#!/usr/bin/env python3
"""e04 -- The calibrated readout race: N=4 LIF readouts, lowest surprisal fires first.

Claim (paper #2, circuit stage 2): a bank of N calibrated LIF readouts, readout j
driven by the calibration current R I(q_j), races to threshold. Because the
first-spike latency from rest is t*(q_j) = -lambda log2 q_j (e01/e02), the readout
with the LARGEST q_j fires FIRST, and its latency equals -lambda log2 max_j q_j.
The argmin-latency readout is therefore argmax_j q_j -- the model's MAP symbol,
read off as a spike *time*.

Over a rover stream this realizes the stream identity of validate.py part B2:
feed each window the predictor q = predictor-row for the previous symbol x_{t-1},
and read the realized symbol x_t's readout latency. That latency / lambda is the
per-symbol surprisal -log2 q_{x_t}; its pi-weighted mean is the cross-entropy
rate H(p, q). For the PERFECT predictor q = P this is the entropy-rate floor
H_rate = 0.9782 bits; for the MEMORYLESS predictor q = pi it is H_marginal =
1.7500; for a WRONG-momentum predictor (s'=0.4) it is 1.1133. The "stupidity tax"
of a mismatched model is its excess latency (= KL) per symbol.

Three parts:
  (1) Per-context race: drive the bank with q = P[i] for each context i; confirm
      argmax P[i] fires first and at -lambda log2 max P[i] within ~dt.
  (2) Stream identity: the three-row table B2, realized as spike-time means.
  (3) Discrete-clock penalty: at finite dt two readouts can land in the same dt
      bin (a tie). Report the tie/collision rate vs dt (DT_FINE vs DT). This is
      the spiking analogue of the paper's continuous-time no-integer-penalty claim.

Because the bank's readouts are independent and the latency law is exact to ~dt,
we drive the bank once per DISTINCT predictor row (4 for q=P, 1 for q=pi, 4 for
q=P_wrong), read all N first-spike latencies from rest, and assemble the stream
means by (context, outcome) lookup -- a handful of Nengo sims instead of millions.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import nengo

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

from spikecoder import config as cfg            # noqa: E402
from spikecoder import information as info       # noqa: E402
from spikecoder import latency as lat            # noqa: E402
from spikecoder import metrics as met            # noqa: E402
from spikecoder import plotting as plot          # noqa: E402
from spikecoder.source import (PI, S, LABELS, momentum_chain,  # noqa: E402
                               sample_stream, RoverSource)
from spikecoder.networks import build_readout_bank  # noqa: E402

LAM = cfg.LAMBDA
N = 4
P = momentum_chain(PI, S)                 # the true rover chain at s = 0.7
P_WRONG = momentum_chain(PI, 0.4)         # wrong-momentum predictor
H_RATE = info.H_RATE                       # 0.9782
H_MARG = info.H_MARGINAL                   # 1.7500


def bank_first_spikes(q_vec, dt=cfg.DT_FINE, tmax=0.5, seed=cfg.SEED):
    """Drive a fresh N-readout bank with the constant probability vector q_vec from
    rest; return the per-readout first-spike latencies (N,) in seconds (inf if a
    readout never fires within tmax).
    """
    q_vec = np.asarray(q_vec, float)
    with nengo.Network(seed=seed) as net:
        q_node = nengo.Node(q_vec)                 # constant model probabilities
        ens, _drive = build_readout_bank(net, q_node, N=N)
        p = nengo.Probe(ens.neurons)
    with nengo.Simulator(net, dt=dt, progress_bar=False) as sim:
        sim.run(tmax)
    spikes = sim.data[p]                            # (T, N)
    lat_out = np.full(N, np.inf)
    for j in range(N):
        idx = np.flatnonzero(spikes[:, j] > 0)
        if len(idx):
            lat_out[j] = sim.trange()[idx[0]]
    return lat_out, spikes, sim.trange()


# ---------------------------------------------------------------------------
print("=" * 72)
print("e04 -- The calibrated readout race")
print("=" * 72)
print(f"lambda = {LAM} s/bit,  N = {N} readouts,  alpha = {cfg.ALPHA:.4f}")

# ===========================================================================
# (1) Per-context race: for each context i, drive the bank with q = P[i].
#     The argmax_j P[i,j] readout must fire first, at -lambda log2 max_j P[i,j].
# ===========================================================================
print("\n" + "-" * 72)
print("(1) Per-context race  (drive q = P[context], dt = DT_FINE)")
print("-" * 72)
print("ctx  q=P[ctx]                     argmax  winner  win_lat(ms)  theory(ms)  |err|/dt")

ctx_winner_ok = True
ctx_lat_ok = True
ctx_rows = []           # (ctx, winner, theory_winner_lat, meas_winner_lat)
per_ctx_lat = []        # full (N,) latency vector per context (for the raster)
for i in range(N):
    q = P[i]
    lat_vec, spikes, trange = bank_first_spikes(q, dt=cfg.DT_FINE, tmax=0.5)
    winner = int(np.argmin(lat_vec))
    argmax_q = int(np.argmax(q))
    win_lat = lat_vec[winner]
    theory_win = lat.analytic_latency_ideal(q.max())     # -lambda log2 max q
    err_dt = abs(win_lat - theory_win) / cfg.DT_FINE
    ok_w = (winner == argmax_q)
    ok_l = (err_dt <= 2.0)
    ctx_winner_ok &= ok_w
    ctx_lat_ok &= ok_l
    ctx_rows.append((i, winner, theory_win, win_lat))
    per_ctx_lat.append(lat_vec)
    qstr = "[" + " ".join(f"{v:.4f}" for v in q) + "]"
    print(f"{LABELS[i]}    {qstr}   {LABELS[argmax_q]}({argmax_q})  "
          f"{LABELS[winner]}({winner})   {win_lat*1e3:>8.3f}   {theory_win*1e3:>8.3f}   {err_dt:>6.2f}")
per_ctx_lat = np.array(per_ctx_lat)                       # (N contexts, N readouts)

# also confirm the FULL latency ordering matches the surprisal ordering per context
order_ok = True
for i in range(N):
    meas_order = np.argsort(per_ctx_lat[i])
    surp_order = np.argsort(-np.log2(P[i]))               # smallest surprisal first
    order_ok &= bool(np.array_equal(meas_order, surp_order))
print(f"\nfull per-context latency order == surprisal order for all contexts: {order_ok}")
print(f"argmax-q readout fires first for every context: {ctx_winner_ok}")
print(f"winner latency = -lambda log2 max(q) within ~dt for every context: {ctx_lat_ok}")

# ===========================================================================
# (2) Stream identity (B2): mean bits/symbol = cross-entropy rate H(p, q)
#     for three predictors. We measure each DISTINCT predictor row's per-readout
#     latency once (in spikes), then assemble the pi-weighted stream mean.
# ===========================================================================
print("\n" + "-" * 72)
print("(2) Stream identity  (mean spike-latency / lambda = cross-entropy rate)")
print("-" * 72)

n_stream = 4000
src = RoverSource(s=S, seed=cfg.SEED)
x = src.sample(n_stream)                                  # ints in 0..3

# --- measure per-readout latencies for each distinct predictor context row ---
# Perfect predictor q = P[ctx]: 4 distinct rows -> 4 bank sims (reuse part 1).
lat_perfect = per_ctx_lat.copy()                          # (ctx, readout) seconds

# Memoryless predictor q = pi (context-independent): 1 bank sim.
lat_pi_vec, _, _ = bank_first_spikes(PI, dt=cfg.DT_FINE, tmax=1.5)
lat_memoryless = np.tile(lat_pi_vec, (N, 1))              # same row for every ctx

# Wrong-momentum predictor q = P_wrong[ctx]: 4 bank sims.
lat_wrong = np.array([bank_first_spikes(P_WRONG[i], dt=cfg.DT_FINE, tmax=1.5)[0]
                      for i in range(N)])                 # (ctx, readout)


def stream_mean_bits(lat_table):
    """Mean bits/symbol over the realized stream: for each t>=1, take the readout
    for the realized symbol x_t under the predictor row for context x_{t-1}, divide
    its first-spike latency by lambda. Returns (mean bits, per-symbol bits array)."""
    bits = np.empty(n_stream - 1)
    for k, t in enumerate(range(1, n_stream)):
        ctx = x[t - 1]
        out = x[t]
        bits[k] = lat_table[ctx, out] / LAM
    return float(np.nanmean(bits[np.isfinite(bits)])), bits


mean_perfect, bits_perfect = stream_mean_bits(lat_perfect)
mean_memoryless, bits_memoryless = stream_mean_bits(lat_memoryless)
mean_wrong, bits_wrong = stream_mean_bits(lat_wrong)

# theory (cross-entropy rates) from validate.py part B2
xH_perfect = info.cross_entropy_rate(P, PI, P)            # = H_rate = 0.9782
xH_memoryless = sum(PI[i] * info.cross_entropy_bits(P[i], PI) for i in range(N))  # 1.7500
xH_wrong = info.cross_entropy_rate(P, PI, P_WRONG)        # 1.1133

# also the pure-numpy stream surprisal mean (sampling reference, no spikes),
# so we can separate sampling error from the dt timing-resolution overhead.
surp_perfect = np.array([-np.log2(P[x[t - 1], x[t]]) for t in range(1, n_stream)]).mean()
surp_memoryless = np.array([-np.log2(PI[x[t]]) for t in range(1, n_stream)]).mean()
surp_wrong = np.array([-np.log2(P_WRONG[x[t - 1], x[t]]) for t in range(1, n_stream)]).mean()

cats = ["perfect q=P", "memoryless q=pi", "wrong mom. s'=0.4"]
measured = np.array([mean_perfect, mean_memoryless, mean_wrong])
theory = np.array([xH_perfect, xH_memoryless, xH_wrong])
numpy_surp = np.array([surp_perfect, surp_memoryless, surp_wrong])

print(f"n = {n_stream} symbols, seed = {cfg.SEED}")
print(f"{'predictor q':<22}{'spike mean':<12}{'numpy surp':<12}{'theory H(p,q)':<14}{'dt overhead':<12}")
for k, c in enumerate(cats):
    print(f"{c:<22}{measured[k]:<12.4f}{numpy_surp[k]:<12.4f}{theory[k]:<14.4f}"
          f"{measured[k]-numpy_surp[k]:<+12.4f}")
print(f"\nentropy-rate floor H(p)          = {H_RATE:.4f} bits/symbol")
print(f"stupidity tax (KL) memoryless    = {xH_memoryless - H_RATE:.4f} bits/symbol")
print(f"stupidity tax (KL) wrong-momentum= {xH_wrong - H_RATE:.4f} bits/symbol")

# the dt overhead is positive (fast/confident symbols round UP to the dt grid):
dt_overhead = measured - numpy_surp
print(f"\nmeasured spike means sit ABOVE the numpy surprisal by "
      f"{dt_overhead.min()*1e3:.2f}-{dt_overhead.max()*1e3:.2f} mbits "
      f"(the dt timing-resolution overhead)")

# ===========================================================================
# (3) Discrete-clock penalty: tie/collision rate vs dt. At finite dt two readouts
#     can land their first spike in the SAME dt bin -- an unbreakable race the
#     continuous-time code never has. A tie occurs when the top-two surprisal gap
#     lambda*|log2 q1 - log2 q2| < dt.
#
#     First the NATURAL source: the rover's MAP gaps are HUGE (the sticky chain
#     puts ~0.74-0.85 on the repeat move and <=0.15 on the rest), so the winner's
#     race against the runner-up is decided by ~46-70 ms -- no clock down to DT=1ms
#     can fuse them. That is a real finding: the integer-penalty analogue does NOT
#     bite on a source whose distributions are far from uniform.
#
#     Then we DEMONSTRATE the mechanism on a deliberately near-tied distribution
#     whose top-2 surprisal gap straddles the two clocks (gap ~ 0.3 ms, between
#     DT_FINE=0.1 ms and DT=1 ms): ties appear at the coarse clock, vanish at the
#     fine clock. This is the spiking analogue of the continuous-time
#     no-integer-penalty claim made concrete.
# ===========================================================================
print("\n" + "-" * 72)
print("(3) Discrete-clock penalty: readout-tie (collision) rate vs dt")
print("-" * 72)


def tie_rate_for_dt(rows, dt):
    """For each distribution row, drive a bank with q=row and count the fraction
    whose top-2 readouts land in the SAME dt bin (winner unresolved by the clock).
    Returns (tie rate, smallest realized top-2 latency gap in seconds)."""
    rows = np.atleast_2d(rows)
    ties = 0
    min_gaps = []
    for r in rows:
        lat_vec, _, _ = bank_first_spikes(r, dt=dt, tmax=0.6)
        srt = np.sort(lat_vec[np.isfinite(lat_vec)])
        if len(srt) >= 2:
            gap = srt[1] - srt[0]
            min_gaps.append(gap)
            if gap < dt:                                   # same bin -> tie
                ties += 1
    return ties / len(rows), (min(min_gaps) if min_gaps else np.inf)


# --- natural source: the rover's own MAP races (top-2 gaps are large) ---
nat_tie_fine, nat_gap_fine = tie_rate_for_dt(P, cfg.DT_FINE)
nat_tie_coarse, nat_gap_coarse = tie_rate_for_dt(P, cfg.DT)
theory_gaps = np.array([met.latency_gap_margin(P[i], LAM) for i in range(N)])
print(f"natural rover top-2 surprisal gap per context (ms): "
      + ", ".join(f"{LABELS[i]}={theory_gaps[i]*1e3:.1f}" for i in range(N)))
print(f"  -> smallest = {theory_gaps.min()*1e3:.1f} ms >> DT = {cfg.DT*1e3:.1f} ms: "
      f"NO clock down to DT fuses them.")
print(f"  natural tie rate: dt={cfg.DT_FINE:.0e} -> {nat_tie_fine:.2f}, "
      f"dt={cfg.DT:.0e} -> {nat_tie_coarse:.2f}  (finding: integer-penalty analogue")
print(f"  does NOT bite a far-from-uniform source)")

# --- engineered near-tie: gap engineered to straddle DT_FINE and DT ---
# Pick q1, q2 so that lambda*(log2 q1 - log2 q2) ~ 0.3 ms (in (DT_FINE, DT)).
# q2/q1 = 2^(-gap/lambda).  gap = 0.3 ms -> ratio = 2^(-0.015) = 0.9897.
gap_target = 3.0e-4                                        # 0.3 ms, between 0.1 and 1.0 ms
q1 = 0.45
q2 = q1 * 2.0 ** (-gap_target / LAM)                       # slightly smaller -> ~0.3ms slower
q_rest = (1.0 - q1 - q2) / 2.0
q_tied = np.array([q1, q2, q_rest, q_rest])
expected_gap = met.latency_gap_margin(q_tied, LAM)
print(f"\nengineered near-tie q = [{q1:.4f} {q2:.4f} {q_rest:.4f} {q_rest:.4f}], "
      f"top-2 surprisal gap = {expected_gap*1e3:.3f} ms")
tie_fine, gap_fine = tie_rate_for_dt(q_tied[None, :], cfg.DT_FINE)
tie_coarse, gap_coarse = tie_rate_for_dt(q_tied[None, :], cfg.DT)
print(f"  dt = {cfg.DT_FINE:.0e}: tie = {tie_fine:.0f}  "
      f"(realized top-2 gap {gap_fine*1e3:.3f} ms -- clock RESOLVES the race)")
print(f"  dt = {cfg.DT:.0e}: tie = {tie_coarse:.0f}  "
      f"(realized top-2 gap {gap_coarse*1e3:.3f} ms -- both spikes share one bin -> TIE)")
print("  -> the coarse clock FUSES readouts whose surprisal gap < dt: the discrete")
print("     clock is the spiking analogue of the integer-bit penalty, and it only")
print("     bites when two model probabilities are nearly equal.")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# Fig 1: per-context spike raster of the readout bank (context U, the clearest race)
ctx_show = 0
_, spikes_show, trange_show = bank_first_spikes(P[ctx_show], dt=cfg.DT_FINE, tmax=0.06)
fig, ax = plot.new_fig(w=6.5, h=3.2)
order = np.argsort(-P[ctx_show])                            # fastest (highest q) on top
for row, j in enumerate(order):
    ts = trange_show[spikes_show[:, j] > 0]
    if len(ts):
        ax.vlines(ts[0] * 1e3, row + 0.6, row + 1.4, color=plot.C_MEASURED, lw=2.5)
    th = lat.analytic_latency_ideal(P[ctx_show, j]) * 1e3
    ax.plot(th, row + 1, "x", color=plot.C_THEORY, ms=9, mew=2)
ax.set_yticks(range(1, N + 1))
ax.set_yticklabels([f"{LABELS[j]}  q={P[ctx_show, j]:.3f}" for j in order])
ax.set_xlabel("first-spike latency (ms)")
ax.set_title(f"e04: readout race, context {LABELS[ctx_show]} -- highest q fires first "
             f"(x = theory)")
ax.set_ylim(0.4, N + 0.6)
plot.save(fig, RESULTS / "e04_race_raster.pdf")

# Fig 2: winner latency vs theory across contexts
fig, ax = plot.new_fig()
ctx_idx = np.arange(N)
meas_win = np.array([r[3] for r in ctx_rows]) * 1e3
theo_win = np.array([r[2] for r in ctx_rows]) * 1e3
ax.bar(ctx_idx - 0.2, meas_win, 0.4, color=plot.C_MEASURED, label="measured winner")
ax.bar(ctx_idx + 0.2, theo_win, 0.4, color=plot.C_THEORY, alpha=0.7,
       label=r"$-\lambda\log_2\max_j q_j$")
ax.set_xticks(ctx_idx)
ax.set_xticklabels([f"ctx {LABELS[i]}\n(winner {LABELS[ctx_rows[i][1]]})" for i in range(N)])
ax.set_ylabel("winner first-spike latency (ms)")
ax.set_title("e04: winner latency = surprisal of the MAP symbol, per context")
ax.legend(frameon=False)
plot.save(fig, RESULTS / "e04_winner_latency.pdf")

# Fig 3: the B2 three-predictor bar chart, measured vs theory
plot.plot_bar_compare(
    cats, measured, theory, RESULTS / "e04_bits_per_symbol.pdf",
    ylabel="mean bits/symbol", title="e04: stream identity -- spike-time mean = cross-entropy rate")

# Fig 4: tie rate vs dt -- natural rover (no ties) vs engineered near-tie
fig, ax = plot.new_fig(w=6.0, h=4.0)
xlab = [f"dt={cfg.DT_FINE:.0e}", f"dt={cfg.DT:.0e}"]
xpos = np.arange(2)
ax.bar(xpos - 0.2, [nat_tie_fine, nat_tie_coarse], 0.4, color=plot.C_FLOOR,
       label=f"natural rover (gap >= {theory_gaps.min()*1e3:.0f} ms)")
ax.bar(xpos + 0.2, [tie_fine, tie_coarse], 0.4, color=plot.C_MEASURED,
       label=f"engineered near-tie (gap {expected_gap*1e3:.2f} ms)")
for k, v in enumerate([nat_tie_fine, nat_tie_coarse]):
    ax.annotate(f"{v:.0f}", (k - 0.2, v + 0.03), ha="center", fontsize=9)
for k, v in enumerate([tie_fine, tie_coarse]):
    ax.annotate(f"{v:.0f}", (k + 0.2, v + 0.03), ha="center", fontsize=9)
ax.set_xticks(xpos)
ax.set_xticklabels(xlab)
ax.set_ylabel("readout-tie (collision) rate")
ax.set_ylim(0, 1.15)
ax.set_title("e04: the coarse clock fuses readouts only when surprisal gap < dt")
ax.legend(frameon=False, fontsize=8, loc="upper left")
plot.save(fig, RESULTS / "e04_tie_rate.pdf")

# ---------------------------------------------------------------------------
# Save + acceptance
# ---------------------------------------------------------------------------
np.savez(RESULTS / "e04_results.npz",
         P=P, P_WRONG=P_WRONG, PI=PI,
         per_ctx_lat=per_ctx_lat,
         ctx_winner=np.array([r[1] for r in ctx_rows]),
         ctx_win_lat=np.array([r[3] for r in ctx_rows]),
         ctx_win_theory=np.array([r[2] for r in ctx_rows]),
         cats=np.array(cats, dtype=object),
         measured=measured, theory=theory, numpy_surp=numpy_surp,
         dt_overhead=dt_overhead,
         theory_gaps=theory_gaps,
         nat_tie_fine=nat_tie_fine, nat_tie_coarse=nat_tie_coarse,
         tie_fine=tie_fine, tie_coarse=tie_coarse,
         gap_fine=gap_fine, gap_coarse=gap_coarse,
         q_tied=q_tied, expected_gap=expected_gap,
         n_stream=n_stream)

# tolerances: perfect predictor sits slightly ABOVE H_rate (dt rounding of fast
# symbols); allow [floor, floor + 0.05]. memoryless/wrong within sampling+dt.
TOL = 0.05
perfect_ok = (H_RATE - cfg.DT_FINE / LAM) <= mean_perfect <= H_RATE + TOL
memoryless_ok = abs(mean_memoryless - xH_memoryless) <= TOL
wrong_ok = abs(mean_wrong - xH_wrong) <= TOL
# the perfect-predictor spike mean should be the SMALLEST of the three (it is the
# most efficient code) and should beat the memoryless code by ~the mutual info.
ordering_ok = mean_perfect < mean_wrong < mean_memoryless
# dt overhead is real and positive (fast symbols round up)
overhead_ok = dt_overhead.min() >= -1e-9 and dt_overhead.max() > 0
# discrete-clock penalty: the natural rover never ties (gaps >> dt), but an
# engineered near-tie (gap between DT_FINE and DT) ties only at the coarse clock.
tie_natural_ok = (nat_tie_fine == 0.0 and nat_tie_coarse == 0.0)
tie_engineered_ok = (tie_fine == 0.0 and tie_coarse == 1.0)

checks = {
    "argmax-q readout fires first for every context": ctx_winner_ok,
    "winner latency = -lambda log2 max(q) within ~dt (all contexts)": ctx_lat_ok,
    "full per-context latency order == surprisal order": order_ok,
    f"perfect predictor mean in [floor, floor+{TOL}] ({mean_perfect:.4f}, floor {H_RATE:.4f})": perfect_ok,
    f"memoryless mean approaches H_marginal=1.7500 ({mean_memoryless:.4f})": memoryless_ok,
    f"wrong-momentum mean approaches 1.1133 ({mean_wrong:.4f})": wrong_ok,
    "perfect < wrong < memoryless (efficiency ordering)": ordering_ok,
    "dt timing-resolution overhead is real and positive": overhead_ok,
    "natural rover never ties at any clock (gaps >> dt)": tie_natural_ok,
    "engineered near-tie ties at coarse clock, resolves at fine clock": tie_engineered_ok,
}
print("\n" + "=" * 72)
allok = True
for name, ok in checks.items():
    allok &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
print(f"\ne04: {sum(checks.values())}/{len(checks)} acceptance checks passed.")
sys.exit(0 if allok else 1)
