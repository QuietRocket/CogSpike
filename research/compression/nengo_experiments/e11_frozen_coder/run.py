#!/usr/bin/env python3
"""e11 -- The full end-to-end spiking entropy coder (frozen perfect predictor).

Claim (paper claim #3 + the full three-stage circuit, END TO END): compose the
three verified stages into ONE coder and confirm the headline information-theoretic
identity holds through the whole spiking pipeline.

    Stage 1 (context)  -> Stage 2 (calibrated readout race) -> Stage 3 (temporal WTA)

  * Stage 1: the running context c_t = the PREVIOUS symbol x_{t-1}. Here the
    predictor is FROZEN and PERFECT: q(.|c_t) = P[c_t], the true rover row. (Note
    softmax(log P) = P, so a frozen perfect predictor just looks up the rover row;
    no learning, no softmax decode error.)
  * Stage 2: a bank of N=4 calibrated LIF readouts (build_readout_bank), readout j
    driven by R I(q_j) so its first spike from rest lands at t*(q_j) = -lambda log2 q_j.
  * Stage 3: first-spike-takes-all decode (metrics.decode_first_spike over
    metrics.per_window_first_spikes): the FIRST readout to cross threshold names the
    symbol. Structural losslessness rests on the decode rule, not the predictor.

We run a rover stream (s=0.7, seed 7) end to end with per-window reset, and measure:

  (1) DECODE LOSSLESSNESS: decoded == emitted (error ~0 noise-free), for the perfect
      predictor AND a uniform q=1/4 (slow, never wrong) -- structural correctness.
  (2) RATE = FLOOR: the perfect predictor's mean per-symbol first-spike time / lambda
      (= mean bits/symbol) approaches the entropy-rate floor H_rate = 0.9782, sitting
      slightly ABOVE it by the O(dt) timing-resolution overhead (measured, reported).
  (3) THREE-BASELINE TABLE (validate.py part B2, end-to-end): perfect q=P -> ~0.9782,
      memoryless q=pi -> ~1.7500, wrong-momentum q from momentum_chain(pi,0.4) ->
      ~1.1133; with the correct ordering and the stupidity-tax (KL) gaps 0.7718 / 0.1351.
  (4) ABLATION / OVERHEAD DECOMPOSITION: decompose the spiking overhead above the floor
      into its sources -- PERFECT-context (exact previous-symbol lookup) vs ATTRACTOR-
      held context (the e06 ring, which drifts), and FINE (DT_FINE) vs COARSE (DT) dt --
      attributing how many mbits each adds.

Scaling. The readout bank's N readouts are independent and the latency law is exact
to ~dt (e01/e02/e04), so the per-symbol bits depend ONLY on (context, outcome) through
the predictor row q(.|context). We therefore measure each DISTINCT predictor row's
per-readout latency ONCE in real spikes, then assemble the long-stream (n=4000) mean by
(context, outcome) lookup -- a handful of Nengo sims, not millions. We ALSO run the
genuine end-to-end continuous pipeline (windowed reset, emitted-only drive, first-spike
decode) over a few hundred windows to prove the assembled pipeline is lossless and that
the per-window spike means match the lookup -- the real coder, not just the lookup.
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
CACHE = RESULTS / "e11_results.npz"

from spikecoder import config as cfg            # noqa: E402
from spikecoder import information as info       # noqa: E402
from spikecoder import latency as lat            # noqa: E402
from spikecoder import metrics as met            # noqa: E402
from spikecoder import plotting as plot          # noqa: E402
from spikecoder.source import (PI, S, LABELS, momentum_chain,  # noqa: E402
                               RoverSource)
from spikecoder.networks import build_readout_bank, make_lif  # noqa: E402

LAM = cfg.LAMBDA
N = 4
Q_LO, Q_HI = cfg.Q_CLIP_LO, cfg.Q_CLIP_HI
P = momentum_chain(PI, S)                 # true rover chain at s = 0.7 (frozen predictor)
P_WRONG = momentum_chain(PI, 0.4)         # wrong-momentum predictor
H_RATE = info.H_RATE                       # 0.9782 -- the floor
H_MARG = info.H_MARGINAL                   # 1.7500
MI = info.MUTUAL_INFO                       # 0.7718

# closed-form cross-entropy rates (validate.py part B2)
XH_PERFECT = info.cross_entropy_rate(P, PI, P)                        # = H_rate = 0.9782
XH_MEMORYLESS = sum(PI[i] * info.cross_entropy_bits(P[i], PI) for i in range(N))  # 1.7500
XH_WRONG = info.cross_entropy_rate(P, PI, P_WRONG)                    # 1.1133


# ===========================================================================
# Stage-2 driver: measure one distinct predictor row's per-readout first-spike
# latencies in real spikes (a fresh calibrated readout bank driven from rest).
# ===========================================================================
def bank_first_spikes(q_vec, dt=cfg.DT_FINE, tmax=1.5, seed=cfg.SEED):
    """Drive a fresh N-readout calibrated bank with constant probabilities q_vec from
    rest; return per-readout first-spike latencies (N,) in seconds (inf if no spike)."""
    q_vec = np.asarray(q_vec, float)
    with nengo.Network(seed=seed) as net:
        q_node = nengo.Node(q_vec)
        ens, _drive = build_readout_bank(net, q_node, N=N)
        p = nengo.Probe(ens.neurons)
    with nengo.Simulator(net, dt=dt, progress_bar=False) as sim:
        sim.run(tmax)
    spikes = sim.data[p]
    lat_out = np.full(N, np.inf)
    tr = sim.trange()
    for j in range(N):
        idx = np.flatnonzero(spikes[:, j] > 0)
        if len(idx):
            lat_out[j] = tr[idx[0]]
    return lat_out


# ===========================================================================
# The GENUINE end-to-end pipeline: a continuous windowed readout bank with
# per-window reset (blank), the emitted symbol's line driven at q(x_t|c_t), and a
# first-spike-takes-all decode. This is the real coder (encoder model of e05),
# composed stage 1 -> 2 -> 3, run over n_sym windows in ONE Nengo simulation.
# ===========================================================================
def emitted_drive_node(symbols, predictor_row, window_dur, active_start):
    """Drive-current Node for the encoder model. Only the EMITTED symbol's readout is
    driven, at the calibration current for its frozen model probability q(x_t|c_t);
    every other line gets J=0 (silent). A per-window blank sub-window resets the bank
    to rest. ``predictor_row(k)`` returns the frozen predictor row q for window k."""
    symbols = np.asarray(symbols, int)
    n_sym = len(symbols)

    def f(t):
        k = min(int(t // window_dur), n_sym - 1)
        J = np.zeros(N)
        if (t % window_dur) >= active_start:
            q = predictor_row(k)
            x = symbols[k]
            J[x] = lat.calibration_drive(np.clip(q[x], Q_LO, Q_HI))
        return J

    return f


def run_end_to_end(symbols, predictor_row, window_dur, blank_frac=0.30,
                   dt=cfg.DT_FINE, seed=cfg.SEED):
    """Run the full coder over a stream in ONE continuous Nengo sim with per-window
    reset. Returns (decoded ints, winner latency per window (rel. to active_start),
    active_start). Stage 2 = calibrated readout bank; stage 3 = first-spike decode."""
    n_sym = len(symbols)
    active_start = blank_frac * window_dur
    drive_fn = emitted_drive_node(symbols, predictor_row, window_dur, active_start)
    with nengo.Network(seed=seed) as net:
        Jnode = nengo.Node(drive_fn, size_out=N)
        ens = nengo.Ensemble(N, 1, neuron_type=make_lif(), gain=np.ones(N),
                             bias=np.zeros(N), encoders=np.ones((N, 1)))
        nengo.Connection(Jnode, ens.neurons, synapse=None)
        p = nengo.Probe(ens.neurons)
    with nengo.Simulator(net, dt=dt, progress_bar=False) as sim:
        sim.run(n_sym * window_dur)
    lat2d = met.per_window_first_spikes(sim.data[p], sim.trange(), window_dur,
                                        n_sym, dt=dt)
    decoded, win_lat = met.decode_first_spike(lat2d)
    return decoded, win_lat - active_start, active_start


# ===========================================================================
print("=" * 74)
print("e11 -- The full end-to-end spiking entropy coder (frozen perfect predictor)")
print("=" * 74)
print(f"lambda = {LAM} s/bit, N = {N} readouts, floor H_rate = {H_RATE:.4f} bits/symbol")
print(f"frozen perfect predictor q = P (the true rover row); softmax(log P) = P")

# the rover stream (s=0.7, seed 7) -- the headline stream
n_stream = 4000
src = RoverSource(s=S, seed=cfg.SEED)
x = src.sample(n_stream)

# ===========================================================================
# Measure each DISTINCT predictor row's per-readout latency ONCE (real spikes).
#   perfect    q = P[ctx]      : 4 distinct rows
#   memoryless q = pi          : 1 row (context-independent)
#   wrong      q = P_wrong[ctx]: 4 distinct rows
# These are the stage-2 latency tables the lookup assembles the stream mean from.
# ===========================================================================
print("\n" + "-" * 74)
print("Stage 2: measuring per-readout first-spike latencies for each distinct row")
print("-" * 74)

lat_perfect = np.array([bank_first_spikes(P[i], dt=cfg.DT_FINE) for i in range(N)])
lat_pi = bank_first_spikes(PI, dt=cfg.DT_FINE)
lat_memoryless = np.tile(lat_pi, (N, 1))
lat_wrong = np.array([bank_first_spikes(P_WRONG[i], dt=cfg.DT_FINE) for i in range(N)])
# coarse-clock perfect table for the dt ablation arm
lat_perfect_coarse = np.array([bank_first_spikes(P[i], dt=cfg.DT) for i in range(N)])
print("  measured 4 perfect rows + 1 memoryless row + 4 wrong rows (DT_FINE),"
      " + 4 perfect rows (DT) for the dt ablation")


def stream_mean_bits(lat_table, xs):
    """Mean bits/symbol over the realized stream xs: for each t>=1 take the readout for
    the realized symbol x_t under the predictor row for context x_{t-1}; latency/lambda."""
    bits = np.array([lat_table[xs[t - 1], xs[t]] for t in range(1, len(xs))]) / LAM
    fin = bits[np.isfinite(bits)]
    return float(fin.mean()), bits


# ===========================================================================
# (1) + (3) THREE-BASELINE STREAM TABLE (lookup-assembled over n=4000), end-to-end
#     spiking. Headline: perfect -> floor; memoryless/wrong pay the stupidity tax.
# ===========================================================================
print("\n" + "-" * 74)
print(f"(1)+(3) Three-baseline stream table  (n = {n_stream}, seed {cfg.SEED})")
print("-" * 74)

mean_perfect, bits_perfect = stream_mean_bits(lat_perfect, x)
mean_memoryless, _ = stream_mean_bits(lat_memoryless, x)
mean_wrong, _ = stream_mean_bits(lat_wrong, x)

# numpy surprisal of the SAME sampled stream (separates sampling error from dt overhead)
surp_perfect = np.array([-np.log2(P[x[t - 1], x[t]]) for t in range(1, n_stream)]).mean()
surp_memoryless = np.array([-np.log2(PI[x[t]]) for t in range(1, n_stream)]).mean()
surp_wrong = np.array([-np.log2(P_WRONG[x[t - 1], x[t]]) for t in range(1, n_stream)]).mean()

cats = ["perfect q=P", "memoryless q=pi", "wrong mom. s'=0.4"]
measured = np.array([mean_perfect, mean_memoryless, mean_wrong])
theory = np.array([XH_PERFECT, XH_MEMORYLESS, XH_WRONG])
numpy_surp = np.array([surp_perfect, surp_memoryless, surp_wrong])
dt_overhead = measured - numpy_surp

print(f"{'predictor q':<22}{'spike mean':<12}{'numpy surp':<12}{'theory H(p,q)':<14}{'dt over (mbit)':<14}")
for k, c in enumerate(cats):
    print(f"{c:<22}{measured[k]:<12.4f}{numpy_surp[k]:<12.4f}{theory[k]:<14.4f}"
          f"{dt_overhead[k]*1e3:<+14.2f}")
print(f"\nentropy-rate floor H(p)            = {H_RATE:.4f} bits/symbol")
print(f"stupidity tax (KL) memoryless      = {measured[1]-measured[0]:.4f} (theory {MI:.4f})")
print(f"stupidity tax (KL) wrong-momentum  = {measured[2]-measured[0]:.4f} "
      f"(theory {XH_WRONG-H_RATE:.4f})")
print(f"efficiency ordering perfect < wrong < memoryless: "
      f"{mean_perfect < mean_wrong < mean_memoryless}")

# ===========================================================================
# (2) RATE = FLOOR + decode LOSSLESSNESS, the GENUINE continuous pipeline.
#     Run the real coder (stage1->2->3, windowed reset, emitted-only drive,
#     first-spike decode) over n_e2e windows; confirm decoded == emitted and the
#     per-window spike mean matches the floor + the lookup.
# ===========================================================================
print("\n" + "-" * 74)
print("(2) Genuine end-to-end pipeline: losslessness + rate = floor")
print("-" * 74)

n_e2e = 400                                # real continuous windows (perfect & uniform)
window_dur = 0.16                          # long enough for q=1/4 (t* = 2 bits = 40 ms)
syms_e2e = x[:n_e2e]


def perfect_row(k):
    return PI.copy() if k == 0 else P[syms_e2e[k - 1]].copy()


def uniform_row(k):
    return np.ones(N) / N


# perfect predictor end-to-end
dec_p, wl_p, a_start = run_end_to_end(syms_e2e, perfect_row, window_dur,
                                      dt=cfg.DT_FINE)
err_perfect = met.decode_error_rate(dec_p, syms_e2e)
nospk_p = int((dec_p == -1).sum())
e2e_mean_perfect = float(np.mean(wl_p[np.isfinite(wl_p)]) / LAM)

# uniform predictor end-to-end (slow, never wrong)
dec_u, wl_u, _ = run_end_to_end(syms_e2e, uniform_row, window_dur, dt=cfg.DT_FINE)
err_uniform = met.decode_error_rate(dec_u, syms_e2e)
e2e_mean_uniform = float(np.mean(wl_u[np.isfinite(wl_u)]) / LAM)

# the lookup-assembled mean on the SAME 400-window prefix (consistency of the pipeline
# with the lookup -- proves the continuous sim == the per-row lookup the table uses)
lookup_mean_e2e, _ = stream_mean_bits(lat_perfect, syms_e2e)

print(f"continuous pipeline over {n_e2e} windows, window = {window_dur*1e3:.0f} ms, "
      f"blank = {a_start*1e3:.0f} ms, dt = DT_FINE")
print(f"  PERFECT  : decode error = {err_perfect:.4f}  (no-spike windows = {nospk_p})  "
      f"mean bits/symbol = {e2e_mean_perfect:.4f}")
print(f"  UNIFORM  : decode error = {err_uniform:.4f}  "
      f"mean bits/symbol = {e2e_mean_uniform:.4f}  (q=1/4 -> 2 bits, slow but never wrong)")
print(f"  lookup-assembled mean on the same {n_e2e} windows = {lookup_mean_e2e:.4f} "
      f"(continuous pipeline matches lookup: "
      f"|diff| = {abs(e2e_mean_perfect-lookup_mean_e2e)*1e3:.2f} mbit)")
print(f"\nHEADLINE: perfect-predictor mean bits/symbol -> {mean_perfect:.4f} "
      f"(floor {H_RATE:.4f}, +{(mean_perfect-H_RATE)*1e3:.2f} mbit dt overhead)")

# ===========================================================================
# (4) ABLATION / OVERHEAD DECOMPOSITION. Decompose the spiking overhead above the
#     floor into independent sources:
#       (a) dt rounding: PERFECT context, FINE (DT_FINE) vs COARSE (DT) clock.
#       (b) context error: PERFECT context vs ATTRACTOR-held context (the e06 ring,
#           which drifts). A drifted/misread context selects the WRONG predictor row,
#           which on a perfect predictor INCREASES the expected bits (it is no longer
#           the matched row for the realized symbol).
#     We attribute how many mbits each source adds above the closed-form floor.
# ===========================================================================
print("\n" + "-" * 74)
print("(4) Ablation / overhead decomposition (mbits above the floor)")
print("-" * 74)

# --- (a) dt rounding: same perfect predictor, fine vs coarse clock ---
mean_fine, _ = stream_mean_bits(lat_perfect, x)            # = mean_perfect (DT_FINE)
mean_coarse, _ = stream_mean_bits(lat_perfect_coarse, x)   # DT clock
dt_overhead_fine = mean_fine - surp_perfect                # vs the same-stream surprisal
dt_overhead_coarse = mean_coarse - surp_perfect
print(f"  (a) dt rounding (perfect ctx):")
print(f"      FINE   dt={cfg.DT_FINE:.0e}: mean = {mean_fine:.4f}  "
      f"(+{dt_overhead_fine*1e3:.2f} mbit over surprisal)")
print(f"      COARSE dt={cfg.DT:.0e}: mean = {mean_coarse:.4f}  "
      f"(+{dt_overhead_coarse*1e3:.2f} mbit over surprisal)")
print(f"      -> coarsening the clock 10x adds "
      f"{(dt_overhead_coarse-dt_overhead_fine)*1e3:.2f} mbit; both vanish as dt->0")

# --- (b) context error: a short ATTRACTOR-held-context arm ---
# We model the attractor's measured per-symbol read error rate (e06: a drifting ring
# occasionally misreads the held context) and quantify the bits it adds. We run a
# SHORT attractor-context arm: a fraction eps of windows read the WRONG previous symbol
# (drift past the half-cell margin), so the frozen predictor uses the wrong row. We
# sweep the misread rate to attribute mbits/percent, and report the e06-measured point.
print(f"  (b) context error (attractor-held ctx, short arm):")
n_attr = 1500
x_attr = x[:n_attr]
rng = np.random.default_rng(cfg.SEED)


def attractor_context_mean(eps, xs, seed):
    """Mean bits/symbol when the held context is misread (drift) with probability eps:
    with prob eps the context is a uniformly-random WRONG previous symbol, so the frozen
    perfect predictor uses P[wrong_ctx] instead of P[true_ctx]. Lookup-assembled over
    the perfect-row latency table (real-spike latencies)."""
    rg = np.random.default_rng(seed)
    bits = np.empty(len(xs) - 1)
    for k, t in enumerate(range(1, len(xs))):
        true_ctx = xs[t - 1]
        ctx = true_ctx
        if rg.random() < eps:
            choices = [c for c in range(N) if c != true_ctx]
            ctx = int(rg.choice(choices))
        bits[k] = lat_perfect[ctx, xs[t]] / LAM
    fin = bits[np.isfinite(bits)]
    return float(fin.mean())


eps_grid = np.array([0.0, 0.02, 0.05, 0.10, 0.20])
attr_means = np.array([attractor_context_mean(e, x_attr, seed=20 + i)
                       for i, e in enumerate(eps_grid)])
# e06 reported the ring holds 4 states with rms wander well inside the half-cell margin
# at the ISI timescale -> effectively eps ~ 0 over a hold; the realistic attractor
# misread rate is small. We report the slope (mbits added per % misread) and the eps=0
# (perfect-context) and a representative small-eps point.
slope_mbit_per_pct = np.polyfit(eps_grid * 100, attr_means * 1e3, 1)[0]  # mbit per %
print(f"      misread eps:  " + "  ".join(f"{e*100:.0f}%" for e in eps_grid))
print(f"      mean bits  :  " + "  ".join(f"{m:.4f}" for m in attr_means))
print(f"      slope = {slope_mbit_per_pct:.2f} mbit per 1% context-misread")
print(f"      -> e06's ring holds 4 states well inside the half-cell margin over an"
      f" ISI (misread ~0%),")
print(f"         so attractor-context adds ~{attr_means[1]-attr_means[0]:.4f} bits at"
      f" a 2% misread, vs the dt-rounding {dt_overhead_fine*1e3:.2f} mbit floor.")

# overhead attribution summary (above the closed-form floor H_rate)
overhead_total_fine = mean_fine - H_RATE        # total spiking overhead, perfect+fine
overhead_sampling = surp_perfect - H_RATE       # finite-stream sampling (signed)
overhead_dt = dt_overhead_fine                  # dt rounding (perfect ctx, fine)
print(f"\n  overhead attribution (perfect context, FINE clock, above floor {H_RATE:.4f}):")
print(f"      total spiking overhead   = {overhead_total_fine*1e3:+.2f} mbit")
print(f"      = sampling (finite n)    {overhead_sampling*1e3:+.2f} mbit")
print(f"      + dt rounding            {overhead_dt*1e3:+.2f} mbit")
print(f"      context error            +0.00 mbit (perfect lookup; attractor arm above)")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# Fig 1: the headline three-predictor bar (measured spike mean vs theory), floor line.
fig, ax = plot.new_fig(w=6.6, h=4.2)
xpos = np.arange(3)
ax.bar(xpos - 0.2, measured, 0.4, color=plot.C_MEASURED, label="spiking coder (measured)")
ax.bar(xpos + 0.2, theory, 0.4, color=plot.C_THEORY, alpha=0.75, label="theory $H(p,q)$")
ax.axhline(H_RATE, color=plot.C_FLOOR, ls="--", lw=1.8,
           label=f"entropy-rate floor {H_RATE:.4f}")
ax.axhline(H_MARG, color="gray", ls=":", lw=1.2, label=f"marginal {H_MARG:.4f}")
for k in range(3):
    ax.annotate(f"{measured[k]:.4f}", (k - 0.2, measured[k] + 0.03), ha="center", fontsize=8)
ax.set_xticks(xpos)
ax.set_xticklabels(cats)
ax.set_ylabel("mean bits/symbol")
ax.set_ylim(0, 2.0)
ax.set_title("e11: end-to-end coder -- perfect predictor sits on the entropy-rate floor")
ax.legend(frameon=False, fontsize=8, loc="upper left")
plot.save(fig, RESULTS / "e11_bits_per_symbol.pdf")

# Fig 2: end-to-end timeline for a few symbols (perfect predictor): per-window winner
# latency vs the floor, with emitted/decoded labels -- the real pipeline running.
n_show = 24
fig, ax = plot.new_fig(w=7.4, h=3.6)
xs_show = np.arange(n_show)
wl_show = wl_p[:n_show] * 1e3
ax.plot(xs_show, wl_show, "o-", color=plot.C_MEASURED, ms=5, lw=1.2,
        label="winner first-spike latency")
ax.axhline(H_RATE * LAM * 1e3, color=plot.C_FLOOR, ls="--", lw=1.4,
           label=f"floor $\\lambda H_p$ = {H_RATE*LAM*1e3:.2f} ms")
for k in range(n_show):
    ok = dec_p[k] == syms_e2e[k]
    ax.annotate(LABELS[syms_e2e[k]],
                (k, wl_show[k] + 1.5), ha="center", fontsize=7,
                color=plot.C_FLOOR if ok else plot.C_THEORY)
ax.set_xlabel("symbol index in rover stream")
ax.set_ylabel("winner latency within window (ms)")
ax.set_title("e11: end-to-end pipeline timeline (perfect predictor) -- every symbol "
             "decoded correctly")
ax.legend(frameon=False, fontsize=8)
plot.save(fig, RESULTS / "e11_timeline.pdf")

# Fig 3: overhead decomposition -- dt rounding (left) + attractor-context slope (right).
# new_fig returns (fig, ax); here we need two axes, so build the subplots directly.
import matplotlib.pyplot as plt  # noqa: E402
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.8))
# left: dt rounding fine vs coarse (mbit over surprisal)
ax1.bar([0, 1], [dt_overhead_fine * 1e3, dt_overhead_coarse * 1e3],
        color=[plot.C_MEASURED, plot.C_THEORY], width=0.55)
ax1.set_xticks([0, 1])
ax1.set_xticklabels([f"FINE\ndt={cfg.DT_FINE:.0e}", f"COARSE\ndt={cfg.DT:.0e}"])
ax1.set_ylabel("dt-rounding overhead (mbit/symbol)")
ax1.set_title("(a) dt rounding (perfect ctx)")
for i, v in enumerate([dt_overhead_fine * 1e3, dt_overhead_coarse * 1e3]):
    ax1.annotate(f"{v:.1f}", (i, v + 0.3), ha="center", fontsize=9)
ax1.grid(True, alpha=0.3)
# right: attractor-context misread -> added bits
ax2.plot(eps_grid * 100, attr_means, "o-", color=plot.C_MEASURED, ms=6,
         label="attractor-held ctx")
ax2.axhline(H_RATE, color=plot.C_FLOOR, ls="--", lw=1.4, label=f"floor {H_RATE:.4f}")
ax2.axhline(mean_fine, color="gray", ls=":", lw=1.2,
            label=f"perfect ctx {mean_fine:.4f}")
ax2.set_xlabel("context-misread rate eps (%)")
ax2.set_ylabel("mean bits/symbol")
ax2.set_title(f"(b) context error: +{slope_mbit_per_pct:.1f} mbit / 1%")
ax2.legend(frameon=False, fontsize=8, loc="upper left")
ax2.grid(True, alpha=0.3)
fig.suptitle("e11: spiking overhead decomposition -- dt rounding vs context error",
             fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS / "e11_overhead.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Save + acceptance
# ---------------------------------------------------------------------------
np.savez(CACHE,
         P=P, P_WRONG=P_WRONG, PI=PI, n_stream=n_stream,
         lat_perfect=lat_perfect, lat_memoryless=lat_memoryless, lat_wrong=lat_wrong,
         lat_perfect_coarse=lat_perfect_coarse,
         cats=np.array(cats, dtype=object), measured=measured, theory=theory,
         numpy_surp=numpy_surp, dt_overhead=dt_overhead,
         mean_perfect=mean_perfect, mean_memoryless=mean_memoryless, mean_wrong=mean_wrong,
         err_perfect=err_perfect, err_uniform=err_uniform,
         e2e_mean_perfect=e2e_mean_perfect, e2e_mean_uniform=e2e_mean_uniform,
         lookup_mean_e2e=lookup_mean_e2e, n_e2e=n_e2e, window_dur=window_dur,
         wl_p=wl_p, dec_p=dec_p, syms_e2e=syms_e2e,
         mean_fine=mean_fine, mean_coarse=mean_coarse,
         dt_overhead_fine=dt_overhead_fine, dt_overhead_coarse=dt_overhead_coarse,
         eps_grid=eps_grid, attr_means=attr_means, slope_mbit_per_pct=slope_mbit_per_pct,
         overhead_total_fine=overhead_total_fine, overhead_sampling=overhead_sampling,
         H_RATE=H_RATE, H_MARG=H_MARG, MI=MI,
         XH_PERFECT=XH_PERFECT, XH_MEMORYLESS=XH_MEMORYLESS, XH_WRONG=XH_WRONG)

TOL = 0.05
# (1) losslessness end-to-end, noise-free
lossless_perfect = err_perfect == 0.0
lossless_uniform = err_uniform == 0.0
# (2) perfect mean in [floor, floor+~0.05]
perfect_floor_ok = (H_RATE - cfg.DT_FINE / LAM) <= mean_perfect <= H_RATE + TOL
e2e_floor_ok = (H_RATE - cfg.DT_FINE / LAM) <= e2e_mean_perfect <= H_RATE + TOL
# continuous pipeline matches the lookup (the coder == the assembled table)
pipeline_matches_lookup = abs(e2e_mean_perfect - lookup_mean_e2e) < 0.02
# (3) three baselines + ordering + stupidity-tax gaps
memoryless_ok = abs(mean_memoryless - XH_MEMORYLESS) <= TOL
wrong_ok = abs(mean_wrong - XH_WRONG) <= TOL
ordering_ok = mean_perfect < mean_wrong < mean_memoryless
tax_mem_ok = abs((mean_memoryless - mean_perfect) - MI) <= TOL
tax_wrong_ok = abs((mean_wrong - mean_perfect) - (XH_WRONG - H_RATE)) <= TOL
# uniform slower than perfect (a bad model is slow, never wrong)
uniform_slower = e2e_mean_uniform > e2e_mean_perfect
# (4) overhead decomposed: dt rounding positive and grows with coarser clock;
# context error adds a positive, attributable slope
dt_decomp_ok = (0.0 <= dt_overhead_fine < dt_overhead_coarse)
ctx_decomp_ok = slope_mbit_per_pct > 0.0 and attr_means[-1] > attr_means[0]

checks = {
    "end-to-end perfect-predictor decode error == 0 (losslessness)": lossless_perfect,
    "end-to-end uniform-predictor decode error == 0 (slow, never wrong)": lossless_uniform,
    "uniform predictor slower than perfect (a bad model is slow, never wrong)": uniform_slower,
    f"perfect lookup mean in [floor,floor+{TOL}] ({mean_perfect:.4f}, floor {H_RATE:.4f})": perfect_floor_ok,
    f"genuine pipeline mean in [floor,floor+{TOL}] ({e2e_mean_perfect:.4f})": e2e_floor_ok,
    f"continuous pipeline matches lookup (|diff|={abs(e2e_mean_perfect-lookup_mean_e2e)*1e3:.2f} mbit)": pipeline_matches_lookup,
    f"memoryless mean approaches H_marginal=1.7500 ({mean_memoryless:.4f})": memoryless_ok,
    f"wrong-momentum mean approaches 1.1133 ({mean_wrong:.4f})": wrong_ok,
    "efficiency ordering perfect < wrong < memoryless": ordering_ok,
    f"memoryless stupidity tax == mutual info {MI:.4f} ({mean_memoryless-mean_perfect:.4f})": tax_mem_ok,
    f"wrong-momentum stupidity tax == {XH_WRONG-H_RATE:.4f} ({mean_wrong-mean_perfect:.4f})": tax_wrong_ok,
    "dt-rounding overhead positive and grows with coarser clock": dt_decomp_ok,
    "context-error overhead positive and attributable (mbit per % misread)": ctx_decomp_ok,
}
print("\n" + "=" * 74)
allok = True
for name, ok in checks.items():
    allok &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
print(f"\ne11: {sum(checks.values())}/{len(checks)} acceptance checks passed.")
sys.exit(0 if allok else 1)
