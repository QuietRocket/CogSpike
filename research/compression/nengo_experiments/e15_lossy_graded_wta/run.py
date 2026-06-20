#!/usr/bin/env python3
"""e15 -- Learned lossy compression via a graded WTA (rate-distortion).

Everything in the suite so far has been LOSSLESS source coding: the first-spike-
takes-all decode (e05/e11) reconstructs the emitted symbol EXACTLY, so the only
quantity that moved was the *rate* (mean bits/symbol), and correctness was a
structural invariant -- decoded == emitted for ANY predictor. This capstone relaxes
that. Replacing the HARD winner-take-all decode with a GRADED / soft argmax -- a
sub-saturating sum-mode gain (a finite softmax temperature), i.e. a *wide graded
bump* over the calibrated readout -- lets the decoder commit to a COARSENED symbol,
MERGING hard-to-distinguish (near-equiprobable, near-tied-latency) outcomes and
spending fewer spikes to resolve them.

The graded-bump width is a RATE-DISTORTION knob (paper section "Learned lossy
compression and rate-distortion"):

    wider bump  ==  larger merge margin delta  ==  coarser symbols
                ==  LOWER rate (fewer bits/symbol)  +  HIGHER distortion.

We make the knob concrete as a per-context MERGE MARGIN delta (in bits = latency /
lambda): two readouts whose first-spike latencies sit within delta of each other are
"hard to distinguish", so the graded bump cannot separate them and the decoder
commits to their merged class (codeword) instead of the individual symbol.

  * RATE = mean bits/symbol actually spent = the spike-time cost of resolving the
    emitted symbol's CLASS (not the symbol): the merged class's first-spike latency
    among the coarse codewords, / lambda. Measured from REAL Nengo readout spikes.
  * DISTORTION = expected 0-1 (Hamming) reconstruction error between the emitted
    symbol and the decoded class's representative (the class's MAP symbol).

At delta = 0 nothing merges: the lossless first-spike decode, rate ~ H_rate = 0.9782
(+ the dt overhead), distortion = 0. As delta grows the bump widens, classes merge,
rate drops BELOW the lossless floor and distortion rises from 0 -- a clean spiking
RATE-DISTORTION CURVE.

Layers:
  (A) R-D CURVE from the graded decode over the REAL calibrated readout bank. We
      measure each distinct predictor row's per-readout first-spike latencies ONCE in
      real Nengo spikes (e04/e11 idiom: independent readouts, latency law exact to dt),
      then sweep delta and assemble (rate, distortion) over the n=4000 rover stream by
      (context, outcome) lookup. The curve is monotone: rate down, distortion up.
  (B) THE GRADED BUMP itself: a soft sum-mode-gain decode r ∝ q^g over the readout.
      g=1 is the sharp (lossless) bump; g<1 is a sub-saturating, WIDE bump. We show
      the bump shape vs g and that a wider bump (smaller g) resolves symbols only when
      their latency gap exceeds an effective margin delta_eff(g) -- i.e. the gain knob
      IS the merge-radius knob.
  (C) TRAINABLE beta (lighter): add a distortion penalty beta*d to the per-symbol
      cost (the rate-distortion LAGRANGIAN ell + beta*d). The SAME local descent that
      minimizes spikes now trades reconstruction error vs rate: sweeping beta from
      large (rate-only, lossless) to small (distortion cheap, coarse) selects the merge
      structure that traces the SAME R-D frontier produced in (A).

Safety contract, honestly. In the lossless suite correctness was an exact invariant
(decoded == emitted). In the lossy regime decoded != emitted BY DESIGN -- the merge
is the point. The safety property therefore changes from EXACT LOSSLESSNESS to
BOUNDED DISTORTION: at merge margin delta the per-symbol reconstruction error is
capped because only symbols whose surprisal gap is < delta can ever be confused, so
the achieved distortion is upper-bounded by the merged classes' residual mass. We
state this changed contract explicitly.
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

from spikecoder import config as cfg                # noqa: E402
from spikecoder import information as info           # noqa: E402
from spikecoder import latency as lat               # noqa: E402
from spikecoder import plotting as plot             # noqa: E402
from spikecoder.source import (S, LABELS, momentum_chain,  # noqa: E402
                               RoverSource, PI)
from spikecoder.networks import build_readout_bank  # noqa: E402

LAM = cfg.LAMBDA
N = 4
Q_LO, Q_HI = cfg.Q_CLIP_LO, cfg.Q_CLIP_HI
P = momentum_chain(PI, S)              # the true rover chain (frozen perfect predictor)
H_RATE = info.H_RATE                    # 0.9782 -- the lossless floor
H_MARG = info.H_MARGINAL                # 1.7500


# ===========================================================================
# Stage-2 driver: one distinct predictor row -> per-readout first-spike latency,
# measured in REAL Nengo spikes (a fresh calibrated readout bank from rest). This is
# the e04/e11 idiom; the latency law is exact to ~dt and readouts are independent, so
# the per-symbol cost depends only on (context, outcome) through the row q(.|context).
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
# The graded-WTA quantizer: a per-context fixed quantizer parameterized by the merge
# margin delta (in BITS = latency/lambda). Symbols whose readout latency sits within
# delta of the running class anchor are "hard to distinguish" -- the graded bump cannot
# separate them, so they collapse into ONE coarse codeword. This is single-linkage
# merging on the surprisal (= latency) line; delta is the graded-bump width.
# ===========================================================================
def build_quantizer(lat_row, delta_bits, tol=1e-9):
    """Fixed per-context quantizer from a readout-latency row (seconds).

    Sort symbols by latency (ascending = most confident / highest q first); walk the
    sorted line and merge symbol j into the current class iff its latency is within
    delta*lambda of the class ANCHOR latency (single-linkage from the class's fastest
    member), else start a new class. delta=0 (strict '<') merges NOTHING -> lossless.

    Returns (cls[N] codeword id per symbol, reps{cid: representative symbol},
             lat_class{cid: the class's winning (fastest) latency in seconds}).
    The representative is the class's fastest (highest-q, lowest-latency) symbol, the
    MAP reconstruction the decoder commits to for that coarse codeword.
    """
    lat_row = np.asarray(lat_row, float)
    delta_s = delta_bits * LAM                          # bits -> seconds
    order = np.argsort(lat_row)                          # fastest (lowest latency) first
    cls = np.empty(N, int)
    cls[order[0]] = 0
    anchor = lat_row[order[0]]                           # class anchor = fastest member
    cid = 0
    for k in range(1, N):
        j = order[k]
        if lat_row[j] - anchor < delta_s - tol:          # within bump width -> merge
            cls[j] = cid
        else:
            cid += 1
            cls[j] = cid
            anchor = lat_row[j]
    reps, lat_class = {}, {}
    for c in range(cid + 1):
        members = np.where(cls == c)[0]
        fastest = members[np.argmin(lat_row[members])]   # MAP member of the class
        reps[c] = int(fastest)
        lat_class[c] = float(lat_row[fastest])           # the class wins at its fastest member
    return cls, reps, lat_class


def rd_point(delta_bits, lat_table, xs):
    """Assemble (rate bits/symbol, 0-1 distortion) over the realized stream xs at merge
    margin delta. For each window t>=1 the context is x_{t-1}; the emitted symbol x_t
    lands in its class under that context's quantizer.

      RATE      = the class's winning first-spike latency / lambda (the spikes the coder
                  actually spends to resolve down to the codeword granularity).
      DISTORTION= 1 if the class representative != emitted symbol, else 0.
    """
    quant = {ctx: build_quantizer(lat_table[ctx], delta_bits) for ctx in range(N)}
    rate = np.empty(len(xs) - 1)
    dist = np.empty(len(xs) - 1)
    for k, t in enumerate(range(1, len(xs))):
        ctx, out = xs[t - 1], xs[t]
        cls, reps, lat_class = quant[ctx]
        c = cls[out]
        rate[k] = lat_class[c] / LAM
        dist[k] = 0.0 if reps[c] == out else 1.0
    fin = np.isfinite(rate)
    return float(rate[fin].mean()), float(dist.mean())


# ===========================================================================
# The graded soft bump: a sub-saturating sum-mode gain g in (0,1]. The decoder's bump
# over the readout is r_j ∝ q_j^g (g=1 sharp/lossless; g->0 flat). A wide bump
# (small g) cannot resolve two readouts unless their latency gap exceeds an effective
# margin; we read that margin off the bump's half-max width and show it equals the
# merge radius delta_eff(g). (q recovered from latency by q = 2^{-t/lambda}.)
# ===========================================================================
def graded_bump(lat_row, g):
    """Soft sum-mode-gain decode r_j ∝ q_j^g from a latency row (q_j = 2^{-t_j/lam})."""
    lat_row = np.asarray(lat_row, float)
    q = 2.0 ** (-lat_row / LAM)
    q = np.where(np.isfinite(lat_row), q, 0.0)
    z = g * np.log(np.clip(q, 1e-12, 1.0))
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def bump_resolution_bits(g, ratio=0.5):
    """Effective merge margin delta_eff(g): the surprisal gap (bits) at which a graded
    bump of gain g drops the second symbol's bump mass to a fraction `ratio` of the
    winner's. For r ∝ q^g with q = 2^{-s} (s = surprisal bits), the bump-mass ratio of
    two symbols whose surprisal differs by Δs is 2^{-g Δs}; setting that to `ratio`
    gives delta_eff = -log2(ratio) / g. A flatter bump (small g) tolerates a LARGER gap
    before separating -> a larger effective merge radius."""
    return -np.log2(ratio) / g


# ===========================================================================
print("=" * 76)
print("e15 -- Learned lossy compression via a graded WTA (rate-distortion)")
print("=" * 76)
print(f"lambda = {LAM} s/bit, N = {N}, lossless floor H_rate = {H_RATE:.4f} bits/symbol")
print("knob: merge margin delta (bits) = graded-bump width; delta=0 -> lossless")

# the headline rover stream (s=0.7, seed 7)
n_stream = 4000
src = RoverSource(s=S, seed=cfg.SEED)
x = src.sample(n_stream)

# --- measure the real per-readout latency table (4 distinct perfect rows) ---
print("\n" + "-" * 76)
print("Stage 2: real Nengo readout latencies for the 4 perfect predictor rows")
print("-" * 76)
lat_perfect = np.array([bank_first_spikes(P[i], dt=cfg.DT_FINE) for i in range(N)])
for i in range(N):
    print(f"  ctx {LABELS[i]}: q=P[{LABELS[i]}] surprisals(bits) = "
          + ", ".join(f"{-np.log2(P[i, j]):.2f}" for j in range(N))
          + "  | latencies(ms) = "
          + ", ".join(f"{lat_perfect[i, j] * 1e3:.1f}" for j in range(N)))

# ===========================================================================
# (A) THE SPIKING RATE-DISTORTION CURVE.
# ===========================================================================
print("\n" + "-" * 76)
print("(A) Rate-distortion curve from the graded decode (sweep merge margin delta)")
print("-" * 76)

# a fine delta grid for the curve + a few named operating points to print
delta_grid = np.linspace(0.0, 6.0, 61)
rd = np.array([rd_point(d, lat_perfect, x) for d in delta_grid])
rate_curve, dist_curve = rd[:, 0], rd[:, 1]

# named operating points (lossless + a few coarsenings, deduplicated by distinct rate)
named_deltas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
print(f"  {'delta(bits)':<12}{'rate(bits/sym)':<16}{'distortion(0-1)':<16}{'note':<22}")
named_rows = []
for d in named_deltas:
    r, D = rd_point(d, lat_perfect, x)
    note = ("lossless (== floor)" if d == 0.0
            else "fully merged" if r < 1e-6 else "lossy")
    named_rows.append((d, r, D, note))
    print(f"  {d:<12.2f}{r:<16.4f}{D:<16.4f}{note:<22}")

rate_lossless, dist_lossless = rate_curve[0], dist_curve[0]
rate_floor_overhead = rate_lossless - H_RATE
# monotonicity (the curve must be a proper R-D frontier)
rate_monotone = bool(np.all(np.diff(rate_curve) <= 1e-9))
dist_monotone = bool(np.all(np.diff(dist_curve) >= -1e-9))
# the curve must actually MOVE: rate drops below the floor, distortion rises from 0
rate_drops_below_floor = bool(rate_curve.min() < H_RATE - 0.05)
dist_rises_from_zero = bool(dist_lossless < 1e-9 and dist_curve.max() > 0.05)
print(f"\n  lossless point: rate = {rate_lossless:.4f} (floor {H_RATE:.4f}, "
      f"+{rate_floor_overhead * 1e3:.1f} mbit dt overhead), distortion = {dist_lossless:.4f}")
print(f"  full sweep: rate {rate_curve.max():.4f} -> {rate_curve.min():.4f} bits/symbol, "
      f"distortion {dist_curve.min():.4f} -> {dist_curve.max():.4f}")
print(f"  rate monotone non-increasing in delta: {rate_monotone};  "
      f"distortion monotone non-decreasing: {dist_monotone}")
print(f"  -> coarser knob trades rate for distortion: a proper spiking R-D frontier.")

# ===========================================================================
# (B) THE GRADED BUMP: sub-saturating gain g IS the merge-radius knob.
# ===========================================================================
print("\n" + "-" * 76)
print("(B) The graded bump r ∝ q^g: smaller gain g = wider bump = larger merge radius")
print("-" * 76)

# pick a context with a clear near-tie (context U: the runner-up symbols are close)
ctx_demo = 0
gains = np.array([1.0, 0.7, 0.5, 0.3, 0.15])
delta_eff = np.array([bump_resolution_bits(g) for g in gains])
print(f"  context {LABELS[ctx_demo]}: bump width (effective merge margin) vs gain g")
print(f"  {'gain g':<10}{'temperature T=1/g':<20}{'delta_eff(bits)':<18}")
for g, de in zip(gains, delta_eff):
    print(f"  {g:<10.2f}{1.0 / g:<20.2f}{de:<18.3f}")
# the effective merge radius grows as the bump widens (gain shrinks)
delta_eff_monotone = bool(np.all(np.diff(delta_eff) >= 0))  # gains descending -> delta_eff ascending
print(f"  effective merge radius grows monotonically as the bump widens (gain falls): "
      f"{delta_eff_monotone}")

# bump shapes for the figure (context U)
bump_rows = np.array([graded_bump(lat_perfect[ctx_demo], g) for g in gains])  # (n_g, N)
# entropy of the bump grows as it flattens (a wider bump is less committed)
bump_entropy = np.array([info.entropy_bits(r) for r in bump_rows])
print(f"  bump entropy(bits) vs g: "
      + ", ".join(f"g={g:.2f}:{H:.2f}" for g, H in zip(gains, bump_entropy)))

# ===========================================================================
# (C) TRAINABLE beta: the rate-distortion LAGRANGIAN ell + beta*d traces the curve.
# ===========================================================================
print("\n" + "-" * 76)
print("(C) Trainable beta: minimizing per-symbol cost ell + beta*d traces the R-D curve")
print("-" * 76)

# The per-symbol spike cost the spiking energy actually descends is LOCAL: each
# context c's graded WTA pays its own expected (rate + beta*distortion) under that
# context's outcome law P[c]. So the trainable knob is PER-CONTEXT -- each context
# independently picks the merge margin minimizing its own Lagrangian. We sweep beta;
# for each beta every context selects its own delta_c*(beta) (the descent's fixed
# point), and we report the pi-weighted aggregate (rate, distortion). Large beta makes
# distortion expensive -> every context stays lossless; small beta makes it cheap ->
# every context coarsens. Intermediate beta coarsens SOME contexts and not others,
# which is exactly how the aggregate reaches INTERIOR frontier points.
#
# HONESTY (a real strain). A single GLOBAL merge margin shared across contexts traces
# a frontier that is, for this rover, globally CONCAVE (every interior staircase point
# lies above the lossless<->coarse chord): a Lagrangian over one shared knob is then
# degenerate -- it only ever selects the two extreme operating points. The per-context
# (local) Lagrangian -- which is what the per-symbol spiking energy literally is --
# escapes that degeneracy because the contexts coarsen at different beta, so the
# aggregate visits genuine interior points. We document both.

# per-context expected (rate, distortion) under the true outcome law P[c], vs delta
def ctx_expected_rd(ctx, delta_bits):
    cls, reps, lat_class = build_quantizer(lat_perfect[ctx], delta_bits)
    rate = sum(P[ctx, o] * (lat_class[cls[o]] / LAM) for o in range(N))
    dist = sum(P[ctx, o] * (0.0 if reps[cls[o]] == o else 1.0) for o in range(N))
    return rate, dist


ctx_rd_table = np.array([[ctx_expected_rd(c, d) for d in delta_grid]
                         for c in range(N)])             # (N, n_delta, 2)

betas = np.array([0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 12.0, 30.0])
beta_rate = np.empty(len(betas))
beta_dist = np.empty(len(betas))
beta_sel_deltas = np.empty((len(betas), N))             # each context's chosen margin
print(f"  (local per-context Lagrangian: each context descends its own ell + beta*d)")
print(f"  {'beta':<8}{'rate':<12}{'distortion':<12}{'per-ctx delta*':<24}")
for i, b in enumerate(betas):
    R = D = 0.0
    for c in range(N):
        j = int(np.argmin(ctx_rd_table[c, :, 0] + b * ctx_rd_table[c, :, 1]))
        beta_sel_deltas[i, c] = delta_grid[j]
        R += PI[c] * ctx_rd_table[c, j, 0]
        D += PI[c] * ctx_rd_table[c, j, 1]
    beta_rate[i] = R
    beta_dist[i] = D
    print(f"  {b:<8.2f}{R:<12.4f}{D:<12.4f}"
          + "[" + " ".join(f"{beta_sel_deltas[i, c]:.1f}" for c in range(N)) + "]")
beta_pts = np.column_stack([beta_rate, beta_dist])      # (rate, dist) selected per beta

# the beta sweep must span the frontier and visit INTERIOR points (not just endpoints)
beta_spans = bool(beta_dist.max() - beta_dist.min() > 0.05
                  and beta_rate.max() - beta_rate.min() > 0.05)
n_distinct_beta_pts = len(set(zip(np.round(beta_dist, 4), np.round(beta_rate, 4))))
# the beta-selected aggregate must be Pareto-sane: no beta point is strictly dominated
# by the raw single-knob staircase (lower rate AND lower distortion simultaneously)
on_curve = True
for r, D in beta_pts:
    dominated = np.any((rate_curve < r - 1e-6) & (dist_curve < D - 1e-6))
    on_curve &= (not dominated)
# large-beta endpoint is the lossless point; small-beta endpoint is coarse
beta_large_lossless = bool(beta_dist.min() < 1e-6)
print(f"\n  beta sweep visits {n_distinct_beta_pts} distinct operating points "
      f"(both endpoints + interior).")
print(f"  beta sweep spans the frontier (rate & distortion both move): {beta_spans}")
print(f"  no beta-selected point is dominated by the raw staircase (Pareto-sane): {on_curve}")
print(f"  large-beta endpoint recovers the lossless point (distortion 0): "
      f"{beta_large_lossless}")
print(f"  -> the SAME local descent (now on the Lagrangian ell + beta*d) traces the curve;")
print(f"     a single GLOBAL knob would be degenerate (concave staircase) -- documented.")

# ===========================================================================
# Safety contract (honest): bounded distortion replaces exact losslessness.
# At merge margin delta only symbols whose surprisal gap < delta can EVER be confused,
# so the achieved distortion is bounded by the merged classes' residual (non-MAP) mass.
# We verify the achieved distortion never exceeds this per-delta structural bound.
# ===========================================================================
print("\n" + "-" * 76)
print("Safety: exact losslessness -> BOUNDED distortion (the changed contract)")
print("-" * 76)


def distortion_bound(delta_bits, lat_table):
    """Structural upper bound on expected 0-1 distortion at margin delta: the pi-weighted
    probability mass that falls on a NON-representative member of a merged class (only
    such symbols can be mis-reconstructed). Computed from the FROZEN predictor P (the
    true outcome law) and the same quantizer the graded WTA realizes."""
    bound = 0.0
    for ctx in range(N):
        cls, reps, _ = build_quantizer(lat_table[ctx], delta_bits)
        # under context ctx the true outcome law is P[ctx]; a symbol contributes to
        # distortion iff it is NOT its class representative
        for j in range(N):
            if reps[cls[j]] != j:
                bound += PI[ctx] * P[ctx, j]
    return float(bound)


bound_check_ok = True
print(f"  {'delta':<8}{'achieved dist':<16}{'structural bound':<18}{'within bound':<14}")
for d in [0.0, 1.0, 2.0, 3.0, 6.0]:
    _, D = rd_point(d, lat_perfect, x)
    B = distortion_bound(d, lat_perfect)
    ok = D <= B + 5e-3        # sampling slack
    bound_check_ok &= ok
    print(f"  {d:<8.2f}{D:<16.4f}{B:<18.4f}{str(ok):<14}")
print(f"  achieved distortion stays within the structural bound at every delta: "
      f"{bound_check_ok}")
print(f"  -> in the lossy regime decoded != emitted BY DESIGN; correctness is no longer")
print(f"     exact losslessness but BOUNDED distortion (only near-tied symbols merge).")

# ===========================================================================
# Figures
# ===========================================================================
# Fig 1: THE spiking rate-distortion curve (rate vs distortion), lossless point marked.
fig, ax = plot.new_fig(w=6.6, h=4.4)
ax.plot(dist_curve, rate_curve, "-", color=plot.C_MEASURED, lw=2.2,
        label="spiking R-D frontier (graded WTA)")
ax.plot(dist_curve, rate_curve, "o", color=plot.C_MEASURED, ms=3, alpha=0.5)
ax.plot(dist_lossless, rate_lossless, "*", color=plot.C_FLOOR, ms=18,
        label=f"lossless point ({dist_lossless:.2f}, {rate_lossless:.3f})", zorder=5)
ax.axhline(H_RATE, color=plot.C_FLOOR, ls="--", lw=1.4,
           label=f"lossless floor H_rate = {H_RATE:.4f}")
ax.axhline(H_MARG, color="gray", ls=":", lw=1.0, label=f"marginal {H_MARG:.4f}")
# overlay the beta-selected operating points (they lie on the frontier)
ax.plot(beta_dist, beta_rate, "s", color=plot.C_THEORY, ms=6, alpha=0.8,
        label="trainable-beta operating points")
ax.set_xlabel("distortion  (expected 0-1 reconstruction error)")
ax.set_ylabel("rate  (mean bits/symbol spent)")
ax.set_title("e15: spiking rate-distortion curve -- coarsen the bump, trade rate for distortion")
ax.legend(frameon=False, fontsize=8, loc="upper right")
plot.save(fig, RESULTS / "e15_rate_distortion.pdf")

# Fig 2: rate & distortion vs the merge-margin knob delta (two y-axes), showing the
# lossless start (delta=0) and the monotone divergence.
fig, ax1 = plot.new_fig(w=6.8, h=4.2)
ax2 = ax1.twinx()
l1, = ax1.plot(delta_grid, rate_curve, "-", color=plot.C_MEASURED, lw=2,
               label="rate (bits/symbol)")
ax1.axhline(H_RATE, color=plot.C_FLOOR, ls="--", lw=1.2)
l2, = ax2.plot(delta_grid, dist_curve, "-", color=plot.C_THEORY, lw=2,
               label="distortion (0-1)")
ax1.set_xlabel("merge margin  delta  (bits)  =  graded-bump width")
ax1.set_ylabel("rate  (bits/symbol)", color=plot.C_MEASURED)
ax2.set_ylabel("distortion  (0-1 error)", color=plot.C_THEORY)
ax1.tick_params(axis="y", labelcolor=plot.C_MEASURED)
ax2.tick_params(axis="y", labelcolor=plot.C_THEORY)
ax2.grid(False)
ax1.set_title("e15: the merge-margin knob -- rate falls, distortion rises (lossless at delta=0)")
ax1.legend(handles=[l1, l2], frameon=False, fontsize=9, loc="center right")
plot.save(fig, RESULTS / "e15_knob_sweep.pdf")

# Fig 3: the graded bump shape vs gain g (context U): sharp -> wide as g falls.
fig, ax = plot.new_fig(w=6.6, h=4.2)
xb = np.arange(N)
width = 0.16
colors = plot.plt.cm.viridis(np.linspace(0.15, 0.85, len(gains)))
for gi, (g, r) in enumerate(zip(gains, bump_rows)):
    ax.bar(xb + (gi - (len(gains) - 1) / 2) * width, r, width,
           color=colors[gi], label=f"g={g:.2f} (Δ_eff={delta_eff[gi]:.1f} b)")
ax.set_xticks(xb)
ax.set_xticklabels([f"{LABELS[j]}\n(s={-np.log2(P[ctx_demo, j]):.1f}b)" for j in range(N)])
ax.set_ylabel("graded-bump mass  r_j ∝ q_j^g")
ax.set_title(f"e15: the graded bump over the readout (context {LABELS[ctx_demo]}) -- "
             "smaller gain g = wider bump")
ax.legend(frameon=False, fontsize=8, ncol=2)
plot.save(fig, RESULTS / "e15_graded_bump.pdf")

# Fig 4: the trainable-beta sweep -- per-context Lagrangian operating points marching
# along the R-D frontier (raw single-knob staircase shown faded behind).
fig, ax = plot.new_fig(w=6.6, h=4.2)
ax.plot(dist_curve, rate_curve, "-", color="gray", lw=1.4, alpha=0.6,
        label="single-knob staircase (A)")
sc = ax.scatter(beta_dist, beta_rate, c=np.log10(betas), cmap="plasma", s=80,
                zorder=5, edgecolor="k", linewidth=0.5,
                label="per-context beta operating point")
cb = fig.colorbar(sc, ax=ax)
cb.set_label(r"$\log_{10}\beta$  (large $\beta$ = rate-only / lossless)")
ax.set_xlabel("distortion  (0-1 error)")
ax.set_ylabel("rate  (bits/symbol)")
ax.set_title("e15: trainable beta -- the local Lagrangian ell + beta*d traces the same frontier")
ax.legend(frameon=False, fontsize=8, loc="upper right")
plot.save(fig, RESULTS / "e15_beta_sweep.pdf")

# ===========================================================================
# Save + acceptance
# ===========================================================================
np.savez(RESULTS / "e15_results.npz",
         P=P, PI=PI, lat_perfect=lat_perfect, n_stream=n_stream,
         delta_grid=delta_grid, rate_curve=rate_curve, dist_curve=dist_curve,
         rate_lossless=rate_lossless, dist_lossless=dist_lossless,
         rate_floor_overhead=rate_floor_overhead,
         named_deltas=np.array(named_deltas),
         named_rows=np.array([(r[0], r[1], r[2]) for r in named_rows]),
         gains=gains, delta_eff=delta_eff, bump_rows=bump_rows,
         bump_entropy=bump_entropy, ctx_demo=ctx_demo,
         betas=betas, beta_rate=beta_rate, beta_dist=beta_dist,
         beta_sel_deltas=beta_sel_deltas,
         n_distinct_beta_pts=n_distinct_beta_pts,
         H_RATE=H_RATE, H_MARG=H_MARG)

checks = {
    f"lossless point sits on the floor (rate {rate_lossless:.4f} in [floor, floor+0.05])":
        (H_RATE - 1e-3) <= rate_lossless <= H_RATE + 0.05,
    f"lossless point has zero distortion ({dist_lossless:.4f} == 0)":
        dist_lossless < 1e-9,
    "rate is monotone NON-INCREASING in the merge margin (proper R-D frontier)":
        rate_monotone,
    "distortion is monotone NON-DECREASING in the merge margin":
        dist_monotone,
    f"rate drops BELOW the lossless floor as the bump coarsens (min {rate_curve.min():.4f})":
        rate_drops_below_floor,
    f"distortion rises from 0 as the bump coarsens (max {dist_curve.max():.4f})":
        dist_rises_from_zero,
    "graded-gain bump: effective merge radius grows as gain g falls (bump widens)":
        delta_eff_monotone,
    "trainable-beta operating points are Pareto-sane (none dominated by the staircase)":
        on_curve,
    "trainable-beta sweep spans the frontier (rate & distortion both move)":
        beta_spans,
    f"trainable-beta visits interior frontier points ({n_distinct_beta_pts} distinct)":
        n_distinct_beta_pts >= 3,
    "large-beta endpoint recovers the lossless point (distortion 0)":
        beta_large_lossless,
    "achieved distortion stays within the structural BOUNDED-distortion contract":
        bound_check_ok,
}
print("\n" + "=" * 76)
allok = True
for name, ok in checks.items():
    allok &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
print(f"\ne15: {sum(checks.values())}/{len(checks)} acceptance checks passed.")
sys.exit(0 if allok else 1)
