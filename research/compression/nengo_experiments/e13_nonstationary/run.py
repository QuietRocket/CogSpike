#!/usr/bin/env python3
"""e13 -- Tracking a drifting source: the constant-lr noise ball as a FEATURE.

Capstone claim (paper, learning-theory outlook). The convergence theorem of e09/e10
assumed a FIXED source and a DECREASING (Robbins-Monro) step: V -> 0 almost surely.
A DRIFTING world instead wants a CONSTANT step. It trades almost-sure convergence
for a steady-state O(eta) noise ball -- and that ball is exactly what makes the
learner ADAPT. A constant step keeps a residual sensitivity that a vanishing step
throws away; when the world changes, the constant-step learner re-tracks, the
decreasing-step learner FREEZES.

The momentum rover is a purpose-built, perfectly controlled non-stationarity:
switching the stickiness s holds the marginal pi = (1/2,1/4,1/8,1/8) FIXED (so the
memoryless cost H(pi) = 1.7500 never moves) and moves ONLY the conditional
structure -- the thing the predictive cycle / the learned decoders capture. The
entropy-rate FLOOR, however, moves with s:

    s = 0.70  ->  H_rate = 0.9782 bits/symbol   (the e09/e10 operating point)
    s = 0.40  ->  H_rate = 1.4852 bits/symbol   (diffuse: weak momentum)
    s = 0.85  ->  H_rate = 0.5959 bits/symbol   (concentrated: strong momentum)

We switch s along the schedule 0.70 -> 0.40 -> 0.85 -> 0.70 at regular intervals.
After each switch the energy JUMPS (excess KL appears: the model is now wrong for
the new s) and then RE-DESCENDS to the NEW floor H_rate(s).

FIVE measurements:
  (1) SPIKING RE-TRACKING (centerpiece): a Nengo PES learner (the e09 idiom) at a
      CONSTANT learning rate, on the switching stream. The decoded energy tracks
      the moving floor: it jumps at each switch and re-descends to H_rate(s_new).
      We mark each per-segment floor on the energy-vs-time plot.
  (2) TRACKING TIME-CONSTANT: after each switch, fit excess(t) ~ ss + A e^{-t/tau}
      and report tau (the adaptation speed) and the steady-state tracking error ss.
      (Measured on the exact numpy delta-rule twin -- the float64 analogue of the
      spiking PES decoder -- so the small tail residual is readable.)
  (3) STEADY-STATE ERROR SCALES WITH lr (the O(eta) ball, now as ADAPTATION budget):
      sweep the constant lr; the converged steady-state tracking error grows ~O(eta).
      Smaller step = tighter ball = slower tracking; a real bias/variance dial.
  (4) DECREASING / TINY lr FREEZES (the motivation): a Robbins-Monro decreasing
      step lr0/(1+t/t0) re-tracks the EARLY switches (large lr early) but, once the
      step has decayed, FREEZES and FAILS to re-track a LATE switch (the s=0.40->0.85
      sharpening): its excess stays large where the constant-lr learner re-descends.
      A tiny CONSTANT step is even worse -- too small to descend at all.
  (5) THE TRADE-OFF, named: constant-lr keeps adaptation at the price of a permanent
      O(eta) ball; decreasing-lr buys an arbitrarily small ball at the price of
      adaptation. The noise ball is not a bug -- on a drifting source it is the
      mechanism.

Twin policy: the SPIKING PES learner carries claim (1) (energy re-tracks the moving
floor, physically, on the substrate). The numpy delta-rule twin -- the exact float64
analogue of the learned spiking decoder, W[i] += lr*(onehot(j)-softmax(W[i])) --
carries (2) time-constants, (3) the lr-sweep, and (4) the freeze, because those need
a clean, denoised tail residual and many runs that a multi-hour LIF sweep cannot
afford. Both obey the same delta rule; we state which carries which.

Outputs: three figures + a results .npz, and a PASS/FAIL line per acceptance check.
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

from spikecoder import config as cfg                              # noqa: E402
from spikecoder import plotting as plot                           # noqa: E402
from spikecoder.source import RoverSource, momentum_chain, PI, LABELS  # noqa: E402
from spikecoder.information import (softmax, entropy_rate, model_energy_and_kl,  # noqa: E402
                                    H_MARGINAL, H_RATE)
from spikecoder.metrics import energy_trajectory                  # noqa: E402

# --- the switching schedule (pi FIXED; only the conditional / floor moves) ---
SCHEDULE = [0.70, 0.40, 0.85, 0.70]          # stickiness s per segment
LATE_SWITCH_SEG = 2                          # the 0.40 -> 0.85 sharpen (the freeze test)
WINDOW = 0.03                                # s per symbol window
DT = cfg.DT                                  # 1e-3 (throughput/learning)
SEED = cfg.SEED
N_CTX = 600
CTX_RADIUS = 1.3

# spiking run: 2500 symbols/segment x 4 = 10000 symbols (the e09/e10 budget)
SEG_LEN_SPIKE = 2500
LR_SPIKE = 1.5e-3                            # constant PES lr (re-tracks inside a segment)

# numpy twin: longer segments so the tail steady-state is clean
SEG_LEN_NP = 6000
LR_NP = 0.02                                 # constant delta-rule lr (the twin's healthy step)
TAIL_FRAC = 0.60                             # tail fraction used for steady-state reads

# per-segment transition matrices + moving floors (pi held fixed)
PS = [momentum_chain(PI, s) for s in SCHEDULE]
FLOORS = [entropy_rate(P, PI) for P in PS]


# ===========================================================================
# the exact numpy delta-rule twin on the SWITCHING stream
#   W[i] += lr*(onehot(j) - softmax(W[i])) -- the float64 analogue of the PES decoder
# ===========================================================================
def make_switch_stream(seg_len, seed=SEED):
    """Sample a stream whose stickiness s switches every seg_len symbols.

    pi is held fixed across all segments; only the conditional law P (and hence the
    entropy-rate floor) moves. Returns (x, seg) with seg[t] the active segment index.
    """
    n = seg_len * len(SCHEDULE)
    rng = np.random.default_rng(seed)
    x = np.empty(n, dtype=int)
    seg = np.empty(n, dtype=int)
    x[0] = rng.choice(4, p=PI)
    seg[0] = 0
    for t in range(1, n):
        s = min(t // seg_len, len(SCHEDULE) - 1)
        seg[t] = s
        x[t] = rng.choice(4, p=PS[s][x[t - 1]])
    return x, seg


def run_numpy_twin(lr_fn, seg_len, seed=SEED):
    """Online delta rule on the switching stream; record per-step excess KL against
    the CURRENT segment's true law (so the excess is honest 'how wrong am I NOW')."""
    x, seg = make_switch_stream(seg_len, seed)
    n = len(x)
    eye = np.eye(4)
    W = np.zeros((4, 4))                       # uniform init (energy starts at 2.0)
    E = np.empty(n)
    exc = np.empty(n)
    for t in range(n):
        s = seg[t]
        Et, klt = model_energy_and_kl(W, PI, PS[s], FLOORS[s], is_logits=True)
        E[t] = Et
        exc[t] = klt
        i = x[t - 1] if t > 0 else x[0]
        j = x[t]
        W[i] += lr_fn(t) * (eye[j] - softmax(W[i]))
    return seg, E, exc


def tail_excess(seg, exc, s_i, frac=TAIL_FRAC):
    """Steady-state (tail) excess of segment s_i."""
    idx = np.flatnonzero(seg == s_i)
    tail = idx[int(frac * len(idx)):]
    return float(exc[tail].mean())


def fit_time_constant(exc, t_switch, seg_len, bin_w=60):
    """After a switch at t_switch, fit excess(t) ~ ss + A e^{-t/tau} over the segment.

    Bin to denoise, estimate ss from the segment tail, log-fit the decaying part.
    Returns (tau in symbols, steady-state ss, binned curve, bin centers, fit curve).
    """
    w = exc[t_switch:t_switch + seg_len]
    nb = (len(w) // bin_w) * bin_w
    b = w[:nb].reshape(-1, bin_w).mean(1)
    centers = np.arange(len(b)) * bin_w
    ss = float(b[-max(8, len(b) // 6):].mean())
    y = b - ss
    mask = y > (y[0] * 0.05)                    # decaying portion above the floor
    if mask.sum() < 3:
        return np.nan, ss, b, centers, None
    coef = np.polyfit(centers[mask], np.log(np.clip(y[mask], 1e-9, None)), 1)
    tau = float(-1.0 / coef[0]) if coef[0] < 0 else np.nan
    fit = ss + np.exp(coef[1]) * np.exp(coef[0] * centers)
    return tau, ss, b, centers, fit


def numpy_lr_sweep(lrs, n=80_000, tail=0.40, nseed=6):
    """The O(eta) steady-state ball at s=0.70, isolated cleanly in numpy.

    Run the exact delta rule to CONVERGENCE on the stationary s=0.70 source and
    tail-average the excess over nseed streams (to denoise the small residual). For
    converged rates the tail residual is the BALL, and it scales as O(eta): smaller
    (still-converged) step -> smaller steady-state tracking error. This is the same
    asymptotic-face reasoning as e10 -- fast rates that settle inside the budget."""
    P = momentum_chain(PI, 0.70)
    floor = entropy_rate(P, PI)
    eye = np.eye(4)
    mark = int((1 - tail) * n)
    out = {}
    for lr in lrs:
        vals = []
        for sd in range(nseed):
            rng = np.random.default_rng(400 + sd)
            x = np.empty(n, dtype=int)
            x[0] = rng.choice(4, p=PI)
            for t in range(1, n):
                x[t] = rng.choice(4, p=P[x[t - 1]])
            W = np.zeros((4, 4))
            ex = []
            for t in range(n):
                i = x[t - 1] if t > 0 else x[0]
                j = x[t]
                z = W[i] - W[i].max()
                e = np.exp(z)
                W[i] += lr * (eye[j] - e / e.sum())
                if t > mark:
                    ex.append(model_energy_and_kl(W, PI, P, floor)[1])
            vals.append(float(np.mean(ex)))
        out[lr] = float(np.mean(vals))
    return out


# ===========================================================================
# the spiking PES learner on the SWITCHING stream (the e09 idiom, constant lr)
# ===========================================================================
def build_pes_net(ctx_fn, y_fn, lr, seed=SEED):
    """One 600-neuron LIF context population whose DECODERS are learned by PES at a
    CONSTANT rate lr, on the switching stream. Uniform init (energy starts at 2.0);
    error = q - y; PES descends -lr*error*activity, driving q toward E[y|c] = P of
    whatever segment is currently active."""
    net = nengo.Network(seed=seed)
    with net:
        ci = nengo.Node(ctx_fn, size_out=4)
        ctx = nengo.Ensemble(N_CTX, 4, radius=CTX_RADIUS)
        nengo.Connection(ci, ctx, synapse=0.005)

        q = nengo.Node(size_in=4)                       # decoded prediction q(.|c)
        conn = nengo.Connection(
            ctx, q, function=lambda c: np.ones(4) * 0.25,   # uniform init
            learning_rule_type=nengo.PES(learning_rate=lr), synapse=0.01)

        yi = nengo.Node(y_fn, size_out=4)               # teacher: one-hot realized symbol
        err = nengo.Node(size_in=4)                     # error = q - y
        nengo.Connection(q, err, synapse=None)
        nengo.Connection(yi, err, transform=-1, synapse=None)
        nengo.Connection(err, conn.learning_rule, synapse=None)

        pq = nengo.Probe(q, synapse=0.02)
        pctx = nengo.Probe(ci)
    return net, pq, pctx


def per_window_q_and_ctx(qd, cd, n_symbols):
    """Reduce raw probes to one settled-tail q and one context index per window."""
    win_q = np.zeros((n_symbols, 4))
    win_ctx = np.full(n_symbols, -1)
    for k in range(n_symbols):
        i0 = int(round(k * WINDOW / DT))
        i1 = int(round((k + 1) * WINDOW / DT))
        if i1 > len(qd):
            break
        seg = qd[i0:i1]
        win_q[k] = seg[len(seg) // 2:].mean(0)          # settled tail
        csum = cd[i0:i1].sum(0)
        if csum.max() > 0:
            win_ctx[k] = int(np.argmax(csum))
    return win_q, win_ctx


def learned_matrix_window(win_q, win_ctx, lo, hi):
    """Average decoded q per context over windows [lo, hi); clip>0 & renormalize."""
    rows = {i: [] for i in range(4)}
    for k in range(lo, hi):
        if win_ctx[k] >= 0:
            rows[win_ctx[k]].append(win_q[k])
    Q = np.ones((4, 4)) / 4.0
    for i in range(4):
        if rows[i]:
            v = np.clip(np.mean(rows[i], axis=0), 1e-6, None)
            Q[i] = v / v.sum()
    return Q


def spiking_energy_trace(win_q, win_ctx, seg_spike, n_symbols, sub=40):
    """Trailing-window decoded energy vs symbol index, measured against the CURRENT
    segment's true law. A short trailing window (sub symbols) makes the post-switch
    JUMP and RE-DESCENT visible without over-smoothing across a boundary."""
    steps, Es = [], []
    for k in range(sub, n_symbols):
        s = seg_spike[k]
        Q = learned_matrix_window(win_q, win_ctx, k - sub, k)
        Es.append(energy_trajectory([Q], PI, PS[s])[0])
        steps.append(k)
    return np.array(steps), np.array(Es)


# ===========================================================================
print("=" * 70)
print("e13 -- Tracking a drifting source: the constant-lr noise ball as a FEATURE")
print("=" * 70)
print(f"\nschedule s = {SCHEDULE} (pi FIXED at {tuple(PI)}; only the conditional moves)")
print(f"marginal H(pi) = {H_MARGINAL:.4f} bits/symbol -- INVARIANT across all switches")
print("moving entropy-rate floors:")
for s, fl in zip(SCHEDULE, FLOORS):
    print(f"  s={s:.2f}  ->  H_rate = {fl:.4f} bits/symbol")

# ---------------------------------------------------------------------------
# (1) SPIKING re-tracking (centerpiece): constant-lr PES on the switching stream
# ---------------------------------------------------------------------------
print(f"\n[spiking PES] constant lr={LR_SPIKE:.1e}, {SEG_LEN_SPIKE} symbols/segment "
      f"x {len(SCHEDULE)} = {SEG_LEN_SPIKE*len(SCHEDULE)} symbols, {N_CTX} ctx neurons...")
n_spike = SEG_LEN_SPIKE * len(SCHEDULE)
x_sp, seg_sp = make_switch_stream(SEG_LEN_SPIKE, seed=SEED)
rs = RoverSource(s=SCHEDULE[0], seed=SEED)        # used only for its input builders
ctx_fn = rs.context_input(x_sp, WINDOW)
y_fn = rs.outcome_input(x_sp, WINDOW)
net, pq, pctx = build_pes_net(ctx_fn, y_fn, LR_SPIKE)
with nengo.Simulator(net, dt=DT, progress_bar=False) as sim:
    sim.run(n_spike * WINDOW)
win_q, win_ctx = per_window_q_and_ctx(sim.data[pq], sim.data[pctx], n_spike)
sp_steps, sp_E = spiking_energy_trace(win_q, win_ctx, seg_sp, n_spike, sub=40)

# per-segment converged spiking energy (tail of each segment) vs the moving floor
sp_seg_E, sp_seg_exc = [], []
for s_i in range(len(SCHEDULE)):
    lo = s_i * SEG_LEN_SPIKE + int(TAIL_FRAC * SEG_LEN_SPIKE)
    hi = (s_i + 1) * SEG_LEN_SPIKE
    Q = learned_matrix_window(win_q, win_ctx, lo, hi)
    E_seg = energy_trajectory([Q], PI, PS[s_i])[0]
    sp_seg_E.append(float(E_seg))
    sp_seg_exc.append(float(E_seg - FLOORS[s_i]))
sp_seg_E = np.array(sp_seg_E)
sp_seg_exc = np.array(sp_seg_exc)
print("[spiking] per-segment converged energy vs the moving floor:")
for s_i in range(len(SCHEDULE)):
    print(f"  seg {s_i} s={SCHEDULE[s_i]:.2f}: E={sp_seg_E[s_i]:.4f}  floor={FLOORS[s_i]:.4f}"
          f"  excess={sp_seg_exc[s_i]:.4f}")
spiking_tracks = bool(np.all(sp_seg_exc < 0.10))
print(f"  spiking re-tracks every floor (all per-segment excess < 0.10): "
      f"{'YES' if spiking_tracks else 'NO'}")

# ---------------------------------------------------------------------------
# (2) numpy twin: tracking time-constants + steady-state error after each switch
# ---------------------------------------------------------------------------
print(f"\n[numpy twin] constant lr={LR_NP}, {SEG_LEN_NP} symbols/segment (clean tail)...")
seg_np, E_np, exc_np = run_numpy_twin(lambda t: LR_NP, SEG_LEN_NP, seed=SEED)
taus, sss = [], []
fits = []
print("[time constants] after each switch  excess(t) ~ ss + A e^{-t/tau}:")
for k in range(1, len(SCHEDULE)):
    t_sw = k * SEG_LEN_NP
    tau, ss, b, centers, fit = fit_time_constant(exc_np, t_sw, SEG_LEN_NP)
    taus.append(tau)
    sss.append(ss)
    fits.append((k, t_sw, b, centers, fit))
    print(f"  switch {k} (s {SCHEDULE[k-1]:.2f}->{SCHEDULE[k]:.2f}) at t={t_sw}: "
          f"tau~{tau:.0f} symbols, steady-state excess={ss:.4f}")
taus = np.array(taus)
sss = np.array(sss)
taus_finite = bool(np.all(np.isfinite(taus)) and np.all(taus > 0) and np.all(taus < SEG_LEN_NP))
ss_small = bool(np.all(sss < 0.02))
# per-segment numpy tail steady-state (against the moving floor), incl. segment 0
np_seg_exc = np.array([tail_excess(seg_np, exc_np, s_i) for s_i in range(len(SCHEDULE))])
print(f"  all tracking constants finite & < one segment: {'YES' if taus_finite else 'NO'}")
print(f"  all steady-state tracking errors small (< 0.02): {'YES' if ss_small else 'NO'}")

# ---------------------------------------------------------------------------
# (3) steady-state tracking error scales with the constant lr (the O(eta) ball)
# ---------------------------------------------------------------------------
SWEEP_LRS = [5e-3, 8e-3, 1.2e-2, 1.6e-2, 2.4e-2]
print(f"\n[lr sweep] converged steady-state tracking error vs constant lr "
      f"(s=0.70, 6 seeds)...")
sweep = numpy_lr_sweep(SWEEP_LRS)
sweep_lrs = np.array(sorted(sweep.keys()))
sweep_err = np.array([sweep[lr] for lr in sweep_lrs])
for lr, er in zip(sweep_lrs, sweep_err):
    print(f"  lr={lr:.4f}  ->  steady-state tracking error = {er:.5f} bits/symbol")
sweep_oeta = bool(np.all(np.diff(sweep_err) >= -1e-4))   # non-decreasing in lr
# slope of log-log fit (O(eta) => slope ~ 1)
slope = float(np.polyfit(np.log(sweep_lrs), np.log(sweep_err), 1)[0])
print(f"  steady-state error increases with lr (O(eta) ball): {'YES' if sweep_oeta else 'NO'}")
print(f"  log-log slope = {slope:.2f}  (O(eta) prediction: ~1)")

# ---------------------------------------------------------------------------
# (4) decreasing / tiny lr FREEZES -- fails to re-track a LATE switch
# ---------------------------------------------------------------------------
print(f"\n[decreasing lr] Robbins-Monro lr0/(1+t/t0): re-tracks early, FREEZES late...")
LR0, T0 = 0.08, 800.0
dec_lr = lambda t: LR0 / (1.0 + t / T0)
seg_dec, E_dec, exc_dec = run_numpy_twin(dec_lr, SEG_LEN_NP, seed=SEED)
dec_seg_exc = np.array([tail_excess(seg_dec, exc_dec, s_i) for s_i in range(len(SCHEDULE))])
lr_at_switch = np.array([dec_lr(k * SEG_LEN_NP) for k in range(len(SCHEDULE))])
print("  lr at each switch (decaying):",
      "  ".join(f"{lr:.4f}" for lr in lr_at_switch))
print("  seg | floor | constant-lr tail-exc | decreasing-lr tail-exc")
for s_i in range(len(SCHEDULE)):
    print(f"   {s_i}  | {FLOORS[s_i]:.4f} |    {np_seg_exc[s_i]:.4f}        |   "
          f"{dec_seg_exc[s_i]:.4f}")
# the late switch (0.40 -> 0.85 sharpen): decreasing-lr freezes, constant-lr re-tracks
late = LATE_SWITCH_SEG
freeze_gap = float(dec_seg_exc[late] - np_seg_exc[late])
decreasing_freezes = bool(dec_seg_exc[late] > 5 * np_seg_exc[late] and dec_seg_exc[late] > 0.02)
print(f"  LATE switch (seg {late}, s {SCHEDULE[late-1]:.2f}->{SCHEDULE[late]:.2f}): "
      f"decreasing-lr excess {dec_seg_exc[late]:.4f} vs constant-lr {np_seg_exc[late]:.4f} "
      f"(gap {freeze_gap:.4f})")
print(f"  decreasing-lr FREEZES on the late switch (>5x worse & >0.02): "
      f"{'YES' if decreasing_freezes else 'NO'}")

# a tiny CONSTANT lr is even worse -- too small to descend at all within a segment
TINY_LR = 2e-4
seg_tiny, E_tiny, exc_tiny = run_numpy_twin(lambda t: TINY_LR, SEG_LEN_NP, seed=SEED)
tiny_seg_exc = np.array([tail_excess(seg_tiny, exc_tiny, s_i) for s_i in range(len(SCHEDULE))])
tiny_frozen = bool(np.all(tiny_seg_exc > 0.10))
print(f"  tiny constant lr={TINY_LR:.0e} never tracks (all segs excess > 0.10): "
      f"{'YES' if tiny_frozen else 'NO'}  "
      f"({', '.join(f'{e:.3f}' for e in tiny_seg_exc)})")

# ===========================================================================
# FIGURES
# ===========================================================================
sw_x = [k * SEG_LEN_SPIKE for k in range(1, len(SCHEDULE))]   # spiking switch indices

# Fig 1: SPIKING energy vs time across the switches, per-segment floors marked
fig, ax = plot.new_fig(w=7.2, h=4.4)
ax.plot(sp_steps, sp_E, "-", color=plot.C_MEASURED, lw=1.3, alpha=0.85,
        label="spiking decoded energy (trailing window)")
ax.axhline(H_MARGINAL, color="gray", ls=":", lw=1.2,
           label=f"marginal H(pi)={H_MARGINAL:.4f} (INVARIANT)")
seg_colors = ["#2ca02c", "#9467bd", "#e377c2", "#2ca02c"]
for s_i in range(len(SCHEDULE)):
    x0 = s_i * SEG_LEN_SPIKE
    x1 = (s_i + 1) * SEG_LEN_SPIKE
    ax.hlines(FLOORS[s_i], x0, x1, color=seg_colors[s_i], ls="--", lw=2.0,
              label=(f"floor H_rate(s)" if s_i == 0 else None))
    ax.axvspan(x0, x1, color=seg_colors[s_i], alpha=0.05)
    ax.annotate(f"s={SCHEDULE[s_i]:.2f}\nfloor {FLOORS[s_i]:.3f}",
                (x0 + 0.5 * SEG_LEN_SPIKE, FLOORS[s_i]), fontsize=7.5, ha="center",
                va="bottom", color=seg_colors[s_i])
for xs in sw_x:
    ax.axvline(xs, color="black", ls="-", lw=0.8, alpha=0.4)
ax.set_ylim(0.3, 2.15)
ax.set_xlabel("symbol index (s switches at the vertical lines)")
ax.set_ylabel("energy (bits/symbol)")
ax.set_title("e13: spiking energy re-tracks the MOVING floor after each s-switch")
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
plot.save(fig, RESULTS / "e13_spiking_tracking.pdf")

# Fig 2: numpy twin excess vs time with the per-switch exponential fits + tau labels
fig, ax = plot.new_fig(w=7.2, h=4.2)
# bin the full excess trace for a clean line
nb = (len(exc_np) // 40) * 40
exc_binned = exc_np[:nb].reshape(-1, 40).mean(1)
tb = np.arange(len(exc_binned)) * 40
ax.plot(tb, exc_binned, "-", color=plot.C_MEASURED, lw=1.0, alpha=0.7,
        label="excess KL (binned)")
for (k, t_sw, b, centers, fit) in fits:
    if fit is not None:
        ax.plot(t_sw + centers, fit, "-", color=plot.C_THEORY, lw=2.0,
                label=("exp fit  ss + A e^{-t/tau}" if k == 1 else None))
        ax.annotate(f"tau~{taus[k-1]:.0f}\nss={sss[k-1]:.4f}",
                    (t_sw + 0.18 * SEG_LEN_NP, max(b)), fontsize=7.5,
                    color=plot.C_THEORY)
for k in range(1, len(SCHEDULE)):
    ax.axvline(k * SEG_LEN_NP, color="black", ls="-", lw=0.8, alpha=0.4)
for s_i in range(len(SCHEDULE)):
    x0, x1 = s_i * SEG_LEN_NP, (s_i + 1) * SEG_LEN_NP
    ax.axvspan(x0, x1, color=seg_colors[s_i], alpha=0.05)
    ax.annotate(f"s={SCHEDULE[s_i]:.2f}", (x0 + 0.5 * SEG_LEN_NP, exc_binned.max() * 0.92),
                fontsize=8, ha="center", color=seg_colors[s_i])
ax.set_xlabel("symbol index (numpy twin; s switches at the vertical lines)")
ax.set_ylabel("excess KL above the current floor (bits/symbol)")
ax.set_title("e13: each switch -> jump then exponential re-descent (tracking time-constant)")
ax.legend(frameon=False, fontsize=8)
plot.save(fig, RESULTS / "e13_time_constants.pdf")

# Fig 3 (L): steady-state error vs lr (O(eta)); (R): constant vs decreasing freeze bars
fig, (axL, axR) = plot.plt.subplots(1, 2, figsize=(11.0, 4.0))
axL.loglog(sweep_lrs, sweep_err, "o-", color=plot.C_MEASURED, lw=2, ms=8,
           label="converged steady-state tracking error")
ref_x = np.array([sweep_lrs.min() * 0.7, sweep_lrs.max() * 1.4])
ref_y = sweep_err[0] * (ref_x / sweep_lrs[0])
axL.loglog(ref_x, ref_y, ":", color="gray", lw=1.3, label=r"slope-1 $O(\eta)$ guide")
for lr, er in zip(sweep_lrs, sweep_err):
    axL.annotate(f"{er:.4f}", (lr, er), fontsize=7.5, xytext=(0, 7),
                 textcoords="offset points", ha="center")
axL.set_xlabel(r"constant learning rate $\eta$")
axL.set_ylabel(r"steady-state tracking error $V_\infty$ (bits/symbol)")
axL.set_title(f"(3) ball $\\sim O(\\eta)$  (log-log slope {slope:.2f})")
axL.legend(frameon=False, fontsize=8)
# right: per-segment tail excess, constant vs decreasing, with the floor-relative jump
xb = np.arange(len(SCHEDULE))
axR.bar(xb - 0.2, np_seg_exc, 0.4, color=plot.C_FLOOR, label="constant lr (adapts)")
axR.bar(xb + 0.2, dec_seg_exc, 0.4, color=plot.C_THEORY, alpha=0.85,
        label="decreasing lr (freezes)")
axR.axvline(LATE_SWITCH_SEG, color="black", ls=":", lw=1.0, alpha=0.0)
axR.annotate("FROZEN\n(late switch)", (LATE_SWITCH_SEG + 0.2, dec_seg_exc[LATE_SWITCH_SEG]),
             fontsize=8, ha="center", va="bottom", color=plot.C_THEORY,
             xytext=(0, 6), textcoords="offset points")
axR.set_xticks(xb)
axR.set_xticklabels([f"seg {i}\ns={SCHEDULE[i]:.2f}" for i in range(len(SCHEDULE))], fontsize=8)
axR.set_ylabel("steady-state excess (bits/symbol)")
axR.set_title("(4) decreasing lr freezes on the late switch")
axR.legend(frameon=False, fontsize=8)
plot.save(fig, RESULTS / "e13_ball_and_freeze.pdf")

# ===========================================================================
# SAVE + ACCEPTANCE
# ===========================================================================
np.savez(RESULTS / "e13_results.npz",
         schedule=np.array(SCHEDULE), floors=np.array(FLOORS), H_MARGINAL=H_MARGINAL,
         sp_steps=sp_steps, sp_E=sp_E, sp_seg_E=sp_seg_E, sp_seg_exc=sp_seg_exc,
         spiking_tracks=spiking_tracks, seg_len_spike=SEG_LEN_SPIKE, lr_spike=LR_SPIKE,
         exc_np=exc_np, E_np=E_np, seg_np=seg_np, taus=taus, sss=sss,
         np_seg_exc=np_seg_exc, taus_finite=taus_finite, ss_small=ss_small,
         sweep_lrs=sweep_lrs, sweep_err=sweep_err, sweep_oeta=sweep_oeta, slope=slope,
         dec_seg_exc=dec_seg_exc, lr_at_switch=lr_at_switch, freeze_gap=freeze_gap,
         decreasing_freezes=decreasing_freezes, late_switch_seg=LATE_SWITCH_SEG,
         tiny_seg_exc=tiny_seg_exc, tiny_frozen=tiny_frozen, tiny_lr=TINY_LR,
         lr0=LR0, t0=T0)

checks = {
    "spiking energy re-tracks the moving floor (every per-segment excess < 0.10)":
        spiking_tracks,
    "spiking re-descends below the INVARIANT marginal in every segment":
        bool(np.all(sp_seg_E < H_MARGINAL)),
    "each switch has a finite tracking time-constant (0 < tau < one segment)":
        taus_finite,
    "constant-lr steady-state tracking error is small in every segment (< 0.02)":
        ss_small,
    "steady-state tracking error scales with lr (O(eta), monotone non-decreasing)":
        sweep_oeta,
    "the O(eta) ball has log-log slope ~1 (0.6 < slope < 1.4)":
        0.6 < slope < 1.4,
    "decreasing (Robbins-Monro) lr FREEZES on a late switch (>5x the constant-lr excess)":
        decreasing_freezes,
    "a tiny constant lr never tracks any segment (the over-frozen extreme)":
        tiny_frozen,
}
print("\n" + "=" * 70)
allok = True
for name, ok in checks.items():
    allok &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
print(f"\ne13: {sum(checks.values())}/{len(checks)} acceptance checks passed.")
sys.exit(0 if allok else 1)
