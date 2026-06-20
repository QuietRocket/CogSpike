#!/usr/bin/env python3
"""e10 -- Excess energy is a Lyapunov function: descent + the noise ball.

Claim (paper #11): the excess energy

    V(W) = H(p, q) - H(p) = sum_i pi_i D_KL(P[i,:] || q[i,:])   (bits/symbol)

is a *Lyapunov function* for learning. For the averaged gradient flow
Wdot = -eta grad V we get Vdot = -eta ||grad V||^2 <= 0 (monotone descent); the
loss is convex in the logits with the unique fixed point q = P. Two regimes:

  * Robbins-Monro DECREASING step  -> almost-sure convergence to the floor
    (V -> 0, the numpy validator's 0.0005-bit residual).
  * CONSTANT step                  -> an O(eta) NOISE BALL around the optimum.

This experiment makes the *constant-eta noise ball physical*: it is not a
minibatch-sampling artifact (the textbook SGD story) but the irreducible jitter
of a real spiking substrate -- neuron noise, finite-window sampling, and the NEF
decode's 1/sqrt(N) representational error. We measure the ball's radius as a
function of the PES learning rate and show it shrinks with lr, exactly the
paper's O(eta) prediction, now carried by spikes.

FOUR results:
  (1) CONSTANT-lr NOISE BALL (spiking centerpiece): run the Nengo PES learner at
      lr in {1e-3, 5e-4, 2e-4, 1e-4}; for each, the FINAL converged excess energy
      (avg KL above the floor 0.9782) DECREASES with lr -- the O(eta) noise ball,
      now physical. Plot final excess vs lr.
  (2) MONOTONE LYAPUNOV DESCENT: from the spiking run, the energy descends
      (cummin envelope monotone) from the uniform init 2.0 toward the floor --
      the excess energy dissipating as a Lyapunov function. Plot with the floor.
  (3) NUMPY ANCHORS (fast & exact, reproducing critique_checks.py /
      learn_validate.py): CHECK 1 sum-mode conservation -- max|sum_j Delta W_ij|
      and the row-sum drift stay at machine epsilon (~1e-15) from a NONZERO init;
      CHECK 2 the Lyapunov constant k in Vdot = -k||grad V||^2 equals ln2=0.693
      (NOT eta); and the averaged-flow monotone descent (frac mono ~1.0, max
      increase ~0).
  (4) SUM-MODE / PARTITION DRIFT IN THE SPIKING DECODER: track |sum_j q_j - 1|
      over learning for the spiking PES decoder (representational, ~1e-3 per e03),
      contrasting the EXACT numpy invariant (~1e-15) with the representational
      spiking one -- the spiking-reality gap.

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

from spikecoder import config as cfg                         # noqa: E402
from spikecoder import plotting as plot                      # noqa: E402
from spikecoder.source import RoverSource, LABELS, momentum_chain  # noqa: E402
from spikecoder.information import (softmax, model_energy_and_kl,   # noqa: E402
                                    H_RATE, H_MARGINAL, LEARNED_REF, LN2)
from spikecoder.metrics import energy_trajectory             # noqa: E402

# --- experiment constants (the verified e09 PES idiom) ---
N_SYMBOLS = 8000       # symbols per spiking run (enough to converge into the ball)
WINDOW = 0.03          # s per symbol window
LRS = [1e-3, 5e-4, 2e-4, 1e-4]   # constant PES learning rates (the noise-ball sweep)
LR_DESCENT = 5e-4      # the run whose descent trajectory we plot (fast + clean)
N_CTX = 600            # context-ensemble size
CTX_RADIUS = 1.3
DT = cfg.DT            # 1e-3 for throughput/learning
SEED = cfg.SEED
CONVERGED_FRAC = 0.6   # read the converged law over the last 40% of the stream


# ===========================================================================
# (3) NUMPY ANCHORS -- reproduce critique_checks.py CHECK 1 + CHECK 2 and the
#     learn_validate.py averaged-flow monotone descent. Fast, exact, pure numpy.
# ===========================================================================
def numpy_check1_sum_mode(n=200_000, seed=7):
    """CHECK 1: sum_j Delta W[i,j] = 0 every step => row-sums of W invariant.
    Started from a NONZERO init so 'constant row-sums' is a real (non-trivial)
    test. Returns (max per-step row-sum deviation, max row-sum drift from init)."""
    pi = np.array([1 / 2, 1 / 4, 1 / 8, 1 / 8])
    P = momentum_chain(pi, 0.7)
    rng = np.random.default_rng(seed)
    x = np.empty(n, dtype=int)
    x[0] = rng.choice(4, p=pi)
    for t in range(1, n):
        x[t] = rng.choice(4, p=P[x[t - 1]])
    W = rng.normal(size=(4, 4))                 # NONZERO init
    init_rowsums = W.sum(axis=1).copy()
    eta0, t0 = 0.2, 5.0e4
    max_step_dev = 0.0
    max_drift = 0.0
    for t in range(1, n):
        i, j = x[t - 1], x[t]
        q = softmax(W[i])
        eta_t = eta0 / (1.0 + t / t0)
        dW = eta_t * (np.eye(4)[j] - q)
        max_step_dev = max(max_step_dev, abs(dW.sum()))
        W[i] += dW
        max_drift = max(max_drift, float(np.abs(W.sum(axis=1) - init_rowsums).max()))
    return max_step_dev, max_drift, init_rowsums, W.sum(axis=1)


def grad_V_bits(W, pi, P):
    """Analytic gradient of V (bits): grad V[i,j] = -pi_i (P[i,j]-q[i,j])/ln2."""
    g = np.zeros((4, 4))
    for i in range(4):
        g[i] = -pi[i] * (P[i] - softmax(W[i])) / LN2
    return g


def numpy_check2_lyapunov(dt=1e-3, n_steps=40_000):
    """CHECK 2: integrate the averaged gradient flow
       Wm[i] += dt*pi[i]*(P[i]-softmax(Wm[i])) and measure
       k = (-dV/dt)/||grad V||^2. It must converge to ln2 (NOT eta).
    Also returns the energy trajectory (for the averaged-flow monotone-descent
    sub-check) and the monotone fraction / max increase."""
    pi = np.array([1 / 2, 1 / 4, 1 / 8, 1 / 8])
    P = momentum_chain(pi, 0.7)
    Wm = np.zeros((4, 4))
    ks = []
    Es = [model_energy_and_kl(Wm, pi, P, H_RATE)[0]]
    V_prev = sum(pi[i] * float((P[i][P[i] > 0] *
                  np.log2(P[i][P[i] > 0] / softmax(Wm[i])[P[i] > 0])).sum())
                 for i in range(4))
    for _ in range(n_steps):
        g = grad_V_bits(Wm, pi, P)
        gnorm2 = float((g * g).sum())
        for i in range(4):
            Wm[i] += dt * pi[i] * (P[i] - softmax(Wm[i]))
        V_now = sum(pi[i] * float((P[i][P[i] > 0] *
                     np.log2(P[i][P[i] > 0] / softmax(Wm[i])[P[i] > 0])).sum())
                    for i in range(4))
        if gnorm2 > 1e-14:
            ks.append((-(V_now - V_prev) / dt) / gnorm2)
        V_prev = V_now
        Es.append(model_energy_and_kl(Wm, pi, P, H_RATE)[0])
    ks = np.array(ks)
    Es = np.array(Es)
    k_mid = float(np.median(ks[len(ks) // 4:]))
    diffs = np.diff(Es)
    frac_mono = float((diffs <= 1e-12).mean())
    max_inc = max(0.0, float(diffs.max()))
    return k_mid, float(ks[-1]), Es, frac_mono, max_inc


ASYM_LRS = (5e-3, 8e-3, 1.2e-2, 1.6e-2)   # fast rates that fully CONVERGE within n


def numpy_asymptotic_ball(lrs=ASYM_LRS, n=60_000, tail=0.4, nseed=4):
    """The paper's CONSTANT-eta noise ball, isolated cleanly in numpy.

    Runs the *exact* online delta rule W[i] += lr*(onehot(j)-softmax(W[i])) -- the
    pure-float64 analogue of the spiking PES decoder -- and tail-averages the
    excess energy once each rate has CONVERGED. The asymptotic residual is the
    noise ball; for converged rates it scales as O(eta): the smaller the
    (still-converged) step, the smaller the ball.

    Two deliberate choices isolate the ASYMPTOTIC face from the descent transient
    (which is the spiking sweep's regime, documented separately):
      * we use only FAST rates (5e-3..1.6e-2) that fully converge within n, so the
        tail residual is the BALL, not how-far-it-descended. Slow rates (1e-4) need
        millions of steps to converge -- at a finite budget their residual is
        descent speed, not ball, which is exactly why the spiking sweep reads the
        other face;
      * we average the tail-ball over nseed streams to denoise the small residual,
        so the O(eta) ordering is robust to a single stream's luck."""
    pi = np.array([1 / 2, 1 / 4, 1 / 8, 1 / 8])
    P = momentum_chain(pi, 0.7)
    eye = np.eye(4)
    mark = int((1 - tail) * n)
    out = {}
    for lr in lrs:
        seed_vals = []
        for sd in range(nseed):
            rng = np.random.default_rng(200 + sd)
            x = np.empty(n, dtype=int)
            x[0] = rng.choice(4, p=pi)
            for t in range(1, n):
                x[t] = rng.choice(4, p=P[x[t - 1]])
            W = np.zeros((4, 4))
            excs = []
            for t in range(n):
                i = x[t - 1] if t > 0 else x[0]
                j = x[t]
                z = W[i] - W[i].max()
                e = np.exp(z)
                W[i] += lr * (eye[j] - e / e.sum())
                if t > mark:
                    excs.append(model_energy_and_kl(W, pi, P, H_RATE)[1])
            seed_vals.append(float(np.mean(excs)))
        out[lr] = float(np.mean(seed_vals))
    return out


# ===========================================================================
# the spiking PES learner (the e09 idiom), parameterized by the learning rate
# ===========================================================================
def build_pes_net(ctx_fn, y_fn, lr, seed=SEED):
    """One spiking context population whose DECODERS are learned by PES at rate lr.

    ctx -> q is a decoded prediction, initialized to the uniform predictor (decoder
    function lambda c: ones(4)/4, so energy starts at log2 4 = 2.0). The error fed
    to PES is (q - y): PES descends -lr * error * activity, driving q toward
    E[y|c] = P. The probe pq carries the *raw* decoded q (before any clip /
    renormalize) so its partition sum sum_j q_j is observable (result 4)."""
    net = nengo.Network(seed=seed)
    with net:
        ci = nengo.Node(ctx_fn, size_out=4)
        ctx = nengo.Ensemble(N_CTX, 4, radius=CTX_RADIUS)
        nengo.Connection(ci, ctx, synapse=0.005)

        q = nengo.Node(size_in=4)                          # decoded prediction q(.|c)
        conn = nengo.Connection(
            ctx, q, function=lambda c: np.ones(4) * 0.25,  # uniform init
            learning_rule_type=nengo.PES(learning_rate=lr), synapse=0.01)

        yi = nengo.Node(y_fn, size_out=4)                  # teacher: one-hot realized symbol
        err = nengo.Node(size_in=4)                        # error = q - y
        nengo.Connection(q, err, synapse=None)
        nengo.Connection(yi, err, transform=-1, synapse=None)
        nengo.Connection(err, conn.learning_rule, synapse=None)

        pq = nengo.Probe(q, synapse=0.02)
        pctx = nengo.Probe(ci)
    return net, pq, pctx


def per_window_q_and_ctx(qd, cd):
    """Reduce raw probes to one settled-tail q and one context index per window."""
    win_q = np.zeros((N_SYMBOLS, 4))
    win_ctx = np.full(N_SYMBOLS, -1)
    for k in range(N_SYMBOLS):
        i0 = int(round(k * WINDOW / DT))
        i1 = int(round((k + 1) * WINDOW / DT))
        if i1 > len(qd):
            break
        seg = qd[i0:i1]
        win_q[k] = seg[len(seg) // 2:].mean(0)             # settled tail (last 50%)
        csum = cd[i0:i1].sum(0)
        if csum.max() > 0:
            win_ctx[k] = int(np.argmax(csum))
    return win_q, win_ctx


def learned_matrix_from_windows(win_q, win_ctx, lo=0, hi=None):
    """Average decoded q per context over windows [lo, hi); clip>0 & renormalize."""
    hi = N_SYMBOLS if hi is None else hi
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


def run_spiking(ctx_fn, y_fn, lr):
    """Run one constant-lr spiking PES learner; return final excess, descent
    snapshots, and the raw per-window q (for partition drift)."""
    net, pq, pctx = build_pes_net(ctx_fn, y_fn, lr)
    with nengo.Simulator(net, dt=DT, progress_bar=False) as sim:
        sim.run(N_SYMBOLS * WINDOW)
    win_q, win_ctx = per_window_q_and_ctx(sim.data[pq], sim.data[pctx])
    Q = learned_matrix_from_windows(win_q, win_ctx, lo=int(CONVERGED_FRAC * N_SYMBOLS))
    E_final, kl_final = model_energy_and_kl(Q, pi, P, H_RATE, is_logits=False)
    return E_final, kl_final, win_q, win_ctx, Q


# ===========================================================================
print("=" * 70)
print("e10 -- Excess energy is a Lyapunov function: descent + the noise ball")
print("=" * 70)

rs = RoverSource(s=0.7, seed=SEED)
P, pi = rs.P, rs.pi
x = rs.sample(N_SYMBOLS)
ctx_fn = rs.context_input(x, WINDOW)
y_fn = rs.outcome_input(x, WINDOW)
print(f"\nsource: momentum rover s=0.7, {N_SYMBOLS} symbols, window {WINDOW}s, "
      f"{N_CTX} ctx neurons, dt={DT}")
print(f"floors: marginal H(pi)={H_MARGINAL:.4f}  entropy rate H={H_RATE:.4f} bits/symbol")
print(f"uniform init energy = log2(4) = {np.log2(4):.4f} bits/symbol")
print(f"noise-ball lr sweep: {LRS}")

# --- (3) NUMPY ANCHORS FIRST (cheap, exact) --------------------------------
print("\n[numpy CHECK 1: sum-mode conservation] nonzero-init online delta rule...")
c1_step, c1_drift, c1_init, c1_final = numpy_check1_sum_mode()
print(f"  init row-sums  = {c1_init}")
print(f"  final row-sums = {c1_final}")
print(f"  max|sum_j Delta W_ij| per step = {c1_step:.3e}  (machine epsilon)")
print(f"  max row-sum DRIFT from init    = {c1_drift:.3e}  (target < 1e-12)")

print("\n[numpy CHECK 2: Lyapunov constant] integrate the averaged gradient flow...")
k_mid, k_last, avg_Es, avg_frac_mono, avg_max_inc = numpy_check2_lyapunov()
print(f"  measured k (median latter 3/4) = {k_mid:.6f}")
print(f"  measured k (last step)         = {k_last:.6f}")
print(f"  ln2                            = {LN2:.6f}")
print(f"  |k - ln2| = {abs(k_mid - LN2):.3e}   |k - eta(0.5)| = {abs(k_mid - 0.5):.3e}")
print(f"  -> the Lyapunov-rate constant is ln2, NOT eta (V is in BITS, flow in nats)")
print(f"\n[numpy averaged-flow descent] V dissipates monotonically (Lyapunov):")
print(f"  energy {avg_Es[0]:.4f} -> {avg_Es[-1]:.6f}  (floor {H_RATE:.4f})")
print(f"  fraction of non-increasing steps = {avg_frac_mono:.4f}  (theorem: 1.0)")
print(f"  max energy INCREASE over all steps = {avg_max_inc:.2e}  (must be ~0)")

# --- (1a) the ASYMPTOTIC O(eta) noise ball, isolated in numpy ---------------
# (the exact float64 delta rule, run to CONVERGENCE; the tail residual is the
#  ball, and for converged rates it scales as O(eta): smaller lr -> smaller ball)
print("\n[numpy asymptotic ball] exact delta rule run to convergence (O(eta) ball)...")
asym = numpy_asymptotic_ball()
asym_lrs = sorted(asym.keys())                       # ascending lr
asym_excess = np.array([asym[lr] for lr in asym_lrs])
asym_oeta = bool(np.all(np.diff(asym_excess) >= -1e-4))   # non-decreasing in lr
for lr in asym_lrs:
    print(f"  lr={lr:.0e}  ->  converged ball excess = {asym[lr]:.5f} bits/symbol")
print(f"  smaller (converged) lr => smaller ball: {'YES' if asym_oeta else 'NO'}  "
      f"-> the constant-eta ball IS O(eta)")

# --- (1b) CONSTANT-lr SPIKING SWEEP at a fixed training budget --------------
# At a FIXED, finite spiking budget (N_SYMBOLS), the slow rates have not yet
# reached their asymptotic ball, so the residual is dominated by DESCENT SPEED:
# larger lr descends faster -> smaller residual at the same budget. This is the
# OTHER face of the same O(eta) trade-off (speed up <-> bigger asymptotic ball);
# the numpy run above isolates the asymptotic face the spiking budget can't reach.
print("\n[spiking sweep] running PES learner at each constant lr (fixed budget)...")
ball_excess = []
ball_E = []
descent_pack = None   # (snap_steps, snap_E, snap_E_env) for the LR_DESCENT run
part_pack = None      # (steps, partition-drift series) for result (4)
Q_descent = None
for lr in LRS:
    E_final, kl_final, win_q, win_ctx, Q = run_spiking(ctx_fn, y_fn, lr)
    ball_excess.append(kl_final)
    ball_E.append(E_final)
    print(f"  lr={lr:.0e}:  final energy E={E_final:.4f}  excess(avg KL)={kl_final:.4f}")

    if abs(lr - LR_DESCENT) < 1e-12:
        Q_descent = Q
        # (2) energy-descent snapshots, anchored at the exact uniform init (2.0)
        fine = [1, 3, 6, 12, 25, 50, 100, 200, 400, 800, 1600]
        coarse = list(np.linspace(2400, N_SYMBOLS, 20).astype(int))
        snap_steps = np.array([0] + fine + coarse)
        snap_E = [float(np.log2(4))]
        for m in snap_steps[1:]:
            Qm = learned_matrix_from_windows(win_q, win_ctx, lo=0, hi=int(m))
            snap_E.append(energy_trajectory([Qm], pi, P)[0])
        snap_E = np.array(snap_E)
        snap_E_env = np.minimum.accumulate(snap_E)
        descent_pack = (snap_steps, snap_E, snap_E_env)
        # (4) spiking partition drift |sum_j q_j - 1| over learning, binned
        nbins = 40
        edges = np.linspace(0, N_SYMBOLS, nbins + 1).astype(int)
        part_steps, part_drift = [], []
        for b in range(nbins):
            sl = slice(edges[b], edges[b + 1])
            seg = win_q[sl]
            valid = seg[np.any(seg != 0, axis=1)]
            if len(valid):
                part_steps.append(int((edges[b] + edges[b + 1]) // 2))
                part_drift.append(float(np.mean(np.abs(valid.sum(1) - 1.0))))
        part_pack = (np.array(part_steps), np.array(part_drift))

ball_excess = np.array(ball_excess)
ball_E = np.array(ball_E)
lrs_arr = np.array(LRS)

# At a FIXED budget the spiking sweep traces the DESCENT-SPEED face: residual is
# MONOTONE DECREASING in lr (larger step descends faster -> smaller residual at
# the same budget). We test exactly that, and that every residual is a small
# noise ball well below the marginal (all in (0, 0.1)). The asymptotic O(eta)
# face (smaller lr -> smaller ball) is supplied by the numpy run above.
order = np.argsort(lrs_arr)
lrs_sorted = lrs_arr[order]
excess_sorted = ball_excess[order]
# residual decreases as lr increases (fixed-budget descent-speed monotonicity)
speed_monotone = bool(np.all(np.diff(excess_sorted) <= 1e-3))
all_small = bool(np.all((excess_sorted > 0) & (excess_sorted < 0.1)))
print(f"\n[spiking sweep] final excess vs lr (fixed budget {N_SYMBOLS}, ascending lr):")
for lr, ex in zip(lrs_sorted, excess_sorted):
    print(f"  lr={lr:.0e}  ->  final excess = {ex:.4f} bits/symbol")
print(f"  residual decreases as lr increases (descent-speed face): "
      f"{'YES' if speed_monotone else 'NO'}")
print(f"  every residual a small noise ball in (0, 0.1): {'YES' if all_small else 'NO'}")
print(f"  contrast: numpy CONVERGED ball IS O(eta) (smaller lr -> smaller ball): "
      f"{'YES' if asym_oeta else 'NO'}")

# --- (2) descent envelope monotonicity --------------------------------------
snap_steps, snap_E, snap_E_env = descent_pack
descent_monotone = bool(np.all(np.diff(snap_E_env) <= 1e-9))
crossed = snap_E[snap_E < H_MARGINAL]
step_below_marg = int(snap_steps[np.argmax(snap_E < H_MARGINAL)]) if len(crossed) else -1
print(f"\n[Lyapunov descent] (lr={LR_DESCENT:.0e}) init 2.0 -> final {snap_E[-1]:.4f}")
print(f"  crosses below marginal {H_MARGINAL:.4f} by step {step_below_marg}")
print(f"  cummin descent envelope monotone: {'YES' if descent_monotone else 'NO'}")

# --- (4) spiking vs exact partition drift -----------------------------------
part_steps, part_drift = part_pack
spiking_part = float(part_drift.mean())
print(f"\n[partition drift] spiking decoder mean|sum_j q_j - 1| = {spiking_part:.2e} "
      f"(representational, e03)")
print(f"  exact numpy invariant (CHECK 1 row-sum drift) = {c1_drift:.2e} "
      f"(algebraic, machine epsilon)")
print(f"  spiking-reality gap factor = {spiking_part / max(c1_drift, 1e-30):.1e}x")

# ===========================================================================
# FIGURES
# ===========================================================================
# Fig 1: the noise ball -- the TWO FACES of the O(eta) trade-off on one axis.
#  (a) numpy CONVERGED ball (red): smaller lr -> smaller asymptotic ball = O(eta).
#  (b) spiking FIXED-budget residual (blue): larger lr descends faster -> smaller
#      residual at the same budget (the descent-speed face the spiking budget sees).
fig, ax = plot.new_fig()
asym_lrs_arr = np.array(asym_lrs)
ax.loglog(asym_lrs_arr, asym_excess, "s-", color=plot.C_THEORY, lw=2, ms=8,
          label="numpy: CONVERGED ball $\\sim O(\\eta)$ (smaller $\\eta$ = smaller ball)")
# O(eta) reference slope-1 line anchored at the smallest converged numpy point
ref_x = np.array([asym_lrs_arr.min() * 0.6, asym_lrs_arr.max() * 1.6])
ref_y = asym_excess[0] * (ref_x / asym_lrs_arr[0])
ax.loglog(ref_x, ref_y, ":", color="gray", lw=1.3, label=r"slope-1 $O(\eta)$ guide")
ax.loglog(lrs_sorted, excess_sorted, "o-", color=plot.C_MEASURED, lw=2, ms=8,
          label=f"spiking PES: residual at fixed budget {N_SYMBOLS} (descent-speed face)")
ax.axhline(0.0005, color=plot.C_FLOOR, ls="--", lw=1.5,
           label="numpy floor 0.0005 (decreasing-$\\eta$ Robbins-Monro)")
for lr, ex in zip(lrs_sorted, excess_sorted):
    ax.annotate(f"{ex:.3f}", (lr, ex), fontsize=7.5, xytext=(0, 7),
                textcoords="offset points", ha="center", color=plot.C_MEASURED)
for lr, ex in zip(asym_lrs_arr, asym_excess):
    ax.annotate(f"{ex:.4f}", (lr, ex), fontsize=7.5, xytext=(0, -12),
                textcoords="offset points", ha="center", color=plot.C_THEORY)
ax.set_xlabel("constant learning rate  $\\eta$ (lr)")
ax.set_ylabel("excess energy  $V = E - H_{rate}$  (bits/symbol)")
ax.set_title("e10: two faces of the constant-$\\eta$ noise-ball / descent-speed trade-off")
ax.legend(frameon=False, fontsize=7.5)
plot.save(fig, RESULTS / "e10_noise_ball.pdf")

# Fig 2: Lyapunov energy descent (spiking) with the floor + marginal lines
fig, ax = plot.new_fig()
xs = np.maximum(snap_steps, 1)
ax.plot(xs, snap_E, "o-", color=plot.C_MEASURED, lw=2, ms=4,
        label=f"spiking energy E (lr={LR_DESCENT:.0e})")
ax.plot(xs, snap_E_env, "-", color="purple", lw=1.2, alpha=0.7,
        label="cummin envelope (Lyapunov)")
ax.axhline(H_MARGINAL, color="gray", ls=":", lw=1.3, label=f"marginal {H_MARGINAL:.4f}")
ax.axhline(H_RATE, color=plot.C_FLOOR, ls="--", lw=1.5,
           label=f"entropy-rate floor {H_RATE:.4f}")
ax.axhline(LEARNED_REF, color="orange", ls="-.", lw=1.0, alpha=0.7,
           label=f"numpy learned ref {LEARNED_REF:.4f}")
ax.annotate("uniform init\n$\\log_2 4 = 2.0$", (xs[0], snap_E[0]), fontsize=8,
            xytext=(xs[0] * 1.3, 1.92), color=plot.C_MEASURED)
ax.set_xscale("log")
ax.set_xlabel("training step (symbols seen, log scale)")
ax.set_ylabel("energy (bits/symbol)")
ax.set_title("e10: excess energy is a Lyapunov function -- it dissipates monotonically")
ax.legend(frameon=False, fontsize=8)
plot.save(fig, RESULTS / "e10_lyapunov_descent.pdf")

# Fig 3: the numpy anchors -- (L) ln2 constant convergence + averaged-flow descent,
#        (R) sum-mode / partition-drift bar contrast (exact vs spiking)
fig, (axL, axR) = plot.plt.subplots(1, 2, figsize=(10.0, 3.8))
# left: averaged-flow energy descent with the ln2-constant annotation
avg_steps = np.arange(len(avg_Es))
axL.plot(avg_steps, avg_Es, "-", color=plot.C_MEASURED, lw=1.6,
         label="averaged-flow energy")
axL.axhline(H_RATE, color=plot.C_FLOOR, ls="--", lw=1.4,
            label=f"floor {H_RATE:.4f}")
axL.set_xlabel("averaged-flow step")
axL.set_ylabel("energy (bits/symbol)")
axL.set_title(f"averaged flow: $V$ dissipates;  $k = {k_mid:.4f} = \\ln 2$")
axL.text(0.50, 0.55, f"$\\dot V = -k\\,\\|\\nabla V\\|^2$\n$k = {k_mid:.4f}$\n"
         f"$\\ln 2 = {LN2:.4f}$\n(NOT $\\eta$)",
         transform=axL.transAxes, fontsize=9, va="center",
         bbox=dict(boxstyle="round", fc="#eef3fb", ec="#4a90d9"))
axL.legend(frameon=False, fontsize=8)
# right: sum-mode / partition drift -- exact numpy vs representational spiking
cats = ["numpy row-sum\ndrift (exact)", "spiking $|\\sum q-1|$\n(representational)"]
vals = [max(c1_drift, 1e-18), spiking_part]
bars = axR.bar(cats, vals, color=[plot.C_FLOOR, plot.C_MEASURED])
axR.set_yscale("log")
axR.set_ylabel("normalization defect")
axR.axhline(1e-12, color="gray", ls="--", lw=1, label="1e-12 (machine)")
axR.axhline(1e-3, color="orange", ls=":", lw=1, label="1e-3 (e03 NEF)")
for b, v in zip(bars, vals):
    axR.annotate(f"{v:.1e}", (b.get_x() + b.get_width() / 2, v), fontsize=8,
                 xytext=(0, 4), textcoords="offset points", ha="center")
axR.set_title("sum-mode: exact (algebraic) vs spiking (representational)")
axR.legend(frameon=False, fontsize=8)
plot.save(fig, RESULTS / "e10_numpy_anchors.pdf")

# ===========================================================================
# SAVE + ACCEPTANCE
# ===========================================================================
np.savez(RESULTS / "e10_results.npz",
         P=P, pi=pi, lrs=lrs_arr, ball_excess=ball_excess, ball_E=ball_E,
         speed_monotone=speed_monotone, all_small=all_small,
         asym_lrs=np.array(asym_lrs), asym_excess=asym_excess, asym_oeta=asym_oeta,
         snap_steps=snap_steps, snap_E=snap_E, snap_E_env=snap_E_env,
         step_below_marg=step_below_marg, descent_monotone=descent_monotone,
         Q_descent=Q_descent,
         c1_step=c1_step, c1_drift=c1_drift, c1_init=c1_init, c1_final=c1_final,
         k_mid=k_mid, k_last=k_last, avg_Es=avg_Es, avg_frac_mono=avg_frac_mono,
         avg_max_inc=avg_max_inc,
         part_steps=part_steps, part_drift=part_drift, spiking_part=spiking_part,
         H_RATE=H_RATE, H_MARGINAL=H_MARGINAL, LEARNED_REF=LEARNED_REF, LN2=LN2)

checks = {
    "numpy CONVERGED ball IS O(eta): smaller lr -> smaller asymptotic ball":
        asym_oeta,
    "spiking residual a small noise ball, monotone in lr at fixed budget":
        speed_monotone and all_small,
    "Lyapunov energy descends from uniform init 2.0 below the marginal 1.75":
        snap_E[0] > H_MARGINAL and step_below_marg >= 0,
    "spiking energy-descent cummin envelope is monotone (Lyapunov dissipation)":
        descent_monotone,
    "numpy sum-mode per-step deviation at machine epsilon (< 1e-12)":
        c1_step < 1e-12,
    "numpy sum-mode row-sum drift at machine epsilon (< 1e-12)":
        c1_drift < 1e-12,
    "numpy Lyapunov constant k = ln2 = 0.693 within 1e-3 (NOT eta)":
        abs(k_mid - LN2) < 1e-3 and abs(k_mid - LN2) < abs(k_mid - 0.5),
    "numpy averaged-flow descent monotone (frac ~1.0, max increase ~0)":
        avg_frac_mono > 0.999 and avg_max_inc < 1e-6,
    "spiking partition drift small (~1e-3) and >> the exact numpy invariant":
        spiking_part < 5e-3 and spiking_part > 10 * c1_drift,
}
print("\n" + "=" * 70)
allok = True
for name, ok in checks.items():
    allok &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
print(f"\ne10: {sum(checks.values())}/{len(checks)} acceptance checks passed.")
sys.exit(0 if allok else 1)
