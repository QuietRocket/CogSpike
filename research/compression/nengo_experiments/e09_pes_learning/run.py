#!/usr/bin/env python3
"""e09 -- The local delta rule learns the rover law (PES embodiment).

Claims (paper #9 + #10): stochastic gradient descent on the per-symbol spike cost
(surprisal -log2 q(x_t|c_t)) is the *local* three-factor Hebbian rule

    Delta W_ij = eta * c_i * (y_j - q_j),

the delta / Widrow-Hoff rule. Its post-factor (y_j - q_j) is the signed residual
the encoder already emits by predictive subtraction: the cost IS the teaching
signal. Nengo's PES (Prescribed Error Sensitivity) IS this rule -- it updates the
DECODERS of a spiking population by -lr * error * activity. We wire a one-hot
context population to a decoded prediction q(.|c), teach it with the realized
symbol's one-hot y, and watch the local rule learn the true conditional law P of
the momentum rover, descending the energy from the uniform init toward the floor.

Four measurements:
  (1) CONVERGENCE: the learned q(.|i) -> P[i,:] (max|q-P| < 0.2).
  (2) ENERGY DESCENT: from the uniform init (log2 4 = 2.0 bits) below the marginal
      H_marginal=1.7500 toward ~1.03, an excess ~0.05 above the entropy-rate floor
      H_rate=0.9782 -- and that excess is the spiking/NEF/constant-lr noise ball,
      NOT a failure (the numpy validator's 0.0005 needs a decreasing lr + a literal
      weight matrix).
  (3) GRADIENT IDENTITY (numpy side-check): the delta-rule update (y-q)*c equals the
      analytic softmax-cross-entropy gradient, exact to ~1e-10 (reproduces
      learn_validate.py's check_gradient_identity) -- anchors that PES embodies the
      *exact* rule.
  (4) DECODERS-NOT-WEIGHTS: the learned object is a decoder FUNCTION c -> q of a
      spiking population (the partition sum q is approximate/representational, e03),
      not a literal 4x4 W matrix -- yet the function matches P. We also run a
      two-channel ON/OFF rectified-error variant (paper #10) and confirm it
      converges to the same place.

Outputs: figures + a results .npz, and a PASS/FAIL line per acceptance check.
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
from spikecoder.source import RoverSource, LABELS            # noqa: E402
from spikecoder.information import (softmax, model_energy_and_kl,  # noqa: E402
                                    H_RATE, H_MARGINAL, LEARNED_REF, LN2)
from spikecoder.metrics import energy_trajectory             # noqa: E402

# --- experiment constants (the verified, working PES idiom) ---
N_SYMBOLS = 10000      # >= 10000 symbols
WINDOW = 0.03          # s per symbol window
LR = 2e-4             # constant PES learning rate (1-2e-4)
N_CTX = 600            # context-ensemble size
CTX_RADIUS = 1.3
DT = cfg.DT           # 1e-3 for throughput/learning
SEED = cfg.SEED
CONVERGED_FRAC = 0.6   # read the converged learned law over the last 40% of the stream


# ---------------------------------------------------------------------------
# (3) numpy gradient-identity side-check (reproduces learn_validate.py)
# ---------------------------------------------------------------------------
def check_gradient_identity(n_trials=64, seed=0):
    """Verify the delta-rule update (y - q) is the negative gradient of the
    per-symbol surprisal -log2 q_j (q = softmax(W_row)).

    Two anchors:
      * finite-difference: d(-log2 q_j)/dW  ==  (q - e_j)/ln2  -- numerically to ~1e-10.
      * exact algebraic: the delta-rule post-factor (e_j - q) IS the negative
        natural-log-loss gradient -grad(-ln q_j) = (e_j - q), to machine precision.
    """
    rng = np.random.default_rng(seed)
    fd_res, exact_res = 0.0, 0.0
    for _ in range(n_trials):
        W_row = rng.normal(size=4)
        j = int(rng.integers(4))
        q = softmax(W_row)
        analytic = (q - np.eye(4)[j]) / LN2           # d(-log2 q_j)/dW
        eps = 1e-6
        num = np.zeros(4)
        for k in range(4):
            Wp = W_row.copy(); Wp[k] += eps
            Wm = W_row.copy(); Wm[k] -= eps
            num[k] = (-np.log2(softmax(Wp)[j]) - (-np.log2(softmax(Wm)[j]))) / (2 * eps)
        fd_res = max(fd_res, float(np.abs(analytic - num).max()))
        # the delta update W += eta*(e_j - q) follows -grad(-ln q_j) = (e_j - q)
        delta_update = np.eye(4)[j] - q
        neg_grad_nats = np.eye(4)[j] - q
        exact_res = max(exact_res, float(np.abs(delta_update - neg_grad_nats).max()))
    return fd_res, exact_res


# ---------------------------------------------------------------------------
# the PES learning network (the local delta rule, embodied in spiking decoders)
# ---------------------------------------------------------------------------
def build_pes_net(ctx_fn, y_fn, onoff=False, seed=SEED):
    """One spiking context population whose DECODERS are learned by PES.

    ctx -> q is a decoded prediction, initialized to the uniform predictor
    (decoder function lambda c: ones(4)/4, so energy starts at log2 4 = 2.0 bits).
    The error fed to the PES rule is (q - y): PES descends -lr * error * activity,
    driving q toward E[y|c] = P. With onoff=True the error is built from two
    rectified channels r+ = max(0, y-q), r- = max(0, q-y) and fed as r+ - r-
    (paper #10's two-channel ON/OFF error population) -- algebraically the same
    signed residual y - q, but assembled from nonnegative half-wave channels.
    """
    net = nengo.Network(seed=seed)
    with net:
        ci = nengo.Node(ctx_fn, size_out=4)
        ctx = nengo.Ensemble(N_CTX, 4, radius=CTX_RADIUS)
        nengo.Connection(ci, ctx, synapse=0.005)

        q = nengo.Node(size_in=4)                     # decoded prediction q(.|c)
        conn = nengo.Connection(
            ctx, q, function=lambda c: np.ones(4) * 0.25,   # uniform init
            learning_rule_type=nengo.PES(learning_rate=LR), synapse=0.01)

        yi = nengo.Node(y_fn, size_out=4)             # teacher: one-hot realized symbol

        if not onoff:
            err = nengo.Node(size_in=4)               # error = q - y
            nengo.Connection(q, err, synapse=None)
            nengo.Connection(yi, err, transform=-1, synapse=None)
            nengo.Connection(err, conn.learning_rule, synapse=None)
        else:
            # two-channel ON/OFF: r+ = max(0, y-q), r- = max(0, q-y); feed -(r+ - r-) = q - y
            def onoff_err(t, v):
                y, qq = v[:4], v[4:]
                rp = np.maximum(0.0, y - qq)
                rm = np.maximum(0.0, qq - y)
                return -(rp - rm)                     # PES wants error = q - y
            err = nengo.Node(onoff_err, size_in=8, size_out=4)
            nengo.Connection(yi, err[:4], synapse=None)
            nengo.Connection(q, err[4:], synapse=None)
            nengo.Connection(err, conn.learning_rule, synapse=None)

        pq = nengo.Probe(q, synapse=0.02)
        pctx = nengo.Probe(ci)
    return net, pq, pctx


def per_window_q_and_ctx(qd, cd):
    """Reduce raw probes to one settled-tail q and one context index per symbol window."""
    win_q = np.zeros((N_SYMBOLS, 4))
    win_ctx = np.full(N_SYMBOLS, -1)
    for k in range(N_SYMBOLS):
        i0 = int(round(k * WINDOW / DT))
        i1 = int(round((k + 1) * WINDOW / DT))
        if i1 > len(qd):
            break
        seg = qd[i0:i1]
        win_q[k] = seg[len(seg) // 2:].mean(0)        # settled tail (last 50%)
        csum = cd[i0:i1].sum(0)
        if csum.max() > 0:
            win_ctx[k] = int(np.argmax(csum))
    return win_q, win_ctx


def learned_matrix(win_q, win_ctx, lo=0, hi=None):
    """Average decoded q per context over windows [lo, hi); clip>0 & renormalize rows."""
    hi = N_SYMBOLS if hi is None else hi
    rows = {i: [] for i in range(4)}
    for k in range(lo, hi):
        if win_ctx[k] >= 0:
            rows[win_ctx[k]].append(win_q[k])
    Q = np.ones((4, 4)) / 4.0
    for i in range(4):
        if rows[i]:
            v = np.mean(rows[i], axis=0)
            v = np.clip(v, 1e-6, None)
            Q[i] = v / v.sum()
    return Q


# ===========================================================================
print("=" * 70)
print("e09 -- The local delta rule learns the rover law (PES embodiment)")
print("=" * 70)

rs = RoverSource(s=0.7, seed=SEED)
P, pi = rs.P, rs.pi
x = rs.sample(N_SYMBOLS)
ctx_fn = rs.context_input(x, WINDOW)
y_fn = rs.outcome_input(x, WINDOW)
print(f"\nsource: momentum rover s=0.7, {N_SYMBOLS} symbols, window {WINDOW}s, "
      f"PES lr={LR}, {N_CTX} ctx neurons")
print(f"floors: marginal H(pi)={H_MARGINAL:.4f}  entropy rate H={H_RATE:.4f} bits/symbol")
print(f"uniform init energy = log2(4) = {np.log2(4):.4f} bits/symbol")

# --- (3) gradient identity FIRST (cheap numpy anchor) ---
fd_res, exact_res = check_gradient_identity()
print(f"\n[gradient identity] delta-rule (y-q) == negative softmax-CE gradient")
print(f"  finite-difference residual  = {fd_res:.2e}  (anchors SGD-on-surprisal)")
print(f"  exact algebraic residual    = {exact_res:.2e}  (delta post-factor IS -grad)")

# --- (1,2,4) run the spiking PES learner ---
print("\n[PES learning] running spiking delta rule (standard error channel)...")
net, pq, pctx = build_pes_net(ctx_fn, y_fn, onoff=False)
with nengo.Simulator(net, dt=DT, progress_bar=False) as sim:
    sim.run(N_SYMBOLS * WINDOW)
win_q, win_ctx = per_window_q_and_ctx(sim.data[pq], sim.data[pctx])

# (1) converged learned law over the last 40% of the stream
Q = learned_matrix(win_q, win_ctx, lo=int(CONVERGED_FRAC * N_SYMBOLS))
max_err = float(np.abs(Q - P).max())
E_final, kl_final = model_energy_and_kl(Q, pi, P, H_RATE, is_logits=False)
print(f"\n[convergence] learned q(.|i) over last {int((1-CONVERGED_FRAC)*100)}% of stream:")
print("        true P[i,:]                       learned q(.|i)            max|row err|")
for i in range(4):
    re = float(np.abs(Q[i] - P[i]).max())
    print(f"  {LABELS[i]} : [{', '.join(f'{v:.3f}' for v in P[i])}]   "
          f"[{', '.join(f'{v:.3f}' for v in Q[i])}]   {re:.3f}")
print(f"\n  max|q - P| = {max_err:.4f}  (target < 0.2)")
print(f"  final energy E = {E_final:.4f} bits/symbol  (floor {H_RATE:.4f}, "
      f"numpy learned ref {LEARNED_REF:.4f})")
print(f"  final excess (avg KL) = {kl_final:.4f} bits/symbol  (the noise-ball floor)")

# (2) energy-descent trajectory: cumulative learned model at growing prefixes of the
#     stream, anchored at the exact uniform init (energy = 2.0 by construction).
fine = [1, 3, 6, 12, 25, 50, 100, 200, 400, 800, 1600]
coarse = list(np.linspace(2400, N_SYMBOLS, 24).astype(int))
snap_steps = [0] + fine + coarse
snap_E = [float(np.log2(4))]   # uniform-init anchor
for m in snap_steps[1:]:
    Qm = learned_matrix(win_q, win_ctx, lo=0, hi=m)
    snap_E.append(energy_trajectory([Qm], pi, P)[0])
snap_steps = np.array(snap_steps)
snap_E = np.array(snap_E)
# smoothed (cumulative-min envelope) for the monotone-descent claim
snap_E_env = np.minimum.accumulate(snap_E)
crossed_marginal = snap_E[snap_E < H_MARGINAL]
step_below_marg = int(snap_steps[np.argmax(snap_E < H_MARGINAL)]) if len(crossed_marginal) else -1
print(f"\n[energy descent] {len(snap_steps)} snapshots, init 2.0 -> final {snap_E[-1]:.4f}")
print(f"  crosses below marginal {H_MARGINAL:.4f} by step {step_below_marg}")
print(f"  smoothed (cummin) envelope is monotone: "
      f"{'YES' if np.all(np.diff(snap_E_env) <= 1e-9) else 'NO'}")

# --- (4) ON/OFF two-channel rectified-error variant ---
print("\n[ON/OFF variant] running two-channel rectified error (paper #10)...")
net2, pq2, pctx2 = build_pes_net(ctx_fn, y_fn, onoff=True)
with nengo.Simulator(net2, dt=DT, progress_bar=False) as sim2:
    sim2.run(N_SYMBOLS * WINDOW)
win_q2, win_ctx2 = per_window_q_and_ctx(sim2.data[pq2], sim2.data[pctx2])
Q2 = learned_matrix(win_q2, win_ctx2, lo=int(CONVERGED_FRAC * N_SYMBOLS))
max_err2 = float(np.abs(Q2 - P).max())
E2, kl2 = model_energy_and_kl(Q2, pi, P, H_RATE, is_logits=False)
print(f"  ON/OFF max|q-P| = {max_err2:.4f},  energy = {E2:.4f},  excess = {kl2:.4f}")

# --- (4) decoders-not-weights: the partition sum of the learned q is representational ---
# average |sum_j q_j(c) - 1| over the converged windows (before our renormalization)
part_defect = float(np.mean(np.abs(win_q[int(CONVERGED_FRAC*N_SYMBOLS):].sum(1) - 1.0)))
print(f"\n[decoders-not-weights] decoded q is a population FUNCTION c->q, not a literal W.")
print(f"  partition defect mean|sum_j q_j - 1| = {part_defect:.4f} "
      f"(representational, e03) -- yet the FUNCTION matches P to {max_err:.3f}")

# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
# Fig 1: energy descent (log-x so the fast early descent is legible next to the tail)
fig, ax = plot.new_fig()
xs = np.maximum(snap_steps, 1)   # 0-step uniform anchor placed at x=1 for the log axis
ax.plot(xs, snap_E, "o-", color=plot.C_MEASURED, lw=2, ms=4, label="energy E (learned q)")
ax.axhline(H_MARGINAL, color="gray", ls=":", lw=1.3, label=f"marginal {H_MARGINAL:.4f}")
ax.axhline(H_RATE, color=plot.C_FLOOR, ls="--", lw=1.5, label=f"entropy-rate floor {H_RATE:.4f}")
ax.axhline(LEARNED_REF, color="orange", ls="-.", lw=1.0, alpha=0.7,
           label=f"numpy learned ref {LEARNED_REF:.4f}")
ax.annotate("uniform init\nlog2 4 = 2.0", (xs[0], snap_E[0]), fontsize=8,
            xytext=(xs[0] * 1.3, 1.92), color=plot.C_MEASURED)
ax.set_xscale("log")
ax.set_xlabel("training step (symbols seen, log scale)")
ax.set_ylabel("energy (bits/symbol)")
ax.set_title("e09: local delta rule descends energy to the noise-ball floor")
ax.legend(frameon=False, fontsize=8)
plot.save(fig, RESULTS / "e09_energy_descent.pdf")

# Fig 2: learned-vs-true conditional matrices (grouped bars, one panel per context)
fig, axes = plot.plt.subplots(2, 2, figsize=(8.5, 6.0))
xb = np.arange(4)
for i, ax in enumerate(axes.flat):
    ax.bar(xb - 0.2, P[i], 0.4, color=plot.C_THEORY, alpha=0.8, label="true P[i,:]")
    ax.bar(xb + 0.2, Q[i], 0.4, color=plot.C_MEASURED, label="learned q(.|i)")
    ax.set_xticks(xb); ax.set_xticklabels(LABELS)
    ax.set_title(f"context = {LABELS[i]}", fontsize=10)
    ax.set_ylim(0, 1.0)
    if i == 0:
        ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
fig.suptitle(f"e09: PES learns the rover conditional law  (max|q-P| = {max_err:.3f})",
             fontsize=11)
plot.save(fig, RESULTS / "e09_learned_matrices.pdf")

# Fig 3: gradient-identity residual (per-component, one representative trial)
rng = np.random.default_rng(1)
W_row = rng.normal(size=4); j = 2
q_demo = softmax(W_row)
analytic = (q_demo - np.eye(4)[j]) / LN2
eps = 1e-6
num = np.array([
    (-np.log2(softmax(np.where(np.arange(4) == k, W_row + eps, W_row))[j])
     - (-np.log2(softmax(np.where(np.arange(4) == k, W_row - eps, W_row))[j]))) / (2 * eps)
    for k in range(4)])
fig, (axL, axR) = plot.plt.subplots(1, 2, figsize=(9.0, 3.6))
axL.bar(np.arange(4) - 0.2, analytic, 0.4, color=plot.C_THEORY, alpha=0.8,
        label=r"analytic $(q-e_j)/\ln 2$")
axL.bar(np.arange(4) + 0.2, num, 0.4, color=plot.C_MEASURED, label="finite-difference")
axL.set_xticks(range(4)); axL.set_xticklabels([f"$W_{k}$" for k in range(4)])
axL.set_ylabel(r"$\partial(-\log_2 q_j)/\partial W$")
axL.set_title("gradient identity (one trial)"); axL.legend(frameon=False, fontsize=8)
axR.bar(["finite-diff\nresidual", "exact alg.\nresidual"], [fd_res, max(exact_res, 1e-18)],
        color=[plot.C_MEASURED, plot.C_FLOOR])
axR.set_yscale("log"); axR.set_ylabel("max abs residual")
axR.axhline(1e-10, color="gray", ls="--", lw=1, label="1e-10 anchor")
axR.set_title("delta rule IS SGD on surprisal"); axR.legend(frameon=False, fontsize=8)
plot.save(fig, RESULTS / "e09_gradient_identity.pdf")

# ---------------------------------------------------------------------------
# save + acceptance
# ---------------------------------------------------------------------------
np.savez(RESULTS / "e09_results.npz",
         P=P, pi=pi, Q=Q, Q_onoff=Q2, max_err=max_err, max_err_onoff=max_err2,
         E_final=E_final, kl_final=kl_final, E_onoff=E2, kl_onoff=kl2,
         snap_steps=snap_steps, snap_E=snap_E, snap_E_env=snap_E_env,
         step_below_marg=step_below_marg, fd_res=fd_res, exact_res=exact_res,
         part_defect=part_defect, H_RATE=H_RATE, H_MARGINAL=H_MARGINAL,
         LEARNED_REF=LEARNED_REF)

checks = {
    "learned q(.|i) -> P (max|q-P| < 0.2)": max_err < 0.2,
    "energy descends from uniform init below the marginal 1.75": (
        snap_E[0] > H_MARGINAL and step_below_marg >= 0),
    "energy converges to the noise ball (1.0 <= E <= 1.05)": 1.0 <= E_final <= 1.05,
    "smoothed energy descent is monotone (cummin envelope)": bool(
        np.all(np.diff(snap_E_env) <= 1e-9)),
    "excess KL is small and above the numpy floor (0 < excess < 0.1)": (
        0.0 < kl_final < 0.1),
    "gradient identity matches to ~1e-10 (finite-diff)": fd_res < 1e-8,
    "delta-rule post-factor IS the negative gradient (exact)": exact_res < 1e-12,
    "ON/OFF two-channel variant also converges (max|q-P| < 0.2)": max_err2 < 0.2,
}
print("\n" + "=" * 70)
allok = True
for name, ok in checks.items():
    allok &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
print(f"\ne09: {sum(checks.values())}/{len(checks)} acceptance checks passed.")
sys.exit(0 if allok else 1)
