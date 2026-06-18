#!/usr/bin/env python3
"""
Numerical validation for the LEARNING note (`learning_to_be_unsurprised.typ`).

Companion note 1 (`spiking_entropy_coder.typ`) showed that a calibrated spiking
circuit's per-symbol energy equals the surprisal -log2 q(x_t | context), so the
expected energy is the cross-entropy rate  H(p) + D_KL(p||q), the excess being
the model error. This note DERIVES a local plasticity rule that descends that
excess. The rule is the delta / error-modulated-Hebbian update

    Delta W[i,j] = eta * c_i * ( y_j - q_j ),      (*)

where c_i is the pre-synaptic context activity (1 if last symbol = i), y_j is
the post-synaptic outcome (1 if this symbol = j), and q_j = softmax_j(W[i,:]) is
the predicted/expected post activity. The factor (y_j - q_j) is exactly the
*residual error spike* the encoder already emits by predictive subtraction:
the cost IS the teaching signal.

This script confirms:

  (A) Gradient identity. (*) is exactly stochastic gradient descent on the
      per-symbol cross-entropy (surprisal), for a softmax predictor whose logits
      are the lateral weights W[i,:] of a first-order (previous-symbol) context.

  (B) Lyapunov / energy descent. The deterministic expected energy
      E(W) = sum_i pi_i sum_j P[i,j] (-log2 softmax_j(W[i,:]))
      decreases monotonically under the averaged rule and converges to the
      entropy-rate floor; the gap E(W) - H_rate = excess energy = avg KL -> 0.

  (C) Convergence of the model. The learned conditional law q[i,:] = softmax(W[i,:])
      converges to the true transition matrix P (max abs error -> 0).

Run:  deq/.venv/bin/python research/compression/learn_validate.py
Pure numpy.
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)
LABELS = ["U", "D", "L", "R"]


def xlog2x(p):
    p = np.asarray(p, float)
    out = np.zeros_like(p)
    nz = p > 0
    out[nz] = p[nz] * np.log2(p[nz])
    return out


def entropy_bits(p):
    return float(-xlog2x(p).sum())


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def momentum_chain(pi, s):
    n = len(pi)
    return s * np.eye(n) + (1.0 - s) * np.tile(pi, (n, 1))


def model_energy_and_kl(W, pi, P, H_rate):
    """Deterministic expected per-symbol energy (cross-entropy rate) under the
    current predictor q[i,:] = softmax(W[i,:]), and the average KL (excess)."""
    E = 0.0
    for i in range(4):
        q = softmax(W[i])
        E += pi[i] * float(-(P[i] * np.log2(q)).sum())
    return E, E - H_rate


def learned_matrix(W):
    return np.vstack([softmax(W[i]) for i in range(4)])


# ---------------------------------------------------------------------------
# (A) gradient identity check: analytic SGD gradient == update (*)
# ---------------------------------------------------------------------------
def check_gradient_identity():
    """For loss ell = -log2 q_j with q = softmax(W[i,:]), verify
       d ell / d W[i,k] = (q_k - 1[k=j]) / ln2,
    so -gradient-descent step with rate eta' = eta*ln2 reproduces (*)."""
    rng = np.random.default_rng(0)
    W_row = rng.normal(size=4)
    j = 2
    q = softmax(W_row)
    analytic = (q - np.eye(4)[j]) / np.log(2.0)  # d(-log2 q_j)/dW

    # numerical gradient of -log2 softmax_j
    eps = 1e-6
    num = np.zeros(4)
    for k in range(4):
        Wp = W_row.copy(); Wp[k] += eps
        Wm = W_row.copy(); Wm[k] -= eps
        lp = -np.log2(softmax(Wp)[j])
        lm = -np.log2(softmax(Wm)[j])
        num[k] = (lp - lm) / (2 * eps)

    print("=" * 70)
    print("(A) GRADIENT IDENTITY:  d(-log2 q_j)/dW  ==  (q - onehot_j)/ln2")
    print("=" * 70)
    print("analytic (q - e_j)/ln2 :", analytic)
    print("numerical d ell / dW   :", num)
    print("max abs difference     : %.2e" % np.abs(analytic - num).max())
    print("  -> the delta rule (*) IS SGD on the surprisal; the post-factor")
    print("     (y_j - q_j) is the negative gradient (the residual error spike).")
    print()


# ---------------------------------------------------------------------------
# (B,C) run the online local rule on the momentum rover
# ---------------------------------------------------------------------------
def run_learning():
    pi = np.array([1 / 2, 1 / 4, 1 / 8, 1 / 8])
    s = 0.7
    P = momentum_chain(pi, s)
    H_marg = entropy_bits(pi)
    H_rate = float(pi @ np.array([entropy_bits(P[i]) for i in range(4)]))

    n = 2_000_000
    rng = np.random.default_rng(7)

    # sample the stream
    x = np.empty(n, dtype=int)
    x[0] = rng.choice(4, p=pi)
    for t in range(1, n):
        x[t] = rng.choice(4, p=P[x[t - 1]])

    # online delta rule on lateral weights W (logits per context row)
    W = np.zeros((4, 4))           # init -> uniform predictor q = 1/4
    eta0, t0 = 0.2, 5.0e4          # Robbins-Monro-style decreasing step
    checkpoints = [0, 1_000, 10_000, 100_000, 1_000_000, n - 1]

    print("=" * 70)
    print("(B,C) ONLINE LOCAL LEARNING on the momentum rover (s=0.7)")
    print("      rule:  W[i,:] += eta_t * ( onehot(j) - softmax(W[i,:]) )")
    print("=" * 70)
    print("Floors:  marginal H(pi) = %.4f   entropy rate H = %.4f bits/symbol"
          % (H_marg, H_rate))
    print("Init predictor uniform -> energy = log2(4) = %.4f bits/symbol" %
          np.log2(4))
    print()
    print("%-12s %-10s %-14s %-14s %-12s" %
          ("step t", "eta_t", "energy E(W)", "excess=avg KL", "max|q-P|"))

    snap0 = None
    for t in range(n):
        if t in checkpoints:
            E, kl = model_energy_and_kl(W, pi, P, H_rate)
            mae = np.abs(learned_matrix(W) - P).max()
            eta_t = eta0 / (1.0 + t / t0)
            print("%-12d %-10.5f %-14.4f %-14.4f %-12.4f"
                  % (t, eta_t, E, kl, mae))
            if t == 0:
                snap0 = (E, kl)
        i, j = x[t - 1] if t > 0 else x[0], x[t]
        q = softmax(W[i])
        eta_t = eta0 / (1.0 + t / t0)
        W[i] += eta_t * (np.eye(4)[j] - q)   # the local update (*)

    Q = learned_matrix(W)
    print()
    print("True transition matrix P:")
    for i in range(4):
        print("   from %s : %s" % (LABELS[i], P[i]))
    print("Learned predictor q = softmax(W):")
    for i in range(4):
        print("   from %s : %s" % (LABELS[i], Q[i]))
    print()
    E_final, kl_final = model_energy_and_kl(W, pi, P, H_rate)
    print("FINAL energy            = %.4f bits/symbol (floor %.4f)" %
          (E_final, H_rate))
    print("FINAL excess (avg KL)   = %.4f bits/symbol  (started at %.4f)" %
          (kl_final, snap0[1]))
    print("FINAL max|q - P|        = %.4f" % np.abs(Q - P).max())
    print()
    # Lyapunov check on the AVERAGED (expected / gradient-flow) dynamics:
    #   E[Delta W[i,:]] = eta * pi_i * (P[i,:] - softmax(W[i,:])).
    # This is deterministic gradient descent on E(W); the theorem predicts
    # STRICTLY monotone descent of the excess energy for small enough eta.
    Wm = np.zeros((4, 4))
    eta = 0.5
    energies = []
    for _ in range(3000):
        for i in range(4):
            q = softmax(Wm[i])
            Wm[i] += eta * pi[i] * (P[i] - q)
        energies.append(model_energy_and_kl(Wm, pi, P, H_rate)[0])
    energies = np.array(energies)
    diffs = np.diff(energies)
    frac_mono = float((diffs <= 1e-12).mean())
    print("Lyapunov check on the AVERAGED gradient flow (deterministic):")
    print("   E(W):  start %.4f  ->  end %.6f   (floor %.4f)"
          % (energies[0], energies[-1], H_rate))
    print("   fraction of strictly non-increasing steps = %.4f  (theorem: 1.0)"
          % frac_mono)
    print("   max energy INCREASE over all steps = %.2e  (must be ~0)"
          % max(0.0, float(diffs.max())))
    print("   -> excess energy E - H_rate is a Lyapunov function; it dissipates")
    print("      monotonically. SGD (above) follows it in expectation, to a")
    print("      small noise ball set by the step size.")
    print()


if __name__ == "__main__":
    check_gradient_identity()
    run_learning()
    print("All checks complete.")
