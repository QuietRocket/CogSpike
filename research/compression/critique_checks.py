#!/usr/bin/env python3
"""
Two MISSING numerical checks for the LEARNING note
(`learning_to_be_unsurprised.typ`), complementing `learn_validate.py`.

Conventions reused from learn_validate.py:
    pi = (1/2, 1/4, 1/8, 1/8),  s = 0.7
    momentum_chain(pi, s) = s*I + (1-s) * 1 pi^T
    softmax over lateral weights W (logits)
    online rule:    W[i] += eta * ( onehot(j) - softmax(W[i]) )
    averaged flow:  Wm[i] += dt * pi[i] * ( P[i] - softmax(Wm[i]) )

CHECK 1 (sum-mode invariance):
    Because sum_j onehot(j) = sum_j softmax(W[i,:]) = 1, every online update
    leaves each row-sum of W invariant:  sum_j Delta W[i,j] = 0 exactly.
    We verify the row-sums of W never drift from their initial values along a
    full online run (to machine precision).

CHECK 2 (Lyapunov-rate constant):
    Along the AVERAGED gradient flow  Wm[i] += dt*pi[i]*(P[i]-softmax(Wm[i]))
    with small dt, V(W) = sum_i pi_i KL(P[i] || softmax(W[i]))  measured in BITS.
    Its analytic gradient is  grad V[i,j] = -pi_i (P[i,j]-q[i,j]) / ln2.
    The flow direction is  dW[i,j]/dt = pi_i (P[i,j]-q[i,j]) = -ln2 * grad V[i,j].
    Hence  Vdot = <grad V, dW/dt> = -ln2 * ||grad V||^2,  so the constant is ln2,
    NOT eta. We estimate k = (-dV/dt) / ||grad V||^2 numerically and check it
    converges to ln2 (~0.6931) rather than to eta. This tests the seed
    hypothesis that the paper's "Vdot = -eta ||grad V||^2" dropped a ln2 factor.

Run: deq/.venv/bin/python research/compression/critique_checks.py
Pure numpy.
"""

import numpy as np

np.set_printoptions(precision=6, suppress=True)
LN2 = np.log(2.0)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def momentum_chain(pi, s):
    n = len(pi)
    return s * np.eye(n) + (1.0 - s) * np.tile(pi, (n, 1))


def learned_matrix(W):
    return np.vstack([softmax(W[i]) for i in range(4)])


def V_bits(W, pi, P):
    """V(W) = sum_i pi_i KL(P[i] || softmax(W[i])) in BITS."""
    v = 0.0
    for i in range(4):
        q = softmax(W[i])
        nz = P[i] > 0
        v += pi[i] * float((P[i][nz] * np.log2(P[i][nz] / q[nz])).sum())
    return v


def grad_V(W, pi, P):
    """Analytic gradient of V (bits):  grad V[i,j] = -pi_i (P[i,j]-q[i,j])/ln2."""
    g = np.zeros((4, 4))
    for i in range(4):
        q = softmax(W[i])
        g[i] = -pi[i] * (P[i] - q) / LN2
    return g


# ---------------------------------------------------------------------------
# CHECK 1: sum-mode invariance of the online rule
# ---------------------------------------------------------------------------
def check1_sum_invariance():
    print("=" * 70)
    print("CHECK 1 (sum-mode invariance): sum_j Delta W[i,j] = 0 every step")
    print("=" * 70)
    pi = np.array([1 / 2, 1 / 4, 1 / 8, 1 / 8])
    s = 0.7
    P = momentum_chain(pi, s)

    n = 200_000
    rng = np.random.default_rng(7)
    x = np.empty(n, dtype=int)
    x[0] = rng.choice(4, p=pi)
    for t in range(1, n):
        x[t] = rng.choice(4, p=P[x[t - 1]])

    # start from a NON-zero init so "constant row-sums" is a real test, not
    # trivially zero (zero init would keep row-sums at 0 too).
    W = rng.normal(size=(4, 4))
    init_rowsums = W.sum(axis=1).copy()

    eta0, t0 = 0.2, 5.0e4
    max_step_dev = 0.0   # max |sum_j Delta W[i,j]| over all steps
    max_drift = 0.0      # max |rowsum_t - rowsum_0| over all steps
    for t in range(1, n):
        i, j = x[t - 1], x[t]
        q = softmax(W[i])
        eta_t = eta0 / (1.0 + t / t0)
        dW = eta_t * (np.eye(4)[j] - q)
        step_dev = abs(dW.sum())            # per-step update row-sum
        if step_dev > max_step_dev:
            max_step_dev = step_dev
        W[i] += dW
        drift = np.abs(W.sum(axis=1) - init_rowsums).max()
        if drift > max_drift:
            max_drift = drift

    tol = 1e-9
    passed = (max_step_dev < tol) and (max_drift < tol)
    print("steps run                              : %d" % (n - 1))
    print("initial row-sums of W                  : %s" % init_rowsums)
    print("final   row-sums of W                  : %s" % W.sum(axis=1))
    print("max |sum_j Delta W[i,j]| per step      : %.3e" % max_step_dev)
    print("max abs row-sum DRIFT from init        : %.3e" % max_drift)
    print("tolerance                              : %.1e" % tol)
    print("RESULT: %s" % ("PASS" if passed else "FAIL"))
    print()
    return passed, max_step_dev, max_drift


# ---------------------------------------------------------------------------
# CHECK 2: Lyapunov-rate constant k in Vdot = -k ||grad V||^2  (expect ln2)
# ---------------------------------------------------------------------------
def check2_lyapunov_rate():
    print("=" * 70)
    print("CHECK 2 (Lyapunov-rate constant): Vdot = -k ||grad V||^2,  k = ?")
    print("=" * 70)
    pi = np.array([1 / 2, 1 / 4, 1 / 8, 1 / 8])
    s = 0.7
    P = momentum_chain(pi, s)

    dt = 1e-3            # small step so finite-diff -dV/dt ~ continuous Vdot
    n_steps = 40_000
    Wm = np.zeros((4, 4))

    ks = []
    V_prev = V_bits(Wm, pi, P)
    for step in range(n_steps):
        g = grad_V(Wm, pi, P)
        gnorm2 = float((g * g).sum())
        # advance the averaged gradient flow
        for i in range(4):
            q = softmax(Wm[i])
            Wm[i] += dt * pi[i] * (P[i] - q)
        V_now = V_bits(Wm, pi, P)
        dVdt = (V_now - V_prev) / dt
        V_prev = V_now
        if gnorm2 > 1e-14:
            ks.append((-dVdt) / gnorm2)

    ks = np.array(ks)
    # use a robust window away from the very start (transient finite-diff)
    k_mid = float(np.median(ks[len(ks) // 4 :]))
    k_last = float(ks[-1])
    eta_paper = 0.5      # the "eta" the paper's claimed Vdot=-eta||gradV||^2 uses

    err_ln2 = abs(k_mid - LN2)
    err_eta = abs(k_mid - eta_paper)
    # PASS if measured constant matches ln2 (and is clearly NOT eta)
    passed = (err_ln2 < 1e-3) and (err_ln2 < err_eta)

    print("dt                                     : %.1e" % dt)
    print("steps                                  : %d" % n_steps)
    print("measured k (median, latter 3/4)        : %.6f" % k_mid)
    print("measured k (last step)                 : %.6f" % k_last)
    print("ln2                                    : %.6f" % LN2)
    print("eta (paper's claimed constant)         : %.6f" % eta_paper)
    print("|k - ln2|                              : %.3e" % err_ln2)
    print("|k - eta|                              : %.3e" % err_eta)
    print("conclusion: constant is ln2, NOT eta -> paper's 'Vdot = -eta||gradV||^2'")
    print("            dropped a ln2 factor (V is in BITS; flow uses nats).")
    print("RESULT: %s" % ("PASS" if passed else "FAIL"))
    print()
    return passed, k_mid


if __name__ == "__main__":
    p1, dev, drift = check1_sum_invariance()
    p2, k = check2_lyapunov_rate()
    print("=" * 70)
    print("SUMMARY")
    print("  CHECK 1 (sum-mode invariance): %s" % ("PASS" if p1 else "FAIL"))
    print("  CHECK 2 (Lyapunov-rate = ln2): %s" % ("PASS" if p2 else "FAIL"))
    print("=" * 70)
