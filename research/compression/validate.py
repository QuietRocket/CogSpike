#!/usr/bin/env python3
"""
Numerical validation for the spiking entropy coder note.

Confirms the two load-bearing quantitative claims of
`spiking_entropy_coder.typ`:

  (A) The momentum/mixture rover source  P = s I + (1-s) 1 pi^T  has
      stationary distribution exactly pi = (1/2, 1/4, 1/8, 1/8), marginal
      symbol entropy H(pi) = 1.75 bits, and a *conditional* entropy rate
      strictly below it. The gap is what the recurrent cycle recovers.

  (B) A leaky integrate-and-fire (LIF) readout, driven with the calibrated
      current  R I(q) = theta / (1 - q^alpha),  fires its first spike at
      latency t*(q) = -lambda * log2(q) -- exactly the Shannon codeword
      length. Hence total stream latency = total surprisal, and expected
      per-symbol latency = the cross-entropy rate of the predictor q against
      the source p. With q = p this equals the entropy rate (the floor);
      with q != p it exceeds it by the KL "stupidity tax".

Run:  deq/.venv/bin/python research/compression/validate.py
Pure numpy, no other dependencies.
"""

import numpy as np

np.set_printoptions(precision=6, suppress=True)
LABELS = ["U", "D", "L", "R"]


def xlog2x(p):
    """p * log2 p with the 0*log0 = 0 convention, elementwise."""
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p)
    nz = p > 0
    out[nz] = p[nz] * np.log2(p[nz])
    return out


def entropy_bits(p):
    """Shannon entropy H(p) in bits."""
    return float(-xlog2x(p).sum())


def cross_entropy_bits(p, q):
    """Cross-entropy H(p, q) = -sum p log2 q in bits (q must be > 0 where p > 0)."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    mask = p > 0
    return float(-(p[mask] * np.log2(q[mask])).sum())


def momentum_chain(pi, s):
    """P_ij = s*delta_ij + (1-s)*pi_j  (mixture / 'stay-or-resample' chain)."""
    n = len(pi)
    return s * np.eye(n) + (1.0 - s) * np.tile(pi, (n, 1))


# ---------------------------------------------------------------------------
# (A) Source: stationary distribution, marginal entropy, entropy rate
# ---------------------------------------------------------------------------
def part_A():
    pi = np.array([1 / 2, 1 / 4, 1 / 8, 1 / 8])
    s = 0.7
    P = momentum_chain(pi, s)

    # Stationarity:  pi P = pi
    drift = np.abs(pi @ P - pi).max()

    # Marginal (memoryless) symbol entropy
    H_marg = entropy_bits(pi)

    # Conditional entropies H(.|i) and the entropy rate
    H_cond = np.array([entropy_bits(P[i]) for i in range(4)])
    H_rate = float(pi @ H_cond)

    print("=" * 68)
    print("(A) MOMENTUM ROVER SOURCE   P = s*I + (1-s)*1 pi^T,  s =", s)
    print("=" * 68)
    print("pi (stationary target)      :", pi)
    print("max|pi P - pi| (drift)      : %.2e   <- must be ~0" % drift)
    print()
    print("Transition matrix P (rows = from U,D,L,R):")
    for i, row in enumerate(P):
        print("   from %s : %s" % (LABELS[i], row))
    print()
    print("Per-context conditional entropies H(.|x):")
    for i in range(4):
        print("   H(.|%s) = %.4f bits" % (LABELS[i], H_cond[i]))
    print()
    print("Marginal symbol entropy   H(pi)   = %.4f bits/symbol" % H_marg)
    print("Conditional entropy RATE  H(rate) = %.4f bits/symbol" % H_rate)
    print("Compression the cycle recovers    = %.4f bits/symbol (%.1f%%)"
          % (H_marg - H_rate, 100 * (H_marg - H_rate) / H_marg))
    print()
    return pi, s, P, H_marg, H_rate


# ---------------------------------------------------------------------------
# (B1) LIF first-passage calibration:  t*(q) = -lambda log2 q, exactly
# ---------------------------------------------------------------------------
def lif_first_passage(RI, tau=1.0, theta=1.0, dt=1e-5, t_max=60.0):
    """Numerically integrate tau V' = -V + RI, V(0)=0; return first-crossing
    time of V = theta. Forward Euler on a fine grid (subthreshold, so stable)."""
    if RI <= theta:
        return np.inf  # never reaches threshold (sub-rheobase)
    V, t = 0.0, 0.0
    n = int(t_max / dt)
    for _ in range(n):
        V += dt * (-V + RI) / tau
        t += dt
        if V >= theta:
            return t
    return np.inf


def part_B1():
    tau, theta, lam = 1.0, 1.0, 1.0
    alpha = lam / (tau * np.log(2.0))  # so that t* = -lambda log2 q

    print("=" * 68)
    print("(B1) LIF CALIBRATION  R I(q) = theta/(1 - q^alpha),  alpha = %.4f"
          % alpha)
    print("=" * 68)
    print("Drive each readout so first-spike latency = Shannon length -log2 q.")
    print("%-8s %-12s %-14s %-14s %-10s" %
          ("q", "R*I(q)", "t* numeric", "-log2 q", "abs err"))
    max_err = 0.0
    for q in [0.5, 0.25, 0.125, 0.85, 0.075, 0.0375, 0.98]:
        RI = theta / (1.0 - q ** alpha)
        t_num = lif_first_passage(RI, tau=tau, theta=theta)
        target = -np.log2(q)
        err = abs(t_num - target)
        max_err = max(max_err, err)
        print("%-8.4f %-12.4f %-14.5f %-14.5f %-10.2e"
              % (q, RI, t_num, target, err))
    print()
    print("max |t* - (-log2 q)| over the table = %.2e (Euler dt=1e-5)" % max_err)
    print("  -> latency IS surprisal; the only error is the ODE step size.")
    print()
    return alpha


# ---------------------------------------------------------------------------
# (B2) Stream identity: total latency = total surprisal = cross-entropy
# ---------------------------------------------------------------------------
def sample_stream(P, pi, n, seed=0):
    rng = np.random.default_rng(seed)
    x = np.empty(n, dtype=int)
    x[0] = rng.choice(4, p=pi)
    for t in range(1, n):
        x[t] = rng.choice(4, p=P[x[t - 1]])
    return x


def part_B2(pi, P, H_rate):
    n = 2_000_000
    x = sample_stream(P, pi, n, seed=7)

    # Perfect predictor q = P  (the recurrent loop has learned the true chain)
    surp_perfect = np.array([-np.log2(P[x[t - 1], x[t]]) for t in range(1, n)])

    # Mismatched predictor: a memoryless model q' = pi (ignores the cycle)
    surp_memoryless = np.array([-np.log2(pi[x[t]]) for t in range(1, n)])

    # Mismatched predictor: wrong momentum guess  s' = 0.4
    P_wrong = momentum_chain(pi, 0.4)
    surp_wrong = np.array([-np.log2(P_wrong[x[t - 1], x[t]]) for t in range(1, n)])

    # Theoretical cross-entropy rates (averaged over stationary contexts)
    xH_perfect = float(sum(pi[i] * cross_entropy_bits(P[i], P[i]) for i in range(4)))
    xH_memoryless = float(sum(pi[i] * cross_entropy_bits(P[i], pi) for i in range(4)))
    xH_wrong = float(sum(pi[i] * cross_entropy_bits(P[i], P_wrong[i]) for i in range(4)))

    # KL "stupidity tax" of each mismatched model (= excess latency per symbol)
    kl_memoryless = xH_memoryless - H_rate
    kl_wrong = xH_wrong - H_rate

    print("=" * 68)
    print("(B2) STREAM IDENTITY   mean latency/lambda = cross-entropy rate")
    print("     (n = %d symbols, latency law t* = -log2 q)" % n)
    print("=" * 68)
    print("%-26s %-14s %-14s" % ("predictor q", "empirical", "theory H(p,q)"))
    print("%-26s %-14.4f %-14.4f"
          % ("perfect  q = P", surp_perfect.mean(), xH_perfect))
    print("%-26s %-14.4f %-14.4f"
          % ("memoryless q = pi", surp_memoryless.mean(), xH_memoryless))
    print("%-26s %-14.4f %-14.4f"
          % ("wrong momentum s'=0.4", surp_wrong.mean(), xH_wrong))
    print()
    print("Entropy-rate floor  H(p)            = %.4f bits/symbol" % H_rate)
    print("Stupidity tax of memoryless model   = %.4f bits/symbol (KL)" % kl_memoryless)
    print("Stupidity tax of wrong-momentum     = %.4f bits/symbol (KL)" % kl_wrong)
    print()
    print("Reading: a memoryless spike code spends %.4f extra bits (=spikes/joules)"
          % kl_memoryless)
    print("per symbol relative to the recurrent predictor that captured the cycle.")
    print()


if __name__ == "__main__":
    pi, s, P, H_marg, H_rate = part_A()
    part_B1()
    part_B2(pi, P, H_rate)
    print("All checks complete.")
