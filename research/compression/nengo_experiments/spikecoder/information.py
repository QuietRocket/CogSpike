"""Information measures (bits) and the suite's reference constants.

The functions are lifted byte-compatible from ``validate.py`` / ``learn_validate.py``
so spiking measurements are compared against the identical numbers. The module also
exposes the closed-form constants of the momentum rover at s = 0.7 as *named*
attributes, so every experiment asserts against one canonical value.
"""

import numpy as np

from .source import PI, S, momentum_chain

LN2 = float(np.log(2.0))


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
    """Cross-entropy H(p, q) = -sum p log2 q in bits (q > 0 where p > 0)."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    mask = p > 0
    return float(-(p[mask] * np.log2(q[mask])).sum())


def kl_bits(p, q):
    """Kullback-Leibler divergence D_KL(p || q) in bits."""
    return cross_entropy_bits(p, q) - entropy_bits(p)


def softmax(z):
    """Numerically stable softmax (the exact simplex normalizer)."""
    z = np.asarray(z, float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def entropy_rate(P, pi):
    """Conditional entropy rate H(P) = sum_i pi_i H(P[i,:]) in bits/symbol."""
    pi = np.asarray(pi, float)
    return float(pi @ np.array([entropy_bits(P[i]) for i in range(len(pi))]))


def cross_entropy_rate(P, pi, Q):
    """Cross-entropy rate H(p, q) = sum_i pi_i sum_j P_ij (-log2 Q_ij)."""
    pi = np.asarray(pi, float)
    return float(sum(pi[i] * cross_entropy_bits(P[i], Q[i]) for i in range(len(pi))))


def model_energy_and_kl(W_or_Q, pi, P, H_rate, is_logits=True):
    """Expected per-symbol energy (cross-entropy rate) and excess (avg KL).

    ``W_or_Q`` is either logit rows (is_logits=True; q[i]=softmax(W[i])) or already
    a probability matrix Q (is_logits=False).
    """
    pi = np.asarray(pi, float)
    E = 0.0
    for i in range(len(pi)):
        q = softmax(W_or_Q[i]) if is_logits else np.asarray(W_or_Q[i], float)
        E += pi[i] * float(-(P[i] * np.log2(q)).sum())
    return E, E - H_rate


def learned_matrix(W):
    """Reconstruct the conditional law q[i,:] = softmax(W[i,:]) from logit rows."""
    return np.vstack([softmax(W[i]) for i in range(len(W))])


# --- canonical reference constants of the momentum rover at s = 0.7 -----------
_P = momentum_chain(PI, S)
H_MARGINAL = entropy_bits(PI)          # 1.7500 bits/symbol (memoryless cost)
H_RATE = entropy_rate(_P, PI)          # 0.9782 bits/symbol (conditional floor)
MUTUAL_INFO = H_MARGINAL - H_RATE      # 0.7718 bits/symbol (what the cycle recovers)
LEARNED_REF = 0.9787                   # converged learned energy (learn_validate.py)

# sanity: these must match the paper / validators to 4 decimals
assert abs(H_MARGINAL - 1.7500) < 5e-5, H_MARGINAL
assert abs(H_RATE - 0.9782) < 5e-5, H_RATE
assert abs(MUTUAL_INFO - 0.7718) < 5e-5, MUTUAL_INFO
