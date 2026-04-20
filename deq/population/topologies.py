"""Archetype weight-matrix builders for the population-level study.

Each builder returns the pair (W, I) needed by the Wilson-Cowan simulator.
Convention: entry W[i, j] is the influence of population j on population i.
"""

from __future__ import annotations

import numpy as np


def contralateral_inhibition(w12: float, w21: float, drive: float = 1.5):
    """Two mutually inhibiting populations with equal external drive.

    w12, w21 are magnitudes of inhibition; both must be non-negative.
    w12 is the strength with which population 1 inhibits population 2.
    w21 is the strength with which population 2 inhibits population 1.
    """
    if w12 < 0 or w21 < 0:
        raise ValueError("Inhibition magnitudes must be non-negative")
    W = np.array([[0.0, -w21], [-w12, 0.0]])
    I = np.array([drive, drive])
    return W, I


def negative_loop(
    w_xa: float,
    w_ai: float,
    w_ia: float,
    u: float = 1.0,
    w_aa: float = 0.0,
    w_ii: float = 0.0,
):
    """Activator-inhibitor negative-feedback loop with constant input u.

    Row 0 = activator A (receives external drive ``w_xa * u`` and
    inhibition ``-w_ia`` from the inhibitor, plus optional self-excitation
    ``w_aa``). Row 1 = inhibitor I (receives excitation ``w_ai`` from A,
    plus optional self-inhibition ``-w_ii``).

    The bare 2-population negative loop with no self-connections has
    trace(J) = -2/tau at every fixed point, so Hopf bifurcation is
    impossible and the activator-inhibitor population cannot oscillate on
    its own. This mirrors the classical Wilson-Cowan result and is
    resolved by including within-population recurrence (``w_aa > 0``),
    which is the natural population-level analogue of lateral excitation
    within the activator pool. ``w_aa`` defaults to zero so existing
    callers see no behavior change.
    """
    W = np.array([[w_aa, -w_ia], [w_ai, -w_ii]])
    I = np.array([w_xa * u, 0.0])
    return W, I


def simple_series(weights, drive: float = 1.5):
    """Feed-forward chain of n populations.

    weights is a length-(n-1) iterable giving the gain from population i to i+1.
    Only population 0 receives the external drive.
    """
    weights = list(weights)
    n = len(weights) + 1
    W = np.zeros((n, n))
    for i, w in enumerate(weights):
        W[i + 1, i] = w
    I = np.zeros(n)
    I[0] = drive
    return W, I


def parallel_composition(w_in, n: int, drive: float = 1.5):
    """One driver feeding n parallel downstream populations.

    w_in may be a scalar (equal gains) or a length-n iterable.
    State ordering: index 0 is the driver; indices 1..n are the parallel pool.
    """
    if np.isscalar(w_in):
        w_in_vec = np.full(n, float(w_in))
    else:
        w_in_vec = np.asarray(w_in, dtype=float)
        if w_in_vec.size != n:
            raise ValueError("w_in length must equal n")
    size = n + 1
    W = np.zeros((size, size))
    for k in range(n):
        W[k + 1, 0] = w_in_vec[k]
    I = np.zeros(size)
    I[0] = drive
    return W, I


def positive_loop(w_12: float, w_21: float, drive: float = 0.0):
    """Two mutually exciting populations (for saddle-node contrast study)."""
    if w_12 < 0 or w_21 < 0:
        raise ValueError("Excitation magnitudes must be non-negative")
    W = np.array([[0.0, w_21], [w_12, 0.0]])
    I = np.array([drive, drive])
    return W, I
