"""The momentum-rover Markov source.

Lifted byte-compatible from ``research/compression/validate.py`` so every spiking
result is asserted against the *same* numbers the paper and the numpy validators
use. A rover moves on the alphabet {U, D, L, R} with long-run move frequencies
``pi = (1/2, 1/4, 1/8, 1/8)`` but with *inertia*: with stickiness ``s`` it repeats
its last move, otherwise it draws a fresh move from ``pi``. Its transition matrix
is ``P_ij = s * 1[i=j] + (1-s) * pi_j`` -- a convex blend of the identity (pure
momentum) and the rank-one ``1 pi^T`` (pure memorylessness).

At s = 0.7: marginal entropy H(pi) = 1.7500 bits, entropy rate H(P) = 0.9782 bits,
mutual information I = 0.7718 bits (44% of the marginal cost the cycle can recover).
"""

import numpy as np

from .config import SEED

LABELS = ["U", "D", "L", "R"]
PI = np.array([1 / 2, 1 / 4, 1 / 8, 1 / 8])
S = 0.7


def momentum_chain(pi, s):
    """P_ij = s*delta_ij + (1-s)*pi_j  (the 'stay-or-resample' chain)."""
    pi = np.asarray(pi, float)
    n = len(pi)
    return s * np.eye(n) + (1.0 - s) * np.tile(pi, (n, 1))


def sample_stream(P, pi, n, seed=0):
    """Sample an n-symbol stream from the chain P started at pi."""
    rng = np.random.default_rng(seed)
    N = len(pi)
    x = np.empty(n, dtype=int)
    x[0] = rng.choice(N, p=pi)
    for t in range(1, n):
        x[t] = rng.choice(N, p=P[x[t - 1]])
    return x


class RoverSource:
    """A momentum-rover source with streaming helpers for Nengo input Nodes.

    Parameters
    ----------
    s : float
        Stickiness in [0, 1). Default 0.7 (the paper's worked operating point).
    pi : array
        Stationary move frequencies. Default (1/2, 1/4, 1/8, 1/8).
    seed : int
        Stream sampling seed. Default config.SEED (= 7, the validators' seed).
    """

    def __init__(self, s=S, pi=PI, seed=SEED):
        self.pi = np.asarray(pi, float)
        self.N = len(self.pi)
        self.seed = seed
        self.set_s(s)

    def set_s(self, s):
        """(Re)build the transition matrix for stickiness s (for non-stationary runs)."""
        self.s = float(s)
        self.P = momentum_chain(self.pi, self.s)
        return self

    # convenience alias used by the non-stationary capstone
    switch_s = set_s

    def sample(self, n, seed=None):
        """Sample an n-symbol stream (ints in 0..N-1)."""
        return sample_stream(self.P, self.pi, n, seed=self.seed if seed is None else seed)

    # ---- Nengo input-function builders -------------------------------------
    def context_input(self, symbols, window_dur):
        """Return f(t) -> one-hot of the *previous* symbol (the context c_t).

        Each symbol occupies a window of length ``window_dur`` seconds. During
        window t the context is the one-hot encoding of symbol x_{t-1} (for t=0,
        the all-zeros / uniform context).
        """
        symbols = np.asarray(symbols, int)
        N = self.N

        def f(t):
            k = int(t // window_dur)
            out = np.zeros(N)
            if 0 < k < len(symbols):
                out[symbols[k - 1]] = 1.0
            return out

        return f

    def outcome_input(self, symbols, window_dur):
        """Return f(t) -> one-hot of the *current* symbol x_t (the teacher y)."""
        symbols = np.asarray(symbols, int)
        N = self.N

        def f(t):
            k = int(t // window_dur)
            out = np.zeros(N)
            if 0 <= k < len(symbols):
                out[symbols[k]] = 1.0
            return out

        return f
