"""N-neuron stochastic LI&F oracle.

Wraps `../archetypes/lif_fcs.py:simulate` over a population of N copies of
each "logical" neuron in a topology, with two sources of stochasticity that
together inject Poisson-like input statistics so that Siegert's diffusion
approximation can apply:

1. Per-neuron threshold heterogeneity -- each copy draws an integer threshold
   tau_i ~ tau_0 + Uniform{-eps, ..., +eps}. This randomizes f-I curves
   across the population without changing the FCS-deterministic per-neuron
   rule. With eps small the population mean rate matches the deterministic
   rate; with eps large it smears.
2. Bernoulli per-tick input thinning -- each external-input tick is
   independently dropped with probability 1 - p_thin, so an external drive
   of constant 1 becomes a Bernoulli(p_thin) process. The variance of the
   thinned drive injects sigma^2 into the diffusion approximation.

Network architecture: each logical neuron k in the topology W (n x n) is
replaced by an *independent* sub-population of N copies (no within-population
recurrent coupling -- copies receive the same population-mean input from
other populations). Cross-population coupling is mean-field: the input to a
copy of population i from population j is W[i,j] * mean_spike_rate_j(t-1).

This gives us a finite-N, finite-topology oracle: the population-rate
output A_k(t) = (1/N) sum_{c=1}^N spike_{k,c}(t) is what Siegert's Phi
should predict at steady state, and what the Naud-Gerstner mesoscopic
equation should predict in transients.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARCHETYPES_DIR = HERE.parent / "archetypes"
if str(ARCHETYPES_DIR) not in sys.path:
    sys.path.insert(0, str(ARCHETYPES_DIR))

from lif_fcs import DEFAULT_RVECTOR, DEFAULT_TAU  # noqa: E402


def simulate_population(
    W,
    B,
    external_inputs,
    N=100,
    p_thin=1.0,
    tau_jitter=0,
    tau_mean=DEFAULT_TAU,
    rvector=DEFAULT_RVECTOR,
    T=200,
    seed=None,
):
    """Simulate N independent copies of an n-population network.

    The topology W (n, n) describes coupling between *populations*. Each
    population has N copies; copies within a population are uncoupled and
    receive the same mean-field input from other populations.

    Args:
        W: (n, n) integer coupling matrix. W[i, j] is the weight from
           population j's mean-spike output to each copy in population i.
        B: (n, m) integer external-input weight matrix.
        external_inputs: (m, T) integer/bool array of external drive.
        N: number of copies per logical population.
        p_thin: Bernoulli retention probability for external input ticks
                (1.0 = no thinning, deterministic).
        tau_jitter: integer threshold jitter epsilon. Each copy draws
                    tau_i ~ Uniform{tau_mean - eps, ..., tau_mean + eps}.
                    0 = no jitter.
        tau_mean: mean integer threshold (default 105 = FCS).
        rvector: leak coefficient vector (default FCS [10,5,3,2,1]).
        T: number of ticks.
        seed: numpy seed for reproducibility.

    Returns:
        rates: (n, T) float array. rates[k, t] = (1/N) sum over copies
               of spike_kc(t). The population-mean firing rate per tick.
        spikes: (n, N, T) bool array. Per-copy spike output.
    """
    W = np.asarray(W, dtype=np.int64)
    B = np.asarray(B, dtype=np.int64)
    ext = np.asarray(external_inputs, dtype=np.int64)
    rvec = np.asarray(rvector, dtype=np.int64)

    n = W.shape[0]
    sigma_len = len(rvec)
    assert W.shape == (n, n)
    assert B.shape[0] == n
    m = B.shape[1]
    assert ext.shape == (m, T)

    rng = np.random.default_rng(seed)

    # Per-copy threshold draws (n, N).
    if tau_jitter > 0:
        thresholds = rng.integers(
            tau_mean - tau_jitter, tau_mean + tau_jitter + 1, size=(n, N)
        )
    else:
        thresholds = np.full((n, N), tau_mean, dtype=np.int64)

    # Per-copy, per-tick input thinning. Same external_inputs applied to
    # every copy, independently thinned per (population, copy, tick).
    # Shape: (n, N, m, T).
    if p_thin < 1.0:
        thinning = rng.binomial(1, p_thin, size=(n, N, m, T)).astype(np.int64)
    else:
        thinning = np.ones((n, N, m, T), dtype=np.int64)

    # State buffers.
    mem = np.zeros((n, N, sigma_len), dtype=np.int64)
    localS_prev = np.zeros((n, N), dtype=bool)
    spikes = np.zeros((n, N, T), dtype=bool)
    rates = np.zeros((n, T), dtype=np.float64)

    # Population-mean spike output (used as cross-population recurrent input,
    # mean-field). At each tick t, the recurrent input to each copy of
    # population i is sum_j W[i,j] * Spike_j_mean(t), where Spike_j_mean(t)
    # = (1/N) sum_c spike_{j,c}(t) [emitted Spike, i.e., localS at t-1].
    #
    # To stay integer-faithful, we use the population-mean *count*
    # round(N * mean_spike_rate). When W is integer, this gives an
    # integer-valued recurrent input per copy when divided by N. We
    # implement it as a float-valued input that then gets rounded.

    for t in range(T):
        emitted = localS_prev  # (n, N)
        spikes[:, :, t] = emitted
        # Population-mean spike rate (n,) at tick t.
        mean_emitted = emitted.mean(axis=1)  # (n,)

        # Recurrent input to each copy of population i: scalar W[i,:] @ mean_emitted.
        # Same scalar applied to all N copies of population i.
        recurrent_per_pop = (W.astype(np.float64) @ mean_emitted)  # (n,)

        # External input per (population, copy, tick): B[i,:] @ thinned_external.
        # thinning[i, c, k, t] is the (i,c)'s retention of input k at tick t.
        # External_inputs[k, t] is shared by all (i, c).
        # input_per_copy[i, c] = sum_k B[i, k] * thinning[i, c, k, t] * ext[k, t]
        # Vectorized:
        thinned_ext = thinning[:, :, :, t] * ext[None, None, :, t]  # (n, N, m)
        external_per_copy = np.einsum("ik,nck->nc", B, thinned_ext)  # (n, N) int

        # Total weighted input per copy = round(recurrent_per_pop) + external_per_copy.
        # Using nearest-integer rounding for recurrent (FCS arithmetic is integer).
        recurrent_int = np.rint(recurrent_per_pop).astype(np.int64)  # (n,)
        weighted_input = recurrent_int[:, None] + external_per_copy  # (n, N)

        # Update mem buffer: prepend weighted_input as new mem[0], shift
        # the rest with reset-after-spike semantics.
        if t == 0:
            new_tail = np.zeros((n, N, sigma_len - 1), dtype=np.int64)
        else:
            new_tail = np.where(
                localS_prev[:, :, None], 0, mem[:, :, :-1]
            )
        mem = np.concatenate(
            [weighted_input[:, :, None], new_tail], axis=2
        )  # (n, N, sigma_len)

        # V = mem @ rvector, per copy.
        V = mem @ rvec  # (n, N)
        # Per-copy threshold.
        localS = V >= thresholds  # (n, N)
        localS_prev = localS

        rates[:, t] = localS.mean(axis=1)

    return rates, spikes


def isi_cv(spike_train):
    """Coefficient of variation of inter-spike intervals for a single train.

    Args:
        spike_train: (T,) bool array.

    Returns:
        CV (float). NaN if fewer than 2 spikes.
    """
    times = np.where(spike_train)[0]
    if len(times) < 2:
        return float("nan")
    isi = np.diff(times)
    if isi.mean() == 0:
        return float("nan")
    return float(isi.std() / isi.mean())


def population_isi_cv(spikes_per_pop):
    """Mean ISI CV across copies of one population.

    Args:
        spikes_per_pop: (N, T) bool array of N spike trains for one population.

    Returns:
        Mean CV across copies that have at least 2 spikes (NaN if none do).
    """
    cvs = []
    for c in range(spikes_per_pop.shape[0]):
        cv = isi_cv(spikes_per_pop[c])
        if not np.isnan(cv):
            cvs.append(cv)
    if not cvs:
        return float("nan")
    return float(np.mean(cvs))


def steady_state_rate(rates, tail=None):
    """Time-average a rates array over its tail.

    Args:
        rates: (n, T) array.
        tail: number of trailing ticks to average over. Default T // 2.

    Returns:
        (n,) mean rate per population.
    """
    T = rates.shape[1]
    if tail is None:
        tail = T // 2
    return rates[:, -tail:].mean(axis=1)
