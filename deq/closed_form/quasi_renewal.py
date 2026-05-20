"""Quasi-renewal mesoscopic equation (Naud-Gerstner 2012, single integral).

For a finite population of N LI&F neurons, the population activity
A(t) = (1/N) sum_n delta(t - t_n_last) follows the integral equation

    A(t) = int_0^infty rho(tau; mu(t)) * S(tau; mu(t)) * A(t - tau) dtau
           + (1/sqrt(N)) * xi(t) * sqrt(A(t))

where
    rho(tau; mu) = hazard rate of firing at age tau under input mu
    S(tau; mu)   = exp(-int_0^tau rho(s; mu) ds)  =  survival probability

In the *Markov / quasi-renewal* approximation we discretize age tau in
units of dt and track the age-distribution

    m_k(t) = fraction of population with last spike k ticks ago

with update

    A(t) = sum_k m_k(t-1) * h(k; mu(t)) * dt
    m_0(t) = A(t)
    m_k(t) = m_{k-1}(t-1) * (1 - h(k-1; mu(t)) * dt)  for k >= 1

The hazard h(k; mu, sigma) at age k is the probability density of crossing
threshold under noisy input. For k >= tau_ref it equals the Siegert
firing rate Phi(mu, sigma) (memoryless approximation; exact only at
t -> infinity steady state). For k < tau_ref it is zero (refractory
period). This is the simplest mesoscopic correction to mean-field WC
that includes BOTH refractoriness AND finite-size noise.

Reduces to mean-field WC at large N (noise -> 0) and to deterministic
LI&F-like dynamics when sigma -> 0 (hazard becomes a step function).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from siegert import Siegert  # noqa: E402


@dataclass
class QuasiRenewal:
    siegert: Siegert
    K_max: int = 30  # maximum age tracked (in ticks)
    tau_ref_ticks: int = 1  # refractory period in ticks
    dt: float = 1.0

    def equilibrium_init(self, n_pops: int, A0: float = 0.1) -> np.ndarray:
        """Geometric age-distribution m_k = A0 (1 - A0)^k, normalized."""
        m = np.zeros((n_pops, self.K_max))
        for k in range(self.K_max):
            m[:, k] = A0 * (1 - A0) ** k
        # Normalize.
        m /= m.sum(axis=1, keepdims=True)
        return m

    def step(
        self,
        m: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        N: int,
        rng=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advance one tick.

        Args:
            m: (n_pops, K_max) age-distribution at start of tick.
            mu: (n_pops,) per-pop input mean.
            sigma: (n_pops,) per-pop input std.
            N: population size for finite-size noise (None = no noise).
            rng: numpy default_rng (or None for deterministic).

        Returns:
            (m_new, A_new) where A_new is per-pop firing rate this tick.
        """
        n_pops = m.shape[0]
        # Per-population hazard h = Phi(mu, sigma).
        h = np.array([self.siegert.phi(mu[i], sigma[i]) for i in range(n_pops)])
        h = np.clip(h, 0.0, 1.0 / self.dt)

        # Per-age effective hazard: 0 for age < tau_ref, h otherwise.
        h_eff = np.zeros((n_pops, self.K_max))
        for k in range(self.K_max):
            if k >= self.tau_ref_ticks:
                h_eff[:, k] = h

        # Mass that fires this tick.
        pf = m * h_eff * self.dt  # (n_pops, K_max)
        A_new = pf.sum(axis=1)  # (n_pops,)

        # Finite-size noise.
        if rng is not None and N is not None and N > 0:
            noise = rng.standard_normal(n_pops) * np.sqrt(np.clip(A_new, 0, 1) / N)
            A_new = np.clip(A_new + noise, 0.0, 1.0)

        # Update age-distribution: survivors advance one age, new fires reset to 0.
        m_new = np.zeros_like(m)
        m_new[:, 0] = A_new
        for k in range(1, self.K_max):
            m_new[:, k] = m[:, k - 1] * (1.0 - h_eff[:, k - 1] * self.dt)

        # Renormalize against drift (numerical / noise leakage).
        sums = m_new.sum(axis=1, keepdims=True)
        sums = np.where(sums > 1e-9, sums, 1.0)
        m_new = m_new / sums

        return m_new, A_new


def simulate_contralateral(
    w12: float,
    w21: float,
    drive: float,
    p_thin: float,
    qr: QuasiRenewal,
    alpha: float,
    beta: float,
    N: int,
    T: int,
    seed: int = 0,
    init_A: Tuple[float, float] = (0.5, 0.1),
) -> np.ndarray:
    """Run quasi-renewal contralateral motif for T ticks.

    Initial condition: asymmetric init_A = (high, low) -- the symmetry
    breaker plays the role of N1's first-fire advantage in the LI&F oracle.

    Args:
        w12, w21: contralateral inhibition weights (FCS units, negative).
        drive: external drive scalar (FCS units).
        p_thin: input thinning rate.
        qr: QuasiRenewal instance.
        alpha, beta: calibration mapping FCS -> Siegert.
        N: population size for finite-size noise.
        T: number of ticks.
        seed: rng seed.
        init_A: (A1_init, A2_init) initial activity asymmetry.

    Returns:
        rates: (2, T) array of per-tick population rates.
    """
    rng = np.random.default_rng(seed)
    # Pre-initialize each population at its initial activity.
    m = np.zeros((2, qr.K_max))
    for i, A0 in enumerate(init_A):
        # Geometric distribution.
        for k in range(qr.K_max):
            m[i, k] = max(A0, 1e-3) * (1 - max(A0, 1e-3)) ** k
        m[i] /= m[i].sum()
    A_prev = np.array(init_A, dtype=float)
    rates = np.zeros((2, T))

    for t in range(T):
        # Inputs: mean and variance per population.
        # Population 1 receives w_21 * A_2; population 2 receives w_12 * A_1.
        mean_in_1 = drive * p_thin + w21 * A_prev[1]
        mean_in_2 = drive * p_thin + w12 * A_prev[0]
        var_in_1 = (drive ** 2) * p_thin * (1 - p_thin) + (w21 ** 2) * A_prev[1] * (1 - A_prev[1])
        var_in_2 = (drive ** 2) * p_thin * (1 - p_thin) + (w12 ** 2) * A_prev[0] * (1 - A_prev[0])
        mu = np.array([alpha * mean_in_1, alpha * mean_in_2])
        sigma = np.array([
            math.sqrt(max(beta * var_in_1, 0.0)),
            math.sqrt(max(beta * var_in_2, 0.0)),
        ])
        m, A = qr.step(m, mu, sigma, N, rng=rng)
        rates[:, t] = A
        A_prev = A

    return rates


def simulate_uniform_inhibition(
    N_neurons: int,
    w: float,
    drive: float,
    p_thin: float,
    qr: QuasiRenewal,
    alpha: float,
    beta: float,
    N_pop: int,
    T: int,
    seed: int = 0,
    init_A=None,
    drive_bump: float = 0.0,
) -> np.ndarray:
    """Run quasi-renewal N-neuron all-to-all inhibition for T ticks.

    Generalizes simulate_contralateral to arbitrary N. Each population i
    receives recurrent inhibition w * sum_{j != i} A_prev[j] and external
    drive (drive + drive_bump_i) * p_thin with drive_bump_i = drive_bump if
    i == 0 else 0. Uses qr.step() unchanged (already vectorized over
    populations).

    Default init_A = linspace(0.5, 0.1, N_neurons) gives neuron 0 the lead,
    matching the 2-neuron init_A=(0.5, 0.1) convention.

    Args:
        N_neurons: number of populations.
        w: per-edge inhibitory weight in FCS units.
        drive, p_thin, alpha, beta: FCS->Siegert calibration constants.
        qr: QuasiRenewal instance.
        N_pop: per-population size (sqrt-N finite-size noise).
        T: number of ticks.
        seed: rng seed.
        init_A: optional (N_neurons,) initial activity.
        drive_bump: float added to neuron 0's external drive (symmetric-breaker).

    Returns:
        rates: (N_neurons, T) array of per-tick population activities.
    """
    rng = np.random.default_rng(seed)
    if init_A is None:
        init_A = np.linspace(0.5, 0.1, N_neurons)
    else:
        init_A = np.asarray(init_A, dtype=float)
        assert init_A.shape == (N_neurons,)

    m = np.zeros((N_neurons, qr.K_max))
    for i, A0 in enumerate(init_A):
        A0_safe = max(float(A0), 1e-3)
        for k in range(qr.K_max):
            m[i, k] = A0_safe * (1 - A0_safe) ** k
        m[i] /= m[i].sum()
    A_prev = init_A.copy()
    rates = np.zeros((N_neurons, T))

    # Per-neuron external drive (drive_bump on neuron 0 only).
    drives_ext = np.full(N_neurons, float(drive))
    drives_ext[0] += float(drive_bump)

    for t in range(T):
        # Sum of all others: w * (sum(A_prev) - A_prev[i]).
        sum_A = A_prev.sum()
        sum_A_var = (A_prev * (1.0 - A_prev)).sum()
        mean_recur = w * (sum_A - A_prev)
        var_recur = (w ** 2) * (sum_A_var - A_prev * (1.0 - A_prev))
        mean_ext = drives_ext * p_thin
        var_ext = (drives_ext ** 2) * p_thin * (1.0 - p_thin)
        mu = alpha * (mean_ext + mean_recur)
        sigma_sq = beta * (var_ext + var_recur)
        sigma = np.sqrt(np.clip(sigma_sq, 0.0, None))
        m, A = qr.step(m, mu, sigma, N_pop, rng=rng)
        rates[:, t] = A
        A_prev = A

    return rates
