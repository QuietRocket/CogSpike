r"""Siegert formula for stationary LI&F firing rate under diffusion approximation.

Standard form (Brunel 2000):

    nu^{-1} = tau_ref + tau_m * sqrt(pi) * \int_{(V_r - mu)/sigma}^{(V_th - mu)/sigma}
              exp(u^2) (1 + erf(u)) du

where mu, sigma are the mean and standard deviation of the (white-noise)
input that drives the membrane. The integrand exp(u^2) (1 + erf(u)) is
identically equal to erfcx(-u) (with erfcx(x) = exp(x^2) erfc(x), the scaled
complementary error function, available in scipy.special); we use the
erfcx form which is numerically stable for both u > 0 and u < 0.

For sigma -> 0 the formula degenerates to the deterministic LI&F rate
1 / (tau_ref + tau_m * log((mu - V_r) / (mu - V_th))) when mu > V_th, and
zero otherwise. We special-case sigma below a small floor to avoid the
quadrature blowing up.

Parameters (V_th, V_r, tau_m, tau_ref) are normalized: V_r = 0, V_th = 1
unless overridden. The mapping from FCS-LI&F (drive, thinning) to
(mu, sigma) is calibrated externally (see `phase1_siegert.py`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.special import erfcx


@dataclass
class Siegert:
    V_th: float = 1.0
    V_r: float = 0.0
    tau_m: float = 1.0
    tau_ref: float = 0.0
    sigma_floor: float = 1e-3

    def phi(self, mu: float, sigma: float) -> float:
        """Stationary firing rate for given input (mu, sigma)."""
        if sigma <= self.sigma_floor:
            # Deterministic LI&F limit.
            if mu <= self.V_th:
                return 0.0
            denom = self.V_r if self.V_r > -1e9 else 0.0
            try:
                period = self.tau_ref + self.tau_m * math.log(
                    (mu - denom) / (mu - self.V_th)
                )
                return 1.0 / period if period > 0 else 1.0
            except (ValueError, ZeroDivisionError):
                return 0.0
        y_th = (self.V_th - mu) / sigma
        y_r = (self.V_r - mu) / sigma
        # Quadrature of erfcx(-u) from y_r to y_th.
        # erfcx is well-conditioned everywhere; quad is reliable.
        try:
            integral, _err = quad(lambda u: erfcx(-u), y_r, y_th, limit=200)
        except Exception:
            return 0.0
        period = self.tau_ref + self.tau_m * math.sqrt(math.pi) * integral
        if period <= 0:
            return 1.0  # cap at unit rate per time unit
        return 1.0 / period

    def phi_array(self, mu, sigma):
        """Vectorized phi over arrays of mu, sigma (broadcast)."""
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        out_shape = np.broadcast_shapes(mu.shape, sigma.shape)
        mu_b = np.broadcast_to(mu, out_shape).ravel()
        sigma_b = np.broadcast_to(sigma, out_shape).ravel()
        out = np.array([self.phi(float(m), float(s)) for m, s in zip(mu_b, sigma_b)])
        return out.reshape(out_shape)


def fcs_to_siegert_input(
    weighted_input_mean: float,
    weighted_input_var: float,
    alpha: float,
    beta: float,
) -> Tuple[float, float]:
    """Map FCS-units (mean, variance) of input to normalized Siegert (mu, sigma).

    Args:
        weighted_input_mean: sum_k w_k * E[x_k] where x_k is each input
            (external bernoulli-thinned drive or recurrent population rate).
        weighted_input_var: sum_k w_k^2 * Var[x_k]. Variance is
            p_thin * (1 - p_thin) for external, nu * (1 - nu) for recurrent.
        alpha: linear scale on mean (calibrated).
        beta: linear scale on variance (calibrated).

    Returns:
        (mu, sigma) in Siegert's normalized units.
    """
    mu = alpha * weighted_input_mean
    sigma_sq = beta * weighted_input_var
    sigma = math.sqrt(max(sigma_sq, 0.0))
    return mu, sigma


def find_fixed_point_contralateral(
    w12: float,
    w21: float,
    drive: float,
    p_thin: float,
    siegert: Siegert,
    alpha: float,
    beta: float,
    nu_guess=(0.5, 0.5),
    tol: float = 1e-9,
) -> Tuple[np.ndarray, bool]:
    """Self-consistent Siegert FP for 2-population contralateral inhibition.

    The contralateral motif: each population i receives external drive
    `drive * p_thin` (mean per tick) plus inhibition w_ji * nu_j from the
    other population. Variance: external Bernoulli + recurrent Bernoulli.

    nu_i = Phi(mu_i, sigma_i)
    mu_i      = alpha * (drive * p_thin + w_ji * nu_j)
    sigma_i^2 = beta * (drive^2 * p_thin*(1-p_thin) + w_ji^2 * nu_j*(1-nu_j))

    Note: w_ji here is the weight from population j into population i.
    Convention matches contralateral(): row i column j of the W matrix.
    """

    def residual(nu):
        nu = np.clip(nu, 0.0, 1.0)
        mu_var = []
        for i in range(2):
            j = 1 - i
            w_ji = w21 if i == 0 else w12  # population j -> i
            mean_in = drive * p_thin + w_ji * nu[j]
            var_in = (drive ** 2) * p_thin * (1 - p_thin) + (w_ji ** 2) * nu[j] * (
                1 - nu[j]
            )
            mu_var.append((mean_in, var_in))
        mu = np.array([alpha * mv[0] for mv in mu_var])
        sigma = np.array([math.sqrt(max(beta * mv[1], 0.0)) for mv in mu_var])
        nu_pred = np.array([siegert.phi(mu[k], sigma[k]) for k in range(2)])
        return nu - nu_pred

    nu_star, _info, ier, _msg = fsolve(
        residual, np.asarray(nu_guess, dtype=float), full_output=True, xtol=tol
    )
    success = ier == 1 and np.max(np.abs(residual(nu_star))) < 1e-7
    return nu_star, bool(success)


def find_all_fixed_points_contralateral(
    w12: float,
    w21: float,
    drive: float,
    p_thin: float,
    siegert: Siegert,
    alpha: float,
    beta: float,
    n_samples: int = 161,
) -> list:
    """Enumerate all fixed points via 1D scalar reduction (mirrors WC pattern).

    For nu_2 = Phi(mu_2(nu_1), sigma_2(nu_1)), substitute back to get a
    scalar equation in nu_1. Find sign changes on a grid + brentq refine.
    """
    from scipy.optimize import brentq

    def nu_2_of_nu_1(r1: float) -> float:
        # Population 2 sees inhibition w_12 from population 1.
        mean_in = drive * p_thin + w12 * r1
        var_in = (drive ** 2) * p_thin * (1 - p_thin) + (w12 ** 2) * r1 * (1 - r1)
        mu = alpha * mean_in
        sigma = math.sqrt(max(beta * var_in, 0.0))
        return siegert.phi(mu, sigma)

    def g(r1: float) -> float:
        r2 = nu_2_of_nu_1(r1)
        # Population 1 sees inhibition w_21 from population 2.
        mean_in = drive * p_thin + w21 * r2
        var_in = (drive ** 2) * p_thin * (1 - p_thin) + (w21 ** 2) * r2 * (1 - r2)
        mu = alpha * mean_in
        sigma = math.sqrt(max(beta * var_in, 0.0))
        return r1 - siegert.phi(mu, sigma)

    xs = np.linspace(1e-4, 1 - 1e-4, n_samples)
    ys = np.array([g(x) for x in xs])
    roots = []
    for i in range(len(xs) - 1):
        a, b = ys[i], ys[i + 1]
        if a * b < 0:
            try:
                r = brentq(g, float(xs[i]), float(xs[i + 1]), xtol=1e-10)
                roots.append(float(r))
            except Exception:
                pass
        elif a == 0:
            roots.append(float(xs[i]))
    # Deduplicate.
    roots = sorted(set(round(r, 8) for r in roots))
    fps = []
    for r1 in roots:
        r2 = nu_2_of_nu_1(r1)
        fps.append(np.array([r1, r2]))
    return fps
