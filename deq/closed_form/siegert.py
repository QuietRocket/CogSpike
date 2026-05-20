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


def _phi_from_means(siegert, alpha, beta, mean_in, var_in):
    """Helper: map (FCS-units mean_in, var_in) -> Phi via Siegert."""
    mu = alpha * mean_in
    sigma = math.sqrt(max(beta * var_in, 0.0))
    return siegert.phi(mu, sigma)


def find_all_fixed_points_uniform_inhibition(
    N: int,
    w: float,
    drive: float,
    p_thin: float,
    siegert: Siegert,
    alpha: float,
    beta: float,
    n_samples: int = 81,
    dedup_tol: float = 1e-4,
) -> list:
    """Enumerate symmetric-orbit fixed points for N-neuron uniform inhibition.

    Topology: every neuron self-drives with `drive` and inhibits every other
    with weight `w`. Under S_N permutation symmetry of W, every fixed point
    lies in an orbit of the form

        nu = (nu_W repeated k times,  nu_L repeated N-k times)

    for some k in {0, ..., N}, up to permutation. For each k we solve the
    closed 2D system

        nu_W = Phi(mu_W, sigma_W)
            mu_W      = alpha * (drive*p_thin + w*((k-1)*nu_W + (N-k)*nu_L))
            sigma_W^2 = beta  * (drive^2*p_thin*(1-p_thin)
                                  + w^2*((k-1)*nu_W*(1-nu_W) + (N-k)*nu_L*(1-nu_L)))
        nu_L = Phi(mu_L, sigma_L)
            mu_L      = alpha * (drive*p_thin + w*(k*nu_W + (N-k-1)*nu_L))
            sigma_L^2 = beta  * (drive^2*p_thin*(1-p_thin)
                                  + w^2*(k*nu_W*(1-nu_W) + (N-k-1)*nu_L*(1-nu_L)))

    Method (mirrors the 2-neuron 1D-reduction pattern, nested one level):
    for each k in {1, ..., N-1} (genuine WTA shapes), scan nu_L on a grid;
    at each nu_L solve the 1D residual r_W(nu_W) := nu_W - Phi(mu_W) = 0 by
    brentq sign-change, taking the LARGEST root (consistent winner branch).
    Substitute back to form residual r_L(nu_L) = nu_L - Phi(mu_L(nu_W*(nu_L), nu_L))
    and brentq again. For k = 0 and k = N (fully symmetric) the 2D system
    collapses to a 1D fixed point in either nu_L or nu_W.

    For N = 2 and k = 1 this reduces exactly to find_all_fixed_points_contralateral
    with w12 = w21 = w (up to (W, L) <-> (1, 2) relabeling). Unit-checked.

    Returns:
        list of dicts {'k': int, 'nu_W': float, 'nu_L': float, 'spread': float,
                       'orbit_size': int} where orbit_size = C(N, k). Sorted by
        descending spread. Deduplicated across k using dedup_tol on (nu_W, nu_L).
    """
    from math import comb
    from scipy.optimize import brentq

    def fp_eq_2d(k_winners: int):
        """Return list of (nu_W, nu_L) roots for orbit shape k_winners."""
        k = k_winners
        Nm = N - k

        def phi_W(nu_W, nu_L):
            mean_in = drive * p_thin + w * ((k - 1) * nu_W + Nm * nu_L)
            var_in = ((drive ** 2) * p_thin * (1 - p_thin)
                      + (w ** 2) * ((k - 1) * nu_W * (1 - nu_W)
                                     + Nm * nu_L * (1 - nu_L)))
            return _phi_from_means(siegert, alpha, beta, mean_in, var_in)

        def phi_L(nu_W, nu_L):
            mean_in = drive * p_thin + w * (k * nu_W + (Nm - 1) * nu_L)
            var_in = ((drive ** 2) * p_thin * (1 - p_thin)
                      + (w ** 2) * (k * nu_W * (1 - nu_W)
                                     + (Nm - 1) * nu_L * (1 - nu_L)))
            return _phi_from_means(siegert, alpha, beta, mean_in, var_in)

        def nu_W_branches(nu_L):
            """All nu_W solutions of nu_W = Phi_W(nu_W, nu_L) at fixed nu_L."""
            xs = np.linspace(1e-4, 1 - 1e-4, n_samples)
            ys = np.array([x - phi_W(x, nu_L) for x in xs])
            branches = []
            for i in range(len(xs) - 1):
                a, b = ys[i], ys[i + 1]
                if a == 0:
                    branches.append(float(xs[i]))
                elif a * b < 0:
                    try:
                        r = brentq(
                            lambda x: x - phi_W(x, nu_L),
                            float(xs[i]), float(xs[i + 1]), xtol=1e-10,
                        )
                        branches.append(float(r))
                    except Exception:
                        pass
            return branches

        def winner_branch(nu_L):
            """The consistent 'winner' branch: largest nu_W root."""
            br = nu_W_branches(nu_L)
            return max(br) if br else None

        def loser_branch(nu_L):
            """The consistent 'loser' branch: smallest nu_W root."""
            br = nu_W_branches(nu_L)
            return min(br) if br else None

        roots_kL = []
        # We need to find nu_L such that nu_L = Phi_L(nu_W_winner(nu_L), nu_L)
        xs_L = np.linspace(1e-4, 1 - 1e-4, n_samples)
        # Scan winner branch.
        for branch_picker in (winner_branch, loser_branch):
            ys_L = []
            valid_xs = []
            for x in xs_L:
                nu_W_star = branch_picker(x)
                if nu_W_star is None:
                    ys_L.append(np.nan)
                else:
                    ys_L.append(x - phi_L(nu_W_star, x))
                valid_xs.append(x)
            ys_L = np.array(ys_L)
            for i in range(len(xs_L) - 1):
                a, b = ys_L[i], ys_L[i + 1]
                if np.isnan(a) or np.isnan(b):
                    continue
                if a == 0:
                    nu_L_star = float(xs_L[i])
                    nu_W_star = branch_picker(nu_L_star)
                    if nu_W_star is not None:
                        roots_kL.append((nu_W_star, nu_L_star))
                elif a * b < 0:
                    try:
                        def residual(x):
                            nu_W_star = branch_picker(x)
                            if nu_W_star is None:
                                return float('nan')
                            return x - phi_L(nu_W_star, x)
                        r = brentq(residual, float(xs_L[i]), float(xs_L[i + 1]),
                                   xtol=1e-10)
                        nu_W_star = branch_picker(r)
                        if nu_W_star is not None:
                            roots_kL.append((nu_W_star, float(r)))
                    except Exception:
                        pass
        return roots_kL

    def fp_eq_1d_symmetric():
        """Fully symmetric FPs: all neurons fire at the same rate nu*.

        nu* = Phi(alpha*(drive*p_thin + w*(N-1)*nu*),
                  sqrt(beta*(drive^2*p_thin*(1-p_thin)
                              + w^2*(N-1)*nu**(1-nu*))))
        """
        def residual(x):
            mean_in = drive * p_thin + w * (N - 1) * x
            var_in = ((drive ** 2) * p_thin * (1 - p_thin)
                      + (w ** 2) * (N - 1) * x * (1 - x))
            return x - _phi_from_means(siegert, alpha, beta, mean_in, var_in)
        xs = np.linspace(1e-4, 1 - 1e-4, n_samples)
        ys = np.array([residual(x) for x in xs])
        roots = []
        for i in range(len(xs) - 1):
            a, b = ys[i], ys[i + 1]
            if a == 0:
                roots.append(float(xs[i]))
            elif a * b < 0:
                try:
                    r = brentq(residual, float(xs[i]), float(xs[i + 1]), xtol=1e-10)
                    roots.append(float(r))
                except Exception:
                    pass
        return roots

    all_fps = []
    # Fully symmetric: treat as k = N (or equivalently k = 0); we tag k = N.
    for nu_s in fp_eq_1d_symmetric():
        all_fps.append({
            'k': N, 'nu_W': nu_s, 'nu_L': nu_s,
            'spread': 0.0, 'orbit_size': 1,
        })
    # All k in {1, ..., N-1}.
    for k in range(1, N):
        for nu_W_star, nu_L_star in fp_eq_2d(k):
            # Filter: must satisfy nu_W >= nu_L (orbit-label convention).
            if nu_W_star < nu_L_star - 1e-6:
                continue
            all_fps.append({
                'k': k, 'nu_W': float(nu_W_star), 'nu_L': float(nu_L_star),
                'spread': float(nu_W_star - nu_L_star),
                'orbit_size': comb(N, k),
            })

    # Deduplicate: distinct (round(nu_W), round(nu_L)) tuples.
    seen = []
    deduped = []
    for fp in sorted(all_fps, key=lambda d: -d['spread']):
        key = (round(fp['nu_W'] / dedup_tol), round(fp['nu_L'] / dedup_tol),
               fp['k'])
        # Allow same (nu_W, nu_L) under different k (different orbits).
        if key in seen:
            continue
        seen.append(key)
        deduped.append(fp)
    return deduped


def find_fp_uniform_inhibition_bumped(
    N: int,
    w: float,
    drive: float,
    drive_bump: float,
    p_thin: float,
    siegert: Siegert,
    alpha: float,
    beta: float,
    n_samples: int = 81,
) -> list:
    """FP enumeration for N-neuron uniform inhibition with a `drive_bump` on
    neuron 0 (S_N -> S_(N-1) broken symmetry).

    Neuron 0 receives drive (drive + drive_bump) * p_thin; neurons 1..N-1 each
    receive drive * p_thin. We enumerate three orbit classes (the WTA-relevant
    ones; the spontaneously-broken k in {2..N-1} orbits are rare for small
    drive_bump and not WTA shapes anyway):

      - 'bumped_winner' (k=1 with bumped neuron as the winner): 2D FP in
        (nu_0, nu_L) where neurons 1..N-1 all tie at nu_L. Same nested-brentq
        machinery as the symmetric case.
      - 'all_equal_others': 1D FP in nu_L assuming nu_0 = nu_L (i.e. the
        bumped neuron does not deviate from the symmetric all-fire FP, which
        can happen at weak |w|).
      - 'all_fire_symmetric': 1D FP ignoring the bump (returns the symmetric
        FP that would exist at drive_bump=0). Diagnostic only.

    Returns list of dicts {'shape': str, 'nu_0': float, 'nu_L': float,
    'spread': float}, sorted by descending spread.
    """
    from scipy.optimize import brentq

    D0 = drive + drive_bump
    D = drive

    def phi_in(mu_in, var_in):
        return _phi_from_means(siegert, alpha, beta, mu_in, var_in)

    # ---- Orbit: bumped winner, others tie (k=1 with bumped as winner) ----
    def phi_0(nu_0, nu_L):
        mean_in = D0 * p_thin + w * (N - 1) * nu_L
        var_in = (D0 ** 2) * p_thin * (1 - p_thin) + (w ** 2) * (N - 1) * nu_L * (1 - nu_L)
        return phi_in(mean_in, var_in)

    def phi_L(nu_0, nu_L):
        mean_in = D * p_thin + w * (nu_0 + (N - 2) * nu_L)
        var_in = ((D ** 2) * p_thin * (1 - p_thin)
                  + (w ** 2) * (nu_0 * (1 - nu_0) + (N - 2) * nu_L * (1 - nu_L)))
        return phi_in(mean_in, var_in)

    def nu_0_branches(nu_L):
        """All nu_0 solutions of nu_0 = phi_0(nu_0, nu_L) at fixed nu_L.

        phi_0 doesn't depend on nu_0 (the bumped neuron's input is fully
        determined by nu_L), so this is a *direct* evaluation, not a
        root-find. Returns [phi_0(0, nu_L)] (or empty if numerically bad).
        """
        try:
            return [phi_0(0.0, nu_L)]
        except Exception:
            return []

    bumped_winner_fps = []
    xs_L = np.linspace(1e-4, 1 - 1e-4, n_samples)

    def residual_L(nu_L):
        branches = nu_0_branches(nu_L)
        if not branches:
            return float('nan')
        nu_0 = branches[0]
        return nu_L - phi_L(nu_0, nu_L)

    ys_L = np.array([residual_L(x) for x in xs_L])
    for i in range(len(xs_L) - 1):
        a, b = ys_L[i], ys_L[i + 1]
        if np.isnan(a) or np.isnan(b):
            continue
        if a == 0:
            nu_L_star = float(xs_L[i])
            nu_0_star = nu_0_branches(nu_L_star)[0]
            bumped_winner_fps.append((nu_0_star, nu_L_star))
        elif a * b < 0:
            try:
                r = brentq(residual_L, float(xs_L[i]), float(xs_L[i + 1]),
                           xtol=1e-10)
                nu_0_star = nu_0_branches(r)[0]
                bumped_winner_fps.append((nu_0_star, float(r)))
            except Exception:
                pass

    # ---- 1D symmetric (all equal, including bumped neuron) ----
    def residual_sym(x):
        # All N neurons fire at x, even neuron 0 (ignoring drive_bump).
        # Actually for the truly symmetric FP under broken symmetry, only
        # the others (N-1) tie; we just compute the analog of the symmetric
        # FP at drive_bump=0 here for diagnostic.
        mean_in = D * p_thin + w * (N - 1) * x
        var_in = (D ** 2) * p_thin * (1 - p_thin) + (w ** 2) * (N - 1) * x * (1 - x)
        return x - phi_in(mean_in, var_in)
    ys_sym = np.array([residual_sym(x) for x in xs_L])
    sym_fps = []
    for i in range(len(xs_L) - 1):
        a, b = ys_sym[i], ys_sym[i + 1]
        if a == 0:
            sym_fps.append(float(xs_L[i]))
        elif a * b < 0:
            try:
                r = brentq(residual_sym, float(xs_L[i]), float(xs_L[i + 1]),
                           xtol=1e-10)
                sym_fps.append(float(r))
            except Exception:
                pass

    fps = []
    for nu_0_star, nu_L_star in bumped_winner_fps:
        fps.append({
            'shape': 'bumped_winner',
            'nu_0': float(nu_0_star),
            'nu_L': float(nu_L_star),
            'spread': float(nu_0_star - nu_L_star),
        })
    for nu_s in sym_fps:
        fps.append({
            'shape': 'symmetric_all_fire',
            'nu_0': float(nu_s),
            'nu_L': float(nu_s),
            'spread': 0.0,
        })
    fps.sort(key=lambda d: -d['spread'])
    return fps
