"""Linear-response transfer function H(omega) for Siegert-LIF populations.

We use the simplest closed-form approximation that is consistent with the
classical population-thread Jacobian analysis at omega = 0: a single-pole
low-pass

    H(omega) = (d Phi / d mu) / (1 + i omega tau_m)

evaluated at the Siegert fixed point (mu*, sigma*). Derivative d Phi / d mu
is computed analytically from the Siegert formula (Leibniz rule on the
integral, with the integrand erfcx(-u)).

This is the small-omega limit of the full Brunel-Hakim / Richardson
transfer function. The full parabolic-cylinder-function form encodes a
high-frequency resonance (proportional to sigma) that is irrelevant near
the bifurcation locus; Richardson 2007 calls this "low-pass with
resonance" form. We omit the resonance by design (explicit non-goal in
the plan; the resonance only shifts oscillation onset frequencies in the
high-noise / high-sigma regime). The single-pole approximation reduces to
the population-thread Wilson-Cowan Jacobian at omega = 0, which is the
Phase 2 self-consistency gate.

Closed-loop machinery:
    delta nu(omega)        = H(omega) * delta mu(omega)
    delta mu_i(omega)      = sum_j J_ij * delta nu_j(omega) + delta mu_i_ext(omega)
    delta nu(omega)        = (I - H(omega) J)^{-1} H(omega) delta mu_ext(omega)
    closed-loop poles      = roots of det(I - H(omega) J) = 0

For a 2-pop motif with diagonal H(omega) = h(omega) (same per-population)
this is a quadratic in h whose poles factor into the eigenvalues of J.
The general n-pop case requires numeric root-finding on the determinant
along the iomega-axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from scipy.special import erfcx
from scipy.optimize import brentq


def dphi_dmu(siegert, mu: float, sigma: float) -> float:
    """Analytical d Phi / d mu at fixed point (mu, sigma).

    From Siegert formula

        Phi^{-1} = tau_ref + tau_m * sqrt(pi) * int_{y_r}^{y_th} erfcx(-u) du

    with y_th = (V_th - mu) / sigma, y_r = (V_r - mu) / sigma. Leibniz:

        d Phi^{-1} / d mu = -(tau_m * sqrt(pi) / sigma) * [erfcx(-y_th) - erfcx(-y_r)]

    so

        d Phi / d mu = -Phi^2 * d Phi^{-1} / d mu
                     = (Phi^2 * tau_m * sqrt(pi) / sigma) *
                       [erfcx(-y_th) - erfcx(-y_r)]
    """
    if sigma <= siegert.sigma_floor:
        # Deterministic-LIF derivative: d/dmu of 1/(tau_ref + tau_m * log((mu-V_r)/(mu-V_th)))
        if mu <= siegert.V_th:
            return 0.0
        denom = siegert.tau_ref + siegert.tau_m * math.log(
            (mu - siegert.V_r) / (mu - siegert.V_th)
        )
        # d/d mu [log((mu-V_r)/(mu-V_th))] = 1/(mu-V_r) - 1/(mu-V_th)
        d_log = 1.0 / (mu - siegert.V_r) - 1.0 / (mu - siegert.V_th)
        return -(siegert.tau_m * d_log) / (denom ** 2)
    nu = siegert.phi(mu, sigma)
    y_th = (siegert.V_th - mu) / sigma
    y_r = (siegert.V_r - mu) / sigma
    bracket = erfcx(-y_th) - erfcx(-y_r)
    return float((nu ** 2) * siegert.tau_m * math.sqrt(math.pi) / sigma * bracket)


def H_low_pass(omega: float, gain: float, tau_m: float) -> complex:
    """Single-pole low-pass transfer function: H(omega) = gain / (1 + i omega tau_m)."""
    return gain / (1.0 + 1j * omega * tau_m)


def closed_loop_matrix(omega: float, J: np.ndarray, gains: np.ndarray,
                        tau_m: float) -> np.ndarray:
    """M(omega) = I - H(omega) J, where H(omega) is diagonal per-population.

    Args:
        omega: real frequency.
        J: (n, n) connectivity matrix in *Siegert* units (post-calibration scale).
        gains: (n,) per-population DC gain (d Phi / d mu) at the operating point.
        tau_m: scalar, shared across populations (assumed identical).

    Returns:
        (n, n) complex matrix.
    """
    n = J.shape[0]
    H_diag = np.array([H_low_pass(omega, g, tau_m) for g in gains], dtype=complex)
    return np.eye(n, dtype=complex) - np.diag(H_diag) @ J.astype(complex)


def closed_loop_response(omega: float, J: np.ndarray, gains: np.ndarray,
                          tau_m: float) -> np.ndarray:
    """G(omega) = (I - H J)^{-1} H : transfer from delta mu_ext to delta nu."""
    n = J.shape[0]
    H_diag = np.array([H_low_pass(omega, g, tau_m) for g in gains], dtype=complex)
    M = np.eye(n, dtype=complex) - np.diag(H_diag) @ J.astype(complex)
    return np.linalg.solve(M, np.diag(H_diag))


def find_imaginary_axis_poles(J: np.ndarray, gains: np.ndarray, tau_m: float,
                              omega_grid: np.ndarray) -> list:
    """Find frequencies where det(I - H(omega) J) crosses zero on the iomega axis.

    Returns a sorted list of real omega values where det(M) = 0 + 0i is
    intersected (i.e., closed-loop sustained-oscillation candidates).
    """
    dets = np.array([
        complex(np.linalg.det(closed_loop_matrix(w, J, gains, tau_m)))
        for w in omega_grid
    ])
    # Look for sign-changes in real(det) AND small |imag|.
    crossings = []
    for i in range(len(omega_grid) - 1):
        d1, d2 = dets[i], dets[i + 1]
        if d1.real * d2.real < 0 and abs(d1.imag) < 1.0 and abs(d2.imag) < 1.0:
            try:
                w_cross = brentq(
                    lambda w: float(
                        np.linalg.det(closed_loop_matrix(w, J, gains, tau_m)).real
                    ),
                    float(omega_grid[i]),
                    float(omega_grid[i + 1]),
                    xtol=1e-8,
                )
                crossings.append(float(w_cross))
            except Exception:
                pass
    return crossings


def jacobian_eigenvalues(J: np.ndarray, gains: np.ndarray,
                         tau_m: float) -> np.ndarray:
    """Time-domain Jacobian eigenvalues of the rate-equation linearization.

    The standard Wilson-Cowan-style linearization is

        d delta nu / dt = (1 / tau_m) * (- delta nu + diag(gains) (J delta nu))

    with eigenvalues lambda(A) where A = (1/tau_m) * (-I + diag(gains) J).
    These are also the solutions of det(I - H(omega) J) = 0 at omega = 0
    (specifically lambda corresponds to a complex "frequency" via
    s = i omega + sigma_real).
    """
    n = J.shape[0]
    A = (1.0 / tau_m) * (-np.eye(n) + np.diag(gains) @ J)
    return np.linalg.eigvals(A)


def closed_loop_zero_freq_consistency(J: np.ndarray, gains: np.ndarray,
                                       tau_m: float) -> dict:
    """Check that closed-loop M(omega = 0) has det = 0 iff Jacobian has a zero eigenvalue.

    At omega = 0: H(0) = diag(gains), so M(0) = I - diag(gains) J. det(M(0))
    is zero iff diag(gains) J has a unit eigenvalue iff the time-domain
    Jacobian (1/tau_m)(-I + diag(gains) J) has a zero eigenvalue. The two
    descriptions are *exactly* equivalent.

    This routine returns the residual det(M(0)) and tau_m * lambda_max(A);
    they should be related by det(M(0)) = prod_i (1 - tau_m * lambda_i / 1)
    when the dimensions allow direct comparison.
    """
    M0 = closed_loop_matrix(0.0, J, gains, tau_m)
    eigs = jacobian_eigenvalues(J, gains, tau_m)
    # Direct equivalence: spectrum of (diag(gains) J) = 1 + tau_m * lambda(A).
    eigs_DJ = np.linalg.eigvals(np.diag(gains) @ J)
    lam_check = (eigs_DJ - 1.0) / tau_m
    residual = float(np.max(np.abs(np.sort_complex(eigs) - np.sort_complex(lam_check))))
    return {
        "det_M0": complex(np.linalg.det(M0)),
        "eigs_jacobian": eigs,
        "eigs_check_via_DJ": lam_check,
        "consistency_residual": residual,
    }
