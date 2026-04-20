"""Bifurcation-curve routines for the Wilson-Cowan archetype analyses.

Phase 1 uses:
  - ``pitchfork_diagonal``: closed form of the diagonal (symmetric-weight)
    pitchfork for contralateral inhibition.
  - ``pitchfork_curve_contralateral``: fine-grained numerical trace of the
    locus where ``max Re(lambda) = 0`` for the asymmetric 2D case.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import brentq

from linearization import jacobian, spectrum
from wilson_cowan import Sigmoid, find_fixed_point, find_saddle_contralateral


def _dominant_real_at(
    w12: float,
    w21: float,
    drive: float,
    tau: float,
    sigmoid: Sigmoid,
    rho_guess=None,
) -> float:
    del rho_guess  # unused; the saddle is selected by middle-branch root
    W = np.array([[0.0, -w21], [-w12, 0.0]])
    I = np.array([drive, drive])
    rho_star, n_fp = find_saddle_contralateral(w12, w21, drive, sigmoid)
    if n_fp == 0 or np.any(np.isnan(rho_star)):
        return float("nan")
    J = jacobian(W, I, rho_star, tau, sigmoid)
    eigvals, _ = spectrum(J)
    return float(eigvals[0].real)


def pitchfork_diagonal(
    drive: float, tau: float, sigmoid: Sigmoid
) -> Tuple[float, float, float]:
    """Find the diagonal pitchfork weight w* (where w12 = w21 = w*).

    Returns (w_star, rho_star_at_w_star, gain_g_at_w_star) such that
    w_star * g = 1 with g = f'(I - w_star * rho*).
    """
    f = lambda w: _dominant_real_at(w, w, drive, tau, sigmoid)
    w_lo, w_hi = 0.01, 10.0
    if f(w_lo) >= 0 or f(w_hi) <= 0:
        # Expand until a sign change is bracketed.
        for w_hi_try in (20.0, 50.0, 200.0):
            if f(w_hi_try) > 0:
                w_hi = w_hi_try
                break
    w_star = brentq(f, w_lo, w_hi, xtol=1e-10)

    W = np.array([[0.0, -w_star], [-w_star, 0.0]])
    I = np.array([drive, drive])
    rho_star, _ = find_fixed_point(W, I, sigmoid, rho_guess=np.array([0.5, 0.5]))
    g = float(sigmoid.f_prime(drive - w_star * rho_star[0]))
    return w_star, float(rho_star[0]), g


def pitchfork_curve_contralateral(
    drive: float,
    tau: float,
    sigmoid: Sigmoid,
    w_max: float = 5.0,
    n_angles: int = 181,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fine-grained numerical bifurcation curve in the (w12, w21) plane.

    The symmetric fixed point of the asymmetric contralateral system loses
    stability along a curve in weight space. We parameterize candidate
    radial slices from the origin by angle theta and find the radius at
    which the dominant real part crosses zero by bisection.

    Returns two equal-length arrays (w12s, w21s) describing the curve.
    Angles that do not yield a sign change within [0, w_max * sqrt(2)] are
    omitted.
    """
    w12s = []
    w21s = []
    # restrict to the open quadrant - both weights must be > 0
    angles = np.linspace(np.deg2rad(1.0), np.deg2rad(89.0), n_angles)
    r_max = np.sqrt(2.0) * w_max

    def f_along(r, theta):
        w12 = r * np.cos(theta)
        w21 = r * np.sin(theta)
        return _dominant_real_at(w12, w21, drive, tau, sigmoid)

    for theta in angles:
        val_lo = f_along(0.01, theta)
        val_hi = f_along(r_max, theta)
        if np.isnan(val_lo) or np.isnan(val_hi):
            continue
        if val_lo * val_hi >= 0:
            continue  # no sign change within the disk
        try:
            r_star = brentq(lambda r: f_along(r, theta), 0.01, r_max, xtol=1e-8)
        except Exception:
            continue
        w12 = float(r_star * np.cos(theta))
        w21 = float(r_star * np.sin(theta))
        if w12 <= w_max and w21 <= w_max:
            w12s.append(w12)
            w21s.append(w21)

    return np.array(w12s), np.array(w21s)
