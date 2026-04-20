"""Pole placement / inverse-design for the negative-loop Hopf locus.

Given a target oscillation frequency omega* and fixed scalar parameters
(w_xa, w_aa, w_ii, tau) and the sigmoid, solve for (w_ai, w_ia) such
that the WC Jacobian at the fixed point has a pure-imaginary pair
+/- i omega*. The Hopf conditions are

    tr J = w_aa g_A - 2 = 0
    det J = 1 - w_aa g_A + w_ai w_ia g_A g_I = w_ai w_ia g_A g_I - 1
    (omega*)^2 = det J

so at the Hopf locus g_A = 2 / w_aa, which pins r_A to one of two
branches (lower/upper) via r_A (1 - r_A) = g_A / k.

Given the branch, the activator-FP constraint fixes the product
w_ia r_I; the inhibitor-FP constraint gives w_ai = f^{-1}(r_I) / r_A;
and the frequency condition gives one scalar equation in r_I. We solve
by scipy.optimize.brentq over each monotone sub-branch and pick the
minimum-magnitude feasible solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

from wilson_cowan import Sigmoid


def _finv(y: float, sigmoid: Sigmoid) -> float:
    return sigmoid.theta - (1.0 / sigmoid.k) * np.log(1.0 / y - 1.0)


@dataclass
class HopfDesign:
    """An inverse-design candidate on the Hopf locus."""

    omega_target: float
    w_ai: float
    w_ia: float
    w_xa: float
    w_aa: float
    r_A: float
    r_I: float
    branch: str  # 'lower' or 'upper' r_A branch
    residual: float  # verification residual of the omega^2 equation


def _branch_r_A(w_aa: float, sigmoid: Sigmoid) -> List[Tuple[float, str]]:
    """Return the (r_A, label) pairs on the Hopf locus for given w_aa.

    Empty list if tr J = 0 is unreachable within the sigmoid's slope
    budget (g_A_max = k / 4).
    """
    target_gA = 2.0 / w_aa
    if target_gA > sigmoid.k / 4.0:
        return []
    disc = 1.0 - 4.0 * target_gA / sigmoid.k
    if disc < 0:
        return []
    s = 0.5 * float(np.sqrt(disc))
    out = []
    lo, hi = 0.5 - s, 0.5 + s
    if 0.0 < lo < 1.0:
        out.append((lo, "lower"))
    if 0.0 < hi < 1.0 and hi != lo:
        out.append((hi, "upper"))
    return out


def design_negative_loop(
    omega_target: float,
    sigmoid: Sigmoid,
    w_xa: float,
    w_aa: float,
    w_max: float = 5.0,
    tau: float = 1.0,
    n_samples: int = 4001,
    prefer_branch: Optional[str] = "lower",
) -> Optional[HopfDesign]:
    """Solve for (w_ai, w_ia) realizing omega* on the Hopf locus.

    When both r_A branches offer feasible solutions, prefer_branch picks
    between them; otherwise the unique branch is used. The lower branch
    is preferred by default because its Hopf eigenvalue pair drifts
    slower in imaginary part when the FP is perturbed away from the
    locus, giving cleaner simulation behavior; the upper branch at low
    omega* sits near a codim-2 neighborhood where the linear frequency
    loses meaning rapidly off-Hopf. Within a branch we pick the
    minimum-norm (w_ai, w_ia) pair.

    Returns None if no feasible solution exists in (0, w_max]^2.
    """
    del tau  # omega* is in units where tau = 1; tau enters only through simulation.
    target_gA = 2.0 / w_aa
    target_omega_sq = omega_target**2

    def residual_in_rI(r_A: float, r_I: float) -> float:
        x_A = _finv(r_A, sigmoid)
        x_I = _finv(r_I, sigmoid)
        B = w_xa + w_aa * r_A - x_A  # = w_ia r_I
        if B <= 0 or r_A <= 0 or r_I <= 0 or r_I >= 1:
            return float("nan")
        w_ai = x_I / r_A
        w_ia = B / r_I
        if w_ai <= 0 or w_ia <= 0:
            return float("nan")
        g_I = float(sigmoid.f_prime(x_I))
        det = w_ai * w_ia * target_gA * g_I - 1.0
        return det - target_omega_sq

    candidates: List[HopfDesign] = []
    for r_A, label in _branch_r_A(w_aa, sigmoid):
        x_A = _finv(r_A, sigmoid)
        B = w_xa + w_aa * r_A - x_A
        if B <= 0:
            continue

        probe = np.linspace(1e-4, 1 - 1e-4, n_samples)
        vals = np.array([residual_in_rI(r_A, float(rI)) for rI in probe])
        finite = np.isfinite(vals)
        for i in range(len(probe) - 1):
            if not (finite[i] and finite[i + 1]):
                continue
            if vals[i] == 0.0:
                rI_star = float(probe[i])
            elif vals[i] * vals[i + 1] >= 0:
                continue
            else:
                try:
                    rI_star = brentq(
                        lambda rI: residual_in_rI(r_A, float(rI)),
                        float(probe[i]),
                        float(probe[i + 1]),
                        xtol=1e-12,
                    )
                except Exception:
                    continue
            x_I = _finv(rI_star, sigmoid)
            w_ai = x_I / r_A
            w_ia = B / rI_star
            if not (0 < w_ai <= w_max and 0 < w_ia <= w_max):
                continue
            g_I = float(sigmoid.f_prime(x_I))
            omega_sq = w_ai * w_ia * target_gA * g_I - 1.0
            if omega_sq <= 0:
                continue
            residual = float(abs(omega_sq - target_omega_sq))
            candidates.append(
                HopfDesign(
                    omega_target=omega_target,
                    w_ai=float(w_ai),
                    w_ia=float(w_ia),
                    w_xa=w_xa,
                    w_aa=w_aa,
                    r_A=float(r_A),
                    r_I=float(rI_star),
                    branch=label,
                    residual=residual,
                )
            )

    if not candidates:
        return None
    if prefer_branch is not None:
        preferred = [c for c in candidates if c.branch == prefer_branch]
        if preferred:
            preferred.sort(key=lambda d: d.w_ai**2 + d.w_ia**2)
            return preferred[0]
    candidates.sort(key=lambda d: d.w_ai**2 + d.w_ia**2)
    return candidates[0]


def achievable_omega_range(
    sigmoid: Sigmoid,
    w_xa: float,
    w_aa: float,
    w_max: float = 5.0,
    n_samples: int = 2001,
) -> Tuple[float, float]:
    """Return (omega_min, omega_max) attainable at the Hopf locus inside
    (w_ai, w_ia) in (0, w_max]^2.
    """
    omegas: List[float] = []
    for r_A, _label in _branch_r_A(w_aa, sigmoid):
        x_A = _finv(r_A, sigmoid)
        B = w_xa + w_aa * r_A - x_A
        if B <= 0:
            continue
        target_gA = 2.0 / w_aa
        for r_I in np.linspace(1e-3, 1 - 1e-3, n_samples):
            x_I = _finv(r_I, sigmoid)
            w_ai = x_I / r_A
            w_ia = B / r_I
            if not (0 < w_ai <= w_max and 0 < w_ia <= w_max):
                continue
            g_I = float(sigmoid.f_prime(x_I))
            det = w_ai * w_ia * target_gA * g_I - 1.0
            if det <= 0:
                continue
            omegas.append(float(np.sqrt(det)))
    if not omegas:
        return float("nan"), float("nan")
    return float(min(omegas)), float(max(omegas))
