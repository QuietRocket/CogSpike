"""Wilson-Cowan rate-equation primitives.

Implements the population-level reduction of a leaky integrate-and-fire
ensemble with distributed thresholds: a smooth sigmoidal gain replaces the
non-smooth spike-reset of the microscopic LIF dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


@dataclass
class Sigmoid:
    k: float = 4.0
    theta: float = 1.0

    def f(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return 1.0 / (1.0 + np.exp(-self.k * (x - self.theta)))

    def f_prime(self, x: np.ndarray) -> np.ndarray:
        fx = self.f(x)
        return self.k * fx * (1.0 - fx)


def wc_rhs(t, rho, W, I, tau, sigmoid: Sigmoid):
    rho = np.asarray(rho, dtype=float)
    return (-rho + sigmoid.f(W @ rho + I)) / tau


def simulate(
    W: np.ndarray,
    I: np.ndarray,
    tau: float,
    sigmoid: Sigmoid,
    t_span: Tuple[float, float],
    rho0: np.ndarray,
    max_step: float = 0.01,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    t_eval: np.ndarray | None = None,
):
    W = np.asarray(W, dtype=float)
    I = np.asarray(I, dtype=float)
    rho0 = np.asarray(rho0, dtype=float)

    sol = solve_ivp(
        fun=lambda t, rho: wc_rhs(t, rho, W, I, tau, sigmoid),
        t_span=t_span,
        y0=rho0,
        method="RK45",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        t_eval=t_eval,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol.t, sol.y


def find_fixed_point(
    W: np.ndarray,
    I: np.ndarray,
    sigmoid: Sigmoid,
    rho_guess: np.ndarray,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, bool]:
    W = np.asarray(W, dtype=float)
    I = np.asarray(I, dtype=float)
    rho_guess = np.asarray(rho_guess, dtype=float)

    def residual(rho):
        return rho - sigmoid.f(W @ rho + I)

    rho_star, infodict, ier, _msg = fsolve(
        residual, rho_guess, full_output=True, xtol=tol
    )
    success = ier == 1 and np.max(np.abs(residual(rho_star))) < 1e-8
    return rho_star, bool(success)


def find_all_fixed_points_contralateral(
    w12: float,
    w21: float,
    drive: float,
    sigmoid: Sigmoid,
    n_samples: int = 401,
) -> list[np.ndarray]:
    """Enumerate all fixed points of the 2D contralateral archetype.

    Uses the scalar reduction rho_1 = f(I - w_21 f(I - w_12 rho_1)), which
    is equivalent to the 2D fixed-point system; every 1D root pairs with a
    unique rho_2 = f(I - w_12 rho_1). The generic bistable regime has three
    roots (two stable, one saddle); the monostable regime has one.

    Returns a list of (rho_1, rho_2) arrays, sorted ascending by rho_1.
    """
    from scipy.optimize import brentq

    def g(r1: float) -> float:
        r2 = float(sigmoid.f(drive - w12 * r1))
        return r1 - float(sigmoid.f(drive - w21 * r2))

    xs = np.linspace(0.0, 1.0, n_samples)
    ys = np.array([g(x) for x in xs])
    roots: list[float] = []
    for i in range(len(xs) - 1):
        a, b = ys[i], ys[i + 1]
        if a == 0.0:
            roots.append(float(xs[i]))
        elif a * b < 0:
            try:
                r = brentq(g, float(xs[i]), float(xs[i + 1]), xtol=1e-12)
                roots.append(float(r))
            except Exception:
                pass
    # Deduplicate within rounding tolerance.
    roots = sorted(set(round(r, 10) for r in roots))
    fps = []
    for r1 in roots:
        r2 = float(sigmoid.f(drive - w12 * r1))
        fps.append(np.array([r1, r2]))
    return fps


def find_saddle_contralateral(
    w12: float,
    w21: float,
    drive: float,
    sigmoid: Sigmoid,
) -> Tuple[np.ndarray, int]:
    """Return the middle-branch fixed point and the total FP count.

    In the bistable regime the middle root of the scalar reduction is the
    unstable saddle; in the monostable regime there is a single root.
    """
    fps = find_all_fixed_points_contralateral(w12, w21, drive, sigmoid)
    n = len(fps)
    if n == 0:
        return np.array([np.nan, np.nan]), 0
    return fps[n // 2], n
