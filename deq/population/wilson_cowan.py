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
