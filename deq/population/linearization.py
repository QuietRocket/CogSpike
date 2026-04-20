"""Linearization primitives for the Wilson-Cowan rate system.

The Jacobian at a fixed point rho* of tau * d(rho)/dt = -rho + f(W rho + I) is
    J = (1/tau) * ( -I_n + diag(f'(W rho* + I)) @ W ).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from wilson_cowan import Sigmoid


def jacobian(
    W: np.ndarray, I: np.ndarray, rho_star: np.ndarray, tau: float, sigmoid: Sigmoid
) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    I = np.asarray(I, dtype=float)
    rho_star = np.asarray(rho_star, dtype=float)
    n = rho_star.size
    drive = W @ rho_star + I
    f_prime = sigmoid.f_prime(drive)
    return (-np.eye(n) + np.diag(f_prime) @ W) / tau


def spectrum(J: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eig(np.asarray(J, dtype=float))
    order = np.argsort(-eigvals.real)
    return eigvals[order], eigvecs[:, order]


def spectral_gap(J: np.ndarray) -> float:
    eigvals, _ = spectrum(J)
    if eigvals.size < 2:
        return float("inf")
    return float(eigvals[0].real - eigvals[1].real)


def is_stable(J: np.ndarray, tol: float = 1e-10) -> bool:
    eigvals, _ = spectrum(J)
    return bool(np.all(eigvals.real < -tol))
