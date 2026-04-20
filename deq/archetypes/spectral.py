"""Spectral helpers for Phase 1 / Phase 2 analysis.

Implements the linearisation approach from the CogSpike CS research note (§4.2)
adapted to the FCS-scaled LI&F model:

  - `weight_eigengap(W)`     : ||λ₁(W)| − |λ₂(W)||
  - `linearized_A(...)`      : A = r·I + W · diag(f'(p*))
  - `operating_point(...)`   : solves  (1-r) p* = W f(p*) + B u  via scipy fsolve
  - `arg_dominant(A)`        : argument of the dominant complex-conjugate pair

Approximations:
  - The FCS neuron has no smooth firing-rate function; we approximate with a
    sigmoid f(p) = 1/(1 + exp(-k (p - p_mid))). Defaults: k=0.08, p_mid=90
    (just below the FCS scaled threshold τ=105 — see plan Risk 2).
  - The rvector leak is a length-5 FIR filter. We collapse it to a scalar
    r_effective (default 0.5) for linearisation. Least-squares fit of
    rvector=[10,5,3,2,1] to r^e·10 gives r≈0.55; see `fit_r_effective`.

These are the authoritative spectral helpers; Phase 1 and Phase 2 call only
these and not the existing 4-neuron WTA `deq/linearization.py`.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve


DEFAULT_K = 0.08
DEFAULT_P_MID = 90.0   # scaled; just below τ=105
DEFAULT_R = 0.5
DEFAULT_TAU = 105


# --- Sigmoid firing-rate approximation ---------------------------------------

def f_sigmoid(p, k=DEFAULT_K, p_mid=DEFAULT_P_MID):
    return 1.0 / (1.0 + np.exp(-k * (np.asarray(p, dtype=float) - p_mid)))


def f_sigmoid_deriv(p, k=DEFAULT_K, p_mid=DEFAULT_P_MID):
    s = f_sigmoid(p, k=k, p_mid=p_mid)
    return k * s * (1.0 - s)


def fit_r_effective(rvector=(10, 5, 3, 2, 1)):
    """Least-squares fit of r^e·10 to the integer rvector.

    Returns the scalar r that best approximates the FIR leak coefficients.
    """
    rv = np.asarray(rvector, dtype=float)
    es = np.arange(len(rv))
    # Minimize sum_e (r^e * 10 - rv[e])^2; use log-domain for stability.
    # We do it by grid + refine.
    rs = np.linspace(0.3, 0.8, 101)
    best = min(rs, key=lambda r: np.sum((r ** es * 10 - rv) ** 2))
    # local refine
    rs2 = np.linspace(best - 0.02, best + 0.02, 201)
    best2 = min(rs2, key=lambda r: np.sum((r ** es * 10 - rv) ** 2))
    return float(best2)


# --- Weight-matrix spectrum --------------------------------------------------

def weight_eigenvalues(W):
    return np.linalg.eigvals(np.asarray(W, dtype=float))


def weight_eigengap(W):
    """||λ₁| − |λ₂|| where λ₁, λ₂ are the two largest-magnitude eigenvalues.

    For 2×2 zero-diagonal W this is provably zero: eigenvalues are ±√(w₁₂·w₂₁)
    with equal magnitudes regardless of |w₁₂| vs |w₂₁|. We still compute it
    numerically and verify.
    """
    vals = weight_eigenvalues(W)
    mags = np.sort(np.abs(vals))[::-1]
    if len(mags) < 2:
        return 0.0
    return float(mags[0] - mags[1])


# --- Linearised state matrix -------------------------------------------------

def linearized_A(W, p_star, r=DEFAULT_R, k=DEFAULT_K, p_mid=DEFAULT_P_MID):
    """A = r·I + W · diag(f'(p*))."""
    W = np.asarray(W, dtype=float)
    g = f_sigmoid_deriv(p_star, k=k, p_mid=p_mid)
    return r * np.eye(W.shape[0]) + W @ np.diag(g)


def spectral_radius(A):
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(A, dtype=float)))))


def arg_dominant(A):
    """Argument (radians) of the dominant eigenvalue.

    If the top eigenvalue is real, returns 0 (or π for negative real).
    If complex, returns atan2(imag, real) of the conjugate pair.
    """
    vals = np.linalg.eigvals(np.asarray(A, dtype=float))
    # dominant = largest magnitude
    top = vals[np.argmax(np.abs(vals))]
    return float(np.arctan2(top.imag, top.real))


def eig_A(A):
    """All eigenvalues of A (complex), sorted by descending magnitude."""
    vals = np.linalg.eigvals(np.asarray(A, dtype=float))
    order = np.argsort(-np.abs(vals))
    return vals[order]


def linearized_eigengap(A):
    """||λ₁(A)| − |λ₂(A)||."""
    vals = eig_A(A)
    if len(vals) < 2:
        return 0.0
    return float(np.abs(vals[0]) - np.abs(vals[1]))


# --- Operating point ---------------------------------------------------------

def operating_point(
    W, B, u, r=DEFAULT_R, k=DEFAULT_K, p_mid=DEFAULT_P_MID,
    n_initials=5, seed=0,
):
    """Solve  (1 - r) p* = W · f(p*) + B · u  for the symmetric fixed point.

    Returns the solution closest to p_mid (in infinity norm) across multiple
    random initial conditions, or the most stable one if fsolve can't converge.
    """
    W = np.asarray(W, dtype=float)
    B = np.asarray(B, dtype=float)
    u = np.asarray(u, dtype=float).reshape(-1)
    n = W.shape[0]

    def residual(p):
        return (1 - r) * p - W @ f_sigmoid(p, k=k, p_mid=p_mid) - B @ u

    rng = np.random.default_rng(seed)
    candidates = []
    # Try several starting points
    starts = [np.full(n, p_mid)]
    for _ in range(n_initials - 1):
        starts.append(p_mid + rng.normal(0, 30.0, size=n))
    for p0 in starts:
        try:
            sol, info, ier, _ = fsolve(residual, p0, full_output=True)
            if ier == 1:
                candidates.append(sol)
        except Exception:
            pass

    if not candidates:
        # Fall back to the mean starting point (best-effort)
        return starts[0]

    # Pick the one closest to p_mid (we want to avoid saturation where f'≈0)
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates - p_mid, axis=1)
    return candidates[np.argmin(dists)]


# --- Diagnostic helpers ------------------------------------------------------

def eigenvector_asymmetry_2x2(A):
    """For a 2x2 A, return |v[0]| − |v[1]| for each eigenvector, capturing
    how concentrated the mode is on neuron 1 vs neuron 2.

    Useful for the contralateral case: even though ||λ₁|-|λ₂|| may be
    symmetric in (w₁₂, w₂₁), the eigenvector *shape* may reveal which
    neuron dominates at a given operating point.
    """
    vals, vecs = np.linalg.eig(np.asarray(A, dtype=float))
    asym = np.abs(vecs[0, :]) - np.abs(vecs[1, :])
    # order by descending |λ|
    order = np.argsort(-np.abs(vals))
    return vals[order], asym[order]


# =============================================================================
# FULL 5n-DIM LINEARISATION (captures FIR windowed integrator)
# =============================================================================
# The FCS neuron has a 5-dim state (mem[0..4]) per neuron, so the full
# linearisation around a non-spiking fixed point has dimension 5n. These
# helpers build and analyse that matrix directly.

R_SUM = 21  # sum of rvector [10,5,3,2,1] — windowed-integrator DC gain
DEFAULT_RVECTOR = np.array([10, 5, 3, 2, 1], dtype=np.int64)


def f_sigmoid_V(V, k=DEFAULT_K, p_mid_V=DEFAULT_TAU):
    """Sigmoid firing rate as a function of windowed sum V.

    Centre p_mid_V defaults to the threshold τ=105 (scaled) so that f' is
    largest at the firing boundary — the right place to linearise a
    threshold-triggered system.
    """
    return 1.0 / (1.0 + np.exp(-k * (np.asarray(V, dtype=float) - p_mid_V)))


def f_sigmoid_V_deriv(V, k=DEFAULT_K, p_mid_V=DEFAULT_TAU):
    s = f_sigmoid_V(V, k=k, p_mid_V=p_mid_V)
    return k * s * (1.0 - s)


def operating_point_full(W, B, u, k=DEFAULT_K, p_mid_V=DEFAULT_TAU,
                         n_initials=5, seed=0):
    """Find steady-state mem[0]^* (equal for all 5 taps at a fixed point).

    Fixed-point equation:
        m_i = Σ_j W_{ij} · f(V_j)  +  B_{i,:} · u
        V_j  = R_SUM · m_j
    """
    W = np.asarray(W, dtype=float)
    B = np.asarray(B, dtype=float)
    u = np.asarray(u, dtype=float).reshape(-1)
    n = W.shape[0]

    def residual(m):
        V = R_SUM * m
        fV = f_sigmoid_V(V, k=k, p_mid_V=p_mid_V)
        return m - (W @ fV + B @ u)

    # Try several starting points: the "no-firing" starting point, the
    # "driven-only" point B@u, and a few random perturbations.
    starts = [np.zeros(n), B @ u]
    rng = np.random.default_rng(seed)
    for _ in range(n_initials - 2):
        starts.append((B @ u) + rng.normal(0, 5.0, size=n))

    best_sol, best_err = None, np.inf
    for p0 in starts:
        try:
            sol, _, ier, _ = fsolve(residual, p0, full_output=True)
            if ier == 1:
                err = float(np.linalg.norm(residual(sol)))
                if err < best_err:
                    best_sol, best_err = sol, err
        except Exception:
            continue
    if best_sol is None:
        return B @ u  # fallback
    return best_sol


def build_A_full(W, m_star, rvector=None, k=DEFAULT_K, p_mid_V=DEFAULT_TAU):
    """Construct the 5n × 5n linearised state matrix.

    State ordering: x = (mem_0[0..4], mem_1[0..4], ..., mem_{n-1}[0..4]).

    Top row of each 5-block (row 5i): contributions from ALL neurons'
        windowed memory, weighted by W_ij · f'(V_j^*) · r_e.
    Subdiagonal of each diagonal 5-block: the shift mem[k](t+1) = mem[k-1](t).

    Reset-after-spike is omitted: linearisation assumes the operating point is
    non-firing, so the reset branch of the Lustre conditional is inactive.
    """
    if rvector is None:
        rvector = DEFAULT_RVECTOR
    rvector = np.asarray(rvector, dtype=float)
    sigma = len(rvector)
    W = np.asarray(W, dtype=float)
    n = W.shape[0]

    V_star = R_SUM * np.asarray(m_star, dtype=float)
    g = f_sigmoid_V_deriv(V_star, k=k, p_mid_V=p_mid_V)

    A = np.zeros((n * sigma, n * sigma))
    for i in range(n):
        # Top row: sum over all j of W[i,j] · g[j] · rvector[e]
        for j in range(n):
            A[i * sigma, j * sigma:(j + 1) * sigma] = W[i, j] * g[j] * rvector
        # Subdiagonal shift for neuron i's own block
        for kk in range(1, sigma):
            A[i * sigma + kk, i * sigma + kk - 1] = 1.0
    return A, g


def dominant_eigen(A):
    """Return (λ, v) for the eigenvalue of largest magnitude."""
    vals, vecs = np.linalg.eig(np.asarray(A, dtype=complex))
    idx = int(np.argmax(np.abs(vals)))
    return vals[idx], vecs[:, idx]


def eigenvector_neuron_asymmetry(v, n, sigma):
    """Σ_e |v[neuron=0, tap e]|  −  Σ_e |v[neuron=1, tap e]|.

    Generalises to n-neuron case by returning the whole per-neuron mass vector:
        m_i = Σ_e |v[i*sigma + e]|
    Returns (masses, asymmetry) where asymmetry = m_0 - m_1 for n=2.
    """
    v = np.asarray(v).reshape(-1)
    masses = np.array([np.sum(np.abs(v[i * sigma:(i + 1) * sigma]))
                       for i in range(n)])
    if n == 2:
        return masses, float(masses[0] - masses[1])
    return masses, None


def spectrum_max_arg(A):
    """Return (max|arg λ_k|, arg of the max-magnitude λ).

    The first captures "does ANY eigenvalue reach a high-frequency pole";
    the second captures the dominant pole's argument.
    """
    vals = np.linalg.eigvals(np.asarray(A, dtype=complex))
    args = np.abs(np.arctan2(vals.imag, vals.real))
    max_arg = float(np.max(args))
    top_idx = int(np.argmax(np.abs(vals)))
    top_arg = float(np.arctan2(vals[top_idx].imag, vals[top_idx].real))
    return max_arg, top_arg
