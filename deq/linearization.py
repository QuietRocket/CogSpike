"""
Mean-Field Linearization of the WTA Network
=============================================

CHALLENGE: The weights (±100, ±70) are large relative to the firing
threshold range (0-80), so operating points land in saturated regions
where the piecewise-linear f-I curve has zero derivative. This makes
naive linearization degenerate (A = r*I, no coupling).

SOLUTION: Two complementary approaches:

1. SIGMOID APPROXIMATION: Use a smooth sigmoid f-I curve that has non-zero
   derivative everywhere. This captures the essential nonlinear gain.

2. RATE-BASED MODEL: Instead of linearizing the potential dynamics, work
   directly with firing rates as the state variable. This provides a
   complementary view where the weight matrix structure is explicit.

The LIF dynamics: p(t+1) = r * p(t) + W * f(p(t)) + B * u(t)

Linearizing around operating point p*:
  A_potential = r*I + W * diag(f'(p*))     [potential-space state matrix]

At steady state: p* = (W * f(p*) + B * u) / (1 - r)
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from network import (
    NUM_NEURONS, NEURON_LABELS, LEAK_R, W, B_IN, C_STATE, D_ZERO,
    firing_rate_smooth, firing_rate_derivative, firing_rate,
    firing_rate_sigmoid, firing_rate_sigmoid_derivative,
    THRESHOLD_LEVELS, FIRING_PROBS, SIGMOID_K, SIGMOID_MID,
    lif_step_meanfield_sigmoid
)


# ═══════════════════════════════════════════════════════════════
# OPERATING POINT FINDING
# ═══════════════════════════════════════════════════════════════

def find_operating_point(input_rates, max_iter=500, tol=1e-8):
    """Find mean-field steady-state using sigmoid f-I curve.

    Uses dampened iteration to handle the strongly nonlinear regime.
    p* = (W * f_sig(p*) + B * u) / (1 - r)
    """
    p = np.ones(NUM_NEURONS) * SIGMOID_MID  # Start at midpoint
    alpha = 0.3  # Dampening factor

    for _ in range(max_iter):
        rates = firing_rate_sigmoid(p)
        p_target = (W @ rates + B_IN @ input_rates) / (1 - LEAK_R)
        p_new = alpha * p_target + (1 - alpha) * p
        if np.max(np.abs(p_new - p)) < tol:
            break
        p = p_new

    rates = firing_rate_sigmoid(p)
    return p, rates


def find_operating_point_fsolve(input_rates):
    """Find operating point using scipy's fsolve (more robust).

    Solves: (1-r)*p - W*f(p) - B*u = 0
    """
    def residual(p):
        rates = firing_rate_sigmoid(p)
        return (1 - LEAK_R) * p - W @ rates - B_IN @ input_rates

    # Try multiple initial conditions
    best_p = None
    best_res = np.inf

    initials = [
        np.ones(NUM_NEURONS) * SIGMOID_MID,
        np.array([100, 0, 0, 0]),      # N1 winner
        np.array([0, 100, 0, 0]),      # N2 winner
        np.zeros(NUM_NEURONS),
        np.ones(NUM_NEURONS) * 80,
    ]

    for p0 in initials:
        try:
            p_sol = fsolve(residual, p0, full_output=False)
            res = np.max(np.abs(residual(p_sol)))
            if res < best_res:
                best_res = res
                best_p = p_sol
        except Exception:
            continue

    rates = firing_rate_sigmoid(best_p)
    return best_p, rates


def find_operating_point_piecewise(input_rates, max_iter=500, tol=1e-8):
    """Find operating point using the piecewise-linear f-I curve with dampening."""
    p = np.ones(NUM_NEURONS) * 40.0
    alpha = 0.1  # Strong dampening needed for piecewise

    for _ in range(max_iter):
        rates = firing_rate_smooth(p)
        p_target = (W @ rates + B_IN @ input_rates) / (1 - LEAK_R)
        p_new = alpha * p_target + (1 - alpha) * p
        if np.max(np.abs(p_new - p)) < tol:
            break
        p = p_new

    rates = firing_rate_smooth(p)
    return p, rates


# ═══════════════════════════════════════════════════════════════
# STATE MATRIX CONSTRUCTION
# ═══════════════════════════════════════════════════════════════

def build_state_matrix(operating_potentials, use_sigmoid=True):
    """Build the linearized state matrix A = r*I + W * diag(f'(p*)).

    Args:
        operating_potentials: (4,) steady-state membrane potentials
        use_sigmoid: if True, use sigmoid derivative (non-zero everywhere)

    Returns:
        A: (4,4) state matrix
        G: (4,) gain vector (derivatives of f-I curve at operating point)
    """
    if use_sigmoid:
        G = firing_rate_sigmoid_derivative(operating_potentials)
    else:
        G = firing_rate_derivative(operating_potentials)
    A = LEAK_R * np.eye(NUM_NEURONS) + W @ np.diag(G)
    return A, G


def build_state_space(operating_potentials, use_sigmoid=True):
    """Build complete (A, B, C, D) state-space representation."""
    A, G = build_state_matrix(operating_potentials, use_sigmoid)
    B = B_IN
    C = C_STATE
    D = D_ZERO
    return A, B, C, D


# ═══════════════════════════════════════════════════════════════
# RATE-BASED STATE SPACE (alternative formulation)
# ═══════════════════════════════════════════════════════════════

def build_rate_jacobian(operating_potentials):
    """Build the Jacobian of the rate dynamics.

    The rate update is: r(t+1) = f(r * p(t) + W * r(t) + B * u)
    where p(t) is implicitly a function of past rates.

    At steady state, the effective Jacobian for rate perturbations:
    J_rate = diag(f'(p*)) * (r * diag(dp*/dr) + W)

    For the simplified case (dominant effect through W):
    J_rate ≈ diag(f'(p*)) * W / (1 - r)  [amplified by steady-state gain]

    But the correct discrete-time formulation treats the full loop:
    δp(t+1) = r * δp(t) + W * diag(f'(p*)) * δp(t)
    So A = r*I + W*G where G = diag(f'(p*)).
    """
    G = firing_rate_sigmoid_derivative(operating_potentials)
    # The rate Jacobian maps rate perturbations to rate perturbations
    # dr(t+1)/dr(t) = f'(p*) * dp(t+1)/dp(t) * dp/dr
    # For potential: A_p = r*I + W*G
    # Rate output: r(t) = f(p(t)), so δr = G * δp
    # Full loop: δr(t+1) = G * A_p * G^{-1} * δr(t) if G is invertible
    # Simplification: eigenvalues of G * A_p * G^{-1} = eigenvalues of A_p
    # So the rate dynamics have the same eigenvalues as the potential dynamics.
    return G


# ═══════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_operating_regimes(save_dir="plots"):
    """Analyze different operating regimes and their linearizations."""
    print("\n" + "=" * 60)
    print("MEAN-FIELD LINEARIZATION ANALYSIS")
    print("=" * 60)

    regimes = {
        'All inputs on (u=[1,1,1,1])': np.ones(NUM_NEURONS),
        'Only N1 input (u=[1,0,0,0])': np.array([1, 0, 0, 0]),
        'N1+N2 inputs (u=[1,1,0,0])': np.array([1, 1, 0, 0]),
        'No inputs (u=[0,0,0,0])': np.zeros(NUM_NEURONS),
    }

    results = {}
    for name, u in regimes.items():
        print(f"\n--- {name} ---")

        # Sigmoid operating point (robust, differentiable everywhere)
        p_star, r_star = find_operating_point(u)
        A, G = build_state_matrix(p_star, use_sigmoid=True)

        print(f"  Sigmoid operating point:")
        print(f"    Potentials p*: {np.round(p_star, 2).tolist()}")
        print(f"    Firing rates:  {np.round(r_star, 4).tolist()}")
        print(f"    Gains g=f'(p*): {np.round(G, 6).tolist()}")

        print(f"\n  State matrix A = r*I + W*G:")
        for i in range(NUM_NEURONS):
            row = "    ["
            row += "  ".join(f"{A[i,j]:>8.4f}" for j in range(NUM_NEURONS))
            row += f" ]  ({NEURON_LABELS[i]})"
            print(row)

        eigenvalues = np.linalg.eigvals(A)
        print(f"\n  Eigenvalues of A: {np.round(eigenvalues, 6).tolist()}")
        print(f"  |λ| (magnitudes): {np.round(np.abs(eigenvalues), 6).tolist()}")
        stable = np.all(np.abs(eigenvalues) < 1)
        print(f"  Stable (all |λ| < 1): {stable}")

        # Also try fsolve
        p_fsolve, r_fsolve = find_operating_point_fsolve(u)
        A_f, G_f = build_state_matrix(p_fsolve, use_sigmoid=True)
        eig_f = np.linalg.eigvals(A_f)

        if np.max(np.abs(p_fsolve - p_star)) > 0.1:
            print(f"\n  fsolve operating point (different):")
            print(f"    Potentials: {np.round(p_fsolve, 2).tolist()}")
            print(f"    Rates:      {np.round(r_fsolve, 4).tolist()}")
            print(f"    Eigenvalues: {np.round(eig_f, 6).tolist()}")

        results[name] = {
            'u': u, 'p_star': p_star, 'rates': r_star,
            'G': G, 'A': A, 'eigenvalues': eigenvalues,
        }

    # --- Plot f-I curves ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    p_range = np.linspace(-100, 160, 1000)

    # All three f-I curves
    ax = axes[0]
    ax.plot(p_range, firing_rate_smooth(p_range), 'b-', linewidth=2,
            label='Piecewise-linear')
    ax.plot(p_range, firing_rate(p_range), 'r--', linewidth=2,
            label='Discrete (step)')
    ax.plot(p_range, firing_rate_sigmoid(p_range), 'g-', linewidth=2,
            label=f'Sigmoid (k={SIGMOID_K})')
    for thresh, prob in zip(THRESHOLD_LEVELS, FIRING_PROBS):
        ax.axvline(thresh, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('Membrane Potential p')
    ax.set_ylabel('Firing Rate f(p)')
    ax.set_title('f-I Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Sigmoid derivative
    ax = axes[1]
    ax.plot(p_range, firing_rate_sigmoid_derivative(p_range), 'g-', linewidth=2,
            label='Sigmoid derivative')
    ax.plot(p_range, firing_rate_derivative(p_range), 'b--', linewidth=2,
            label='Piecewise-linear derivative')
    # Mark operating points
    for name, res in results.items():
        for i in range(NUM_NEURONS):
            g = res['G'][i]
            if g > 1e-6:
                ax.plot(res['p_star'][i], g, 'o', markersize=8,
                        label=f"{NEURON_LABELS[i]} ({name[:15]})")
    ax.set_xlabel('Membrane Potential p')
    ax.set_ylabel("f'(p)")
    ax.set_title('Linearization Gain')
    ax.legend(fontsize=6, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Operating points on f-I curve
    ax = axes[2]
    ax.plot(p_range, firing_rate_sigmoid(p_range), 'g-', linewidth=2)
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
    for name, res in results.items():
        for i in range(NUM_NEURONS):
            ax.plot(res['p_star'][i], res['rates'][i], 'o', color=colors[i],
                    markersize=8)
            # Draw tangent line
            g = res['G'][i]
            dp = 30
            p_tan = np.linspace(res['p_star'][i] - dp, res['p_star'][i] + dp, 50)
            f_tan = res['rates'][i] + g * (p_tan - res['p_star'][i])
            ax.plot(p_tan, f_tan, '--', color=colors[i], alpha=0.3, linewidth=1)
    ax.set_xlabel('Membrane Potential p')
    ax.set_ylabel('Firing Rate')
    ax.set_title('Operating Points & Tangent Lines')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.1])

    plt.tight_layout()
    plt.savefig(f"{save_dir}/linearization_fi_curve.png", dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {save_dir}/linearization_fi_curve.png")
    plt.close()

    # --- Raw weight matrix analysis ---
    print(f"\n  RAW WEIGHT MATRIX EIGENANALYSIS")
    print(f"  ─────────────────────────────")
    eig_W = np.linalg.eigvals(W)
    eigvals_W, eigvecs_W = np.linalg.eig(W)
    idx = np.argsort(-np.abs(eigvals_W))
    eigvals_W = eigvals_W[idx]
    eigvecs_W = eigvecs_W[:, idx]

    print(f"  Eigenvalues of W: {np.round(eigvals_W, 4).tolist()}")
    print(f"  |λ|:              {np.round(np.abs(eigvals_W), 4).tolist()}")
    print(f"\n  Eigenvectors (columns):")
    for i in range(NUM_NEURONS):
        v = eigvecs_W[:, i].real
        v_norm = v / np.max(np.abs(v))
        print(f"    Mode {i+1} (λ={eigvals_W[i]:.2f}):", end="")
        for j in range(NUM_NEURONS):
            print(f"  {NEURON_LABELS[j]}={v_norm[j]:+.3f}", end="")
        print()

    print(f"\n  The weight matrix has a near-symmetric structure (3 neurons at -70,")
    print(f"  1 neuron at -100). The eigenvalue decomposition reveals:")
    print(f"    λ₁={eigvals_W[0]:.1f}: strongest mode — governs overall inhibition strength")
    print(f"    λ₂={eigvals_W[1]:.1f}: N1's asymmetric advantage mode")
    print(f"    λ₃,₄={eigvals_W[2]:.1f}: degenerate modes — N2/N3/N4 symmetry")

    return results


if __name__ == "__main__":
    analyze_operating_regimes()
