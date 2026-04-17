"""
Steady-State and Equilibrium Analysis
======================================

Finds and characterizes the fixed points of the nonlinear WTA system.

At steady state with constant input u:
  p* = r * p* + W * f(p*) + B * u
  => p* = (W * f(p*) + B * u) / (1 - r)

Multiple equilibria may exist:
  - N1-winner: N1 at high rate, others suppressed
  - N2/N3/N4-winner: alternative winner states (less stable due to weight asymmetry)
  - Symmetric: all neurons at same rate (unstable saddle point)

The Jacobian at each fixed point determines local stability:
  J = r*I + W * diag(f'(p*))

Phase portrait shows trajectories and basins of attraction.
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from network import (
    NUM_NEURONS, NEURON_LABELS, LEAK_R, W, B_IN,
    firing_rate_smooth, firing_rate_derivative, firing_rate,
    firing_rate_sigmoid, firing_rate_sigmoid_derivative,
    lif_step_meanfield, lif_step_meanfield_sigmoid
)
from linearization import build_state_matrix


def fixed_point_equation(p, u):
    """F(p) = 0 at the fixed point.

    p* = (W * f(p*) + B * u) / (1 - r)
    => (1-r)*p* - W*f(p*) - B*u = 0
    """
    rates = firing_rate_smooth(p)
    return (1 - LEAK_R) * p - W @ rates - B_IN @ u


def find_all_fixed_points(u, n_random=50, seed=42):
    """Search for multiple fixed points using different initial conditions.

    Returns list of unique fixed points.
    """
    rng = np.random.default_rng(seed)
    fixed_points = []

    # Structured initial conditions
    initials = [
        np.zeros(NUM_NEURONS),                    # From rest
        np.ones(NUM_NEURONS) * 50,                # All at midpoint
        np.ones(NUM_NEURONS) * 100,               # All above threshold
    ]
    # Single-winner initial conditions
    for i in range(NUM_NEURONS):
        ic = np.zeros(NUM_NEURONS)
        ic[i] = 100
        initials.append(ic)

    # Random initial conditions
    for _ in range(n_random):
        initials.append(rng.uniform(-50, 150, NUM_NEURONS))

    for p0 in initials:
        try:
            p_sol, info, ier, msg = fsolve(fixed_point_equation, p0, args=(u,),
                                            full_output=True)
            if ier == 1:  # Converged
                # Check it's actually a fixed point
                residual = np.max(np.abs(fixed_point_equation(p_sol, u)))
                if residual < 1e-6:
                    # Check if it's a new fixed point
                    is_new = True
                    for fp in fixed_points:
                        if np.max(np.abs(fp['p'] - p_sol)) < 1e-3:
                            is_new = False
                            break
                    if is_new:
                        rates = firing_rate_smooth(p_sol)
                        A, G = build_state_matrix(p_sol)
                        eigenvalues = np.linalg.eigvals(A)
                        stable = np.all(np.abs(eigenvalues) < 1.0)

                        fixed_points.append({
                            'p': p_sol,
                            'rates': rates,
                            'A': A,
                            'eigenvalues': eigenvalues,
                            'stable': stable,
                            'residual': residual,
                        })
        except Exception:
            continue

    return fixed_points


def classify_fixed_point(fp):
    """Classify a fixed point based on its eigenvalues and firing pattern."""
    rates = fp['rates']
    eigs = fp['eigenvalues']

    # Firing pattern
    active = np.where(rates > 0.1)[0]
    if len(active) == 0:
        pattern = "Quiescent"
    elif len(active) == 1:
        pattern = f"WTA: {NEURON_LABELS[active[0]]} wins"
    elif len(active) == NUM_NEURONS:
        if np.std(rates) < 0.05:
            pattern = "Symmetric (all equal)"
        else:
            winner = np.argmax(rates)
            pattern = f"Partial WTA: {NEURON_LABELS[winner]} dominant"
    else:
        pattern = f"Mixed ({len(active)} active)"

    # Stability type
    if fp['stable']:
        if np.all(np.isreal(eigs)):
            stability = "Stable node"
        else:
            stability = "Stable spiral"
    else:
        real_positive = np.sum(np.abs(eigs) > 1)
        if np.all(np.isreal(eigs)):
            stability = f"Saddle point ({real_positive} unstable dims)"
        else:
            stability = f"Unstable spiral ({real_positive} unstable dims)"

    return pattern, stability


def plot_phase_portrait(u, fixed_points, save_dir="plots"):
    """Plot 2D phase portrait projected onto N1-N2 membrane potential plane.

    Simulates trajectories from a grid of initial conditions and colors
    them by which attractor they reach.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left: Trajectory plot ---
    ax = axes[0]
    grid_size = 15
    p1_range = np.linspace(-50, 150, grid_size)
    p2_range = np.linspace(-50, 150, grid_size)

    colors_winner = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9E9E9E']
    T_sim = 100

    for p1_init in p1_range:
        for p2_init in p2_range:
            p = np.array([p1_init, p2_init, 30, 30])  # N3, N4 at midpoint
            rates = firing_rate_sigmoid(p)
            trajectory_p1 = [p[0]]
            trajectory_p2 = [p[1]]

            for _ in range(T_sim):
                p, rates = lif_step_meanfield_sigmoid(p, rates, u)
                trajectory_p1.append(p[0])
                trajectory_p2.append(p[1])

            # Color by winner at end
            winner = np.argmax(rates)
            if np.max(rates) < 0.05:
                color = colors_winner[4]  # Gray for quiescent
            else:
                color = colors_winner[winner]

            ax.plot(trajectory_p1, trajectory_p2, color=color, alpha=0.3, linewidth=0.5)
            ax.plot(trajectory_p1[-1], trajectory_p2[-1], '.', color=color,
                    markersize=3, alpha=0.5)

    # Mark fixed points
    for fp in fixed_points:
        marker = 'o' if fp['stable'] else 'x'
        size = 15 if fp['stable'] else 12
        pattern, _ = classify_fixed_point(fp)
        ax.plot(fp['p'][0], fp['p'][1], marker, color='black',
                markersize=size, markeredgewidth=2,
                label=f"{'●' if fp['stable'] else '✕'} {pattern}")

    ax.set_xlabel('N1 Potential', fontsize=12)
    ax.set_ylabel('N2 Potential', fontsize=12)
    ax.set_title('Phase Portrait (N1-N2 plane)\nTrajectories colored by winner')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

    # --- Right: Basin of attraction ---
    ax = axes[1]
    grid_size_basin = 40
    p1_range = np.linspace(-50, 150, grid_size_basin)
    p2_range = np.linspace(-50, 150, grid_size_basin)
    basin = np.zeros((grid_size_basin, grid_size_basin))

    for i, p2_init in enumerate(p2_range):
        for j, p1_init in enumerate(p1_range):
            p = np.array([p1_init, p2_init, 30, 30])
            rates = firing_rate_sigmoid(p)
            for _ in range(200):
                p, rates = lif_step_meanfield_sigmoid(p, rates, u)
            winner = np.argmax(rates)
            if np.max(rates) < 0.05:
                basin[i, j] = -1  # Quiescent
            else:
                basin[i, j] = winner

    cmap = ListedColormap(['#BDBDBD', '#2196F3', '#FF9800', '#4CAF50', '#E91E63'])
    im = ax.imshow(basin, origin='lower', aspect='auto',
                   extent=[p1_range[0], p1_range[-1], p2_range[0], p2_range[-1]],
                   cmap=cmap, vmin=-1, vmax=3)

    # Mark fixed points
    for fp in fixed_points:
        marker = 'o' if fp['stable'] else 'x'
        ax.plot(fp['p'][0], fp['p'][1], marker, color='white',
                markersize=12, markeredgewidth=2, markeredgecolor='black')

    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1, 2, 3])
    cbar.ax.set_yticklabels(['Quiescent', 'N1 wins', 'N2 wins', 'N3 wins', 'N4 wins'])
    ax.set_xlabel('N1 Initial Potential', fontsize=12)
    ax.set_ylabel('N2 Initial Potential', fontsize=12)
    ax.set_title('Basin of Attraction Map\n(N3, N4 initialized at 30)')

    fig.suptitle(f'Steady-State Analysis — u={u.tolist()}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/steady_state_phase.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_dir}/steady_state_phase.png")
    plt.close()


def run_steady_state_analysis(save_dir="plots"):
    """Run complete steady-state analysis."""
    print("\n" + "=" * 60)
    print("STEADY-STATE / EQUILIBRIUM ANALYSIS")
    print("=" * 60)

    u = np.ones(NUM_NEURONS)  # All inputs on

    print(f"\n  Searching for fixed points with u = {u.tolist()} ...")
    fixed_points = find_all_fixed_points(u)

    print(f"\n  Found {len(fixed_points)} distinct fixed point(s):")
    for i, fp in enumerate(fixed_points):
        pattern, stability = classify_fixed_point(fp)
        print(f"\n  Fixed Point {i+1}: {pattern}")
        print(f"    Potentials: {np.round(fp['p'], 2).tolist()}")
        print(f"    Rates:      {np.round(fp['rates'], 4).tolist()}")
        print(f"    Eigenvalues: {np.round(fp['eigenvalues'], 4).tolist()}")
        print(f"    |λ|:         {np.round(np.abs(fp['eigenvalues']), 4).tolist()}")
        print(f"    Stability:   {stability}")
        print(f"    Residual:    {fp['residual']:.2e}")

    # Hydraulic analogy
    print(f"\n  Hydraulic Analogy (water flow through neurons):")
    print(f"  ─────────────────────────────────────────────")
    if len(fixed_points) > 0:
        # Find the dominant equilibrium
        stable_fps = [fp for fp in fixed_points if fp['stable']]
        if stable_fps:
            fp = max(stable_fps, key=lambda x: np.max(x['rates']))
            winner = np.argmax(fp['rates'])
            print(f"    At equilibrium, {NEURON_LABELS[winner]} acts as the main channel:")
            print(f"    - Input flow:  100 units/step from each S_i")
            print(f"    - Leak drain:  {LEAK_R*100:.0f}% of stored potential per step")
            for i in range(NUM_NEURONS):
                inflow = 100 * u[i]
                inhibition_received = sum(W[i, j] * fp['rates'][j]
                                          for j in range(NUM_NEURONS) if j != i)
                net_input = inflow + inhibition_received
                leak = (1 - LEAK_R) * fp['p'][i]
                print(f"    - {NEURON_LABELS[i]}: net_input={net_input:+.1f}, "
                      f"leak_drain={leak:.1f}, "
                      f"rate={fp['rates'][i]:.3f}")

    # Phase portrait
    plot_phase_portrait(u, fixed_points, save_dir)

    # Also analyze the no-competition case
    print(f"\n  --- Single-input case u=[1,0,0,0] ---")
    u_single = np.array([1, 0, 0, 0])
    fps_single = find_all_fixed_points(u_single)
    for i, fp in enumerate(fps_single):
        pattern, stability = classify_fixed_point(fp)
        print(f"  FP{i+1}: {pattern} | p={np.round(fp['p'],1).tolist()} "
              f"| rates={np.round(fp['rates'],3).tolist()} | {stability}")

    return {
        'fixed_points': fixed_points,
        'fixed_points_single': fps_single,
    }


if __name__ == "__main__":
    run_steady_state_analysis()
