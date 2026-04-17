"""
Transient Analysis — Time-Domain Validation
=============================================

Compares the linearized system response with the actual nonlinear simulation:
  - Step response (all inputs on from rest)
  - Impulse response (single-step input)
  - Linear vs nonlinear overlay
  - Monte Carlo stochastic vs mean-field prediction
  - Settling time measurement

This validates the entire linearization approach by quantifying where the
linear approximation matches and diverges from the spiking dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from network import (
    NUM_NEURONS, NEURON_LABELS, LEAK_R, W, B_IN, P_THRESHOLD,
    firing_rate_smooth, firing_rate_derivative
)
from linearization import find_operating_point, build_state_space
from lif_simulation import (
    simulate_meanfield, monte_carlo_spike_counts, simulate_stochastic
)


def simulate_linear_system(A, B, u, T=50, p0=None):
    """Simulate the linearized discrete-time system.

    p(t+1) = A*p(t) + B*u(t)

    If p0 is None, starts from the operating point offset (i.e., p0=0 in
    deviation coordinates, which means the system starts AT the operating point).
    """
    if p0 is None:
        p0 = np.zeros(NUM_NEURONS)

    potentials = np.zeros((T + 1, NUM_NEURONS))
    potentials[0] = p0

    if u.ndim == 1:
        u = np.tile(u, (T, 1))

    for t in range(T):
        potentials[t + 1] = A @ potentials[t] + B @ u[t]

    return potentials


def step_response_comparison(save_dir="plots"):
    """Compare step responses: linear, mean-field, and stochastic.

    Step input: u = [1,1,1,1] applied at t=0 from rest.
    """
    T = 80
    u = np.ones(NUM_NEURONS)

    # --- Mean-field simulation (sigmoid, nonlinear) ---
    mf = simulate_meanfield(u, T=T, use_sigmoid=True)

    # --- Linearized simulation ---
    # Linearize around the steady-state operating point
    p_star, r_star = find_operating_point(u)
    A, B, C, D = build_state_space(p_star)

    # The linear system works in deviation coordinates: δp = p - p*
    # Step response from rest means initial δp = -p* (starting far from equilibrium)
    # But for the full-state linear system, we simulate directly:
    # p(t+1) = A*p(t) + B*u(t), starting from p(0) = 0
    linear_p = simulate_linear_system(A, B, u, T=T)

    # --- Monte Carlo stochastic ---
    mc = monte_carlo_spike_counts(u, T=T, n_trials=500, seed=42)

    # Also get a few individual stochastic traces
    stoch_traces = []
    for seed in [42, 123, 456]:
        result = simulate_stochastic(u, T=T, seed=seed)
        stoch_traces.append(result)

    # --- Plot comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

    # Top-left: Mean-field potentials
    ax = axes[0, 0]
    for i in range(NUM_NEURONS):
        ax.plot(mf['potentials'][:, i], color=colors[i], linewidth=2,
                label=NEURON_LABELS[i])
    ax.axhline(P_THRESHOLD, color='red', linestyle='--', alpha=0.3, label='Threshold')
    ax.set_ylabel('Membrane Potential')
    ax.set_title('Mean-Field (Nonlinear) Step Response')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: Linear potentials
    ax = axes[0, 1]
    for i in range(NUM_NEURONS):
        ax.plot(linear_p[:, i], color=colors[i], linewidth=2,
                label=NEURON_LABELS[i], linestyle='--')
    ax.axhline(P_THRESHOLD, color='red', linestyle='--', alpha=0.3)
    ax.set_ylabel('Membrane Potential')
    ax.set_title('Linearized System Step Response')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-left: Mean-field firing rates
    ax = axes[1, 0]
    for i in range(NUM_NEURONS):
        ax.plot(mf['rates'][:, i], color=colors[i], linewidth=2,
                label=NEURON_LABELS[i])
    # Overlay Monte Carlo mean rates as bars
    mc_rates = mc['mean_counts'] / T
    ax.axhline(mc_rates[0], color=colors[0], linestyle=':', alpha=0.7,
               label=f'MC avg N1={mc_rates[0]:.3f}')
    ax.axhline(mc_rates[1], color=colors[1], linestyle=':', alpha=0.7,
               label=f'MC avg N2={mc_rates[1]:.3f}')
    ax.set_ylabel('Firing Rate')
    ax.set_xlabel('Time Step')
    ax.set_title('Mean-Field Rates vs Monte Carlo Average')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Stochastic spike rasters (3 trials)
    ax = axes[1, 1]
    for trial_idx, result in enumerate(stoch_traces):
        for i in range(NUM_NEURONS):
            spike_times = np.where(result['spikes'][:, i] == 1)[0]
            y_pos = trial_idx * NUM_NEURONS + i
            ax.scatter(spike_times, np.full_like(spike_times, y_pos),
                       marker='|', s=30, color=colors[i], linewidths=1)
    # Labels
    ytick_pos = []
    ytick_labels = []
    for trial_idx in range(3):
        for i in range(NUM_NEURONS):
            ytick_pos.append(trial_idx * NUM_NEURONS + i)
            ytick_labels.append(f'T{trial_idx+1}:{NEURON_LABELS[i]}')
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels, fontsize=7)
    ax.set_xlabel('Time Step')
    ax.set_title('Stochastic Spike Rasters (3 trials)')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Step Response Comparison: Linear vs Nonlinear vs Stochastic', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/transient_step_response.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_dir}/transient_step_response.png")
    plt.close()

    return {
        'meanfield': mf,
        'linear': linear_p,
        'monte_carlo': mc,
    }


def impulse_response_analysis(save_dir="plots"):
    """Analyze impulse response: single spike at t=0, observe ring-down."""
    T = 30

    # Impulse: all inputs fire once at t=0, then silence
    u_impulse = np.zeros((T, NUM_NEURONS))
    u_impulse[0] = 1.0

    # Mean-field (sigmoid)
    from network import lif_step_meanfield_sigmoid
    potentials_mf = np.zeros((T + 1, NUM_NEURONS))
    rates_mf = np.zeros((T + 1, NUM_NEURONS))
    for t in range(T):
        potentials_mf[t+1], rates_mf[t+1] = lif_step_meanfield_sigmoid(
            potentials_mf[t], rates_mf[t], u_impulse[t]
        )

    # Linear
    u = np.ones(NUM_NEURONS)
    p_star, _ = find_operating_point(u)
    A, B, C, D = build_state_space(p_star)
    linear_p = simulate_linear_system(A, B, u_impulse, T=T)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

    ax = axes[0]
    for i in range(NUM_NEURONS):
        ax.plot(potentials_mf[:, i], color=colors[i], linewidth=2,
                label=NEURON_LABELS[i])
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Membrane Potential')
    ax.set_title('Impulse Response — Mean-Field')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i in range(NUM_NEURONS):
        ax.plot(linear_p[:, i], color=colors[i], linewidth=2,
                label=NEURON_LABELS[i], linestyle='--')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Membrane Potential')
    ax.set_title('Impulse Response — Linearized')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Impulse Response (single spike at t=0)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/transient_impulse.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_dir}/transient_impulse.png")
    plt.close()


def settling_time_analysis(save_dir="plots"):
    """Measure settling time and compare with eigenvalue predictions."""
    T = 100
    u = np.ones(NUM_NEURONS)

    # Mean-field simulation (sigmoid)
    mf = simulate_meanfield(u, T=T, use_sigmoid=True)

    # Find steady-state values
    ss_rates = mf['rates'][-1]
    ss_potentials = mf['potentials'][-1]

    # Settling time: first time the rate stays within 5% of steady state
    settling_times = {}
    for i in range(NUM_NEURONS):
        rate_trace = mf['rates'][:, i]
        target = ss_rates[i]
        if abs(target) < 1e-6:
            settling_times[NEURON_LABELS[i]] = 0
            continue

        tolerance = 0.05 * abs(target) + 1e-4
        settled = np.abs(rate_trace - target) < tolerance

        # Find the last time it wasn't settled
        not_settled = np.where(~settled)[0]
        if len(not_settled) > 0:
            settling_times[NEURON_LABELS[i]] = not_settled[-1] + 1
        else:
            settling_times[NEURON_LABELS[i]] = 0

    # Eigenvalue prediction
    p_star, _ = find_operating_point(u)
    from linearization import build_state_matrix
    A, G = build_state_matrix(p_star)
    eigenvalues = np.linalg.eigvals(A)
    dominant_mag = np.max(np.abs(eigenvalues))
    predicted_tau = -1 / np.log(dominant_mag) if dominant_mag > 1e-10 and dominant_mag < 1 else np.inf
    predicted_settling = 3 * predicted_tau  # 3 time constants ≈ 95% settling

    print(f"\n  Settling Time Analysis:")
    print(f"  ─────────────────────")
    for name, st in settling_times.items():
        print(f"    {name}: {st} steps")
    print(f"\n  Eigenvalue prediction:")
    print(f"    Dominant |λ| = {dominant_mag:.6f}")
    print(f"    Time constant τ = {predicted_tau:.3f} steps")
    print(f"    Predicted settling (3τ) = {3*predicted_tau:.1f} steps")

    # Plot settling
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
    for i in range(NUM_NEURONS):
        ax.plot(mf['rates'][:, i], color=colors[i], linewidth=2,
                label=f'{NEURON_LABELS[i]} (settle @ t={settling_times[NEURON_LABELS[i]]})')
        # Mark steady state
        ax.axhline(ss_rates[i], color=colors[i], linestyle=':', alpha=0.3)

    ax.axvline(predicted_settling, color='red', linestyle='--', alpha=0.5,
               label=f'Predicted 3τ = {predicted_settling:.1f}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Firing Rate')
    ax.set_title('Settling Time — Mean-Field Firing Rates')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/transient_settling.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_dir}/transient_settling.png")
    plt.close()

    return settling_times, predicted_settling


def run_transient_analysis(save_dir="plots"):
    """Run complete transient analysis."""
    print("\n" + "=" * 60)
    print("TRANSIENT ANALYSIS")
    print("=" * 60)

    print("\n  1. Step Response Comparison:")
    step_results = step_response_comparison(save_dir)

    print(f"\n  Monte Carlo spike counts (500 trials, T=50):")
    mc = step_results['monte_carlo']
    print(f"    Mean: {np.round(mc['mean_counts'], 2).tolist()}")
    print(f"    Std:  {np.round(mc['std_counts'], 2).tolist()}")
    print(f"    PRISM (3-neuron): N1=37.16, N2=7.77, N3=7.77")

    print(f"\n  2. Impulse Response:")
    impulse_response_analysis(save_dir)

    print(f"\n  3. Settling Time:")
    settling_times, predicted = settling_time_analysis(save_dir)

    return {
        'step': step_results,
        'settling_times': settling_times,
        'predicted_settling': predicted,
    }


if __name__ == "__main__":
    run_transient_analysis()
