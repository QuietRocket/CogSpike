"""
Discrete-Time LIF Simulator
============================

Provides deterministic and stochastic simulation of the 4-neuron WTA network.
Validates against PRISM spike counts from the case study analysis.

PRISM reference (3-neuron case, all inputs always-on):
  N1 (winner): 37.16 spikes / 50 steps
  N2 (loser):   7.77 spikes / 50 steps
  N3 (loser):   7.77 spikes / 50 steps
"""

import numpy as np
import matplotlib.pyplot as plt
from network import (
    NUM_NEURONS, NEURON_LABELS, LEAK_R, P_THRESHOLD, P_RESET,
    W, B_IN, P_MIN, P_MAX, firing_rate,
    lif_step_deterministic, lif_step_stochastic, lif_step_meanfield,
    lif_step_meanfield_sigmoid, firing_rate_smooth, firing_rate_sigmoid
)


def simulate_deterministic(input_spikes, T=50):
    """Run deterministic LIF simulation for T steps.

    Args:
        input_spikes: (4,) constant input spike vector, or (T, 4) time-varying
        T: number of time steps

    Returns:
        dict with 'potentials' (T+1, 4), 'spikes' (T, 4), 'spike_counts' (4,)
    """
    if input_spikes.ndim == 1:
        input_spikes = np.tile(input_spikes, (T, 1))

    potentials = np.zeros((T + 1, NUM_NEURONS))
    spikes = np.zeros((T, NUM_NEURONS), dtype=int)
    prev_spikes = np.zeros(NUM_NEURONS)

    for t in range(T):
        new_p, new_s = lif_step_deterministic(
            potentials[t], prev_spikes, input_spikes[t]
        )
        potentials[t + 1] = new_p
        spikes[t] = new_s
        prev_spikes = new_s

    return {
        'potentials': potentials,
        'spikes': spikes,
        'spike_counts': spikes.sum(axis=0),
    }


def simulate_stochastic(input_spikes, T=50, seed=None):
    """Run stochastic LIF simulation with probabilistic firing.

    Args:
        input_spikes: (4,) constant input spike vector, or (T, 4) time-varying
        T: number of time steps
        seed: random seed

    Returns:
        dict with 'potentials', 'spikes', 'spike_counts'
    """
    rng = np.random.default_rng(seed)
    if input_spikes.ndim == 1:
        input_spikes = np.tile(input_spikes, (T, 1))

    potentials = np.zeros((T + 1, NUM_NEURONS))
    spikes = np.zeros((T, NUM_NEURONS), dtype=int)
    prev_spikes = np.zeros(NUM_NEURONS)

    for t in range(T):
        new_p, new_s = lif_step_stochastic(
            potentials[t], prev_spikes, input_spikes[t], rng
        )
        potentials[t + 1] = new_p
        spikes[t] = new_s
        prev_spikes = new_s

    return {
        'potentials': potentials,
        'spikes': spikes,
        'spike_counts': spikes.sum(axis=0),
    }


def simulate_meanfield(input_rates, T=50, use_sigmoid=False):
    """Run mean-field (rate-based) simulation.

    Uses continuous firing rates instead of binary spikes.

    Args:
        input_rates: (4,) constant input rates, or (T, 4) time-varying
        T: number of time steps
        use_sigmoid: if True, use sigmoid f-I curve (better for mean-field)

    Returns:
        dict with 'potentials' (T+1, 4), 'rates' (T+1, 4)
    """
    if input_rates.ndim == 1:
        input_rates = np.tile(input_rates, (T, 1))

    step_fn = lif_step_meanfield_sigmoid if use_sigmoid else lif_step_meanfield

    potentials = np.zeros((T + 1, NUM_NEURONS))
    rates = np.zeros((T + 1, NUM_NEURONS))

    for t in range(T):
        new_p, new_r = step_fn(potentials[t], rates[t], input_rates[t])
        potentials[t + 1] = new_p
        rates[t + 1] = new_r

    return {
        'potentials': potentials,
        'rates': rates,
    }


def monte_carlo_spike_counts(input_spikes, T=50, n_trials=1000, seed=42):
    """Run Monte Carlo simulation and return mean spike counts.

    Args:
        input_spikes: (4,) constant input spike vector
        T: time steps per trial
        n_trials: number of Monte Carlo trials
        seed: base random seed

    Returns:
        dict with 'mean_counts' (4,), 'std_counts' (4,), 'all_counts' (n_trials, 4)
    """
    rng = np.random.default_rng(seed)
    all_counts = np.zeros((n_trials, NUM_NEURONS))

    for trial in range(n_trials):
        result = simulate_stochastic(input_spikes, T, seed=rng.integers(0, 2**31))
        all_counts[trial] = result['spike_counts']

    return {
        'mean_counts': all_counts.mean(axis=0),
        'std_counts': all_counts.std(axis=0),
        'all_counts': all_counts,
    }


def plot_simulation(result, title="LIF Simulation", save_path=None):
    """Plot membrane potentials and spike raster from a simulation result."""
    T = result['spikes'].shape[0]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})

    # Membrane potentials
    ax = axes[0]
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
    for i in range(NUM_NEURONS):
        ax.plot(range(T + 1), result['potentials'][:, i],
                label=NEURON_LABELS[i], color=colors[i], linewidth=1.5)
    ax.axhline(y=P_THRESHOLD, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_ylabel('Membrane Potential')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Spike raster
    ax = axes[1]
    for i in range(NUM_NEURONS):
        spike_times = np.where(result['spikes'][:, i] == 1)[0]
        ax.scatter(spike_times, np.full_like(spike_times, i), marker='|',
                   s=100, color=colors[i], linewidths=2)
    ax.set_yticks(range(NUM_NEURONS))
    ax.set_yticklabels(NEURON_LABELS)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Neuron')
    ax.set_title(f'Spike Raster (counts: {result["spike_counts"].tolist()})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


def run_validation(save_dir="plots"):
    """Run simulations and validate against PRISM results."""
    print("\n" + "=" * 60)
    print("LIF SIMULATION VALIDATION")
    print("=" * 60)

    u = np.ones(NUM_NEURONS)  # All inputs always on

    # --- Deterministic simulation ---
    print("\n1. Deterministic simulation (T=50):")
    det_result = simulate_deterministic(u, T=50)
    print(f"   Spike counts: {det_result['spike_counts'].tolist()}")
    plot_simulation(det_result,
                    "Deterministic LIF Simulation (T=50, all inputs on)",
                    f"{save_dir}/sim_deterministic.png")

    # --- Single stochastic trial ---
    print("\n2. Single stochastic trial (T=50):")
    stoch_result = simulate_stochastic(u, T=50, seed=42)
    print(f"   Spike counts: {stoch_result['spike_counts'].tolist()}")
    plot_simulation(stoch_result,
                    "Stochastic LIF Simulation (single trial, T=50)",
                    f"{save_dir}/sim_stochastic_single.png")

    # --- Monte Carlo ---
    print("\n3. Monte Carlo (1000 trials, T=50):")
    mc = monte_carlo_spike_counts(u, T=50, n_trials=1000, seed=42)
    print(f"   Mean spike counts: {np.round(mc['mean_counts'], 2).tolist()}")
    print(f"   Std dev:           {np.round(mc['std_counts'], 2).tolist()}")
    print(f"\n   PRISM reference (3-neuron case):")
    print(f"     N1: 37.16, N2: 7.77, N3: 7.77")
    print(f"   Note: 4-neuron case has additional inhibition on N1 from N4")

    # Monte Carlo histogram
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
    for i in range(NUM_NEURONS):
        axes[i].hist(mc['all_counts'][:, i], bins=20, color=colors[i],
                     alpha=0.7, edgecolor='black')
        axes[i].axvline(mc['mean_counts'][i], color='red', linestyle='--',
                        label=f'mean={mc["mean_counts"][i]:.1f}')
        axes[i].set_title(f'{NEURON_LABELS[i]}')
        axes[i].set_xlabel('Spike Count')
        axes[i].legend()
    axes[0].set_ylabel('Frequency')
    fig.suptitle('Monte Carlo Spike Count Distribution (1000 trials, T=50)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sim_monte_carlo_hist.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_dir}/sim_monte_carlo_hist.png")
    plt.close()

    # --- Mean-field simulation (sigmoid) ---
    print("\n4. Mean-field (sigmoid f-I, rate-based) simulation (T=100):")
    mf = simulate_meanfield(u, T=100, use_sigmoid=True)
    print(f"   Final potentials: {np.round(mf['potentials'][-1], 2).tolist()}")
    print(f"   Final rates:      {np.round(mf['rates'][-1], 4).tolist()}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for i in range(NUM_NEURONS):
        axes[0].plot(mf['potentials'][:, i], label=NEURON_LABELS[i],
                     color=colors[i], linewidth=1.5)
        axes[1].plot(mf['rates'][:, i], label=NEURON_LABELS[i],
                     color=colors[i], linewidth=1.5)
    axes[0].set_ylabel('Membrane Potential')
    axes[0].set_title('Mean-Field Simulation (rate-based)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel('Firing Rate')
    axes[1].set_xlabel('Time Step')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sim_meanfield.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_dir}/sim_meanfield.png")
    plt.close()

    return {
        'deterministic': det_result,
        'stochastic': stoch_result,
        'monte_carlo': mc,
        'meanfield': mf,
    }


if __name__ == "__main__":
    run_validation()
