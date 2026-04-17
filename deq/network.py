"""
4-Neuron Winner-Take-All (WTA) Competitive Inhibition Network
==============================================================

Network topology:
  - 4 input neurons S1..S4, each feeding one core neuron N1..N4 (weight +100)
  - 4 output neurons O1..O4, each driven by one core neuron (weight +100)
  - Full inhibitory interconnection among N1..N4:
      N1 inhibits others at -100 (stronger)
      N2, N3, N4 inhibit others at -70 (weaker)

LIF dynamics (discrete-time, multiplicative leak):
  p_i(t+1) = floor(r * p_i(t) + sum_j W[i,j] * y_j(t) + B[i] * u_i(t))

Parameters from PRISM model:
  r = 0.5, threshold = 80, reset = 0
  4 probabilistic firing levels at potentials 20, 40, 60, 80
"""

import numpy as np

# --- LIF Parameters ---
LEAK_R = 0.5
P_THRESHOLD = 80
P_RESET = 0
P_REST = 0
NUM_NEURONS = 4  # Core neurons N1..N4

# Probabilistic firing thresholds (4 levels)
THRESHOLD_LEVELS = np.array([20, 40, 60, 80])
FIRING_PROBS = np.array([0.25, 0.50, 0.75, 1.00])

# Potential bounds (from PRISM model)
P_MIN = -360
P_MAX = 150

# --- Weight Matrix ---
# W[i,j] = synaptic weight from neuron j to neuron i
# Core inhibitory interconnection only (N1..N4)
W = np.array([
    [   0, -70, -70, -70],   # N1 receives -70 from N2, N3, N4
    [-100,   0, -70, -70],   # N2 receives -100 from N1, -70 from N3, N4
    [-100, -70,   0, -70],   # N3 receives -100 from N1, -70 from N2, N4
    [-100, -70, -70,   0],   # N4 receives -100 from N1, -70 from N2, N3
], dtype=float)

# Input gain matrix: each S_i feeds N_i with weight 100
B_IN = 100.0 * np.eye(NUM_NEURONS)

# Output gain matrix: each N_i feeds O_i with weight 100
C_OUT = 100.0 * np.eye(NUM_NEURONS)

# For state-space analysis, observe membrane potentials directly
C_STATE = np.eye(NUM_NEURONS)
D_ZERO = np.zeros((NUM_NEURONS, NUM_NEURONS))

NEURON_LABELS = ['N1', 'N2', 'N3', 'N4']
INPUT_LABELS = ['S1', 'S2', 'S3', 'S4']
OUTPUT_LABELS = ['O1', 'O2', 'O3', 'O4']


# ═══════════════════════════════════════════════════════════════
# FIRING RATE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def firing_probability(potential):
    """Return firing probability given membrane potential (scalar or array).
    Piecewise linear interpolation between threshold levels."""
    p = np.asarray(potential, dtype=float)
    prob = np.zeros_like(p)
    for i, (thresh, fp) in enumerate(zip(THRESHOLD_LEVELS, FIRING_PROBS)):
        if i == 0:
            mask = (p > 0) & (p <= thresh)
            prob[mask] = fp * p[mask] / thresh
        else:
            prev_thresh = THRESHOLD_LEVELS[i - 1]
            prev_fp = FIRING_PROBS[i - 1]
            mask = (p > prev_thresh) & (p <= thresh)
            prob[mask] = prev_fp + (fp - prev_fp) * (p[mask] - prev_thresh) / (thresh - prev_thresh)
    prob[p > THRESHOLD_LEVELS[-1]] = 1.0
    prob[p <= 0] = 0.0
    return prob


def firing_rate(potential):
    """Mean-field firing rate function f(p) — step function version.

    Uses the PRISM model's discrete threshold levels:
      p <= 20:  rate = 0
      20 < p <= 40:  rate = 0.25
      40 < p <= 60:  rate = 0.50
      60 < p <= 80:  rate = 0.75
      p > 80:  rate = 1.0
    """
    p = np.asarray(potential, dtype=float)
    rate = np.zeros_like(p)
    rate[(p > 20) & (p <= 40)] = 0.25
    rate[(p > 40) & (p <= 60)] = 0.50
    rate[(p > 60) & (p <= 80)] = 0.75
    rate[p > 80] = 1.0
    return rate


def firing_rate_smooth(potential):
    """Smoothed (piecewise-linear) firing rate for linearization."""
    return firing_probability(potential)


def firing_rate_derivative(potential):
    """Derivative df/dp of the smooth firing rate curve."""
    p = np.asarray(potential, dtype=float)
    dp = np.zeros_like(p)
    for i, (thresh, fp) in enumerate(zip(THRESHOLD_LEVELS, FIRING_PROBS)):
        if i == 0:
            mask = (p > 0) & (p <= thresh)
            dp[mask] = fp / thresh  # 0.25/20 = 0.0125
        else:
            prev_thresh = THRESHOLD_LEVELS[i - 1]
            prev_fp = FIRING_PROBS[i - 1]
            mask = (p > prev_thresh) & (p <= thresh)
            dp[mask] = (fp - prev_fp) / (thresh - prev_thresh)
    return dp


# ═══════════════════════════════════════════════════════════════
# SIGMOID APPROXIMATION (for linearization in saturated regimes)
# ═══════════════════════════════════════════════════════════════

# Sigmoid parameters: f(p) ≈ 1/(1 + exp(-k*(p - p_mid)))
# Chosen to match the discrete f-I curve: f(0)≈0, f(40)=0.5, f(80)≈1
SIGMOID_K = 0.08    # steepness (matched to 4-level curve)
SIGMOID_MID = 40.0  # midpoint of firing range


def firing_rate_sigmoid(potential):
    """Sigmoid approximation of the f-I curve.

    f(p) = 1 / (1 + exp(-k * (p - p_mid)))

    This is differentiable everywhere with non-zero derivative,
    making it suitable for linearization even when operating
    points are in saturated regions.
    """
    p = np.asarray(potential, dtype=float)
    return 1.0 / (1.0 + np.exp(-SIGMOID_K * (p - SIGMOID_MID)))


def firing_rate_sigmoid_derivative(potential):
    """Derivative of sigmoid f-I curve: f'(p) = k * f(p) * (1 - f(p))."""
    f = firing_rate_sigmoid(potential)
    return SIGMOID_K * f * (1.0 - f)


# ═══════════════════════════════════════════════════════════════
# LIF UPDATE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def lif_step_deterministic(potentials, spikes_internal, input_spikes):
    """Single deterministic LIF update step."""
    syn_input = W @ spikes_internal + B_IN @ input_spikes
    new_p = np.floor(LEAK_R * potentials + syn_input).astype(int)
    new_p = np.clip(new_p, P_MIN, P_MAX)
    new_spikes = (new_p >= P_THRESHOLD).astype(int)
    new_p[new_spikes == 1] = P_RESET
    return new_p.astype(float), new_spikes


def lif_step_stochastic(potentials, spikes_internal, input_spikes, rng=None):
    """Single stochastic LIF update step with probabilistic firing."""
    if rng is None:
        rng = np.random.default_rng()
    syn_input = W @ spikes_internal + B_IN @ input_spikes
    new_p = np.floor(LEAK_R * potentials + syn_input).astype(int)
    new_p = np.clip(new_p, P_MIN, P_MAX)
    probs = firing_rate(new_p.astype(float))
    draws = rng.random(NUM_NEURONS)
    new_spikes = (draws < probs).astype(int)
    new_p[new_spikes == 1] = P_RESET
    return new_p.astype(float), new_spikes


def lif_step_meanfield(potentials, rates, input_rates):
    """Mean-field LIF update using continuous firing rates."""
    syn_input = W @ rates + B_IN @ input_rates
    new_p = LEAK_R * potentials + syn_input
    new_rates = firing_rate_smooth(new_p)
    return new_p, new_rates


def lif_step_meanfield_sigmoid(potentials, rates, input_rates):
    """Mean-field LIF update using sigmoid firing rates."""
    syn_input = W @ rates + B_IN @ input_rates
    new_p = LEAK_R * potentials + syn_input
    new_rates = firing_rate_sigmoid(new_p)
    return new_p, new_rates


def print_network_summary():
    """Print a summary of the network configuration."""
    print("=" * 60)
    print("4-NEURON WTA COMPETITIVE INHIBITION NETWORK")
    print("=" * 60)
    print(f"\nLIF Parameters:")
    print(f"  Leak rate (r):     {LEAK_R}")
    print(f"  Threshold:         {P_THRESHOLD}")
    print(f"  Reset potential:   {P_RESET}")
    print(f"  Firing levels:     {THRESHOLD_LEVELS.tolist()}")
    print(f"  Firing probs:      {FIRING_PROBS.tolist()}")
    print(f"\nInhibitory Weight Matrix W (row i <- col j):")
    print(f"  {'':>4}", end="")
    for l in NEURON_LABELS:
        print(f"  {l:>6}", end="")
    print()
    for i, label in enumerate(NEURON_LABELS):
        print(f"  {label:>4}", end="")
        for j in range(NUM_NEURONS):
            print(f"  {W[i,j]:>6.0f}", end="")
        print()
    print(f"\nInput weights: {np.diag(B_IN).tolist()}")
    print(f"Output weights: {np.diag(C_OUT).tolist()}")
    print(f"\nKey asymmetry: N1 inhibits at -100, others inhibit at -70")
    print(f"  N1's inhibitory advantage: 30 units per connection")


if __name__ == "__main__":
    print_network_summary()
