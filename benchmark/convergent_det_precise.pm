// Auto-generated PRISM model from CogSpike
// Neurons: 3, Edges: 2
//
// Node name mapping (GUI label -> PRISM variables):
//            In1 [Input]  ->  x_In1
//            In2 [Input]  ->  x_In2
//             N1 [Output]  ->  y_N1, p_N1
//
dtmc

// Global neuron parameters
const int P_rth = 100;
const int P_rest = 0;
const int P_reset = 0;
const double r = 0.95;
const int P_MIN = 0;
const int P_MAX = 200;

// Per-neuron potential bounds (optimized for state space)
const int P_MIN_N1 = 0;
const int P_MAX_N1 = 180;
const int T_MAX = 100;

// Firing probability thresholds (1 levels)
// Isomorphic with simulation: threshold_i = (i * P_rth) / levels
formula threshold1 = 100;

// Synaptic weights
const int w_In1_N1 = 60;
const int w_In2_N1 = 60;

// Spike propagation: neurons read y(source) directly (1-tick delay)

// Membrane potential update formulas (using per-neuron bounds)
formula newPotential_N1 = max(P_MIN_N1, min(P_MAX_N1, floor((w_In1_N1 * x_In1 + w_In2_N1 * x_In2) + r * p_N1)));

// Input generator formulas
formula in_In1_g0_fires = true;
formula in_In2_g0_fires = true;

module Inputs
  x_In1 : [0..1] init 0;
  x_In2 : [0..1] init 0;

  // Multi-generator input transitions
  [tick] in_In1_g0_fires -> (x_In1' = 1);
  [tick] !(in_In1_g0_fires) -> (x_In1' = 0);
  [tick] in_In2_g0_fires -> (x_In2' = 1);
  [tick] !(in_In2_g0_fires) -> (x_In2' = 0);
endmodule

module N1
  // No refractory periods - simplified model
  y_N1 : [0..1] init 0;  // spike output
  p_N1 : [P_MIN_N1..P_MAX_N1] init 0;  // membrane potential

  // Normal period - firing on newPotential (1 levels)
  [tick] newPotential_N1 <= threshold1 -> (y_N1' = 0) & (p_N1' = newPotential_N1);
  [tick] newPotential_N1 > threshold1 -> 1.0:(y_N1' = 1) & (p_N1' = P_reset);

endmodule


// Spike count rewards
rewards "spike_N1_count"
  y_N1 = 1 : 1;
endrewards

// Labels for PCTL properties
label "spike_N1" = (y_N1 = 1);
label "output_spike" = (y_N1 = 1);
