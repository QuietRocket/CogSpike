// Auto-generated PRISM model from CogSpike
// Neurons: 5, Edges: 4
//
// Node name mapping (GUI label -> PRISM variables):
//             In [Input]  ->  x_In
//             N1 [Neuron]  ->  y_N1, p_N1
//             N2 [Neuron]  ->  y_N2, p_N2
//             N3 [Neuron]  ->  y_N3, p_N3
//             N4 [Output]  ->  y_N4, p_N4
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
const int P_MAX_N1 = 120;
const int P_MIN_N2 = 0;
const int P_MAX_N2 = 120;
const int P_MIN_N3 = 0;
const int P_MAX_N3 = 120;
const int P_MIN_N4 = 0;
const int P_MAX_N4 = 120;
const int T_MAX = 100;

// Firing probability thresholds (4 levels)
// Isomorphic with simulation: threshold_i = (i * P_rth) / levels
formula threshold1 = 25;
formula threshold2 = 50;
formula threshold3 = 75;
formula threshold4 = 100;

// Synaptic weights
const int w_In_N1 = 80;
const int w_N1_N2 = 80;
const int w_N2_N3 = 80;
const int w_N3_N4 = 80;

// Spike propagation: neurons read y(source) directly (1-tick delay)

// Membrane potential update formulas (using per-neuron bounds)
formula newPotential_N1 = max(P_MIN_N1, min(P_MAX_N1, floor((w_In_N1 * x_In) + r * p_N1)));
formula newPotential_N2 = max(P_MIN_N2, min(P_MAX_N2, floor((w_N1_N2 * y_N1) + r * p_N2)));
formula newPotential_N3 = max(P_MIN_N3, min(P_MAX_N3, floor((w_N2_N3 * y_N2) + r * p_N3)));
formula newPotential_N4 = max(P_MIN_N4, min(P_MAX_N4, floor((w_N3_N4 * y_N3) + r * p_N4)));

// Input generator formulas
formula in_In_g0_fires = true;

module Inputs
  x_In : [0..1] init 0;

  // Multi-generator input transitions
  [tick] in_In_g0_fires -> (x_In' = 1);
  [tick] !(in_In_g0_fires) -> (x_In' = 0);
endmodule

module N1
  // No refractory periods - simplified model
  y_N1 : [0..1] init 0;  // spike output
  p_N1 : [P_MIN_N1..P_MAX_N1] init 0;  // membrane potential

  // Normal period - firing on newPotential (4 levels)
  [tick] newPotential_N1 <= threshold1 -> (y_N1' = 0) & (p_N1' = newPotential_N1);
  [tick] newPotential_N1 > threshold1 & newPotential_N1 <= threshold2 -> 0.7500:(y_N1' = 0) & (p_N1' = newPotential_N1) + 0.2500:(y_N1' = 1) & (p_N1' = P_reset);
  [tick] newPotential_N1 > threshold2 & newPotential_N1 <= threshold3 -> 0.5000:(y_N1' = 0) & (p_N1' = newPotential_N1) + 0.5000:(y_N1' = 1) & (p_N1' = P_reset);
  [tick] newPotential_N1 > threshold3 & newPotential_N1 <= threshold4 -> 0.2500:(y_N1' = 0) & (p_N1' = newPotential_N1) + 0.7500:(y_N1' = 1) & (p_N1' = P_reset);
  [tick] newPotential_N1 > threshold4 -> 1.0:(y_N1' = 1) & (p_N1' = P_reset);

endmodule

module N2
  // No refractory periods - simplified model
  y_N2 : [0..1] init 0;  // spike output
  p_N2 : [P_MIN_N2..P_MAX_N2] init 0;  // membrane potential

  // Normal period - firing on newPotential (4 levels)
  [tick] newPotential_N2 <= threshold1 -> (y_N2' = 0) & (p_N2' = newPotential_N2);
  [tick] newPotential_N2 > threshold1 & newPotential_N2 <= threshold2 -> 0.7500:(y_N2' = 0) & (p_N2' = newPotential_N2) + 0.2500:(y_N2' = 1) & (p_N2' = P_reset);
  [tick] newPotential_N2 > threshold2 & newPotential_N2 <= threshold3 -> 0.5000:(y_N2' = 0) & (p_N2' = newPotential_N2) + 0.5000:(y_N2' = 1) & (p_N2' = P_reset);
  [tick] newPotential_N2 > threshold3 & newPotential_N2 <= threshold4 -> 0.2500:(y_N2' = 0) & (p_N2' = newPotential_N2) + 0.7500:(y_N2' = 1) & (p_N2' = P_reset);
  [tick] newPotential_N2 > threshold4 -> 1.0:(y_N2' = 1) & (p_N2' = P_reset);

endmodule

module N3
  // No refractory periods - simplified model
  y_N3 : [0..1] init 0;  // spike output
  p_N3 : [P_MIN_N3..P_MAX_N3] init 0;  // membrane potential

  // Normal period - firing on newPotential (4 levels)
  [tick] newPotential_N3 <= threshold1 -> (y_N3' = 0) & (p_N3' = newPotential_N3);
  [tick] newPotential_N3 > threshold1 & newPotential_N3 <= threshold2 -> 0.7500:(y_N3' = 0) & (p_N3' = newPotential_N3) + 0.2500:(y_N3' = 1) & (p_N3' = P_reset);
  [tick] newPotential_N3 > threshold2 & newPotential_N3 <= threshold3 -> 0.5000:(y_N3' = 0) & (p_N3' = newPotential_N3) + 0.5000:(y_N3' = 1) & (p_N3' = P_reset);
  [tick] newPotential_N3 > threshold3 & newPotential_N3 <= threshold4 -> 0.2500:(y_N3' = 0) & (p_N3' = newPotential_N3) + 0.7500:(y_N3' = 1) & (p_N3' = P_reset);
  [tick] newPotential_N3 > threshold4 -> 1.0:(y_N3' = 1) & (p_N3' = P_reset);

endmodule

module N4
  // No refractory periods - simplified model
  y_N4 : [0..1] init 0;  // spike output
  p_N4 : [P_MIN_N4..P_MAX_N4] init 0;  // membrane potential

  // Normal period - firing on newPotential (4 levels)
  [tick] newPotential_N4 <= threshold1 -> (y_N4' = 0) & (p_N4' = newPotential_N4);
  [tick] newPotential_N4 > threshold1 & newPotential_N4 <= threshold2 -> 0.7500:(y_N4' = 0) & (p_N4' = newPotential_N4) + 0.2500:(y_N4' = 1) & (p_N4' = P_reset);
  [tick] newPotential_N4 > threshold2 & newPotential_N4 <= threshold3 -> 0.5000:(y_N4' = 0) & (p_N4' = newPotential_N4) + 0.5000:(y_N4' = 1) & (p_N4' = P_reset);
  [tick] newPotential_N4 > threshold3 & newPotential_N4 <= threshold4 -> 0.2500:(y_N4' = 0) & (p_N4' = newPotential_N4) + 0.7500:(y_N4' = 1) & (p_N4' = P_reset);
  [tick] newPotential_N4 > threshold4 -> 1.0:(y_N4' = 1) & (p_N4' = P_reset);

endmodule


// Spike count rewards
rewards "spike_N1_count"
  y_N1 = 1 : 1;
endrewards

rewards "spike_N2_count"
  y_N2 = 1 : 1;
endrewards

rewards "spike_N3_count"
  y_N3 = 1 : 1;
endrewards

rewards "spike_N4_count"
  y_N4 = 1 : 1;
endrewards

// Labels for PCTL properties
label "spike_N1" = (y_N1 = 1);
label "spike_N2" = (y_N2 = 1);
label "spike_N3" = (y_N3 = 1);
label "spike_N4" = (y_N4 = 1);
label "output_spike" = (y_N4 = 1);
