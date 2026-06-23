// Auto-generated PRISM model from CogSpike
// Neurons: 9, Edges: 12
//
// Node name mapping (GUI label -> PRISM variables):
//             S1 [Input]  ->  x_S1
//             S2 [Input]  ->  x_S2
//             S3 [Input]  ->  x_S3
//             N1 [Neuron]  ->  y_N1, p_N1
//             N2 [Neuron]  ->  y_N2, p_N2
//             N3 [Neuron]  ->  y_N3, p_N3
//             O1 [Output]  ->  y_O1, p_O1
//             O2 [Output]  ->  y_O2, p_O2
//             O3 [Output]  ->  y_O3, p_O3
//
dtmc

// Global neuron parameters
const int P_rth = 80;
const int P_rest = 0;
const int P_reset = 0;
const double r = 0.5;
const int P_MIN = -200;
const int P_MAX = 200;

// Per-neuron potential bounds (optimized for state space)
const int P_MIN_N1 = -210;
const int P_MAX_N1 = 150;
const int P_MIN_N2 = -255;
const int P_MAX_N2 = 150;
const int P_MIN_N3 = -255;
const int P_MAX_N3 = 150;
const int P_MIN_O1 = 0;
const int P_MAX_O1 = 150;
const int P_MIN_O2 = 0;
const int P_MAX_O2 = 150;
const int P_MIN_O3 = 0;
const int P_MAX_O3 = 150;
const int T_MAX = 100;

// Firing probability thresholds (4 levels)
// Isomorphic with simulation: threshold_i = (i * P_rth) / levels
formula threshold1 = 20;
formula threshold2 = 40;
formula threshold3 = 60;
formula threshold4 = 80;

// Synaptic weights
const int w_S1_N1 = 100;
const int w_S2_N2 = 100;
const int w_S3_N3 = 100;
const int w_N1_O1 = 100;
const int w_N2_O2 = 100;
const int w_N3_O3 = 100;
const int w_N1_N2 = -100;
const int w_N1_N3 = -100;
const int w_N2_N1 = -70;
const int w_N2_N3 = -70;
const int w_N3_N1 = -70;
const int w_N3_N2 = -70;

// Spike propagation: neurons read y(source) directly (1-tick delay)

// Membrane potential update formulas (using per-neuron bounds)
formula newPotential_N1 = max(P_MIN_N1, min(P_MAX_N1, floor((w_S1_N1 * x_S1 + w_N2_N1 * y_N2 + w_N3_N1 * y_N3) + r * p_N1)));
formula newPotential_N2 = max(P_MIN_N2, min(P_MAX_N2, floor((w_S2_N2 * x_S2 + w_N1_N2 * y_N1 + w_N3_N2 * y_N3) + r * p_N2)));
formula newPotential_N3 = max(P_MIN_N3, min(P_MAX_N3, floor((w_S3_N3 * x_S3 + w_N1_N3 * y_N1 + w_N2_N3 * y_N2) + r * p_N3)));
formula newPotential_O1 = max(P_MIN_O1, min(P_MAX_O1, floor((w_N1_O1 * y_N1) + r * p_O1)));
formula newPotential_O2 = max(P_MIN_O2, min(P_MAX_O2, floor((w_N2_O2 * y_N2) + r * p_O2)));
formula newPotential_O3 = max(P_MIN_O3, min(P_MAX_O3, floor((w_N3_O3 * y_N3) + r * p_O3)));

// Input generator formulas
formula in_S1_g0_fires = true;
formula in_S2_g0_fires = true;
formula in_S3_g0_fires = true;

module Inputs
  x_S1 : [0..1] init 0;
  x_S2 : [0..1] init 0;
  x_S3 : [0..1] init 0;

  // Multi-generator input transitions
  [tick] in_S1_g0_fires -> (x_S1' = 1);
  [tick] !(in_S1_g0_fires) -> (x_S1' = 0);
  [tick] in_S2_g0_fires -> (x_S2' = 1);
  [tick] !(in_S2_g0_fires) -> (x_S2' = 0);
  [tick] in_S3_g0_fires -> (x_S3' = 1);
  [tick] !(in_S3_g0_fires) -> (x_S3' = 0);
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

module O1
  // No refractory periods - simplified model
  y_O1 : [0..1] init 0;  // spike output
  p_O1 : [P_MIN_O1..P_MAX_O1] init 0;  // membrane potential

  // Normal period - firing on newPotential (4 levels)
  [tick] newPotential_O1 <= threshold1 -> (y_O1' = 0) & (p_O1' = newPotential_O1);
  [tick] newPotential_O1 > threshold1 & newPotential_O1 <= threshold2 -> 0.7500:(y_O1' = 0) & (p_O1' = newPotential_O1) + 0.2500:(y_O1' = 1) & (p_O1' = P_reset);
  [tick] newPotential_O1 > threshold2 & newPotential_O1 <= threshold3 -> 0.5000:(y_O1' = 0) & (p_O1' = newPotential_O1) + 0.5000:(y_O1' = 1) & (p_O1' = P_reset);
  [tick] newPotential_O1 > threshold3 & newPotential_O1 <= threshold4 -> 0.2500:(y_O1' = 0) & (p_O1' = newPotential_O1) + 0.7500:(y_O1' = 1) & (p_O1' = P_reset);
  [tick] newPotential_O1 > threshold4 -> 1.0:(y_O1' = 1) & (p_O1' = P_reset);

endmodule

module O2
  // No refractory periods - simplified model
  y_O2 : [0..1] init 0;  // spike output
  p_O2 : [P_MIN_O2..P_MAX_O2] init 0;  // membrane potential

  // Normal period - firing on newPotential (4 levels)
  [tick] newPotential_O2 <= threshold1 -> (y_O2' = 0) & (p_O2' = newPotential_O2);
  [tick] newPotential_O2 > threshold1 & newPotential_O2 <= threshold2 -> 0.7500:(y_O2' = 0) & (p_O2' = newPotential_O2) + 0.2500:(y_O2' = 1) & (p_O2' = P_reset);
  [tick] newPotential_O2 > threshold2 & newPotential_O2 <= threshold3 -> 0.5000:(y_O2' = 0) & (p_O2' = newPotential_O2) + 0.5000:(y_O2' = 1) & (p_O2' = P_reset);
  [tick] newPotential_O2 > threshold3 & newPotential_O2 <= threshold4 -> 0.2500:(y_O2' = 0) & (p_O2' = newPotential_O2) + 0.7500:(y_O2' = 1) & (p_O2' = P_reset);
  [tick] newPotential_O2 > threshold4 -> 1.0:(y_O2' = 1) & (p_O2' = P_reset);

endmodule

module O3
  // No refractory periods - simplified model
  y_O3 : [0..1] init 0;  // spike output
  p_O3 : [P_MIN_O3..P_MAX_O3] init 0;  // membrane potential

  // Normal period - firing on newPotential (4 levels)
  [tick] newPotential_O3 <= threshold1 -> (y_O3' = 0) & (p_O3' = newPotential_O3);
  [tick] newPotential_O3 > threshold1 & newPotential_O3 <= threshold2 -> 0.7500:(y_O3' = 0) & (p_O3' = newPotential_O3) + 0.2500:(y_O3' = 1) & (p_O3' = P_reset);
  [tick] newPotential_O3 > threshold2 & newPotential_O3 <= threshold3 -> 0.5000:(y_O3' = 0) & (p_O3' = newPotential_O3) + 0.5000:(y_O3' = 1) & (p_O3' = P_reset);
  [tick] newPotential_O3 > threshold3 & newPotential_O3 <= threshold4 -> 0.2500:(y_O3' = 0) & (p_O3' = newPotential_O3) + 0.7500:(y_O3' = 1) & (p_O3' = P_reset);
  [tick] newPotential_O3 > threshold4 -> 1.0:(y_O3' = 1) & (p_O3' = P_reset);

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

rewards "spike_O1_count"
  y_O1 = 1 : 1;
endrewards

rewards "spike_O2_count"
  y_O2 = 1 : 1;
endrewards

rewards "spike_O3_count"
  y_O3 = 1 : 1;
endrewards

// Labels for PCTL properties
label "spike_N1" = (y_N1 = 1);
label "spike_N2" = (y_N2 = 1);
label "spike_N3" = (y_N3 = 1);
label "spike_O1" = (y_O1 = 1);
label "spike_O2" = (y_O2 = 1);
label "spike_O3" = (y_O3 = 1);
label "output_spike" = (y_O1 = 1 | y_O2 = 1 | y_O3 = 1);
