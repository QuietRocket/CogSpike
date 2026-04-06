// Auto-generated DISCRETIZED PRISM model from CogSpike
// Weight discretization: WL=4, T_d=4, lambda_d=-2
// Neurons: 12, Edges: 20, Threshold levels: 4
// Preserves ALL PCTL properties (paper section 7)
dtmc

// Discretization parameters (paper sections 3-4)
const int WL = 4;       // Weight discretization levels
const int T_d = 4;      // Discretized threshold (paper section 3.2)
const int lambda_d = -2; // Additive leak factor (paper section 4.2)
const int K = 4;        // Number of threshold levels

// Per-neuron potential bounds (paper section 7)
const int P_MAX_N1 = 8;  // T_d + max_excitatory_input
const int P_MAX_N2 = 8;  // T_d + max_excitatory_input
const int P_MAX_N3 = 8;  // T_d + max_excitatory_input
const int P_MAX_O1 = 8;  // T_d + max_excitatory_input
const int P_MAX_O2 = 8;  // T_d + max_excitatory_input
const int P_MAX_O3 = 8;  // T_d + max_excitatory_input
const int P_MAX_N4 = 8;  // T_d + max_excitatory_input
const int P_MAX_O4 = 8;  // T_d + max_excitatory_input
const int T_MAX = 100;

// Discretized synaptic weights (paper section 3)
const int W_S1_N1 = 4;  // delta_4(100)
const int W_S2_N2 = 4;  // delta_4(100)
const int W_S3_N3 = 4;  // delta_4(100)
const int W_N1_O1 = 4;  // delta_4(100)
const int W_N2_O2 = 4;  // delta_4(100)
const int W_N3_O3 = 4;  // delta_4(100)
const int W_N1_N2 = -4;  // delta_4(-100)
const int W_N2_N1 = -3;  // delta_4(-70)
const int W_N2_N3 = -3;  // delta_4(-70)
const int W_N3_N2 = -3;  // delta_4(-70)
const int W_N1_N3 = -4;  // delta_4(-100)
const int W_N3_N1 = -3;  // delta_4(-70)
const int W_N4_N2 = -3;  // delta_4(-70)
const int W_N2_N4 = -3;  // delta_4(-70)
const int W_N4_N3 = -3;  // delta_4(-70)
const int W_N3_N4 = -3;  // delta_4(-70)
const int W_N4_N1 = -3;  // delta_4(-70)
const int W_N1_N4 = -4;  // delta_4(-100)
const int W_S4_N4 = 4;  // delta_4(100)
const int W_N4_O4 = 4;  // delta_4(100)

// Contribution formulas (paper section 4.1)
formula contrib_N1 = W_S1_N1 * x_S1 + W_N2_N1 * y_N2 + W_N3_N1 * y_N3 + W_N4_N1 * y_N4;
formula contrib_N2 = W_S2_N2 * x_S2 + W_N1_N2 * y_N1 + W_N3_N2 * y_N3 + W_N4_N2 * y_N4;
formula contrib_N3 = W_S3_N3 * x_S3 + W_N2_N3 * y_N2 + W_N1_N3 * y_N1 + W_N4_N3 * y_N4;
formula contrib_O1 = W_N1_O1 * y_N1;
formula contrib_O2 = W_N2_O2 * y_N2;
formula contrib_O3 = W_N3_O3 * y_N3;
formula contrib_N4 = W_N2_N4 * y_N2 + W_N3_N4 * y_N3 + W_N1_N4 * y_N1 + W_S4_N4 * x_S4;
formula contrib_O4 = W_N4_O4 * y_N4;

// Potential update with additive leak (paper section 4.2)
formula newP_N1 = max(0, min(P_MAX_N1, p_N1 + contrib_N1 + lambda_d));
formula newP_N2 = max(0, min(P_MAX_N2, p_N2 + contrib_N2 + lambda_d));
formula newP_N3 = max(0, min(P_MAX_N3, p_N3 + contrib_N3 + lambda_d));
formula newP_O1 = max(0, min(P_MAX_O1, p_O1 + contrib_O1 + lambda_d));
formula newP_O2 = max(0, min(P_MAX_O2, p_O2 + contrib_O2 + lambda_d));
formula newP_O3 = max(0, min(P_MAX_O3, p_O3 + contrib_O3 + lambda_d));
formula newP_N4 = max(0, min(P_MAX_N4, p_N4 + contrib_N4 + lambda_d));
formula newP_O4 = max(0, min(P_MAX_O4, p_O4 + contrib_O4 + lambda_d));

// Input module
module Inputs
  x_S1 : [0..1] init 0;
  x_S2 : [0..1] init 0;
  x_S3 : [0..1] init 0;
  x_S4 : [0..1] init 0;

  [tick] true -> (x_S1' = 1);
  [tick] true -> (x_S2' = 1);
  [tick] true -> (x_S3' = 1);
  [tick] true -> (x_S4' = 1);
endmodule

module N1
  y_N1 : [0..1] init 0;
  p_N1 : [0..P_MAX_N1] init 0;

  [tick] newP_N1 <= 1 -> (y_N1' = 0) & (p_N1' = newP_N1);
  [tick] newP_N1 > 1 & newP_N1 <= 2 -> 0.250000:(y_N1' = 1) & (p_N1' = 0) + 0.750000:(y_N1' = 0) & (p_N1' = newP_N1);
  [tick] newP_N1 > 2 & newP_N1 <= 3 -> 0.500000:(y_N1' = 1) & (p_N1' = 0) + 0.500000:(y_N1' = 0) & (p_N1' = newP_N1);
  [tick] newP_N1 > 3 & newP_N1 <= 4 -> 0.750000:(y_N1' = 1) & (p_N1' = 0) + 0.250000:(y_N1' = 0) & (p_N1' = newP_N1);
  [tick] newP_N1 > 4 -> 1.0:(y_N1' = 1) & (p_N1' = 0);
endmodule

module N2
  y_N2 : [0..1] init 0;
  p_N2 : [0..P_MAX_N2] init 0;

  [tick] newP_N2 <= 1 -> (y_N2' = 0) & (p_N2' = newP_N2);
  [tick] newP_N2 > 1 & newP_N2 <= 2 -> 0.250000:(y_N2' = 1) & (p_N2' = 0) + 0.750000:(y_N2' = 0) & (p_N2' = newP_N2);
  [tick] newP_N2 > 2 & newP_N2 <= 3 -> 0.500000:(y_N2' = 1) & (p_N2' = 0) + 0.500000:(y_N2' = 0) & (p_N2' = newP_N2);
  [tick] newP_N2 > 3 & newP_N2 <= 4 -> 0.750000:(y_N2' = 1) & (p_N2' = 0) + 0.250000:(y_N2' = 0) & (p_N2' = newP_N2);
  [tick] newP_N2 > 4 -> 1.0:(y_N2' = 1) & (p_N2' = 0);
endmodule

module N3
  y_N3 : [0..1] init 0;
  p_N3 : [0..P_MAX_N3] init 0;

  [tick] newP_N3 <= 1 -> (y_N3' = 0) & (p_N3' = newP_N3);
  [tick] newP_N3 > 1 & newP_N3 <= 2 -> 0.250000:(y_N3' = 1) & (p_N3' = 0) + 0.750000:(y_N3' = 0) & (p_N3' = newP_N3);
  [tick] newP_N3 > 2 & newP_N3 <= 3 -> 0.500000:(y_N3' = 1) & (p_N3' = 0) + 0.500000:(y_N3' = 0) & (p_N3' = newP_N3);
  [tick] newP_N3 > 3 & newP_N3 <= 4 -> 0.750000:(y_N3' = 1) & (p_N3' = 0) + 0.250000:(y_N3' = 0) & (p_N3' = newP_N3);
  [tick] newP_N3 > 4 -> 1.0:(y_N3' = 1) & (p_N3' = 0);
endmodule

module O1
  y_O1 : [0..1] init 0;
  p_O1 : [0..P_MAX_O1] init 0;

  [tick] newP_O1 <= 1 -> (y_O1' = 0) & (p_O1' = newP_O1);
  [tick] newP_O1 > 1 & newP_O1 <= 2 -> 0.250000:(y_O1' = 1) & (p_O1' = 0) + 0.750000:(y_O1' = 0) & (p_O1' = newP_O1);
  [tick] newP_O1 > 2 & newP_O1 <= 3 -> 0.500000:(y_O1' = 1) & (p_O1' = 0) + 0.500000:(y_O1' = 0) & (p_O1' = newP_O1);
  [tick] newP_O1 > 3 & newP_O1 <= 4 -> 0.750000:(y_O1' = 1) & (p_O1' = 0) + 0.250000:(y_O1' = 0) & (p_O1' = newP_O1);
  [tick] newP_O1 > 4 -> 1.0:(y_O1' = 1) & (p_O1' = 0);
endmodule

module O2
  y_O2 : [0..1] init 0;
  p_O2 : [0..P_MAX_O2] init 0;

  [tick] newP_O2 <= 1 -> (y_O2' = 0) & (p_O2' = newP_O2);
  [tick] newP_O2 > 1 & newP_O2 <= 2 -> 0.250000:(y_O2' = 1) & (p_O2' = 0) + 0.750000:(y_O2' = 0) & (p_O2' = newP_O2);
  [tick] newP_O2 > 2 & newP_O2 <= 3 -> 0.500000:(y_O2' = 1) & (p_O2' = 0) + 0.500000:(y_O2' = 0) & (p_O2' = newP_O2);
  [tick] newP_O2 > 3 & newP_O2 <= 4 -> 0.750000:(y_O2' = 1) & (p_O2' = 0) + 0.250000:(y_O2' = 0) & (p_O2' = newP_O2);
  [tick] newP_O2 > 4 -> 1.0:(y_O2' = 1) & (p_O2' = 0);
endmodule

module O3
  y_O3 : [0..1] init 0;
  p_O3 : [0..P_MAX_O3] init 0;

  [tick] newP_O3 <= 1 -> (y_O3' = 0) & (p_O3' = newP_O3);
  [tick] newP_O3 > 1 & newP_O3 <= 2 -> 0.250000:(y_O3' = 1) & (p_O3' = 0) + 0.750000:(y_O3' = 0) & (p_O3' = newP_O3);
  [tick] newP_O3 > 2 & newP_O3 <= 3 -> 0.500000:(y_O3' = 1) & (p_O3' = 0) + 0.500000:(y_O3' = 0) & (p_O3' = newP_O3);
  [tick] newP_O3 > 3 & newP_O3 <= 4 -> 0.750000:(y_O3' = 1) & (p_O3' = 0) + 0.250000:(y_O3' = 0) & (p_O3' = newP_O3);
  [tick] newP_O3 > 4 -> 1.0:(y_O3' = 1) & (p_O3' = 0);
endmodule

module N4
  y_N4 : [0..1] init 0;
  p_N4 : [0..P_MAX_N4] init 0;

  [tick] newP_N4 <= 1 -> (y_N4' = 0) & (p_N4' = newP_N4);
  [tick] newP_N4 > 1 & newP_N4 <= 2 -> 0.250000:(y_N4' = 1) & (p_N4' = 0) + 0.750000:(y_N4' = 0) & (p_N4' = newP_N4);
  [tick] newP_N4 > 2 & newP_N4 <= 3 -> 0.500000:(y_N4' = 1) & (p_N4' = 0) + 0.500000:(y_N4' = 0) & (p_N4' = newP_N4);
  [tick] newP_N4 > 3 & newP_N4 <= 4 -> 0.750000:(y_N4' = 1) & (p_N4' = 0) + 0.250000:(y_N4' = 0) & (p_N4' = newP_N4);
  [tick] newP_N4 > 4 -> 1.0:(y_N4' = 1) & (p_N4' = 0);
endmodule

module O4
  y_O4 : [0..1] init 0;
  p_O4 : [0..P_MAX_O4] init 0;

  [tick] newP_O4 <= 1 -> (y_O4' = 0) & (p_O4' = newP_O4);
  [tick] newP_O4 > 1 & newP_O4 <= 2 -> 0.250000:(y_O4' = 1) & (p_O4' = 0) + 0.750000:(y_O4' = 0) & (p_O4' = newP_O4);
  [tick] newP_O4 > 2 & newP_O4 <= 3 -> 0.500000:(y_O4' = 1) & (p_O4' = 0) + 0.500000:(y_O4' = 0) & (p_O4' = newP_O4);
  [tick] newP_O4 > 3 & newP_O4 <= 4 -> 0.750000:(y_O4' = 1) & (p_O4' = 0) + 0.250000:(y_O4' = 0) & (p_O4' = newP_O4);
  [tick] newP_O4 > 4 -> 1.0:(y_O4' = 1) & (p_O4' = 0);
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
rewards "spike_N4_count"
  y_N4 = 1 : 1;
endrewards
rewards "spike_O4_count"
  y_O4 = 1 : 1;
endrewards

// Labels for PCTL properties
label "spike_N1" = (y_N1 = 1);
label "spike_N2" = (y_N2 = 1);
label "spike_N3" = (y_N3 = 1);
label "spike_O1" = (y_O1 = 1);
label "spike_O2" = (y_O2 = 1);
label "spike_O3" = (y_O3 = 1);
label "spike_N4" = (y_N4 = 1);
label "spike_O4" = (y_O4 = 1);
label "output_spike" = (y_O1 = 1 | y_O2 = 1 | y_O3 = 1 | y_O4 = 1);
