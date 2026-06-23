// Auto-generated DISCRETIZED PRISM model from CogSpike
// Weight discretization: WL=6, T_d=6, r=0.95
// Neurons: 3, Edges: 2, Threshold levels: 4
// Preserves ALL PCTL properties (paper section 7)
dtmc

// Discretization parameters (paper sections 3-4)
const int WL = 6;       // Weight discretization levels
const int T_d = 6;      // Discretized threshold (paper section 3.2)
const double r = 0.95;     // Retention rate — multiplicative leak (paper section 4.2)
const int K = 4;        // Number of threshold levels

// Per-neuron potential bounds (paper section 7)
const int P_MAX_N1 = 11;  // T_d + max_excitatory_input
const int P_MIN_N1 = 0;  // sum of inhibitory inputs
const int P_MAX_N2 = 11;  // T_d + max_excitatory_input
const int P_MIN_N2 = 0;  // sum of inhibitory inputs
const int T_MAX = 100;

// Discretized synaptic weights (paper section 3)
const int W_In_N1 = 5;  // delta_6(80)
const int W_N1_N2 = 5;  // delta_6(80)

// Contribution formulas (paper section 4.1)
// contrib_n = sum of (discretized_weight * spike_variable)
formula contrib_N1 = W_In_N1 * x_In;
formula contrib_N2 = W_N1_N2 * y_N1;

// Potential update with multiplicative leak (isomorphic with simulation engine)
// newP_n = max(P_MIN_n, min(P_MAX_n, floor(r * p_n) + contrib_n))
formula newP_N1 = max(P_MIN_N1, min(P_MAX_N1, floor(r * p_N1) + contrib_N1));
formula newP_N2 = max(P_MIN_N2, min(P_MAX_N2, floor(r * p_N2) + contrib_N2));


// Feasibility analysis (paper section 5)
// N1: FEASIBLE (multi-step, min 2 steps)
// N2: FEASIBLE (multi-step, min 2 steps)

// Input module
module Inputs
  x_In : [0..1] init 0;

  [tick] true -> (x_In' = 1);
endmodule

module N1
  y_N1 : [0..1] init 0;  // spike output
  p_N1 : [P_MIN_N1..P_MAX_N1] init 0;  // membrane potential (discretized domain)

  // Normal period - firing on newP (4 levels, no reset tick)
  // Level 0: newP_N1 <= 1 -> no fire
  [tick] newP_N1 <= 1 -> (y_N1' = 0) & (p_N1' = newP_N1);
  // Level 1: 1 < newP_N1 <= 3 -> fire P=0.25
  [tick] newP_N1 > 1 & newP_N1 <= 3 -> 0.250000:(y_N1' = 1) & (p_N1' = 0) + 0.750000:(y_N1' = 0) & (p_N1' = newP_N1);
  // Level 2: 3 < newP_N1 <= 4 -> fire P=0.50
  [tick] newP_N1 > 3 & newP_N1 <= 4 -> 0.500000:(y_N1' = 1) & (p_N1' = 0) + 0.500000:(y_N1' = 0) & (p_N1' = newP_N1);
  // Level 3: 4 < newP_N1 <= 6 -> fire P=0.75
  [tick] newP_N1 > 4 & newP_N1 <= 6 -> 0.750000:(y_N1' = 1) & (p_N1' = 0) + 0.250000:(y_N1' = 0) & (p_N1' = newP_N1);
  // Level 4: newP_N1 > 6 -> certain fire
  [tick] newP_N1 > 6 -> 1.0:(y_N1' = 1) & (p_N1' = 0);

endmodule

module N2
  y_N2 : [0..1] init 0;  // spike output
  p_N2 : [P_MIN_N2..P_MAX_N2] init 0;  // membrane potential (discretized domain)

  // Normal period - firing on newP (4 levels, no reset tick)
  // Level 0: newP_N2 <= 1 -> no fire
  [tick] newP_N2 <= 1 -> (y_N2' = 0) & (p_N2' = newP_N2);
  // Level 1: 1 < newP_N2 <= 3 -> fire P=0.25
  [tick] newP_N2 > 1 & newP_N2 <= 3 -> 0.250000:(y_N2' = 1) & (p_N2' = 0) + 0.750000:(y_N2' = 0) & (p_N2' = newP_N2);
  // Level 2: 3 < newP_N2 <= 4 -> fire P=0.50
  [tick] newP_N2 > 3 & newP_N2 <= 4 -> 0.500000:(y_N2' = 1) & (p_N2' = 0) + 0.500000:(y_N2' = 0) & (p_N2' = newP_N2);
  // Level 3: 4 < newP_N2 <= 6 -> fire P=0.75
  [tick] newP_N2 > 4 & newP_N2 <= 6 -> 0.750000:(y_N2' = 1) & (p_N2' = 0) + 0.250000:(y_N2' = 0) & (p_N2' = newP_N2);
  // Level 4: newP_N2 > 6 -> certain fire
  [tick] newP_N2 > 6 -> 1.0:(y_N2' = 1) & (p_N2' = 0);

endmodule


// Spike count rewards
rewards "spike_N1_count"
  y_N1 = 1 : 1;
endrewards

rewards "spike_N2_count"
  y_N2 = 1 : 1;
endrewards

// Labels for PCTL properties
label "spike_N1" = (y_N1 = 1);
label "spike_N2" = (y_N2 = 1);
label "output_spike" = (y_N2 = 1);
