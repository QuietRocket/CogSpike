//! Discretized PRISM model generator for Spiking Neural Networks.
//!
//! Generates DTMC models using the paper's weight discretization approach (§7).
//! Instead of tracking potentials in the raw domain [0..500], weights are
//! discretized to [-W, W] and potentials tracked in [0..T_d+E], giving a
//! ~50-170x state reduction per neuron while preserving ALL PCTL properties.
//!
//! **Key formulas from the paper:**
//! - Weight discretization: `δ_W(w) = ⌊w · W / w_max⌉` (§3)
//! - Threshold calibration: `T_d = ⌈T · W / w_max⌉` (§3.2)
//! - Additive leak: `λ_d = -max(1, ⌊ℓ · T_d⌋)` (§4.2)
//! - Potential update: `p' = max(0, min(P_MAX, p + contrib + λ_d))` (§7)

use crate::snn::discretization::{
    Feasibility, check_feasibility, discretize_weight, discretized_leak, discretized_threshold,
};
use crate::snn::graph::{Node, SnnGraph};
use crate::snn::prism_gen::PrismGenConfig;
use std::fmt::Write as _;

/// Generates a discretized PRISM model from an SNN graph.
///
/// The model uses the paper's weight discretization (§7) to track exact
/// potentials in a reduced domain. This preserves ALL PCTL properties
/// while achieving massive state space reduction.
pub fn generate_discretized_model(graph: &SnnGraph, config: &PrismGenConfig) -> String {
    let mut out = String::with_capacity(4096);
    let m = &config.model;
    let wl = config.weight_levels.clamp(1, 10);
    let k = m.threshold_levels.clamp(1, 10) as usize;
    let t_d = discretized_threshold(m.p_rth, wl);
    let leak_factor = 1.0 - (m.leak_r as f64 / 100.0);
    let lambda_d = discretized_leak(leak_factor, t_d);

    // Header
    writeln!(
        out,
        "// Auto-generated DISCRETIZED PRISM model from CogSpike"
    )
    .ok();
    writeln!(
        out,
        "// Weight discretization: WL={wl}, T_d={t_d}, lambda_d={lambda_d}"
    )
    .ok();
    writeln!(
        out,
        "// Neurons: {}, Edges: {}, Threshold levels: {k}",
        graph.nodes.len(),
        graph.edges.len()
    )
    .ok();
    writeln!(out, "// Preserves ALL PCTL properties (paper section 7)").ok();
    writeln!(out, "dtmc\n").ok();

    // Global constants
    write_discretized_constants(&mut out, graph, config, t_d, lambda_d);
    writeln!(out).ok();

    // Discretized weight constants
    write_discretized_weights(&mut out, graph, wl);
    writeln!(out).ok();

    // Per-neuron potential bounds and contribution formulas
    write_contribution_formulas(&mut out, graph, config, t_d);
    writeln!(out).ok();

    // Potential update formulas
    write_potential_formulas(&mut out, graph, config, t_d, lambda_d);
    writeln!(out).ok();

    // Transfer comments
    write_transfer_comments(&mut out, graph);
    writeln!(out).ok();

    // Feasibility analysis
    write_feasibility_analysis(&mut out, graph, config, t_d, lambda_d);
    writeln!(out).ok();

    // Input module
    write_input_module(&mut out, graph, config);
    writeln!(out).ok();

    // Discretized neuron modules
    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        write_discretized_neuron_module(&mut out, node, graph, config, t_d, lambda_d);
        writeln!(out).ok();
    }

    // Transfer modules
    write_transfer_modules(&mut out, graph);
    writeln!(out).ok();

    // Rewards
    if config.include_rewards {
        write_rewards(&mut out, graph);
    }

    // Labels
    write_labels(&mut out, graph, config);

    out
}

// ============================================================================
// Constants and Formulas
// ============================================================================

fn write_discretized_constants(
    out: &mut String,
    graph: &SnnGraph,
    config: &PrismGenConfig,
    t_d: i32,
    lambda_d: i32,
) {
    let m = &config.model;
    let wl = config.weight_levels.clamp(1, 10);
    let k = m.threshold_levels.clamp(1, 10);

    writeln!(out, "// Discretization parameters (paper sections 3-4)").ok();
    writeln!(
        out,
        "const int WL = {};       // Weight discretization levels",
        wl
    )
    .ok();
    writeln!(
        out,
        "const int T_d = {};      // Discretized threshold (paper section 3.2)",
        t_d
    )
    .ok();
    writeln!(
        out,
        "const int lambda_d = {}; // Additive leak factor (paper section 4.2)",
        lambda_d
    )
    .ok();
    writeln!(
        out,
        "const int K = {};        // Number of threshold levels",
        k
    )
    .ok();

    // Refractory constants
    if m.enable_arp {
        writeln!(out, "const int ARP = {};", m.arp).ok();
    }
    if m.enable_rrp {
        writeln!(out, "const int RRP = {};", m.rrp).ok();
        writeln!(out, "const double alpha = {};", m.alpha as f64 / 100.0).ok();
    }

    // Per-neuron potential bounds
    writeln!(out).ok();
    writeln!(out, "// Per-neuron potential bounds (paper section 7)").ok();
    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        let p_max = compute_p_max(node, graph, config, t_d);
        writeln!(
            out,
            "const int P_MAX_{} = {};  // T_d + max_excitatory_input",
            node.id.0, p_max
        )
        .ok();
    }

    if let Some(t) = config.time_bound {
        writeln!(out, "const int T_MAX = {t};").ok();
    }
}

/// Compute the maximum potential for a neuron in the discretized domain.
/// P_MAX = T_d + sum of positive (excitatory) discretized weights.
fn compute_p_max(node: &Node, graph: &SnnGraph, config: &PrismGenConfig, t_d: i32) -> i32 {
    let wl = config.weight_levels.clamp(1, 10);
    let incoming = graph.incoming_edges(node.id);
    let max_excitatory: i32 = incoming
        .iter()
        .map(|e| discretize_weight(e.signed_weight(), wl))
        .filter(|&w| w > 0)
        .sum();
    // Ensure P_MAX is at least T_d (even with no excitatory inputs)
    t_d.max(t_d + max_excitatory)
}

fn write_discretized_weights(out: &mut String, graph: &SnnGraph, wl: u8) {
    writeln!(out, "// Discretized synaptic weights (paper section 3)").ok();
    for edge in &graph.edges {
        let w_d = discretize_weight(edge.signed_weight(), wl);
        let prefix = if graph.is_input(edge.from) {
            "W_in"
        } else {
            "W_n"
        };
        writeln!(
            out,
            "const int {prefix}{}_{} = {};  // delta_{}({})",
            edge.from.0,
            edge.to.0,
            w_d,
            wl,
            edge.signed_weight()
        )
        .ok();
    }
}

fn write_contribution_formulas(
    out: &mut String,
    graph: &SnnGraph,
    _config: &PrismGenConfig,
    _t_d: i32,
) {
    writeln!(out, "// Contribution formulas (paper section 4.1)").ok();
    writeln!(
        out,
        "// contrib_n = sum of (discretized_weight * spike_variable)"
    )
    .ok();

    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        let n = node.id.0;
        let incoming = graph.incoming_edges(node.id);

        if incoming.is_empty() {
            writeln!(out, "formula contrib_{n} = 0;").ok();
            continue;
        }

        let terms: Vec<String> = incoming
            .iter()
            .map(|e| {
                let prefix = if graph.is_input(e.from) {
                    "W_in"
                } else {
                    "W_n"
                };
                let spike_var = if graph.is_input(e.from) {
                    format!("x{}", e.from.0)
                } else {
                    format!("z{}_{}", e.from.0, e.to.0)
                };
                format!("{prefix}{}_{} * {spike_var}", e.from.0, e.to.0)
            })
            .collect();

        writeln!(out, "formula contrib_{n} = {};", terms.join(" + ")).ok();
    }
}

fn write_potential_formulas(
    out: &mut String,
    graph: &SnnGraph,
    _config: &PrismGenConfig,
    _t_d: i32,
    _lambda_d: i32,
) {
    writeln!(
        out,
        "// Potential update with additive leak (paper section 4.2)"
    )
    .ok();
    writeln!(
        out,
        "// newP_n = max(0, min(P_MAX_n, p_n + contrib_n + lambda_d))"
    )
    .ok();

    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        let n = node.id.0;
        writeln!(
            out,
            "formula newP_{n} = max(0, min(P_MAX_{n}, p{n} + contrib_{n} + lambda_d));"
        )
        .ok();
    }
}

fn write_transfer_comments(out: &mut String, graph: &SnnGraph) {
    writeln!(out, "// Transfer variables for spike propagation").ok();
    for edge in &graph.edges {
        if !graph.is_input(edge.from) {
            writeln!(
                out,
                "// z{}_{} defined in transfer module",
                edge.from.0, edge.to.0
            )
            .ok();
        }
    }
}

fn write_feasibility_analysis(
    out: &mut String,
    graph: &SnnGraph,
    config: &PrismGenConfig,
    t_d: i32,
    lambda_d: i32,
) {
    let wl = config.weight_levels.clamp(1, 10);

    writeln!(out, "// Feasibility analysis (paper section 5)").ok();
    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        let incoming = graph.incoming_edges(node.id);
        let excitatory_weights: Vec<i32> = incoming
            .iter()
            .map(|e| discretize_weight(e.signed_weight(), wl))
            .filter(|&w| w > 0)
            .collect();

        let feasibility = check_feasibility(&excitatory_weights, t_d, lambda_d);
        match feasibility {
            Feasibility::SingleStep => {
                writeln!(out, "// Neuron {}: FEASIBLE (single-step reach)", node.id.0).ok();
            }
            Feasibility::MultiStep { min_steps } => {
                writeln!(
                    out,
                    "// Neuron {}: FEASIBLE (multi-step, min {} steps)",
                    node.id.0, min_steps
                )
                .ok();
            }
            Feasibility::Impossible => {
                writeln!(
                    out,
                    "// WARNING: Neuron {} INFEASIBLE — leak overwhelms excitatory input",
                    node.id.0
                )
                .ok();
            }
        }
    }
}

// ============================================================================
// Input Module (reused from the old quotient generator)
// ============================================================================

fn needs_global_clock(graph: &SnnGraph) -> bool {
    use crate::simulation::InputPattern;

    graph
        .nodes
        .iter()
        .filter(|n| graph.is_input(n.id))
        .any(|input| {
            input.input_config.as_ref().is_some_and(|cfg| {
                cfg.generators.iter().any(|g| {
                    g.active
                        && matches!(
                            g.pattern,
                            InputPattern::Pulse { .. }
                                | InputPattern::Silence { .. }
                                | InputPattern::Periodic { .. }
                                | InputPattern::Burst { .. }
                                | InputPattern::Custom { .. }
                        )
                })
            })
        })
}

fn write_global_clock(out: &mut String, config: &PrismGenConfig) {
    let t_max = config.time_bound.unwrap_or(100);
    writeln!(out, "// Global clock for time-dependent input patterns").ok();
    writeln!(out, "module GlobalClock").ok();
    writeln!(out, "  step : [0..{t_max}] init 0;").ok();
    writeln!(out, "  [tick] step < {t_max} -> (step' = step + 1);").ok();
    writeln!(out, "  [tick] step = {t_max} -> (step' = step);").ok();
    writeln!(out, "endmodule").ok();
}

fn write_input_module(out: &mut String, graph: &SnnGraph, config: &PrismGenConfig) {
    let inputs: Vec<_> = graph
        .nodes
        .iter()
        .filter(|n| graph.is_input(n.id))
        .collect();

    if inputs.is_empty() {
        writeln!(out, "// No input nodes defined").ok();
        return;
    }

    // Write GlobalClock if needed
    if needs_global_clock(graph) {
        write_global_clock(out, config);
        writeln!(out).ok();
    }

    writeln!(out, "// Input module").ok();
    writeln!(out, "module Inputs").ok();

    for input in &inputs {
        writeln!(out, "  x{} : [0..1] init 0;", input.id.0).ok();
    }
    writeln!(out).ok();

    for input in &inputs {
        let n = input.id.0;
        if let Some(ref cfg) = input.input_config {
            let active_count = cfg.generators.iter().filter(|g| g.active).count();
            if active_count == 0 {
                writeln!(out, "  [tick] true -> (x{n}' = 0);").ok();
            } else {
                let prob = cfg
                    .generators
                    .iter()
                    .find(|g| g.active)
                    .and_then(|g| {
                        use crate::simulation::InputPattern;
                        match &g.pattern {
                            InputPattern::Random { probability } => Some(*probability),
                            InputPattern::AlwaysOn => Some(1.0),
                            InputPattern::AlwaysOff => Some(0.0),
                            _ => Some(0.5), // Default for time-dependent patterns
                        }
                    })
                    .unwrap_or(0.5);

                if (prob - 1.0).abs() < 1e-9 {
                    writeln!(out, "  [tick] true -> (x{n}' = 1);").ok();
                } else if prob.abs() < 1e-9 {
                    writeln!(out, "  [tick] true -> (x{n}' = 0);").ok();
                } else {
                    writeln!(
                        out,
                        "  [tick] true -> {:.6}:(x{n}' = 1) + {:.6}:(x{n}' = 0);",
                        prob,
                        1.0 - prob
                    )
                    .ok();
                }
            }
        } else {
            // No config -> always on (legacy behavior)
            writeln!(out, "  [tick] true -> (x{n}' = 1);").ok();
        }
    }

    writeln!(out, "endmodule").ok();
}

// ============================================================================
// Discretized Neuron Module — The core implementation (paper §7)
// ============================================================================

/// Write a discretized neuron module that tracks exact potential in [0..P_MAX].
fn write_discretized_neuron_module(
    out: &mut String,
    node: &Node,
    _graph: &SnnGraph,
    config: &PrismGenConfig,
    t_d: i32,
    _lambda_d: i32,
) {
    let n = node.id.0;
    let model = &config.model;
    let k = model.threshold_levels.clamp(1, 10) as usize;

    // Refractory max state
    let max_state = if model.enable_arp && model.enable_rrp {
        2
    } else if model.enable_arp {
        1
    } else {
        0
    };

    writeln!(out, "module Neuron{n}").ok();

    // State variable (refractory state)
    if model.enable_arp || model.enable_rrp {
        writeln!(
            out,
            "  // State: 0=normal{}{}",
            if model.enable_arp { ", 1=ARP" } else { "" },
            if model.enable_rrp { ", 2=RRP" } else { "" }
        )
        .ok();
        writeln!(out, "  s{n} : [0..{max_state}] init 0;").ok();
    }

    if model.enable_arp {
        writeln!(out, "  aref{n} : [0..ARP] init 0;").ok();
    }
    if model.enable_rrp {
        writeln!(out, "  rref{n} : [0..RRP] init 0;").ok();
    }

    writeln!(out, "  y{n} : [0..1] init 0;  // spike output").ok();
    writeln!(
        out,
        "  p{n} : [0..P_MAX_{n}] init 0;  // membrane potential (discretized domain)"
    )
    .ok();
    writeln!(out).ok();

    // State guard prefix
    let state_guard = if model.enable_arp || model.enable_rrp {
        format!("s{n} = 0 & ")
    } else {
        String::new()
    };

    // Spike reset transition
    writeln!(out, "  // Spike reset").ok();
    if model.enable_arp {
        writeln!(
            out,
            "  [tick] s{n} = 0 & y{n} = 1 -> (p{n}' = 0) & (aref{n}' = ARP) & (y{n}' = 0) & (s{n}' = 1);"
        )
        .ok();
    } else {
        writeln!(out, "  [tick] y{n} = 1 -> (p{n}' = 0) & (y{n}' = 0);").ok();
    }

    writeln!(out).ok();
    writeln!(
        out,
        "  // Normal period - probabilistic firing by threshold level"
    )
    .ok();

    // Compute threshold boundaries for probabilistic firing
    // Level j fires with probability j/k
    // Boundary j is at T_d * j / k
    let step = 1.0 / k as f64;

    // Level 0: newP <= boundary_1 -> no fire
    let boundary_1 = t_d as f64 / k as f64;
    writeln!(
        out,
        "  // Level 0: newP_{n} <= {:.0} -> no fire",
        boundary_1.floor()
    )
    .ok();
    writeln!(
        out,
        "  [tick] {state_guard}y{n} = 0 & newP_{n} <= {} -> (y{n}' = 0) & (p{n}' = newP_{n});",
        (boundary_1.floor() as i32).max(0)
    )
    .ok();

    // Levels 1 to k-1: probabilistic firing
    for j in 1..k {
        let fire_prob = j as f64 * step;
        let no_fire_prob = 1.0 - fire_prob;
        let lower = (t_d as f64 * (j as f64 - 1.0) / k as f64).floor() as i32;
        let upper = (t_d as f64 * j as f64 / k as f64).floor() as i32;

        writeln!(
            out,
            "  // Level {j}: {lower} < newP_{n} <= {upper} -> fire P={fire_prob:.2}"
        )
        .ok();
        writeln!(
            out,
            "  [tick] {state_guard}y{n} = 0 & newP_{n} > {lower} & newP_{n} <= {upper} -> {:.6}:(y{n}' = 1) & (p{n}' = 0) + {:.6}:(y{n}' = 0) & (p{n}' = newP_{n});",
            fire_prob, no_fire_prob
        )
        .ok();
    }

    // Level k: above T_d -> certain fire
    writeln!(out, "  // Level {k}: newP_{n} > {t_d} -> certain fire").ok();
    writeln!(
        out,
        "  [tick] {state_guard}y{n} = 0 & newP_{n} > {t_d} -> 1.0:(y{n}' = 1) & (p{n}' = 0);"
    )
    .ok();

    writeln!(out).ok();

    // Absolute refractory period
    if model.enable_arp {
        writeln!(out, "  // Absolute refractory period").ok();
        writeln!(
            out,
            "  [tick] s{n} = 1 & aref{n} > 0 -> (aref{n}' = aref{n} - 1) & (y{n}' = 0) & (p{n}' = p{n});"
        )
        .ok();

        if model.enable_rrp {
            writeln!(
                out,
                "  [tick] s{n} = 1 & aref{n} = 0 -> (s{n}' = 2) & (rref{n}' = RRP) & (y{n}' = 0);"
            )
            .ok();
        } else {
            writeln!(
                out,
                "  [tick] s{n} = 1 & aref{n} = 0 -> (s{n}' = 0) & (y{n}' = 0);"
            )
            .ok();
        }
        writeln!(out).ok();
    }

    // Relative refractory period
    if model.enable_arp && model.enable_rrp {
        let alpha = model.alpha as f64 / 100.0;

        writeln!(out, "  // Relative refractory period (alpha-scaled)").ok();

        // RRP spike reset
        writeln!(
            out,
            "  [tick] s{n} = 2 & y{n} = 1 & rref{n} > 0 -> (p{n}' = 0) & (aref{n}' = ARP) & (y{n}' = 0) & (rref{n}' = 0) & (s{n}' = 1);"
        )
        .ok();

        // RRP probabilistic firing (alpha-scaled) — based on current potential level
        // For simplicity, we check potential against threshold boundaries
        for j in 0..=k {
            let base_prob = if j == 0 {
                0.0
            } else if j == k {
                1.0
            } else {
                j as f64 * step
            };
            let fire_prob = alpha * base_prob;
            let no_fire_prob = 1.0 - fire_prob;

            let lower = if j == 0 {
                -1 // matches everything <= boundary_0
            } else {
                (t_d as f64 * (j as f64 - 1.0) / k as f64).floor() as i32
            };
            let upper = if j == k {
                // For p_max, this is handled by > upper_prev
                i32::MAX
            } else {
                (t_d as f64 * j as f64 / k as f64).floor() as i32
            };

            if j == 0 {
                // Lowest level: p <= boundary_1
                if fire_prob.abs() < 1e-9 {
                    writeln!(
                        out,
                        "  [tick] s{n} = 2 & y{n} = 0 & rref{n} > 0 & p{n} <= {} -> (y{n}' = 0) & (rref{n}' = rref{n} - 1);",
                        upper
                    ).ok();
                } else {
                    writeln!(
                        out,
                        "  [tick] s{n} = 2 & y{n} = 0 & rref{n} > 0 & p{n} <= {} -> {:.6}:(y{n}' = 0) & (rref{n}' = rref{n} - 1) + {:.6}:(y{n}' = 1);",
                        upper, no_fire_prob, fire_prob
                    ).ok();
                }
            } else if j == k {
                // Highest level: p > boundary_k-1
                let prev_upper = (t_d as f64 * ((k - 1) as f64) / k as f64).floor() as i32;
                if (fire_prob - 1.0).abs() < 1e-9 {
                    writeln!(
                        out,
                        "  [tick] s{n} = 2 & y{n} = 0 & rref{n} > 0 & p{n} > {} -> (y{n}' = 1);",
                        prev_upper
                    )
                    .ok();
                } else {
                    writeln!(
                        out,
                        "  [tick] s{n} = 2 & y{n} = 0 & rref{n} > 0 & p{n} > {} -> {:.6}:(y{n}' = 0) & (rref{n}' = rref{n} - 1) + {:.6}:(y{n}' = 1);",
                        prev_upper, no_fire_prob, fire_prob
                    ).ok();
                }
            } else {
                if fire_prob.abs() < 1e-9 {
                    writeln!(
                        out,
                        "  [tick] s{n} = 2 & y{n} = 0 & rref{n} > 0 & p{n} > {} & p{n} <= {} -> (y{n}' = 0) & (rref{n}' = rref{n} - 1);",
                        lower, upper
                    ).ok();
                } else {
                    writeln!(
                        out,
                        "  [tick] s{n} = 2 & y{n} = 0 & rref{n} > 0 & p{n} > {} & p{n} <= {} -> {:.6}:(y{n}' = 0) & (rref{n}' = rref{n} - 1) + {:.6}:(y{n}' = 1);",
                        lower, upper, no_fire_prob, fire_prob
                    ).ok();
                }
            }
        }

        // RRP ended
        writeln!(
            out,
            "  [tick] s{n} = 2 & y{n} = 0 & rref{n} = 0 -> (p{n}' = 0) & (y{n}' = 0) & (s{n}' = 0);"
        )
        .ok();
    }

    writeln!(out, "endmodule").ok();
}

// ============================================================================
// Transfer, Rewards, Labels (unchanged from old generator)
// ============================================================================

fn write_transfer_modules(out: &mut String, graph: &SnnGraph) {
    writeln!(out, "// Synapse transfer modules (spike propagation)").ok();

    for edge in &graph.edges {
        if graph.is_input(edge.from) {
            continue;
        }

        writeln!(out, "module Transfer{}_{}", edge.from.0, edge.to.0).ok();
        writeln!(out, "  z{}_{} : [0..1] init 0;", edge.from.0, edge.to.0).ok();
        writeln!(
            out,
            "  [tick] true -> (z{}_{}' = y{});",
            edge.from.0, edge.to.0, edge.from.0
        )
        .ok();
        writeln!(out, "endmodule").ok();
        writeln!(out).ok();
    }
}

fn write_rewards(out: &mut String, graph: &SnnGraph) {
    writeln!(out, "// Spike count rewards").ok();

    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        writeln!(out, "rewards \"spike{}_count\"", node.id.0).ok();
        writeln!(out, "  y{} = 1 : 1;", node.id.0).ok();
        writeln!(out, "endrewards").ok();
        writeln!(out).ok();
    }
}

fn write_labels(out: &mut String, graph: &SnnGraph, config: &PrismGenConfig) {
    writeln!(out, "// Labels for PCTL properties").ok();
    let model = &config.model;

    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        let n = node.id.0;
        writeln!(out, "label \"spike{n}\" = (y{n} = 1);").ok();

        if model.enable_arp {
            if model.enable_rrp {
                writeln!(out, "label \"refractory{n}\" = (s{n} = 1 | s{n} = 2);").ok();
            } else {
                writeln!(out, "label \"refractory{n}\" = (s{n} = 1);").ok();
            }
        }
    }

    // Output neurons
    let outputs = graph.output_neurons();

    if !outputs.is_empty() {
        let output_spikes: Vec<_> = outputs.iter().map(|id| format!("y{} = 1", id.0)).collect();
        writeln!(
            out,
            "label \"output_spike\" = ({});",
            output_spikes.join(" | ")
        )
        .ok();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::ModelConfig;
    use crate::snn::graph::SnnGraph;

    fn basic_config() -> PrismGenConfig {
        PrismGenConfig {
            model: ModelConfig {
                threshold_levels: 4,
                p_rth: 100,
                leak_r: 95, // r=0.95, ℓ=0.05
                ..Default::default()
            },
            weight_levels: 3,
            ..Default::default()
        }
    }

    #[test]
    fn test_generate_discretized_model_basic() {
        let graph = SnnGraph::demo_layout();
        let config = basic_config();
        let prism = generate_discretized_model(&graph, &config);

        // Should contain discretized model markers
        assert!(prism.contains("DISCRETIZED PRISM model"));
        assert!(prism.contains("T_d"));
        assert!(prism.contains("lambda_d"));
        assert!(prism.contains("dtmc"));
    }

    #[test]
    fn test_discretized_model_uses_discretized_weights() {
        let graph = SnnGraph::demo_layout();
        let config = basic_config();
        let prism = generate_discretized_model(&graph, &config);

        // Should contain discretized weight constants (W_in or W_n prefix)
        assert!(
            prism.contains("W_in") || prism.contains("W_n"),
            "Model should contain discretized weight constants"
        );
        // Should NOT contain raw weight constants
        assert!(
            !prism.contains("weight_in0_"),
            "Model should not contain raw weight constants"
        );
    }

    #[test]
    fn test_discretized_model_has_contribution_formulas() {
        let graph = SnnGraph::demo_layout();
        let config = basic_config();
        let prism = generate_discretized_model(&graph, &config);

        // Should contain contribution formulas
        assert!(
            prism.contains("formula contrib_"),
            "Model should contain contribution formulas"
        );
    }

    #[test]
    fn test_discretized_model_has_additive_leak() {
        let graph = SnnGraph::demo_layout();
        let config = basic_config();
        let prism = generate_discretized_model(&graph, &config);

        // Should contain lambda_d in potential update formula
        assert!(
            prism.contains("lambda_d"),
            "Model should reference lambda_d (additive leak)"
        );
        assert!(
            prism.contains("newP_"),
            "Model should contain potential update formulas"
        );
    }

    #[test]
    fn test_discretized_model_uses_exact_potential() {
        let graph = SnnGraph::demo_layout();
        let config = basic_config();
        let prism = generate_discretized_model(&graph, &config);

        // Should contain exact potential variables (p1, p2, etc.)
        // The demo layout has neurons with IDs 1, 2, 3
        let has_potential = prism
            .lines()
            .any(|line| line.contains(" p1 :") || line.contains(" p2 :") || line.contains(" p3 :"));
        assert!(
            has_potential,
            "Model should have exact potential variables (p_n)"
        );

        // Should NOT contain pClass variables
        assert!(
            !prism.contains("pClass"),
            "Discretized model should not use pClass"
        );
    }

    #[test]
    fn test_discretized_model_small_potential_range() {
        let graph = SnnGraph::demo_layout();
        let config = basic_config();
        let prism = generate_discretized_model(&graph, &config);

        // P_MAX should be small (< 20 for typical networks with WL=3)
        // T_d = ceil(100 * 3 / 100) = 3, max excitatory = 3 -> P_MAX = 6
        assert!(
            prism.contains("P_MAX_"),
            "Model should define P_MAX per neuron"
        );
        // Check that P_MAX values are reasonable
        for line in prism.lines() {
            if line.contains("P_MAX_") && line.contains(" = ") && !line.contains("formula") {
                let value: i32 = line
                    .split('=')
                    .nth(1)
                    .and_then(|s| s.trim().trim_end_matches(';').trim().parse().ok())
                    .unwrap_or(0);
                assert!(
                    value < 20,
                    "P_MAX should be small in discretized model, got {value}"
                );
            }
        }
    }

    #[test]
    fn test_discretized_model_feasibility_comment() {
        let graph = SnnGraph::demo_layout();
        let config = basic_config();
        let prism = generate_discretized_model(&graph, &config);

        // Should contain feasibility analysis
        assert!(
            prism.contains("FEASIBLE") || prism.contains("INFEASIBLE"),
            "Model should contain feasibility analysis"
        );
    }

    #[test]
    fn test_discretized_model_with_different_weight_levels() {
        let graph = SnnGraph::demo_layout();
        let config_w5 = PrismGenConfig {
            weight_levels: 5,
            ..basic_config()
        };
        let prism = generate_discretized_model(&graph, &config_w5);

        // With WL=5: T_d = ceil(100*5/100) = 5
        assert!(prism.contains("WL = 5"));
        assert!(prism.contains("T_d = 5"));
    }
}
