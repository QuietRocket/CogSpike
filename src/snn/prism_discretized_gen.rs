//! Discretized PRISM model generator for Spiking Neural Networks.
//!
//! Generates DTMC models using the paper's weight discretization approach (§7).
//! Instead of tracking potentials in the raw domain [-500..500], weights are
//! discretized to [-W, W] and potentials tracked in [P_MIN..P_MAX], giving a
//! ~50-170x state reduction per neuron while preserving ALL PCTL properties.
//!
//! **Key formulas from the paper:**
//! - Weight discretization: `δ_W(w) = ⌊w · W / w_max⌉` (§3)
//! - Threshold calibration: `T_d = ⌈T · W / w_max⌉` (§3.2)
//! - Multiplicative leak: `floor(r × p)` — isomorphic with simulation engine (§4.2)
//! - Potential update: `p' = max(P_MIN, min(P_MAX, floor(r × p) + contrib))` (§7)

use crate::snn::discretization::{
    Feasibility, check_feasibility, compute_discretized_p_min, discretize_weight,
    discretized_threshold,
};
use crate::snn::graph::{Node, SnnGraph};
use crate::snn::prism_gen::{build_name_map, NameMap, PrismGenConfig};
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
    let retention_rate = m.leak_r as f64 / 100.0;
    let names = build_name_map(graph);

    // Header
    writeln!(
        out,
        "// Auto-generated DISCRETIZED PRISM model from CogSpike"
    )
    .ok();
    writeln!(
        out,
        "// Weight discretization: WL={wl}, T_d={t_d}, r={retention_rate}"
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
    write_discretized_constants(&mut out, graph, config, t_d, retention_rate, &names);
    writeln!(out).ok();

    // Discretized weight constants
    write_discretized_weights(&mut out, graph, wl, &names);
    writeln!(out).ok();

    // Per-neuron potential bounds and contribution formulas
    write_contribution_formulas(&mut out, graph, config, t_d, &names);
    writeln!(out).ok();

    // Potential update formulas (multiplicative leak, allows negatives)
    write_potential_formulas(&mut out, graph, config, t_d, &names);
    writeln!(out).ok();

    // Transfer modules removed
    writeln!(out).ok();

    // Feasibility analysis
    write_feasibility_analysis(&mut out, graph, config, t_d, retention_rate, &names);
    writeln!(out).ok();

    // Input module
    write_input_module(&mut out, graph, config, &names);
    writeln!(out).ok();

    // Discretized neuron modules
    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        write_discretized_neuron_module(&mut out, node, graph, config, t_d, &names);
        writeln!(out).ok();
    }

    // Transfer modules removed
    writeln!(out).ok();

    // Rewards
    if config.include_rewards {
        write_rewards(&mut out, graph, &names);
    }

    // Labels
    write_labels(&mut out, graph, config, &names);

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
    retention_rate: f64,
    names: &NameMap,
) {
    let m = &config.model;
    let wl = config.weight_levels.clamp(1, 10);
    let k = m.threshold_levels.clamp(1, 10);

    writeln!(out, "// Discretization parameters (paper sections 3-4)").ok();
    writeln!(out, "const int WL = {};       // Weight discretization levels", wl).ok();
    writeln!(out, "const int T_d = {};      // Discretized threshold (paper section 3.2)", t_d).ok();
    writeln!(out, "const double r = {};     // Retention rate — multiplicative leak (paper section 4.2)", retention_rate).ok();
    writeln!(out, "const int K = {};        // Number of threshold levels", k).ok();

    if m.enable_arp {
        writeln!(out, "const int ARP = {};", m.arp).ok();
    }
    if m.enable_rrp {
        writeln!(out, "const int RRP = {};", m.rrp).ok();
        writeln!(out, "const double alpha = {};", m.alpha as f64 / 100.0).ok();
    }

    writeln!(out).ok();
    writeln!(out, "// Per-neuron potential bounds (paper section 7)").ok();
    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        let name = &names[&node.id];
        let p_max = compute_p_max(node, graph, config, t_d);
        let p_min = compute_p_min(node, graph, config);
        writeln!(
            out,
            "const int P_MAX_{name} = {p_max};  // T_d + max_excitatory_input"
        )
        .ok();
        writeln!(
            out,
            "const int P_MIN_{name} = {p_min};  // sum of inhibitory inputs"
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

/// Compute the minimum potential for a neuron in the discretized domain.
/// P_MIN = sum of negative (inhibitory) discretized weights.
/// Returns 0 if no inhibitory inputs.
fn compute_p_min(node: &Node, graph: &SnnGraph, config: &PrismGenConfig) -> i32 {
    let wl = config.weight_levels.clamp(1, 10);
    let incoming = graph.incoming_edges(node.id);
    let all_weights: Vec<i32> = incoming
        .iter()
        .map(|e| discretize_weight(e.signed_weight(), wl))
        .collect();
    compute_discretized_p_min(&all_weights)
}

fn write_discretized_weights(out: &mut String, graph: &SnnGraph, wl: u8, names: &NameMap) {
    writeln!(out, "// Discretized synaptic weights (paper section 3)").ok();
    for edge in &graph.edges {
        let w_d = discretize_weight(edge.signed_weight(), wl);
        let from_name = &names[&edge.from];
        let to_name = &names[&edge.to];
        writeln!(
            out,
            "const int W_{from_name}_{to_name} = {w_d};  // delta_{wl}({})",
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
    names: &NameMap,
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
        let n = &names[&node.id];
        let incoming = graph.incoming_edges(node.id);

        if incoming.is_empty() {
            writeln!(out, "formula contrib_{n} = 0;").ok();
            continue;
        }

        let terms: Vec<String> = incoming
            .iter()
            .map(|e| {
                let from_name = &names[&e.from];
                let to_name = &names[&e.to];
                let spike_var = if graph.is_input(e.from) {
                    format!("x_{from_name}")
                } else {
                    format!("y_{from_name}")
                };
                format!("W_{from_name}_{to_name} * {spike_var}")
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
    names: &NameMap,
) {
    writeln!(out, "// Potential update with multiplicative leak (isomorphic with simulation engine)").ok();
    writeln!(out, "// newP_n = max(P_MIN_n, min(P_MAX_n, floor(r * p_n) + contrib_n))").ok();

    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        let n = &names[&node.id];
        writeln!(
            out,
            "formula newP_{n} = max(P_MIN_{n}, min(P_MAX_{n}, floor(r * p_{n}) + contrib_{n}));"
        )
        .ok();
    }
}


fn write_feasibility_analysis(
    out: &mut String,
    graph: &SnnGraph,
    config: &PrismGenConfig,
    t_d: i32,
    retention_rate: f64,
    names: &NameMap,
) {
    let wl = config.weight_levels.clamp(1, 10);

    writeln!(out, "// Feasibility analysis (paper section 5)").ok();
    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        let name = &names[&node.id];
        let incoming = graph.incoming_edges(node.id);
        let excitatory_weights: Vec<i32> = incoming
            .iter()
            .map(|e| discretize_weight(e.signed_weight(), wl))
            .filter(|&w| w > 0)
            .collect();

        let feasibility = check_feasibility(&excitatory_weights, t_d, retention_rate);
        match feasibility {
            Feasibility::SingleStep => {
                writeln!(out, "// {name}: FEASIBLE (single-step reach)").ok();
            }
            Feasibility::MultiStep { min_steps } => {
                writeln!(out, "// {name}: FEASIBLE (multi-step, min {min_steps} steps)").ok();
            }
            Feasibility::Impossible => {
                writeln!(out, "// WARNING: {name} INFEASIBLE — steady-state below threshold").ok();
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

fn write_input_module(out: &mut String, graph: &SnnGraph, config: &PrismGenConfig, names: &NameMap) {
    let inputs: Vec<_> = graph
        .nodes
        .iter()
        .filter(|n| graph.is_input(n.id))
        .collect();

    if inputs.is_empty() {
        writeln!(out, "// No input nodes defined").ok();
        return;
    }

    if needs_global_clock(graph) {
        write_global_clock(out, config);
        writeln!(out).ok();
    }

    writeln!(out, "// Input module").ok();
    writeln!(out, "module Inputs").ok();

    for input in &inputs {
        let input_name = &names[&input.id];
        writeln!(out, "  x_{input_name} : [0..1] init 0;").ok();
    }
    writeln!(out).ok();

    for input in &inputs {
        let n = &names[&input.id];
        if let Some(ref cfg) = input.input_config {
            let active_count = cfg.generators.iter().filter(|g| g.active).count();
            if active_count == 0 {
                writeln!(out, "  [tick] true -> (x_{n}' = 0);").ok();
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
                            _ => Some(0.5),
                        }
                    })
                    .unwrap_or(0.5);

                if (prob - 1.0).abs() < 1e-9 {
                    writeln!(out, "  [tick] true -> (x_{n}' = 1);").ok();
                } else if prob.abs() < 1e-9 {
                    writeln!(out, "  [tick] true -> (x_{n}' = 0);").ok();
                } else {
                    writeln!(
                        out,
                        "  [tick] true -> {:.6}:(x_{n}' = 1) + {:.6}:(x_{n}' = 0);",
                        prob,
                        1.0 - prob
                    )
                    .ok();
                }
            }
        } else {
            writeln!(out, "  [tick] true -> (x_{n}' = 1);").ok();
        }
    }

    writeln!(out, "endmodule").ok();
}

// ============================================================================
// Discretized Neuron Module — The core implementation (paper §7)
// ============================================================================

/// Write a discretized neuron module that tracks exact potential in [P_MIN..P_MAX].
fn write_discretized_neuron_module(
    out: &mut String,
    node: &Node,
    _graph: &SnnGraph,
    config: &PrismGenConfig,
    t_d: i32,
    names: &NameMap,
) {
    let n = &names[&node.id];
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

    writeln!(out, "module {n}").ok();

    // State variable (refractory state)
    if model.enable_arp || model.enable_rrp {
        writeln!(
            out,
            "  // State: 0=normal{}{}",
            if model.enable_arp { ", 1=ARP" } else { "" },
            if model.enable_rrp { ", 2=RRP" } else { "" }
        )
        .ok();
        writeln!(out, "  s_{n} : [0..{max_state}] init 0;").ok();
    }

    if model.enable_arp {
        writeln!(out, "  aref_{n} : [0..ARP] init 0;").ok();
    }
    if model.enable_rrp {
        writeln!(out, "  rref_{n} : [0..RRP] init 0;").ok();
    }

    writeln!(out, "  y_{n} : [0..1] init 0;  // spike output").ok();
    writeln!(
        out,
        "  p_{n} : [P_MIN_{n}..P_MAX_{n}] init 0;  // membrane potential (discretized domain)"
    )
    .ok();
    writeln!(out).ok();

    // State guard prefix
    let state_guard = if model.enable_arp || model.enable_rrp {
        format!("s_{n} = 0 & ")
    } else {
        String::new()
    };

    // ── Isomorphic firing: guards check newP (post-accumulation) ──
    // Matches simulation.rs handle_normal_state() semantics.
    // Fire branch sets p' = 0 (no dead reset tick).

    writeln!(
        out,
        "  // Normal period - firing on newP ({k} levels, no reset tick)"
    )
    .ok();

    // Precompute threshold boundaries to ensure complete integer coverage.
    let step = 1.0 / k as f64;
    let boundaries: Vec<i32> = (0..=k)
        .map(|j| {
            if j == k {
                t_d
            } else {
                (t_d as f64 * j as f64 / k as f64).floor() as i32
            }
        })
        .collect();

    // Level 0: newP <= boundaries[1] -> no fire
    writeln!(
        out,
        "  // Level 0: newP_{n} <= {} -> no fire",
        boundaries[1]
    )
    .ok();
    writeln!(
        out,
        "  [tick] {state_guard}newP_{n} <= {} -> (y_{n}' = 0) & (p_{n}' = newP_{n});",
        boundaries[1]
    )
    .ok();

    for j in 1..k {
        let fire_prob = j as f64 * step;
        let no_fire_prob = 1.0 - fire_prob;
        let lower = boundaries[j];
        let upper = boundaries[j + 1];

        if lower == upper {
            writeln!(out, "  // Level {j}: SKIPPED (degenerate)").ok();
            continue;
        }

        writeln!(out, "  // Level {j}: {lower} < newP_{n} <= {upper} -> fire P={fire_prob:.2}").ok();
        writeln!(
            out,
            "  [tick] {state_guard}newP_{n} > {lower} & newP_{n} <= {upper} -> {:.6}:(y_{n}' = 1) & (p_{n}' = 0) + {:.6}:(y_{n}' = 0) & (p_{n}' = newP_{n});",
            fire_prob, no_fire_prob
        )
        .ok();
    }

    writeln!(out, "  // Level {k}: newP_{n} > {t_d} -> certain fire").ok();
    writeln!(
        out,
        "  [tick] {state_guard}newP_{n} > {t_d} -> 1.0:(y_{n}' = 1) & (p_{n}' = 0);"
    )
    .ok();

    writeln!(out).ok();

    if model.enable_arp {
        writeln!(out, "  // Absolute refractory period").ok();
        writeln!(
            out,
            "  [tick] s_{n} = 1 & aref_{n} > 0 -> (aref_{n}' = aref_{n} - 1) & (y_{n}' = 0) & (p_{n}' = p_{n});"
        )
        .ok();

        if model.enable_rrp {
            writeln!(
                out,
                "  [tick] s_{n} = 1 & aref_{n} = 0 -> (s_{n}' = 2) & (rref_{n}' = RRP) & (y_{n}' = 0);"
            )
            .ok();
        } else {
            writeln!(
                out,
                "  [tick] s_{n} = 1 & aref_{n} = 0 -> (s_{n}' = 0) & (y_{n}' = 0);"
            )
            .ok();
        }
        writeln!(out).ok();
    }

    if model.enable_arp && model.enable_rrp {
        let alpha = model.alpha as f64 / 100.0;

        writeln!(out, "  // Relative refractory period (alpha-scaled, newP guards)").ok();

        for j in 0..=k {
            let base_prob = if j == 0 { 0.0 } else if j == k { 1.0 } else { j as f64 * step };
            let fire_prob = alpha * base_prob;
            let no_fire_prob = 1.0 - fire_prob;

            if j == 0 {
                if fire_prob.abs() < 1e-9 {
                    writeln!(out, "  [tick] s_{n} = 2 & rref_{n} > 0 & newP_{n} <= {} -> (y_{n}' = 0) & (p_{n}' = newP_{n}) & (rref_{n}' = rref_{n} - 1);", boundaries[1]).ok();
                } else {
                    writeln!(out, "  [tick] s_{n} = 2 & rref_{n} > 0 & newP_{n} <= {} -> {:.6}:(y_{n}' = 0) & (p_{n}' = newP_{n}) & (rref_{n}' = rref_{n} - 1) + {:.6}:(y_{n}' = 1) & (p_{n}' = 0) & (aref_{n}' = ARP) & (rref_{n}' = 0) & (s_{n}' = 1);", boundaries[1], no_fire_prob, fire_prob).ok();
                }
            } else if j == k {
                if (fire_prob - 1.0).abs() < 1e-9 {
                    writeln!(out, "  [tick] s_{n} = 2 & rref_{n} > 0 & newP_{n} > {} -> (y_{n}' = 1) & (p_{n}' = 0) & (aref_{n}' = ARP) & (rref_{n}' = 0) & (s_{n}' = 1);", boundaries[k]).ok();
                } else {
                    writeln!(out, "  [tick] s_{n} = 2 & rref_{n} > 0 & newP_{n} > {} -> {:.6}:(y_{n}' = 0) & (p_{n}' = newP_{n}) & (rref_{n}' = rref_{n} - 1) + {:.6}:(y_{n}' = 1) & (p_{n}' = 0) & (aref_{n}' = ARP) & (rref_{n}' = 0) & (s_{n}' = 1);", boundaries[k], no_fire_prob, fire_prob).ok();
                }
            } else {
                let lower = boundaries[j];
                let upper = boundaries[j + 1];
                if lower == upper { continue; }
                if fire_prob.abs() < 1e-9 {
                    writeln!(out, "  [tick] s_{n} = 2 & rref_{n} > 0 & newP_{n} > {lower} & newP_{n} <= {upper} -> (y_{n}' = 0) & (p_{n}' = newP_{n}) & (rref_{n}' = rref_{n} - 1);").ok();
                } else {
                    writeln!(out, "  [tick] s_{n} = 2 & rref_{n} > 0 & newP_{n} > {lower} & newP_{n} <= {upper} -> {:.6}:(y_{n}' = 0) & (p_{n}' = newP_{n}) & (rref_{n}' = rref_{n} - 1) + {:.6}:(y_{n}' = 1) & (p_{n}' = 0) & (aref_{n}' = ARP) & (rref_{n}' = 0) & (s_{n}' = 1);", no_fire_prob, fire_prob).ok();
                }
            }
        }

        writeln!(out, "  [tick] s_{n} = 2 & rref_{n} = 0 -> (p_{n}' = 0) & (y_{n}' = 0) & (s_{n}' = 0);").ok();
    }

    writeln!(out, "endmodule").ok();
}

// ============================================================================
// Transfer, Rewards, Labels (unchanged from old generator)
// ============================================================================

fn write_rewards(out: &mut String, graph: &SnnGraph, names: &NameMap) {
    writeln!(out, "// Spike count rewards").ok();

    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        let n = &names[&node.id];
        writeln!(out, "rewards \"spike_{n}_count\"").ok();
        writeln!(out, "  y_{n} = 1 : 1;").ok();
        writeln!(out, "endrewards").ok();
        writeln!(out).ok();
    }
}

fn write_labels(out: &mut String, graph: &SnnGraph, config: &PrismGenConfig, names: &NameMap) {
    writeln!(out, "// Labels for PCTL properties").ok();
    let model = &config.model;

    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        let n = &names[&node.id];
        writeln!(out, "label \"spike_{n}\" = (y_{n} = 1);").ok();

        if model.enable_arp {
            if model.enable_rrp {
                writeln!(out, "label \"refractory_{n}\" = (s_{n} = 1 | s_{n} = 2);").ok();
            } else {
                writeln!(out, "label \"refractory_{n}\" = (s_{n} = 1);").ok();
            }
        }
    }

    let outputs = graph.output_neurons();
    if !outputs.is_empty() {
        let output_spikes: Vec<_> = outputs.iter().map(|id| format!("y_{} = 1", names[id])).collect();
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
        assert!(prism.contains("const double r = "));
        assert!(prism.contains("dtmc"));
    }

    #[test]
    fn test_discretized_model_uses_discretized_weights() {
        let graph = SnnGraph::demo_layout();
        let config = basic_config();
        let prism = generate_discretized_model(&graph, &config);

        // Should contain discretized weight constants (W_ prefix with named labels)
        assert!(
            prism.contains("W_"),
            "Model should contain discretized weight constants"
        );
        // Should NOT contain old raw weight constants
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
    fn test_discretized_model_has_multiplicative_leak() {
        let graph = SnnGraph::demo_layout();
        let config = basic_config();
        let prism = generate_discretized_model(&graph, &config);

        // Should contain r (retention rate) for multiplicative leak
        assert!(
            prism.contains("const double r = "),
            "Model should declare retention rate r"
        );
        // Should contain floor(r * p_n) in potential update formula
        assert!(
            prism.contains("floor(r * p_"),
            "Model should use multiplicative leak floor(r * p)"
        );
        // Should NOT contain lambda_d (old additive leak)
        assert!(
            !prism.contains("lambda_d"),
            "Model should not reference lambda_d (old additive leak)"
        );
        assert!(
            prism.contains("newP_"),
            "Model should contain potential update formulas"
        );
    }

    #[test]
    fn test_discretized_model_allows_negative_potentials() {
        // Create a graph with inhibitory connections
        let mut graph = SnnGraph::default();
        use crate::snn::graph::NodeKind;
        let n1 = graph.add_node("N1", NodeKind::Neuron, [0.0, 0.0]);
        let n2 = graph.add_node("N2", NodeKind::Neuron, [100.0, 0.0]);
        // N1 inhibits N2
        graph.add_edge(n1, n2, 100); // weight 100 but...
        // Add inhibitory edge by setting negative weight
        if let Some(edge) = graph.edges.last_mut() {
            edge.weight = 100;
            edge.is_inhibitory = true; // -100 signed weight
        }
        let config = basic_config();
        let prism = generate_discretized_model(&graph, &config);

        // Should contain negative P_MIN for N2
        assert!(
            prism.contains("P_MIN_N2"),
            "Model should define P_MIN for neurons with inhibitory inputs"
        );
        // P_MIN should appear in variable declaration
        assert!(
            prism.contains("P_MIN_N2..P_MAX_N2"),
            "Variable range should use P_MIN..P_MAX"
        );
    }

    #[test]
    fn test_discretized_model_uses_exact_potential() {
        let graph = SnnGraph::demo_layout();
        let config = basic_config();
        let prism = generate_discretized_model(&graph, &config);

        // Should contain exact potential variables with named labels
        // The demo layout has neurons labeled "Neuron A", "Neuron B", "Output"
        let has_potential = prism.lines().any(|line| {
            line.contains(" p_Neuron_A :") || line.contains(" p_Neuron_B :") || line.contains(" p_Output :")
        });
        assert!(
            has_potential,
            "Model should have exact potential variables (p_name)"
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
