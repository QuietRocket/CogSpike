//! PRISM model generator for Spiking Neural Networks.
//!
//! Generates DTMC (Discrete-Time Markov Chain) models from [`SnnGraph`] for
//! probabilistic model checking with PRISM.

use crate::simulation::{InputPattern, ModelConfig};
use crate::snn::graph::{NeuronParams, Node, SnnGraph};
use std::fmt::Write as _;

/// Configuration for PRISM model generation.
#[derive(Debug, Clone)]
pub struct PrismGenConfig {
    /// Potential range for membrane potential variable.
    pub potential_range: (i32, i32),
    /// Whether to include spike count rewards.
    pub include_rewards: bool,
    /// Global time bound for bounded properties.
    pub time_bound: Option<u32>,
    /// Model configuration (threshold levels, refractory toggles).
    pub model: ModelConfig,
}

impl Default for PrismGenConfig {
    fn default() -> Self {
        Self {
            potential_range: (-500, 500),
            include_rewards: true,
            time_bound: Some(100),
            model: ModelConfig::default(),
        }
    }
}

/// Generates a complete PRISM model from an SNN graph.
pub fn generate_prism_model(graph: &SnnGraph, config: &PrismGenConfig) -> String {
    let mut out = String::with_capacity(4096);

    // Header
    writeln!(out, "// Auto-generated PRISM model from CogSpike").ok();
    writeln!(
        out,
        "// Neurons: {}, Edges: {}",
        graph.nodes.len(),
        graph.edges.len()
    )
    .ok();
    writeln!(out, "dtmc\n").ok();

    // Global constants from first neuron (or defaults)
    let default_params = NeuronParams::default();
    let params = graph
        .nodes
        .first()
        .map(|n| &n.params)
        .unwrap_or(&default_params);

    write_global_constants(&mut out, params, config);
    writeln!(out).ok();

    // Threshold formulas
    write_threshold_formulas(&mut out, params, config);
    writeln!(out).ok();

    // Weight constants for each edge
    write_weight_constants(&mut out, graph);
    writeln!(out).ok();

    // Transfer variables (spike propagation between neurons)
    write_transfer_formulas(&mut out, graph);
    writeln!(out).ok();

    // Potential formulas for each neuron
    write_potential_formulas(&mut out, graph, config);
    writeln!(out).ok();

    // Input module
    write_input_module(&mut out, graph);
    writeln!(out).ok();

    // Neuron modules
    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue; // Inputs handled separately
        }
        write_neuron_module(&mut out, node, graph, config);
        writeln!(out).ok();
    }

    // Transfer modules (synapse spike propagation)
    write_transfer_modules(&mut out, graph);
    writeln!(out).ok();

    // Rewards for spike counting
    if config.include_rewards {
        write_rewards(&mut out, graph);
    }

    // Labels for common properties
    write_labels(&mut out, graph, config);

    out
}

fn write_global_constants(out: &mut String, params: &NeuronParams, config: &PrismGenConfig) {
    writeln!(out, "// Global neuron parameters").ok();
    // Values are already in 0-100 range
    writeln!(out, "const int P_rth = {};", params.p_rth).ok();
    writeln!(out, "const int P_rest = {};", params.p_rest).ok();
    writeln!(out, "const int P_reset = {};", params.p_reset).ok();
    // leak_r is 0-100, representing 0.0-1.0, so divide by 100 for PRISM double
    writeln!(out, "const double r = {};", params.leak_r as f64 / 100.0).ok();
    writeln!(out, "const int ARP = {};", params.arp).ok();
    writeln!(out, "const int RRP = {};", params.rrp).ok();
    // alpha is 0-100, representing 0.0-1.0
    writeln!(out, "const double alpha = {};", params.alpha as f64 / 100.0).ok();
    writeln!(out, "const int P_MIN = {};", config.potential_range.0).ok();
    writeln!(out, "const int P_MAX = {};", config.potential_range.1).ok();
    if let Some(t) = config.time_bound {
        writeln!(out, "const int T_MAX = {t};").ok();
    }
}

fn write_threshold_formulas(out: &mut String, params: &NeuronParams, config: &PrismGenConfig) {
    let levels = config.model.threshold_levels.clamp(1, 10);
    writeln!(out, "// Firing probability thresholds ({levels} levels)").ok();
    for i in 1..=levels {
        // Generate thresholds matching simulation.rs generate_thresholds()
        let th = (i as u32 * params.p_rth as u32) / levels as u32;
        writeln!(out, "formula threshold{i} = {th} * P_rth / 100;").ok();
    }
}

fn write_weight_constants(out: &mut String, graph: &SnnGraph) {
    writeln!(out, "// Synaptic weights").ok();
    for edge in &graph.edges {
        let is_input = graph.is_input(edge.from);
        // signed_weight() now returns i16 directly in -100..100 range
        let effective_weight = edge.signed_weight();

        if is_input {
            // Input to neuron weight
            writeln!(
                out,
                "const int weight_in{}_{} = {};",
                edge.from.0, edge.to.0, effective_weight
            )
            .ok();
        } else {
            // Neuron to neuron weight
            writeln!(
                out,
                "const int weight_n{}_{} = {};",
                edge.from.0, edge.to.0, effective_weight
            )
            .ok();
        }
    }
}

fn write_transfer_formulas(out: &mut String, graph: &SnnGraph) {
    writeln!(out, "// Transfer variables for spike propagation").ok();
    for edge in &graph.edges {
        if !graph.is_input(edge.from) {
            // Neuron-to-neuron edges need transfer variables
            writeln!(
                out,
                "// z{}_{} defined in transfer module",
                edge.from.0, edge.to.0
            )
            .ok();
        }
    }
}

fn write_potential_formulas(out: &mut String, graph: &SnnGraph, _config: &PrismGenConfig) {
    writeln!(out, "// Membrane potential update formulas").ok();

    for node in &graph.nodes {
        // Skip Input nodes
        if graph.is_input(node.id) {
            continue;
        }

        let incoming: Vec<_> = graph.incoming_edges(node.id).into_iter().collect();

        if incoming.is_empty() {
            // No inputs, potential just decays
            writeln!(
                out,
                "formula newPotential_{} = max(P_MIN, min(P_MAX, floor(r * p{})));",
                node.id.0, node.id.0
            )
            .ok();
            continue;
        }

        // Build input sum: weighted inputs + weighted neuron spikes
        let mut terms = Vec::new();

        for edge in incoming {
            if graph.is_input(edge.from) {
                // Input contribution
                terms.push(format!(
                    "weight_in{}_{} * x{}",
                    edge.from.0, edge.to.0, edge.from.0
                ));
            } else {
                // Neuron spike contribution (via transfer variable)
                terms.push(format!(
                    "weight_n{}_{} * z{}_{}",
                    edge.from.0, edge.to.0, edge.from.0, edge.to.0
                ));
            }
        }

        let input_sum = if terms.is_empty() {
            "0".to_owned()
        } else {
            terms.join(" + ")
        };

        writeln!(
            out,
            "formula newPotential_{} = max(P_MIN, min(P_MAX, floor(({}) + r * p{})));",
            node.id.0, input_sum, node.id.0
        )
        .ok();
    }
}

fn write_input_module(out: &mut String, graph: &SnnGraph) {
    let inputs: Vec<_> = graph
        .nodes
        .iter()
        .filter(|n| graph.is_input(n.id))
        .collect();

    if inputs.is_empty() {
        writeln!(out, "// No input nodes defined").ok();
        return;
    }

    writeln!(out, "module Inputs").ok();

    // Input variables
    for input in &inputs {
        writeln!(out, "  x{} : [0..1] init 0;", input.id.0).ok();
    }
    writeln!(out).ok();

    // Generate tick action for each input based on its pattern
    writeln!(out, "  // Input pattern transitions").ok();

    // Collect all input updates
    let mut input_updates: Vec<(u32, String)> = Vec::new();

    for input in &inputs {
        let n = input.id.0;
        let update = if let Some(ref cfg) = input.input_config {
            // Use first active generator's pattern (single generator support for now)
            if let Some(generator) = cfg.generators.iter().find(|g| g.active) {
                match &generator.pattern {
                    InputPattern::AlwaysOn => format!("(x{n}' = 1)"),
                    InputPattern::AlwaysOff => format!("(x{n}' = 0)"),
                    InputPattern::Random { probability } => {
                        // Generate probabilistic transition
                        format!("RAND:{probability}:{n}")
                    }
                    // For other patterns, default to always on (future enhancement)
                    _ => format!("(x{n}' = 1)"),
                }
            } else {
                // No active generators, default off
                format!("(x{n}' = 0)")
            }
        } else {
            // No config, default to always on (legacy behavior)
            format!("(x{n}' = 1)")
        };
        input_updates.push((n, update));
    }

    // Check if any inputs have random patterns
    let has_random = input_updates.iter().any(|(_, u)| u.starts_with("RAND:"));

    if has_random {
        // Generate separate transitions for each random input
        for (n, update) in &input_updates {
            if update.starts_with("RAND:") {
                let parts: Vec<&str> = update.split(':').collect();
                if parts.len() >= 3 {
                    if let Ok(prob) = parts[1].parse::<f64>() {
                        let no_spike = 1.0 - prob;
                        writeln!(
                            out,
                            "  [tick] true -> {:.4}:(x{n}' = 1) + {:.4}:(x{n}' = 0);",
                            prob, no_spike
                        )
                        .ok();
                    }
                }
            } else {
                // Non-random pattern, emit simple update
                writeln!(out, "  [tick] true -> 1: {};", update).ok();
            }
        }
    } else {
        // All inputs are deterministic, combine into one action
        write!(out, "  [tick] true -> 1: ").ok();
        for (i, (_, update)) in input_updates.iter().enumerate() {
            if i > 0 {
                write!(out, " & ").ok();
            }
            write!(out, "{update}").ok();
        }
        writeln!(out, ";").ok();
    }

    writeln!(out, "endmodule").ok();
}

#[expect(clippy::needless_range_loop)]
fn write_neuron_module(out: &mut String, node: &Node, _graph: &SnnGraph, config: &PrismGenConfig) {
    let n = node.id.0;
    let params = &node.params;
    let model = &config.model;
    let levels = model.threshold_levels.clamp(1, 10);

    // Determine max state based on enabled refractory periods
    let max_state = if model.enable_arp && model.enable_rrp {
        2
    } else if model.enable_arp {
        1
    } else {
        0
    };

    writeln!(out, "module Neuron{n}").ok();
    writeln!(
        out,
        "  // State: 0=normal{}{}",
        if model.enable_arp { ", 1=ARP" } else { "" },
        if model.enable_rrp { ", 2=RRP" } else { "" }
    )
    .ok();
    writeln!(out, "  s{n} : [0..{max_state}] init 0;").ok();

    if model.enable_arp {
        writeln!(out, "  aref{n} : [0..ARP] init 0;").ok();
    }
    if model.enable_rrp {
        writeln!(out, "  rref{n} : [0..RRP] init 0;").ok();
    }

    writeln!(out, "  y{n} : [0..1] init 0;  // spike output").ok();
    writeln!(
        out,
        "  p{} : [P_MIN..P_MAX] init {};  // membrane potential",
        n, params.p_rest
    )
    .ok();
    writeln!(out).ok();

    // Generate probability step size based on configured levels
    let step = 1.0 / levels as f64;

    // Normal period - spike action (what happens after a spike is decided)
    writeln!(out, "  // Normal period - spike action").ok();
    if model.enable_arp {
        writeln!(
            out,
            "  [spike{n}] s{n} = 0 & y{n} = 1 -> (p{n}' = P_reset) & (aref{n}' = ARP) & (y{n}' = 0) & (s{n}' = 1);"
        ).ok();
    } else {
        // No ARP: spike and stay in normal state
        writeln!(
            out,
            "  [spike{n}] s{n} = 0 & y{n} = 1 -> (p{n}' = P_reset) & (y{n}' = 0);"
        )
        .ok();
    }

    // Normal period - probabilistic transitions based on thresholds
    writeln!(
        out,
        "  // Normal period - probabilistic firing ({levels} levels)"
    )
    .ok();

    // Below threshold1: no spike
    writeln!(
        out,
        "  [tick] s{n} = 0 & y{n} = 0 & p{n} <= threshold1 -> (y{n}' = 0) & (p{n}' = newPotential_{n});"
    ).ok();

    // Threshold-based probabilistic firing (variable levels)
    for i in 0..(levels - 1) {
        let prob = (i + 1) as f64 * step;
        let no_spike_prob = 1.0 - prob;
        let spike_prob = prob;
        writeln!(
            out,
            "  [tick] s{n} = 0 & y{n} = 0 & p{n} > threshold{} & p{n} <= threshold{} -> {:.4}:(y{n}' = 0) & (p{n}' = newPotential_{n}) + {:.4}:(y{n}' = 1);",
            i + 1, i + 2, no_spike_prob, spike_prob
        ).ok();
    }

    // Above top threshold: always spike
    writeln!(
        out,
        "  [tick] s{n} = 0 & y{n} = 0 & p{n} > threshold{levels} -> 1.0:(y{n}' = 1);"
    )
    .ok();

    writeln!(out).ok();

    // Absolute refractory period (only if enabled)
    if model.enable_arp {
        writeln!(out, "  // Absolute refractory period").ok();
        writeln!(
            out,
            "  [tick] s{n} = 1 & aref{n} > 0 -> (aref{n}' = aref{n} - 1) & (y{n}' = 0) & (p{n}' = newPotential_{n});"
        ).ok();

        if model.enable_rrp {
            // ARP finished -> RRP
            writeln!(
                out,
                "  [tick] s{n} = 1 & aref{n} = 0 -> (s{n}' = 2) & (rref{n}' = RRP) & (y{n}' = 0);"
            )
            .ok();
        } else {
            // ARP finished -> Normal (skip RRP)
            writeln!(
                out,
                "  [tick] s{n} = 1 & aref{n} = 0 -> (s{n}' = 0) & (y{n}' = 0);"
            )
            .ok();
        }

        writeln!(out).ok();
    }

    // Relative refractory period (only if both ARP and RRP enabled)
    if model.enable_arp && model.enable_rrp {
        let alpha = params.alpha as f64 / 100.0;

        writeln!(
            out,
            "  // Relative refractory period (alpha-scaled probabilities)"
        )
        .ok();
        writeln!(
            out,
            "  [spike{n}] s{n} = 2 & y{n} = 1 & rref{n} > 0 -> (p{n}' = P_reset) & (aref{n}' = ARP) & (y{n}' = 0) & (rref{n}' = 0) & (s{n}' = 1);"
        ).ok();

        // RRP probabilistic firing (alpha-scaled)
        writeln!(
            out,
            "  [tick] s{n} = 2 & y{n} = 0 & rref{n} > 0 & p{n} <= threshold1 -> (y{n}' = 0) & (p{n}' = newPotential_{n}) & (rref{n}' = rref{n} - 1);"
        ).ok();

        for i in 0..(levels - 1) {
            let base_prob = (i + 1) as f64 * step;
            let spike_prob = alpha * base_prob;
            let no_spike_prob = 1.0 - spike_prob;
            writeln!(
                out,
                "  [tick] s{n} = 2 & y{n} = 0 & rref{n} > 0 & p{n} > threshold{} & p{n} <= threshold{} -> {:.4}:(y{n}' = 0) & (p{n}' = newPotential_{n}) & (rref{n}' = rref{n} - 1) + {:.4}:(y{n}' = 1);",
                i + 1, i + 2, no_spike_prob, spike_prob
            ).ok();
        }

        // RRP ended - return to normal
        writeln!(
            out,
            "  [tick] s{n} = 2 & y{n} = 0 & rref{n} = 0 -> (p{n}' = P_reset) & (y{n}' = 0) & (s{n}' = 0);"
        ).ok();
    }

    writeln!(out, "endmodule").ok();
}

fn write_transfer_modules(out: &mut String, graph: &SnnGraph) {
    writeln!(out, "// Synapse transfer modules (spike propagation)").ok();

    for edge in &graph.edges {
        // Skip edges from Input nodes (they don't need transfer modules)
        if graph.is_input(edge.from) {
            continue;
        }

        writeln!(out, "module Transfer{}_{}", edge.from.0, edge.to.0).ok();
        writeln!(out, "  z{}_{} : [0..1] init 0;", edge.from.0, edge.to.0).ok();
        writeln!(
            out,
            "  [spike{}] true -> 1: (z{}_{}' = 1);",
            edge.from.0, edge.from.0, edge.to.0
        )
        .ok();
        writeln!(
            out,
            "  [tick] true -> 1: (z{}_{}' = 0);",
            edge.from.0, edge.to.0
        )
        .ok();
        writeln!(out, "endmodule").ok();
        writeln!(out).ok();
    }
}

fn write_rewards(out: &mut String, graph: &SnnGraph) {
    writeln!(out, "// Spike count rewards").ok();

    for node in &graph.nodes {
        // Skip Input nodes - they don't have neuron modules
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
        // Skip Input nodes - they don't have neuron modules
        if graph.is_input(node.id) {
            continue;
        }
        let n = node.id.0;
        writeln!(out, "label \"spike{n}\" = (y{n} = 1);").ok();

        // Only generate refractory label if ARP is enabled
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

/// Generates PCTL property specifications for common verification queries.
pub fn generate_pctl_properties(graph: &SnnGraph) -> String {
    let mut out = String::with_capacity(1024);

    writeln!(out, "// Auto-generated PCTL properties from CogSpike").ok();
    writeln!(out, "const int T;  // Time bound parameter\n").ok();

    // Labels
    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        writeln!(out, "label \"spike{}\" = (y{} = 1);", node.id.0, node.id.0).ok();
    }
    writeln!(out).ok();

    // Basic reachability
    writeln!(out, "// Reachability: Can neuron spike?").ok();
    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        writeln!(out, "P=? [ F \"spike{}\" ]", node.id.0).ok();
    }
    writeln!(out).ok();

    // Bounded reachability
    writeln!(out, "// Bounded reachability: Spike within T steps").ok();
    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        writeln!(out, "P=? [ F<=T \"spike{}\" ]", node.id.0).ok();
    }
    writeln!(out).ok();

    // Liveness / persistence
    writeln!(out, "// Persistence: Neuron keeps spiking").ok();
    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        writeln!(out, "P>=1 [ G (F \"spike{}\") ]", node.id.0).ok();
    }
    writeln!(out).ok();

    // Refractory correctness check
    writeln!(out, "// Safety: No spikes during absolute refractory").ok();
    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        writeln!(
            out,
            "P>=1 [ G ((s{} = 1) => (y{} = 0)) ]",
            node.id.0, node.id.0
        )
        .ok();
    }
    writeln!(out).ok();

    // Reward queries
    writeln!(out, "// Cumulative spike count rewards").ok();
    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        writeln!(out, "R{{\"spike{}_count\"}}=? [ C<=T ]", node.id.0).ok();
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::ModelConfig;
    use crate::snn::graph::SnnGraph;

    #[test]
    fn test_generate_demo_model() {
        let graph = SnnGraph::demo_layout();
        let config = PrismGenConfig::default();
        let prism = generate_prism_model(&graph, &config);

        assert!(prism.contains("dtmc"));
        assert!(prism.contains("module Inputs"));
        assert!(prism.contains("module Neuron"));
        assert!(prism.contains("[tick]"));
        assert!(prism.contains("[spike"));
    }

    #[test]
    fn test_generate_pctl() {
        let graph = SnnGraph::demo_layout();
        let pctl = generate_pctl_properties(&graph);

        assert!(pctl.contains("P=?"));
        assert!(pctl.contains("F<=T"));
    }

    #[test]
    fn test_variable_threshold_levels() {
        let graph = SnnGraph::demo_layout();

        // Test with 5 threshold levels
        let config = PrismGenConfig {
            model: ModelConfig {
                threshold_levels: 5,
                enable_arp: true,
                enable_rrp: true,
            },
            ..Default::default()
        };
        let prism = generate_prism_model(&graph, &config);

        // Should contain 5 threshold formulas
        assert!(prism.contains("threshold1"));
        assert!(prism.contains("threshold5"));
        assert!(!prism.contains("threshold6"));
        assert!(prism.contains("// Firing probability thresholds (5 levels)"));
    }

    #[test]
    fn test_arp_disabled() {
        let graph = SnnGraph::demo_layout();

        let config = PrismGenConfig {
            model: ModelConfig {
                threshold_levels: 10,
                enable_arp: false,
                enable_rrp: false,
            },
            ..Default::default()
        };
        let prism = generate_prism_model(&graph, &config);

        // State should only go to 0 (normal)
        assert!(prism.contains("s2 : [0..0] init 0;"));
        // Should NOT have aref variable declaration
        assert!(!prism.contains("aref2 : [0..ARP]"));
        // Should NOT have refractory label (since ARP is disabled)
        assert!(!prism.contains("label \"refractory2\""));
    }

    #[test]
    fn test_rrp_disabled() {
        let graph = SnnGraph::demo_layout();

        let config = PrismGenConfig {
            model: ModelConfig {
                threshold_levels: 10,
                enable_arp: true,
                enable_rrp: false,
            },
            ..Default::default()
        };
        let prism = generate_prism_model(&graph, &config);

        // State should go to 0..1 (normal, ARP)
        assert!(prism.contains("s2 : [0..1] init 0;"));
        // Should have ARP variable
        assert!(prism.contains("aref2 : [0..ARP] init 0;"));
        // Should NOT have RRP variable
        assert!(!prism.contains("rref2 : [0..RRP]"));
        // Refractory label should only reference s=1
        assert!(prism.contains("label \"refractory2\" = (s2 = 1);"));
    }

    #[test]
    fn test_rrp_probabilities_are_valid() {
        // Regression test: RRP probabilities must be in [0, 1]
        // Previously, alpha was not divided by 100, causing negative probabilities
        let graph = SnnGraph::demo_layout();
        let config = PrismGenConfig::default();
        let prism = generate_prism_model(&graph, &config);

        // Should NOT contain negative probabilities
        assert!(
            !prism.contains("-> -"),
            "RRP should not have negative probabilities"
        );

        // Should NOT contain probabilities > 1 (except 1.0 which is valid)
        // Check for patterns like "5.000:" or "10.000:" which are invalid
        for line in prism.lines() {
            if line.contains("s2 = 2") && line.contains("->") {
                // RRP transition line
                assert!(!line.contains("+ 5."), "RRP probability should not be 5.x");
                assert!(
                    !line.contains("+ 10."),
                    "RRP probability should not be 10.x"
                );
                assert!(
                    !line.contains("-4."),
                    "RRP probability should not be negative"
                );
            }
        }

        // Verify alpha-scaled probabilities are correct (alpha=0.5, first level=0.1)
        // Expected: spike_prob = 0.5 * 0.1 = 0.05, no_spike = 0.95
        assert!(
            prism.contains("0.9500") || prism.contains("0.95"),
            "RRP first level should have ~0.95 no-spike probability"
        );
    }
}
