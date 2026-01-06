//! PRISM model generator for Spiking Neural Networks.
//!
//! Generates DTMC (Discrete-Time Markov Chain) models from [`SnnGraph`] for
//! probabilistic model checking with PRISM.

use crate::snn::graph::{NeuronParams, Node, SnnGraph};
use std::fmt::Write;

/// Configuration for PRISM model generation.
#[derive(Debug, Clone)]
pub struct PrismGenConfig {
    /// Potential range for membrane potential variable.
    pub potential_range: (i32, i32),
    /// Whether to include spike count rewards.
    pub include_rewards: bool,
    /// Global time bound for bounded properties.
    pub time_bound: Option<u32>,
}

impl Default for PrismGenConfig {
    fn default() -> Self {
        Self {
            potential_range: (-500, 500),
            include_rewards: true,
            time_bound: Some(100),
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
    write_threshold_formulas(&mut out, params);
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
    write_labels(&mut out, graph);

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
        writeln!(out, "const int T_MAX = {};", t).ok();
    }
}

fn write_threshold_formulas(out: &mut String, params: &NeuronParams) {
    writeln!(out, "// Firing probability thresholds (fraction of P_rth)").ok();
    for (i, th) in params.thresholds.iter().enumerate() {
        // thresholds are 0-100, convert to fraction for formula
        writeln!(out, "formula threshold{} = {} * P_rth / 100;", i + 1, th).ok();
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
            "0".to_string()
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

    // Input variables (constant 1 for now, could be extended to patterns)
    for input in &inputs {
        writeln!(out, "  x{} : [0..1] init 1;", input.id.0).ok();
    }

    // Tick action maintains inputs
    write!(out, "  [tick] true -> 1: ").ok();
    for (i, input) in inputs.iter().enumerate() {
        if i > 0 {
            write!(out, " & ").ok();
        }
        write!(out, "(x{}' = 1)", input.id.0).ok();
    }
    writeln!(out, ";").ok();
    writeln!(out, "endmodule").ok();
}

fn write_neuron_module(out: &mut String, node: &Node, _graph: &SnnGraph, _config: &PrismGenConfig) {
    let n = node.id.0;
    let params = &node.params;

    writeln!(out, "module Neuron{}", n).ok();
    writeln!(out, "  // State: 0=normal, 1=ARP, 2=RRP").ok();
    writeln!(out, "  s{} : [0..2] init 0;", n).ok();
    writeln!(out, "  aref{} : [0..ARP] init 0;", n).ok();
    writeln!(out, "  rref{} : [0..RRP] init 0;", n).ok();
    writeln!(out, "  y{} : [0..1] init 0;  // spike output", n).ok();
    writeln!(
        out,
        "  p{} : [P_MIN..P_MAX] init {};  // membrane potential",
        n,
        params.p_rest // Already in 0-100 range
    )
    .ok();
    writeln!(out).ok();

    // Normal period - spike action
    writeln!(out, "  // Normal period - spike action").ok();
    writeln!(
        out,
        "  [spike{}] s{} = 0 & y{} = 1 -> (p{}' = P_reset) & (aref{}' = ARP) & (y{}' = 0) & (s{}' = 1);",
        n, n, n, n, n, n, n
    )
    .ok();

    // Normal period - probabilistic transitions based on thresholds
    writeln!(out, "  // Normal period - probabilistic firing").ok();

    // Below threshold1: no spike
    writeln!(
        out,
        "  [tick] s{} = 0 & y{} = 0 & p{} <= threshold1 -> (y{}' = 0) & (p{}' = newPotential_{});",
        n, n, n, n, n, n
    )
    .ok();

    // Threshold-based probabilistic firing (10 levels)
    let probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    for i in 0..9 {
        let no_spike_prob = 1.0 - probs[i];
        let spike_prob = probs[i];
        writeln!(
            out,
            "  [tick] s{} = 0 & y{} = 0 & p{} > threshold{} & p{} <= threshold{} -> {:.1}:(y{}' = 0) & (p{}' = newPotential_{}) + {:.1}:(y{}' = 1);",
            n, n, n, i + 1, n, i + 2, no_spike_prob, n, n, n, spike_prob, n
        )
        .ok();
    }

    // Above threshold10: always spike
    writeln!(
        out,
        "  [tick] s{} = 0 & y{} = 0 & p{} > threshold10 -> 1.0:(y{}' = 1);",
        n, n, n, n
    )
    .ok();

    writeln!(out).ok();

    // Absolute refractory period
    writeln!(out, "  // Absolute refractory period").ok();
    writeln!(
        out,
        "  [tick] s{} = 1 & aref{} > 0 -> (aref{}' = aref{} - 1) & (y{}' = 0) & (p{}' = newPotential_{});",
        n, n, n, n, n, n, n
    )
    .ok();
    writeln!(
        out,
        "  [tick] s{} = 1 & aref{} = 0 -> (s{}' = 2) & (rref{}' = RRP) & (y{}' = 0);",
        n, n, n, n, n
    )
    .ok();

    writeln!(out).ok();

    // Relative refractory period
    writeln!(
        out,
        "  // Relative refractory period (alpha-scaled probabilities)"
    )
    .ok();
    writeln!(
        out,
        "  [spike{}] s{} = 2 & y{} = 1 & rref{} > 0 -> (p{}' = P_reset) & (aref{}' = ARP) & (y{}' = 0) & (rref{}' = 0) & (s{}' = 1);",
        n, n, n, n, n, n, n, n, n
    )
    .ok();

    // RRP probabilistic firing (alpha-scaled)
    writeln!(
        out,
        "  [tick] s{} = 2 & y{} = 0 & rref{} > 0 & p{} <= threshold1 -> (y{}' = 0) & (p{}' = newPotential_{}) & (rref{}' = rref{} - 1);",
        n, n, n, n, n, n, n, n, n
    )
    .ok();

    for i in 0..9 {
        let base_spike = probs[i];
        let no_spike_prob = 1.0 - (params.alpha as f64 * base_spike);
        let spike_prob = params.alpha as f64 * base_spike;
        writeln!(
            out,
            "  [tick] s{} = 2 & y{} = 0 & rref{} > 0 & p{} > threshold{} & p{} <= threshold{} -> {:.3}:(y{}' = 0) & (p{}' = newPotential_{}) & (rref{}' = rref{} - 1) + {:.3}:(y{}' = 1);",
            n, n, n, n, i + 1, n, i + 2, no_spike_prob, n, n, n, n, n, spike_prob, n
        )
        .ok();
    }

    // RRP ended - return to normal
    writeln!(
        out,
        "  [tick] s{} = 2 & y{} = 0 & rref{} = 0 -> (p{}' = P_reset) & (y{}' = 0) & (s{}' = 0);",
        n, n, n, n, n, n
    )
    .ok();

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

fn write_labels(out: &mut String, graph: &SnnGraph) {
    writeln!(out, "// Labels for PCTL properties").ok();

    for node in &graph.nodes {
        // Skip Input nodes - they don't have neuron modules
        if graph.is_input(node.id) {
            continue;
        }
        writeln!(out, "label \"spike{}\" = (y{} = 1);", node.id.0, node.id.0).ok();
        writeln!(
            out,
            "label \"refractory{}\" = (s{} = 1 | s{} = 2);",
            node.id.0, node.id.0, node.id.0
        )
        .ok();
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

    // Safety: refractory correctness
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
}
