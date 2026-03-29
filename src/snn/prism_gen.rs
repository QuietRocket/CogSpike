//! PRISM model generator for Spiking Neural Networks.
//!
//! Generates DTMC (Discrete-Time Markov Chain) models from [`SnnGraph`] for
//! probabilistic model checking with PRISM.

use crate::simulation::{GeneratorCombineMode, InputNeuronConfig, InputPattern, ModelConfig};
use crate::snn::graph::{Node, NodeId, SnnGraph};
use std::collections::HashMap;
use std::fmt::Write as _;

// ============================================================================
// Named Label Infrastructure
// ============================================================================

/// PRISM language reserved words that cannot be used as identifiers.
const PRISM_RESERVED: &[&str] = &[
    "module", "endmodule", "dtmc", "ctmc", "mdp", "const", "formula",
    "label", "rewards", "endrewards", "init", "endinit", "global",
    "true", "false", "int", "double", "bool", "min", "max",
    "filter", "func", "ceil", "floor", "log", "mod", "pow", "sqrt",
    "rate", "prob",
];

/// Maps `NodeId` → PRISM-safe identifier derived from the node's GUI label.
pub type NameMap = HashMap<NodeId, String>;

/// Sanitize a GUI label into a valid PRISM identifier.
///
/// - Non-alphanumeric characters become underscores
/// - Consecutive / leading / trailing underscores are collapsed
/// - Names starting with a digit get an `n` prefix
/// - PRISM reserved words get a `_node` suffix
pub fn sanitize_prism_label(label: &str) -> String {
    let mapped: String = label
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() || ch == '_' { ch } else { '_' })
        .collect();
    // Collapse runs of underscores
    let mut result = String::with_capacity(mapped.len());
    let mut prev_underscore = true; // treat start as underscore to trim leading
    for ch in mapped.chars() {
        if ch == '_' {
            if !prev_underscore {
                result.push('_');
            }
            prev_underscore = true;
        } else {
            prev_underscore = false;
            result.push(ch);
        }
    }
    // Trim trailing underscore
    if result.ends_with('_') {
        result.pop();
    }
    // Handle empty or digit-leading
    if result.is_empty() {
        result = "unnamed".to_string();
    } else if result.starts_with(|c: char| c.is_ascii_digit()) {
        result = format!("n{result}");
    }
    // Avoid PRISM reserved words
    if PRISM_RESERVED.contains(&result.to_lowercase().as_str()) {
        result.push_str("_node");
    }
    result
}

/// Build a unique name map from the graph's node labels.
///
/// If two nodes sanitize to the same identifier, a `_<id>` suffix is
/// appended to disambiguate.
pub fn build_name_map(graph: &SnnGraph) -> NameMap {
    let mut map = NameMap::new();
    let mut seen: HashMap<String, Vec<NodeId>> = HashMap::new();
    for node in &graph.nodes {
        let sanitized = sanitize_prism_label(&node.label);
        seen.entry(sanitized.clone()).or_default().push(node.id);
        map.insert(node.id, sanitized);
    }
    // Disambiguate collisions
    for (base, ids) in &seen {
        if ids.len() > 1 {
            for id in ids {
                map.insert(*id, format!("{}_{}", base, id.0));
            }
        }
    }
    map
}

/// Write a comment header documenting the GUI-label → PRISM-identifier mapping.
fn write_name_mapping_header(out: &mut String, graph: &SnnGraph, names: &NameMap) {
    writeln!(out, "//").ok();
    writeln!(out, "// Node name mapping (GUI label -> PRISM variables):").ok();
    for node in &graph.nodes {
        let name = &names[&node.id];
        let role = if graph.is_input(node.id) {
            "Input"
        } else if graph.is_output(node.id) {
            "Output"
        } else {
            "Neuron"
        };
        if graph.is_input(node.id) {
            writeln!(out, "//   {name:>12} [{role}]  ->  x_{name}").ok();
        } else {
            writeln!(out, "//   {name:>12} [{role}]  ->  y_{name}, p_{name}").ok();
        }
    }
    writeln!(out, "//").ok();
}

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
    /// Weight discretization levels (W) for discretized mode.
    /// Default 3 → weights mapped to [-3, 3]. Only used by the discretized generator.
    pub weight_levels: u8,
}

impl Default for PrismGenConfig {
    fn default() -> Self {
        Self {
            // Tighter default range; overridden by model.derive_potential_range() if set
            potential_range: (-200, 200),
            include_rewards: true,
            time_bound: Some(100),
            model: ModelConfig::default(),
            weight_levels: 3,
        }
    }
}

/// Generates a complete PRISM model from an SNN graph.
pub fn generate_prism_model(graph: &SnnGraph, config: &PrismGenConfig) -> String {
    let mut out = String::with_capacity(4096);
    let names = build_name_map(graph);

    // Header
    writeln!(out, "// Auto-generated PRISM model from CogSpike").ok();
    writeln!(
        out,
        "// Neurons: {}, Edges: {}",
        graph.nodes.len(),
        graph.edges.len()
    )
    .ok();
    write_name_mapping_header(&mut out, graph, &names);
    writeln!(out, "dtmc\n").ok();

    // Global constants from ModelConfig
    write_global_constants(&mut out, graph, config, &names);
    writeln!(out).ok();

    // Threshold formulas
    write_threshold_formulas(&mut out, config);
    writeln!(out).ok();

    // Weight constants for each edge
    write_weight_constants(&mut out, graph, &names);
    writeln!(out).ok();

    // Transfer variables (spike propagation between neurons)
    write_transfer_formulas(&mut out, graph);
    writeln!(out).ok();

    // Potential formulas for each neuron
    write_potential_formulas(&mut out, graph, config, &names);
    writeln!(out).ok();

    // Input module
    write_input_module(&mut out, graph, config, &names);
    writeln!(out).ok();

    // Neuron modules
    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue; // Inputs handled separately
        }
        write_neuron_module(&mut out, node, graph, config, &names);
        writeln!(out).ok();
    }

    // Transfer modules removed — neurons read y{source} directly for
    // 1-tick synaptic delay isomorphic with simulation engine.
    writeln!(out).ok();

    // Rewards for spike counting
    if config.include_rewards {
        write_rewards(&mut out, graph, &names);
    }

    // Labels for common properties
    write_labels(&mut out, graph, config, &names);

    out
}

fn write_global_constants(out: &mut String, graph: &SnnGraph, config: &PrismGenConfig, names: &NameMap) {
    let m = &config.model;
    writeln!(out, "// Global neuron parameters").ok();
    // Values are already in 0-100 range
    writeln!(out, "const int P_rth = {};", m.p_rth).ok();
    writeln!(out, "const int P_rest = {};", m.p_rest).ok();
    writeln!(out, "const int P_reset = {};", m.p_reset).ok();
    // leak_r is 0-100, representing 0.0-1.0, so divide by 100 for PRISM double
    writeln!(out, "const double r = {};", m.leak_r as f64 / 100.0).ok();
    // Only include ARP/RRP constants if enabled
    if m.enable_arp {
        writeln!(out, "const int ARP = {};", m.arp).ok();
    }
    if m.enable_rrp {
        writeln!(out, "const int RRP = {};", m.rrp).ok();
        // alpha is only needed for RRP
        writeln!(out, "const double alpha = {};", m.alpha as f64 / 100.0).ok();
    }

    // Compute global potential range (for fallback in formulas)
    let (mut p_min, p_max) = m.derive_potential_range().unwrap_or(config.potential_range);
    if !graph.has_inhibitory_synapses() {
        p_min = 0;
    }
    writeln!(out, "const int P_MIN = {p_min};").ok();
    writeln!(out, "const int P_MAX = {p_max};").ok();

    // Per-neuron optimized potential ranges
    writeln!(out).ok();
    writeln!(
        out,
        "// Per-neuron potential bounds (optimized for state space)"
    )
    .ok();
    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        let name = &names[&node.id];
        let (n_min, n_max) = compute_neuron_potential_range(node, graph, config);
        writeln!(out, "const int P_MIN_{name} = {n_min};").ok();
        writeln!(out, "const int P_MAX_{name} = {n_max};").ok();
    }

    if let Some(t) = config.time_bound {
        writeln!(out, "const int T_MAX = {t};").ok();
    }
}

/// Compute the optimal potential range for a specific neuron based on incoming weights.
/// Returns (p_min, p_max) where:
/// - p_max = max(P_rth, sum of positive incoming weights) with headroom
/// - p_min = -(sum of negative incoming weights) or 0 if no inhibitory
fn compute_neuron_potential_range(
    node: &Node,
    graph: &SnnGraph,
    config: &PrismGenConfig,
) -> (i32, i32) {
    let incoming = graph.incoming_edges(node.id);

    // Calculate max excitatory and inhibitory input per step
    let mut max_excitatory: i32 = 0;
    let mut max_inhibitory: i32 = 0;

    for edge in &incoming {
        let w = edge.signed_weight() as i32;
        if w > 0 {
            max_excitatory += w;
        } else {
            max_inhibitory += w.abs();
        }
    }

    // P_MAX: at least P_rth (firing threshold), plus max input with 1.5x headroom
    // The 1.5x accounts for potential accumulation before firing
    let p_rth = config.model.p_rth as i32;
    let p_max = std::cmp::max(p_rth, (max_excitatory * 3) / 2);

    // P_MIN: only negative if there are inhibitory inputs
    let p_min = if max_inhibitory > 0 {
        -((max_inhibitory * 3) / 2)
    } else {
        0
    };

    (p_min, p_max)
}

fn write_threshold_formulas(out: &mut String, config: &PrismGenConfig) {
    let m = &config.model;
    let levels = m.threshold_levels.clamp(1, 10);
    writeln!(out, "// Firing probability thresholds ({levels} levels)").ok();
    writeln!(
        out,
        "// Isomorphic with simulation: threshold_i = (i * P_rth) / levels"
    )
    .ok();
    for i in 1..=levels {
        // Exact match of simulation.rs generate_thresholds(): (i * p_rth) / levels
        let th = (i as u32 * m.p_rth as u32) / levels as u32;
        writeln!(out, "formula threshold{i} = {th};").ok();
    }
}

fn write_weight_constants(out: &mut String, graph: &SnnGraph, names: &NameMap) {
    writeln!(out, "// Synaptic weights").ok();
    for edge in &graph.edges {
        let from_name = &names[&edge.from];
        let to_name = &names[&edge.to];
        let effective_weight = edge.signed_weight();
        writeln!(
            out,
            "const int w_{from_name}_{to_name} = {effective_weight};"
        )
        .ok();
    }
}

fn write_transfer_formulas(out: &mut String, _graph: &SnnGraph) {
    // Transfer modules removed for isomorphism with simulation.
    // Neuron-to-neuron spike propagation now reads y{source} directly
    // in the newPotential formula, giving 1-tick synaptic delay matching
    // the simulation engine's Phase 4 → Phase 2 transfer semantics.
    writeln!(
        out,
        "// Spike propagation: neurons read y(source) directly (1-tick delay)"
    )
    .ok();
}

fn write_potential_formulas(out: &mut String, graph: &SnnGraph, _config: &PrismGenConfig, names: &NameMap) {
    writeln!(
        out,
        "// Membrane potential update formulas (using per-neuron bounds)"
    )
    .ok();

    for node in &graph.nodes {
        // Skip Input nodes
        if graph.is_input(node.id) {
            continue;
        }

        let n = &names[&node.id];
        let incoming: Vec<_> = graph.incoming_edges(node.id).into_iter().collect();

        if incoming.is_empty() {
            // No inputs, potential just decays
            writeln!(
                out,
                "formula newPotential_{n} = max(P_MIN_{n}, min(P_MAX_{n}, floor(r * p_{n})));"
            )
            .ok();
            continue;
        }

        // Build input sum: weighted inputs + weighted neuron spikes
        let mut terms = Vec::new();

        for edge in incoming {
            let from_name = &names[&edge.from];
            let to_name = &names[&edge.to];
            if graph.is_input(edge.from) {
                // Input contribution
                terms.push(format!("w_{from_name}_{to_name} * x_{from_name}"));
            } else {
                // Neuron spike contribution (read y directly — 1-tick delay)
                terms.push(format!("w_{from_name}_{to_name} * y_{from_name}"));
            }
        }

        let input_sum = if terms.is_empty() {
            "0".to_owned()
        } else {
            terms.join(" + ")
        };

        writeln!(
            out,
            "formula newPotential_{n} = max(P_MIN_{n}, min(P_MAX_{n}, floor(({input_sum}) + r * p_{n})));"
        )
        .ok();
    }
}

// ============================================================================
// Multi-Generator Input Support
// ============================================================================

/// Check if any input neuron needs a global step counter (has time-dependent patterns).
fn needs_global_clock(graph: &SnnGraph) -> bool {
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

/// Write the GlobalClock module for time-dependent input patterns.
fn write_global_clock(out: &mut String, config: &PrismGenConfig) {
    let t_max = config.time_bound.unwrap_or(100);
    writeln!(out, "// Global clock for time-dependent input patterns").ok();
    writeln!(out, "module GlobalClock").ok();
    writeln!(out, "  step : [0..{t_max}] init 0;").ok();
    writeln!(out, "  [tick] step < {t_max} -> (step' = step + 1);").ok();
    writeln!(out, "  [tick] step = {t_max} -> (step' = step);").ok();
    writeln!(out, "endmodule").ok();
}

/// Convert a deterministic pattern to a PRISM formula expression.
/// Returns None for probabilistic patterns (Random, Poisson) and InternalFiring.
fn pattern_to_formula(pattern: &InputPattern, input_name: &str, gen_idx: usize) -> Option<String> {
    let formula_name = format!("in_{input_name}_g{gen_idx}");
    match pattern {
        InputPattern::AlwaysOn => Some(format!("formula {formula_name}_fires = true;")),
        InputPattern::AlwaysOff => Some(format!("formula {formula_name}_fires = false;")),
        InputPattern::Pulse { duration } => Some(format!(
            "formula {formula_name}_fires = (step < {duration});"
        )),
        InputPattern::Silence { duration } => Some(format!(
            "formula {formula_name}_fires = (step >= {duration});"
        )),
        InputPattern::Periodic { period, phase } => {
            if *period == 0 {
                Some(format!("formula {formula_name}_fires = false;"))
            } else {
                Some(format!(
                    "formula {formula_name}_fires = (mod(step + {phase}, {period}) = 0);"
                ))
            }
        }
        InputPattern::Burst {
            burst_length,
            silence_length,
        } => {
            let cycle = burst_length + silence_length;
            if cycle == 0 {
                Some(format!("formula {formula_name}_fires = false;"))
            } else {
                Some(format!(
                    "formula {formula_name}_fires = (mod(step, {cycle}) < {burst_length});"
                ))
            }
        }
        InputPattern::Custom { spike_times } => {
            if spike_times.is_empty() {
                Some(format!("formula {formula_name}_fires = false;"))
            } else if spike_times.len() > 20 {
                // Too many spike times - warn and use first 20
                let times: Vec<_> = spike_times.iter().take(20).collect();
                let conditions: Vec<_> = times.iter().map(|t| format!("step = {t}")).collect();
                Some(format!(
                    "formula {formula_name}_fires = ({});  // WARNING: truncated to 20 times",
                    conditions.join(" | ")
                ))
            } else {
                let conditions: Vec<_> =
                    spike_times.iter().map(|t| format!("step = {t}")).collect();
                Some(format!(
                    "formula {formula_name}_fires = ({});",
                    conditions.join(" | ")
                ))
            }
        }
        // Probabilistic patterns don't get formulas
        InputPattern::Random { .. }
        | InputPattern::Poisson { .. }
        | InputPattern::InternalFiring => None,
    }
}

/// Extract probability from a probabilistic pattern.
fn pattern_probability(pattern: &InputPattern, dt_ms: f32) -> Option<f64> {
    match pattern {
        InputPattern::Random { probability } => Some(*probability),
        InputPattern::Poisson { rate_hz } => {
            let step_duration_s = dt_ms as f64 / 1000.0;
            Some((*rate_hz * step_duration_s).min(1.0))
        }
        _ => None,
    }
}

/// Compute combined probability for multiple probabilistic generators.
fn compute_combined_probability(probabilities: &[f64], mode: GeneratorCombineMode) -> f64 {
    if probabilities.is_empty() {
        return 0.0;
    }

    match mode {
        GeneratorCombineMode::Or => {
            // P(at least one) = 1 - Π(1 - pᵢ)
            1.0 - probabilities.iter().map(|p| 1.0 - p).product::<f64>()
        }
        GeneratorCombineMode::And => {
            // P(all) = Πpᵢ
            probabilities.iter().product()
        }
        GeneratorCombineMode::Xor => {
            // P(odd number fire) - computed iteratively
            // Let P_odd be probability of odd count, P_even = 1 - P_odd
            // For each generator with prob p: new_P_odd = P_even * p + P_odd * (1-p)
            let mut p_odd = 0.0;
            for &p in probabilities {
                p_odd = (1.0 - p_odd) * p + p_odd * (1.0 - p);
            }
            p_odd
        }
    }
}

/// Categorized generators for an input neuron.
struct CategorizedGenerators {
    /// Indices and formula names of deterministic generators
    deterministic: Vec<(usize, String)>,
    /// Probabilities of probabilistic generators
    probabilistic: Vec<f64>,
}

/// Categorize active generators into deterministic and probabilistic.
fn categorize_generators(
    cfg: &InputNeuronConfig,
    input_name: &str,
    dt_ms: f32,
) -> CategorizedGenerators {
    let mut deterministic = Vec::new();
    let mut probabilistic = Vec::new();

    for (idx, generator) in cfg.generators.iter().enumerate() {
        if !generator.active {
            continue;
        }
        if let Some(prob) = pattern_probability(&generator.pattern, dt_ms) {
            probabilistic.push(prob);
        } else {
            deterministic.push((idx, format!("in_{input_name}_g{idx}_fires")));
        }
    }

    CategorizedGenerators {
        deterministic,
        probabilistic,
    }
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

    // Write GlobalClock if needed
    if needs_global_clock(graph) {
        write_global_clock(out, config);
        writeln!(out).ok();
    }

    // Write per-generator formulas for deterministic patterns
    writeln!(out, "// Input generator formulas").ok();
    for input in &inputs {
        let input_name = &names[&input.id];
        if let Some(ref cfg) = input.input_config {
            for (idx, generator) in cfg.generators.iter().enumerate() {
                if !generator.active {
                    continue;
                }
                if let Some(formula) = pattern_to_formula(&generator.pattern, input_name, idx) {
                    writeln!(out, "{formula}").ok();
                }
            }
        }
    }
    writeln!(out).ok();

    // Derive dt_ms from first neuron
    let dt_ms = graph
        .nodes
        .first()
        .map(|n| n.params.dt as f32 / 10.0)
        .unwrap_or(1.0);

    writeln!(out, "module Inputs").ok();

    // Input variables
    for input in &inputs {
        let input_name = &names[&input.id];
        writeln!(out, "  x_{input_name} : [0..1] init 0;").ok();
    }
    writeln!(out).ok();

    writeln!(out, "  // Multi-generator input transitions").ok();

    // Generate transitions for each input
    for input in &inputs {
        let input_name = &names[&input.id];

        if let Some(ref cfg) = input.input_config {
            let active_count = cfg.generators.iter().filter(|g| g.active).count();

            if active_count == 0 {
                // No active generators -> always off
                writeln!(out, "  [tick] true -> (x_{input_name}' = 0);").ok();
                continue;
            }

            let cats = categorize_generators(cfg, input_name, dt_ms);
            let mode = cfg.combine_mode;

            // Generate transitions based on mode and generator types
            write_input_transitions(out, input_name, &cats, mode);
        } else {
            // No config -> legacy behavior (always on)
            writeln!(out, "  [tick] true -> (x_{input_name}' = 1);").ok();
        }
    }

    writeln!(out, "endmodule").ok();
}

/// Write PRISM transitions for a single input neuron with categorized generators.
fn write_input_transitions(
    out: &mut String,
    input_name: &str,
    cats: &CategorizedGenerators,
    mode: GeneratorCombineMode,
) {
    let n = input_name;
    let has_det = !cats.deterministic.is_empty();
    let has_prob = !cats.probabilistic.is_empty();

    if !has_det && !has_prob {
        // No active generators
        writeln!(out, "  [tick] true -> (x_{n}' = 0);").ok();
        return;
    }

    if !has_det {
        // All probabilistic - simple combined probability
        let p_fire = compute_combined_probability(&cats.probabilistic, mode);
        let p_no_fire = 1.0 - p_fire;
        writeln!(
            out,
            "  [tick] true -> {:.6}:(x_{n}' = 1) + {:.6}:(x_{n}' = 0);",
            p_fire, p_no_fire
        )
        .ok();
        return;
    }

    if !has_prob {
        // All deterministic - just use formula
        let combined_formula = combine_deterministic_formulas(&cats.deterministic, mode);
        writeln!(out, "  [tick] {combined_formula} -> (x_{n}' = 1);").ok();
        writeln!(out, "  [tick] !({combined_formula}) -> (x_{n}' = 0);").ok();
        return;
    }

    // Mixed deterministic and probabilistic
    let p_prob = compute_combined_probability(&cats.probabilistic, mode);

    match mode {
        GeneratorCombineMode::Or => {
            let det_formula =
                combine_deterministic_formulas(&cats.deterministic, GeneratorCombineMode::Or);
            writeln!(out, "  [tick] ({det_formula}) -> (x_{n}' = 1);").ok();
            writeln!(
                out,
                "  [tick] !({det_formula}) -> {:.6}:(x_{n}' = 1) + {:.6}:(x_{n}' = 0);",
                p_prob,
                1.0 - p_prob
            )
            .ok();
        }
        GeneratorCombineMode::And => {
            let det_formula =
                combine_deterministic_formulas(&cats.deterministic, GeneratorCombineMode::And);
            writeln!(out, "  [tick] !({det_formula}) -> (x_{n}' = 0);").ok();
            writeln!(
                out,
                "  [tick] ({det_formula}) -> {:.6}:(x_{n}' = 1) + {:.6}:(x_{n}' = 0);",
                p_prob,
                1.0 - p_prob
            )
            .ok();
        }
        GeneratorCombineMode::Xor => {
            write_xor_mixed_transitions(out, n, cats);
        }
    }
}

/// Combine deterministic formula names with the given mode.
fn combine_deterministic_formulas(
    formulas: &[(usize, String)],
    mode: GeneratorCombineMode,
) -> String {
    if formulas.is_empty() {
        return "true".to_owned();
    }
    if formulas.len() == 1 {
        return formulas[0].1.clone();
    }

    let op = match mode {
        GeneratorCombineMode::Or => " | ",
        GeneratorCombineMode::And => " & ",
        GeneratorCombineMode::Xor => {
            // XOR of multiple formulas is complex; handle separately
            // For now, do pairwise XOR: (a ^ b) = (a | b) & !(a & b)
            // Better: count odd occurrences
            return format!("/* XOR of {} deterministic generators */", formulas.len());
        }
    };

    formulas
        .iter()
        .map(|(_, f)| f.clone())
        .collect::<Vec<_>>()
        .join(op)
}

/// Write XOR transitions for mixed deterministic/probabilistic generators.
/// This enumerates all 2^|D| deterministic states.
fn write_xor_mixed_transitions(out: &mut String, input_name: &str, cats: &CategorizedGenerators) {
    let n = input_name;
    let det_count = cats.deterministic.len();

    if det_count > 4 {
        writeln!(
            out,
            "  // WARNING: XOR with {} deterministic generators may cause state explosion",
            det_count
        )
        .ok();
    }

    for mask in 0..(1u32 << det_count) {
        let det_fire_count = mask.count_ones() as usize;
        let det_parity = det_fire_count % 2;

        let mut guard_parts = Vec::new();
        for (bit, (_, formula)) in cats.deterministic.iter().enumerate() {
            if (mask >> bit) & 1 == 1 {
                guard_parts.push(formula.clone());
            } else {
                guard_parts.push(format!("!{formula}"));
            }
        }
        let guard = guard_parts.join(" & ");

        let target_prob_parity = 1 - det_parity;
        let p_prob_odd =
            compute_combined_probability(&cats.probabilistic, GeneratorCombineMode::Xor);

        let p_fire = if target_prob_parity == 1 {
            p_prob_odd
        } else {
            1.0 - p_prob_odd
        };
        let p_no_fire = 1.0 - p_fire;

        if (p_fire - 1.0).abs() < 1e-9 {
            writeln!(out, "  [tick] ({guard}) -> (x_{n}' = 1);").ok();
        } else if p_fire.abs() < 1e-9 {
            writeln!(out, "  [tick] ({guard}) -> (x_{n}' = 0);").ok();
        } else {
            writeln!(
                out,
                "  [tick] ({guard}) -> {:.6}:(x_{n}' = 1) + {:.6}:(x_{n}' = 0);",
                p_fire, p_no_fire
            )
            .ok();
        }
    }
}

#[expect(clippy::needless_range_loop)]
fn write_neuron_module(out: &mut String, node: &Node, _graph: &SnnGraph, config: &PrismGenConfig, names: &NameMap) {
    let n = &names[&node.id];
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

    writeln!(out, "module {n}").ok();

    // Only declare state variable if refractory periods are enabled
    if model.enable_arp || model.enable_rrp {
        writeln!(
            out,
            "  // State: 0=normal{}{}",
            if model.enable_arp { ", 1=ARP" } else { "" },
            if model.enable_rrp { ", 2=RRP" } else { "" }
        )
        .ok();
        writeln!(out, "  s_{n} : [0..{max_state}] init 0;").ok();
    } else {
        writeln!(out, "  // No refractory periods - simplified model").ok();
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
        "  p_{n} : [P_MIN_{n}..P_MAX_{n}] init {};  // membrane potential",
        model.p_rest
    )
    .ok();
    writeln!(out).ok();

    let step = 1.0 / levels as f64;

    // State guard prefix - only needed if refractory periods are enabled
    let state_guard = if model.enable_arp || model.enable_rrp {
        format!("s_{n} = 0 & ")
    } else {
        String::new()
    };

    writeln!(
        out,
        "  // Normal period - firing on newPotential ({levels} levels)"
    )
    .ok();

    // Below threshold1: no spike
    writeln!(
        out,
        "  [tick] {state_guard}newPotential_{n} <= threshold1 -> (y_{n}' = 0) & (p_{n}' = newPotential_{n});"
    ).ok();

    // Threshold-based probabilistic firing (variable levels)
    for i in 0..(levels - 1) {
        let prob = (i + 1) as f64 * step;
        let no_spike_prob = 1.0 - prob;
        let spike_prob = prob;
        writeln!(
            out,
            "  [tick] {state_guard}newPotential_{n} > threshold{} & newPotential_{n} <= threshold{} -> {:.4}:(y_{n}' = 0) & (p_{n}' = newPotential_{n}) + {:.4}:(y_{n}' = 1) & (p_{n}' = P_reset);",
            i + 1, i + 2, no_spike_prob, spike_prob
        ).ok();
    }

    // Above top threshold: always spike
    writeln!(
        out,
        "  [tick] {state_guard}newPotential_{n} > threshold{levels} -> 1.0:(y_{n}' = 1) & (p_{n}' = P_reset);"
    )
    .ok();

    writeln!(out).ok();

    // Absolute refractory period (only if enabled)
    if model.enable_arp {
        writeln!(out, "  // Absolute refractory period").ok();
        writeln!(
            out,
            "  [tick] s_{n} = 1 & aref_{n} > 0 -> (aref_{n}' = aref_{n} - 1) & (y_{n}' = 0) & (p_{n}' = newPotential_{n});"
        ).ok();

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

    // Relative refractory period (only if both ARP and RRP enabled)
    if model.enable_arp && model.enable_rrp {
        let alpha = model.alpha as f64 / 100.0;

        writeln!(
            out,
            "  // Relative refractory period (alpha-scaled, newPotential guards)"
        )
        .ok();

        writeln!(
            out,
            "  [tick] s_{n} = 2 & rref_{n} > 0 & newPotential_{n} <= threshold1 -> (y_{n}' = 0) & (p_{n}' = newPotential_{n}) & (rref_{n}' = rref_{n} - 1);"
        ).ok();

        for i in 0..(levels - 1) {
            let base_prob = (i + 1) as f64 * step;
            let spike_prob = alpha * base_prob;
            let no_spike_prob = 1.0 - spike_prob;
            writeln!(
                out,
                "  [tick] s_{n} = 2 & rref_{n} > 0 & newPotential_{n} > threshold{} & newPotential_{n} <= threshold{} -> {:.4}:(y_{n}' = 0) & (p_{n}' = newPotential_{n}) & (rref_{n}' = rref_{n} - 1) + {:.4}:(y_{n}' = 1) & (p_{n}' = P_reset) & (aref_{n}' = ARP) & (rref_{n}' = 0) & (s_{n}' = 1);",
                i + 1, i + 2, no_spike_prob, spike_prob
            ).ok();
        }

        let max_spike_prob = alpha;
        let max_no_spike_prob = 1.0 - alpha;
        writeln!(
            out,
            "  [tick] s_{n} = 2 & rref_{n} > 0 & newPotential_{n} > threshold{levels} -> {:.4}:(y_{n}' = 0) & (p_{n}' = newPotential_{n}) & (rref_{n}' = rref_{n} - 1) + {:.4}:(y_{n}' = 1) & (p_{n}' = P_reset) & (aref_{n}' = ARP) & (rref_{n}' = 0) & (s_{n}' = 1);",
            max_no_spike_prob, max_spike_prob
        ).ok();

        // RRP ended - return to normal
        writeln!(
            out,
            "  [tick] s_{n} = 2 & rref_{n} = 0 -> (p_{n}' = P_reset) & (y_{n}' = 0) & (s_{n}' = 0);"
        ).ok();
    }

    writeln!(out, "endmodule").ok();
}

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

    // Output neurons
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

/// Generates PCTL property specifications for common verification queries.
pub fn generate_pctl_properties(graph: &SnnGraph) -> String {
    let mut out = String::with_capacity(1024);
    let names = build_name_map(graph);

    writeln!(out, "// Auto-generated PCTL properties from CogSpike").ok();
    writeln!(out, "const int T;  // Time bound parameter\n").ok();

    // Labels
    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        let n = &names[&node.id];
        writeln!(out, "label \"spike_{n}\" = (y_{n} = 1);").ok();
    }
    writeln!(out).ok();

    // Basic reachability
    writeln!(out, "// Reachability: Can neuron spike?").ok();
    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        let n = &names[&node.id];
        writeln!(out, "P=? [ F \"spike_{n}\" ]").ok();
    }
    writeln!(out).ok();

    // Bounded reachability
    writeln!(out, "// Bounded reachability: Spike within T steps").ok();
    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        let n = &names[&node.id];
        writeln!(out, "P=? [ F<=T \"spike_{n}\" ]").ok();
    }
    writeln!(out).ok();

    // Liveness / persistence
    writeln!(out, "// Persistence: Neuron keeps spiking").ok();
    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        let n = &names[&node.id];
        writeln!(out, "P>=1 [ G (F \"spike_{n}\") ]").ok();
    }
    writeln!(out).ok();

    // Refractory correctness check
    writeln!(out, "// Safety: No spikes during absolute refractory").ok();
    for node in &graph.nodes {
        if graph.is_input(node.id) {
            continue;
        }
        let n = &names[&node.id];
        writeln!(
            out,
            "P>=1 [ G ((s_{n} = 1) => (y_{n} = 0)) ]"
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
        let n = &names[&node.id];
        writeln!(out, "R{{\"spike_{n}_count\"}}=? [ C<=T ]").ok();
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
                ..Default::default()
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
                ..Default::default()
            },
            ..Default::default()
        };
        let prism = generate_prism_model(&graph, &config);

        // Should NOT have state variable when refractory is disabled
        assert!(!prism.contains("s_Neuron_A : [0.."));
        // Should have simplified model comment
        assert!(prism.contains("// No refractory periods - simplified model"));
        // Should NOT have aref variable declaration
        assert!(!prism.contains("aref_Neuron_A : [0..ARP]"));
        // Should NOT have refractory label (since ARP is disabled)
        assert!(!prism.contains("label \"refractory_Neuron_A\""));
    }

    #[test]
    fn test_rrp_disabled() {
        let graph = SnnGraph::demo_layout();

        let config = PrismGenConfig {
            model: ModelConfig {
                threshold_levels: 10,
                enable_arp: true,
                enable_rrp: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let prism = generate_prism_model(&graph, &config);

        // State should go to 0..1 (normal, ARP)
        assert!(prism.contains("s_Neuron_A : [0..1] init 0;"));
        // Should have ARP variable
        assert!(prism.contains("aref_Neuron_A : [0..ARP] init 0;"));
        // Should NOT have RRP variable
        assert!(!prism.contains("rref_Neuron_A : [0..RRP]"));
        // Refractory label should only reference s=1
        assert!(prism.contains("label \"refractory_Neuron_A\" = (s_Neuron_A = 1);"));
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

    // =========================================================================
    // Multi-Generator Tests
    // =========================================================================

    #[test]
    fn test_compute_combined_probability_or() {
        let probs = vec![0.3, 0.5];
        let result = compute_combined_probability(&probs, GeneratorCombineMode::Or);
        assert!((result - 0.65).abs() < 0.0001);
    }

    #[test]
    fn test_compute_combined_probability_and() {
        let probs = vec![0.3, 0.5];
        let result = compute_combined_probability(&probs, GeneratorCombineMode::And);
        assert!((result - 0.15).abs() < 0.0001);
    }

    #[test]
    fn test_compute_combined_probability_xor() {
        let probs = vec![0.3, 0.5];
        let result = compute_combined_probability(&probs, GeneratorCombineMode::Xor);
        assert!((result - 0.5).abs() < 0.0001);
    }

    #[test]
    fn test_pattern_to_formula_periodic() {
        let pattern = InputPattern::Periodic {
            period: 5,
            phase: 2,
        };
        let formula = pattern_to_formula(&pattern, "S1", 0);
        assert!(formula.is_some());
        let f = formula.unwrap();
        assert!(f.contains("mod(step + 2, 5) = 0"));
    }

    #[test]
    fn test_pattern_to_formula_burst() {
        let pattern = InputPattern::Burst {
            burst_length: 3,
            silence_length: 2,
        };
        let formula = pattern_to_formula(&pattern, "S2", 0);
        assert!(formula.is_some());
        let f = formula.unwrap();
        assert!(f.contains("mod(step, 5) < 3"));
    }

    #[test]
    fn test_pattern_to_formula_random_returns_none() {
        let pattern = InputPattern::Random { probability: 0.5 };
        let formula = pattern_to_formula(&pattern, "S1", 0);
        assert!(formula.is_none());
    }

    #[test]
    fn test_global_clock_added_for_periodic() {
        use crate::simulation::{GeneratorId, InputGenerator, InputNeuronConfig};
        use crate::snn::graph::NodeKind;

        // Create a simple graph with an input that has Periodic pattern
        let mut graph = SnnGraph::default();
        let input_id = graph.add_node("Input0", NodeKind::Neuron, [0.0, 0.0]);

        // Configure the input with a periodic pattern
        if let Some(node) = graph.node_mut(input_id) {
            let mut config = InputNeuronConfig::default();
            config.generators.push(InputGenerator {
                id: GeneratorId(0),
                label: "Periodic".to_string(),
                pattern: InputPattern::Periodic {
                    period: 10,
                    phase: 0,
                },
                active: true,
            });
            node.input_config = Some(config);
        }

        let prism_config = PrismGenConfig::default();
        let prism = generate_prism_model(&graph, &prism_config);

        // Should contain GlobalClock module
        assert!(
            prism.contains("module GlobalClock"),
            "GlobalClock module missing"
        );
        assert!(prism.contains("step :"), "step variable missing");

        // Should contain formula for the periodic pattern (label: "Input0")
        assert!(prism.contains("in_Input0_g0_fires"), "Generator formula missing");
    }

    #[test]
    fn test_multi_generator_or_mode() {
        use crate::simulation::{GeneratorId, InputGenerator, InputNeuronConfig};
        use crate::snn::graph::NodeKind;

        let mut graph = SnnGraph::default();
        let input_id = graph.add_node("Input0", NodeKind::Neuron, [0.0, 0.0]);

        // Configure with two random generators in OR mode
        if let Some(node) = graph.node_mut(input_id) {
            let mut config = InputNeuronConfig::default();
            config.combine_mode = GeneratorCombineMode::Or;
            config.generators.push(InputGenerator {
                id: GeneratorId(0),
                label: "Random1".to_string(),
                pattern: InputPattern::Random { probability: 0.3 },
                active: true,
            });
            config.generators.push(InputGenerator {
                id: GeneratorId(1),
                label: "Random2".to_string(),
                pattern: InputPattern::Random { probability: 0.5 },
                active: true,
            });
            node.input_config = Some(config);
        }

        let prism_config = PrismGenConfig::default();
        let prism = generate_prism_model(&graph, &prism_config);

        // Should contain combined probability (0.65 for OR)
        assert!(
            prism.contains("0.65") || prism.contains("0.650000"),
            "Combined OR probability missing"
        );
    }

    #[test]
    fn test_per_neuron_potential_bounds() {
        use crate::snn::graph::NodeKind;

        // Create a diamond topology: Input -> A -> (B, C) -> D
        let mut graph = SnnGraph::default();
        let input = graph.add_node("Input", NodeKind::Neuron, [0.0, 0.0]);
        let _a = graph.add_node("A", NodeKind::Neuron, [100.0, 0.0]);
        let b = graph.add_node("B", NodeKind::Neuron, [200.0, -50.0]);
        let c = graph.add_node("C", NodeKind::Neuron, [200.0, 50.0]);
        let d = graph.add_node("D", NodeKind::Neuron, [300.0, 0.0]);

        graph.add_edge(input, _a, 100);
        graph.add_edge(_a, b, 100);
        graph.add_edge(_a, c, 100);
        graph.add_edge(b, d, 100);
        graph.add_edge(c, d, 100);

        let config = PrismGenConfig::default();
        let prism = generate_prism_model(&graph, &config);

        // Check per-neuron constants are generated with named labels
        assert!(prism.contains("// Per-neuron potential bounds"));

        // A has 1 input: P_MAX should be based on weight 100 -> 150
        assert!(prism.contains("P_MAX_A = 150"));

        // D has 2 inputs: P_MAX should be based on weight 200 -> 300
        assert!(prism.contains("P_MAX_D = 300"));

        // All neurons should have P_MIN = 0 (no inhibitory)
        assert!(prism.contains("P_MIN_A = 0"));
        assert!(prism.contains("P_MIN_D = 0"));

        // Neuron modules should use per-neuron bounds with named labels
        assert!(prism.contains("p_D : [P_MIN_D..P_MAX_D]"));
    }
}
