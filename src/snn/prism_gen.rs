//! PRISM model generator for Spiking Neural Networks.
//!
//! Generates DTMC (Discrete-Time Markov Chain) models from [`SnnGraph`] for
//! probabilistic model checking with PRISM.

use crate::simulation::{GeneratorCombineMode, InputNeuronConfig, InputPattern, ModelConfig};
use crate::snn::graph::{Node, SnnGraph};
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

    // Global constants from ModelConfig
    write_global_constants(&mut out, graph, config);
    writeln!(out).ok();

    // Threshold formulas
    write_threshold_formulas(&mut out, config);
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
    write_input_module(&mut out, graph, config);
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

fn write_global_constants(out: &mut String, graph: &SnnGraph, config: &PrismGenConfig) {
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
        let (n_min, n_max) = compute_neuron_potential_range(node, graph, config);
        writeln!(out, "const int P_MIN_{} = {};", node.id.0, n_min).ok();
        writeln!(out, "const int P_MAX_{} = {};", node.id.0, n_max).ok();
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
    for i in 1..=levels {
        // Generate thresholds matching simulation.rs generate_thresholds()
        let th = (i as u32 * m.p_rth as u32) / levels as u32;
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

        let n = node.id.0;
        let incoming: Vec<_> = graph.incoming_edges(node.id).into_iter().collect();

        if incoming.is_empty() {
            // No inputs, potential just decays
            writeln!(
                out,
                "formula newPotential_{n} = max(P_MIN_{n}, min(P_MAX_{n}, floor(r * p{n})));"
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
            "formula newPotential_{n} = max(P_MIN_{n}, min(P_MAX_{n}, floor(({input_sum}) + r * p{n})));"
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
fn pattern_to_formula(pattern: &InputPattern, input_id: u32, gen_idx: usize) -> Option<String> {
    let formula_name = format!("in{input_id}_g{gen_idx}");
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
    input_id: u32,
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
            deterministic.push((idx, format!("in{input_id}_g{idx}_fires")));
        }
    }

    CategorizedGenerators {
        deterministic,
        probabilistic,
    }
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

    // Write per-generator formulas for deterministic patterns
    writeln!(out, "// Input generator formulas").ok();
    for input in &inputs {
        let n = input.id.0;
        if let Some(ref cfg) = input.input_config {
            for (idx, generator) in cfg.generators.iter().enumerate() {
                if !generator.active {
                    continue;
                }
                if let Some(formula) = pattern_to_formula(&generator.pattern, n, idx) {
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
        writeln!(out, "  x{} : [0..1] init 0;", input.id.0).ok();
    }
    writeln!(out).ok();

    writeln!(out, "  // Multi-generator input transitions").ok();

    // Generate transitions for each input
    for input in &inputs {
        let n = input.id.0;

        if let Some(ref cfg) = input.input_config {
            let active_count = cfg.generators.iter().filter(|g| g.active).count();

            if active_count == 0 {
                // No active generators -> always off
                writeln!(out, "  [tick] true -> (x{n}' = 0);").ok();
                continue;
            }

            let cats = categorize_generators(cfg, n, dt_ms);
            let mode = cfg.combine_mode;

            // Generate transitions based on mode and generator types
            write_input_transitions(out, n, &cats, mode);
        } else {
            // No config -> legacy behavior (always on)
            writeln!(out, "  [tick] true -> (x{n}' = 1);").ok();
        }
    }

    writeln!(out, "endmodule").ok();
}

/// Write PRISM transitions for a single input neuron with categorized generators.
fn write_input_transitions(
    out: &mut String,
    input_id: u32,
    cats: &CategorizedGenerators,
    mode: GeneratorCombineMode,
) {
    let n = input_id;
    let has_det = !cats.deterministic.is_empty();
    let has_prob = !cats.probabilistic.is_empty();

    if !has_det && !has_prob {
        // No active generators
        writeln!(out, "  [tick] true -> (x{n}' = 0);").ok();
        return;
    }

    if !has_det {
        // All probabilistic - simple combined probability
        let p_fire = compute_combined_probability(&cats.probabilistic, mode);
        let p_no_fire = 1.0 - p_fire;
        writeln!(
            out,
            "  [tick] true -> {:.6}:(x{n}' = 1) + {:.6}:(x{n}' = 0);",
            p_fire, p_no_fire
        )
        .ok();
        return;
    }

    if !has_prob {
        // All deterministic - just use formula
        let combined_formula = combine_deterministic_formulas(&cats.deterministic, mode);
        writeln!(out, "  [tick] {combined_formula} -> (x{n}' = 1);").ok();
        writeln!(out, "  [tick] !({combined_formula}) -> (x{n}' = 0);").ok();
        return;
    }

    // Mixed deterministic and probabilistic
    let p_prob = compute_combined_probability(&cats.probabilistic, mode);

    match mode {
        GeneratorCombineMode::Or => {
            // OR: if any deterministic fires -> fire; else -> probabilistic
            let det_formula =
                combine_deterministic_formulas(&cats.deterministic, GeneratorCombineMode::Or);
            writeln!(out, "  [tick] ({det_formula}) -> (x{n}' = 1);").ok();
            writeln!(
                out,
                "  [tick] !({det_formula}) -> {:.6}:(x{n}' = 1) + {:.6}:(x{n}' = 0);",
                p_prob,
                1.0 - p_prob
            )
            .ok();
        }
        GeneratorCombineMode::And => {
            // AND: if any deterministic doesn't fire -> no fire; else -> probabilistic
            let det_formula =
                combine_deterministic_formulas(&cats.deterministic, GeneratorCombineMode::And);
            writeln!(out, "  [tick] !({det_formula}) -> (x{n}' = 0);").ok();
            writeln!(
                out,
                "  [tick] ({det_formula}) -> {:.6}:(x{n}' = 1) + {:.6}:(x{n}' = 0);",
                p_prob,
                1.0 - p_prob
            )
            .ok();
        }
        GeneratorCombineMode::Xor => {
            // XOR: need to enumerate deterministic parities
            // For simplicity with mixed patterns, we enumerate all 2^|D| states
            // This may cause state explosion for many deterministic generators
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
fn write_xor_mixed_transitions(out: &mut String, input_id: u32, cats: &CategorizedGenerators) {
    let n = input_id;
    let det_count = cats.deterministic.len();

    if det_count > 4 {
        // Warn about potential state explosion
        writeln!(
            out,
            "  // WARNING: XOR with {} deterministic generators may cause state explosion",
            det_count
        )
        .ok();
    }

    // For each subset of firing deterministic generators
    for mask in 0..(1u32 << det_count) {
        let det_fire_count = mask.count_ones() as usize;
        let det_parity = det_fire_count % 2;

        // Build guard: which formulas fire and which don't
        let mut guard_parts = Vec::new();
        for (bit, (_, formula)) in cats.deterministic.iter().enumerate() {
            if (mask >> bit) & 1 == 1 {
                guard_parts.push(formula.clone());
            } else {
                guard_parts.push(format!("!{formula}"));
            }
        }
        let guard = guard_parts.join(" & ");

        // Compute probabilistic contribution to parity
        // We need probability that prob generators produce parity that makes total odd
        let target_prob_parity = 1 - det_parity; // Need this parity from prob to get odd total

        // P(odd prob fires)
        let p_prob_odd =
            compute_combined_probability(&cats.probabilistic, GeneratorCombineMode::Xor);

        let p_fire = if target_prob_parity == 1 {
            p_prob_odd
        } else {
            1.0 - p_prob_odd
        };
        let p_no_fire = 1.0 - p_fire;

        if (p_fire - 1.0).abs() < 1e-9 {
            writeln!(out, "  [tick] ({guard}) -> (x{n}' = 1);").ok();
        } else if p_fire.abs() < 1e-9 {
            writeln!(out, "  [tick] ({guard}) -> (x{n}' = 0);").ok();
        } else {
            writeln!(
                out,
                "  [tick] ({guard}) -> {:.6}:(x{n}' = 1) + {:.6}:(x{n}' = 0);",
                p_fire, p_no_fire
            )
            .ok();
        }
    }
}

#[expect(clippy::needless_range_loop)]
fn write_neuron_module(out: &mut String, node: &Node, _graph: &SnnGraph, config: &PrismGenConfig) {
    let n = node.id.0;
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

    // Only declare state variable if refractory periods are enabled
    if model.enable_arp || model.enable_rrp {
        writeln!(
            out,
            "  // State: 0=normal{}{}",
            if model.enable_arp { ", 1=ARP" } else { "" },
            if model.enable_rrp { ", 2=RRP" } else { "" }
        )
        .ok();
        writeln!(out, "  s{n} : [0..{max_state}] init 0;").ok();
    } else {
        writeln!(out, "  // No refractory periods - simplified model").ok();
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
        "  p{} : [P_MIN_{}..P_MAX_{}] init {};  // membrane potential (per-neuron bounds)",
        n, n, n, model.p_rest
    )
    .ok();
    writeln!(out).ok();

    // Generate probability step size based on configured levels
    let step = 1.0 / levels as f64;

    // Normal period - spike reset (during tick after spike was set)
    // Note: We handle spike reset during [tick] to avoid deadlock with Inputs module
    writeln!(out, "  // Normal period - spike reset").ok();
    if model.enable_arp {
        writeln!(
            out,
            "  [tick] s{n} = 0 & y{n} = 1 -> (p{n}' = P_reset) & (aref{n}' = ARP) & (y{n}' = 0) & (s{n}' = 1);"
        ).ok();
    } else {
        // No ARP: spike and stay in normal state (no state variable to guard)
        writeln!(out, "  [tick] y{n} = 1 -> (p{n}' = P_reset) & (y{n}' = 0);").ok();
    }

    // Normal period - probabilistic transitions based on thresholds
    writeln!(
        out,
        "  // Normal period - probabilistic firing ({levels} levels)"
    )
    .ok();

    // State guard prefix - only needed if refractory periods are enabled
    let state_guard = if model.enable_arp || model.enable_rrp {
        format!("s{n} = 0 & ")
    } else {
        String::new()
    };

    // Below threshold1: no spike
    writeln!(
        out,
        "  [tick] {state_guard}y{n} = 0 & p{n} <= threshold1 -> (y{n}' = 0) & (p{n}' = newPotential_{n});"
    ).ok();

    // Threshold-based probabilistic firing (variable levels)
    for i in 0..(levels - 1) {
        let prob = (i + 1) as f64 * step;
        let no_spike_prob = 1.0 - prob;
        let spike_prob = prob;
        writeln!(
            out,
            "  [tick] {state_guard}y{n} = 0 & p{n} > threshold{} & p{n} <= threshold{} -> {:.4}:(y{n}' = 0) & (p{n}' = newPotential_{n}) + {:.4}:(y{n}' = 1);",
            i + 1, i + 2, no_spike_prob, spike_prob
        ).ok();
    }

    // Above top threshold: always spike
    writeln!(
        out,
        "  [tick] {state_guard}y{n} = 0 & p{n} > threshold{levels} -> 1.0:(y{n}' = 1);"
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
        let alpha = model.alpha as f64 / 100.0;

        writeln!(
            out,
            "  // Relative refractory period (alpha-scaled probabilities)"
        )
        .ok();
        // RRP spike reset (during tick after spike was set)
        writeln!(
            out,
            "  [tick] s{n} = 2 & y{n} = 1 & rref{n} > 0 -> (p{n}' = P_reset) & (aref{n}' = ARP) & (y{n}' = 0) & (rref{n}' = 0) & (s{n}' = 1);"
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

        // RRP above max threshold: spike with alpha-scaled probability (alpha * 1.0 = alpha)
        let max_spike_prob = alpha;
        let max_no_spike_prob = 1.0 - alpha;
        writeln!(
            out,
            "  [tick] s{n} = 2 & y{n} = 0 & rref{n} > 0 & p{n} > threshold{levels} -> {:.4}:(y{n}' = 0) & (p{n}' = newPotential_{n}) & (rref{n}' = rref{n} - 1) + {:.4}:(y{n}' = 1);",
            max_no_spike_prob, max_spike_prob
        ).ok();

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
    writeln!(
        out,
        "// Transfer variables sample source neuron's y output during tick"
    )
    .ok();

    for edge in &graph.edges {
        // Skip edges from Input nodes (they don't need transfer modules)
        if graph.is_input(edge.from) {
            continue;
        }

        // Transfer module captures source neuron's spike state (y) during each tick.
        // This replaces the previous [spike] action to avoid synchronization deadlocks.
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
        assert!(!prism.contains("s2 : [0.."));
        // Should have simplified model comment
        assert!(prism.contains("// No refractory periods - simplified model"));
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
                ..Default::default()
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

    // =========================================================================
    // Multi-Generator Tests
    // =========================================================================

    #[test]
    fn test_compute_combined_probability_or() {
        // OR: P(at least one) = 1 - (1-0.3)*(1-0.5) = 1 - 0.35 = 0.65
        let probs = vec![0.3, 0.5];
        let result = compute_combined_probability(&probs, GeneratorCombineMode::Or);
        assert!((result - 0.65).abs() < 0.0001);
    }

    #[test]
    fn test_compute_combined_probability_and() {
        // AND: P(all) = 0.3 * 0.5 = 0.15
        let probs = vec![0.3, 0.5];
        let result = compute_combined_probability(&probs, GeneratorCombineMode::And);
        assert!((result - 0.15).abs() < 0.0001);
    }

    #[test]
    fn test_compute_combined_probability_xor() {
        // XOR: P(exactly one) = 0.3*(1-0.5) + (1-0.3)*0.5 = 0.15 + 0.35 = 0.5
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
        let formula = pattern_to_formula(&pattern, 0, 0);
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
        let formula = pattern_to_formula(&pattern, 1, 0);
        assert!(formula.is_some());
        let f = formula.unwrap();
        assert!(f.contains("mod(step, 5) < 3"));
    }

    #[test]
    fn test_pattern_to_formula_random_returns_none() {
        let pattern = InputPattern::Random { probability: 0.5 };
        let formula = pattern_to_formula(&pattern, 0, 0);
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

        // Should contain formula for the periodic pattern (node id is 1 from add_node)
        assert!(prism.contains("in1_g0_fires"), "Generator formula missing");
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
        // A has 1 input (weight 100), D has 2 inputs (weight 100 each)
        let mut graph = SnnGraph::default();
        let input = graph.add_node("Input", NodeKind::Neuron, [0.0, 0.0]);
        let a = graph.add_node("A", NodeKind::Neuron, [100.0, 0.0]);
        let b = graph.add_node("B", NodeKind::Neuron, [200.0, -50.0]);
        let c = graph.add_node("C", NodeKind::Neuron, [200.0, 50.0]);
        let d = graph.add_node("D", NodeKind::Neuron, [300.0, 0.0]);

        graph.add_edge(input, a, 100);
        graph.add_edge(a, b, 100);
        graph.add_edge(a, c, 100);
        graph.add_edge(b, d, 100);
        graph.add_edge(c, d, 100);

        let config = PrismGenConfig::default();
        let prism = generate_prism_model(&graph, &config);

        // Check per-neuron constants are generated
        assert!(prism.contains("// Per-neuron potential bounds"));

        // A has 1 input: P_MAX should be based on weight 100 -> 150 (100 * 1.5)
        assert!(prism.contains(&format!("P_MAX_{} = 150", a.0)));

        // D has 2 inputs: P_MAX should be based on weight 200 -> 300 (200 * 1.5)
        assert!(prism.contains(&format!("P_MAX_{} = 300", d.0)));

        // All neurons should have P_MIN = 0 (no inhibitory)
        assert!(prism.contains(&format!("P_MIN_{} = 0", a.0)));
        assert!(prism.contains(&format!("P_MIN_{} = 0", d.0)));

        // Neuron modules should use per-neuron bounds
        assert!(prism.contains(&format!("p{} : [P_MIN_{}..P_MAX_{}]", d.0, d.0, d.0)));
    }
}
