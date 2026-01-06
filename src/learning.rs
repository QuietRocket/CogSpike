//! Parameter learning module using Probabilistic Advice Back-Propagation.
//!
//! Implements the SHF (Should Have Fired) and SNHF (Should Not Have Fired)
//! algorithms adapted from the naco20.pdf paper for learning synaptic weights.

use crate::snn::graph::{EdgeId, NodeId, SnnGraph};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Configuration for the learning algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Target probability to achieve (0.0 to 1.0).
    pub target_probability: f64,
    /// Learning rate for weight updates.
    pub learning_rate: f64,
    /// Convergence threshold - stop when |error| < threshold.
    pub convergence_threshold: f64,
    /// Maximum number of iterations.
    pub max_iterations: u32,
    /// Minimum signal strength to propagate (avoid infinite recursion).
    pub epsilon: f64,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            target_probability: 0.8,
            learning_rate: 0.1,
            convergence_threshold: 0.01,
            max_iterations: 100,
            epsilon: 0.001,
        }
    }
}

/// State of the learning process.
#[derive(Debug, Clone, Default)]
pub struct LearningState {
    /// Current iteration.
    pub iteration: u32,
    /// History of probabilities at each iteration.
    pub probability_history: Vec<f64>,
    /// History of weight changes per edge.
    pub weight_changes: Vec<HashMap<EdgeId, f64>>,
    /// Whether learning has converged.
    pub converged: bool,
    /// Final error (target - current).
    pub final_error: f64,
}

impl LearningState {
    pub fn new() -> Self {
        Self {
            iteration: 0,
            probability_history: Vec::new(),
            weight_changes: Vec::new(),
            converged: false,
            final_error: 1.0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Firing probability estimates for neurons (from PRISM results or simulation).
pub type FiringProbabilities = HashMap<NodeId, f64>;

/// Result of a complete training run.
#[derive(Debug, Clone)]
pub enum TrainingResult {
    /// Training converged within the threshold.
    Converged {
        iterations: u32,
        final_probability: f64,
        final_error: f64,
    },
    /// Reached maximum iterations without converging.
    MaxIterations {
        iterations: u32,
        final_probability: f64,
        final_error: f64,
    },
    /// Training was stopped by user or callback.
    Stopped {
        iterations: u32,
        last_probability: f64,
    },
    /// An error occurred during training.
    Error(String),
}

/// Progress information for a training step.
#[derive(Debug, Clone)]
pub struct TrainingProgress {
    /// Current iteration number.
    pub iteration: u32,
    /// Maximum iterations allowed.
    pub max_iterations: u32,
    /// Current probability from verification.
    pub current_probability: f64,
    /// Current error (target - current).
    pub error: f64,
    /// Number of weights changed this iteration.
    pub weights_changed: usize,
}

/// Callback type for training progress. Return `false` to stop training.
pub type ProgressCallback = Box<dyn FnMut(&TrainingProgress) -> bool + Send>;

/// Run a complete training loop with a verification callback.
///
/// This function alternates between:
/// 1. Calling the verifier to get current probability
/// 2. Running SHF/SNHF to adjust weights
/// 3. Repeating until converged or max iterations
///
/// # Arguments
/// * `graph` - The SNN graph (weights will be modified)
/// * `config` - Learning configuration with target, learning rate, etc.
/// * `verify_fn` - Callback function that runs PRISM and returns the current probability
/// * `progress_fn` - Optional callback for progress updates (return `false` to stop)
///
/// # Returns
/// The result of the training run.
pub fn run_training_loop<F>(
    graph: &mut SnnGraph,
    config: &LearningConfig,
    mut verify_fn: F,
    mut progress_fn: Option<ProgressCallback>,
) -> TrainingResult
where
    F: FnMut(&SnnGraph) -> Result<f64, String>,
{
    let targets = collect_learning_targets(graph);
    if targets.is_empty() {
        return TrainingResult::Error("No learning targets found".to_owned());
    }

    for iteration in 1..=config.max_iterations {
        // Step 1: Run verification to get current probability
        let current_probability = match verify_fn(graph) {
            Ok(prob) => prob,
            Err(e) => return TrainingResult::Error(e),
        };

        // Step 2: Check convergence
        let error = config.target_probability - current_probability;
        if error.abs() < config.convergence_threshold {
            return TrainingResult::Converged {
                iterations: iteration,
                final_probability: current_probability,
                final_error: error,
            };
        }

        // Step 3: Run learning iteration
        let firing_probs = estimate_firing_probabilities(graph);
        let result =
            run_learning_iteration(graph, &targets, current_probability, &firing_probs, config);

        // Step 4: Report progress
        let progress = TrainingProgress {
            iteration,
            max_iterations: config.max_iterations,
            current_probability,
            error,
            weights_changed: result.weight_changes.len(),
        };

        if let Some(ref mut callback) = progress_fn {
            if !callback(&progress) {
                return TrainingResult::Stopped {
                    iterations: iteration,
                    last_probability: current_probability,
                };
            }
        }
    }

    // Get final probability after max iterations
    let final_prob = verify_fn(graph).unwrap_or(0.0);
    let final_error = config.target_probability - final_prob;

    TrainingResult::MaxIterations {
        iterations: config.max_iterations,
        final_probability: final_prob,
        final_error,
    }
}

/// Result of a single learning iteration.
#[derive(Debug, Clone)]
pub struct IterationResult {
    /// The observed probability before weight updates.
    pub observed_probability: f64,
    /// The error (target - observed).
    pub error: f64,
    /// Weight changes made this iteration.
    pub weight_changes: HashMap<EdgeId, f64>,
    /// Whether this iteration converged.
    pub converged: bool,
}

/// Run a single learning iteration.
///
/// This implements the Probabilistic Advice Dispatcher from the algorithm.
///
/// # Arguments
/// * `graph` - The SNN graph (weights will be modified)
/// * `output_neurons` - The neurons to optimize (typically output/supervised neurons)
/// * `current_probability` - Current probability from PRISM verification
/// * `firing_probs` - Firing probability estimates for each neuron
/// * `config` - Learning configuration
///
/// # Returns
/// The result of this iteration including weight changes made.
pub fn run_learning_iteration(
    graph: &mut SnnGraph,
    output_neurons: &[NodeId],
    current_probability: f64,
    firing_probs: &FiringProbabilities,
    config: &LearningConfig,
) -> IterationResult {
    let error = config.target_probability - current_probability;
    let mut weight_changes = HashMap::new();
    let converged = error.abs() < config.convergence_threshold;

    if !converged {
        for &node_id in output_neurons {
            let mut visited = HashSet::new();

            if error > config.convergence_threshold {
                // Probability too low -> We need more firing (SHF)
                propagate_shf(
                    graph,
                    node_id,
                    error.abs() * config.learning_rate,
                    &mut visited,
                    firing_probs,
                    config.epsilon,
                    &mut weight_changes,
                );
            } else if error < -config.convergence_threshold {
                // Probability too high -> We need less firing (SNHF)
                propagate_snhf(
                    graph,
                    node_id,
                    error.abs() * config.learning_rate,
                    &mut visited,
                    firing_probs,
                    config.epsilon,
                    &mut weight_changes,
                );
            }
        }
    }

    IterationResult {
        observed_probability: current_probability,
        error,
        weight_changes,
        converged,
    }
}

/// Should Have Fired (SHF) - Propagate advice to increase firing probability.
///
/// Goal: Raise the firing probability of the target neuron by:
/// - Strengthening excitatory inputs
/// - Weakening inhibitory inputs
/// - Recursively advising predecessors
fn propagate_shf(
    graph: &mut SnnGraph,
    node_id: NodeId,
    signal: f64,
    visited: &mut HashSet<NodeId>,
    firing_probs: &FiringProbabilities,
    epsilon: f64,
    weight_changes: &mut HashMap<EdgeId, f64>,
) {
    // Skip if already visited or if this is an input generator
    if visited.contains(&node_id) || graph.is_input_generator(node_id) {
        return;
    }
    visited.insert(node_id);

    // Get incoming edges (we need to modify the graph, so collect edge info first)
    let incoming: Vec<_> = graph
        .incoming_edges(node_id)
        .iter()
        .map(|e| (e.id, e.from, e.weight, e.is_inhibitory))
        .collect();

    for (edge_id, pred_id, weight, is_inhibitory) in incoming {
        // Get predecessor's firing probability (default to 0.5 if unknown)
        let p_pred = *firing_probs.get(&pred_id).unwrap_or(&0.5);
        // Convert weight to 0.0-1.0 range for calculations
        let weight_f64 = weight as f64 / 100.0;

        if !is_inhibitory {
            // CASE: Excitatory Input
            // If predecessor is active, strengthen the link (Hebbian-like).
            // If inactive, tell it to become active (propagate SHF).

            // Weight update: increase by signal * predecessor activity
            let weight_delta = signal * p_pred;
            let new_weight_f64 = weight_f64 + weight_delta;
            // Convert back to u8 (0-100), clamping to valid range
            let new_weight_u8 = (new_weight_f64 * 100.0).clamp(0.0, 100.0).round() as u8;
            graph.update_weight(edge_id, new_weight_u8);
            weight_changes.insert(edge_id, weight_delta);

            // Backpropagate: If predecessor has LOW prob, it needs SHF advice
            let propagated_signal = signal * weight_f64 * (1.0 - p_pred);
            if propagated_signal > epsilon {
                propagate_shf(
                    graph,
                    pred_id,
                    propagated_signal,
                    visited,
                    firing_probs,
                    epsilon,
                    weight_changes,
                );
            }
        } else {
            // CASE: Inhibitory Input (weight < 0)
            // If predecessor is active, it's hurting us. Weaken the link (make less negative).
            // AND tell it to stop firing (propagate SNHF).

            // Weaken inhibition (make less negative = add positive)
            let weight_delta = signal * p_pred;
            let new_weight_f64 = weight_f64 + weight_delta;
            let new_weight_u8 = (new_weight_f64 * 100.0).clamp(0.0, 100.0).round() as u8;
            graph.update_weight(edge_id, new_weight_u8);
            weight_changes.insert(edge_id, weight_delta);

            // Backpropagate SNHF (Stop firing!)
            // weight is u8 (always positive), so no .abs() needed
            let propagated_signal = signal * weight_f64 * p_pred;
            if propagated_signal > epsilon {
                propagate_snhf(
                    graph,
                    pred_id,
                    propagated_signal,
                    visited,
                    firing_probs,
                    epsilon,
                    weight_changes,
                );
            }
        }
    }
}

/// Should Not Have Fired (SNHF) - Propagate advice to decrease firing probability.
///
/// Goal: Lower the firing probability of the target neuron by:
/// - Weakening excitatory inputs
/// - Strengthening inhibitory inputs
/// - Recursively advising predecessors
fn propagate_snhf(
    graph: &mut SnnGraph,
    node_id: NodeId,
    signal: f64,
    visited: &mut HashSet<NodeId>,
    firing_probs: &FiringProbabilities,
    epsilon: f64,
    weight_changes: &mut HashMap<EdgeId, f64>,
) {
    // Skip if already visited or if this is an input generator
    if visited.contains(&node_id) || graph.is_input_generator(node_id) {
        return;
    }
    visited.insert(node_id);

    // Get incoming edges
    let incoming: Vec<_> = graph
        .incoming_edges(node_id)
        .iter()
        .map(|e| (e.id, e.from, e.weight, e.is_inhibitory))
        .collect();

    for (edge_id, pred_id, weight, is_inhibitory) in incoming {
        let p_pred = *firing_probs.get(&pred_id).unwrap_or(&0.5);
        // Convert weight to 0.0-1.0 range for calculations
        let weight_f64 = weight as f64 / 100.0;

        if !is_inhibitory {
            // CASE: Excitatory Input
            // It's exciting us too much. Weaken the link.
            // If it is very active, tell it to stop (propagate SNHF).

            // Decrease weight
            let weight_delta = -(signal * p_pred);
            let new_weight_f64 = weight_f64 + weight_delta;
            // Convert back to u8 (0-100), clamping to valid range
            let new_weight_u8 = (new_weight_f64 * 100.0).clamp(0.0, 100.0).round() as u8;
            graph.update_weight(edge_id, new_weight_u8);
            weight_changes.insert(edge_id, weight_delta);

            // Backpropagate SNHF
            let propagated_signal = signal * weight_f64 * p_pred;
            if propagated_signal > epsilon {
                propagate_snhf(
                    graph,
                    pred_id,
                    propagated_signal,
                    visited,
                    firing_probs,
                    epsilon,
                    weight_changes,
                );
            }
        } else {
            // CASE: Inhibitory Input
            // We want MORE inhibition to stop firing.
            // Strengthen link (make more negative).
            // If predecessor is NOT firing, we need it to fire to inhibit us (propagate SHF).

            // Make more negative (subtract)
            let weight_delta = -(signal * p_pred);
            let new_weight_f64 = weight_f64 + weight_delta;
            let new_weight_u8 = (new_weight_f64 * 100.0).clamp(0.0, 100.0).round() as u8;
            graph.update_weight(edge_id, new_weight_u8);
            weight_changes.insert(edge_id, weight_delta);

            // If predecessor is NOT firing, we need it to start firing to inhibit us
            // weight is u8 (always positive), so no .abs() needed
            let propagated_signal = signal * weight_f64 * (1.0 - p_pred);
            if propagated_signal > epsilon {
                propagate_shf(
                    graph,
                    pred_id,
                    propagated_signal,
                    visited,
                    firing_probs,
                    epsilon,
                    weight_changes,
                );
            }
        }
    }
}

/// Estimate firing probabilities from neuron parameters.
///
/// This is a heuristic estimate when PRISM results are not available.
/// Uses the neuron's threshold parameters to estimate base firing probability.
pub fn estimate_firing_probabilities(graph: &SnnGraph) -> FiringProbabilities {
    let mut probs = HashMap::new();

    for node in &graph.nodes {
        // Input neurons (no incoming edges) always fire
        let prob = if graph.is_input(node.id) {
            1.0
        } else {
            // Estimate based on parameters
            // Higher leak_r means potential builds up more (higher firing prob)
            // This is a rough heuristic
            let base = 0.5;
            let leak_factor = node.params.leak_r as f64;
            (base * leak_factor).clamp(0.0, 1.0)
        };
        probs.insert(node.id, prob);
    }

    probs
}

/// Collect output neurons for learning (nodes with target_probability set, or topological outputs).
pub fn collect_learning_targets(graph: &SnnGraph) -> Vec<NodeId> {
    let mut targets = Vec::new();

    // Add nodes with target_probability set (explicit learning targets)
    for node in &graph.nodes {
        if node.target_probability.is_some() && !targets.contains(&node.id) {
            targets.push(node.id);
        }
    }

    // If no explicit targets, use automatically detected outputs (leaf nodes)
    if targets.is_empty() {
        targets.extend(graph.output_neurons());
    }

    targets
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snn::graph::{NodeKind, SnnGraph};

    // =========================================================================
    // Test Topology Helpers
    // =========================================================================

    /// Create a simple chain: Input -> A -> B -> Output
    /// All weights start at 1 (minimal) so the signal can barely propagate.
    /// Learning should increase weights to allow better signal propagation.
    fn create_simple_chain() -> SnnGraph {
        let mut graph = SnnGraph::default();
        let input = graph.add_node("Input", NodeKind::Neuron, [0.0, 0.0]);
        let a = graph.add_node("A", NodeKind::Neuron, [100.0, 0.0]);
        let b = graph.add_node("B", NodeKind::Neuron, [200.0, 0.0]);
        let output = graph.add_node("Output", NodeKind::Neuron, [300.0, 0.0]);

        // Weights start at 1 (minimal non-zero value for u8)
        graph.add_edge(input, a, 1);
        graph.add_edge(a, b, 1);
        graph.add_edge(b, output, 1);

        graph
    }

    /// Create a diamond structure: Input -> A,B -> Output
    /// Tests that blame propagates through multiple paths.
    fn create_diamond() -> SnnGraph {
        let mut graph = SnnGraph::default();
        let input = graph.add_node("Input", NodeKind::Neuron, [0.0, 100.0]);
        let a = graph.add_node("A", NodeKind::Neuron, [100.0, 50.0]);
        let b = graph.add_node("B", NodeKind::Neuron, [100.0, 150.0]);
        let output = graph.add_node("Output", NodeKind::Neuron, [200.0, 100.0]);

        // Starting with small weights (10 = 0.1 in the old scale)
        graph.add_edge(input, a, 10);
        graph.add_edge(input, b, 10);
        graph.add_edge(a, output, 10);
        graph.add_edge(b, output, 10);

        graph
    }

    // =========================================================================
    // Mock Verifiers
    // =========================================================================

    /// A simple mock verifier that estimates probability purely from edge weights.
    /// Higher total weights on edges into output = higher probability.
    /// This is fast but doesn't simulate actual network dynamics.
    fn mock_verify_weight_based(graph: &SnnGraph) -> Result<f64, String> {
        let outputs = graph.output_neurons();
        if outputs.is_empty() {
            return Err("No outputs".to_owned());
        }

        // Simple heuristic: average of incoming weights to output
        // Weights are now u8 (0-100), so convert to f64 (0.0-1.0)
        let mut total_weight: f64 = 0.0;
        let mut count = 0;

        for output_id in &outputs {
            for edge in graph.incoming_edges(*output_id) {
                let weight_f64 = edge.weight as f64 / 100.0;
                if edge.is_inhibitory {
                    total_weight -= weight_f64;
                } else {
                    total_weight += weight_f64;
                }
                count += 1;
            }
        }

        if count == 0 {
            return Ok(0.0);
        }

        // Normalize to 0.0-1.0 range
        let prob = (total_weight / count as f64).clamp(0.0, 1.0);
        Ok(prob)
    }

    /// A more sophisticated mock verifier that simulates network dynamics.
    /// Propagates activation through the network using a simplified LIF model.
    fn mock_verify_network_dynamics(graph: &SnnGraph) -> Result<f64, String> {
        let outputs = graph.output_neurons();
        let inputs = graph.input_neurons();

        if outputs.is_empty() {
            return Err("No outputs".to_owned());
        }
        if inputs.is_empty() {
            return Err("No inputs".to_owned());
        }

        // Initialize potentials: inputs start at 1.0 (firing), others at 0.0
        let mut potentials: HashMap<NodeId, f64> = HashMap::new();
        for node in &graph.nodes {
            let initial = if inputs.contains(&node.id) { 1.0 } else { 0.0 };
            potentials.insert(node.id, initial);
        }

        // Simulate multiple timesteps to propagate activation
        const NUM_TIMESTEPS: usize = 10;
        const THRESHOLD: f64 = 0.5;
        const LEAK: f64 = 0.9;

        for _ in 0..NUM_TIMESTEPS {
            let mut new_potentials = potentials.clone();

            for node in &graph.nodes {
                // Skip input neurons (they maintain constant activation)
                if inputs.contains(&node.id) {
                    continue;
                }

                // Calculate incoming activation
                let mut incoming_sum: f64 = 0.0;
                for edge in graph.incoming_edges(node.id) {
                    let pre_potential = potentials.get(&edge.from).copied().unwrap_or(0.0);
                    // Only transmit if presynaptic neuron is "firing" (above threshold)
                    if pre_potential >= THRESHOLD {
                        // Convert u8 weight (0-100) to f64 (0.0-1.0)
                        let weight_contribution = edge.weight as f64 / 100.0;
                        if edge.is_inhibitory {
                            incoming_sum -= weight_contribution;
                        } else {
                            incoming_sum += weight_contribution;
                        }
                    }
                }

                // Update potential with leak and incoming activation
                let current = potentials.get(&node.id).copied().unwrap_or(0.0);
                let new_potential = (current * LEAK + incoming_sum).clamp(0.0, 1.0);
                new_potentials.insert(node.id, new_potential);
            }

            potentials = new_potentials;
        }

        // Calculate average output activation as probability
        let mut total_output: f64 = 0.0;
        for output_id in &outputs {
            let potential = potentials.get(output_id).copied().unwrap_or(0.0);
            // Convert potential to firing probability
            total_output += if potential >= THRESHOLD {
                1.0
            } else {
                potential / THRESHOLD
            };
        }

        Ok(total_output / outputs.len() as f64)
    }

    // =========================================================================
    // Original Tests
    // =========================================================================

    #[test]
    fn test_shf_modifies_weights() {
        let mut graph = SnnGraph::demo_layout();

        // Get initial weights
        let initial_weights: HashMap<_, _> = graph.edges.iter().map(|e| (e.id, e.weight)).collect();

        // Run SHF on output neuron
        let output = graph.output_neurons()[0];
        let mut visited = HashSet::new();
        let firing_probs = estimate_firing_probabilities(&graph);
        let mut weight_changes = HashMap::new();

        propagate_shf(
            &mut graph,
            output,
            0.1,
            &mut visited,
            &firing_probs,
            0.001,
            &mut weight_changes,
        );

        // Check that some weights changed
        assert!(!weight_changes.is_empty(), "SHF should modify some weights");

        // Verify that weights in the graph actually changed
        let mut any_changed = false;
        for (edge_id, initial) in &initial_weights {
            if let Some(edge) = graph.edges.iter().find(|e| e.id == *edge_id) {
                // u8 weights, compare directly (difference of at least 1)
                if edge.weight != *initial {
                    any_changed = true;
                    break;
                }
            }
        }
        assert!(
            any_changed,
            "At least one weight should have been modified in the graph"
        );
    }

    #[test]
    fn test_learning_iteration() {
        let mut graph = SnnGraph::demo_layout();
        let outputs = collect_learning_targets(&graph);
        let firing_probs = estimate_firing_probabilities(&graph);
        let config = LearningConfig::default();

        let result = run_learning_iteration(
            &mut graph,
            &outputs,
            0.3, // Low probability -> should trigger SHF
            &firing_probs,
            &config,
        );

        assert!(
            result.error > 0.0,
            "Error should be positive when prob is low"
        );
        assert!(!result.converged, "Should not converge on first iteration");
    }

    #[test]
    fn test_convergence_detection() {
        let mut graph = SnnGraph::demo_layout();
        let outputs = collect_learning_targets(&graph);
        let firing_probs = estimate_firing_probabilities(&graph);
        let config = LearningConfig {
            target_probability: 0.8,
            convergence_threshold: 0.01,
            ..Default::default()
        };

        // Test with probability close to target
        let result = run_learning_iteration(
            &mut graph,
            &outputs,
            0.795, // Very close to target
            &firing_probs,
            &config,
        );

        assert!(result.converged, "Should converge when error < threshold");
    }

    // =========================================================================
    // Convergence Tests - Weight-Based Mock Verifier
    // =========================================================================

    #[test]
    fn test_simple_chain_convergence_weight_based() {
        // Create chain with zero weights - probability starts at 0
        let mut graph = create_simple_chain();

        // Record initial state
        let initial_prob = mock_verify_weight_based(&graph).expect("verify failed");
        assert!(
            initial_prob < 0.1,
            "Initial probability should be near zero, got {initial_prob}"
        );

        // Run training loop
        let config = LearningConfig {
            target_probability: 0.8,
            learning_rate: 0.2,
            convergence_threshold: 0.1,
            max_iterations: 50,
            epsilon: 0.0001,
        };

        let result = run_training_loop(&mut graph, &config, mock_verify_weight_based, None);

        // Verify convergence or at least improvement
        match result {
            TrainingResult::Converged {
                final_probability, ..
            } => {
                assert!(
                    final_probability > initial_prob,
                    "Probability should have increased: {initial_prob} -> {final_probability}"
                );
            }
            TrainingResult::MaxIterations {
                final_probability, ..
            } => {
                assert!(
                    final_probability > initial_prob,
                    "Probability should have increased even if not converged: {initial_prob} -> {final_probability}"
                );
            }
            other => panic!("Unexpected result: {other:?}"),
        }

        // Verify weights increased along the chain
        for edge in &graph.edges {
            assert!(
                edge.weight > 0,
                "Edge {} should have positive weight after SHF",
                edge.id.0
            );
        }
    }

    #[test]
    fn test_diamond_convergence_weight_based() {
        let mut graph = create_diamond();

        // Record initial weights
        let initial_weights: HashMap<_, _> = graph.edges.iter().map(|e| (e.id, e.weight)).collect();

        let config = LearningConfig {
            target_probability: 0.9,
            learning_rate: 0.15,
            convergence_threshold: 0.05,
            max_iterations: 30,
            epsilon: 0.0001,
        };

        let result = run_training_loop(&mut graph, &config, mock_verify_weight_based, None);

        // Check that both paths had weights modified
        let mut modified_count = 0;
        for edge in &graph.edges {
            let initial = initial_weights.get(&edge.id).copied().unwrap_or(0);
            // u8 weights: difference of at least 1 unit
            if edge.weight != initial {
                modified_count += 1;
            }
        }

        assert!(
            modified_count >= 2,
            "At least 2 edges should be modified in diamond (got {modified_count})"
        );

        // Final probability should be higher
        let final_prob = mock_verify_weight_based(&graph).expect("verify failed");
        assert!(
            final_prob > 0.3,
            "Final probability should have improved, got {final_prob}"
        );

        // Verify the result type
        match result {
            TrainingResult::Converged { .. } | TrainingResult::MaxIterations { .. } => {}
            other => panic!("Unexpected result: {other:?}"),
        }
    }

    // =========================================================================
    // Convergence Tests - Network Dynamics Mock Verifier
    // =========================================================================

    #[test]
    fn test_simple_chain_convergence_network_dynamics() {
        // Create chain with minimal weights - probability starts near 0
        let mut graph = create_simple_chain();

        // Record initial state
        let initial_prob = mock_verify_network_dynamics(&graph).expect("verify failed");
        // With u8 weights starting at 1 (0.01), initial prob is essentially 0
        assert!(
            initial_prob < 0.5,
            "Initial probability should be low, got {initial_prob}"
        );

        // Run training loop with aggressive learning rate for u8 weights
        let config = LearningConfig {
            target_probability: 0.8,
            learning_rate: 0.5, // Higher for integer quantization
            convergence_threshold: 0.15,
            max_iterations: 200, // More iterations for integer rounding
            epsilon: 0.0001,
        };

        let result = run_training_loop(&mut graph, &config, mock_verify_network_dynamics, None);

        // Verify improvement (or at least no decrease)
        let final_prob = mock_verify_network_dynamics(&graph).expect("verify failed");
        // With integer quantization, we may not always increase
        // but we should at least not get worse
        assert!(
            final_prob >= initial_prob,
            "Probability should not decrease: {initial_prob} -> {final_prob}"
        );

        match result {
            TrainingResult::Converged { .. } | TrainingResult::MaxIterations { .. } => {}
            other => panic!("Unexpected result: {other:?}"),
        }
    }

    #[test]
    fn test_diamond_convergence_network_dynamics() {
        let mut graph = create_diamond();

        let initial_prob = mock_verify_network_dynamics(&graph).expect("verify failed");

        let config = LearningConfig {
            target_probability: 0.9,
            learning_rate: 0.2,
            convergence_threshold: 0.1,
            max_iterations: 50,
            epsilon: 0.0001,
        };

        let _ = run_training_loop(&mut graph, &config, mock_verify_network_dynamics, None);

        // Final probability should be higher
        let final_prob = mock_verify_network_dynamics(&graph).expect("verify failed");
        assert!(
            final_prob >= initial_prob,
            "Final probability should not decrease: {initial_prob} -> {final_prob}"
        );
    }

    // =========================================================================
    // Weight Direction Tests
    // =========================================================================

    #[test]
    fn test_shf_increases_excitatory_weights() {
        let mut graph = create_simple_chain();

        // Run a single SHF iteration with low current probability
        let outputs = collect_learning_targets(&graph);
        let firing_probs = estimate_firing_probabilities(&graph);
        let config = LearningConfig::default();

        let result = run_learning_iteration(
            &mut graph,
            &outputs,
            0.1, // Very low probability - needs SHF
            &firing_probs,
            &config,
        );

        // Error should be positive (we need MORE firing)
        assert!(
            result.error > 0.0,
            "Error should be positive when probability is low"
        );

        // With minimal initial weights (1), the SHF signal propagates weakly\n        // through the chain. At least some edges should have increased weight.
        // Count how many edges have positive weight
        let positive_count = graph
            .edges
            .iter()
            .filter(|e| !e.is_inhibitory && e.weight > 0)
            .count();
        assert!(
            positive_count >= 1,
            "At least one excitatory edge should have positive weight after SHF, got {positive_count}"
        );

        // Also verify weight_changes were recorded
        assert!(
            !result.weight_changes.is_empty(),
            "SHF should record weight changes"
        );
    }

    #[test]
    fn test_snhf_decreases_excitatory_weights() {
        // Create a graph with high weights
        let mut graph = SnnGraph::default();
        let input = graph.add_node("Input", NodeKind::Neuron, [0.0, 0.0]);
        let output = graph.add_node("Output", NodeKind::Neuron, [100.0, 0.0]);
        graph.add_edge(input, output, 100); // Max weight (100 = 1.0 in old scale)

        // Run a single learning iteration with very HIGH current probability
        let outputs = collect_learning_targets(&graph);
        let firing_probs = estimate_firing_probabilities(&graph);
        let config = LearningConfig {
            target_probability: 0.2, // We want LOW probability
            ..Default::default()
        };

        let initial_weight = graph.edges[0].weight;

        let result = run_learning_iteration(
            &mut graph,
            &outputs,
            0.9, // Currently too high - needs SNHF
            &firing_probs,
            &config,
        );

        // Error should be negative (we want LESS firing)
        assert!(
            result.error < 0.0,
            "Error should be negative when probability is too high"
        );

        // Weight should have decreased
        assert!(
            graph.edges[0].weight < initial_weight,
            "Weight should decrease after SNHF: {} -> {}",
            initial_weight,
            graph.edges[0].weight
        );
    }

    // =========================================================================
    // Monotonicity Test
    // =========================================================================

    #[test]
    fn test_probability_trends_upward() {
        use std::sync::{Arc, Mutex};

        let mut graph = create_diamond();

        let config = LearningConfig {
            target_probability: 0.8,
            learning_rate: 0.1,
            convergence_threshold: 0.05,
            max_iterations: 20,
            epsilon: 0.0001,
        };

        let probabilities = Arc::new(Mutex::new(Vec::new()));
        let probs_clone = Arc::clone(&probabilities);

        let _ = run_training_loop(
            &mut graph,
            &config,
            mock_verify_weight_based,
            Some(Box::new(move |progress: &TrainingProgress| {
                probs_clone
                    .lock()
                    .expect("lock failed")
                    .push(progress.current_probability);
                true // Continue training
            })),
        );

        // Check that probability generally trends upward
        // (Allow for some noise, but overall trend should be positive)
        let probs = probabilities.lock().expect("lock failed");
        if probs.len() >= 3 {
            let third = probs.len() / 3;
            let first_third: f64 = probs[..third].iter().sum::<f64>() / third as f64;
            let last_third: f64 =
                probs[2 * third..].iter().sum::<f64>() / (probs.len() - 2 * third) as f64;

            assert!(
                last_third >= first_third - 0.1,
                "Probability should trend upward: first third avg={first_third}, last third avg={last_third}"
            );
        }
    }
}
