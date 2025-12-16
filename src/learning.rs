//! Parameter learning module using Probabilistic Advice Back-Propagation.
//!
//! Implements the SHF (Should Have Fired) and SNHF (Should Not Have Fired)
//! algorithms adapted from the naco20.pdf paper for learning synaptic weights.

use crate::snn::graph::{EdgeId, NodeId, NodeKind, SnnGraph};
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

        if !is_inhibitory {
            // CASE: Excitatory Input
            // If predecessor is active, strengthen the link (Hebbian-like).
            // If inactive, tell it to become active (propagate SHF).

            // Weight update: increase by signal * predecessor activity
            let weight_delta = signal * p_pred;
            let new_weight = weight + weight_delta as f32;
            graph.update_weight(edge_id, new_weight);
            weight_changes.insert(edge_id, weight_delta);

            // Backpropagate: If predecessor has LOW prob, it needs SHF advice
            let propagated_signal = signal * weight as f64 * (1.0 - p_pred);
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
            let new_weight = weight + weight_delta as f32;
            graph.update_weight(edge_id, new_weight);
            weight_changes.insert(edge_id, weight_delta);

            // Backpropagate SNHF (Stop firing!)
            let propagated_signal = signal * weight.abs() as f64 * p_pred;
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

        if !is_inhibitory {
            // CASE: Excitatory Input
            // It's exciting us too much. Weaken the link.
            // If it is very active, tell it to stop (propagate SNHF).

            // Decrease weight
            let weight_delta = -(signal * p_pred);
            let new_weight = weight + weight_delta as f32;
            graph.update_weight(edge_id, new_weight);
            weight_changes.insert(edge_id, weight_delta);

            // Backpropagate SNHF
            let propagated_signal = signal * weight as f64 * p_pred;
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
            let new_weight = weight + weight_delta as f32;
            graph.update_weight(edge_id, new_weight);
            weight_changes.insert(edge_id, weight_delta);

            // If predecessor is NOT firing, we need it to start firing to inhibit us
            let propagated_signal = signal * weight.abs() as f64 * (1.0 - p_pred);
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
        let prob = match node.kind {
            NodeKind::Input => 1.0, // Inputs always fire
            _ => {
                // Estimate based on parameters
                // Higher leak_r means potential builds up more (higher firing prob)
                // This is a rough heuristic
                let base = 0.5;
                let leak_factor = node.params.leak_r as f64;
                (base * leak_factor).clamp(0.0, 1.0)
            }
        };
        probs.insert(node.id, prob);
    }

    probs
}

/// Collect output neurons for learning (nodes with target_probability set, or Output nodes).
pub fn collect_learning_targets(graph: &SnnGraph) -> Vec<NodeId> {
    let mut targets = Vec::new();

    // Add nodes with target_probability set (explicit learning targets)
    for node in &graph.nodes {
        if node.target_probability.is_some() && !targets.contains(&node.id) {
            targets.push(node.id);
        }
    }

    // If no explicit targets, use output neurons
    if targets.is_empty() {
        targets.extend(
            graph
                .nodes
                .iter()
                .filter(|n| n.kind == NodeKind::Output)
                .map(|n| n.id),
        );
    }

    // If still empty, use automatically detected outputs
    if targets.is_empty() {
        targets.extend(graph.output_neurons());
    }

    targets
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snn::graph::SnnGraph;

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
                if (edge.weight - initial).abs() > 0.0001 {
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
}
