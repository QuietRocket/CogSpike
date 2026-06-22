//! Simulation engine for Spiking Neural Networks.
//!
//! This module provides a discrete-time simulation engine that is mathematically
//! isomorphic with the PRISM verification model. The neuron model is a Leaky
//! Integrate-and-Fire (LIF) neuron with configurable refractory periods and
//! probabilistic firing thresholds.

use crate::snn::graph::{EdgeId, NodeId, SnnGraph};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Model Configuration
// ============================================================================

/// Global simulation/verification model configuration.
///
/// These settings affect both state space complexity and simulation behavior.
/// Fewer threshold levels and disabled refractory periods lead to smaller
/// PRISM state spaces.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Number of discrete firing probability levels (1-10).
    /// - 1 level = deterministic firing (fire if p > threshold)
    /// - 10 levels = original probabilistic model with 0.1 increments
    pub threshold_levels: u8,

    /// Enable Absolute Refractory Period.
    /// When disabled, neurons can fire immediately after spiking.
    pub enable_arp: bool,

    /// Enable Relative Refractory Period.
    /// When disabled, neurons skip the RRP state after ARP.
    pub enable_rrp: bool,

    // === Global Neuron Parameters (moved from per-neuron storage) ===
    /// Resting threshold potential (0-100, representing 0.0-1.0).
    /// Determines the potential level at which firing probability increases.
    pub p_rth: u8,

    /// Resting potential (0-100).
    pub p_rest: u8,

    /// Reset potential after spike (0-100).
    pub p_reset: u8,

    /// Leak rate (0-100, representing 0.0-1.0).
    /// Controls membrane potential decay each time step.
    pub leak_r: u8,

    /// Absolute refractory period duration (time steps).
    pub arp: u32,

    /// Relative refractory period duration (time steps).
    pub rrp: u32,

    /// Alpha scaling factor for RRP (0-100, representing 0.0-1.0).
    /// Scales firing probability during relative refractory period.
    pub alpha: u8,

    // === PRISM State Space Optimization ===
    /// Optional threshold for deriving PRISM potential range.
    /// When set, P_MIN = -2*threshold, P_MAX = 2*threshold.
    /// When None, uses the default range from PrismGenConfig.
    #[serde(default)]
    pub prism_potential_threshold: Option<u16>,
}

impl Default for ModelConfig {
    /// Default configuration optimized for fast PRISM verification.
    /// For biologically accurate models, use `ModelConfig::full()`.
    fn default() -> Self {
        Self {
            // Optimized for execution speed: 4 levels provides probabilistic
            // behavior while reducing state space by 60% vs 10 levels
            threshold_levels: 4,
            // Disabled by default for faster verification; enable via full()
            enable_arp: false,
            enable_rrp: false,
            // Global neuron parameters
            p_rth: 100,
            p_rest: 0,
            p_reset: 0,
            leak_r: 95,
            arp: 2,
            rrp: 4,
            alpha: 50,
            // Auto-derive range: P_MIN=-200, P_MAX=200 (401 states vs 1001)
            prism_potential_threshold: Some(100),
        }
    }
}

impl ModelConfig {
    /// Full biologically accurate configuration with refractory periods.
    /// Use when model accuracy is more important than verification speed.
    pub fn full() -> Self {
        Self {
            threshold_levels: 10,
            enable_arp: true,
            enable_rrp: true,
            prism_potential_threshold: None,
            ..Default::default()
        }
    }

    /// Configuration for minimal PRISM state space (deterministic firing, no refractory).
    /// Use for fastest possible verification with binary firing behavior.
    pub fn deterministic() -> Self {
        Self {
            threshold_levels: 1,
            enable_arp: false,
            enable_rrp: false,
            prism_potential_threshold: Some(100),
            ..Default::default()
        }
    }

    /// Validate and clamp threshold_levels to valid range.
    pub fn validate(&mut self) {
        self.threshold_levels = self.threshold_levels.clamp(1, 10);
    }

    /// Derive PRISM potential range from threshold, or None to use default.
    /// Range is centered at 0: (-2*threshold, 2*threshold)
    pub fn derive_potential_range(&self) -> Option<(i32, i32)> {
        self.prism_potential_threshold.map(|th| {
            let bound = i32::from(th) * 2;
            (-bound, bound)
        })
    }
}

// ============================================================================
// Input Pattern System
// ============================================================================

/// Unique identifier for an input generator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GeneratorId(pub u32);

/// Input spike generation patterns.
///
/// These patterns define how input neurons generate spikes over time.
/// External pattern injection is the primary mode; internal probabilistic
/// firing is also supported.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum InputPattern {
    /// Always firing: 1, 1, 1, 1, ... (constant high)
    AlwaysOn,

    /// Never firing: 0, 0, 0, 0, ... (constant low)
    AlwaysOff,

    /// Pulse of n ones followed by zeros: 1^n, 0, 0, 0, ...
    Pulse { duration: u32 },

    /// Silence of n zeros followed by ones: 0^n, 1, 1, 1, ...
    Silence { duration: u32 },

    /// Random (Bernoulli): Each step fires with probability p.
    Random { probability: f64 },

    /// Periodic: Fires every `period` steps with optional phase offset.
    Periodic { period: u32, phase: u32 },

    /// Burst: `burst_length` spikes, then silent for `silence_length` steps, repeat.
    Burst {
        burst_length: u32,
        silence_length: u32,
    },

    /// Poisson process with given rate (spikes/second).
    Poisson { rate_hz: f64 },

    /// Custom spike train (explicit time steps when the input fires).
    Custom { spike_times: Vec<u32> },

    /// Internal probabilistic firing (same model as regular neurons).
    InternalFiring,
}

impl Default for InputPattern {
    fn default() -> Self {
        Self::AlwaysOn
    }
}

impl InputPattern {
    /// Evaluate whether the input fires at the given time step.
    pub fn fires_at(&self, step: u32, dt_ms: f32, rng: &mut impl Rng) -> bool {
        match self {
            Self::AlwaysOn => true,
            Self::AlwaysOff => false,
            Self::Pulse { duration } => step < *duration,
            Self::Silence { duration } => step >= *duration,
            Self::Random { probability } => rng.r#gen::<f64>() < *probability,
            Self::Periodic { period, phase } => {
                if *period == 0 {
                    return false;
                }
                (step + *phase) % *period == 0
            }
            Self::Burst {
                burst_length,
                silence_length,
            } => {
                let cycle = burst_length + silence_length;
                if cycle == 0 {
                    return false;
                }
                (step % cycle) < *burst_length
            }
            Self::Poisson { rate_hz } => {
                // Convert rate to probability per step
                let step_duration_s = dt_ms as f64 / 1000.0;
                let prob = *rate_hz * step_duration_s;
                rng.r#gen::<f64>() < prob.min(1.0)
            }
            Self::Custom { spike_times } => spike_times.binary_search(&step).is_ok(),
            Self::InternalFiring => false, // Handled separately by neuron logic
        }
    }

    /// Human-readable label for this pattern type.
    pub fn label(&self) -> &'static str {
        match self {
            Self::AlwaysOn => "Always On",
            Self::AlwaysOff => "Always Off",
            Self::Pulse { .. } => "Pulse",
            Self::Silence { .. } => "Silence",
            Self::Random { .. } => "Random",
            Self::Periodic { .. } => "Periodic",
            Self::Burst { .. } => "Burst",
            Self::Poisson { .. } => "Poisson",
            Self::Custom { .. } => "Custom",
            Self::InternalFiring => "Internal",
        }
    }
}

/// Generator assigned to an input neuron.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputGenerator {
    /// Unique identifier for this generator.
    pub id: GeneratorId,
    /// Human-readable label.
    pub label: String,
    /// The pattern this generator produces.
    pub pattern: InputPattern,
    /// Whether this generator is currently active.
    pub active: bool,
}

impl InputGenerator {
    /// Create a new generator with the given pattern.
    pub fn new(id: GeneratorId, label: impl Into<String>, pattern: InputPattern) -> Self {
        Self {
            id,
            label: label.into(),
            pattern,
            active: true,
        }
    }
}

/// How multiple generators combine for a single input neuron.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GeneratorCombineMode {
    /// OR: Fire if ANY generator fires.
    #[default]
    Or,
    /// AND: Fire only if ALL active generators fire.
    And,
    /// XOR: Fire if ODD number of generators fire.
    Xor,
}

impl GeneratorCombineMode {
    /// All available combine modes.
    pub const ALL: [Self; 3] = [Self::Or, Self::And, Self::Xor];

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Or => "OR",
            Self::And => "AND",
            Self::Xor => "XOR",
        }
    }

    /// Combine a list of boolean fire values.
    pub fn combine(self, fires: &[bool]) -> bool {
        match self {
            Self::Or => fires.iter().any(|&f| f),
            Self::And => !fires.is_empty() && fires.iter().all(|&f| f),
            Self::Xor => fires.iter().filter(|&&f| f).count() % 2 == 1,
        }
    }
}

/// Input configuration per input neuron.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InputNeuronConfig {
    /// Generators attached to this input neuron.
    pub generators: Vec<InputGenerator>,
    /// How generators combine.
    pub combine_mode: GeneratorCombineMode,
    /// Counter for generating unique generator IDs.
    #[serde(default)]
    next_generator_id: u32,
}

impl InputNeuronConfig {
    /// Create with a single default generator.
    pub fn with_default_generator() -> Self {
        let mut config = Self::default();
        config.add_generator("Default", InputPattern::AlwaysOn);
        config
    }

    /// Add a generator with the given pattern.
    pub fn add_generator(
        &mut self,
        label: impl Into<String>,
        pattern: InputPattern,
    ) -> GeneratorId {
        let id = GeneratorId(self.next_generator_id);
        self.next_generator_id += 1;
        self.generators
            .push(InputGenerator::new(id, label, pattern));
        id
    }

    /// Remove a generator by ID.
    pub fn remove_generator(&mut self, id: GeneratorId) {
        self.generators.retain(|g| g.id != id);
    }

    /// Evaluate whether this input fires at the given step.
    pub fn fires_at(&self, step: u32, dt_ms: f32, rng: &mut impl Rng) -> bool {
        let active_fires: Vec<bool> = self
            .generators
            .iter()
            .filter(|g| g.active)
            .map(|g| g.pattern.fires_at(step, dt_ms, rng))
            .collect();

        if active_fires.is_empty() {
            return false;
        }

        self.combine_mode.combine(&active_fires)
    }
}

// ============================================================================
// Simulation State
// ============================================================================

/// State of a single neuron during simulation.
#[derive(Clone, Debug)]
pub struct NeuronSimState {
    /// Current state machine state:
    /// - 0 = Normal
    /// - 1 = ARP (Absolute Refractory Period)
    /// - 2 = RRP (Relative Refractory Period)
    pub state: u8,
    /// Membrane potential (P_MIN..P_MAX range).
    pub potential: i32,
    /// Spike output for current time step (0 or 1).
    pub spike_output: u8,
    /// Absolute refractory counter (only used if enable_arp).
    pub arp_counter: u32,
    /// Relative refractory counter (only used if enable_rrp).
    pub rrp_counter: u32,
}

impl NeuronSimState {
    /// Create initial state for a neuron with given rest potential.
    pub fn new(p_rest: i32) -> Self {
        Self {
            state: 0,
            potential: p_rest,
            spike_output: 0,
            arp_counter: 0,
            rrp_counter: 0,
        }
    }
}

/// Transfer state for spike propagation (one per synapse).
#[derive(Clone, Debug, Default)]
pub struct TransferState {
    /// Current value of transfer variable (0 or 1).
    pub z: u8,
}

/// Record of a single spike event.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpikeEvent {
    pub neuron_id: NodeId,
    pub time_step: u32,
}

/// Recorded history for visualization.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SimulationHistory {
    /// All spike events (sorted by time).
    pub spikes: Vec<SpikeEvent>,
    /// Membrane potential traces per neuron (only if recording enabled).
    pub potentials: HashMap<NodeId, Vec<i32>>,
    /// Spike counts per neuron.
    pub spike_counts: HashMap<NodeId, u32>,
}

impl SimulationHistory {
    /// Get firing rate for a neuron in Hz.
    pub fn firing_rate(&self, neuron_id: NodeId, total_time_ms: f32) -> f64 {
        let count = self.spike_counts.get(&neuron_id).copied().unwrap_or(0);
        (count as f64 / total_time_ms as f64) * 1000.0
    }
}

// ============================================================================
// Simulation Configuration
// ============================================================================

/// Configuration for simulation runs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Duration in milliseconds.
    pub duration_ms: f32,
    /// Potential range for clamping.
    pub potential_range: (i32, i32),
    /// Random seed for reproducibility (None = random).
    pub seed: Option<u64>,
    /// Whether to record membrane potentials (memory intensive).
    pub record_potentials: bool,
    /// Whether to record spike events.
    pub record_spikes: bool,
    /// Model configuration (thresholds, refractory toggles).
    pub model: ModelConfig,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            duration_ms: 250.0,
            potential_range: (-500, 500),
            seed: None,
            record_potentials: false,
            record_spikes: true,
            model: ModelConfig::default(),
        }
    }
}

// ============================================================================
// Simulation Result
// ============================================================================

/// Result of a simulation run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationResult {
    /// Recorded history for visualization.
    pub history: SimulationHistory,
    /// Total time steps simulated.
    pub total_steps: u32,
    /// Time step duration in ms.
    pub dt_ms: f32,
    /// Configuration used.
    pub config: SimulationConfig,
}

impl SimulationResult {
    /// Total simulation time in milliseconds.
    pub fn total_time_ms(&self) -> f32 {
        self.total_steps as f32 * self.dt_ms
    }
}

// ============================================================================
// Background Simulation Job
// ============================================================================

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc,
};

/// Progress update from a running simulation.
#[derive(Clone, Debug)]
pub struct SimulationProgress {
    pub current_step: u32,
    pub total_steps: u32,
    pub spike_count: u32,
}

impl SimulationProgress {
    /// Progress as a fraction 0.0 to 1.0.
    pub fn fraction(&self) -> f32 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.current_step as f32 / self.total_steps as f32
        }
    }
}

/// Handle for a background simulation job.
pub struct SimulationJob {
    /// When the simulation started.
    pub started_at: std::time::Instant,
    /// Receiver for progress updates.
    progress_rx: mpsc::Receiver<SimulationProgress>,
    /// Receiver for the final result.
    result_rx: mpsc::Receiver<SimulationResult>,
    /// Latest progress received.
    pub latest_progress: Option<SimulationProgress>,
    /// Flag to request cancellation.
    stop_requested: Arc<AtomicBool>,
}

impl SimulationJob {
    /// Request the simulation to stop.
    pub fn request_stop(&self) {
        self.stop_requested.store(true, Ordering::SeqCst);
    }

    /// Check if stop has been requested.
    pub fn is_stop_requested(&self) -> bool {
        self.stop_requested.load(Ordering::SeqCst)
    }

    /// Poll for progress updates (non-blocking).
    pub fn poll_progress(&mut self) -> Option<SimulationProgress> {
        let mut latest = None;
        while let Ok(progress) = self.progress_rx.try_recv() {
            latest = Some(progress);
        }
        if latest.is_some() {
            self.latest_progress = latest.clone();
        }
        latest
    }

    /// Check if simulation completed and get result (non-blocking).
    pub fn try_get_result(&self) -> Option<SimulationResult> {
        self.result_rx.try_recv().ok()
    }
}

/// Start a simulation in a background thread.
pub fn start_simulation_job(graph: SnnGraph, config: SimulationConfig) -> SimulationJob {
    let (progress_tx, progress_rx) = mpsc::channel();
    let (result_tx, result_rx) = mpsc::channel();
    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_flag_clone = stop_flag.clone();

    std::thread::spawn(move || {
        let result = run_simulation_with_progress(&graph, &config, &progress_tx, &stop_flag_clone);
        let _ = result_tx.send(result);
    });

    SimulationJob {
        started_at: std::time::Instant::now(),
        progress_rx,
        result_rx,
        latest_progress: None,
        stop_requested: stop_flag,
    }
}

/// Run simulation with progress reporting.
fn run_simulation_with_progress(
    graph: &SnnGraph,
    config: &SimulationConfig,
    progress_tx: &mpsc::Sender<SimulationProgress>,
    stop_flag: &Arc<AtomicBool>,
) -> SimulationResult {
    let dt_ms = graph
        .nodes
        .first()
        .map(|n| n.params.dt as f32 / 10.0)
        .unwrap_or(0.1);
    let total_steps = (config.duration_ms / dt_ms).ceil() as u32;

    let mut rng: rand::rngs::StdRng = match config.seed {
        Some(seed) => rand::SeedableRng::seed_from_u64(seed),
        None => rand::SeedableRng::from_entropy(),
    };

    let mut neurons: HashMap<NodeId, NeuronSimState> = graph
        .nodes
        .iter()
        .map(|n| (n.id, NeuronSimState::new(config.model.p_rest as i32)))
        .collect();

    let mut transfers: HashMap<EdgeId, TransferState> = graph
        .edges
        .iter()
        .filter(|e| !graph.is_input(e.from))
        .map(|e| (e.id, TransferState::default()))
        .collect();

    let mut history = SimulationHistory::default();
    for node in &graph.nodes {
        history.spike_counts.insert(node.id, 0);
        if config.record_potentials {
            history
                .potentials
                .insert(node.id, Vec::with_capacity(total_steps as usize));
        }
    }

    let progress_interval = (total_steps / 100).max(1);

    for step in 0..total_steps {
        // Check for cancellation
        if stop_flag.load(Ordering::SeqCst) {
            break;
        }

        generate_input_spikes(graph, &mut neurons, config, step, dt_ms, &mut rng);

        // Record input spikes to history
        for node in &graph.nodes {
            if graph.is_input(node.id) {
                if let Some(neuron) = neurons.get(&node.id) {
                    if neuron.spike_output == 1 {
                        *history.spike_counts.entry(node.id).or_insert(0) += 1;
                        if config.record_spikes {
                            history.spikes.push(SpikeEvent {
                                neuron_id: node.id,
                                time_step: step,
                            });
                        }
                    }
                }
            }
        }

        let updates = compute_all_neuron_updates(graph, &neurons, &transfers, config, &mut rng);

        for (node_id, update) in updates {
            if let Some(neuron) = neurons.get_mut(&node_id) {
                neuron.state = update.new_state;
                neuron.potential = update.new_potential;
                neuron.spike_output = update.new_spike;
                neuron.arp_counter = update.new_arp;
                neuron.rrp_counter = update.new_rrp;

                if update.new_spike == 1 {
                    *history.spike_counts.entry(node_id).or_insert(0) += 1;
                    if config.record_spikes {
                        history.spikes.push(SpikeEvent {
                            neuron_id: node_id,
                            time_step: step,
                        });
                    }
                }
            }
        }

        for edge in &graph.edges {
            if graph.is_input(edge.from) {
                continue;
            }
            if let Some(transfer) = transfers.get_mut(&edge.id) {
                if let Some(source) = neurons.get(&edge.from) {
                    transfer.z = source.spike_output;
                }
            }
        }

        if config.record_potentials {
            for (node_id, neuron) in &neurons {
                if let Some(trace) = history.potentials.get_mut(node_id) {
                    trace.push(neuron.potential);
                }
            }
        }

        // Send progress update
        if step % progress_interval == 0 {
            let total_spikes: u32 = history.spike_counts.values().sum();
            let _ = progress_tx.send(SimulationProgress {
                current_step: step,
                total_steps,
                spike_count: total_spikes,
            });
        }
    }

    SimulationResult {
        history,
        total_steps,
        dt_ms,
        config: config.clone(),
    }
}

// ============================================================================
// Simulation Engine
// ============================================================================

/// Potential range constants.
const P_MIN: i32 = -500;
const P_MAX: i32 = 500;

/// Run a complete simulation.
pub fn run_simulation(graph: &SnnGraph, config: &SimulationConfig) -> SimulationResult {
    // Determine time step from first neuron or use default
    let dt_ms = graph
        .nodes
        .first()
        .map(|n| n.params.dt as f32 / 10.0)
        .unwrap_or(0.1);
    let total_steps = (config.duration_ms / dt_ms).ceil() as u32;

    // Initialize RNG
    let mut rng: rand::rngs::StdRng = match config.seed {
        Some(seed) => rand::SeedableRng::seed_from_u64(seed),
        None => rand::SeedableRng::from_entropy(),
    };

    // Initialize neuron states
    let mut neurons: HashMap<NodeId, NeuronSimState> = graph
        .nodes
        .iter()
        .map(|n| (n.id, NeuronSimState::new(config.model.p_rest as i32)))
        .collect();

    // Initialize transfer states for neuron-to-neuron edges
    let mut transfers: HashMap<EdgeId, TransferState> = graph
        .edges
        .iter()
        .filter(|e| !graph.is_input(e.from))
        .map(|e| (e.id, TransferState::default()))
        .collect();

    // Initialize history
    let mut history = SimulationHistory::default();
    for node in &graph.nodes {
        history.spike_counts.insert(node.id, 0);
        if config.record_potentials {
            history
                .potentials
                .insert(node.id, Vec::with_capacity(total_steps as usize));
        }
    }

    // Main simulation loop
    for step in 0..total_steps {
        // Phase 1: Generate input spikes
        generate_input_spikes(graph, &mut neurons, config, step, dt_ms, &mut rng);

        // Phase 1b: Record input spikes to history
        for node in &graph.nodes {
            if graph.is_input(node.id) {
                if let Some(neuron) = neurons.get(&node.id) {
                    if neuron.spike_output == 1 {
                        *history.spike_counts.entry(node.id).or_insert(0) += 1;
                        if config.record_spikes {
                            history.spikes.push(SpikeEvent {
                                neuron_id: node.id,
                                time_step: step,
                            });
                        }
                    }
                }
            }
        }

        // Phase 2: Compute neuron updates
        let updates = compute_all_neuron_updates(graph, &neurons, &transfers, config, &mut rng);

        // Phase 3: Apply updates
        for (node_id, update) in updates {
            if let Some(neuron) = neurons.get_mut(&node_id) {
                neuron.state = update.new_state;
                neuron.potential = update.new_potential;
                neuron.spike_output = update.new_spike;
                neuron.arp_counter = update.new_arp;
                neuron.rrp_counter = update.new_rrp;

                // Record spike event
                if update.new_spike == 1 {
                    *history.spike_counts.entry(node_id).or_insert(0) += 1;
                    if config.record_spikes {
                        history.spikes.push(SpikeEvent {
                            neuron_id: node_id,
                            time_step: step,
                        });
                    }
                }
            }
        }

        // Phase 4: Update transfer variables
        for edge in &graph.edges {
            if graph.is_input(edge.from) {
                continue;
            }
            if let Some(transfer) = transfers.get_mut(&edge.id) {
                if let Some(source) = neurons.get(&edge.from) {
                    transfer.z = source.spike_output;
                }
            }
        }

        // Phase 5: Record potentials
        if config.record_potentials {
            for (node_id, neuron) in &neurons {
                if let Some(trace) = history.potentials.get_mut(node_id) {
                    trace.push(neuron.potential);
                }
            }
        }
    }

    SimulationResult {
        history,
        total_steps,
        dt_ms,
        config: config.clone(),
    }
}

/// Generate input spikes for input neurons.
fn generate_input_spikes(
    graph: &SnnGraph,
    neurons: &mut HashMap<NodeId, NeuronSimState>,
    _config: &SimulationConfig,
    step: u32,
    dt_ms: f32,
    rng: &mut impl Rng,
) {
    for node in &graph.nodes {
        if !graph.is_input(node.id) {
            continue;
        }

        // Read input config from the node's stored config (configured in Design tab)
        let fires = if let Some(ref input_config) = node.input_config {
            input_config.fires_at(step, dt_ms, rng)
        } else {
            // Default: always on if no config set
            true
        };

        if let Some(neuron) = neurons.get_mut(&node.id) {
            neuron.spike_output = if fires { 1 } else { 0 };
        }
    }
}

/// Neuron update result.
#[derive(Clone, Debug)]
struct NeuronUpdate {
    new_state: u8,
    new_potential: i32,
    new_spike: u8,
    new_arp: u32,
    new_rrp: u32,
}

/// Compute updates for all non-input neurons.
fn compute_all_neuron_updates(
    graph: &SnnGraph,
    neurons: &HashMap<NodeId, NeuronSimState>,
    transfers: &HashMap<EdgeId, TransferState>,
    config: &SimulationConfig,
    rng: &mut impl Rng,
) -> Vec<(NodeId, NeuronUpdate)> {
    graph
        .nodes
        .iter()
        .filter(|n| !graph.is_input(n.id))
        .filter_map(|node| {
            let neuron = neurons.get(&node.id)?;
            let update =
                compute_neuron_update(node, neuron, graph, neurons, transfers, config, rng);
            Some((node.id, update))
        })
        .collect()
}

/// Compute update for a single neuron.
fn compute_neuron_update(
    node: &crate::snn::graph::Node,
    neuron: &NeuronSimState,
    graph: &SnnGraph,
    neurons: &HashMap<NodeId, NeuronSimState>,
    transfers: &HashMap<EdgeId, TransferState>,
    config: &SimulationConfig,
    rng: &mut impl Rng,
) -> NeuronUpdate {
    let model = &config.model;

    // Convert parameters from global ModelConfig
    let leak_r = model.leak_r as f64 / 100.0;
    let alpha = model.alpha as f64 / 100.0;

    // Generate thresholds based on configured levels
    let thresholds = generate_thresholds(model);

    // Calculate weighted input sum
    let input_sum = calculate_weighted_inputs(node.id, graph, neurons, transfers);

    // New potential with leak and input
    let new_potential = ((leak_r * neuron.potential as f64) + input_sum as f64).floor() as i32;
    let new_potential = new_potential.clamp(P_MIN, P_MAX);

    match neuron.state {
        0 => handle_normal_state(neuron, new_potential, &thresholds, model, rng),
        1 => handle_arp_state(neuron, new_potential, model),
        2 => handle_rrp_state(neuron, new_potential, &thresholds, alpha, model, rng),
        _ => NeuronUpdate {
            new_state: 0,
            new_potential,
            new_spike: 0,
            new_arp: 0,
            new_rrp: 0,
        },
    }
}

/// Generate threshold values based on configured levels.
fn generate_thresholds(model: &ModelConfig) -> Vec<i32> {
    let p_rth = model.p_rth as i32;
    let levels = model.threshold_levels.clamp(1, 10);
    (1..=levels)
        .map(|i| (i as i32 * p_rth) / levels as i32)
        .collect()
}

/// Determine firing probability based on potential and thresholds.
fn determine_fire_probability(potential: i32, thresholds: &[i32]) -> f64 {
    let levels = thresholds.len();
    if levels == 0 {
        return if potential > 0 { 1.0 } else { 0.0 };
    }

    let step = 1.0 / levels as f64;

    for (i, &th) in thresholds.iter().enumerate() {
        if potential <= th {
            return i as f64 * step;
        }
    }
    1.0
}

/// Calculate weighted input sum for a neuron.
fn calculate_weighted_inputs(
    node_id: NodeId,
    graph: &SnnGraph,
    neurons: &HashMap<NodeId, NeuronSimState>,
    transfers: &HashMap<EdgeId, TransferState>,
) -> i32 {
    graph
        .incoming_edges(node_id)
        .iter()
        .map(|edge| {
            let spike = if graph.is_input(edge.from) {
                neurons.get(&edge.from).map(|n| n.spike_output).unwrap_or(0)
            } else {
                transfers.get(&edge.id).map(|t| t.z).unwrap_or(0)
            };
            edge.signed_weight() as i32 * spike as i32
        })
        .sum()
}

/// Handle normal state neuron update.
fn handle_normal_state(
    _neuron: &NeuronSimState,
    new_potential: i32,
    thresholds: &[i32],
    model: &ModelConfig,
    rng: &mut impl Rng,
) -> NeuronUpdate {
    let fire_prob = determine_fire_probability(new_potential, thresholds);
    let fires = rng.r#gen::<f64>() < fire_prob;

    if fires {
        let next_state = if model.enable_arp { 1 } else { 0 };
        NeuronUpdate {
            new_state: next_state,
            new_potential: model.p_reset as i32,
            new_spike: 1,
            new_arp: if model.enable_arp { model.arp } else { 0 },
            new_rrp: 0,
        }
    } else {
        NeuronUpdate {
            new_state: 0,
            new_potential,
            new_spike: 0,
            new_arp: 0,
            new_rrp: 0,
        }
    }
}

/// Handle ARP state neuron update.
fn handle_arp_state(
    neuron: &NeuronSimState,
    new_potential: i32,
    model: &ModelConfig,
) -> NeuronUpdate {
    if neuron.arp_counter > 1 {
        NeuronUpdate {
            new_state: 1,
            new_potential,
            new_spike: 0,
            new_arp: neuron.arp_counter - 1,
            new_rrp: 0,
        }
    } else {
        // ARP finished
        let next_state = if model.enable_rrp { 2 } else { 0 };
        let next_rrp = if model.enable_rrp { model.rrp } else { 0 };
        NeuronUpdate {
            new_state: next_state,
            new_potential,
            new_spike: 0,
            new_arp: 0,
            new_rrp: next_rrp,
        }
    }
}

/// Handle RRP state neuron update.
fn handle_rrp_state(
    neuron: &NeuronSimState,
    new_potential: i32,
    thresholds: &[i32],
    alpha: f64,
    model: &ModelConfig,
    rng: &mut impl Rng,
) -> NeuronUpdate {
    if neuron.rrp_counter == 0 {
        // RRP finished
        return NeuronUpdate {
            new_state: 0,
            new_potential: model.p_reset as i32,
            new_spike: 0,
            new_arp: 0,
            new_rrp: 0,
        };
    }

    // Alpha-scaled firing probability
    let base_prob = determine_fire_probability(new_potential, thresholds);
    let fire_prob = alpha * base_prob;
    let fires = rng.r#gen::<f64>() < fire_prob;

    if fires {
        NeuronUpdate {
            new_state: 1,
            new_potential: model.p_reset as i32,
            new_spike: 1,
            new_arp: if model.enable_arp { model.arp } else { 0 },
            new_rrp: 0,
        }
    } else {
        NeuronUpdate {
            new_state: 2,
            new_potential,
            new_spike: 0,
            new_arp: 0,
            new_rrp: neuron.rrp_counter - 1,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snn::graph::{NodeKind, SnnGraph};

    #[test]
    fn test_input_pattern_always_on() {
        let mut rng = rand::thread_rng();
        let pattern = InputPattern::AlwaysOn;
        for step in 0..100 {
            assert!(pattern.fires_at(step, 0.1, &mut rng));
        }
    }

    #[test]
    fn test_input_pattern_pulse() {
        let mut rng = rand::thread_rng();
        let pattern = InputPattern::Pulse { duration: 5 };
        assert!(pattern.fires_at(0, 0.1, &mut rng));
        assert!(pattern.fires_at(4, 0.1, &mut rng));
        assert!(!pattern.fires_at(5, 0.1, &mut rng));
        assert!(!pattern.fires_at(100, 0.1, &mut rng));
    }

    #[test]
    fn test_input_pattern_periodic() {
        let mut rng = rand::thread_rng();
        let pattern = InputPattern::Periodic {
            period: 3,
            phase: 0,
        };
        assert!(pattern.fires_at(0, 0.1, &mut rng));
        assert!(!pattern.fires_at(1, 0.1, &mut rng));
        assert!(!pattern.fires_at(2, 0.1, &mut rng));
        assert!(pattern.fires_at(3, 0.1, &mut rng));
        assert!(pattern.fires_at(6, 0.1, &mut rng));
    }

    #[test]
    fn test_input_pattern_burst() {
        let mut rng = rand::thread_rng();
        let pattern = InputPattern::Burst {
            burst_length: 2,
            silence_length: 3,
        };
        // Cycle: 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, ...
        assert!(pattern.fires_at(0, 0.1, &mut rng));
        assert!(pattern.fires_at(1, 0.1, &mut rng));
        assert!(!pattern.fires_at(2, 0.1, &mut rng));
        assert!(!pattern.fires_at(3, 0.1, &mut rng));
        assert!(!pattern.fires_at(4, 0.1, &mut rng));
        assert!(pattern.fires_at(5, 0.1, &mut rng));
    }

    #[test]
    fn test_generator_combine_or() {
        let fires = vec![false, true, false];
        assert!(GeneratorCombineMode::Or.combine(&fires));
        assert!(!GeneratorCombineMode::Or.combine(&[false, false]));
    }

    #[test]
    fn test_generator_combine_and() {
        assert!(GeneratorCombineMode::And.combine(&[true, true]));
        assert!(!GeneratorCombineMode::And.combine(&[true, false]));
    }

    #[test]
    fn test_generator_combine_xor() {
        assert!(GeneratorCombineMode::Xor.combine(&[true, false, false]));
        assert!(!GeneratorCombineMode::Xor.combine(&[true, true]));
        assert!(GeneratorCombineMode::Xor.combine(&[true, true, true]));
    }

    #[test]
    fn test_variable_thresholds() {
        // Default (4 levels - optimized for speed)
        let model4 = ModelConfig::default();
        let th4 = generate_thresholds(&model4);
        assert_eq!(th4.len(), 4);
        assert_eq!(th4[0], 25); // 25% of p_rth
        assert_eq!(th4[3], 100); // 100% of p_rth

        // 10 levels (full model)
        let model10 = ModelConfig::full();
        let th10 = generate_thresholds(&model10);
        assert_eq!(th10.len(), 10);
        assert_eq!(th10[0], 10); // 10% of p_rth
        assert_eq!(th10[9], 100); // 100% of p_rth

        // 5 levels
        let model5 = ModelConfig {
            threshold_levels: 5,
            ..Default::default()
        };
        let th5 = generate_thresholds(&model5);
        assert_eq!(th5.len(), 5);
        assert_eq!(th5[0], 20); // 20% of p_rth
        assert_eq!(th5[4], 100);

        // 1 level (deterministic)
        let model1 = ModelConfig {
            threshold_levels: 1,
            ..Default::default()
        };
        let th1 = generate_thresholds(&model1);
        assert_eq!(th1.len(), 1);
        assert_eq!(th1[0], 100);
    }

    #[test]
    fn test_fire_probability_levels() {
        let thresholds = vec![20, 40, 60, 80, 100];

        // Below first threshold
        assert!((determine_fire_probability(10, &thresholds) - 0.0).abs() < 0.01);

        // Between first and second
        assert!((determine_fire_probability(25, &thresholds) - 0.2).abs() < 0.01);

        // Above all thresholds
        assert!((determine_fire_probability(150, &thresholds) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_simple_simulation() {
        let mut graph = SnnGraph::default();
        let input = graph.add_node("Input", NodeKind::Neuron, [0.0, 0.0]);
        let output = graph.add_node("Output", NodeKind::Neuron, [100.0, 0.0]);
        graph.add_edge(input, output, 100);

        let config = SimulationConfig {
            duration_ms: 10.0,
            seed: Some(12345),
            record_spikes: true,
            ..Default::default()
        };

        let result = run_simulation(&graph, &config);

        assert!(result.total_steps > 0);
        assert!(
            !result.history.spikes.is_empty()
                || result.history.spike_counts.values().all(|&c| c == 0)
        );
    }

    #[test]
    fn test_model_config_deterministic() {
        let deterministic = ModelConfig::deterministic();
        assert_eq!(deterministic.threshold_levels, 1);
        assert!(!deterministic.enable_arp);
        assert!(!deterministic.enable_rrp);
    }
}
