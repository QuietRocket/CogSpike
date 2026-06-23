//! An OpenAI-gym-shaped surface over the latency entropy coder.
//!
//! `obs` is the context one-hot `c` (the previous symbol). The agent predicts a
//! distribution `q`, decodes a symbol, and pays a per-symbol cost equal to the
//! surprisal `-log2 q[emitted]` (bits); the first-spike latency is `lambda *` that.
//! The "money plot" is the running mean of `bits`, which descends to the entropy
//! rate floor `H_RATE` as the local delta rule learns `q -> P`.

pub mod agent;
pub mod rover;
pub mod scenario;

pub use agent::DeltaAgent;
pub use rover::{OnlineRover, RoverEnv};
pub use scenario::Scenario;

use crate::coder::{cross_entropy_rate, one_hot};

/// An agent's output for one context: the predicted distribution and its decode.
#[derive(Clone, Debug)]
pub struct Prediction {
    /// Predicted next-symbol distribution `q = softmax(W c)`.
    pub q: Vec<f64>,
    /// First-spike-takes-all decode (argmin latency = argmax `q`); `None` if silent.
    pub decoded: Option<usize>,
}

/// The outcome of one environment step.
#[derive(Clone, Debug)]
pub struct StepOut {
    /// Observation for the NEXT step: one-hot of the just-emitted symbol.
    pub obs: Vec<f64>,
    /// RL reward `= -lambda * bits` (negative latency); maximizing the return
    /// minimizes the total code length.
    pub reward: f64,
    /// Surprisal paid on this symbol `= -log2 q[emitted]` (bits) -- the money-plot value.
    pub bits: f64,
    /// Whether the decoded symbol matched the emitted symbol.
    pub correct: bool,
    /// The emitted (ground-truth) symbol.
    pub emitted: usize,
    /// Whether the episode has ended (horizon reached).
    pub done: bool,
}

/// A symbol-source environment.
pub trait Env {
    /// Reset to the start of an episode; returns the first observation (context `c`).
    fn reset(&mut self, seed: u64) -> Vec<f64>;
    /// Advance one symbol, scoring the agent's `prediction`.
    fn step(&mut self, prediction: &Prediction) -> StepOut;
    /// Alphabet size.
    fn n(&self) -> usize;
}

/// A predict-and-learn agent.
pub trait Agent {
    /// Predict a distribution and decode a symbol for observation `obs`.
    fn act(&mut self, obs: &[f64]) -> Prediction;
    /// Apply the local update from context `c`, outcome one-hot `y`, prediction `q`.
    fn learn(&mut self, c: &[f64], y: &[f64], q: &[f64]);
}

/// Summary of a headless episode, for asserting convergence against the validators.
#[derive(Clone, Debug)]
pub struct EpisodeResult {
    /// Number of scored symbols.
    pub steps: usize,
    /// Mean bits/symbol over the final window.
    pub mean_bits_window: f64,
    /// Decode accuracy over the final window.
    pub accuracy_window: f64,
    /// Energy `E = H(p, learned_q)` of the final learned law (bits/symbol).
    pub final_energy: f64,
    /// Initial energy with the uniform predictor `= log2(n)`.
    pub initial_energy: f64,
    /// Max absolute deviation of the learned conditional law from the true `P`.
    pub max_abs_q_minus_p: f64,
}

/// Run a full headless episode (`reset -> act -> step -> learn` loop) and summarize
/// the convergence of the rover agent against the true chain.
#[must_use]
pub fn run_episode(
    env: &mut RoverEnv,
    agent: &mut DeltaAgent,
    seed: u64,
    window: usize,
) -> EpisodeResult {
    let n = env.n();
    let mut obs = env.reset(seed);
    let mut bits_log: Vec<f64> = Vec::new();
    let mut correct_log: Vec<bool> = Vec::new();

    loop {
        let prediction = agent.act(&obs);
        let step = env.step(&prediction);
        let y = one_hot(step.emitted, n);
        agent.learn(&obs, &y, &prediction.q);
        bits_log.push(step.bits);
        correct_log.push(step.correct);
        obs = step.obs;
        if step.done {
            break;
        }
    }

    let steps = bits_log.len();
    let w = window.min(steps).max(1);
    let mean_bits_window = bits_log.iter().rev().take(w).sum::<f64>() / w as f64;
    let accuracy_window =
        correct_log.iter().rev().take(w).filter(|&&c| c).count() as f64 / w as f64;

    let law = agent.net.conditional_law();
    let final_energy = cross_entropy_rate(&env.source.p, &env.source.pi, &law);
    let initial_energy = (n as f64).log2();
    let max_abs_q_minus_p = env
        .source
        .p
        .iter()
        .zip(&law)
        .flat_map(|(prow, qrow)| prow.iter().zip(qrow).map(|(&a, &b)| (a - b).abs()))
        .fold(0.0_f64, f64::max);

    EpisodeResult {
        steps,
        mean_bits_window,
        accuracy_window,
        final_energy,
        initial_energy,
        max_abs_q_minus_p,
    }
}
