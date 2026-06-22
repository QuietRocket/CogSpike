//! The rover environment: emits the momentum-rover stream and scores the agent's
//! per-symbol prediction by surprisal.

use crate::coder::{LAMBDA, RoverSource, one_hot};

use super::{Env, Prediction, StepOut};

/// The momentum-rover environment over a pre-sampled symbol stream.
#[derive(Clone, Debug)]
pub struct RoverEnv {
    /// The underlying Markov source (holds `pi`, `s`, `P`).
    pub source: RoverSource,
    /// Time-per-bit constant for the latency/reward.
    pub lambda: f64,
    /// Number of scored symbols per episode.
    pub horizon: usize,
    stream: Vec<usize>,
    t: usize,
}

impl RoverEnv {
    /// Construct from a source, `lambda`, and horizon.
    #[must_use]
    pub fn new(source: RoverSource, lambda: f64, horizon: usize) -> Self {
        Self {
            source,
            lambda,
            horizon,
            stream: Vec::new(),
            t: 0,
        }
    }

    /// The paper's rover (`s = 0.7`, `pi = (1/2,1/4,1/8,1/8)`) with `lambda = LAMBDA`.
    #[must_use]
    pub fn paper(horizon: usize) -> Self {
        Self::new(RoverSource::paper(), LAMBDA, horizon)
    }
}

impl Env for RoverEnv {
    fn reset(&mut self, seed: u64) -> Vec<f64> {
        self.source.seed = seed;
        self.stream = self.source.sample(self.horizon + 1);
        self.t = 1;
        let first = self.stream.first().copied().unwrap_or(0);
        one_hot(first, self.n())
    }

    fn step(&mut self, prediction: &Prediction) -> StepOut {
        let emitted = self
            .stream
            .get(self.t)
            .copied()
            .expect("step is called within the horizon");
        let q_emit = prediction.q.get(emitted).copied().unwrap_or(0.0);
        let bits = -q_emit.log2();
        let reward = -self.lambda * bits;
        let correct = prediction.decoded == Some(emitted);
        self.t += 1;
        let done = self.t > self.horizon;
        let obs = one_hot(emitted, self.n());
        StepOut {
            obs,
            reward,
            bits,
            correct,
            emitted,
            done,
        }
    }

    fn n(&self) -> usize {
        self.source.n()
    }
}
