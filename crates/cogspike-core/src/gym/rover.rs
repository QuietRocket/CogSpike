//! The rover environment: emits the momentum-rover stream and scores the agent's
//! per-symbol prediction by surprisal.

use rand::SeedableRng as _;
use rand::rngs::StdRng;

use crate::coder::{LAMBDA, Q_CLIP_HI, Q_CLIP_LO, RoverSource, one_hot};

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
        // The encoder cannot represent q outside [Q_CLIP_LO, Q_CLIP_HI], so per-symbol
        // surprisal is bounded by -log2(Q_CLIP_LO) = MAX_SURPRISAL_BITS (clip-enforced).
        let q_emit = prediction
            .q
            .get(emitted)
            .copied()
            .unwrap_or(0.0)
            .clamp(Q_CLIP_LO, Q_CLIP_HI);
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

/// An unbounded, online variant of the rover for interactive (frame-stepped) use.
///
/// It samples one symbol at a time from the live transition matrix, never `done`,
/// and exposes the current context so the UI can step it indefinitely.
#[derive(Clone, Debug)]
pub struct OnlineRover {
    /// The underlying Markov source (its `P` can be retuned live via [`Self::set_s`]).
    pub source: RoverSource,
    /// Time-per-bit constant for the latency/reward.
    pub lambda: f64,
    rng: StdRng,
    prev: usize,
}

impl OnlineRover {
    /// Construct from a source, `lambda`, and seed (samples the initial symbol).
    #[must_use]
    pub fn new(source: RoverSource, lambda: f64, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let prev = source.sample_initial(&mut rng);
        Self {
            source,
            lambda,
            rng,
            prev,
        }
    }

    /// The paper's rover, seeded.
    #[must_use]
    pub fn paper(seed: u64) -> Self {
        Self::new(RoverSource::paper(), LAMBDA, seed)
    }

    /// Alphabet size.
    #[must_use]
    pub fn n(&self) -> usize {
        self.source.n()
    }

    /// The current context observation (one-hot of the previous symbol).
    #[must_use]
    pub fn context(&self) -> Vec<f64> {
        one_hot(self.prev, self.n())
    }

    /// Retune the source stickiness `s` live (rebuilds the transition matrix).
    pub fn set_s(&mut self, s: f64) {
        self.source.set_s(s);
    }

    /// Emit the next symbol and score the agent's `prediction` by surprisal.
    pub fn step(&mut self, prediction: &Prediction) -> StepOut {
        let emitted = self.source.sample_next(self.prev, &mut self.rng);
        // Clip-enforced bound: q is clamped to [Q_CLIP_LO, Q_CLIP_HI], so per-symbol
        // surprisal never exceeds -log2(Q_CLIP_LO) = MAX_SURPRISAL_BITS.
        let q_emit = prediction
            .q
            .get(emitted)
            .copied()
            .unwrap_or(0.0)
            .clamp(Q_CLIP_LO, Q_CLIP_HI);
        let bits = -q_emit.log2();
        let reward = -self.lambda * bits;
        let correct = prediction.decoded == Some(emitted);
        self.prev = emitted;
        StepOut {
            obs: one_hot(emitted, self.n()),
            reward,
            bits,
            correct,
            emitted,
            done: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coder::MAX_SURPRISAL_BITS;

    #[test]
    fn surprisal_is_clip_bounded() {
        // A maximally-wrong prediction: q -> 0 for whichever symbol the source emits.
        // The q-clip must cap the surprisal at -log2(Q_CLIP_LO) = MAX_SURPRISAL_BITS.
        let mut env = OnlineRover::paper(7);
        let pred = Prediction {
            q: vec![1e-12; env.n()],
            decoded: None,
        };
        for _ in 0..32 {
            let out = env.step(&pred);
            assert!(
                out.bits <= *MAX_SURPRISAL_BITS + 1e-9,
                "bits {} exceeded clip-enforced cap {}",
                out.bits,
                *MAX_SURPRISAL_BITS
            );
        }
        // the cap is the expected ~13.29 bits (= -log2(1e-4))
        assert!((*MAX_SURPRISAL_BITS - 13.287_712_379_549_45).abs() < 1e-9);
    }

    #[test]
    fn typical_surprisal_is_unaffected_by_the_clip() {
        // A well-inside-range prediction is not perturbed by the clamp.
        let mut env = OnlineRover::paper(7);
        let pred = Prediction {
            q: vec![0.25; env.n()],
            decoded: None,
        };
        let out = env.step(&pred);
        assert!(
            (out.bits - 2.0).abs() < 1e-12,
            "uniform q -> log2(4) = 2 bits"
        );
    }
}
