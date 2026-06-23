//! The delta-rule agent: a softmax predictor learned online by
//! `dW = eta * c * (y - q)`, with a Robbins-Monro decreasing step
//! `eta_t = eta0 / (1 + t/t0)`.

use crate::coder::LAMBDA;
use crate::net::CoderNet;
use crate::substrate::{Idealized, Substrate as _};

use super::{Agent, Prediction};

/// A predict-and-learn agent over a [`CoderNet`], decoding via an [`Idealized`]
/// substrate and learning with a decreasing-step delta rule.
#[derive(Clone, Debug)]
pub struct DeltaAgent {
    /// The predictor network.
    pub net: CoderNet,
    /// The fidelity substrate used to decode (first-spike-takes-all).
    pub substrate: Idealized,
    /// Initial learning rate.
    pub eta0: f64,
    /// Robbins-Monro time constant.
    pub t0: f64,
    /// Number of learning steps applied so far.
    pub t: u64,
}

impl DeltaAgent {
    /// A zero-initialized agent for `n` symbols with the paper's schedule
    /// (`eta0 = 0.2`, `t0 = 5e4`) and `lambda = LAMBDA`.
    #[must_use]
    pub fn paper(n: usize) -> Self {
        Self {
            net: CoderNet::zeros(n),
            substrate: Idealized::new(LAMBDA),
            eta0: 0.2,
            t0: 5.0e4,
            t: 0,
        }
    }

    /// The current Robbins-Monro learning rate `eta_t = eta0 / (1 + t/t0)`.
    #[must_use]
    pub fn eta(&self) -> f64 {
        self.eta0 / (1.0 + self.t as f64 / self.t0)
    }
}

impl Agent for DeltaAgent {
    fn act(&mut self, obs: &[f64]) -> Prediction {
        let q = self.net.predict(obs);
        let latencies = self.substrate.encode_latencies(&q);
        let decoded = self.substrate.decode(&latencies);
        Prediction { q, decoded }
    }

    fn learn(&mut self, c: &[f64], y: &[f64], q: &[f64]) {
        let eta = self.eta();
        self.net.apply_delta(c, y, q, eta);
        self.t += 1;
    }
}
