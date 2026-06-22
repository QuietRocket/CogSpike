//! The fidelity ladder: one [`Substrate`] trait, three rungs.
//!
//! Only [`Idealized`] (the closed-form, byte-faithful rung) is implemented now.
//! `SpikingPopulation` (M5, wrapping the finite LIF simulation) and `EventStream`
//! (M6, the integer DVS residual map) are reserved variants so the same scenario
//! can later run at any fidelity without redesign.

use crate::coder::{LAMBDA, analytic_latency_ideal, decode_first_spike};

/// Which physical fidelity a substrate realizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fidelity {
    /// Closed-form: `t*(q) = -lambda log2 q` exactly.
    Idealized,
    /// Finite spiking population (Nengo-audited). Reserved for M5.
    SpikingPopulation,
    /// Neuromorphic event stream (integer DVS residuals). Reserved for M6.
    EventStream,
}

/// The provenance/strength of a substrate's numbers, surfaced in the UI as the
/// `=` / `≈` / `⊢` badges.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Parity {
    /// Closed-form, byte-faithful to the numpy oracle within `tol`.
    Exact {
        /// Absolute tolerance against the golden values.
        tol: f64,
    },
    /// Sampled/decomposed; within relative band `rel` of the oracle.
    Band {
        /// Relative band width.
        rel: f64,
    },
    /// Formally proven (PRISM/PCTL).
    Proven,
}

/// A physical realization of the latency code: map predicted probabilities `q` to
/// per-symbol first-spike latencies, and decode the winner.
pub trait Substrate {
    /// Map predicted distribution `q` to per-symbol first-spike latencies (seconds).
    fn encode_latencies(&self, q: &[f64]) -> Vec<f64>;
    /// First-spike-takes-all decode: argmin latency is the decoded symbol
    /// (`None` if no neuron spikes).
    fn decode(&self, latencies: &[f64]) -> Option<usize>;
    /// Which fidelity rung this substrate is.
    fn fidelity(&self) -> Fidelity;
    /// The parity strength of this substrate's numbers.
    fn parity(&self) -> Parity;
}

/// The idealized (closed-form) rung: latency `t*(q) = -lambda log2 q`, exact.
#[derive(Clone, Copy, Debug)]
pub struct Idealized {
    /// Time-per-bit constant `lambda` (s/bit).
    pub lambda: f64,
}

impl Idealized {
    /// Construct with an explicit `lambda`.
    #[must_use]
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

impl Default for Idealized {
    fn default() -> Self {
        Self { lambda: LAMBDA }
    }
}

impl Substrate for Idealized {
    fn encode_latencies(&self, q: &[f64]) -> Vec<f64> {
        q.iter()
            .map(|&qi| analytic_latency_ideal(qi, self.lambda))
            .collect()
    }

    fn decode(&self, latencies: &[f64]) -> Option<usize> {
        decode_first_spike(latencies).0
    }

    fn fidelity(&self) -> Fidelity {
        Fidelity::Idealized
    }

    fn parity(&self) -> Parity {
        Parity::Exact { tol: 1e-12 }
    }
}
