//! The momentum-rover Markov source, byte-faithful to `spikecoder/source.py`.
//!
//! `P_ij = s*1[i=j] + (1-s)*pi_j` -- a convex blend of the identity (pure
//! momentum) and `1 pi^T` (pure memorylessness). See `research/compression/validate.py`.

use rand::SeedableRng as _;
use rand::distributions::{Distribution as _, WeightedIndex};
use rand::rngs::StdRng;

use super::constants::{PI, S, SEED};

/// Transition matrix `P_ij = s*delta_ij + (1-s)*pi_j` (the stay-or-resample chain).
#[must_use]
pub fn momentum_chain(pi: &[f64], s: f64) -> Vec<Vec<f64>> {
    let n = pi.len();
    (0..n)
        .map(|i| {
            pi.iter()
                .enumerate()
                .map(|(j, &pj)| (if i == j { s } else { 0.0 }) + (1.0 - s) * pj)
                .collect()
        })
        .collect()
}

/// Sample an `n`-symbol stream from chain `p` started at `pi`.
///
/// NOTE: this uses Rust's `StdRng` and does NOT reproduce numpy's PCG64 stream
/// byte-for-byte; only the statistics converge (empirical marginal -> `pi`). Exact
/// parity is asserted only on the closed-form quantities (the golden fixtures).
#[must_use]
pub fn sample_stream(p: &[Vec<f64>], pi: &[f64], n: usize, seed: u64) -> Vec<usize> {
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let pi_dist = WeightedIndex::new(pi).expect("pi must be a valid probability distribution");
    let row_dists: Vec<WeightedIndex<f64>> = p
        .iter()
        .map(|row| WeightedIndex::new(row).expect("each P row must be a valid distribution"))
        .collect();
    let mut prev = pi_dist.sample(&mut rng);
    out.push(prev);
    for _ in 1..n {
        let next = row_dists
            .get(prev)
            .expect("row index is a previously sampled symbol, in range")
            .sample(&mut rng);
        out.push(next);
        prev = next;
    }
    out
}

/// One-hot encoding of `symbol` in an `n`-dimensional vector (all-zeros if out of range).
#[must_use]
pub fn one_hot(symbol: usize, n: usize) -> Vec<f64> {
    let mut v = vec![0.0; n];
    if let Some(slot) = v.get_mut(symbol) {
        *slot = 1.0;
    }
    v
}

/// The momentum-rover source: stationary `pi`, stickiness `s`, transition `p`, seed.
#[derive(Clone, Debug)]
pub struct RoverSource {
    /// Stationary move frequencies.
    pub pi: Vec<f64>,
    /// Stickiness in `[0, 1)`.
    pub s: f64,
    /// Transition matrix `momentum_chain(pi, s)`.
    pub p: Vec<Vec<f64>>,
    /// Stream sampling seed.
    pub seed: u64,
}

impl RoverSource {
    /// Build a source with the given stickiness, stationary distribution, and seed.
    #[must_use]
    pub fn new(s: f64, pi: Vec<f64>, seed: u64) -> Self {
        let p = momentum_chain(&pi, s);
        Self { pi, s, p, seed }
    }

    /// The paper's worked rover: `pi = (1/2,1/4,1/8,1/8)`, `s = 0.7`, `seed = 7`.
    #[must_use]
    pub fn paper() -> Self {
        Self::new(S, PI.to_vec(), SEED)
    }

    /// (Re)build the transition matrix for a new stickiness `s` (non-stationary runs).
    pub fn set_s(&mut self, s: f64) {
        self.s = s;
        self.p = momentum_chain(&self.pi, s);
    }

    /// Alphabet size.
    #[must_use]
    pub fn n(&self) -> usize {
        self.pi.len()
    }

    /// Sample an `n`-symbol stream (see [`sample_stream`] for the RNG caveat).
    #[must_use]
    pub fn sample(&self, n: usize) -> Vec<usize> {
        sample_stream(&self.p, &self.pi, n, self.seed)
    }
}
