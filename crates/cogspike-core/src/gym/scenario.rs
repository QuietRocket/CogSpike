//! Playground scenarios: a small library of contrasting symbol sources that make the
//! compression / learning / prediction story intuitive at a glance.
//!
//! Each scenario is a general Markov source `(pi, P)`. The local delta rule discovers
//! `P` from the stream, and the money plot descends from the uniform start `log2 n` to
//! the source's *entropy rate* -- so a more predictable source visibly compresses more.
//! The reference lines (entropy-rate floor, marginal `H(pi)`, Bayes accuracy ceiling)
//! are computed analytically from `(pi, P)`, so every scenario is self-labelling.

use crate::coder::{RoverSource, entropy_bits, entropy_rate, momentum_chain};

/// Alphabet size shared by every playground scenario (the 4 rover moves U/D/L/R).
pub const N: usize = 4;

/// Stickiness of the momentum rover (probability of repeating the previous move).
const ROVER_S: f64 = 0.7;
/// Per-row off-target leak of the near-periodic cycle (keeps it stochastic + learnable).
const PERIODIC_EPS: f64 = 0.02;

/// A contrasting source for the playground, ordered incompressible -> most compressible.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Scenario {
    /// Uniform IID -- no structure; incompressible (bits stay at `log2 n`).
    UniformIid,
    /// Biased IID -- skewed marginal, no memory; compresses to `H(pi)`.
    BiasedIid,
    /// Near-periodic cycle -- almost deterministic; highly compressible (bits -> ~0).
    Periodic,
    /// Momentum rover -- sticky Markov memory; partial compression to the entropy rate.
    MarkovRover,
}

impl Scenario {
    /// All scenarios, in pedagogical order.
    pub const ALL: [Self; 4] = [
        Self::UniformIid,
        Self::BiasedIid,
        Self::Periodic,
        Self::MarkovRover,
    ];

    /// Short selector label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::UniformIid => "Uniform (random)",
            Self::BiasedIid => "Biased (skewed)",
            Self::Periodic => "Periodic (cycle)",
            Self::MarkovRover => "Rover (momentum)",
        }
    }

    /// One-line plain-language description of what the learner faces.
    #[must_use]
    pub fn tagline(self) -> &'static str {
        match self {
            Self::UniformIid => {
                "Pure noise: every symbol equally likely, no pattern. Nothing to learn -- the code stays at log2 4 = 2 bits."
            }
            Self::BiasedIid => {
                "Some symbols are common, but the order is random. The learner finds the bias and compresses to the marginal entropy H(pi)."
            }
            Self::Periodic => {
                "An almost-deterministic cycle U->D->L->R. Highly predictable, so it compresses nearly to zero bits."
            }
            Self::MarkovRover => {
                "A momentum walk: the rover tends to repeat its last move. Partial memory means partial compression (the entropy rate)."
            }
        }
    }

    /// The analytic stationary distribution `pi`.
    #[must_use]
    pub fn stationary(self) -> Vec<f64> {
        match self {
            // A cyclically-symmetric chain has a uniform stationary distribution.
            Self::UniformIid | Self::Periodic => vec![1.0 / N as f64; N],
            Self::BiasedIid | Self::MarkovRover => vec![0.5, 0.25, 0.125, 0.125],
        }
    }

    /// The transition matrix `P` (row `i` = conditional law given previous symbol `i`).
    #[must_use]
    pub fn transition(self) -> Vec<Vec<f64>> {
        match self {
            // s = 0 makes the rover chain memoryless: every row equals `pi` (IID).
            Self::UniformIid | Self::BiasedIid => momentum_chain(&self.stationary(), 0.0),
            Self::MarkovRover => momentum_chain(&self.stationary(), ROVER_S),
            Self::Periodic => (0..N)
                .map(|i| {
                    let next = (i + 1) % N;
                    let leak = PERIODIC_EPS;
                    let hit = 1.0 - leak * (N - 1) as f64;
                    (0..N).map(|j| if j == next { hit } else { leak }).collect()
                })
                .collect(),
        }
    }

    /// Build the source for this scenario with the given stream seed.
    #[must_use]
    pub fn source(self, seed: u64) -> RoverSource {
        RoverSource {
            pi: self.stationary(),
            s: if self.has_stickiness() { ROVER_S } else { 0.0 },
            p: self.transition(),
            seed,
        }
    }

    /// Whether the live stickiness slider applies (only the momentum rover).
    #[must_use]
    pub fn has_stickiness(self) -> bool {
        matches!(self, Self::MarkovRover)
    }

    /// Best achievable bits/symbol: the source entropy rate (the money-plot floor).
    #[must_use]
    pub fn entropy_rate_floor(self) -> f64 {
        entropy_rate(&self.transition(), &self.stationary())
    }

    /// Marginal entropy `H(pi)`: the bits a memoryless coder pays (the no-memory baseline).
    #[must_use]
    pub fn marginal_entropy(self) -> f64 {
        entropy_bits(&self.stationary())
    }

    /// Best achievable next-symbol accuracy: the Bayes ceiling `sum_i pi_i max_j P_ij`.
    #[must_use]
    pub fn bayes_ceiling(self) -> f64 {
        self.transition()
            .iter()
            .zip(self.stationary())
            .map(|(row, pi_i)| pi_i * row.iter().copied().fold(0.0_f64, f64::max))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floors_are_ordered_by_compressibility() {
        // Uniform is incompressible (= log2 4); each later scenario compresses more.
        let uniform = Scenario::UniformIid.entropy_rate_floor();
        let biased = Scenario::BiasedIid.entropy_rate_floor();
        let periodic = Scenario::Periodic.entropy_rate_floor();
        let rover = Scenario::MarkovRover.entropy_rate_floor();
        assert!((uniform - 2.0).abs() < 1e-9, "uniform floor is log2 4 = 2");
        assert!((biased - 1.75).abs() < 1e-9, "biased floor is H(pi) = 1.75");
        assert!(periodic < 0.6, "near-periodic compresses hard: {periodic}");
        assert!(
            (rover - 0.9782).abs() < 1e-3,
            "rover floor is the entropy rate"
        );
        // memory only helps the rover: IID floors equal their marginals.
        assert!((biased - Scenario::BiasedIid.marginal_entropy()).abs() < 1e-12);
    }

    #[test]
    fn every_transition_row_is_a_distribution() {
        for sc in Scenario::ALL {
            for row in sc.transition() {
                let s: f64 = row.iter().sum();
                assert!((s - 1.0).abs() < 1e-12, "{sc:?} row sums to 1");
                assert!(row.iter().all(|&p| p >= 0.0), "{sc:?} row is non-negative");
            }
        }
    }

    #[test]
    fn bayes_ceiling_brackets() {
        // uniform: best you can do is 1/4; periodic: nearly perfect.
        assert!((Scenario::UniformIid.bayes_ceiling() - 0.25).abs() < 1e-9);
        assert!(Scenario::Periodic.bayes_ceiling() > 0.9);
        assert!((Scenario::MarkovRover.bayes_ceiling() - 0.8031).abs() < 1e-3);
    }
}
