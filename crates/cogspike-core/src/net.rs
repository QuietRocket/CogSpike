//! The predictor network: logits `a = W c`, `q = softmax(a)`, learned online by
//! the local three-factor delta rule `dW = eta * c * (y - q)`.
//!
//! With a one-hot context `c` (the previous symbol), only the active row `W[i]`
//! participates, exactly reproducing `learn_validate.py`'s per-context update.

use crate::coder::{learned_matrix, softmax};

/// A linear-softmax predictor: `w[i]` is the logit row used when the context is
/// symbol `i`.
#[derive(Clone, Debug)]
pub struct CoderNet {
    /// Logit rows; `w[i][j]` is the logit for predicting symbol `j` from context `i`.
    pub w: Vec<Vec<f64>>,
    /// Alphabet size.
    pub n: usize,
}

impl CoderNet {
    /// A zero-initialized predictor for an `n`-symbol alphabet (uniform `q = 1/n`).
    #[must_use]
    pub fn zeros(n: usize) -> Self {
        Self {
            w: vec![vec![0.0; n]; n],
            n,
        }
    }

    /// Logits `a_j = sum_i w[i][j] * c_i` for context activity `c`.
    #[must_use]
    pub fn logits(&self, c: &[f64]) -> Vec<f64> {
        let mut a = vec![0.0; self.n];
        for (&ci, row) in c.iter().zip(&self.w) {
            for (aj, &wij) in a.iter_mut().zip(row) {
                *aj += ci * wij;
            }
        }
        a
    }

    /// Predicted distribution `q = softmax(W c)`.
    #[must_use]
    pub fn predict(&self, c: &[f64]) -> Vec<f64> {
        softmax(&self.logits(c))
    }

    /// The local delta / three-factor update `dW[i][j] += eta * c_i * (y_j - q_j)`.
    pub fn apply_delta(&mut self, c: &[f64], y: &[f64], q: &[f64], eta: f64) {
        for (&ci, row) in c.iter().zip(self.w.iter_mut()) {
            for ((wij, &yj), &qj) in row.iter_mut().zip(y).zip(q) {
                *wij += eta * ci * (yj - qj);
            }
        }
    }

    /// The current conditional law `q[i,:] = softmax(W[i,:])`.
    #[must_use]
    pub fn conditional_law(&self) -> Vec<Vec<f64>> {
        learned_matrix(&self.w)
    }
}
