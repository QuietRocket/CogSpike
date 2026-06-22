//! Information measures in bits, lifted byte-faithful from
//! `spikecoder/information.py` and `research/compression/validate.py`.
//!
//! Distributions are `&[f64]` rows; conditional laws are `&[Vec<f64>]` matrices.
//! Every function here is closed-form and is asserted against the golden fixtures
//! (`tests/coder_parity.rs`) to `<= 1e-12`.

/// `p * log2 p` with the `0 * log0 = 0` convention.
fn xlog2x(p: f64) -> f64 {
    if p > 0.0 { p * p.log2() } else { 0.0 }
}

/// Shannon entropy `H(p) = -sum p log2 p` in bits.
#[must_use]
pub fn entropy_bits(p: &[f64]) -> f64 {
    -p.iter().map(|&pi| xlog2x(pi)).sum::<f64>()
}

/// Cross-entropy `H(p, q) = -sum p log2 q` in bits (`q > 0` wherever `p > 0`).
#[must_use]
pub fn cross_entropy_bits(p: &[f64], q: &[f64]) -> f64 {
    -p.iter()
        .zip(q)
        .filter(|&(&pi, _)| pi > 0.0)
        .map(|(&pi, &qi)| pi * qi.log2())
        .sum::<f64>()
}

/// Kullback-Leibler divergence `D_KL(p || q)` in bits.
#[must_use]
pub fn kl_bits(p: &[f64], q: &[f64]) -> f64 {
    cross_entropy_bits(p, q) - entropy_bits(p)
}

/// Numerically stable softmax (the exact simplex normalizer).
#[must_use]
pub fn softmax(z: &[f64]) -> Vec<f64> {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = z.iter().map(|&zi| (zi - m).exp()).collect();
    let total: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / total).collect()
}

/// Conditional entropy rate `H(P) = sum_i pi_i H(P[i,:])` in bits/symbol.
#[must_use]
pub fn entropy_rate(p: &[Vec<f64>], pi: &[f64]) -> f64 {
    pi.iter()
        .zip(p)
        .map(|(&pii, row)| pii * entropy_bits(row))
        .sum()
}

/// Cross-entropy rate `H(p, q) = sum_i pi_i sum_j P_ij (-log2 Q_ij)` in bits/symbol.
#[must_use]
pub fn cross_entropy_rate(p: &[Vec<f64>], pi: &[f64], q: &[Vec<f64>]) -> f64 {
    pi.iter()
        .zip(p)
        .zip(q)
        .map(|((&pii, prow), qrow)| pii * cross_entropy_bits(prow, qrow))
        .sum()
}

/// Expected per-symbol energy (cross-entropy rate) and the excess (average KL).
///
/// Takes predicted conditional laws `q` (already probabilities) under the true
/// `p`. For logit rows `W`, pass `&learned_matrix(&W)`.
#[must_use]
pub fn model_energy_and_kl(q: &[Vec<f64>], pi: &[f64], p: &[Vec<f64>], h_rate: f64) -> (f64, f64) {
    let energy = cross_entropy_rate(p, pi, q);
    (energy, energy - h_rate)
}

/// Reconstruct the conditional law `q[i,:] = softmax(W[i,:])` from logit rows.
#[must_use]
pub fn learned_matrix(w: &[Vec<f64>]) -> Vec<Vec<f64>> {
    w.iter().map(|row| softmax(row)).collect()
}
