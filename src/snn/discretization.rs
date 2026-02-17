//! Weight discretization functions from the research paper (§3-§5).
//!
//! Implements the core mathematical functions for converting continuous-domain
//! SNN parameters to the discretized domain used by the paper's formal model.
//!
//! ## Key Formulas
//! - **Weight discretization**: `δ_W(w) = ⌊w · W / w_max⌉` (§3)
//! - **Threshold calibration**: `T_d = ⌈T · W / w_max⌉` (§3.2)
//! - **Leak factor**: `λ_d = -max(1, ⌊ℓ · T_d⌋)` (§4.2)

/// Maximum weight value in CogSpike's internal scale.
const W_MAX: f64 = 100.0;

/// Discretize a weight to the range `[-W, W]`.
///
/// Paper §3: `δ_W(w) = ⌊w · W / w_max⌉`
///
/// # Examples
/// - `discretize_weight(100, 3) == 3`   (full excitatory → +W)
/// - `discretize_weight(-67, 3) == -2`  (partial inhibitory)
/// - `discretize_weight(0, 3) == 0`     (zero stays zero)
pub fn discretize_weight(w: i16, weight_levels: u8) -> i32 {
    (w as f64 * weight_levels as f64 / W_MAX).round() as i32
}

/// Compute the discretized threshold.
///
/// Paper §3.2: `T_d = ⌈T · W / w_max⌉`
///
/// Uses ceiling to ensure the discretized neuron is at least as hard to fire
/// as the original (Threshold Preservation Theorem).
pub fn discretized_threshold(threshold: u8, weight_levels: u8) -> i32 {
    let t = threshold as i32;
    let w = weight_levels as i32;
    let w_max = W_MAX as i32;
    // Ceiling division: ceil(t * w / w_max) = (t * w + w_max - 1) / w_max
    (t * w + w_max - 1) / w_max
}

/// Compute the discretized additive leak factor.
///
/// Paper §4.2: `λ_d = -max(1, ⌊ℓ · T_d⌋)`
///
/// where `ℓ = 1 - r` is the leak factor (fraction of potential lost per step),
/// and `r` is the retention rate from `ModelConfig::leak_r / 100`.
///
/// The `max(1, ...)` ensures leak is always at least -1 to maintain the
/// Asymptotic Silence property (Soundness Theorem).
///
/// Returns a negative integer.
pub fn discretized_leak(leak_factor: f64, t_d: i32) -> i32 {
    -(1i32.max((leak_factor * t_d as f64).floor() as i32))
}

/// Result of a feasibility analysis for a neuron.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Feasibility {
    /// The neuron can reach threshold in a single step.
    SingleStep,
    /// The neuron can reach threshold but requires accumulation over multiple steps.
    MultiStep {
        /// Minimum number of steps to reach threshold under maximum sustained input.
        min_steps: i32,
    },
    /// The neuron can never reach threshold — leak overwhelms excitatory input.
    Impossible,
}

/// Check whether a neuron can reach its firing threshold given its discretized
/// excitatory weights and leak factor.
///
/// Paper §5: A neuron is feasible if `Σ w_i^d > |λ_d|` for its excitatory inputs.
pub fn check_feasibility(excitatory_weights: &[i32], t_d: i32, lambda_d: i32) -> Feasibility {
    let max_excitation: i32 = excitatory_weights.iter().sum();
    let leak_magnitude = lambda_d.unsigned_abs() as i32;

    if max_excitation >= t_d {
        return Feasibility::SingleStep;
    }

    let net_gain = max_excitation - leak_magnitude;
    if net_gain <= 0 {
        return Feasibility::Impossible;
    }

    // Steps needed: ceil(t_d / net_gain)
    let steps = (t_d + net_gain - 1) / net_gain;
    Feasibility::MultiStep { min_steps: steps }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Weight discretization ---

    #[test]
    fn test_discretize_weight_full_excitatory() {
        // δ_3(100) = round(100 * 3 / 100) = 3
        assert_eq!(discretize_weight(100, 3), 3);
    }

    #[test]
    fn test_discretize_weight_partial_inhibitory() {
        // δ_3(-67) = round(-67 * 3 / 100) = round(-2.01) = -2
        assert_eq!(discretize_weight(-67, 3), -2);
    }

    #[test]
    fn test_discretize_weight_half_w5() {
        // δ_5(50) = round(50 * 5 / 100) = round(2.5) = 3 (rounds to even → 2? No, Rust rounds 2.5 → 3)
        // Actually f64::round rounds 2.5 → 3.0
        assert_eq!(discretize_weight(50, 5), 3);
    }

    #[test]
    fn test_discretize_weight_zero() {
        assert_eq!(discretize_weight(0, 3), 0);
        assert_eq!(discretize_weight(0, 10), 0);
    }

    #[test]
    fn test_discretize_weight_negative_full() {
        // δ_3(-100) = round(-100 * 3 / 100) = -3
        assert_eq!(discretize_weight(-100, 3), -3);
    }

    #[test]
    fn test_discretize_weight_small() {
        // δ_3(10) = round(10 * 3 / 100) = round(0.3) = 0
        assert_eq!(discretize_weight(10, 3), 0);
        // δ_3(20) = round(20 * 3 / 100) = round(0.6) = 1
        assert_eq!(discretize_weight(20, 3), 1);
    }

    // --- Threshold calibration ---

    #[test]
    fn test_discretized_threshold_basic() {
        // T_d(100, 3) = ceil(100 * 3 / 100) = ceil(3) = 3
        assert_eq!(discretized_threshold(100, 3), 3);
    }

    #[test]
    fn test_discretized_threshold_ceiling() {
        // T_d(80, 5) = ceil(80 * 5 / 100) = ceil(4.0) = 4
        assert_eq!(discretized_threshold(80, 5), 4);
        // T_d(70, 3) = ceil(70 * 3 / 100) = ceil(2.1) = 3
        assert_eq!(discretized_threshold(70, 3), 3);
    }

    #[test]
    fn test_discretized_threshold_w1() {
        // T_d(100, 1) = ceil(100 * 1 / 100) = 1
        assert_eq!(discretized_threshold(100, 1), 1);
    }

    // --- Leak factor ---

    #[test]
    fn test_discretized_leak_basic() {
        // λ_d(ℓ=0.1, T_d=3) = -max(1, floor(0.1 * 3)) = -max(1, 0) = -1
        assert_eq!(discretized_leak(0.1, 3), -1);
    }

    #[test]
    fn test_discretized_leak_high() {
        // λ_d(ℓ=0.5, T_d=6) = -max(1, floor(0.5 * 6)) = -max(1, 3) = -3
        assert_eq!(discretized_leak(0.5, 6), -3);
    }

    #[test]
    fn test_discretized_leak_minimum_one() {
        // λ_d(ℓ=0.01, T_d=3) = -max(1, floor(0.01 * 3)) = -max(1, 0) = -1
        assert_eq!(discretized_leak(0.01, 3), -1);
    }

    #[test]
    fn test_discretized_leak_exact_threshold() {
        // λ_d(ℓ=0.05, T_d=20) = -max(1, floor(0.05 * 20)) = -max(1, 1) = -1
        assert_eq!(discretized_leak(0.05, 20), -1);
    }

    // --- Feasibility ---

    #[test]
    fn test_feasibility_single_step() {
        // One excitatory weight of 3, threshold 3 → can fire in one step
        assert_eq!(check_feasibility(&[3], 3, -1), Feasibility::SingleStep);
    }

    #[test]
    fn test_feasibility_multi_step() {
        // One excitatory weight of 2, threshold 5, leak -1 → net gain = 1 per step → 5 steps
        assert_eq!(
            check_feasibility(&[2], 5, -1),
            Feasibility::MultiStep { min_steps: 5 }
        );
    }

    #[test]
    fn test_feasibility_impossible() {
        // One excitatory weight of 1, leak -1 → net gain = 0, can never accumulate
        assert_eq!(check_feasibility(&[1], 3, -1), Feasibility::Impossible);
    }

    #[test]
    fn test_feasibility_impossible_leak_dominates() {
        // One excitatory weight of 1, leak -2 → net gain negative
        assert_eq!(check_feasibility(&[1], 3, -2), Feasibility::Impossible);
    }

    #[test]
    fn test_feasibility_multiple_inputs() {
        // Two excitatory weights [2, 1], threshold 5, leak -1 → sum=3, net_gain=2, steps=3
        assert_eq!(
            check_feasibility(&[2, 1], 5, -1),
            Feasibility::MultiStep { min_steps: 3 }
        );
    }
}
