//! Weight discretization functions from the research paper (§3-§5).
//!
//! Implements the core mathematical functions for converting continuous-domain
//! SNN parameters to the discretized domain used by the paper's formal model.
//!
//! ## Key Formulas
//! - **Weight discretization**: `δ_W(w) = ⌊w · W / w_max⌉` (§3)
//! - **Threshold calibration**: `T_d = ⌈T · W / w_max⌉` (§3.2)
//! - **Multiplicative leak**: `floor(r × p)` — same as simulation engine (§4.2)
//! - **Negative potentials**: `P_MIN = -Σ|w_inhib|` preserves differential inhibition depth

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

/// Compute the minimum potential for a neuron in the discretized domain.
///
/// P_MIN = sum of all negative (inhibitory) discretized incoming weights.
/// This allows the potential to go negative, preserving differential
/// inhibition depth that drives WTA convergence.
///
/// Returns a non-positive integer (0 if no inhibitory inputs).
pub fn compute_discretized_p_min(inhibitory_weights: &[i32]) -> i32 {
    inhibitory_weights.iter().filter(|&&w| w < 0).sum()
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
/// excitatory weights and retention rate.
///
/// With multiplicative leak `r × p`, a neuron reaches steady-state potential
/// `p_ss = excitation / (1 - r)` under sustained maximum input. If `p_ss >= T_d`,
/// the neuron is feasible.
///
/// - `SingleStep`: max excitation in one tick ≥ T_d
/// - `MultiStep`: needs accumulation over multiple ticks to reach T_d
/// - `Impossible`: steady-state potential under sustained max input < T_d
pub fn check_feasibility(excitatory_weights: &[i32], t_d: i32, retention_rate: f64) -> Feasibility {
    let max_excitation: i32 = excitatory_weights.iter().sum();

    if max_excitation >= t_d {
        return Feasibility::SingleStep;
    }

    if max_excitation <= 0 {
        return Feasibility::Impossible;
    }

    // With multiplicative leak, steady-state under sustained input:
    // p_ss = max_excitation / (1 - r)
    let leak_factor = 1.0 - retention_rate;
    if leak_factor <= 1e-9 {
        // r ≈ 1.0, no leak → always accumulates (multi-step if excitation < T_d)
        return Feasibility::MultiStep { min_steps: ((t_d as f64) / max_excitation as f64).ceil() as i32 };
    }

    let p_ss = max_excitation as f64 / leak_factor;
    if p_ss < t_d as f64 {
        return Feasibility::Impossible;
    }

    // Estimate steps: solve r^n * 0 + E * (1 - r^n) / (1 - r) >= T_d
    // Simplification: n = ceil(log(1 - T_d*(1-r)/E) / log(r))... just iterate
    let mut p = 0.0_f64;
    for step in 1..=1000 {
        p = retention_rate * p + max_excitation as f64;
        if p >= t_d as f64 {
            return Feasibility::MultiStep { min_steps: step };
        }
    }
    Feasibility::Impossible
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
        // δ_5(50) = round(50 * 5 / 100) = round(2.5) = 3
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

    // --- P_MIN computation ---

    #[test]
    fn test_p_min_no_inhibitory() {
        assert_eq!(compute_discretized_p_min(&[3, 2]), 0);
    }

    #[test]
    fn test_p_min_with_inhibitory() {
        // Three inhibitory weights of -3 each
        assert_eq!(compute_discretized_p_min(&[-3, -3, -3]), -9);
    }

    #[test]
    fn test_p_min_mixed() {
        // Mix of excitatory (+3) and inhibitory (-2, -3)
        assert_eq!(compute_discretized_p_min(&[3, -2, -3]), -5);
    }

    #[test]
    fn test_p_min_empty() {
        assert_eq!(compute_discretized_p_min(&[]), 0);
    }

    // --- Feasibility (with multiplicative leak) ---

    #[test]
    fn test_feasibility_single_step() {
        // One excitatory weight of 3, threshold 3, r=0.5 → fires in one step
        assert_eq!(check_feasibility(&[3], 3, 0.5), Feasibility::SingleStep);
    }

    #[test]
    fn test_feasibility_multi_step() {
        // Excitation=2 per step, T_d=5, r=0.5
        // Step 1: 0*0.5 + 2 = 2
        // Step 2: 2*0.5 + 2 = 3
        // Step 3: 3*0.5 + 2 = 3.5
        // Step 4: 3.5*0.5 + 2 = 3.75
        // Step 5: 3.75*0.5 + 2 = 3.875
        // Steady state: 2/(1-0.5) = 4 < 5 → Impossible
        assert_eq!(check_feasibility(&[2], 5, 0.5), Feasibility::Impossible);
    }

    #[test]
    fn test_feasibility_multi_step_achievable() {
        // Excitation=2 per step, T_d=3, r=0.5
        // Step 1: 2, Step 2: 3 → reaches threshold
        assert_eq!(
            check_feasibility(&[2], 3, 0.5),
            Feasibility::MultiStep { min_steps: 2 }
        );
    }

    #[test]
    fn test_feasibility_impossible_no_excitation() {
        assert_eq!(check_feasibility(&[], 3, 0.5), Feasibility::Impossible);
    }

    #[test]
    fn test_feasibility_no_leak() {
        // r=1.0 (no leak) → accumulates indefinitely
        // Excitation=1 per step, T_d=3 → 3 steps
        assert_eq!(
            check_feasibility(&[1], 3, 1.0),
            Feasibility::MultiStep { min_steps: 3 }
        );
    }

    #[test]
    fn test_feasibility_multiple_inputs() {
        // Two excitatory weights [2, 1] = 3 total, T_d=5, r=0.5
        // Step 1: 3, Step 2: 1.5+3=4.5, Step 3: 2.25+3=5.25 ≥ 5 → 3 steps
        assert_eq!(
            check_feasibility(&[2, 1], 5, 0.5),
            Feasibility::MultiStep { min_steps: 3 }
        );
    }
}
