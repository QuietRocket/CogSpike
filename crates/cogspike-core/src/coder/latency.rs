//! Latency calibration, byte-faithful to `spikecoder/latency.py`.
//!
//! Paper identity (idealized, first spike from rest): the calibration drive
//! `R I(q) = theta/(1 - q^alpha)` with `alpha = lambda/(tau ln2)` makes the LIF
//! first-spike latency `t*(q) = -lambda log2 q` -- exactly the Shannon length.

use super::constants::LN2;

/// Calibration exponent `alpha = lambda / (tau ln2)`.
#[must_use]
pub fn alpha_of(lam: f64, tau: f64) -> f64 {
    lam / (tau * LN2)
}

/// Calibration drive `R I(q) = theta / (1 - q^alpha)` in threshold units; `q` in `(0, 1)`.
#[must_use]
pub fn calibration_drive(q: f64, lam: f64, tau: f64, theta: f64) -> f64 {
    theta / (1.0 - q.powf(alpha_of(lam, tau)))
}

/// The paper's exact latency `t*(q) = -lambda log2 q` (seconds).
#[must_use]
pub fn analytic_latency_ideal(q: f64, lam: f64) -> f64 {
    -lam * q.log2()
}

/// Nengo LIF first-spike latency from rest: `tau_rc * ln(J/(J-1))`; `inf` for `J <= 1`.
/// No `tau_ref` term -- the refractory period does not delay the first spike from rest.
#[must_use]
pub fn nengo_first_spike_time(j: f64, tau_rc: f64) -> f64 {
    if j > 1.0 {
        tau_rc * (j / (j - 1.0)).ln()
    } else {
        f64::INFINITY
    }
}

/// Invert the calibration drive: the `q` encoded by drive `J = theta/(1 - q^alpha)`.
#[must_use]
pub fn q_for_drive(j: f64, lam: f64, tau: f64, theta: f64) -> f64 {
    (1.0 - theta / j).powf(1.0 / alpha_of(lam, tau))
}

/// Maximum representable probability `q_max = (1 - 1/ceiling)^(1/alpha)`: the `q`
/// whose calibration drive hits the rheobase `ceiling`.
#[must_use]
pub fn q_max_for_ceiling(ceiling: f64, lam: f64, tau: f64) -> f64 {
    (1.0 - 1.0 / ceiling).powf(1.0 / alpha_of(lam, tau))
}

/// Minimum latency (floor cost) on the most-confident representable symbol `q_max`.
#[must_use]
pub fn t_min_for_ceiling(ceiling: f64, lam: f64, tau: f64) -> f64 {
    analytic_latency_ideal(q_max_for_ceiling(ceiling, lam, tau), lam)
}
