//! Canonical constants of the momentum-rover entropy coder at `s = 0.7`.
//!
//! Lifted byte-faithful from `spikecoder/{config,source,information}.py`. The
//! derived reference values (`H_MARGINAL`, `H_RATE`, `MUTUAL_INFO`) are computed
//! once from the same closed forms the numpy validators use.

use std::sync::LazyLock;

use super::information::{entropy_bits, entropy_rate};
use super::source::momentum_chain;

/// `ln 2`.
pub const LN2: f64 = std::f64::consts::LN_2;

/// Alphabet labels.
pub const LABELS: [&str; 4] = ["U", "D", "L", "R"];

/// Stationary move frequencies `pi = (1/2, 1/4, 1/8, 1/8)`.
pub const PI: [f64; 4] = [0.5, 0.25, 0.125, 0.125];

/// Stickiness of the paper's worked operating point.
pub const S: f64 = 0.7;

/// Membrane time constant `tau_rc` (seconds).
pub const TAU_RC: f64 = 0.02;
/// Refractory period (seconds); does not delay the first spike from rest.
pub const TAU_REF: f64 = 0.002;
/// Firing threshold (Nengo normalizes to 1).
pub const THETA: f64 = 1.0;
/// Time-per-bit constant `lambda = tau_rc` (s/bit), giving `alpha = 1/ln2`.
pub const LAMBDA: f64 = TAU_RC;
/// Max drive `R I/theta` a "real" neuron supplies (few-fold rheobase).
pub const RHEOBASE_CEILING: f64 = 10.0;
/// Clamp `q` away from 0 (drive -> rheobase, noise-dominated).
pub const Q_CLIP_LO: f64 = 1e-4;
/// Clamp `q` away from 1 (drive -> infinity).
pub const Q_CLIP_HI: f64 = 0.999;
/// Default timestep (seconds).
pub const DT: f64 = 1e-3;
/// Fine grid for latency-resolution-critical work (seconds).
pub const DT_FINE: f64 = 1e-4;
/// Stream sampling seed (matches the numpy validators).
pub const SEED: u64 = 7;

/// Converged learned energy (`learn_validate.py`): `~0.9787` bits/symbol.
pub const LEARNED_REF: f64 = 0.9787;

/// Marginal (memoryless) symbol entropy `H(pi) = 1.7500` bits/symbol.
pub static H_MARGINAL: LazyLock<f64> = LazyLock::new(|| entropy_bits(&PI));

/// Conditional entropy rate `H(P) = 0.9782` bits/symbol -- the floor the cycle reaches.
pub static H_RATE: LazyLock<f64> = LazyLock::new(|| entropy_rate(&momentum_chain(&PI, S), &PI));

/// Mutual information `I = H_MARGINAL - H_RATE = 0.7718` bits/symbol.
pub static MUTUAL_INFO: LazyLock<f64> = LazyLock::new(|| *H_MARGINAL - *H_RATE);

/// Bounded worst-case per-symbol surprisal `-log2(Q_CLIP_LO) = 13.2877` bits.
///
/// Because the encoder cannot represent a probability below `Q_CLIP_LO`, no symbol can
/// ever cost more than this many bits; on the idealized substrate that is a bounded
/// first-spike latency `t_max = LAMBDA * 13.2877 = 0.2658 s` (the latency neuron always
/// fires within `t_max`). A clip-enforced a-priori bound -- not a PRISM/PCTL machine-check.
pub static MAX_SURPRISAL_BITS: LazyLock<f64> = LazyLock::new(|| -Q_CLIP_LO.log2());
