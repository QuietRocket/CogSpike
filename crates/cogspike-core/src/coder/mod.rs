//! The latency entropy-coder: the idealized (closed-form, byte-faithful) substrate.
//!
//! Ports `research/compression/nengo_experiments/spikecoder/*.py` and the numpy
//! validators (`validate.py`, `learn_validate.py`). The construction chain is:
//! source symbols -> context `c` -> `q = softmax(W c)` -> first-spike latency
//! `t*(q) = -lambda log2 q` -> temporal winner-take-all decode, learned by the
//! local delta rule `dW = eta*c*(y - q)`.
//!
//! Every closed-form quantity is asserted against checked-in golden fixtures
//! (`tests/coder_parity.rs`) to `<= 1e-12`; the named rover constants to `5e-5`.
//! Sampled quantities (which depend on numpy's PCG64) are validated only by
//! statistical convergence, never exact stream equality.

pub mod constants;
pub mod information;
pub mod latency;
pub mod metrics;
pub mod source;

pub use constants::{
    DT, H_MARGINAL, H_RATE, LABELS, LAMBDA, LEARNED_REF, LN2, MAX_SURPRISAL_BITS, MUTUAL_INFO, PI,
    Q_CLIP_HI, Q_CLIP_LO, RHEOBASE_CEILING, S, SEED, TAU_RC, THETA,
};
pub use information::{
    cross_entropy_bits, cross_entropy_rate, entropy_bits, entropy_rate, kl_bits, learned_matrix,
    model_energy_and_kl, softmax,
};
pub use latency::{
    alpha_of, analytic_latency_ideal, calibration_drive, nengo_first_spike_time, q_for_drive,
    q_max_for_ceiling, t_min_for_ceiling,
};
pub use metrics::{
    decode_error_rate, decode_first_spike, energy_trajectory, latency_gap_margin,
    mean_bits_per_symbol,
};
pub use source::{RoverSource, momentum_chain, one_hot, sample_stream};
