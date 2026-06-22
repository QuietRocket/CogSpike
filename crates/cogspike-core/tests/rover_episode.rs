//! Headless rover episode: the `reset -> act -> step -> learn` loop must descend
//! from the uniform-predictor energy `log2(4) = 2.0` toward the entropy-rate floor
//! `H_RATE = 0.9782` and the Bayes accuracy ceiling `0.8031`, reproducing the
//! online local rule of `learn_validate.py` (and capstone e12).

use cog_spike::coder::H_RATE;
use cog_spike::gym::{DeltaAgent, RoverEnv, run_episode};

#[test]
fn rover_episode_converges_to_entropy_rate_floor() {
    let mut env = RoverEnv::paper(1_000_000);
    let mut agent = DeltaAgent::paper(4);
    let r = run_episode(&mut env, &mut agent, 7, 100_000);

    let floor = *H_RATE;

    // Starts at the uniform-predictor energy log2(4) = 2.0 bits/symbol.
    assert!(
        (r.initial_energy - 2.0).abs() < 1e-12,
        "initial energy {} should be log2(4) = 2.0",
        r.initial_energy
    );

    // Energy is a Lyapunov function: it descends, never below the entropy-rate
    // floor (Gibbs), and lands near the learned reference ~0.9787.
    assert!(
        r.final_energy < r.initial_energy,
        "energy must decrease: {} !< {}",
        r.final_energy,
        r.initial_energy
    );
    assert!(
        r.final_energy >= floor - 1e-9,
        "final energy {} fell below the entropy-rate floor {}",
        r.final_energy,
        floor
    );
    assert!(
        r.final_energy < floor + 0.02,
        "final energy {} not near the floor {}",
        r.final_energy,
        floor
    );

    // The learned conditional law matches the true momentum chain.
    assert!(
        r.max_abs_q_minus_p < 0.06,
        "max|q - P| = {} (learned law should match P)",
        r.max_abs_q_minus_p
    );

    // The money-plot quantities over the final window: bits/symbol near the floor,
    // accuracy near the Bayes ceiling 0.8031.
    assert!(
        r.mean_bits_window < floor + 0.05,
        "windowed mean bits/symbol {} not near the floor {}",
        r.mean_bits_window,
        floor
    );
    assert!(
        r.accuracy_window > 0.78,
        "windowed accuracy {} below the Bayes-ceiling neighborhood (~0.803)",
        r.accuracy_window
    );
}
