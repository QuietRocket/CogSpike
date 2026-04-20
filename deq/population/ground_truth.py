"""Behavioral classifiers used by the sweep scripts.

A classifier maps a parameter-space cell to a boolean verdict (did the
qualitative behavior of interest occur?). The classifier *implementation*
is simulation-based; the downstream analysis compares these verdicts
against the spectral prediction.
"""

from __future__ import annotations

import numpy as np

from wilson_cowan import (
    Sigmoid,
    find_fixed_point,
    find_saddle_contralateral,
    simulate,
)


def wta_contralateral(
    w12: float,
    w21: float,
    drive: float,
    tau: float,
    sigmoid: Sigmoid,
    t_final: float = 50.0,
    perturbation: float = 0.05,
    margin: float = 0.3,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_step: float = 0.5,
):
    """Return True iff the two mirror-image perturbations commit to DIFFERENT winners.

    The perturbations start from the saddle (middle root of the scalar
    reduction). In the bistable regime the saddle is unstable and each
    perturbation direction is pulled onto its own stable attractor, so the
    two trajectories end with opposite signs of (rho_1 - rho_2). In the
    monostable regime both perturbations relax to the single attractor and
    end with the same sign, regardless of how asymmetric that single
    attractor happens to be for skew weights.

    Requiring the sign opposition is the faithful reading of the plan's
    parenthetical -- 'the symmetric state is unstable and the system
    commits to one population dominating [which population depends on the
    initial condition]' -- and rules out false positives from strongly
    skewed single-FP regimes.

    Tolerances default to rtol=1e-6, atol=1e-8 and max_step=0.5 for sweep
    use; these are loose relative to wilson_cowan.simulate defaults but
    ample for classifying the sign of the final asymmetry.
    """
    W = np.array([[0.0, -w21], [-w12, 0.0]])
    I = np.array([drive, drive])
    rho_star, n_fp = find_saddle_contralateral(w12, w21, drive, sigmoid)
    if n_fp == 0 or np.any(np.isnan(rho_star)):
        return False, rho_star

    d1 = np.array([perturbation, -perturbation])
    d2 = -d1
    t_eval = np.array([t_final])
    _, y1 = simulate(
        W, I, tau, sigmoid, t_span=(0.0, t_final), rho0=rho_star + d1,
        rtol=rtol, atol=atol, max_step=max_step, t_eval=t_eval,
    )
    _, y2 = simulate(
        W, I, tau, sigmoid, t_span=(0.0, t_final), rho0=rho_star + d2,
        rtol=rtol, atol=atol, max_step=max_step, t_eval=t_eval,
    )

    delta1 = float(y1[0, -1] - y1[1, -1])
    delta2 = float(y2[0, -1] - y2[1, -1])
    both_strong = abs(delta1) > margin and abs(delta2) > margin
    sign_opposite = (delta1 * delta2) < 0
    return bool(both_strong and sign_opposite), rho_star
