"""The latency calibration bridge -- the new spiking-facing piece.

This module connects the paper's exact identity to Nengo's actual LIF neuron.

Paper (idealized, no refractory):
    drive    R I(q) = theta / (1 - q^alpha),   alpha = lambda / (tau ln2)
    latency  t*(q) = -lambda log2 q            (first spike from rest, exact)

Nengo LIF first spike from rest (threshold normalized to 1, input current J):
    t = tau_rc * ln( J / (J - 1) )             (NO tau_ref: the refractory period
                                                is post-spike dead time and does not
                                                delay the first spike from rest)

Composing the calibration drive J = R I(q) with the Nengo law gives, with
lambda = tau_rc and alpha = 1/ln2, exactly t = -lambda log2 q (verified to ~dt in
e01/e02). The honest limits the paper flags become concrete here:
  * finite drive: J = 1/(1 - q^alpha) -> inf as q -> 1, so a few-fold rheobase
    ceiling pins q_max (q_max_for_ceiling);
  * finite timing resolution dt: the smallest latencies (high q) round to dt;
  * q -> 0: drive -> rheobase, latency diverges and is noise-dominated.
"""

import numpy as np

from .config import ALPHA, LAMBDA, RHEOBASE_CEILING, TAU_RC, THETA


def alpha_of(lam=LAMBDA, tau=TAU_RC):
    """Calibration exponent alpha = lambda / (tau ln2)."""
    return lam / (tau * np.log(2.0))


def calibration_drive(q, lam=LAMBDA, tau=TAU_RC, theta=THETA):
    """The calibration drive R I(q) = theta / (1 - q^alpha) (in threshold units).

    Vectorized. q must be in (0, 1); at q -> 1 the drive diverges.
    """
    q = np.asarray(q, float)
    a = alpha_of(lam, tau)
    return theta / (1.0 - q ** a)


def analytic_latency_ideal(q, lam=LAMBDA):
    """The paper's exact latency t*(q) = -lambda log2 q (seconds)."""
    q = np.asarray(q, float)
    return -lam * np.log2(q)


def nengo_first_spike_time(J, tau_rc=TAU_RC):
    """Nengo LIF first-spike latency from rest: tau_rc * ln(J/(J-1)).

    Returns inf for sub-rheobase drive (J <= 1). No tau_ref term -- the refractory
    period does not delay the first spike from rest.
    """
    J = np.asarray(J, float)
    out = np.full_like(J, np.inf)
    sup = J > 1.0
    out[sup] = tau_rc * np.log(J[sup] / (J[sup] - 1.0))
    return out if out.shape else float(out)


def nengo_isi(J, tau_rc=TAU_RC, tau_ref=0.002):
    """Nengo LIF steady-state inter-spike interval: tau_ref + tau_rc ln(J/(J-1)).

    This is the *repeated-firing* (count-code) period; it DOES include tau_ref,
    unlike the first spike from rest. Provided for the count-vs-latency comparison.
    """
    return tau_ref + nengo_first_spike_time(J, tau_rc)


def drive_for_q(q, lam=LAMBDA, tau=TAU_RC, theta=THETA):
    """Alias of calibration_drive (the J to inject for model probability q)."""
    return calibration_drive(q, lam, tau, theta)


def q_for_drive(J, lam=LAMBDA, tau=TAU_RC, theta=THETA):
    """Invert the calibration drive: the q encoded by drive J = theta/(1-q^alpha)."""
    J = np.asarray(J, float)
    a = alpha_of(lam, tau)
    return (1.0 - theta / J) ** (1.0 / a)


def q_max_for_ceiling(ceiling=RHEOBASE_CEILING, lam=LAMBDA, tau=TAU_RC, theta=THETA):
    """Maximum representable probability q_max: the q whose drive hits the ceiling.

    Solve theta/(1 - q^alpha) = ceiling*theta  ->  q_max = (1 - 1/ceiling)^(1/alpha).
    """
    a = alpha_of(lam, tau)
    return (1.0 - 1.0 / ceiling) ** (1.0 / a)


def t_min_for_ceiling(ceiling=RHEOBASE_CEILING, tau_rc=TAU_RC, lam=LAMBDA, tau=TAU_RC):
    """Minimum latency (floor cost) on the most-confident representable symbol q_max."""
    return float(analytic_latency_ideal(q_max_for_ceiling(ceiling, lam, tau), lam))


def drive_table(qs=(0.5, 0.7, 0.9, 0.95, 0.99), lam=LAMBDA, tau=TAU_RC, theta=THETA):
    """Reproduce the paper's calibration-drive table: q -> (R I/theta, surprisal bits)."""
    rows = []
    for q in qs:
        rows.append((q, float(calibration_drive(q, lam, tau, theta) / theta),
                     float(-np.log2(q))))
    return rows
