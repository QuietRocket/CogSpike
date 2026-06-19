"""Reusable Nengo network builders -- the heart of the accumulation layer.

Every builder encapsulates a Nengo idiom validated in a lower-tier experiment, so
higher tiers compose them instead of re-deriving them. The load-bearing idioms:

  * **Pin the initial voltage to rest.** Nengo randomizes LIF membrane voltage in
    [0, 1) to desynchronize neurons; for the first-spike latency code this must be
    pinned to 0 (``make_lif`` does this). The per-symbol-window reset re-establishes
    rest before each symbol, which is what makes the latency code refractory-immune.
  * **Inject a known current** by connecting directly to ``ens.neurons`` with
    ``gain=1, bias=0`` -- then the input current equals the node value (in threshold
    units), so ``J = R I(q)``.
  * **Softmax is the exact normalizer**; the NEF decodes it approximately, so the
    partition of unity becomes representational (e03 measures the gap).
"""

import numpy as np
import nengo

from .config import (ALPHA, DT, LAMBDA, Q_CLIP_HI, Q_CLIP_LO, TAU_RC, TAU_REF, THETA)
from .information import softmax


def make_lif(tau_rc=TAU_RC, tau_ref=TAU_REF, init_voltage=0.0):
    """A Nengo LIF neuron type with the initial membrane voltage pinned.

    init_voltage=0.0 starts the neuron from rest (required for the latency code).
    """
    return nengo.LIF(
        tau_rc=tau_rc,
        tau_ref=tau_ref,
        initial_state={"voltage": nengo.dists.Choice([init_voltage])},
    )


def clamp_q(q, lo=Q_CLIP_LO, hi=Q_CLIP_HI):
    """Clamp model probabilities to the exactly-representable band (keeps drive finite)."""
    return np.clip(q, lo, hi)


def add_current_neuron(net, current, tau_rc=TAU_RC, tau_ref=TAU_REF, label=None):
    """One LIF neuron driven by a constant or Node ``current`` (R I, threshold units).

    Returns the ensemble (1 neuron). Connect a spike probe to ``ens.neurons``.
    """
    with net:
        ens = nengo.Ensemble(
            1, 1, neuron_type=make_lif(tau_rc, tau_ref),
            gain=[1.0], bias=[0.0], encoders=[[1.0]], label=label,
        )
        src = current if isinstance(current, nengo.Node) else nengo.Node(current)
        nengo.Connection(src, ens.neurons, synapse=None)
    return ens


def calibration_drive_node(net, q_source, N, lam=LAMBDA, tau=TAU_RC, theta=THETA,
                           label="calib_drive"):
    """A Node mapping an N-dim model-probability source q -> drive R I(q)=theta/(1-q^a).

    Clamps q to the representable band so the drive stays finite. Returns the Node.
    """
    a = lam / (tau * np.log(2.0))

    def f(t, q):
        q = clamp_q(q)
        return theta / (1.0 - q ** a)

    with net:
        node = nengo.Node(f, size_in=N, size_out=N, label=label)
        nengo.Connection(q_source, node, synapse=None)
    return node


def build_readout_bank(net, q_source, N=4, tau_rc=TAU_RC, tau_ref=TAU_REF,
                       lam=LAMBDA, theta=THETA, label="readout"):
    """N calibrated LIF readout neurons racing to threshold.

    ``q_source`` is an N-dim Node/output giving the model probabilities q_j; readout
    j is driven by the calibration current R I(q_j), so its first spike from rest
    lands at latency -lambda log2 q_j. Returns (ensemble of N neurons, drive Node).

    Each neuron is an independent LIF unit (current injected directly into
    ``.neurons``); the ensemble's ``dimensions`` is a formality (encoders unused).
    """
    drive = calibration_drive_node(net, q_source, N, lam, tau_rc, theta)
    with net:
        ens = nengo.Ensemble(
            N, 1, neuron_type=make_lif(tau_rc, tau_ref),
            gain=np.ones(N), bias=np.zeros(N), encoders=np.ones((N, 1)),
            label=label,
        )
        nengo.Connection(drive, ens.neurons, synapse=None)
    return ens, drive


def build_softmax_predictor(net, logit_source, N=4, n_neurons=400, radius=4.0,
                            synapse=0.01, seed=1, label="softmax_pred"):
    """NEF ensemble that decodes q = softmax(a) from an N-dim logit source a = W c.

    The radius must cover the logit range the weights produce. Returns
    (ensemble, output Node giving the decoded N-dim q). The decoded q is only
    approximately on the simplex -- e03 quantifies |sum q - 1| vs n_neurons.
    """
    with net:
        ens = nengo.Ensemble(n_neurons, N, radius=radius, seed=seed, label=label)
        nengo.Connection(logit_source, ens, synapse=None)
        out = nengo.Node(size_in=N, label=label + "_q")
        nengo.Connection(ens, out, function=lambda a: softmax(a), synapse=synapse)
    return ens, out


def build_divisive_norm(net, drive_source, N=4, sigma=0.1, synapse=0.005,
                        label="divnorm"):
    """Carandini-Heeger divisive normalization r_i = a_i / (sigma + sum_j a_j).

    Sums to sum_i a_i / (sigma + sum_i a_i) < 1 for sigma > 0 -- the biophysical
    approximation to softmax, exact only as sigma -> 0 (e03 contrasts it). Expects
    nonnegative drives ``a``. Returns the output Node.
    """
    def f(t, a):
        a = np.maximum(a, 0.0)
        return a / (sigma + a.sum())

    with net:
        out = nengo.Node(f, size_in=N, size_out=N, label=label)
        nengo.Connection(drive_source, out, synapse=synapse)
    return out


def windowed_drive(active_fn, window_dur, blank_frac=0.25, blank_value=0.0):
    """Wrap a drive function with a per-window blank that resets the LIF to rest.

    Each symbol window of length ``window_dur`` begins with a blank fraction during
    which the drive is ``blank_value`` (default 0), letting the membrane decay to 0
    (Nengo clamps voltage at >= 0), then applies ``active_fn(t)`` for the rest of the
    window. This realizes the paper's ``new_window`` reset inside one continuous
    simulation and makes per-window first-spike latencies start from rest.

    Returns (wrapped_fn, active_start) where active_start = blank_frac*window_dur is
    the within-window offset at which the calibrated drive begins (latencies are
    measured relative to it).
    """
    active_start = blank_frac * window_dur

    def f(t):
        phase = t % window_dur
        if phase < active_start:
            val = active_fn(t)
            return np.full_like(np.atleast_1d(val), blank_value, dtype=float)
        return active_fn(t)

    return f, active_start
