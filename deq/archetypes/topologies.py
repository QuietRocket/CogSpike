"""Archetype topology constructors (FCS §6.2.5, §6.2.7, §6.3.4).

Each constructor returns (W, B, external_inputs) for `lif_fcs.simulate`.
All weights are in FCS scaled integer units (×10 vs. §3 rationals).
"""

import numpy as np


DEFAULT_SELF_DRIVE = 11  # delayer threshold: w * 10 = 110 >= tau=105


def negative_loop(w_XA=11, w_AI=11, w_IA=-11, T=50, x_pattern=None):
    """Negative loop archetype (FCS Fig. 1d, §6.2.5).

    Two neurons: activator A (index 0) with external input X and inhibitory
    feedback from I; inhibitor I (index 1) excited by A.

    Default weights: w_XA=11, w_AI=11, w_IA=-11. Note: FCS appendix A.3 suggests
    w_IA=-20 as starting point, but analytical trace of the Lustre semantics
    shows that -11 (inhibition exactly cancelling the drive) is what reproduces
    Property 5's exact 0,1,1,0,0,1,1,0,0,... pattern. -20 overshoots into the
    negative-leak regime and gives a slower period.

    Args:
        w_XA, w_AI, w_IA: scaled integer weights.
        T: simulation ticks.
        x_pattern: (T,) array of external X signals. Defaults to constant 1.

    Returns:
        (W, B, external) ready for simulate().
        Neuron indexing: 0=A (activator), 1=I (inhibitor).
    """
    W = np.array([
        [0, w_IA],  # A receives from I
        [w_AI, 0],  # I receives from A
    ], dtype=np.int64)
    B = np.array([
        [w_XA],  # A receives external X
        [0],     # I has no external
    ], dtype=np.int64)
    if x_pattern is None:
        x_pattern = np.ones(T, dtype=np.int64)
    external = x_pattern.reshape(1, T)
    return W, B, external


def contralateral(w_12, w_21, T=50, self_drive=DEFAULT_SELF_DRIVE):
    """Contralateral inhibition archetype (FCS Fig. 1f, §6.2.7).

    Two mutually inhibiting neurons, each with its own external excitation.
    FCS Fig. 10 sweeps (w_12, w_21) over negative integer pairs.

    Weight convention per FCS Fig. 10 axes (Appendix A.4):
      w_12 = weight of the edge from N1 to N2 (inhibition of N2 by N1)
      w_21 = weight of the edge from N2 to N1 (inhibition of N1 by N2)

    Args:
        w_12, w_21: scaled integer inhibitory weights (typically negative).
        T: simulation ticks.
        self_drive: external-input weight to each neuron (default 11, delayer threshold).

    Returns:
        (W, B, external) with both external inputs = constant 1.
        Neuron indexing: 0=N1, 1=N2.
    """
    W = np.array([
        [0, w_21],   # N1 receives from N2 with weight w_21
        [w_12, 0],   # N2 receives from N1 with weight w_12
    ], dtype=np.int64)
    B = np.array([
        [self_drive, 0],
        [0, self_drive],
    ], dtype=np.int64)
    external = np.ones((2, T), dtype=np.int64)
    return W, B, external


def all_to_all_inhibition(N, w, T=50, self_drive=DEFAULT_SELF_DRIVE, drive_bump=0):
    """N-neuron all-to-all lateral inhibition (generalizes contralateral to N>2).

    Every neuron drives itself with `self_drive` (default 11, delayer threshold)
    and inhibits every other neuron with scaled-integer weight `w` (typically
    negative). `drive_bump` adds an integer to neuron 0's self-drive — the
    FCS-faithful symmetry breaker, since the FCS Lustre semantics with
    symmetric weights and zero initial condition locks all N neurons into
    identical synchronous trajectories.

    Reduces to contralateral(w, w) at N = 2 with drive_bump = 0 (up to the
    shared-w constraint).

    Args:
        N: number of neurons (>= 2).
        w: scaled-integer inhibitory weight (typically negative).
        T: simulation ticks.
        self_drive: external-input weight to each neuron (default 11).
        drive_bump: integer added to B[0, 0] (default 0). Use 1 to break the
            FCS-faithful S_N symmetry.

    Returns:
        (W, B, external) ready for lif_fcs.simulate.
        Neuron indexing: 0 is the (optionally) drive-bumped neuron.
    """
    W = np.full((N, N), int(w), dtype=np.int64)
    np.fill_diagonal(W, 0)
    B = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        B[i, i] = int(self_drive) + (int(drive_bump) if i == 0 else 0)
    external = np.ones((N, T), dtype=np.int64)
    return W, B, external


def contralateral_delayed(w_12, w_21, T=50, self_drive=DEFAULT_SELF_DRIVE,
                          delayer_drive=DEFAULT_SELF_DRIVE):
    """Contralateral inhibition with a delayer on the N1 -> N2 branch (FCS Fig. 11).

    Topology (per plan Appendix A.5 resolution):
      - N1 (index 0): excited by external X1 (weight self_drive), inhibited by
        N2 (weight w_21).
      - N2 (index 1): excited by external X2 (weight self_drive), inhibited by
        the DELAYER (weight w_12).
      - Delayer (index 2): excited by N1's Spike output (weight delayer_drive,
        default 11 → acts as a unit-gain one-tick buffer).

    The swept parameter w_12 lives on delayer -> N2, not on N1 -> delayer.
    N1's inhibition thus reaches N2 one tick later than N2's reaches N1.

    Returns:
        (W, B, external) for a 3-neuron network. Neuron indexing: 0=N1, 1=N2, 2=D.
    """
    W = np.array([
        [0,    w_21, 0],      # N1 inhibited by N2
        [0,    0,    w_12],   # N2 inhibited by delayer
        [delayer_drive, 0, 0],  # delayer driven by N1 spikes
    ], dtype=np.int64)
    B = np.array([
        [self_drive, 0],
        [0, self_drive],
        [0, 0],
    ], dtype=np.int64)
    external = np.ones((2, T), dtype=np.int64)
    return W, B, external
