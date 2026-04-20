"""FCS-accurate discrete LI&F simulator.

Implements the Lustre semantics from De Maria et al. 2020 (FCS §6.2) verbatim:
length-5 windowed integrator with rvector=[10,5,3,2,1], integer-scaled weights
and threshold, reset-after-spike on mem[1..4], and one-tick spike emission delay.

Authoritative semantics (from the Lustre node in §6.2):
    V(t)      = sum_e mem[e](t) * rvector[e]
    localS(t) = (V(t) >= threshold)
    mem[0](t) = sum_i w_i * x_i(t)              [summed weighted inputs]
    mem[k](t) = 0 if pre(localS)                 [reset-after-spike]
                else pre(mem[k-1])               [otherwise shift]
                for k = 1..4
    Spike(t)  = false (at t=0)
                else pre(localS)                 [one-tick emission delay]

`x_i(t)` at tick t is the Spike output of another neuron (which equals that
neuron's localS(t-1)) for recurrent edges, or the external input at tick t.
"""

import numpy as np


DEFAULT_TAU = 105
DEFAULT_RVECTOR = np.array([10, 5, 3, 2, 1], dtype=np.int64)


def simulate(W, B, external_inputs, tau=DEFAULT_TAU, rvector=DEFAULT_RVECTOR,
             T=50, initial_mem=None):
    """Run the FCS LI&F network for T ticks.

    Args:
        W: (n, n) int array. W[i, j] = weight from neuron j's Spike to neuron i.
           Diagonal typically zero (no self-recurrence).
        B: (n, m) int array. B[i, k] = weight from external input k to neuron i.
        external_inputs: (m, T) int/bool array of external signals per tick.
        tau: integer threshold (scaled ×10, default 105).
        rvector: length-sigma int array of leak coefficients.
        T: number of ticks.
        initial_mem: optional (n, sigma) int array overriding the initial mem
                     buffer. Default is all zeros (the FCS initial condition).
                     Note: mem[0] at t=0 is always overwritten by the current-tick
                     input equation, so only initial_mem[:, 1:] has any effect
                     (it sets the "prior history" visible at t=0 through V(0)).

    Returns:
        spikes: (n, T) bool array. spikes[i, t] is the exported Spike(t) of neuron i
                (equal to localS(t-1), with spikes[:, 0] = False).
        local: (n, T) bool array. local[i, t] is localS(t) for neuron i, useful for
               diagnostics / exact-sequence comparisons.
    """
    W = np.asarray(W, dtype=np.int64)
    B = np.asarray(B, dtype=np.int64)
    external_inputs = np.asarray(external_inputs, dtype=np.int64)
    rvec = np.asarray(rvector, dtype=np.int64)

    n = W.shape[0]
    sigma = len(rvec)
    assert W.shape == (n, n)
    assert B.shape[0] == n
    m = B.shape[1]
    assert external_inputs.shape == (m, T)

    if initial_mem is None:
        mem = np.zeros((n, sigma), dtype=np.int64)
    else:
        mem = np.asarray(initial_mem, dtype=np.int64).copy()
        assert mem.shape == (n, sigma)
    localS_prev = np.zeros(n, dtype=bool)
    spikes = np.zeros((n, T), dtype=bool)
    local = np.zeros((n, T), dtype=bool)

    # If initial_mem was provided, compute V(0) BEFORE overwriting mem[0].
    # The user's specification of initial_mem represents the "prior history"
    # visible to the very first V computation — we use its full shape at t=0.
    initial_provided = initial_mem is not None
    initial_mem_copy = mem.copy() if initial_provided else None

    for t in range(T):
        spikes[:, t] = localS_prev if t > 0 else False

        recurrent = W @ spikes[:, t].astype(np.int64)
        external = B @ external_inputs[:, t]
        weighted_input = recurrent + external

        if t == 0:
            if initial_provided:
                # User specified an initial tail. Keep it for this first tick so
                # V(0) can see the injected prior history.
                new_tail = initial_mem_copy[:, 1:]
            else:
                new_tail = np.zeros((n, sigma - 1), dtype=np.int64)
        else:
            new_tail = np.where(
                localS_prev[:, None],
                0,
                mem[:, :-1],
            )
        mem = np.concatenate([weighted_input[:, None], new_tail], axis=1)

        V = mem @ rvec
        localS = V >= tau
        local[:, t] = localS
        localS_prev = localS

    return spikes, local


def spike_sequence_to_str(seq):
    """Render a boolean spike sequence as a compact '01' string."""
    return "".join("1" if s else "0" for s in seq)
