"""Measurement helpers: first-spike extraction, decode, bits/symbol, energy.

These turn raw Nengo spike probes into the information-theoretic quantities the
experiments compare against the paper's closed-form numbers.
"""

import numpy as np

from .config import DT
from .information import softmax


def first_spike_time(spikes_1d, trange):
    """Time of the first spike in a 1-D spike train (inf if none).

    Nengo spike probes record amplitude 1/dt at the spike step; any positive
    value marks a spike.
    """
    idx = np.flatnonzero(np.asarray(spikes_1d) > 0)
    return float(trange[idx[0]]) if len(idx) else np.inf


def per_window_first_spikes(spikes_2d, trange, window_dur, n_windows, dt=DT):
    """First-spike latency of each neuron within each symbol window.

    Returns an array (n_windows, N) of latencies *relative to the window start*
    (inf where a neuron did not spike in that window).

    spikes_2d : (T, N) spike probe data.
    """
    spikes_2d = np.asarray(spikes_2d)
    T, N = spikes_2d.shape
    out = np.full((n_windows, N), np.inf)
    for k in range(n_windows):
        t0 = k * window_dur
        i0 = int(round(t0 / dt))
        i1 = int(round((t0 + window_dur) / dt))
        i1 = min(i1, T)
        seg = spikes_2d[i0:i1]
        for j in range(N):
            idx = np.flatnonzero(seg[:, j] > 0)
            if len(idx):
                out[k, j] = idx[0] * dt
    return out


def decode_first_spike(latencies_2d):
    """First-spike-takes-all decode: argmin latency per window (the winner).

    latencies_2d : (n_windows, N). Returns (decoded ints, winner latency per window).
    Windows with no spike at all decode to -1 with inf latency.
    """
    lat = np.asarray(latencies_2d, float)
    decoded = np.where(np.isfinite(lat).any(axis=1), np.argmin(lat, axis=1), -1)
    win_lat = np.min(lat, axis=1)
    return decoded, win_lat


def decode_error_rate(decoded, emitted):
    """Fraction of windows whose decoded symbol != emitted symbol."""
    decoded = np.asarray(decoded, int)
    emitted = np.asarray(emitted, int)
    m = min(len(decoded), len(emitted))
    return float(np.mean(decoded[:m] != emitted[:m]))


def mean_bits_per_symbol(win_latencies, lam):
    """Mean per-symbol first-spike time converted to bits (divide latency by lambda)."""
    lat = np.asarray(win_latencies, float)
    lat = lat[np.isfinite(lat)]
    return float(np.mean(lat) / lam) if len(lat) else np.inf


def energy_trajectory(q_snapshots, pi, P):
    """Cross-entropy-rate energy E = sum_i pi_i sum_j P_ij(-log2 q_ij) per snapshot.

    q_snapshots : list of (N,N) conditional-law matrices Q.
    Returns array of energies (bits/symbol).
    """
    pi = np.asarray(pi, float)
    out = []
    for Q in q_snapshots:
        Q = np.clip(np.asarray(Q, float), 1e-12, 1.0)
        E = sum(pi[i] * float(-(P[i] * np.log2(Q[i])).sum()) for i in range(len(pi)))
        out.append(E)
    return np.array(out)


def latency_gap_margin(q, lam):
    """The first-spike latency gap lambda*(log2 q_top - log2 q_2nd) between the two
    most probable symbols of a distribution q. Small margin => hard WTA decode.
    """
    q = np.sort(np.asarray(q, float))[::-1]
    if len(q) < 2 or q[1] <= 0:
        return np.inf
    return float(lam * (np.log2(q[0]) - np.log2(q[1])))
