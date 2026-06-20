#!/usr/bin/env python3
"""e06 -- A ring attractor holds the running context c_t (the previous symbol).

Claim (paper, the recurrent-predictor stage + FIX G / BIO-06): the running
context c_t -- "which previous symbol" -- is held as a bump of GRADED PERSISTENT
ACTIVITY on a line/ring attractor (Seung 1996; Ben-Yishai 1995). The position of
the bump along the manifold *is* the stored value. FIX G is the honesty patch:
such an attractor does NOT store arbitrarily deep history -- it stores a SINGLE
graded analog value, finite and noise-limited, with capacity

    C ~ log2(SNR)  bits,                                      (eq. attractor-capacity)

where SNR = (usable range of the bump) / (positional jitter from drift+noise).
For a 4-symbol alphabet we only need C >= log2(4) = 2 bits.

The spiking-reality gap vs the numpy validator: the validator sets c_t = e_i
*exactly and instantaneously* (a clean one-hot of the previous symbol). Here the
context is an ANALOG bump on a real spiking ring that (a) takes time to WRITE and
SETTLE after each new symbol, and (b) DRIFTS during the hold. We measure all three:
HOLD fidelity, DRIFT (-> effective capacity), and WRITE/SETTLE latency, and show
that 4 context states comfortably fit a first-order source -- while documenting
honestly that the capacity is finite (FIX G made measurable).

Geometry. We encode symbol s in {0,1,2,3} as an angle theta_s = s * 2*pi/4 on the
unit circle, represented in a 2-D NEF ensemble as the point (cos theta, sin theta).
A recurrent connection feeds back the *radially-projected* state -- it pushes the
state back onto the unit circle without rotating it -- creating a continuous RING
of fixed points (a ring attractor). With no input the bump rests wherever it was
written: graded persistent activity (verified below: an arbitrary written angle is
held to within a few degrees with no input).

Writing a NEW symbol onto an already-latched bump. A brief additive write current
cannot overpower the recurrent hold -- for an antipodal target the new input
*cancels* the old bump radially (the state passes through the origin, where the
ring direction is undefined) and the recurrence snaps it back. We therefore load a
new item the way attractor working-memory models do: a brief CLEAR (strong
inhibition of the memory + recurrence) wipes the old bump to zero, then the write
current builds the new bump fresh from the origin -- exactly the regime in which
the ring holds an arbitrary written angle. This clear-then-write is the spiking
realization of the coder's per-symbol `new_window` load of c_t.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import nengo

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

from spikecoder import config as cfg          # noqa: E402
from spikecoder import plotting as plot        # noqa: E402

N_SYM = 4
ANGLES = np.arange(N_SYM) * 2.0 * np.pi / N_SYM          # symbol -> angle on the ring
RING_PTS = np.stack([np.cos(ANGLES), np.sin(ANGLES)], 1)  # (4,2) target points
LABELS = ["U", "D", "L", "R"]

# --- ring + write/clear timing ---
TAU_FB = 0.1            # recurrent (ring-feedback) synapse
N_NEURONS = 700
RADIUS = 1.4
WRITE_GAIN = 8.0        # write-current amplitude (builds the new bump from rest)
CLEAR_INHIB = 20.0      # inhibition that wipes the old bump before a fresh write
CLEAR_DUR = 0.020       # 20 ms clear, then ...
WRITE_DUR = 0.060       # ... write active until 60 ms into the window
HOLD_DUR = 0.140        # 140 ms pure hold -> inter-symbol interval ~ 200 ms
ISI = WRITE_DUR + HOLD_DUR
DT = cfg.DT
half_cell = np.pi / N_SYM   # 45 deg: a symbol is unambiguous while |err| < this


# ---------------------------------------------------------------------------
# The ring attractor builder (experiment-specific; lives here, not in spikecoder).
# ---------------------------------------------------------------------------
def ring_feedback(x):
    """Recurrent transform: pull the 2-D state back onto the unit circle.

    A ring attractor needs the recurrent excitation to balance the leak *along*
    the ring while killing any radial component. With the NEF integrator identity
    Connection(ens, ens, synapse=tau, function=f), the state-update is
    dx/dt ~ (f(x) - x)/tau; choosing f(x) = x/|x| makes every unit-norm point a
    fixed point (the radius is pushed to 1, the tangential direction is untouched)
    => a continuous ring of fixed points. Near the origin (r~0, no bump) the map is
    the identity, so a cleared ring stays cleared until the write builds a bump.
    """
    x = np.asarray(x, float)
    r = np.linalg.norm(x)
    if r < 1e-6:
        return x
    return x / r          # restore radius to 1, keep the angle


def build_ring(net, write_source, clear_source, n_neurons=N_NEURONS, tau=TAU_FB,
               radius=RADIUS, seed=1, label="ring"):
    """A 2-D ring attractor with a clear-then-write load path.

    The recurrent ring feedback runs through a relay ensemble so the hold can be
    momentarily gated off (inhibited) during the clear. write_source is a 2-D Node
    giving the (cos,sin) write current (active only during the write sub-window);
    clear_source is a scalar Node = 1 during the clear sub-window, 0 otherwise, and
    strongly inhibits both the memory and the relay so the old bump is wiped before
    the new one is written from rest. Returns (ens, decoded-(cos,sin) Node).
    """
    with net:
        ens = nengo.Ensemble(n_neurons, 2, radius=radius, seed=seed, label=label)
        relay = nengo.Ensemble(n_neurons, 2, radius=radius, seed=seed + 1,
                               label=label + "_relay")
        # recurrent ring feedback: ens -> relay (ring map) -> ens (hold)
        nengo.Connection(ens, relay, function=ring_feedback, synapse=0.01)
        nengo.Connection(relay, ens, synapse=tau)
        # write: build the bump from rest (scaled so it dominates within the window)
        nengo.Connection(write_source, ens, synapse=0.05, transform=0.05 / tau)
        # clear: wipe both memory and relay during the clear sub-window
        nengo.Connection(clear_source, ens.neurons,
                         transform=-CLEAR_INHIB * np.ones((n_neurons, 1)), synapse=0.003)
        nengo.Connection(clear_source, relay.neurons,
                         transform=-CLEAR_INHIB * np.ones((n_neurons, 1)), synapse=0.003)
        cs_out = nengo.Node(size_in=2, label=label + "_cs")
        nengo.Connection(ens, cs_out, synapse=0.02)
    return ens, cs_out


def make_write_clear(symbol_at, n_windows, isi=ISI):
    """Build (write_fn, clear_fn) for a sequence of per-window symbols.

    symbol_at : callable k -> symbol index (or None to write nothing in window k).
    Within each window: [0, CLEAR_DUR) clear; [CLEAR_DUR, WRITE_DUR) write target;
    [WRITE_DUR, isi) hold.
    """
    def write_fn(t):
        k = int(t // isi)
        if k >= n_windows:
            return np.zeros(2)
        s = symbol_at(k)
        phase = t - k * isi
        if s is not None and CLEAR_DUR <= phase < WRITE_DUR:
            return WRITE_GAIN * RING_PTS[s]
        return np.zeros(2)

    def clear_fn(t):
        k = int(t // isi)
        if k >= n_windows:
            return 0.0
        phase = t - k * isi
        return 1.0 if (symbol_at(k) is not None and phase < CLEAR_DUR) else 0.0

    return write_fn, clear_fn


def decode_angle(cs):
    """(cos,sin) -> angle in [0, 2pi)."""
    return np.mod(np.arctan2(cs[..., 1], cs[..., 0]), 2.0 * np.pi)


def angle_to_symbol(theta):
    """Nearest ring-vertex symbol to an angle (circular nearest neighbour)."""
    d = np.abs(np.mod(theta[..., None] - ANGLES + np.pi, 2 * np.pi) - np.pi)
    return np.argmin(d, axis=-1)


def circ_dist(a, b):
    """Smallest absolute angular distance between two angles (radians)."""
    return np.abs(np.mod(a - b + np.pi, 2 * np.pi) - np.pi)


def run_ring(symbol_at, n_windows, run_dur, seed=1):
    """Simulate the ring over n_windows, writing symbol_at(k) each window.

    Returns (trange, decoded angle(t), decoded (cos,sin)(t)).
    """
    write_fn, clear_fn = make_write_clear(symbol_at, n_windows)
    with nengo.Network(seed=seed) as net:
        write = nengo.Node(write_fn, label="write")
        clear = nengo.Node(clear_fn, label="clear")
        ens, cs_out = build_ring(net, write, clear, seed=seed)
        p_cs = nengo.Probe(cs_out, synapse=0.01)
    with nengo.Simulator(net, dt=DT, progress_bar=False) as sim:
        sim.run(run_dur)
    ts = sim.trange()
    cs = sim.data[p_cs]
    return ts, decode_angle(cs), cs


# ---------------------------------------------------------------------------
print("=" * 70)
print("e06 -- Ring attractor holds the running context c_t")
print("=" * 70)
print(f"alphabet N={N_SYM}, ring angles (deg) = {np.degrees(ANGLES).astype(int)}; "
      f"clear {CLEAR_DUR*1e3:.0f} ms, write {(WRITE_DUR-CLEAR_DUR)*1e3:.0f} ms, "
      f"hold {HOLD_DUR*1e3:.0f} ms, ISI {ISI*1e3:.0f} ms")

# ===========================================================================
# (0) Ring quality: with no input, does the ring hold an ARBITRARY written angle?
#     (the defining property of a continuous ring of fixed points.)
# ===========================================================================
probe_degs = np.array([0, 30, 60, 90, 135, 180, 225, 270, 315, 340])
ring_hold_err = []
for deg in probe_degs:
    a = np.radians(deg)
    tgt = np.array([np.cos(a), np.sin(a)])

    def sym_at(k, tgt=tgt):
        return 0  # placeholder; we override the write target directly below

    # write this arbitrary angle (not a ring vertex) and hold 300 ms
    def write_fn(t, tgt=tgt):
        return WRITE_GAIN * tgt if (CLEAR_DUR <= t < WRITE_DUR) else np.zeros(2)

    def clear_fn(t):
        return 1.0 if t < CLEAR_DUR else 0.0

    with nengo.Network(seed=3) as net:
        w = nengo.Node(write_fn)
        c = nengo.Node(clear_fn)
        ens, cs_out = build_ring(net, w, c, seed=3)
        p = nengo.Probe(cs_out, synapse=0.01)
    with nengo.Simulator(net, dt=DT, progress_bar=False) as sim:
        sim.run(0.35)
    th = decode_angle(sim.data[p])
    err = float(np.degrees(circ_dist(th[-1], a)))
    ring_hold_err.append(err)
ring_hold_err = np.array(ring_hold_err)
print(f"\n[ring quality] holds an arbitrary written angle to within "
      f"max {ring_hold_err.max():.1f} deg (mean {ring_hold_err.mean():.1f} deg) over 300 ms "
      f"-> a genuine continuous ring of fixed points")

# ===========================================================================
# (1) HOLD FIDELITY: write each symbol, then hold for a full inter-symbol
#     interval with NO input and decode the held context across the hold.
# ===========================================================================
hold_err = {}          # symbol -> angular error during the hold (rad)
hold_traces = {}
for s in range(N_SYM):
    ts, th, cs = run_ring(lambda k, s=s: s if k == 0 else None, 1, ISI, seed=10 + s)
    hold_mask = ts >= WRITE_DUR + 0.020   # measure after the write+settle
    err = circ_dist(th[hold_mask], ANGLES[s])
    hold_err[s] = err
    hold_traces[s] = (ts, th)
    dec = angle_to_symbol(th[hold_mask])
    frac_correct = float(np.mean(dec == s))
    print(f"[hold] wrote {LABELS[s]} (theta={np.degrees(ANGLES[s]):.0f} deg): "
          f"mean|err|={np.degrees(err.mean()):.1f} deg, max={np.degrees(err.max()):.1f} deg, "
          f"decoded-correct fraction over hold={frac_correct:.3f}")

all_hold_err = np.concatenate([hold_err[s] for s in range(N_SYM)])
hold_correct_overall = float(np.mean(all_hold_err < half_cell))
print(f"\n[hold] half-cell decode margin = {np.degrees(half_cell):.0f} deg; "
      f"fraction of hold-samples within margin (all symbols) = {hold_correct_overall:.3f}")

# ===========================================================================
# (2) WRITE / SETTLE LATENCY: how long after the window start does the bump arrive
#     at and stay within the half-cell margin of the target?
# ===========================================================================
settle_times = []
for s in range(N_SYM):
    ts, th = hold_traces[s]
    err = circ_dist(th, ANGLES[s])
    within = err < half_cell
    settled_t = np.inf
    for i in range(len(ts)):
        if within[i] and np.all(within[i:]):
            settled_t = ts[i]
            break
    settle_times.append(settled_t)
    print(f"[settle] {LABELS[s]}: settled within half-cell at {settled_t*1e3:.1f} ms "
          f"(write ends at {WRITE_DUR*1e3:.0f} ms, ISI = {ISI*1e3:.0f} ms)")
settle_times = np.array(settle_times)
max_settle = settle_times.max()
print(f"\n[settle] max settle = {max_settle*1e3:.1f} ms  <  ISI = {ISI*1e3:.0f} ms ? "
      f"{max_settle < ISI}")

# ===========================================================================
# (3) DRIFT: write one symbol, then hold for a LONG time with no input, and
#     measure how the decoded angle wanders. The positional jitter sets the SNR
#     and hence the effective capacity C ~ log2(SNR).
# ===========================================================================
LONG_HOLD = 1.0        # 1 s of pure holding, no input
N_DRIFT_SEEDS = 8
drift_curves = []      # (t, signed displacement from the settled value)
for k in range(N_DRIFT_SEEDS):
    s = k % N_SYM
    ts, th, cs = run_ring(lambda kk, s=s: s if kk == 0 else None, 1,
                          WRITE_DUR + LONG_HOLD, seed=100 + k)
    hold_mask = ts >= WRITE_DUR + 0.030
    th_hold = th[hold_mask]
    t_hold = ts[hold_mask] - ts[hold_mask][0]
    ref = th_hold[0]
    disp = np.mod(th_hold - ref + np.pi, 2 * np.pi) - np.pi
    drift_curves.append((t_hold, disp))

Tmin = min(len(c[1]) for c in drift_curves)
tgrid = drift_curves[0][0][:Tmin]
disp_mat = np.stack([c[1][:Tmin] for c in drift_curves])    # (seeds, T)
drift_std = disp_mat.std(axis=0)
rms_drift = np.sqrt((disp_mat ** 2).mean(axis=0))           # rms wander vs hold time

late = tgrid >= 0.2
drift_rate = np.degrees(np.polyfit(tgrid[late], rms_drift[late], 1)[0])  # deg/s
isi_idx = np.searchsorted(tgrid, HOLD_DUR)
jitter_isi = np.degrees(rms_drift[min(isi_idx, Tmin - 1)])  # rms wander after one ISI
jitter_long = np.degrees(rms_drift[-1])                     # rms wander after the long hold
print(f"\n[drift] drift rate (rms-wander slope, late window) = {drift_rate:.2f} deg/s")
print(f"[drift] rms wander after one ISI ({HOLD_DUR*1e3:.0f} ms hold) = {jitter_isi:.2f} deg")
print(f"[drift] rms wander after long hold ({LONG_HOLD*1e3:.0f} ms)   = {jitter_long:.2f} deg")

# ===========================================================================
# Effective capacity C ~ log2(SNR). Usable range = full ring (360 deg). The
# positional jitter is the spread that makes two bump positions confusable.
# Report capacity at the inter-symbol timescale (the relevant one for the coder)
# and after the long hold (the worst case FIX G bounds).
# ===========================================================================
USABLE_RANGE = 360.0
snr_isi = USABLE_RANGE / max(jitter_isi, 1e-6)
snr_long = USABLE_RANGE / max(jitter_long, 1e-6)
C_isi = np.log2(snr_isi)
C_long = np.log2(snr_long)
C_needed = np.log2(N_SYM)
print(f"\n[capacity] usable range = {USABLE_RANGE:.0f} deg")
print(f"[capacity] SNR(ISI)  = {snr_isi:.1f} -> C = log2(SNR) = {C_isi:.2f} bits "
      f"({snr_isi:.0f} distinguishable states)")
print(f"[capacity] SNR(long) = {snr_long:.1f} -> C = log2(SNR) = {C_long:.2f} bits "
      f"({snr_long:.0f} distinguishable states)")
print(f"[capacity] needed for N={N_SYM} symbols: log2({N_SYM}) = {C_needed:.2f} bits")
print(f"[capacity] 4 states comfortably fit at the ISI timescale: {C_isi >= C_needed}")

# ===========================================================================
# (4) A multi-symbol HOLD trace: stream several symbols through ONE continuous
#     ring sim, clearing+writing each at its window start and holding between --
#     the "bump-hold" picture (decoded context vs time across several symbols).
# ===========================================================================
stream = np.array([0, 2, 1, 3, 0, 1], dtype=int)
ts_stream, th_stream, _ = run_ring(lambda k: int(stream[k]) if k < len(stream) else None,
                                   len(stream), len(stream) * ISI, seed=7)

# per-window decode of the held context, read just before the next write
stream_decode = []
for k in range(len(stream)):
    t_read = (k + 1) * ISI - 0.005
    i = min(np.searchsorted(ts_stream, t_read), len(th_stream) - 1)
    stream_decode.append(int(angle_to_symbol(th_stream[i])))
stream_decode = np.array(stream_decode)
stream_err = float(np.mean(stream_decode != stream))
print(f"\n[stream] wrote        {[LABELS[s] for s in stream]}")
print(f"[stream] held/decoded {[LABELS[s] for s in stream_decode]}")
print(f"[stream] per-window hold decode error = {stream_err:.3f}")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# Fig 1: multi-symbol bump-hold trace (decoded context vs time)
fig, ax = plot.new_fig(w=7.5, h=4.0)
ax.plot(ts_stream, np.degrees(th_stream), "-", color=plot.C_MEASURED, lw=1.4,
        label="decoded bump angle")
for k, s in enumerate(stream):
    x0 = (k * ISI) / (len(stream) * ISI)
    x1 = ((k + 1) * ISI) / (len(stream) * ISI)
    ax.axhline(np.degrees(ANGLES[s]), x0, x1, color=plot.C_THEORY, lw=2.5, alpha=0.8)
    ax.axvline(k * ISI, color="gray", ls=":", lw=0.6, alpha=0.6)
    ax.text(k * ISI + WRITE_DUR / 2, 352, LABELS[s], color=plot.C_THEORY,
            ha="center", fontsize=9, weight="bold")
ax.set_yticks(np.degrees(ANGLES))
ax.set_yticklabels([f"{LABELS[s]}\n{int(np.degrees(ANGLES[s]))}°" for s in range(N_SYM)])
ax.set_ylim(-30, 390)
ax.set_xlabel("time (s)")
ax.set_ylabel("held context (ring angle)")
ax.set_title("e06: the ring holds each written symbol across its inter-symbol interval")
ax.legend(frameon=False, loc="lower right")
plot.save(fig, RESULTS / "e06_bump_hold.pdf")

# Fig 2: drift curves (rms wander vs hold time)
fig, ax = plot.new_fig()
for (t_h, disp) in drift_curves:
    ax.plot(t_h, np.degrees(disp), "-", color="gray", lw=0.7, alpha=0.5)
ax.plot(tgrid, np.degrees(rms_drift), "-", color=plot.C_MEASURED, lw=2.2,
        label="rms wander across seeds")
ax.axhline(np.degrees(half_cell), color=plot.C_THEORY, ls="--", lw=1.5,
           label=f"half-cell margin {np.degrees(half_cell):.0f}°")
ax.axvline(HOLD_DUR, color=plot.C_FLOOR, ls=":", lw=1.3,
           label=f"one ISI hold ({HOLD_DUR*1e3:.0f} ms)")
ax.set_xlabel("hold time with no input (s)")
ax.set_ylabel("drift from written angle (deg)")
ax.set_title("e06: the held bump drifts -- bounded, well inside the half-cell margin")
ax.legend(frameon=False, fontsize=8)
plot.save(fig, RESULTS / "e06_drift.pdf")

# Fig 3: effective capacity / SNR
fig, ax = plot.new_fig()
labels = [f"at ISI\n({HOLD_DUR*1e3:.0f} ms hold)", f"after long hold\n({LONG_HOLD*1e3:.0f} ms)"]
caps = [C_isi, C_long]
bars = ax.bar(labels, caps, 0.5, color=plot.C_MEASURED, label="effective capacity log₂(SNR)")
ax.axhline(C_needed, color=plot.C_THEORY, ls="--", lw=2,
           label=f"needed for N={N_SYM}: log₂({N_SYM}) = {C_needed:.0f} bits")
for b, c in zip(bars, caps):
    ax.text(b.get_x() + b.get_width() / 2, c + 0.1, f"{c:.2f}", ha="center", fontsize=9)
ax.set_ylabel("context capacity (bits)")
ax.set_title("e06: effective capacity comfortably exceeds 2 bits (FIX G, measured)")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_ylim(0, max(caps) + 1.2)
plot.save(fig, RESULTS / "e06_capacity.pdf")

# ---------------------------------------------------------------------------
# Save + acceptance
# ---------------------------------------------------------------------------
np.savez(RESULTS / "e06_results.npz",
         angles=ANGLES, ring_hold_err=ring_hold_err, hold_correct_overall=hold_correct_overall,
         all_hold_err=all_hold_err, settle_times=settle_times, ISI=ISI,
         tgrid=tgrid, rms_drift=rms_drift, drift_std=drift_std, drift_rate=drift_rate,
         jitter_isi=jitter_isi, jitter_long=jitter_long,
         C_isi=C_isi, C_long=C_long, C_needed=C_needed, snr_isi=snr_isi, snr_long=snr_long,
         stream=stream, stream_decode=stream_decode, stream_err=stream_err,
         ts_stream=ts_stream, th_stream=th_stream)

checks = {
    "ring holds a written symbol over an ISI (>=95% of hold within half-cell)":
        hold_correct_overall >= 0.95,
    "per-window hold decode of a 6-symbol stream is correct (0 errors)":
        stream_err == 0.0,
    "write/settle latency < inter-symbol interval (all symbols)":
        bool(np.all(settle_times < ISI)),
    "drift is bounded: rms wander after one ISI < half-cell margin":
        jitter_isi < np.degrees(half_cell),
    "effective capacity at ISI >= log2(4) = 2 bits (4 states distinguishable)":
        C_isi >= C_needed,
}
print("\n" + "=" * 70)
allok = True
for name, ok in checks.items():
    allok &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
print(f"\ne06: {sum(checks.values())}/{len(checks)} acceptance checks passed.")
sys.exit(0 if allok else 1)
