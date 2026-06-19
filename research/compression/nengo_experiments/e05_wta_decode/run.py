#!/usr/bin/env python3
"""e05 -- The temporal winner-take-all decoder: first-spike-takes-all.

Claim (paper §"Stage 3", the structural-safety half of the coder): the decoder
commits to the FIRST readout to cross threshold (first-spike-takes-all). This makes
the code *lossless* -- decoded == emitted -- for ANY predictor, even a uniform
q = 1/4 (which decodes correctly, just slowly). Correctness is STRUCTURAL (it rests
on the decode rule, not on the predictor); only the *speed* is quantitative. The
structural-safety property is the SETTLED form
    P_{>=1}[ F G ( sum_j winner_j = 1 ) ]          (paper eq-fg)
-- "eventually exactly one settled winner per window" -- NOT a global "always <= 1
winner", which is false during the integration transient where several readouts
charge toward threshold at once.

We build TWO layers.

  (A) The decode itself = argmin first-spike-time over a calibrated readout bank.
      Encoder model (the paper's transmission picture): the source emits x_t, and
      *that* symbol's readout is driven at q(x_t|c_t); the decoder names whichever
      readout fires first. Over a rover stream we confirm decoded == emitted with
      ZERO errors noise-free (losslessness), for BOTH the perfect predictor
      q = P[x_{t-1}] AND a uniform predictor q = 1/4 (slow, never wrong). Then under
      membrane noise we run a controlled two-readout race and show the decode error
      rate rises and CONCENTRATES at small latency-gap margin
      m = lambda*(log2 q_top - log2 q_2nd), exactly as predicted.

  (B) An actual dynamical temporal-WTA layer: N=4 nonnegative activity channels that
      self-excite and laterally inhibit, fed by the readout spikes. With the emitted
      symbol's line driven, it LATCHES a single winner; we measure the settling time
      (which scales with surprisal). With ALL lines driven (the genuine race), the
      leader crosses the commit threshold FIRST (correct first-spike-takes-all
      decode) -- but if left to free-run, slow near-rheobase losers co-latch into a
      multi-winner steady state. That is exactly the transient the global "<=1
      always" claim mistakes for an invariant, and that first-spike-takes-all
      tolerates: we document it honestly.

Outputs: figures + a results .npz, and a PASS/FAIL line per acceptance check.
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
from spikecoder import latency as lat          # noqa: E402
from spikecoder import metrics as met          # noqa: E402
from spikecoder import plotting as plot        # noqa: E402
from spikecoder.networks import make_lif       # noqa: E402
from spikecoder.source import RoverSource      # noqa: E402

N = 4
LAM = cfg.LAMBDA
Q_LO, Q_HI = cfg.Q_CLIP_LO, cfg.Q_CLIP_HI


# ===========================================================================
# Experiment-specific builders (kept INSIDE run.py; do not touch spikecoder).
# ===========================================================================

def emitted_drive_node(symbols, predictor, window_dur, blank_frac, active_start):
    """A drive-current Node for the encoder model.

    Only the EMITTED symbol's readout is driven, at the calibration current for its
    model probability q(x_t|c_t). Every other line gets J = 0 (truly silent, never
    fires -- NOT the clamped-q rheobase drive). A per-window blank sub-window resets
    every line to rest. ``predictor(k)`` returns the full N-dim model vector q for
    window k; we inject calibration_drive(q[x_k]) on line x_k only.
    """
    symbols = np.asarray(symbols, int)
    n_sym = len(symbols)

    def f(t):
        k = min(int(t // window_dur), n_sym - 1)
        J = np.zeros(N)
        if (t % window_dur) >= active_start:
            q = predictor(k)
            x = symbols[k]
            J[x] = lat.calibration_drive(np.clip(q[x], Q_LO, Q_HI))
        return J

    return f


def run_readout_bank(drive_fn, n_sym, window_dur, sigma=0.0, dt=cfg.DT_FINE,
                     seed=cfg.SEED):
    """Run a direct-current readout bank and return per-window first-spike latencies.

    The drive Node injects an N-dim current vector directly into ``ens.neurons``
    (gain=1, bias=0 => input current == node value in threshold units). Optional
    independent membrane white noise (size_out=N) on each line. Returns
    (latencies (n_windows, N) relative to window start, decoded ints, winner latency).
    """
    with nengo.Network(seed=seed) as net:
        Jnode = nengo.Node(drive_fn, size_out=N)
        ens = nengo.Ensemble(N, 1, neuron_type=make_lif(), gain=np.ones(N),
                             bias=np.zeros(N), encoders=np.ones((N, 1)))
        nengo.Connection(Jnode, ens.neurons, synapse=None)
        if sigma > 0:
            noise = nengo.Node(nengo.processes.WhiteNoise(
                dist=nengo.dists.Gaussian(0, sigma), scale=False, seed=seed),
                size_out=N)
            nengo.Connection(noise, ens.neurons, synapse=None)
        p = nengo.Probe(ens.neurons)
    with nengo.Simulator(net, dt=dt, progress_bar=False) as sim:
        sim.run(n_sym * window_dur)
    lat2d = met.per_window_first_spikes(sim.data[p], sim.trange(), window_dur,
                                        n_sym, dt=dt)
    decoded, win_lat = met.decode_first_spike(lat2d)
    return lat2d, decoded, win_lat


def two_readout_race(q_top, q_2nd, sigma, n_trials, dt=cfg.DT_FINE, dur=0.25,
                     seedbase=0):
    """A controlled noisy two-readout race. Line 0 = correct (emitted) at q_top,
    line 1 = competitor at q_2nd (q_top >= q_2nd). Under membrane noise line 1 may
    cross first -> a decode error. Returns the empirical error rate.
    """
    J = np.array([lat.calibration_drive(q_top), lat.calibration_drive(q_2nd)])
    errs, n_ok = 0, 0
    for s in range(n_trials):
        with nengo.Network(seed=seedbase + s) as net:
            Jn = nengo.Node(J)
            ens = nengo.Ensemble(2, 1, neuron_type=make_lif(), gain=np.ones(2),
                                 bias=np.zeros(2), encoders=np.ones((2, 1)))
            nengo.Connection(Jn, ens.neurons, synapse=None)
            noise = nengo.Node(nengo.processes.WhiteNoise(
                dist=nengo.dists.Gaussian(0, sigma), scale=False,
                seed=1000 + seedbase + s), size_out=2)
            nengo.Connection(noise, ens.neurons, synapse=None)
            p = nengo.Probe(ens.neurons)
        with nengo.Simulator(net, dt=dt, progress_bar=False) as sim:
            sim.run(dur)
        sp, tr = sim.data[p], sim.trange()
        t0 = met.first_spike_time(sp[:, 0], tr)
        t1 = met.first_spike_time(sp[:, 1], tr)
        if np.isinf(t0) and np.isinf(t1):
            continue
        n_ok += 1
        if t1 < t0:                 # competitor crossed first -> wrong symbol decoded
            errs += 1
    return errs / max(n_ok, 1)


def build_wta(net, readout, self_exc=1.0, lat_inh=2.0, fb_syn=0.01, in_syn=0.005,
              evid=1.0, n_per=120):
    """A dynamical temporal-WTA: N nonnegative activity channels (EnsembleArray) with
    self-excitation (+self_exc on the diagonal) and lateral inhibition (-lat_inh off
    the diagonal), fed feedforward by the readout spikes (each spike pumps its own
    channel). The settled stable states of symmetric contralateral inhibition are the
    single-winner configurations (paper eq-fg). Returns a probe on the activity.
    """
    with net:
        wta = nengo.networks.EnsembleArray(
            n_per, N, encoders=nengo.dists.Choice([[1.0]]),
            intercepts=nengo.dists.Uniform(0.05, 0.9),
            eval_points=nengo.dists.Uniform(0, 1), radius=1.0)
        nengo.Connection(readout.neurons, wta.input, transform=evid * np.eye(N),
                         synapse=in_syn)
        Wr = self_exc * np.eye(N) - lat_inh * (np.ones((N, N)) - np.eye(N))
        nengo.Connection(wta.output, wta.input, transform=Wr, synapse=fb_syn)
        p_act = nengo.Probe(wta.output, synapse=0.005)
    return p_act


def run_wta_window(Jvec, dur=0.30, dt=cfg.DT, seed=1, **kw):
    """One decode window: drive the readout bank with current vector Jvec, feed a
    dynamical WTA, return (activity (T,N), trange)."""
    with nengo.Network(seed=seed) as net:
        Jn = nengo.Node(Jvec)
        readout = nengo.Ensemble(N, 1, neuron_type=make_lif(), gain=np.ones(N),
                                 bias=np.zeros(N), encoders=np.ones((N, 1)))
        nengo.Connection(Jn, readout.neurons, synapse=None)
        p_act = build_wta(net, readout, **kw)
    with nengo.Simulator(net, dt=dt, progress_bar=False) as sim:
        sim.run(dur)
    return sim.data[p_act], sim.trange()


def wta_settling_time(act, tr, thr=0.2):
    """Settling time of the F G property: the FIRST time exactly one channel is active
    (> thr) AND it remains the unique active channel, with the same identity, through
    the end of the window. Returns (settle_time, winner) or (inf, -1)."""
    na = (act > thr).sum(axis=1)
    w = np.argmax(act, axis=1)
    for i in range(len(tr)):
        if na[i] == 1 and np.all((act[i:] > thr).sum(axis=1) == 1) \
                and np.all(np.argmax(act[i:], axis=1) == w[i]):
            return float(tr[i]), int(w[i])
    return np.inf, -1


def wta_commit_decode(act, tr, commit=0.5):
    """First-spike-takes-all read-off: the winner is argmax activity at the FIRST time
    any channel crosses the commit level. This is the decode committed at the first
    crossing, before the settling transient resolves. Returns (commit_time, winner)."""
    for i in range(len(tr)):
        if act[i].max() >= commit:
            return float(tr[i]), int(np.argmax(act[i]))
    return np.inf, -1


# ===========================================================================
print("=" * 72)
print("e05 -- Temporal winner-take-all decoder (first-spike-takes-all)")
print("=" * 72)

src = RoverSource()                      # s=0.7, pi=(1/2,1/4,1/8,1/8), seed=7
P = src.P
n_sym = 60
syms = src.sample(n_sym)
window_dur = 0.16                        # long enough for q=1/4 (t* = 2 bits = 40 ms)
blank_frac = 0.30
active_start = blank_frac * window_dur


def perfect_pred(k):
    return src.pi.copy() if k == 0 else P[syms[k - 1]].copy()


def uniform_pred(k):
    return np.ones(N) / N


# ---------------------------------------------------------------------------
# LAYER A.1 -- losslessness: decoded == emitted, noise-free, ANY predictor.
# ---------------------------------------------------------------------------
print("\n[A.1 losslessness] encoder model: only emitted line driven; decode = argmin "
      "first-spike")
lossless = {}
mean_lat = {}
for name, pred in [("perfect", perfect_pred), ("uniform", uniform_pred)]:
    drive_fn = emitted_drive_node(syms, pred, window_dur, blank_frac, active_start)
    lat2d, decoded, win_lat = run_readout_bank(drive_fn, n_sym, window_dur)
    err = met.decode_error_rate(decoded, syms)
    nospk = int((decoded == -1).sum())
    fin = win_lat[np.isfinite(win_lat)] - active_start    # latency after blank
    lossless[name] = err
    mean_lat[name] = float(fin.mean())
    print(f"  {name:8s} predictor: decode error = {err:.4f}  "
          f"(no-spike windows = {nospk})  mean winner latency = "
          f"{fin.mean() * 1e3:6.2f} ms")
print(f"  -> uniform is SLOWER (q=1/4 => t* = {(-LAM * np.log2(0.25)) * 1e3:.1f} ms, "
      f"all symbols equal) but NEVER wrong: 'a bad model is slow, never wrong'.")

# ---------------------------------------------------------------------------
# LAYER A.2 -- noise + margin: decode error concentrates at small latency-gap margin.
# ---------------------------------------------------------------------------
print("\n[A.2 margin] controlled noisy 2-readout race (sigma = 0.5), error vs "
      "latency-gap margin")
sigma = 0.5
n_trials = 120
q_top = 0.5
q_2nds = np.array([0.49, 0.47, 0.45, 0.42, 0.38, 0.33, 0.27, 0.20, 0.13])
margins = np.array([met.latency_gap_margin([q_top, q2, 0.0, 0.0], LAM)
                    for q2 in q_2nds])
err_vs_margin = np.array([two_readout_race(q_top, q2, sigma, n_trials)
                          for q2 in q_2nds])
print("  q_2nd   margin(ms)   error")
for q2, m, e in zip(q_2nds, margins, err_vs_margin):
    print(f"  {q2:.2f}    {m * 1e3:7.2f}    {e:.3f}")
# monotone-ish: error is highest at the smallest margin, ~0 at the largest
err_small = err_vs_margin[0]                     # smallest margin
err_large = err_vs_margin[-1]                    # largest margin
# fraction of all errors that fall in the smallest-margin half
half = len(margins) // 2
small_half_err = err_vs_margin[:half].sum()
large_half_err = err_vs_margin[half:].sum()
conc = small_half_err / max(small_half_err + large_half_err, 1e-9)
print(f"  smallest-margin error = {err_small:.3f}, largest-margin error = "
      f"{err_large:.3f}; {conc * 100:.0f}% of all errors at small-margin half")

# ---------------------------------------------------------------------------
# LAYER B.1 -- dynamical WTA, emitted-only drive: clean single-winner latch.
# ---------------------------------------------------------------------------
print("\n[B.1 dynamical WTA] emitted-only drive => single-winner latch; settling time "
      "scales with surprisal")
wta_single_ok = True
settle_rows = []     # (emit, q, settle_time, winner, n_final, correct&single)
for emit in range(N):
    for q in [0.85, 0.5, 0.25, 0.125]:
        J = np.zeros(N)
        J[emit] = lat.calibration_drive(q)
        act, tr = run_wta_window(J, seed=10 + emit)
        st, w = wta_settling_time(act, tr)
        n_final = int((act[-1] > 0.2).sum())
        ok = (w == emit) and (n_final == 1)
        wta_single_ok &= ok
        settle_rows.append((emit, q, st, w, n_final, ok))
        if emit == 0:
            print(f"  emit={emit} q={q:.3f}: settle = {st * 1e3:5.1f} ms  "
                  f"winner = {w}  final-active = {n_final}  ok = {ok}")
settle_times = np.array([r[2] for r in settle_rows])
settle_qs = np.array([r[1] for r in settle_rows])
finite_settle = settle_times[np.isfinite(settle_times)]
print(f"  all {len(settle_rows)} (emit,q) cases single-winner & correct: "
      f"{wta_single_ok}; settle mean = {finite_settle.mean() * 1e3:.1f} ms, "
      f"max = {finite_settle.max() * 1e3:.1f} ms")
# settling time should grow as q shrinks (surprisal grows): check monotone trend
order = np.argsort(-settle_qs)        # high q -> low q
st_by_q = {}
for q in [0.85, 0.5, 0.25, 0.125]:
    st_by_q[q] = np.mean([r[2] for r in settle_rows if r[1] == q])
settle_monotone = (st_by_q[0.85] < st_by_q[0.5] < st_by_q[0.25] < st_by_q[0.125])
print("  settle(q=0.85,0.5,0.25,0.125) ms = "
      + ", ".join(f"{st_by_q[q] * 1e3:.1f}" for q in [0.85, 0.5, 0.25, 0.125])
      + f"  (monotone in surprisal: {settle_monotone})")

# ---------------------------------------------------------------------------
# LAYER B.2 -- dynamical WTA, FULL race: first-spike-takes-all vs free-run transient.
# ---------------------------------------------------------------------------
print("\n[B.2 dynamical WTA] FULL race (all lines driven): commit-time decode is "
      "correct; free-run co-latches losers")
commit_correct = 0
free_run_single = 0
commit_times = []
b2_rows = []
for ctx in range(N):
    q = P[ctx]
    J = lat.calibration_drive(np.clip(q, Q_LO, Q_HI))
    act, tr = run_wta_window(J, seed=50 + ctx)
    tc, wc = wta_commit_decode(act, tr, commit=0.5)
    st, ws = wta_settling_time(act, tr)
    n_final = int((act[-1] > 0.2).sum())
    expect = int(np.argmax(q))
    if wc == expect:
        commit_correct += 1
    if n_final == 1:
        free_run_single += 1
    commit_times.append(tc)
    b2_rows.append((ctx, expect, wc, tc, n_final))
    print(f"  ctx={ctx} argmax={expect}: commit winner = {wc} @ {tc * 1e3:4.1f} ms "
          f"(correct={wc == expect})  |  free-run final-active = {n_final}")
commit_times = np.array(commit_times)
print(f"  first-spike-takes-all (commit-time) decode correct: "
      f"{commit_correct}/{N};  free-run single-winner: {free_run_single}/{N}")
print("  -> the free-run multi-winner is the integration transient the GLOBAL "
      "'<=1 always' claim mistakes for an invariant (paper fix H);")
print("     first-spike-takes-all reads the leader off at the first crossing, before "
      "slow near-rheobase losers co-latch.")

# ===========================================================================
# Figures
# ===========================================================================

# Fig 1: losslessness -- per-window winner latency, perfect vs uniform, all correct.
drive_p = emitted_drive_node(syms, perfect_pred, window_dur, blank_frac, active_start)
lat2d_p, dec_p, wl_p = run_readout_bank(drive_p, n_sym, window_dur)
drive_u = emitted_drive_node(syms, uniform_pred, window_dur, blank_frac, active_start)
lat2d_u, dec_u, wl_u = run_readout_bank(drive_u, n_sym, window_dur)
fig, ax = plot.new_fig(w=6.6)
xs = np.arange(n_sym)
ax.plot(xs, (wl_p - active_start) * 1e3, "o-", color=plot.C_MEASURED, ms=4,
        lw=1, label="perfect predictor (fast)")
ax.plot(xs, (wl_u - active_start) * 1e3, "s--", color=plot.C_THEORY, ms=4,
        lw=1, alpha=0.8, label="uniform q=1/4 (slow, equal)")
ax.axhline((-LAM * np.log2(0.25)) * 1e3, color="gray", ls=":", lw=1,
           label="uniform t* = 2 bits = 40 ms")
ax.set_xlabel("symbol index in rover stream")
ax.set_ylabel("winner first-spike latency (ms)")
ax.set_title("e05: losslessness — both predictors decode correctly (perfect = fast, "
             "uniform = slow)")
ax.legend(frameon=False, fontsize=8)
plot.save(fig, RESULTS / "e05_losslessness.pdf")

# Fig 2: decode error vs latency-gap margin (errors concentrate at small margin).
fig, ax = plot.new_fig()
ax.plot(margins * 1e3, err_vs_margin, "o-", color=plot.C_MEASURED, ms=7,
        label=f"measured (σ={sigma}, {n_trials} trials)")
ax.axvline(0, color="gray", lw=0.8)
ax.set_xlabel(r"latency-gap margin  $m = \lambda\,(\log_2 q_{top} - \log_2 q_{2nd})$  (ms)")
ax.set_ylabel("decode error rate")
ax.set_title("e05: decode errors concentrate at small latency-gap margin")
ax.legend(frameon=False)
plot.save(fig, RESULTS / "e05_margin.pdf")

# Fig 3: dynamical WTA settling -- one emitted-only window, single winner latches.
J_demo = np.zeros(N)
J_demo[2] = lat.calibration_drive(0.5)
act_demo, tr_demo = run_wta_window(J_demo, seed=12)
st_demo, w_demo = wta_settling_time(act_demo, tr_demo)
fig, ax = plot.new_fig()
labels = ["U", "D", "L", "R"]
for j in range(N):
    ax.plot(tr_demo * 1e3, act_demo[:, j],
            lw=2 if j == 2 else 1.2,
            color=plot.C_MEASURED if j == 2 else "gray",
            alpha=1.0 if j == 2 else 0.6,
            label=f"{labels[j]}" + (" (emitted)" if j == 2 else ""))
ax.axvline(st_demo * 1e3, color=plot.C_THEORY, ls="--", lw=1.2,
           label=f"settle = {st_demo * 1e3:.0f} ms")
ax.axhline(0.2, color="gray", ls=":", lw=0.8)
ax.set_xlabel("time within window (ms)")
ax.set_ylabel("WTA channel activity")
ax.set_title("e05: dynamical WTA latches a single winner (emitted L, q=0.5)")
ax.legend(frameon=False, fontsize=8, ncol=2)
plot.save(fig, RESULTS / "e05_wta_settle.pdf")

# Fig 4: settling time scales with surprisal (emitted-only).
fig, ax = plot.new_fig()
qs_plot = np.array([0.85, 0.5, 0.25, 0.125])
st_plot = np.array([st_by_q[q] for q in qs_plot]) * 1e3
ax.plot(-np.log2(qs_plot), st_plot, "o-", color=plot.C_MEASURED, ms=7,
        label="measured settle time")
ax.set_xlabel(r"surprisal  $-\log_2 q$  (bits)")
ax.set_ylabel("WTA settling time (ms)")
ax.set_title("e05: WTA settling time tracks the latency code (surprisal)")
ax.legend(frameon=False)
plot.save(fig, RESULTS / "e05_settle_vs_surprisal.pdf")

# Fig 5: the full-race free-run multi-winner transient (honest).
q_ctx = P[0]
J_full = lat.calibration_drive(np.clip(q_ctx, Q_LO, Q_HI))
act_full, tr_full = run_wta_window(J_full, seed=50)
tc_full, wc_full = wta_commit_decode(act_full, tr_full, commit=0.5)
fig, ax = plot.new_fig()
for j in range(N):
    ax.plot(tr_full * 1e3, act_full[:, j],
            lw=2 if j == 0 else 1.2,
            color=plot.C_MEASURED if j == 0 else "gray",
            alpha=1.0 if j == 0 else 0.6,
            label=f"{labels[j]}" + (" (leader)" if j == 0 else ""))
ax.axvline(tc_full * 1e3, color=plot.C_FLOOR, ls="--", lw=1.4,
           label=f"commit @ {tc_full * 1e3:.0f} ms → decode={labels[wc_full]}")
ax.axhline(0.2, color="gray", ls=":", lw=0.8)
ax.set_xlabel("time within window (ms)")
ax.set_ylabel("WTA channel activity")
ax.set_title("e05: full race — leader committed first; slow losers co-latch later "
             "(F G transient)")
ax.legend(frameon=False, fontsize=8, ncol=2)
plot.save(fig, RESULTS / "e05_full_race.pdf")

# ===========================================================================
# Save + acceptance
# ===========================================================================
np.savez(RESULTS / "e05_results.npz",
         syms=syms, window_dur=window_dur, active_start=active_start,
         lossless_perfect=lossless["perfect"], lossless_uniform=lossless["uniform"],
         mean_lat_perfect=mean_lat["perfect"], mean_lat_uniform=mean_lat["uniform"],
         q_2nds=q_2nds, margins=margins, err_vs_margin=err_vs_margin, sigma=sigma,
         settle_times=settle_times, settle_qs=settle_qs,
         st_by_q=np.array([st_by_q[q] for q in [0.85, 0.5, 0.25, 0.125]]),
         commit_correct=commit_correct, free_run_single=free_run_single,
         commit_times=commit_times)

checks = {
    "noise-free perfect-predictor decode error == 0 (losslessness)":
        lossless["perfect"] == 0.0,
    "noise-free uniform-predictor decode error == 0 (slow, never wrong)":
        lossless["uniform"] == 0.0,
    "uniform predictor is slower than perfect (higher mean latency)":
        mean_lat["uniform"] > mean_lat["perfect"],
    "decode error rises at small margin (smallest > largest by > 0.1)":
        err_small - err_large > 0.1,
    "decode errors concentrate in the small-margin half (> 80%)":
        conc > 0.80,
    "dynamical WTA latches a single correct winner (emitted-only, all cases)":
        wta_single_ok,
    "WTA settling time grows with surprisal (monotone in q)":
        settle_monotone,
    "first-spike-takes-all (commit-time) decode correct in full race (4/4)":
        commit_correct == N,
}
print("\n" + "=" * 72)
allok = True
for name, ok in checks.items():
    allok &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
print(f"\ne05: {sum(checks.values())}/{len(checks)} acceptance checks passed.")
sys.exit(0 if allok else 1)
