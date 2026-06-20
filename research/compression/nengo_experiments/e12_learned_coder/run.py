#!/usr/bin/env python3
"""e12 -- The LEARNED end-to-end spiking entropy coder (THE MONEY PLOT).

This is the capstone that composes everything. e11 ran the full coder with a
*frozen perfect* predictor (q = P, the true rover row). Here the predictor is no
longer frozen: it is the e09 PES learner, starting from the *uniform* predictor
q = 1/4 (so the coder's bill starts at log2 4 = 2.0 bits/symbol) and learning the
rover law online from the realized symbol stream. The LEARNED q is then the drive
source for the calibrated readout bank -- so as the circuit learns, its own
first-spike spike-time bill falls. Compression, prediction, and learning are ONE
descent on ONE number, read off the real spike clock.

The closed loop. The learned predictor IS the drive source for the latency clock:
a transient bad prediction pushes a readout's drive high (q small -> drive
theta/(1-q^a) grows) or, at q -> 1, toward infinity. We clamp the learned q to
(Q_CLIP_LO, q_max] before it reaches the calibration drive so the bank stays
finite and never silent -- the clamp is what keeps the closed loop stable through
the noisy early-learning transient.

Pipeline at each training checkpoint:
  Stage 1 (context): the running one-hot of the PREVIOUS symbol c_t = x_{t-1}.
  Stage 1' (LEARNED predictor): a 600-neuron spiking context ensemble whose
            decoders are PES-learned (e09 idiom), reading off q(.|c_t). At t=0 the
            decoder is the uniform predictor 1/4.
  Stage 2 (calibrated readout race): build_readout_bank driven by the LEARNED,
            clamped q(.|c_t); readout j first-spikes at -lambda log2 q_j.
  Stage 3 (temporal WTA decode): first-spike-takes-all names the symbol.

We snapshot the learner at a schedule of training times. At each checkpoint we read
the learned conditional law Q_hat(.|c) (settled decode per context) and measure:

  (A) MEAN BITS/SYMBOL FROM REAL READOUT LATENCIES (the headline measurement).
      For each context row we drive a FRESH calibrated readout bank with the
      checkpoint's learned, clamped Q_hat[c] in real Nengo spikes (DT_FINE) and
      record each readout's first-spike latency. The per-symbol cost of emitting
      x_t in context c_t is latency(Q_hat[c_t])[x_t] / lambda bits; we average
      over the realized stream's (context, outcome) pairs. This is the actual
      spike-time bill the coder pays at that point in learning.
  (B) MEAN BITS/SYMBOL FROM -log2 Q_hat (numpy cross-check / the model's own
      surprisal), which the real-latency measurement should track to the dt tax.
  (C) DECODE ACCURACY: first-spike-takes-all over the learned latency table on the
      realized stream (winner == emitted), rising as the peaks sharpen.

THE MONEY PLOT: mean bits/symbol descending from ~2.0 (uniform init = log2 4) past
the marginal 1.7500, toward the noise-ball floor (~1.0-1.05, the e09 constant-lr
ball plus the e11 dt overhead), with decode accuracy rising in lockstep.

Efficiency, honestly. The readout-latency table for a learned row depends only on
that row Q_hat[c]; there are only 4 distinct rows per checkpoint, so we drive 4
banks per checkpoint and assemble the n=4000 stream mean by (context, outcome)
lookup -- the same exact-to-dt assembly e11 justified (independent readouts, latency
law exact to dt). We ALSO run the GENUINE continuous learned pipeline once at the
final checkpoint (windowed reset, emitted-only drive at the learned q, first-spike
decode) to prove the assembled coder is a real, lossless spiking circuit.
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

from spikecoder import config as cfg                              # noqa: E402
from spikecoder import information as info                         # noqa: E402
from spikecoder import latency as lat                             # noqa: E402
from spikecoder import metrics as met                             # noqa: E402
from spikecoder import plotting as plot                           # noqa: E402
from spikecoder.source import RoverSource, LABELS                 # noqa: E402
from spikecoder.information import (model_energy_and_kl,           # noqa: E402
                                    H_RATE, H_MARGINAL, LEARNED_REF)
from spikecoder.networks import build_readout_bank, make_lif      # noqa: E402

# --- constants (the verified e09 PES idiom + e11 readout band) ---
N = 4
N_SYMBOLS = 10000          # training-stream length (>= 10000)
WINDOW = 0.03              # s per symbol window (e09)
LR = 2e-4                  # constant PES learning rate (e09)
N_CTX = 600               # context-ensemble size (e09)
CTX_RADIUS = 1.3          # (e09)
DT = cfg.DT               # 1e-3 for learning/throughput
DT_FINE = cfg.DT_FINE     # 1e-4 for latency timing
LAM = cfg.LAMBDA
SEED = cfg.SEED
Q_LO = cfg.Q_CLIP_LO
Q_MAX = float(lat.q_max_for_ceiling())   # rheobase-ceiling cap (keeps drive finite)
N_STREAM = 4000            # stream over which each checkpoint's mean is assembled


# ===========================================================================
# Stage 1' : the PES learner (the e09 idiom, uniform init). We probe the decoded
# q on a per-window basis so we can reconstruct the learned law Q_hat at any
# training-time checkpoint (the cumulative settled decode up to that window).
# ===========================================================================
def build_pes_net(ctx_fn, y_fn, seed=SEED):
    """One spiking context population whose decoders are PES-learned, uniform init."""
    net = nengo.Network(seed=seed)
    with net:
        ci = nengo.Node(ctx_fn, size_out=N)
        ctx = nengo.Ensemble(N_CTX, N, radius=CTX_RADIUS)
        nengo.Connection(ci, ctx, synapse=0.005)

        q = nengo.Node(size_in=N)                       # decoded prediction q(.|c)
        conn = nengo.Connection(
            ctx, q, function=lambda c: np.ones(N) * 0.25,    # uniform init -> 2.0 bits
            learning_rule_type=nengo.PES(learning_rate=LR), synapse=0.01)

        yi = nengo.Node(y_fn, size_out=N)               # teacher: one-hot realized x_t
        err = nengo.Node(size_in=N)                     # PES error = q - y
        nengo.Connection(q, err, synapse=None)
        nengo.Connection(yi, err, transform=-1, synapse=None)
        nengo.Connection(err, conn.learning_rule, synapse=None)

        pq = nengo.Probe(q, synapse=0.02)
        pctx = nengo.Probe(ci)
    return net, pq, pctx


def per_window_q_and_ctx(qd, cd):
    """Reduce raw probes to one settled-tail q and one context index per window."""
    win_q = np.zeros((N_SYMBOLS, N))
    win_ctx = np.full(N_SYMBOLS, -1)
    for k in range(N_SYMBOLS):
        i0 = int(round(k * WINDOW / DT))
        i1 = int(round((k + 1) * WINDOW / DT))
        if i1 > len(qd):
            break
        seg = qd[i0:i1]
        win_q[k] = seg[len(seg) // 2:].mean(0)          # settled tail (last 50%)
        csum = cd[i0:i1].sum(0)
        if csum.max() > 0:
            win_ctx[k] = int(np.argmax(csum))
    return win_q, win_ctx


def learned_matrix(win_q, win_ctx, lo=0, hi=None):
    """Average decoded q per context over windows [lo, hi); clip>0 & renormalize rows.

    This is the learned conditional law Q_hat available to the coder using the
    windows seen so far -- i.e. the predictor at training-time `hi`.
    """
    hi = N_SYMBOLS if hi is None else hi
    rows = {i: [] for i in range(N)}
    for k in range(lo, hi):
        if win_ctx[k] >= 0:
            rows[win_ctx[k]].append(win_q[k])
    Q = np.ones((N, N)) / N
    for i in range(N):
        if rows[i]:
            v = np.mean(rows[i], axis=0)
            v = np.clip(v, 1e-6, None)
            Q[i] = v / v.sum()
    return Q


# ===========================================================================
# Stage 2 driver: drive a fresh calibrated readout bank with one LEARNED row
# (clamped to the representable band) and record real-spike first-spike latencies.
# This is the actual spike-time bill of that row at that checkpoint.
# ===========================================================================
def bank_first_spikes(q_vec, dt=DT_FINE, tmax=2.5, seed=SEED):
    """Drive a fresh N-readout calibrated bank from rest with constant probabilities
    q_vec (clamped to (Q_LO, Q_MAX]); return per-readout first-spike latencies (N,)
    in seconds (inf if no spike within tmax)."""
    q_vec = np.clip(np.asarray(q_vec, float), Q_LO, Q_MAX)
    with nengo.Network(seed=seed) as net:
        q_node = nengo.Node(q_vec)
        ens, _drive = build_readout_bank(net, q_node, N=N)
        p = nengo.Probe(ens.neurons)
    with nengo.Simulator(net, dt=dt, progress_bar=False) as sim:
        sim.run(tmax)
    spikes = sim.data[p]
    tr = sim.trange()
    out = np.full(N, np.inf)
    for j in range(N):
        idx = np.flatnonzero(spikes[:, j] > 0)
        if len(idx):
            out[j] = tr[idx[0]]
    return out


def latency_table_for_Q(Q_hat, dt=DT_FINE):
    """Real-spike per-readout latency table (N,N): row c driven by learned Q_hat[c]."""
    return np.array([bank_first_spikes(Q_hat[c], dt=dt) for c in range(N)])


def argmin_tiebreak_random(rows, rng, tol=1e-9):
    """First-spike-takes-all winner per row, breaking near-ties UNIFORMLY at random.

    A uniform predictor produces four (near-)equal latencies; a fair temporal-WTA
    decoder has no basis to prefer any readout, so it picks at random among the tied
    winners. This makes the uniform-init decode honestly sit at CHANCE (not an
    accidental argmin-index-0 bias). As learning sharpens one readout ahead of the
    rest the tie breaks and the winner becomes deterministic.
    """
    rows = np.asarray(rows, float)
    out = np.empty(len(rows), int)
    for k, r in enumerate(rows):
        finite = np.isfinite(r)
        if not finite.any():
            out[k] = -1
            continue
        m = np.nanmin(np.where(finite, r, np.inf))
        tied = np.flatnonzero(finite & (r <= m + tol))
        out[k] = int(rng.choice(tied))
    return out


def stream_bits_and_accuracy(lat_table, xs, rng=None):
    """From a real-spike latency table and the realized stream xs (t>=1):
      bits_t   = lat_table[c_t, x_t] / lambda   (the emitted symbol's readout latency)
      decoded_t= first-spike winner of lat_table[c_t] (random tie-break)
    Returns (mean bits/symbol over finite costs, decode accuracy = top-1 hit rate).
    The accuracy here is the model's top-1 NEXT-SYMBOL prediction hit rate, ceilinged
    by the Bayes-optimal rate sum_c pi_c max_j P[c,j] (a stochastic source cannot be
    predicted perfectly); it rises from chance as the learned peaks sharpen."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    cs = xs[:-1]
    os = xs[1:]
    bits = lat_table[cs, os] / LAM
    fin = np.isfinite(bits)
    mean_bits = float(bits[fin].mean()) if fin.any() else np.inf
    winners = argmin_tiebreak_random(lat_table[cs], rng)
    accuracy = float(np.mean(winners == os))
    return mean_bits, accuracy


# ===========================================================================
# The GENUINE continuous learned pipeline (final checkpoint): emitted-only drive at
# the learned q, per-window blank reset, first-spike-takes-all decode. The real coder.
# ===========================================================================
def run_end_to_end(symbols, predictor_row, window_dur, blank_frac=0.30,
                   dt=DT_FINE, seed=SEED):
    n_sym = len(symbols)
    active_start = blank_frac * window_dur
    symbols = np.asarray(symbols, int)

    def drive_fn(t):
        k = min(int(t // window_dur), n_sym - 1)
        J = np.zeros(N)
        if (t % window_dur) >= active_start:
            q = predictor_row(k)
            x = symbols[k]
            J[x] = lat.calibration_drive(np.clip(q[x], Q_LO, Q_MAX))
        return J

    with nengo.Network(seed=seed) as net:
        Jnode = nengo.Node(drive_fn, size_out=N)
        ens = nengo.Ensemble(N, 1, neuron_type=make_lif(), gain=np.ones(N),
                             bias=np.zeros(N), encoders=np.ones((N, 1)))
        nengo.Connection(Jnode, ens.neurons, synapse=None)
        p = nengo.Probe(ens.neurons)
    with nengo.Simulator(net, dt=dt, progress_bar=False) as sim:
        sim.run(n_sym * window_dur)
    lat2d = met.per_window_first_spikes(sim.data[p], sim.trange(), window_dur,
                                        n_sym, dt=dt)
    decoded, win_lat = met.decode_first_spike(lat2d)
    return decoded, win_lat - active_start, active_start


# ===========================================================================
print("=" * 76)
print("e12 -- The LEARNED end-to-end spiking entropy coder (THE MONEY PLOT)")
print("=" * 76)
print(f"lambda = {LAM} s/bit, N = {N} readouts, floor H_rate = {H_RATE:.4f} bits/symbol")
print(f"marginal H(pi) = {H_MARGINAL:.4f}, uniform-init energy = log2(4) = {np.log2(4):.4f}")
print(f"q clamp band for the drive: ({Q_LO:.0e}, q_max = {Q_MAX:.4f}] (keeps drive finite)")

# the rover stream that BOTH trains the learner and is coded
rs = RoverSource(s=0.7, seed=SEED)
P, pi = rs.P, rs.pi
x_train = rs.sample(N_SYMBOLS)
ctx_fn = rs.context_input(x_train, WINDOW)
y_fn = rs.outcome_input(x_train, WINDOW)
x_stream = x_train[:N_STREAM]              # the stream each checkpoint is scored on
print(f"\nsource: momentum rover s=0.7, training {N_SYMBOLS} symbols (window {WINDOW}s, "
      f"PES lr={LR}, {N_CTX} ctx neurons), scoring stream n={N_STREAM}")

# ---------------------------------------------------------------------------
# Run the PES learner ONCE over the whole stream, recording the per-window
# decoded q so we can reconstruct the learned predictor at every checkpoint.
# ---------------------------------------------------------------------------
print("\n[learning] running the spiking PES learner from the uniform init...")
net, pq, pctx = build_pes_net(ctx_fn, y_fn)
with nengo.Simulator(net, dt=DT, progress_bar=False) as sim:
    sim.run(N_SYMBOLS * WINDOW)
win_q, win_ctx = per_window_q_and_ctx(sim.data[pq], sim.data[pctx])

# final converged law (last 40%) -- the predictor the coder settles to
CONVERGED_FRAC = 0.6
Q_final = learned_matrix(win_q, win_ctx, lo=int(CONVERGED_FRAC * N_SYMBOLS))
max_err_final = float(np.abs(Q_final - P).max())
E_final_model, kl_final_model = model_energy_and_kl(Q_final, pi, P, H_RATE, is_logits=False)
print(f"  learned law converged: max|Q_hat - P| = {max_err_final:.4f}, "
      f"model energy = {E_final_model:.4f} bits/symbol (excess KL {kl_final_model:.4f})")

# prediction baselines for the decode accuracy: a STOCHASTIC source cannot be
# predicted perfectly. CHANCE = best context-free guess (= max marginal pi); the
# BAYES-OPTIMAL top-1 ceiling = sum_c pi_c max_j P[c,j] (always guess the most
# probable next symbol under the TRUE law). The learned coder's decode accuracy
# rises from chance toward this ceiling and provably cannot exceed it.
CHANCE_ACC = float(pi.max())                                   # 0.50
BAYES_ACC = float(sum(pi[c] * P[c].max() for c in range(N)))   # 0.8031
print(f"\nprediction baselines: chance = {CHANCE_ACC:.4f} (max marginal), "
      f"Bayes-optimal top-1 ceiling = {BAYES_ACC:.4f} (sum_c pi_c max_j P[c,j])")

# ---------------------------------------------------------------------------
# Checkpoint schedule (training symbols seen). At each checkpoint we use the
# learned law Q_hat reconstructed from the windows seen so far, drive 4 real
# readout banks, and measure (A) real-latency bits, (B) -log2 Q_hat bits, (C) acc.
# ---------------------------------------------------------------------------
checkpoints = [1, 25, 100, 300, 800, 2000, 4000, 7000, N_SYMBOLS]
print(f"\n[checkpoints] {len(checkpoints)} training-time checkpoints: {checkpoints}")
print("-" * 76)
print(f"{'ckpt':>7} {'max|Qh-P|':>10} {'bits(real)':>11} {'bits(-log2Qh)':>14} "
      f"{'acc':>7} {'model E':>9}")

ckpt_bits_real = []        # (A) mean bits/symbol from real readout latencies
ckpt_bits_model = []       # (B) mean bits/symbol from -log2 Q_hat (numpy)
ckpt_acc = []              # (C) decode accuracy
ckpt_maxerr = []
ckpt_model_E = []
ckpt_lat_tables = []

acc_rng = np.random.default_rng(SEED)
for m in checkpoints:
    Q_hat = learned_matrix(win_q, win_ctx, lo=0, hi=m)         # predictor at time m
    # (A) drive 4 real readout banks with the learned, clamped rows
    lat_table = latency_table_for_Q(Q_hat, dt=DT_FINE)
    bits_real, acc = stream_bits_and_accuracy(lat_table, x_stream, rng=acc_rng)
    # (B) the model's own surprisal on the same stream (clamped, as the drive sees it)
    Qc = np.clip(Q_hat, Q_LO, Q_MAX)
    Qc = Qc / Qc.sum(axis=1, keepdims=True)
    cs, os = x_stream[:-1], x_stream[1:]
    bits_model = float(np.mean(-np.log2(Qc[cs, os])))
    # model cross-entropy-rate energy (closed form) for reference
    E_m, _ = model_energy_and_kl(Q_hat, pi, P, H_RATE, is_logits=False)
    me = float(np.abs(Q_hat - P).max())

    ckpt_bits_real.append(bits_real)
    ckpt_bits_model.append(bits_model)
    ckpt_acc.append(acc)
    ckpt_maxerr.append(me)
    ckpt_model_E.append(E_m)
    ckpt_lat_tables.append(lat_table)
    print(f"{m:>7} {me:>10.4f} {bits_real:>11.4f} {bits_model:>14.4f} "
          f"{acc:>7.3f} {E_m:>9.4f}")

ckpt_bits_real = np.array(ckpt_bits_real)
ckpt_bits_model = np.array(ckpt_bits_model)
ckpt_acc = np.array(ckpt_acc)
ckpt_maxerr = np.array(ckpt_maxerr)
ckpt_model_E = np.array(ckpt_model_E)
checkpoints_arr = np.array(checkpoints)

bits_init = ckpt_bits_real[0]
bits_final = ckpt_bits_real[-1]
acc_init = ckpt_acc[0]
acc_final = ckpt_acc[-1]
# normalized decode efficiency: 0 at chance, 1 at the Bayes-optimal ceiling
ckpt_norm_eff = (ckpt_acc - CHANCE_ACC) / (BAYES_ACC - CHANCE_ACC)
eff_init, eff_final = float(ckpt_norm_eff[0]), float(ckpt_norm_eff[-1])

# the spiking overhead of the final learned coder above its own model energy = the dt tax
dt_overhead_final = bits_final - ckpt_bits_model[-1]

print("-" * 76)
print(f"[descent] real-latency bits/symbol: init {bits_init:.4f} -> final {bits_final:.4f}")
print(f"          crosses below marginal {H_MARGINAL:.4f}: "
      f"{'YES at ckpt ' + str(int(checkpoints_arr[np.argmax(ckpt_bits_real < H_MARGINAL)])) if (ckpt_bits_real < H_MARGINAL).any() else 'NO'}")
print(f"[accuracy] decode top-1 accuracy: init {acc_init:.3f} -> final {acc_final:.3f} "
      f"(chance {CHANCE_ACC:.3f}, Bayes ceiling {BAYES_ACC:.3f})")
print(f"          normalized decode efficiency: init {eff_init:.3f} -> final {eff_final:.3f} "
      f"(1.0 = Bayes-optimal; the stochastic source forbids more)")
print(f"[overhead] final real-latency bits {bits_final:.4f} = model surprisal "
      f"{ckpt_bits_model[-1]:.4f} + dt tax {dt_overhead_final*1e3:+.2f} mbit")

# monotone-descent envelope (cumulative min) of the real-latency bits
bits_real_env = np.minimum.accumulate(ckpt_bits_real)
descent_monotone_env = bool(np.all(np.diff(bits_real_env) <= 1e-9))

# correlation: bits down as accuracy up (one descent on one number)
if np.std(ckpt_bits_real) > 0 and np.std(ckpt_acc) > 0:
    corr_bits_acc = float(np.corrcoef(ckpt_bits_real, ckpt_acc)[0, 1])
else:
    corr_bits_acc = np.nan

# ---------------------------------------------------------------------------
# The GENUINE continuous learned pipeline at the FINAL checkpoint: prove the
# learned coder is a real, lossless spiking circuit (not just a lookup table).
# ---------------------------------------------------------------------------
print("\n" + "-" * 76)
print("[genuine pipeline] continuous learned coder at the final checkpoint")
print("-" * 76)
n_e2e = 400
window_dur_e2e = 0.16          # long enough for the slowest learned row to first-spike
syms_e2e = x_stream[:n_e2e]


def learned_row(k):
    """Final-checkpoint learned predictor row for window k (context = prev symbol)."""
    if k == 0:
        return pi.copy()
    return Q_final[syms_e2e[k - 1]].copy()


dec_L, wl_L, a_start = run_end_to_end(syms_e2e, learned_row, window_dur_e2e, dt=DT_FINE)
err_e2e = met.decode_error_rate(dec_L, syms_e2e)
nospk_e2e = int((dec_L == -1).sum())
e2e_mean_bits = float(np.mean(wl_L[np.isfinite(wl_L)]) / LAM)
# the lookup-assembled mean on the same 400 windows with the final latency table
lookup_mean_e2e, lookup_acc_e2e = stream_bits_and_accuracy(
    ckpt_lat_tables[-1], syms_e2e, rng=np.random.default_rng(SEED))
print(f"  continuous pipeline over {n_e2e} windows ({window_dur_e2e*1e3:.0f} ms, "
      f"blank {a_start*1e3:.0f} ms, dt=DT_FINE)")
print(f"  decode error = {err_e2e:.4f} (no-spike windows {nospk_e2e}), "
      f"mean bits/symbol = {e2e_mean_bits:.4f}")
print(f"  lookup-assembled mean on same windows = {lookup_mean_e2e:.4f} "
      f"(|diff| = {abs(e2e_mean_bits-lookup_mean_e2e)*1e3:.2f} mbit -> coder == lookup)")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt  # noqa: E402

# === THE MONEY PLOT: bits/symbol descending + accuracy rising, one twin-axis fig ===
fig, axL = plt.subplots(figsize=(7.6, 4.6))
xs = np.maximum(checkpoints_arr, 1)
axL.plot(xs, ckpt_bits_real, "o-", color=plot.C_MEASURED, lw=2.2, ms=6,
         label="mean bits/symbol (real readout latencies)", zorder=5)
axL.plot(xs, ckpt_bits_model, "s--", color="#6a9fd8", lw=1.3, ms=4, alpha=0.9,
         label=r"mean $-\log_2 \hat{q}$ (model surprisal)")
axL.axhline(np.log2(4), color="black", ls=":", lw=1.0, alpha=0.6)
axL.axhline(H_MARGINAL, color="gray", ls=":", lw=1.4, label=f"marginal {H_MARGINAL:.4f}")
axL.axhline(H_RATE, color=plot.C_FLOOR, ls="--", lw=1.8,
            label=f"entropy-rate floor {H_RATE:.4f}")
axL.annotate("uniform init\n$\\log_2 4 = 2.0$", (xs[0], ckpt_bits_real[0]),
             fontsize=8, xytext=(xs[0] * 1.4, np.log2(4) + 0.02), color=plot.C_MEASURED)
axL.set_xscale("log")
axL.set_xlabel("training step (rover symbols seen, log scale)")
axL.set_ylabel("mean bits/symbol", color=plot.C_MEASURED)
axL.tick_params(axis="y", labelcolor=plot.C_MEASURED)
axL.set_ylim(0.85, 2.1)

axR = axL.twinx()
axR.plot(xs, ckpt_acc * 100, "^-", color=plot.C_THEORY, lw=1.8, ms=6,
         label="decode top-1 accuracy", zorder=4)
axR.axhline(BAYES_ACC * 100, color=plot.C_THEORY, ls="--", lw=1.2, alpha=0.7,
            label=f"Bayes-optimal ceiling {BAYES_ACC*100:.1f}%")
axR.axhline(CHANCE_ACC * 100, color="gray", ls="-.", lw=1.0, alpha=0.6,
            label=f"chance {CHANCE_ACC*100:.0f}%")
axR.set_ylabel("decode top-1 accuracy (%)", color=plot.C_THEORY)
axR.tick_params(axis="y", labelcolor=plot.C_THEORY)
axR.set_ylim(20, 102)
axR.grid(False)

lL, llabL = axL.get_legend_handles_labels()
lR, llabR = axR.get_legend_handles_labels()
axL.legend(lL + lR, llabL + llabR, frameon=False, fontsize=8, loc="center right")
axL.set_title("e12: the learned coder -- bits/symbol descend, accuracy rises, ONE descent")
fig.tight_layout()
fig.savefig(RESULTS / "e12_money_plot.pdf", bbox_inches="tight")
plt.close(fig)

# === Fig 2: learned-vs-true conditional law (final), grouped bars per context ===
fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0))
xb = np.arange(N)
for i, ax in enumerate(axes.flat):
    ax.bar(xb - 0.2, P[i], 0.4, color=plot.C_THEORY, alpha=0.8, label="true $P[i,:]$")
    ax.bar(xb + 0.2, Q_final[i], 0.4, color=plot.C_MEASURED,
           label=r"learned $\hat{q}(\cdot|i)$")
    ax.set_xticks(xb); ax.set_xticklabels(LABELS)
    ax.set_title(f"context = {LABELS[i]}", fontsize=10)
    ax.set_ylim(0, 1.0)
    if i == 0:
        ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
fig.suptitle(f"e12: the coder's LEARNED predictor at convergence "
             f"(max$|\\hat{{q}}-P|$ = {max_err_final:.3f})", fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS / "e12_learned_law.pdf", bbox_inches="tight")
plt.close(fig)

# === Fig 3: the closed loop -- per-checkpoint drive headroom + accuracy ladder ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8))
# left: the spiking overhead (real bits - model surprisal) at each checkpoint -- the
# dt tax, plus any clamp distortion in the noisy early transient
overhead = ckpt_bits_real - ckpt_bits_model
ax1.plot(xs, overhead * 1e3, "o-", color=plot.C_MEASURED, ms=5, lw=1.6)
ax1.axhline(0, color="gray", ls=":", lw=1.0)
ax1.set_xscale("log")
ax1.set_xlabel("training step (symbols seen)")
ax1.set_ylabel("real bits $-$ model surprisal (mbit/symbol)")
ax1.set_title("(a) spiking overhead over the closed loop")
ax1.grid(True, alpha=0.3)
# right: accuracy vs max|Qhat - P| -- learning sharpens peaks, decode follows
ax2.plot(ckpt_maxerr, ckpt_acc * 100, "o-", color=plot.C_THEORY, ms=6, lw=1.6)
ax2.axhline(BAYES_ACC * 100, color=plot.C_THEORY, ls="--", lw=1.1, alpha=0.7,
            label=f"Bayes ceiling {BAYES_ACC*100:.1f}%")
ax2.axhline(CHANCE_ACC * 100, color="gray", ls="-.", lw=1.0, alpha=0.6,
            label=f"chance {CHANCE_ACC*100:.0f}%")
for i, m in enumerate(checkpoints):
    ax2.annotate(str(m), (ckpt_maxerr[i], ckpt_acc[i] * 100 + 1.2),
                 fontsize=6.5, ha="center", color="gray")
ax2.set_xlabel(r"max$|\hat{q} - P|$ (learning error)")
ax2.set_ylabel("decode top-1 accuracy (%)")
ax2.set_title("(b) sharper law $->$ Bayes-optimal decode")
ax2.legend(frameon=False, fontsize=7.5, loc="lower left")
ax2.grid(True, alpha=0.3)
fig.suptitle("e12: the closed loop is stable (clamp keeps drive finite) and decode "
             "tracks learning", fontsize=10.5)
fig.tight_layout()
fig.savefig(RESULTS / "e12_closed_loop.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Save + acceptance
# ---------------------------------------------------------------------------
np.savez(RESULTS / "e12_results.npz",
         P=P, pi=pi, Q_final=Q_final, max_err_final=max_err_final,
         E_final_model=E_final_model, kl_final_model=kl_final_model,
         checkpoints=checkpoints_arr, ckpt_bits_real=ckpt_bits_real,
         ckpt_bits_model=ckpt_bits_model, ckpt_acc=ckpt_acc,
         ckpt_maxerr=ckpt_maxerr, ckpt_model_E=ckpt_model_E,
         bits_real_env=bits_real_env, corr_bits_acc=corr_bits_acc,
         bits_init=bits_init, bits_final=bits_final,
         acc_init=acc_init, acc_final=acc_final,
         ckpt_norm_eff=ckpt_norm_eff, eff_init=eff_init, eff_final=eff_final,
         CHANCE_ACC=CHANCE_ACC, BAYES_ACC=BAYES_ACC,
         dt_overhead_final=dt_overhead_final,
         err_e2e=err_e2e, e2e_mean_bits=e2e_mean_bits,
         lookup_mean_e2e=lookup_mean_e2e, n_e2e=n_e2e, window_dur_e2e=window_dur_e2e,
         wl_L=wl_L, dec_L=dec_L, syms_e2e=syms_e2e,
         H_RATE=H_RATE, H_MARGINAL=H_MARGINAL, LEARNED_REF=LEARNED_REF,
         Q_MAX=Q_MAX, N_SYMBOLS=N_SYMBOLS, N_STREAM=N_STREAM)

below_marg = (ckpt_bits_real < H_MARGINAL).any()
checks = {
    "real-latency bits start near uniform init 2.0 (>= 1.9)": bits_init >= 1.9,
    "real-latency bits descend below the marginal 1.7500": below_marg,
    "real-latency bits reach the noise ball (1.0 <= final <= 1.10)": (
        1.0 <= bits_final <= 1.10),
    "smoothed (cummin) bits descent is monotone": descent_monotone_env,
    "decode top-1 accuracy improves over training (final - init >= 0.10)": (
        acc_final - acc_init >= 0.10),
    "final decode accuracy reaches the Bayes-optimal ceiling (|acc-Bayes| < 0.02)": (
        abs(acc_final - BAYES_ACC) < 0.02),
    "normalized decode efficiency reaches Bayes-optimal (final >= 0.98)": (
        eff_final >= 0.98),
    "bits and accuracy anti-correlate (one descent: corr <= -0.7)": (
        corr_bits_acc <= -0.7),
    "real-latency bits track the model surprisal (final |gap| < 0.05)": (
        abs(dt_overhead_final) < 0.05),
    "learned predictor converged to P (max|Q_hat-P| < 0.2)": max_err_final < 0.2,
    "genuine continuous learned pipeline is lossless (decode error == 0)": (
        err_e2e == 0.0),
    "continuous pipeline matches the lookup (|diff| < 0.02)": (
        abs(e2e_mean_bits - lookup_mean_e2e) < 0.02),
}
print("\n" + "=" * 76)
allok = True
for name, ok in checks.items():
    allok &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
print(f"\ne12: {sum(checks.values())}/{len(checks)} acceptance checks passed.")
print(f"\nHEADLINE: the learned coder's spike-time bill descends "
      f"{bits_init:.3f} -> {bits_final:.3f} bits/symbol as decode top-1 accuracy rises "
      f"{acc_init*100:.0f}% -> {acc_final*100:.0f}% (Bayes-optimal {BAYES_ACC*100:.1f}%) "
      f"-- one descent on one number, in real spikes.")
sys.exit(0 if allok else 1)
