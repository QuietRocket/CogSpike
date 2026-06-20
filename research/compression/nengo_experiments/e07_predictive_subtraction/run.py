#!/usr/bin/env python3
"""e07 -- Predictive subtraction: the one operation that needs a cycle.

Claim (paper #5 + FIX F, the sign): the recurrent loop cancels the EXPECTED drive
via INHIBITORY feedback -- NOT an additive depolarization "equal to the
prediction" -- so the soma integrates only the UNPREDICTED residual

    I_eff(t) = I_in(t) - I_pred(t)                                   (eq-predsub)

with the prediction entering with a MINUS sign (feedback inhibition / shunting).
A perfectly predicted symbol meets balanced excitation + inhibition: the soma sees
~0 net drive and is near-silent. A surprising symbol produces a large residual and
fires. The emitted spike train stops being a copy of the symbol stream and becomes
a SURPRISE stream.

This is the one experiment that is *impossible* to see in pure numpy: the sign of
the feedback is invisible in the algebra `input - prediction` (numpy just computes
the number); only a real spiking soma, whose voltage is clamped at V >= 0 and which
fires when V crosses threshold, distinguishes "subtract the prediction" (the soma
quiets) from "add the prediction" (the soma is driven to runaway early firing).

Three tests:
  (1) SIGN CONTROL (the FIX F demonstration). Run the loop with INHIBITORY feedback
      (correct, J_eff = J_in - J_pred) and with ADDITIVE/EXCITATORY feedback (wrong,
      J_eff = J_in + J_pred). Show the inhibitory sign makes a WELL-PREDICTED symbol
      near-silent (residual ~ 0, balanced E/I) while the wrong (additive) sign
      DOUBLES the drive and fires early -- runaway, no cancellation.
  (2) SURPRISE STREAM. Over a rover stream with a known predictor, show the residual
      output (spike count / 1-latency) tracks the surprisal -log2 q(x_t|c_t), NOT the
      raw symbol identity. Report the correlation r.
  (3) SHUNTING-BALANCE REGIME. The exact additive (input - prediction) idealization
      holds only where the inhibitory conductance tracks the excitatory drive
      linearly. We contrast the idealized SUBTRACTIVE inhibition (a clean negative
      current) against a conductance-based SHUNTING inhibition
      -g_inh (V - E_inh), and quantify where the linear input-minus-prediction
      arithmetic holds vs where E/I balance breaks at large mismatch.

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

from spikecoder import config as cfg            # noqa: E402
from spikecoder import latency as lat            # noqa: E402
from spikecoder import plotting as plot          # noqa: E402
from spikecoder.source import RoverSource, LABELS  # noqa: E402
from spikecoder.networks import make_lif         # noqa: E402

TAU_RC, TAU_REF, THETA = cfg.TAU_RC, cfg.TAU_REF, cfg.THETA
LAM = cfg.LAMBDA


# ---------------------------------------------------------------------------
# Core builder: one LIF soma whose net current is input_excitatory +/- prediction.
# A "sign" of +1 means ADDITIVE feedback (wrong); -1 means INHIBITORY (correct).
# ---------------------------------------------------------------------------
def run_soma(J_in, J_pred, sign=-1.0, dt=cfg.DT_FINE, tmax=0.25, seed=0,
             probe_voltage=False):
    """First-spike latency of a single rest-pinned LIF soma under net current
    J_eff = J_in + sign * J_pred. sign=-1 is inhibitory feedback (subtraction,
    the correct FIX-F sign); sign=+1 is additive feedback (the wrong sign).

    Both currents are injected directly into ens.neurons (gain 1, bias 0), so the
    soma integrates exactly J_eff in threshold units. Nengo clamps V >= 0, so an
    over-inhibited soma (J_eff < 1) charges toward a sub-threshold equilibrium and
    never fires -- the near-silence of a well-predicted symbol.
    """
    with nengo.Network(seed=seed) as net:
        ens = nengo.Ensemble(1, 1, neuron_type=make_lif(), gain=[1.0], bias=[0.0],
                             encoders=[[1.0]])
        nengo.Connection(nengo.Node(J_in), ens.neurons, synapse=None)
        # the feedback prediction current, entering with the chosen sign
        nengo.Connection(nengo.Node(sign * J_pred), ens.neurons, synapse=None)
        p = nengo.Probe(ens.neurons)
        if probe_voltage:
            pv = nengo.Probe(ens.neurons, "voltage")
    with nengo.Simulator(net, dt=dt, progress_bar=False) as sim:
        sim.run(tmax)
    ts = sim.trange()
    idx = np.flatnonzero(sim.data[p][:, 0] > 0)
    t1 = ts[idx[0]] if len(idx) else np.inf
    n_spikes = int(len(idx))
    V = sim.data[pv][:, 0] if probe_voltage else None
    return t1, n_spikes, ts, V


def run_soma_shunt(J_in, g_inh, dt=cfg.DT_FINE, tmax=0.25, E_inh=0.0, seed=0):
    """Conductance-based SHUNTING inhibition (the non-idealized regime).

    The LIF voltage obeys  tau dV = (-V + J_in - g_inh (V - E_inh)) dt, i.e. the
    inhibition adds a term -g_inh (V - E_inh) that *scales with V* (divisive /
    shunting) rather than subtracting a fixed current. We integrate this directly
    (Euler at the fine dt) and report the first-spike latency, to contrast with the
    clean additive subtraction. With E_inh = 0 (reversal at rest) the shunt's
    effective drive is J_in/(1 + g_inh) and its effective leak time constant is
    tau/(1 + g_inh): both the steady state and the speed change.
    """
    nt = int(round(tmax / dt))
    V = 0.0
    for k in range(nt):
        dV = (-V + J_in - g_inh * (V - E_inh)) / TAU_RC
        V = V + dt * dV
        if V < 0.0:
            V = 0.0  # Nengo clamps the membrane at rest
        if V >= THETA:
            return (k + 1) * dt
    return np.inf


def out_rate(t1, n_spikes, tmax):
    """A scalar 'residual output': spike count over the window (the surprise signal).
    A silenced (never-firing) soma outputs 0; an early-firing surprised soma fires
    many times.
    """
    return float(n_spikes)


print("=" * 72)
print("e07 -- Predictive subtraction: the one operation that needs a cycle")
print("=" * 72)

# ===========================================================================
# (1) SIGN CONTROL -- the FIX F demonstration.
#     A symbol arrives with a fixed excitatory sensory drive J_in. The loop
#     predicts it with probability q and feeds back a prediction current
#     J_pred = J_in * (q^alpha) so that a PERFECT prediction (q -> 1) cancels the
#     whole excitatory drive (balanced E/I) and q -> 0 cancels nothing.
#
#     Concretely we set J_in to the calibrated drive of a confident arrival and
#     scale the prediction so that residual J_eff = J_in - J_pred = theta/(1-q^a)
#     * (1 - q^a) ... we instead use the clean, controllable form below.
# ===========================================================================
# Excitatory sensory arrival: a strongly-driven symbol (well above rheobase).
J_in = 8.0                       # ~8x rheobase: a clearly-firing un-cancelled input
# prediction strength: q in [0,1], J_pred = q * J_in (inhibition matched to q).
# q=1 -> full cancellation (J_eff=0, silence); q=0 -> no cancellation (fires hard).
q_grid = np.array([0.0, 0.25, 0.5, 0.7, 0.85, 0.95, 0.99, 1.0])
J_pred_grid = q_grid * J_in

inhib = {"t1": [], "nsp": []}     # correct: J_eff = J_in - J_pred
addit = {"t1": [], "nsp": []}     # wrong:   J_eff = J_in + J_pred
TMAX = 0.2
for Jp in J_pred_grid:
    t1, nsp, _, _ = run_soma(J_in, Jp, sign=-1.0, tmax=TMAX)
    inhib["t1"].append(t1); inhib["nsp"].append(nsp)
    t1a, nspa, _, _ = run_soma(J_in, Jp, sign=+1.0, tmax=TMAX)
    addit["t1"].append(t1a); addit["nsp"].append(nspa)
for d in (inhib, addit):
    d["t1"] = np.array(d["t1"]); d["nsp"] = np.array(d["nsp"])
J_eff_inhib = J_in - J_pred_grid
J_eff_addit = J_in + J_pred_grid

print(f"\n[sign control] excitatory sensory drive J_in = {J_in}x rheobase")
print("q(pred) J_pred  | INHIB J_eff  t1(ms)   nsp | ADDIT J_eff  t1(ms)   nsp")
for k, q in enumerate(q_grid):
    ti = inhib["t1"][k]; ta = addit["t1"][k]
    si = f"{ti*1e3:7.2f}" if np.isfinite(ti) else "  silent"
    sa = f"{ta*1e3:7.2f}" if np.isfinite(ta) else "  silent"
    print(f"  {q:.2f}  {J_pred_grid[k]:5.2f} | {J_eff_inhib[k]:8.2f} {si}  {inhib['nsp'][k]:3d} "
          f"| {J_eff_addit[k]:8.2f} {sa}  {addit['nsp'][k]:3d}")

# A "well-predicted" symbol = high q. Under inhibition it is silenced; under
# addition it fires (and earlier than the un-predicted baseline).
well_pred = q_grid >= 0.95
inhib_silent_wellpred = np.all(~np.isfinite(inhib["t1"][well_pred]))
# the un-cancelled baseline latency (q=0, J_eff = J_in both ways)
base_t1 = inhib["t1"][0]
# additive sign DOUBLES drive at q=1 and fires strictly EARLIER than baseline
addit_fires_earlier = np.all(addit["t1"][well_pred] < base_t1 - cfg.DT_FINE)
addit_never_silent = np.all(np.isfinite(addit["t1"]))
print(f"\n  baseline (q=0, no prediction) latency = {base_t1*1e3:.2f} ms")
print(f"  INHIBITORY: well-predicted (q>=0.95) symbols are SILENT? {inhib_silent_wellpred}")
print(f"  ADDITIVE  : well-predicted symbols fire EARLIER than baseline (runaway)? "
      f"{addit_fires_earlier};  never silent? {addit_never_silent}")

# ===========================================================================
# (2) SURPRISE STREAM -- residual output tracks surprisal, not the symbol.
#     A rover stream with a KNOWN (true) predictor q(x_t|c_t) = P[c_t, x_t].
#     Each symbol arrives with a FIXED excitatory drive J_in; the loop inhibits
#     it with J_pred = J_in * q(x_t|c_t). The residual J_eff = J_in (1 - q) so a
#     well-predicted symbol (q->1) is silenced and a surprising one (q->0) fires.
#     We measure the soma output (spike count) per symbol and correlate it with
#     the surprisal -log2 q(x_t|c_t).
# ===========================================================================
src = RoverSource(s=0.7)
N_SYM = 120
stream = src.sample(N_SYM)              # ints 0..3
P = src.P
# context c_t = previous symbol; q(x_t|c_t) = P[x_{t-1}, x_t]; for t=0 use stationary pi
q_real = np.empty(N_SYM)
for t in range(N_SYM):
    if t == 0:
        q_real[t] = src.pi[stream[t]]
    else:
        q_real[t] = P[stream[t - 1], stream[t]]
q_real = np.clip(q_real, cfg.Q_CLIP_LO, cfg.Q_CLIP_HI)
surprisal = -np.log2(q_real)            # bits

# run each symbol window through the inhibitory soma; J_pred matched to q.
# Choose J_in so the rover's MOST confident "stay" move (q_max = s + (1-s) pi_U
# = 0.85) lands the residual J_eff = J_in (1 - q_max) just BELOW rheobase, so the
# best-predicted symbols are genuinely silenced (balanced E/I) while surprising
# moves keep a supra-threshold residual and fire. J_in*(1-0.85) < 1 => J_in < 6.67.
J_in_stream = 6.5
q_max_rover = float(q_real.max())
resid_nsp = np.empty(N_SYM)
resid_t1 = np.empty(N_SYM)
J_eff_stream = J_in_stream * (1.0 - q_real)     # residual drive after cancellation
for t in range(N_SYM):
    Jp = J_in_stream * q_real[t]
    t1, nsp, _, _ = run_soma(J_in_stream, Jp, sign=-1.0, tmax=0.2, seed=int(stream[t]))
    resid_nsp[t] = nsp
    resid_t1[t] = t1

# correlation of residual output (spike count) with surprisal
fin = resid_nsp > 0  # firing windows have a defined latency; count uses all windows
r_nsp = float(np.corrcoef(surprisal, resid_nsp)[0, 1])
# correlation with the raw symbol identity (should be near zero / weak): a control
r_symbol = float(np.corrcoef(stream.astype(float), resid_nsp)[0, 1])
# fraction of windows that are SILENCED (well predicted: high q, low surprisal)
silenced = resid_nsp == 0
frac_silenced = float(silenced.mean())
# mean surprisal of silenced vs firing windows
mean_surp_silent = float(surprisal[silenced].mean()) if silenced.any() else np.nan
mean_surp_fire = float(surprisal[~silenced].mean()) if (~silenced).any() else np.nan
print(f"\n[surprise stream] {N_SYM}-symbol rover stream (s=0.7), true predictor, "
      f"J_in={J_in_stream}x rheobase")
print(f"  rover q_max (most confident move) = {q_max_rover:.3f}  "
      f"=> residual J_eff = {J_in_stream*(1-q_max_rover):.2f}x rheobase (sub-threshold)")
print(f"  corr(residual spike count, surprisal -log2 q) r = {r_nsp:+.3f}")
print(f"  corr(residual spike count, raw symbol id)     r = {r_symbol:+.3f}  (control)")
print(f"  silenced (well-predicted) windows: {silenced.sum()}/{N_SYM} = {frac_silenced:.2f}")
print(f"  mean surprisal: silenced windows {mean_surp_silent:.3f} bits, "
      f"firing windows {mean_surp_fire:.3f} bits")

# ===========================================================================
# (3) SHUNTING-BALANCE REGIME -- where the linear (input - prediction) holds.
#     Idealized SUBTRACTIVE inhibition: J_eff = J_in - J_pred (a clean current,
#     the analyzed model). Conductance SHUNTING inhibition adds -g_inh (V - E_inh)
#     to the membrane: with E_inh = 0 (reversal at rest) this gives
#         tau dV = -(1 + g_inh) V + J_in,
#     i.e. a DIVISIVE rescaling -- steady state J_in/(1+g_inh) AND an effective
#     leak time constant tau/(1+g_inh).
#
#     Principled match: choose the shunt conductance that reproduces the SAME
#     sub-threshold steady state as a subtractive prediction J_pred = q J_in,
#         J_in/(1+g_inh) = J_in (1 - q)  =>  g_inh = q/(1-q).
#     With this match the two forms have IDENTICAL steady-state residual drive by
#     construction; the ONLY discrepancy is dynamical -- the shunt also divides the
#     membrane time constant by (1+g_inh), so it charges faster and fires earlier.
#     That latency gap is the breakdown of the "clean arithmetic": the subtractive
#     idealization is recovered where g_inh is small relative to the leak (small q,
#     the paper's small-conductance regime) and departs as the prediction
#     approaches balance (q -> 1, g_inh -> infinity).
# ===========================================================================
J_in_b = 6.0
# sweep a grid that resolves the genuinely-small-conductance edge (g = q/(1-q)),
# where the additive idealization is recovered, as well as the near-balance breakdown
q_b = np.unique(np.concatenate([
    np.array([0.0, 0.02, 0.05, 0.09]),             # small g: g = 0, .02, .05, .10
    np.linspace(0.15, 0.95, 12),
]))
E_inh = 0.0
g_b = q_b / (1.0 - q_b)                             # steady-state-matched shunt
# subtractive: J_pred = q * J_in (the idealized linear cancellation)
t_sub = np.array([run_soma(J_in_b, q * J_in_b, sign=-1.0, tmax=0.6)[0] for q in q_b])
# shunting: divisive conductance, steady-state-matched to the subtractive residual
t_shunt = np.array([run_soma_shunt(J_in_b, g, tmax=0.6) for g in g_b])
# steady-state residual drives (IDENTICAL by the match; sanity check)
r_sub = J_in_b * (1.0 - q_b)                        # subtractive residual
r_shunt_ss = J_in_b / (1.0 + g_b)                   # shunt steady-state (== r_sub)
ss_match_err = float(np.max(np.abs(r_sub - r_shunt_ss)))
# latency gap: the dynamical departure (finite, both fire)
both_fire = np.isfinite(t_sub) & np.isfinite(t_shunt)
lat_gap = np.full_like(q_b, np.nan)
lat_gap[both_fire] = np.abs(t_sub[both_fire] - t_shunt[both_fire])
# relative latency departure |t_shunt - t_sub| / t_sub: the dynamical nonlinearity
rel_lat = np.full_like(q_b, np.nan)
rel_lat[both_fire] = lat_gap[both_fire] / t_sub[both_fire]
print(f"\n[shunting-balance regime] J_in = {J_in_b}x rheobase, E_inh = {E_inh} (at rest)")
print(f"  steady-state-matched shunt g=q/(1-q); max |r_sub - r_shunt_ss| = {ss_match_err:.2e}")
print("q(pred)  g_inh   resid r   t_sub(ms)  t_shunt(ms)  |gap|(ms)  rel.dep=|dt|/t_sub")
for k, q in enumerate(q_b):
    ts = f"{t_sub[k]*1e3:8.2f}" if np.isfinite(t_sub[k]) else "  silent"
    th = f"{t_shunt[k]*1e3:8.2f}" if np.isfinite(t_shunt[k]) else "  silent"
    lg = f"{lat_gap[k]*1e3:7.2f}" if np.isfinite(lat_gap[k]) else "    --"
    rd = f"{rel_lat[k]:.3f}" if np.isfinite(rel_lat[k]) else "  --"
    print(f"  {q:.2f}  {g_b[k]:6.2f}  {r_sub[k]:7.3f}  {ts}  {th}  {lg}     {rd}")
# small-conductance regime (g << 1, the paper's "small relative to the leak" regime)
# vs near-balance (g >> 1, the shunt dominates the leak)
small_g = g_b <= 0.10     # conductance small relative to the unit leak
near_bal = g_b >= 4.0     # shunt dominates the leak (near full E/I balance)
rel_small = float(np.nanmax(rel_lat[small_g]))
rel_large = float(np.nanmax(rel_lat[near_bal])) if near_bal.any() else float(np.nanmax(rel_lat))
# the q at which the departure first exceeds 10% (the practical edge of "linear")
edge = np.where(np.isfinite(rel_lat) & (rel_lat > 0.10))[0]
q_linear_edge = float(q_b[edge[0]]) if len(edge) else float(q_b[-1])
print(f"\n  max relative latency departure, small conductance (g<=0.10): {rel_small:.3f}")
print(f"  max relative latency departure, near balance      (g>=4.0):  {rel_large:.3f}")
print(f"  linear idealization edge: rel.departure first exceeds 10% at q = {q_linear_edge:.2f} "
      f"(g = {q_linear_edge/(1-q_linear_edge):.2f})")
print("  -> the linear input-minus-prediction idealization holds only while the shunt "
      "conductance is small relative to the leak (g<<1, small q); it degrades steadily "
      "as the prediction approaches balance (q->1, g->inf), where the divisive "
      "time-constant rescaling makes the shunt fire well before the subtractive model.")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# Fig 1: sign control -- inhibitory vs additive latency/output vs prediction quality
fig, (axL, axR) = plot.plt.subplots(1, 2, figsize=(11, 4.2))
# left: first-spike latency vs prediction quality
ti = inhib["t1"].copy(); ta = addit["t1"].copy()
# plot silenced (inf) at a capped marker above the axis
cap = TMAX * 1e3
ti_plot = np.where(np.isfinite(ti), ti * 1e3, cap)
axL.plot(q_grid, ti_plot, "o-", color=plot.C_MEASURED, lw=2,
         label="inhibitory  $J_{eff}=J_{in}-J_{pred}$ (FIX F)")
axL.plot(q_grid, ta * 1e3, "s--", color=plot.C_THEORY, lw=2,
         label="additive  $J_{eff}=J_{in}+J_{pred}$ (wrong sign)")
axL.axhline(base_t1 * 1e3, color="gray", ls=":", lw=1.2,
            label=f"un-cancelled baseline {base_t1*1e3:.1f} ms")
axL.annotate("SILENCED\n(balanced E/I)", (0.97, cap), color=plot.C_MEASURED,
             fontsize=8, ha="right", va="top")
axL.set_xlabel("prediction quality q (inhibition matched to q)")
axL.set_ylabel("first-spike latency (ms)")
axL.set_title("Latency: inhibition silences, addition runs away")
axL.legend(frameon=False, fontsize=8)
# right: net soma current J_eff
axR.plot(q_grid, J_eff_inhib, "o-", color=plot.C_MEASURED, lw=2, label="inhibitory $J_{eff}$")
axR.plot(q_grid, J_eff_addit, "s--", color=plot.C_THEORY, lw=2, label="additive $J_{eff}$")
axR.axhline(THETA, color=plot.C_FLOOR, ls="--", lw=1.3, label=r"rheobase $\theta=1$")
axR.fill_between(q_grid, 0, THETA, color=plot.C_FLOOR, alpha=0.08)
axR.annotate("sub-threshold:\nsilence", (0.5, 0.4), fontsize=8, color=plot.C_FLOOR)
axR.set_xlabel("prediction quality q")
axR.set_ylabel(r"net soma current $J_{eff}$ (threshold units)")
axR.set_title("Net drive: subtraction balances E/I to ~0")
axR.legend(frameon=False, fontsize=8)
plot.save(fig, RESULTS / "e07_sign_control.pdf")

# Fig 1b: side-by-side voltage traces for a predicted-then-surprising pair
# Predicted symbol: q=0.95 (strong inhibition). Surprising: q=0.1 (weak inhibition).
q_predd, q_surp = 0.95, 0.1
_, _, ts_p_i, V_p_i = run_soma(J_in, q_predd * J_in, sign=-1.0, tmax=0.12, probe_voltage=True)
_, _, ts_p_a, V_p_a = run_soma(J_in, q_predd * J_in, sign=+1.0, tmax=0.12, probe_voltage=True)
_, _, ts_s_i, V_s_i = run_soma(J_in, q_surp * J_in, sign=-1.0, tmax=0.12, probe_voltage=True)
fig, ax = plot.new_fig(7.5, 4.0)
ax.plot(ts_p_i * 1e3, V_p_i, color=plot.C_MEASURED, lw=2,
        label=f"predicted q={q_predd}, INHIB → silenced")
ax.plot(ts_s_i * 1e3, V_s_i, color=plot.C_FLOOR, lw=2,
        label=f"surprising q={q_surp}, INHIB → fires")
ax.plot(ts_p_a * 1e3, V_p_a, color=plot.C_THEORY, lw=2, ls="--",
        label=f"predicted q={q_predd}, ADDITIVE → runaway")
ax.axhline(THETA, color="gray", ls=":", lw=1.2, label=r"threshold $\theta$")
ax.set_xlim(0, 60)   # one charge cycle is enough to read the three fates
ax.set_xlabel("time (ms)")
ax.set_ylabel("membrane voltage V")
ax.set_title("e07: predicted-then-surprising — the sign decides who spikes")
ax.legend(frameon=False, fontsize=8, loc="center right")
plot.save(fig, RESULTS / "e07_voltage_traces.pdf")

# Fig 2: residual output vs surprisal scatter (surprise stream)
fig, ax = plot.new_fig(6.5, 4.2)
jit = np.random.default_rng(0).normal(0, 0.04, size=N_SYM)
sc = ax.scatter(surprisal, resid_nsp + jit, c=stream, cmap="viridis", s=28,
                alpha=0.8, edgecolors="none")
# linear fit line
b1, b0 = np.polyfit(surprisal, resid_nsp, 1)
xx = np.linspace(surprisal.min(), surprisal.max(), 50)
ax.plot(xx, b0 + b1 * xx, "-", color=plot.C_THEORY, lw=2,
        label=f"fit (r={r_nsp:+.2f})")
ax.set_xlabel(r"surprisal $-\log_2 q(x_t \mid c_t)$ (bits)")
ax.set_ylabel("residual output (spike count / window)")
ax.set_title("e07: the residual spike stream IS a surprise stream")
cb = fig.colorbar(sc, ax=ax, ticks=range(4))
cb.ax.set_yticklabels(LABELS)
cb.set_label("emitted symbol")
ax.legend(frameon=False, fontsize=9)
plot.save(fig, RESULTS / "e07_surprise_scatter.pdf")

# Fig 3: shunting-balance regime -- subtractive vs steady-state-matched shunt latency
fig, ax = plot.new_fig(7.2, 4.2)
ax.plot(q_b, t_sub * 1e3, "o-", color=plot.C_MEASURED, lw=2,
        label=r"subtractive $J_{in}-J_{pred}$ (idealized model)")
ax.plot(q_b, t_shunt * 1e3, "s--", color=plot.C_THEORY, lw=2,
        label=r"shunting $-g_{inh}(V-E_{inh})$, $g=q/(1-q)$")
ax.set_xlabel("prediction quality q  (steady-state-matched: same residual drive)")
ax.set_ylabel("first-spike latency (ms)")
ax.set_title("e07: matched steady states, divergent latencies near balance")
ax2 = ax.twinx()
ax2.plot(q_b, rel_lat, ":", color="gray", lw=1.8,
         label=r"rel. departure $|\Delta t|/t_{sub}$")
ax2.set_ylabel("relative latency departure", color="gray")
ax2.tick_params(axis="y", labelcolor="gray")
ax2.grid(False)
# annotate the small-conductance (linear) vs near-balance (broken) regimes
ax.axvspan(0.0, 0.5, color=plot.C_FLOOR, alpha=0.07)
ax.annotate("small g:\nlinear holds", (0.18, t_sub[3] * 1e3 * 1.3),
            fontsize=8, color=plot.C_FLOOR)
ax.annotate("g -> inf:\nbalance breaks", (0.74, t_sub[-3] * 1e3 * 0.4),
            fontsize=8, color="gray")
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, frameon=False, fontsize=8, loc="upper left")
plot.save(fig, RESULTS / "e07_shunt_regime.pdf")

# ---------------------------------------------------------------------------
# Save + acceptance
# ---------------------------------------------------------------------------
np.savez(RESULTS / "e07_results.npz",
         q_grid=q_grid, J_pred_grid=J_pred_grid, J_in=J_in,
         inhib_t1=inhib["t1"], inhib_nsp=inhib["nsp"],
         addit_t1=addit["t1"], addit_nsp=addit["nsp"],
         J_eff_inhib=J_eff_inhib, J_eff_addit=J_eff_addit, base_t1=base_t1,
         stream=stream, q_real=q_real, surprisal=surprisal,
         resid_nsp=resid_nsp, resid_t1=resid_t1, J_eff_stream=J_eff_stream,
         r_nsp=r_nsp, r_symbol=r_symbol, frac_silenced=frac_silenced,
         mean_surp_silent=mean_surp_silent, mean_surp_fire=mean_surp_fire,
         q_b=q_b, t_sub=t_sub, t_shunt=t_shunt, r_sub=r_sub, r_shunt_ss=r_shunt_ss,
         g_b=g_b, lat_gap=lat_gap, rel_lat=rel_lat, ss_match_err=ss_match_err,
         rel_small=rel_small, rel_large=rel_large, q_linear_edge=q_linear_edge)

checks = {
    "INHIBITORY feedback silences well-predicted symbols (q>=0.95 -> no spike)":
        inhib_silent_wellpred,
    "ADDITIVE (wrong) sign never silences -- no cancellation":
        addit_never_silent,
    "ADDITIVE sign fires EARLIER than baseline (doubles drive -> runaway)":
        addit_fires_earlier,
    "residual output tracks surprisal (positive corr, r > 0.5)":
        r_nsp > 0.5,
    "residual tracks surprisal, NOT the raw symbol (|r_symbol| < r_nsp)":
        abs(r_symbol) < r_nsp,
    "well-predicted windows silenced, surprising windows fire (E/I split)":
        (frac_silenced > 0.0) and (mean_surp_silent < mean_surp_fire),
    "subtractive idealization holds at small shunt conductance (rel.dep < 0.1)":
        rel_small < 0.1,
    "E/I balance / linear idealization degrades near balance (large g)":
        rel_large > rel_small,
}
print("\n" + "=" * 72)
allok = True
for name, ok in checks.items():
    allok &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
print(f"\ne07: {sum(checks.values())}/{len(checks)} acceptance checks passed.")
sys.exit(0 if allok else 1)
