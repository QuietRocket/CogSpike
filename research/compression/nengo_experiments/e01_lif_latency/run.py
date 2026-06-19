#!/usr/bin/env python3
"""e01 -- Raw LIF first-spike latency law, in real Nengo spikes.

Claim (paper #1): a leaky integrate-and-fire neuron driven by constant current
charges as V(t) = R I (1 - e^{-t/tau}) and fires its first spike at latency
t*(I) = tau ln( R I / (R I - theta) ).

This is the most primitive rung: before any calibration, does Nengo's actual
spiking LIF obey the latency law? Two spiking-reality facts surface here that the
pure-numpy validator (one clean ODE at Euler dt=1e-5, V(0)=0) never sees:

  (1) Nengo randomizes the initial membrane voltage in [0,1) to desynchronize
      neurons. For a first-spike latency code this must be pinned to rest (V(0)=0),
      or the neuron fires early by a random amount.

  (2) The refractory period tau_ref is *post-spike dead time*: it delays the
      *inter-spike interval*, NOT the first spike from rest. So the first-spike
      latency from rest is t = tau_rc ln(J/(J-1)) with NO tau_ref term -- and
      since the coder resets each readout to rest per symbol window, the latency
      code is refractory-immune.

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
from spikecoder import plotting as plot        # noqa: E402
from spikecoder.networks import make_lif       # noqa: E402

TAU_RC, TAU_REF = cfg.TAU_RC, cfg.TAU_REF


def first_two_spikes(J, dt, pin_voltage=True, tmax=0.2, seed=0):
    """Return (first-spike time, second-spike time) for constant drive J from rest."""
    nt = make_lif(TAU_RC, TAU_REF) if pin_voltage else nengo.LIF(tau_rc=TAU_RC, tau_ref=TAU_REF)
    with nengo.Network(seed=seed) as net:
        ens = nengo.Ensemble(1, 1, neuron_type=nt, gain=[1.0], bias=[0.0], encoders=[[1.0]])
        nengo.Connection(nengo.Node(J), ens.neurons, synapse=None)
        p = nengo.Probe(ens.neurons)
        pv = nengo.Probe(ens.neurons, "voltage")
    with nengo.Simulator(net, dt=dt, progress_bar=False) as sim:
        sim.run(tmax)
    ts = sim.trange()
    idx = np.flatnonzero(sim.data[p][:, 0] > 0)
    t1 = ts[idx[0]] if len(idx) else np.inf
    t2 = ts[idx[1]] if len(idx) > 1 else np.inf
    return t1, t2, ts, sim.data[pv][:, 0]


def analytic_first_spike(J):
    return TAU_RC * np.log(J / (J - 1.0))      # from rest, no tau_ref


def analytic_isi(J):
    return TAU_REF + TAU_RC * np.log(J / (J - 1.0))  # steady-state inter-spike interval


# ---------------------------------------------------------------------------
print("=" * 70)
print("e01 -- Raw LIF first-spike latency law")
print("=" * 70)

# (1) Initial-voltage randomization: pinned vs default, at fixed J
J0 = 2.0
t1_pin, _, ts_pin, V_pin = first_two_spikes(J0, cfg.DT_FINE, pin_voltage=True)
# sample several seeds of the *default* (randomized) init to show the scatter
t1_rand = []
for sd in range(12):
    t1r, _, _, _ = first_two_spikes(J0, cfg.DT_FINE, pin_voltage=False, seed=sd)
    t1_rand.append(t1r)
t1_rand = np.array(t1_rand)
t1_ideal = analytic_first_spike(J0)
print(f"\n[init-voltage] J={J0}: pinned t1={t1_pin:.5f}s (ideal {t1_ideal:.5f}); "
      f"randomized t1 in [{t1_rand.min():.5f}, {t1_rand.max():.5f}] over 12 seeds")

# (2) Latency law: sweep J, pinned from rest, dt fine
Js = np.array([1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0])
dt = cfg.DT_FINE
t1s, t2s = [], []
for J in Js:
    a, b, _, _ = first_two_spikes(J, dt, pin_voltage=True, tmax=0.3)
    t1s.append(a)
    t2s.append(b)
t1s, t2s = np.array(t1s), np.array(t2s)
t1_law = analytic_first_spike(Js)
isi_law = analytic_isi(Js)
err_first_dt = np.abs(t1s - t1_law) / dt
err_first_isi_dt = np.abs(t1s - isi_law) / dt   # how far first spike is from the +tau_ref law
print("\n[latency law]  J     t1_meas   tau_rc ln(J/(J-1))   |err|/dt   (+tau_ref law) |err|/dt")
for k, J in enumerate(Js):
    print(f"             {J:<6} {t1s[k]:.5f}   {t1_law[k]:.5f}            {err_first_dt[k]:.2f}        {err_first_isi_dt[k]:.1f}")
print(f"\nmax |t1 - tau_rc ln(J/(J-1))| / dt = {err_first_dt.max():.2f}  (first spike: NO tau_ref)")

# recover tau_rc by least squares: t1 = tau_rc * x, x = ln(J/(J-1))
xfit = np.log(Js / (Js - 1.0))
tau_rc_fit = float(np.sum(xfit * t1s) / np.sum(xfit * xfit))
print(f"recovered tau_rc (LS fit) = {tau_rc_fit*1000:.3f} ms  (true {TAU_RC*1000:.1f} ms)")

# (3) Refractory shows up in the ISI (2nd-1st spike), matching tau_ref + tau_rc ln(...)
isi_meas = t2s - t1s
isi_err = np.abs(isi_meas - isi_law)
print(f"\n[refractory]  inter-spike interval matches tau_ref+tau_rc ln(J/(J-1)): "
      f"max|err| = {isi_err.max()*1000:.3f} ms  => tau_ref delays the SECOND spike, not the first")

# (4) dt quantization: error floor shrinks with dt
dts = [2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
maxerr_vs_dt = []
for d in dts:
    e = []
    for J in [1.5, 2.0, 5.0]:
        a, _, _, _ = first_two_spikes(J, d, pin_voltage=True, tmax=0.3)
        e.append(abs(a - analytic_first_spike(J)))
    maxerr_vs_dt.append(max(e))
maxerr_vs_dt = np.array(maxerr_vs_dt)
print(f"\n[dt quantization] max latency error vs dt: "
      + ", ".join(f"dt={d:.0e}->{er*1e3:.3f}ms" for d, er in zip(dts, maxerr_vs_dt)))
print("  -> error floor tracks dt (the spike lands on the time grid)")

# --- figures ---
# Fig 1: init-voltage randomization
fig, ax = plot.new_fig()
ax.axhline(t1_ideal * 1e3, color=plot.C_THEORY, lw=2, label=f"ideal (V0=0): {t1_ideal*1e3:.2f} ms")
ax.plot([0] * len(t1_rand), t1_rand * 1e3, "o", color="gray", alpha=0.6, label="randomized V0 (12 seeds)")
ax.plot([0], [t1_pin * 1e3], "s", color=plot.C_MEASURED, ms=10, label="pinned V0=0")
ax.set_xticks([])
ax.set_ylabel("first-spike latency (ms)")
ax.set_title("e01: Nengo randomizes initial voltage; pin it for the latency code")
ax.legend(frameon=False)
plot.save(fig, RESULTS / "e01_init_voltage.pdf")

# Fig 2: latency law vs J
fig, ax = plot.new_fig()
Jdense = np.linspace(1.02, 12, 300)
ax.plot(Jdense, analytic_first_spike(Jdense) * 1e3, "-", color=plot.C_THEORY, lw=2,
        label=r"$\tau_{rc}\ln\frac{J}{J-1}$ (first spike, no $\tau_{ref}$)")
ax.plot(Jdense, analytic_isi(Jdense) * 1e3, "--", color="gray", lw=1.3,
        label=r"$\tau_{ref}+\tau_{rc}\ln\frac{J}{J-1}$ (ISI)")
ax.plot(Js, t1s * 1e3, "o", color=plot.C_MEASURED, ms=7, label="Nengo first spike")
ax.set_xlabel("drive J = R I / θ")
ax.set_ylabel("latency (ms)")
ax.set_title("e01: first-spike latency law (refractory-immune from rest)")
ax.legend(frameon=False)
plot.save(fig, RESULTS / "e01_latency_law.pdf")

# Fig 3: dt quantization
fig, ax = plot.new_fig()
ax.loglog(dts, maxerr_vs_dt, "o-", color=plot.C_MEASURED, label="max latency error")
ax.loglog(dts, dts, "--", color=plot.C_THEORY, label="error = dt")
ax.set_xlabel("timestep dt (s)")
ax.set_ylabel("max |t1 - analytic| (s)")
ax.set_title("e01: latency error floor tracks the timing resolution dt")
ax.legend(frameon=False)
plot.save(fig, RESULTS / "e01_dt_quantization.pdf")

# --- save + acceptance ---
np.savez(RESULTS / "e01_results.npz",
         Js=Js, t1s=t1s, t1_law=t1_law, isi_meas=isi_meas, isi_law=isi_law,
         err_first_dt=err_first_dt, tau_rc_fit=tau_rc_fit,
         t1_rand=t1_rand, t1_pin=t1_pin, dts=dts, maxerr_vs_dt=maxerr_vs_dt)

checks = {
    "first spike matches tau_rc ln(J/(J-1)) within ~dt": err_first_dt.max() <= 2.0,
    "first spike does NOT carry tau_ref (offset law is worse)": err_first_isi_dt.min() > 5.0,
    "recovered tau_rc within 2% of true": abs(tau_rc_fit - TAU_RC) / TAU_RC < 0.02,
    "ISI matches tau_ref+tau_rc ln(...) (refractory in 2nd spike)": isi_err.max() < 2 * dt,
    "randomized V0 spreads first-spike (motivates pinning)": (t1_rand.max() - t1_rand.min()) > 5 * dt,
    "dt error floor decreases with dt": maxerr_vs_dt[-1] < maxerr_vs_dt[0],
}
print("\n" + "=" * 70)
allok = True
for name, ok in checks.items():
    allok &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
print(f"\ne01: {sum(checks.values())}/{len(checks)} acceptance checks passed.")
sys.exit(0 if allok else 1)
