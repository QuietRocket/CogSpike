#!/usr/bin/env python3
"""e02 -- Latency calibration: t*(q) = -lambda log2 q, and the honesty checks.

Claim (paper Theorem 1): drive a readout whose model probability is q with the
calibration current R I(q) = theta / (1 - q^alpha), alpha = lambda/(tau ln2).
Then its first-spike latency is EXACTLY t*(q) = -lambda log2 q -- the surprisal of
q, as a spike time. The paper then flags three physical limits (its §4 honesty
checks): finite drive caps the representable probability at q_max ~ 0.85-0.95;
the rare-symbol (q->0) end is noise-dominated; finite timing resolution dt
replaces the integer-bit penalty.

This experiment turns those prose honesty blocks into measured curves in a real
Nengo LIF neuron. Because the first spike from rest carries no refractory term
(e01), the paper-faithful calibration already gives the exact identity -- no
"refractory-corrected" drive is needed (a simplification over the project plan).
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
from spikecoder import plotting as plot        # noqa: E402
from spikecoder.networks import make_lif       # noqa: E402

TAU_RC, LAM = cfg.TAU_RC, cfg.LAMBDA


def first_spike(J, dt=cfg.DT_FINE, sigma=0.0, tmax=0.25, seed=0):
    """First-spike latency from rest under DC drive J + optional white-noise current."""
    with nengo.Network(seed=seed) as net:
        ens = nengo.Ensemble(1, 1, neuron_type=make_lif(), gain=[1.0], bias=[0.0],
                             encoders=[[1.0]])
        nengo.Connection(nengo.Node(J), ens.neurons, synapse=None)
        if sigma > 0:
            noise = nengo.Node(nengo.processes.WhiteNoise(
                dist=nengo.dists.Gaussian(0, sigma), scale=False, seed=seed))
            nengo.Connection(noise, ens.neurons, synapse=None)
        p = nengo.Probe(ens.neurons)
    with nengo.Simulator(net, dt=dt, progress_bar=False) as sim:
        sim.run(tmax)
    idx = np.flatnonzero(sim.data[p][:, 0] > 0)
    return sim.trange()[idx[0]] if len(idx) else np.inf


print("=" * 70)
print("e02 -- Latency calibration t*(q) = -lambda log2 q")
print("=" * 70)

# (1) the exact identity, clean (no noise), dt fine
qs = np.array([0.98, 0.95, 0.9, 0.85, 0.75, 0.5, 0.25, 0.125, 0.075, 0.0375])
J = lat.calibration_drive(qs)
t_meas = np.array([first_spike(j, dt=cfg.DT_FINE, tmax=0.3) for j in J])
t_ideal = lat.analytic_latency_ideal(qs)
err_dt = np.abs(t_meas - t_ideal) / cfg.DT_FINE
print(f"\nalpha = {cfg.ALPHA:.4f}  lambda = {LAM} s/bit")
print("q       J=RI/θ    t_meas(s)  -λ log2 q   surprisal   |err|/dt")
for k, q in enumerate(qs):
    print(f"{q:<7} {J[k]:<9.3f} {t_meas[k]:<10.5f} {t_ideal[k]:<11.5f} "
          f"{-np.log2(q):<11.3f} {err_dt[k]:.2f}")
print(f"\nmax |t_meas - (-λ log2 q)| / dt = {err_dt.max():.2f}  (calibration exact to ~dt)")

# (2) drive table + q_max ceiling
ceilings = [5.0, 10.0, 15.0]
qmaxes = {c: lat.q_max_for_ceiling(c) for c in ceilings}
print("\n[drive ceiling -> q_max]")
for c in ceilings:
    print(f"  ceiling {c:>4.0f}x rheobase  ->  q_max = {qmaxes[c]:.3f}, "
          f"floor latency t_min = {lat.analytic_latency_ideal(qmaxes[c])*1e3:.3f} ms")
drive_tab = lat.drive_table()  # paper's q=0.5..0.99 table

# (3) q_max demonstration: cap the drive at a ceiling and watch high-q symbols floor
ceiling = cfg.RHEOBASE_CEILING
qmax = lat.q_max_for_ceiling(ceiling)
q_hi = np.array([0.85, 0.9, qmax, 0.95, 0.98, 0.995])
J_uncapped = lat.calibration_drive(q_hi)
J_capped = np.minimum(J_uncapped, ceiling)
t_capped = np.array([first_spike(j, dt=cfg.DT_FINE, tmax=0.1) for j in J_capped])
t_want = lat.analytic_latency_ideal(q_hi)
print(f"\n[q_max floor] ceiling = {ceiling}x, q_max = {qmax:.3f}")
print("q       want t*(ms)  capped t*(ms)  saturated?")
for k, q in enumerate(q_hi):
    sat = "YES (floored)" if J_uncapped[k] > ceiling else "no"
    print(f"{q:<7} {t_want[k]*1e3:<11.3f} {t_capped[k]*1e3:<13.3f} {sat}")

# (4) noise-dominated rare-symbol tail: latency mean/std vs q under membrane noise
sigma = 0.15
n_trials = 40
q_noise = np.array([0.7, 0.5, 0.3, 0.15, 0.08, 0.04])
means, stds = [], []
for q in q_noise:
    j = lat.calibration_drive(q)
    samples = np.array([first_spike(j, dt=cfg.DT_FINE, sigma=sigma, tmax=0.4, seed=s)
                        for s in range(n_trials)])
    samples = samples[np.isfinite(samples)]
    means.append(samples.mean())
    stds.append(samples.std())
means, stds = np.array(means), np.array(stds)
cv = stds / means  # coefficient of variation
print(f"\n[noise tail] sigma={sigma}, {n_trials} trials/q")
print("q       mean t*(ms)  std(ms)   CV=std/mean")
for k, q in enumerate(q_noise):
    print(f"{q:<7} {means[k]*1e3:<11.3f} {stds[k]*1e3:<9.3f} {cv[k]:.3f}")
print("  -> the rarer the symbol (smaller q, near-rheobase drive), the noisier the latency")

# --- figures ---
fig, ax = plot.new_fig()
qd = np.linspace(0.02, 0.99, 300)
ax.plot(-np.log2(qd), lat.analytic_latency_ideal(qd) * 1e3, "-", color=plot.C_THEORY,
        lw=2, label=r"theory $t^* = -\lambda\log_2 q$")
ax.plot(-np.log2(qs), t_meas * 1e3, "o", color=plot.C_MEASURED, ms=6, label="Nengo measured")
ax.set_xlabel(r"surprisal $-\log_2 q$ (bits)")
ax.set_ylabel("first-spike latency (ms)")
ax.set_title("e02: spike time IS surprisal (calibration exact to ~dt)")
ax.legend(frameon=False)
plot.save(fig, RESULTS / "e02_calibration.pdf")

fig, ax = plot.new_fig()
qd = np.linspace(0.05, 0.995, 400)
ax.plot(qd, lat.calibration_drive(qd), "-", color=plot.C_MEASURED, lw=2,
        label=r"$R I/\theta = (1-q^\alpha)^{-1}$")
for c in ceilings:
    ax.axhline(c, ls="--", lw=1, alpha=0.7)
    ax.annotate(f"{c:.0f}x → q_max={qmaxes[c]:.2f}", (0.07, c + 0.3), fontsize=8)
ax.set_ylim(1, 20)
ax.set_xlabel("model probability q")
ax.set_ylabel(r"drive $R I/\theta$")
ax.set_title("e02: finite drive caps the representable probability q_max")
ax.legend(frameon=False)
plot.save(fig, RESULTS / "e02_drive_ceiling.pdf")

fig, ax = plot.new_fig()
ax.errorbar(q_noise, means * 1e3, yerr=stds * 1e3, fmt="o-", color=plot.C_MEASURED,
            capsize=4, label="mean ± std")
ax.plot(q_noise, lat.analytic_latency_ideal(q_noise) * 1e3, "--", color=plot.C_THEORY,
        label="noise-free theory")
ax.set_xlabel("model probability q")
ax.set_ylabel("first-spike latency (ms)")
ax.set_title(f"e02: rare symbols are noise-dominated (σ={sigma})")
ax.legend(frameon=False)
plot.save(fig, RESULTS / "e02_noise_tail.pdf")

# --- save + acceptance ---
np.savez(RESULTS / "e02_results.npz",
         qs=qs, J=J, t_meas=t_meas, t_ideal=t_ideal, err_dt=err_dt,
         qmaxes=np.array([qmaxes[c] for c in ceilings]), ceilings=np.array(ceilings),
         q_hi=q_hi, t_capped=t_capped, t_want=t_want, J_uncapped=J_uncapped,
         q_noise=q_noise, means=means, stds=stds, cv=cv)

# capped high-q symbols should fire LATER than they "want" (floored), i.e. saturate
saturated = J_uncapped > ceiling
floored_ok = np.all(t_capped[saturated] > t_want[saturated] - cfg.DT_FINE)
checks = {
    "calibration t*(q) = -lambda log2 q exact to ~dt (max < 2 dt)": err_dt.max() < 2.0,
    "drive table matches paper (q=0.9 -> ~7.1x)": abs(dict((q, ri) for q, ri, _ in drive_tab)[0.9] - 7.09) < 0.1,
    "q_max in [0.85, 0.95] for a 10x ceiling": 0.85 <= qmaxes[10.0] <= 0.95,
    "symbols above q_max saturate (latency floors)": floored_ok,
    "latency noise (CV) grows as q -> 0": cv[-1] > cv[0],
}
print("\n" + "=" * 70)
allok = True
for name, ok in checks.items():
    allok &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
print(f"\ne02: {sum(checks.values())}/{len(checks)} acceptance checks passed.")
sys.exit(0 if allok else 1)
