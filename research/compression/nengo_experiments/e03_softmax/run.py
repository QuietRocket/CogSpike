#!/usr/bin/env python3
"""e03 -- NEF softmax predictor vs Carandini-Heeger divisive normalization.

Claim (paper #7, the gain-control fix A1, paper §"Gain control: softmax versus
divisive normalization"):

  * The predicted distribution q_j = softmax(W c)_j is the EXACT simplex
    normalizer: sum_j q_j = 1 identically, by construction (the partition $sum
    e^{a_k}$ has no additive slack).
  * Carandini-Heeger divisive normalization r_i = a_i / (sigma + sum_j a_j) sums
    to sum a / (sigma + sum a) < 1 whenever sigma > 0, and -> 1 only as sigma -> 0.
    It is the *biophysical approximation* to softmax, NOT an exact normalizer.

In float64 (the numpy validator, spikecoder.information.softmax) the partition of
unity is free and exact. The spiking-reality twist: in the NEF the softmax must be
DECODED from a finite, heterogeneous LIF population, so sum q = 1 stops being an
algebraic identity and becomes a REPRESENTATIONAL quantity -- it holds only up to
the population's decode error, which shrinks as ~1/sqrt(n_neurons). The headline
gap this experiment quantifies: partition-of-unity is now *bought with neurons*.

We measure three things:
  (1) NEF softmax: sweep n_neurons, measure per-component RMSE of decoded q vs the
      analytic softmax and the normalization defect |sum q - 1|; show the ~1/sqrt(N)
      NEF scaling on log-log axes, and report the bits-of-normalization accuracy
      per neuron.
  (2) Divisive normalization: sweep sigma on a fixed nonnegative drive; show sum(r)
      = sum a/(sigma + sum a) < 1 and -> 1 as sigma -> 0 (it matches its analytic
      value exactly, because it is a deterministic gain node, not a spiking decode).
  (3) Contrast: softmax sums to 1 (exact); divisive norm sums to < 1, for a
      representative vector.
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
from spikecoder import information as info        # noqa: E402
from spikecoder import plotting as plot           # noqa: E402
from spikecoder.networks import (                 # noqa: E402
    build_softmax_predictor, build_divisive_norm)

N = 4                       # alphabet size (U, D, L, R)
RADIUS = 4.0                # NEF representation radius -- must cover the logit range
DT = cfg.DT                 # throughput grid (this is not a latency-timing experiment)
SETTLE = 0.30               # sim time per held logit vector
TAIL = 50                   # timesteps at the settled tail to average the decode over


def decode_softmax(a, n_neurons, seed=cfg.SEED):
    """Run a fresh NEF softmax ensemble on a held logit vector a; return decoded q.

    Builds an n_neurons ensemble representing the N-dim logit a, decodes q =
    softmax(a) (build_softmax_predictor), holds a constant for SETTLE seconds and
    averages the output over the settled tail (rejects the synaptic transient).
    """
    with nengo.Network(seed=seed) as net:
        logit = nengo.Node(a)
        _, qout = build_softmax_predictor(net, logit, N=N, n_neurons=n_neurons,
                                          radius=RADIUS, seed=seed)
        p = nengo.Probe(qout, synapse=0.02)
    with nengo.Simulator(net, dt=DT, progress_bar=False) as sim:
        sim.run(SETTLE)
    return sim.data[p][-TAIL:].mean(axis=0)


def divnorm_sum(a, sigma, seed=cfg.SEED):
    """Decoded sum of the divisive-normalization node on nonnegative drive a."""
    with nengo.Network(seed=seed) as net:
        drv = nengo.Node(np.maximum(a, 0.0))
        out = build_divisive_norm(net, drv, N=N, sigma=sigma)
        p = nengo.Probe(out, synapse=0.005)
    with nengo.Simulator(net, dt=DT, progress_bar=False) as sim:
        sim.run(0.10)
    return sim.data[p][-20:].mean(axis=0)


print("=" * 70)
print("e03 -- NEF softmax (exact simplex normalizer) vs divisive normalization")
print("=" * 70)

# A fixed test set of logit vectors covering the representable ball.
rng = np.random.default_rng(cfg.SEED)
# magnitude up to ~RADIUS along the N-sphere so softmax spans near-uniform to peaked
n_test = 12
test_logits = rng.uniform(-RADIUS / np.sqrt(N), RADIUS / np.sqrt(N), size=(n_test, N))

# --- (1) NEF softmax: error vs n_neurons --------------------------------------
n_neuron_grid = np.array([50, 100, 200, 400, 800, 1600])
rmse_mean, rmse_std = [], []
sumdef_mean, sumdef_std = [], []
for nn in n_neuron_grid:
    rmses, sumdefs = [], []
    for a in test_logits:
        q_dec = decode_softmax(a, nn)
        q_true = info.softmax(a)
        rmses.append(float(np.sqrt(np.mean((q_dec - q_true) ** 2))))
        sumdefs.append(float(abs(q_dec.sum() - 1.0)))
    rmse_mean.append(np.mean(rmses))
    rmse_std.append(np.std(rmses))
    sumdef_mean.append(np.mean(sumdefs))
    sumdef_std.append(np.std(sumdefs))
rmse_mean = np.array(rmse_mean); rmse_std = np.array(rmse_std)
sumdef_mean = np.array(sumdef_mean); sumdef_std = np.array(sumdef_std)

print("\n[NEF softmax]  n_neurons   RMSE(q)      |sum q - 1|   (mean over "
      f"{n_test} test logits)")
for k, nn in enumerate(n_neuron_grid):
    print(f"               {nn:>6d}     {rmse_mean[k]:.4f}       {sumdef_mean[k]:.4f}")

# 1/sqrt(N) scaling: fit log RMSE = b - p log N; the NEF predicts slope p ~ 0.5
logN = np.log(n_neuron_grid.astype(float))
slope_rmse, icpt_rmse = np.polyfit(logN, np.log(rmse_mean), 1)
slope_sum, icpt_sum = np.polyfit(logN, np.log(sumdef_mean), 1)
print(f"\nlog-log slope of RMSE   vs n_neurons = {slope_rmse:+.3f}  (NEF ideal -0.5)")
print(f"log-log slope of |sum-1| vs n_neurons = {slope_sum:+.3f}  (NEF ideal -0.5)")

# --- bits-of-normalization accuracy per neuron --------------------------------
# Normalization defect |sum q - 1| is a probability error. Read it as the precision
# (in bits) to which the partition of unity is enforced: bits = -log2(defect), and
# the marginal value of one neuron is d(bits)/d(n_neurons) ~ p / (n ln2) at the
# 1/sqrt(n) rate. We report the precision achieved per neuron count.
norm_bits = -np.log2(sumdef_mean)         # bits of partition-of-unity precision
bits_per_neuron = norm_bits / n_neuron_grid
print("\n[normalization accuracy as bits]  n_neurons   -log2|sum q-1| (bits)   "
      "bits per neuron")
for k, nn in enumerate(n_neuron_grid):
    print(f"                                  {nn:>6d}        {norm_bits[k]:.2f}"
          f"                  {bits_per_neuron[k]*1e3:.4f}e-3")
# how many neurons buy one extra bit of normalization precision, at this rate
neurons_per_bit = 1.0 / (abs(slope_sum) / np.log(2.0) / n_neuron_grid)  # local
print(f"\nAt the {slope_sum:+.2f} log-log rate, one extra bit of partition-of-unity "
      f"precision costs ~{neurons_per_bit[2]:.0f}x more neurons near n={n_neuron_grid[2]}.")

# representative decode at the largest population, for the report table
q_true_rep = info.softmax(test_logits[0])
q_dec_rep = decode_softmax(test_logits[0], n_neuron_grid[-1])

# --- (2) Divisive normalization: sum vs sigma ---------------------------------
a_drive = np.array([2.0, 1.0, 0.5, 0.5])   # representative nonnegative drive
tot = float(a_drive.sum())
sigmas = np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005])
dn_sum_meas, dn_sum_analytic = [], []
for s in sigmas:
    r = divnorm_sum(a_drive, s)
    dn_sum_meas.append(float(r.sum()))
    dn_sum_analytic.append(tot / (s + tot))
dn_sum_meas = np.array(dn_sum_meas)
dn_sum_analytic = np.array(dn_sum_analytic)
print(f"\n[divisive norm]  total drive sum a = {tot}")
print("sigma     sum(r) meas    sum a/(sigma+sum a)    deficit 1-sum(r)")
for k, s in enumerate(sigmas):
    print(f"{s:<8.3f}  {dn_sum_meas[k]:.4f}         {dn_sum_analytic[k]:.4f}"
          f"                 {1-dn_sum_meas[k]:.4f}")
print("  -> sum(r) < 1 for every sigma > 0, and -> 1 as sigma -> 0 "
      "(the paper's Proposition).")

# --- (3) Contrast: softmax (sums to 1) vs divisive norm (sums to < 1) ---------
# Use the same vector for both: logits a -> softmax; same a as nonnegative drive
# (shift to nonnegative) -> divisive norm at a representative sigma.
a_rep = np.array([2.0, 1.0, 0.5, 0.5])
q_soft = info.softmax(a_rep)               # exact simplex (numpy reference)
q_soft_nef = decode_softmax(a_rep, 800)    # NEF decode at a strong population
sigma_rep = 0.1
r_div = divnorm_sum(a_rep, sigma_rep)

print(f"\n[contrast on a={a_rep.tolist()}]")
print(f"  softmax (numpy exact):  sum = {q_soft.sum():.6f}  -> EXACT 1")
print(f"  softmax (NEF, n=800):   sum = {q_soft_nef.sum():.6f}  -> representational")
print(f"  div-norm (sigma={sigma_rep}):  sum = {r_div.sum():.6f}  -> < 1 by design")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# Fig 1: RMSE and |sum q -1| vs n_neurons, log-log, with the 1/sqrt(N) guide.
fig, ax = plot.new_fig()
ax.errorbar(n_neuron_grid, rmse_mean, yerr=rmse_std, fmt="o-", color=plot.C_MEASURED,
            capsize=3, label=r"per-component RMSE$(q)$")
ax.errorbar(n_neuron_grid, sumdef_mean, yerr=sumdef_std, fmt="s-", color=plot.C_FLOOR,
            capsize=3, label=r"normalization defect $|\sum q - 1|$")
# 1/sqrt(N) reference anchored at the first RMSE point
ref = rmse_mean[0] * np.sqrt(n_neuron_grid[0]) / np.sqrt(n_neuron_grid.astype(float))
ax.plot(n_neuron_grid, ref, "--", color=plot.C_THEORY, lw=1.5,
        label=r"$\propto 1/\sqrt{N}$ (NEF scaling)")
ax.axhline(0.05, ls=":", color="gray", lw=1, label="0.05 target")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("ensemble size $n_{neurons}$")
ax.set_ylabel("error")
ax.set_title("e03: NEF softmax error decreases as $1/\\sqrt{N}$")
ax.legend(frameon=False, fontsize=8)
plot.save(fig, RESULTS / "e03_nef_scaling.pdf")

# Fig 2: divisive-norm sum vs sigma (measured == analytic), with the softmax line.
fig, ax = plot.new_fig()
sd = np.linspace(sigmas.min(), sigmas.max(), 200)
ax.plot(sd, tot / (sd + tot), "-", color=plot.C_THEORY, lw=2,
        label=r"$\sum a/(\sigma + \sum a)$ (analytic)")
ax.plot(sigmas, dn_sum_meas, "o", color=plot.C_MEASURED, ms=7,
        label="divisive-norm node (measured)")
ax.axhline(1.0, ls="--", color=plot.C_FLOOR, lw=1.5, label="softmax: $\\sum q = 1$ exact")
ax.set_xlabel(r"semi-saturation $\sigma$")
ax.set_ylabel(r"$\sum_i r_i$")
ax.set_ylim(0.75, 1.02)
ax.set_title("e03: divisive norm sums to $<1$; $\\to 1$ only as $\\sigma\\to 0$")
ax.legend(frameon=False, fontsize=8)
plot.save(fig, RESULTS / "e03_divnorm_sigma.pdf")

# Fig 3: contrast bars -- softmax (sums to 1) vs divisive norm (sums to <1).
labels = ["U", "D", "L", "R"]
x = np.arange(N)
fig, ax = plot.new_fig(w=6.5, h=4.0)
ax.bar(x - 0.27, q_soft, 0.27, color=plot.C_MEASURED,
       label=f"softmax  (Σ={q_soft.sum():.3f}, exact)")
ax.bar(x, q_soft_nef, 0.27, color="#7fb3e0",
       label=f"softmax NEF n=800  (Σ={q_soft_nef.sum():.3f})")
ax.bar(x + 0.27, r_div, 0.27, color=plot.C_THEORY, alpha=0.8,
       label=f"div-norm σ={sigma_rep}  (Σ={r_div.sum():.3f}, <1)")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("output component")
ax.set_xlabel(f"symbol   (logits a = {a_rep.tolist()})")
ax.set_title("e03: softmax is the exact normalizer; divisive norm leaves a deficit")
ax.legend(frameon=False, fontsize=8)
plot.save(fig, RESULTS / "e03_contrast.pdf")

# ---------------------------------------------------------------------------
# Save + acceptance
# ---------------------------------------------------------------------------
np.savez(RESULTS / "e03_results.npz",
         n_neuron_grid=n_neuron_grid, rmse_mean=rmse_mean, rmse_std=rmse_std,
         sumdef_mean=sumdef_mean, sumdef_std=sumdef_std,
         slope_rmse=slope_rmse, slope_sum=slope_sum,
         norm_bits=norm_bits, bits_per_neuron=bits_per_neuron,
         sigmas=sigmas, dn_sum_meas=dn_sum_meas, dn_sum_analytic=dn_sum_analytic,
         a_drive=a_drive, a_rep=a_rep, q_soft=q_soft, q_soft_nef=q_soft_nef,
         r_div=r_div, sigma_rep=sigma_rep, test_logits=test_logits,
         q_true_rep=q_true_rep, q_dec_rep=q_dec_rep)

# index of n=400 in the grid (the >=400 RMSE target)
idx400 = int(np.where(n_neuron_grid == 400)[0][0])
dn_max_sigma_pos = float(dn_sum_meas[sigmas == sigmas.max()][0])   # largest sigma
dn_min_sigma = float(dn_sum_meas[sigmas == sigmas.min()][0])       # smallest sigma

checks = {
    "NEF softmax RMSE decreases monotonically with n_neurons":
        bool(np.all(np.diff(rmse_mean) < 0)),
    "NEF softmax RMSE < 0.05 at n_neurons >= 400":
        bool(np.all(rmse_mean[idx400:] < 0.05)),
    "RMSE log-log slope ~ -1/2 (NEF scaling, in [-0.7,-0.3])":
        bool(-0.7 <= slope_rmse <= -0.3),
    "normalization defect |sum q-1| shrinks with n_neurons":
        bool(sumdef_mean[-1] < sumdef_mean[0]),
    "divisive norm sums to < 1 for every sigma > 0":
        bool(np.all(dn_sum_meas < 1.0 - 1e-6)),
    "divisive norm sum increases toward 1 as sigma -> 0":
        bool(dn_min_sigma > dn_max_sigma_pos and dn_min_sigma > 0.99),
    "softmax (numpy) sums to 1 exactly (the exact simplex normalizer)":
        bool(abs(q_soft.sum() - 1.0) < 1e-12),
    "divisive-norm node matches its analytic sum (deterministic gain)":
        bool(np.max(np.abs(dn_sum_meas - dn_sum_analytic)) < 1e-3),
}

print("\n" + "=" * 70)
allok = True
for name, ok in checks.items():
    allok &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
print(f"\ne03: {sum(checks.values())}/{len(checks)} acceptance checks passed.")
sys.exit(0 if allok else 1)
