"""Regenerate the note's figures with clean labels (no internal phase
names). Reads the `.npy` artifacts produced by the phase sweeps and
writes into note/figs/.

This script is idempotent; running it overwrites the existing PDFs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

HERE = Path(__file__).resolve().parent
POP = HERE.parent
sys.path.insert(0, str(POP))
RESULTS = POP / "results"
OUT = HERE / "figs"
OUT.mkdir(exist_ok=True)

# FCS-style palette for direct-comparison panels (matches FCS Fig. 10):
# teal = property holds; red = property fails. The note's other figures
# keep the existing greyscale to avoid colour-overload.
FCS_TEAL = "#2ca5a5"
FCS_RED = "#cc3030"
FCS_CMAP = mcolors.ListedColormap([FCS_RED, FCS_TEAL])


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# --- Contralateral pitchfork overlay (p2_pitchfork) --------------------------
sym = np.load(RESULTS / "pitchfork_curve_symbolic.npy")
num = np.load(RESULTS / "bifurcation_curve.npy")
fig, ax = plt.subplots(figsize=(4.5, 4.0))
ax.plot(num[:, 0], num[:, 1], "k.", markersize=2, label="numerical (radial bisection)")
ax.plot(sym[:, 0], sym[:, 1], "r-", linewidth=1.0, label="symbolic ($\\det J = 0$)")
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_xlabel(r"$w_{12}$")
ax.set_ylabel(r"$w_{21}$")
ax.set_title("Contralateral pitchfork: symbolic vs numerical")
ax.legend(fontsize=9)
save(fig, "p2_pitchfork")


# --- Hopf overlay (p2_hopf) --------------------------------------------------
osc = np.load(RESULTS / "oscillation_map_negative_loop.npy")
w_grid = np.load(RESULTS / "w_grid_negative_loop.npy")
hopf = np.load(RESULTS / "hopf_curve_analytical.npy")
extent = (w_grid[0], w_grid[-1], w_grid[0], w_grid[-1])
fig, ax = plt.subplots(figsize=(5.0, 4.0))
ax.imshow(
    osc.T, origin="lower", extent=extent, cmap="Greys",
    vmin=0, vmax=1, aspect="equal", interpolation="nearest", alpha=0.7,
)
ax.plot(hopf[:, 0], hopf[:, 1], "r.", markersize=2.5, label="analytical Hopf locus")
ax.set_xlabel(r"$w_{\mathrm{ai}}$")
ax.set_ylabel(r"$w_{\mathrm{ia}}$")
ax.set_title(r"Activator--inhibitor Hopf ($w_{\mathrm{aa}} = 2.5$)")
ax.legend(loc="upper right", fontsize=9)
save(fig, "p2_hopf")


# --- Frequency comparison (p2_freq) ------------------------------------------
omega_grid = np.load(RESULTS / "oscillation_freq_negative_loop.npy")


def nearest_freq(points_ai, points_ia, curve_ai, curve_ia, curve_om):
    out = np.full(points_ai.size, np.nan)
    for idx, (a, b) in enumerate(zip(points_ai, points_ia)):
        d = np.sqrt((curve_ai - a) ** 2 + (curve_ia - b) ** 2)
        out[idx] = float(curve_om[int(np.argmin(d))])
    return out


mask = np.isfinite(omega_grid)
ai_pts = np.array([w_grid[i] for i in np.where(mask)[0]])
ia_pts = np.array([w_grid[j] for j in np.where(mask)[1]])
omega_meas = omega_grid[mask]
omega_pred = nearest_freq(ai_pts, ia_pts, hopf[:, 0], hopf[:, 1], hopf[:, 2])
fig, ax = plt.subplots(figsize=(4.5, 4.0))
ax.plot(omega_pred, omega_meas, ".", markersize=3, alpha=0.4)
lim = float(max(np.nanmax(omega_meas), np.nanmax(omega_pred)))
ax.plot([0, lim], [0, lim], "k--", linewidth=0.8)
ax.set_xlabel(r"$\omega^*$ analytical (nearest Hopf point)")
ax.set_ylabel(r"$\omega$ measured (FFT)")
ax.set_title("Oscillation frequency: analytical vs measured")
save(fig, "p2_freq")


# --- Pole-placement scatter (p3_scatter) -------------------------------------
import csv

rows = []
with (RESULTS / "pole_placement_table.csv").open() as fh:
    reader = csv.DictReader(fh)
    for r in reader:
        rows.append(r)
targets = np.array([float(r["target_omega"]) for r in rows])
measured = np.array([float(r["measured_omega"]) for r in rows])
fig, ax = plt.subplots(figsize=(4.5, 4.0))
ax.plot(targets, measured, "o")
lim = float(max(targets.max(), measured.max()) * 1.05)
ax.plot([0, lim], [0, lim], "k--", linewidth=0.8)
ax.set_xlabel(r"$\omega^*$ target")
ax.set_ylabel(r"$\omega$ measured (FFT)")
ax.set_title("Pole placement: target vs realised frequency")
save(fig, "p3_scatter")

# ============================================================================
# v2 FIGURES
# ============================================================================

# --- Symmetric fixed-point branches rho*(w) for the contralateral case ------
# Plot the fixed-point structure on the diagonal w12=w21=w. Below the
# pitchfork (w < w*=1) the symmetric state is the unique attractor;
# above, it is the saddle, and two asymmetric stable branches appear.
# Using `find_all_fixed_points_contralateral` to enumerate all roots.
from wilson_cowan import Sigmoid, find_all_fixed_points_contralateral  # noqa: E402

DRIVE = 1.5
sigmoid = Sigmoid(k=4.0, theta=1.0)
w_diag = np.linspace(0.0, 5.0, 401)
rows = []
for w in w_diag:
    fps = find_all_fixed_points_contralateral(float(w), float(w), DRIVE, sigmoid)
    for fp in fps:
        rows.append((float(w), float(fp[0]), float(fp[1])))
rows = np.array(rows)
fig, ax = plt.subplots(figsize=(5.0, 3.6))
# colour by branch type using a simple rule: |rho1 - rho2| > 0.02 => asymmetric
asym = np.abs(rows[:, 1] - rows[:, 2]) > 0.02
ax.scatter(rows[~asym, 0], rows[~asym, 1], s=4, color="#444444", label="symmetric branch")
ax.scatter(rows[asym, 0], rows[asym, 1], s=4, color=FCS_TEAL, label="asymmetric branches")
ax.scatter(rows[asym, 0], rows[asym, 2], s=4, color=FCS_TEAL)
ax.axvline(1.0, color=FCS_RED, linestyle="--", linewidth=1, label=r"$w_*=1$ (pitchfork)")
ax.set_xlabel(r"diagonal weight $w = w_{12} = w_{21}$")
ax.set_ylabel(r"fixed-point activity $\rho^*$")
ax.set_xlim(0, 5)
ax.set_ylim(-0.02, 1.02)
ax.set_title("Contralateral symmetric fixed-point branches")
ax.legend(loc="upper right", fontsize=9)
save(fig, "rho_star_curve")


# --- 4-tick WTA panel (E1) compared with t=50 ground truth ------------------
wta_t4 = np.load(RESULTS / "ground_truth_contralateral_t4.npy")
wta_t50 = np.load(RESULTS / "ground_truth_contralateral.npy")
w_grid = np.load(RESULTS / "w_grid.npy")
extent = (w_grid[0], w_grid[-1], w_grid[0], w_grid[-1])
sym_curve = np.load(RESULTS / "pitchfork_curve_symbolic.npy")

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
for ax, wta_arr, label, t_label in (
    (axes[0], wta_t50, "(a) asymptotic, $t = 50\\,\\tau$", "t=50"),
    (axes[1], wta_t4, "(b) time-bounded, $t = 4\\,\\tau$", "t=4"),
):
    # imshow expects [row=y, col=x]; we plot w12 on x and w21 on y, so transpose.
    ax.imshow(
        wta_arr.T.astype(int),
        origin="lower",
        extent=extent,
        cmap=FCS_CMAP,
        vmin=0, vmax=1,
        aspect="equal",
        interpolation="nearest",
    )
    ax.plot(sym_curve[:, 0], sym_curve[:, 1], "k-", linewidth=1.0, alpha=0.6)
    ax.set_xlim(0, 5); ax.set_ylim(0, 5)
    ax.set_xlabel(r"$w_{12}$")
    if ax is axes[0]:
        ax.set_ylabel(r"$w_{21}$")
    ax.set_title(label, fontsize=10)
fig.suptitle("WTA verdict on the WC contralateral archetype: teal = WTA, red = no WTA",
             fontsize=10)
fig.tight_layout()
save(fig, "p1_t4_panel")


# --- Loop-gain heatmap on the 50x50 grid ------------------------------------
# Show the loop-gain product w12 * w21 * g1 * g2 evaluated at the saddle
# fixed point. The unity contour (loop gain = 1) is the pitchfork locus.
gap = np.load(RESULTS / "spectral_gap_contralateral.npy")  # for reference
# Recompute loop gain from scratch using the saddle FP at each cell.
from wilson_cowan import find_saddle_contralateral  # noqa: E402

N = w_grid.size
loop_gain = np.full((N, N), np.nan)
for i, w12 in enumerate(w_grid):
    for j, w21 in enumerate(w_grid):
        rho_star, n_fp = find_saddle_contralateral(float(w12), float(w21), DRIVE, sigmoid)
        if n_fp == 0 or np.any(np.isnan(rho_star)):
            continue
        # arg_i is the input to neuron i: arg_1 = I - w12 * rho_2, etc.
        arg1 = DRIVE - float(w12) * float(rho_star[1])
        arg2 = DRIVE - float(w21) * float(rho_star[0])
        g1 = float(sigmoid.f_prime(arg1))
        g2 = float(sigmoid.f_prime(arg2))
        loop_gain[i, j] = float(w12) * float(w21) * g1 * g2

fig, ax = plt.subplots(figsize=(4.8, 4.0))
# Use a divergent colour map centred at 1.0.
norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=float(np.nanmax(loop_gain)))
im = ax.imshow(loop_gain.T, origin="lower", extent=extent, cmap="RdBu_r",
               aspect="equal", interpolation="nearest", norm=norm)
cs = ax.contour(w_grid, w_grid, loop_gain.T, levels=[1.0], colors="black", linewidths=1.4)
ax.clabel(cs, fmt={1.0: r"loop gain = 1"}, inline=True, fontsize=8)
ax.set_xlabel(r"$w_{12}$")
ax.set_ylabel(r"$w_{21}$")
ax.set_title(r"Loop gain $w_{12} w_{21} g_1 g_2$ at the saddle FP")
fig.colorbar(im, ax=ax, label="loop gain")
save(fig, "loop_gain_heatmap")


# --- Extreme-weight LI&F sweep (E3) ----------------------------------------
lif_extreme = np.load(RESULTS / "lif_extreme_wta_map.npy")
lif_w = np.load(RESULTS / "lif_extreme_w_vals.npy")
# Phase 4 (the existing 40x40 sweep) for context.
lif40 = np.load(RESULTS / "lif_wta_map.npy")

fig, ax = plt.subplots(figsize=(5.6, 4.6))
extent_e = (lif_w[0], lif_w[-1], lif_w[0], lif_w[-1])
ax.imshow(
    lif_extreme.T.astype(int),
    origin="lower",
    extent=extent_e,
    cmap=FCS_CMAP,
    vmin=0, vmax=1,
    aspect="equal",
    interpolation="nearest",
)
# Outline the existing 40x40 sweep region.
ax.plot([1, 40, 40, 1, 1], [1, 1, 40, 40, 1], "k--", linewidth=0.8,
        label="Phase 4 sweep box (|w| <= 40)")
ax.set_xlim(1, lif_w[-1]); ax.set_ylim(1, lif_w[-1])
ax.set_xlabel(r"$|w_{12}^{\mathrm{LIF}}|$")
ax.set_ylabel(r"$|w_{21}^{\mathrm{LIF}}|$")
ax.set_title(r"LI\&F bistable region out to $|w| \leq 200$")
ax.legend(loc="lower right", fontsize=9)
save(fig, "p4_lif_extreme")


print("Regenerated all figures into", OUT)
