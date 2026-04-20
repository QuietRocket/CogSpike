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
import numpy as np

HERE = Path(__file__).resolve().parent
POP = HERE.parent
sys.path.insert(0, str(POP))
RESULTS = POP / "results"
OUT = HERE / "figs"
OUT.mkdir(exist_ok=True)


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

print("Regenerated 4 figures into", OUT)
