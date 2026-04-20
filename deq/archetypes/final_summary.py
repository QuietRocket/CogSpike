"""Generate the final summary triptych figure.

Three panels on a common axis and colour scale for contralateral inhibition:
  (1) Phase 0 deterministic ground truth (blue/red staircase).
  (2) Combinatorial predictor's ||w_12|-|w_21|| with θ=7 contour
      overlaid on the deterministic GT.
  (3) Phase 1c reachability GT with scalar-r ρ(A) contour (ρ>0.544 cutoff).

This figure is the single-image summary of the project's central finding:
deterministic dynamics is combinatorial, reachability dynamics is spectral,
and the boundary between the two regimes is the scientific result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from archetypes.spectral import operating_point, linearized_A, spectral_radius
from archetypes.topologies import contralateral

RESULTS = HERE / "results"


def main():
    w_vals = np.load(RESULTS / "fcs_fig10_wvals.npy")
    gt_det = np.load(RESULTS / "fcs_fig10_groundtruth.npy")
    gt_reach = np.load(RESULTS / "fcs_fig10_reachable_groundtruth.npy")

    # Recompute scalar-r ρ(A) (fast, 1600 cells)
    print("[Final] Recomputing ρ(A)...")
    n = len(w_vals)
    rho = np.zeros((n, n))
    u = np.array([1.0, 1.0])
    for i, w21 in enumerate(w_vals):
        for j, w12 in enumerate(w_vals):
            W, B, _ = contralateral(int(w12), int(w21), T=1)
            pstar = operating_point(W, B, u, r=0.5, p_mid=30.0)
            A = linearized_A(W, pstar, r=0.5, p_mid=30.0)
            rho[i, j] = spectral_radius(A)

    # Combinatorial asymmetry
    Wv12, Wv21 = np.meshgrid(w_vals, w_vals)
    abs_diff = np.abs(np.abs(Wv12) - np.abs(Wv21))

    # --- Triptych
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    extent = [w_vals.max() + 0.5, w_vals.min() - 0.5,
              w_vals.max() + 0.5, w_vals.min() - 0.5]

    # Panel 1: deterministic GT
    ax = axes[0]
    im = ax.imshow(gt_det.astype(int), cmap="RdBu", vmin=0, vmax=1,
                   extent=extent, origin="upper", aspect="equal",
                   interpolation="nearest")
    ax.set_title("(a) Deterministic ground truth (Phase 0)\n"
                 f"63.4% blue, two asymmetry wings",
                 fontsize=11)
    ax.set_xlabel(r"$w_{12}$ (inhibition N1$\to$N2)")
    ax.set_ylabel(r"$w_{21}$ (inhibition N2$\to$N1)")
    ax.set_xlim(-0.5, -40.5); ax.set_ylim(-0.5, -40.5)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_ticks([0, 1]); cbar.set_ticklabels(["red (no WTA)", "blue (WTA)"])

    # Panel 2: deterministic GT with combinatorial contour
    ax = axes[1]
    im = ax.imshow(gt_det.astype(int), cmap="RdBu", vmin=0, vmax=1,
                   extent=extent, origin="upper", aspect="equal",
                   interpolation="nearest", alpha=0.65)
    Xc = np.linspace(extent[0], extent[1], abs_diff.shape[1])
    Yc = np.linspace(extent[2], extent[3], abs_diff.shape[0])
    Xg, Yg = np.meshgrid(Xc, Yc)
    cs = ax.contour(Xg, Yg, abs_diff, levels=[7], colors="black", linewidths=2.0)
    ax.clabel(cs, fmt=r"$||w_{12}|-|w_{21}|| = 7$", inline=True, fontsize=9)
    ax.set_title("(b) Deterministic GT + combinatorial predictor\n"
                 r"$||w_{12}|-|w_{21}|| > 7$: 83.4% accuracy",
                 fontsize=11)
    ax.set_xlabel(r"$w_{12}$"); ax.set_ylabel(r"$w_{21}$")
    ax.set_xlim(-0.5, -40.5); ax.set_ylim(-0.5, -40.5)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Panel 3: reachability GT with ρ(A) contour
    ax = axes[2]
    im = ax.imshow(gt_reach.astype(int), cmap="RdBu", vmin=0, vmax=1,
                   extent=extent, origin="upper", aspect="equal",
                   interpolation="nearest", alpha=0.65)
    cs = ax.contour(Xg, Yg, rho, levels=[0.544], colors="black", linewidths=2.0)
    ax.clabel(cs, fmt=r"$\rho(A) = 0.544$", inline=True, fontsize=9)
    ax.set_title("(c) Reachability GT + spectral predictor\n"
                 r"$\rho(A) > 0.544$: 98.5% accuracy",
                 fontsize=11)
    ax.set_xlabel(r"$w_{12}$"); ax.set_ylabel(r"$w_{21}$")
    ax.set_xlim(-0.5, -40.5); ax.set_ylim(-0.5, -40.5)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_ticks([0, 1]); cbar.set_ticklabels(["red (no WTA)", "blue (WTA)"])

    fig.suptitle("Spectral Cartography of LI&F Archetypes: central finding\n"
                 "Deterministic dynamics = combinatorial   |   "
                 "Reachability dynamics = spectral",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS / "final_triptych.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"[Final] Saved {RESULTS / 'final_triptych.png'}")


if __name__ == "__main__":
    main()
