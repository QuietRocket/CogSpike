"""Phase 2: H(omega) (dynamic) reading of FCS Property 5.

Converts Phase 1's Jacobian Im(lambda) into a predicted ringing period

    T_pred = 2 * pi / |Im(lambda)|

interpreted as the natural oscillation period of the rate-equation
linearization at the Siegert FP. Compares against FCS's measured period
from Phase 0.

Two lenses:

  • Continuous: T_pred heatmap and scatter of T_pred vs FCS-measured
    period; report the mean ratio T_pred / T_FCS over strict-P5 cells.
  • Boolean: label cells "period-4 blue" iff |T_pred - 4| <= tol. The
    pessimistic check is at tol=0.5 (half a tick); we also sweep tol in
    [0.1, 8.0] to find the best Jaccard.

Output: results/phase2/h_grid.npz, period_predicted_vs_4.pdf,
T_pred_vs_FCS_period.pdf.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "phase2"
RESULTS.mkdir(parents=True, exist_ok=True)


TWO_PI = 2.0 * np.pi
PERIOD_TARGET = 4.0
TOL_DEFAULT = 0.5  # half a tick


def jaccard(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = (a & b).sum(); union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else 1.0


def main():
    # Load Phase 0 (FCS truth) and Phase 1 (Siegert eigenvalues).
    p0 = np.load(HERE / "results" / "phase0" / "fcs_grid.npz",
                 allow_pickle=True)
    p1 = np.load(HERE / "results" / "phase1" / "siegert_grid.npz",
                 allow_pickle=True)

    W_IA = p0["W_IA_VALUES"]
    W_XA = p0["W_XA_VALUES"]
    W_AI = int(p0["W_AI"])
    strict_p5 = p0["strict_p5"].astype(bool)
    broad_osc = p0["broad_osc"].astype(bool)
    fcs_period = p0["period"]
    eig_im = p1["eig_im"]
    eig_re = p1["eig_re"]
    spiral_blue = p1["spiral_blue"]

    nIA, nXA = strict_p5.shape

    # eig_im[..., 0] is the upper conjugate's Im; |Im(lambda)| = magnitude.
    im_dom = np.abs(eig_im[..., 0])
    # Avoid division by zero: where im is essentially 0, T_pred is infinite (= no ringing).
    with np.errstate(divide="ignore", invalid="ignore"):
        T_pred = np.where(im_dom > 1e-8, TWO_PI / im_dom, np.nan)

    print(f"Phase 2: T_pred = 2*pi / |Im(lambda)| on {nIA} x {nXA} grid")
    print(f"  spiral cells: {int(spiral_blue.sum())} / {spiral_blue.size}")

    n_valid = int(np.isfinite(T_pred).sum())
    print(f"  cells with finite T_pred: {n_valid}")

    if n_valid > 0:
        print(f"  T_pred quantiles (across valid cells): "
              f"min={np.nanmin(T_pred):.2f}, "
              f"med={np.nanmedian(T_pred):.2f}, "
              f"mean={np.nanmean(T_pred):.2f}, "
              f"max={np.nanmax(T_pred):.2f}")

    # Boolean label at default tol.
    h_p4_blue = (np.isfinite(T_pred) &
                 (np.abs(T_pred - PERIOD_TARGET) <= TOL_DEFAULT))
    h_p4_blue = h_p4_blue.astype(int)
    j_default = jaccard(h_p4_blue, strict_p5)
    print(f"\nT_pred period-4 gate at tol={TOL_DEFAULT}:")
    print(f"  blue cells: {int(h_p4_blue.sum())}/{h_p4_blue.size}")
    print(f"  Jaccard vs FCS strict_p5: {j_default:.3f}")

    # Sweep tol.
    tols = np.linspace(0.1, 8.0, 80)
    j_curve = []
    for t in tols:
        mask = (np.isfinite(T_pred) & (np.abs(T_pred - PERIOD_TARGET) <= t))
        j_curve.append(jaccard(mask.astype(int), strict_p5))
    j_curve = np.array(j_curve)
    best_i = int(np.argmax(j_curve))
    best_tol = float(tols[best_i])
    best_j = float(j_curve[best_i])
    print(f"\nBest tolerance sweep: tol={best_tol:.2f}, Jaccard={best_j:.3f}")

    # Continuous metric: mean(|T_pred - 4|) over strict-P5 cells.
    over_p5 = strict_p5 & np.isfinite(T_pred)
    if over_p5.any():
        err_over_p5 = float(np.mean(np.abs(T_pred[over_p5] - PERIOD_TARGET)))
        ratio_over_p5 = float(np.mean(T_pred[over_p5] / PERIOD_TARGET))
        print(f"\nOver strict-P5 cells ({int(over_p5.sum())}):")
        print(f"  mean |T_pred - 4| = {err_over_p5:.2f} ticks")
        print(f"  mean T_pred / 4   = {ratio_over_p5:.2f}")

    # Cells whose FCS period equals 4 -- T_pred should ideally equal 4 there.
    fcs_p4_mask = (fcs_period == 4) & np.isfinite(T_pred)
    if fcs_p4_mask.any():
        ratio_p4 = float(np.mean(T_pred[fcs_p4_mask] / PERIOD_TARGET))
        print(f"\nOver FCS-period=4 cells ({int(fcs_p4_mask.sum())}):")
        print(f"  mean T_pred / 4   = {ratio_p4:.2f}")

    # Sanity gate cell.
    i_def = int(np.where(W_IA == -11)[0][0])
    j_def = int(np.where(W_XA == 11)[0][0])
    T_def = float(T_pred[i_def, j_def])
    print(f"\nFCS default cell (w_IA=-11, w_XA=11):")
    print(f"  T_pred = {T_def:.3f} ticks (FCS measured period = 4)")
    print(f"  ratio = T_pred / 4 = {T_def/4:.2f}")

    # Save.
    np.savez(
        RESULTS / "h_grid.npz",
        W_IA_VALUES=W_IA,
        W_XA_VALUES=W_XA,
        W_AI=W_AI,
        T_pred=T_pred,
        h_p4_blue=h_p4_blue,
        tol_default=TOL_DEFAULT,
        period_target=PERIOD_TARGET,
        jaccard_strict=j_default,
        best_tol=best_tol,
        best_jaccard=best_j,
        j_curve=j_curve,
        tols=tols,
    )

    # ---------- plots ----------

    # 4-panel: FCS strict-P5 / FCS period heat / T_pred heat / period-4 blue.
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

    # Panel 0: FCS strict-P5 (boolean).
    ax = axes[0]
    for i, w_IA in enumerate(W_IA):
        for j, w_XA in enumerate(W_XA):
            color = "tab:blue" if strict_p5[i, j] else "tab:red"
            ax.scatter(int(w_XA), int(w_IA), c=color, s=18, edgecolor="none")
    ax.scatter([11], [-11], facecolors="none", edgecolors="gold",
               s=160, linewidths=2.0, zorder=5)
    ax.set_xlabel(r"$w_{XA}$"); ax.set_ylabel(r"$w_{IA}$")
    ax.set_title("FCS strict Property 5", fontsize=10)
    ax.grid(alpha=0.3); ax.set_aspect("equal")

    # Panel 1: FCS measured period (continuous).
    ax = axes[1]
    period_plot = np.ma.masked_where(fcs_period == 0, fcs_period)
    im_p = ax.imshow(
        period_plot, origin="lower",
        cmap=plt.get_cmap("viridis", 12), vmin=1, vmax=12,
        extent=[W_XA[0] - 0.5, W_XA[-1] + 0.5,
                W_IA[0] - 0.5, W_IA[-1] + 0.5],
        aspect="auto",
    )
    cbar = plt.colorbar(im_p, ax=ax, ticks=range(1, 13))
    cbar.set_label("FCS-measured period (ticks)")
    ax.scatter([11], [-11], facecolors="none", edgecolors="gold",
               s=160, linewidths=2.0, zorder=5)
    ax.set_xlabel(r"$w_{XA}$"); ax.set_ylabel(r"$w_{IA}$")
    ax.set_title("FCS measured period", fontsize=10)

    # Panel 2: T_pred heatmap (clipped to [0, 30] for visibility).
    ax = axes[2]
    T_pred_clipped = np.where(np.isfinite(T_pred), np.clip(T_pred, 0, 30), np.nan)
    T_masked = np.ma.masked_invalid(T_pred_clipped)
    im_tp = ax.imshow(
        T_masked, origin="lower", cmap="magma",
        vmin=0, vmax=30,
        extent=[W_XA[0] - 0.5, W_XA[-1] + 0.5,
                W_IA[0] - 0.5, W_IA[-1] + 0.5],
        aspect="auto",
    )
    cbar = plt.colorbar(im_tp, ax=ax)
    cbar.set_label(r"$T_{\rm pred} = 2\pi / |\rm Im(\lambda)|$ (ticks)")
    ax.scatter([11], [-11], facecolors="none", edgecolors="white",
               s=160, linewidths=2.0, zorder=5)
    ax.set_xlabel(r"$w_{XA}$"); ax.set_ylabel(r"$w_{IA}$")
    ax.set_title(r"$T_{\rm pred}$ heatmap (clipped at 30)", fontsize=10)

    # Panel 3: period-4 blue label (boolean) overlay with FCS strict-P5.
    ax = axes[3]
    for i, w_IA in enumerate(W_IA):
        for j, w_XA in enumerate(W_XA):
            color = "tab:blue" if h_p4_blue[i, j] else "tab:red"
            ax.scatter(int(w_XA), int(w_IA), c=color, s=18, edgecolor="none")
    ax.scatter([11], [-11], facecolors="none", edgecolors="gold",
               s=160, linewidths=2.0, zorder=5)
    ax.set_xlabel(r"$w_{XA}$"); ax.set_ylabel(r"$w_{IA}$")
    ax.set_title(
        f"H(ω) period-4 gate (tol={TOL_DEFAULT})\n"
        f"{int(h_p4_blue.sum())}/{h_p4_blue.size} blue, "
        f"J(strict)={j_default:.3f}",
        fontsize=10,
    )
    ax.grid(alpha=0.3); ax.set_aspect("equal")

    fig.suptitle("Phase 2: H(ω) predicted ringing period vs FCS oracle",
                 fontsize=11)
    plt.tight_layout()
    out_pdf = RESULTS / "period_predicted_vs_4.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\n  wrote {out_pdf}")
    plt.close(fig)

    # Scatter: T_pred vs FCS measured period.
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    valid = np.isfinite(T_pred) & (fcs_period > 0)
    xs = fcs_period[valid]
    ys = T_pred[valid]
    is_p5 = strict_p5[valid]
    # Slight horizontal jitter so dots don't all stack.
    rng = np.random.default_rng(0)
    xj = xs + rng.uniform(-0.15, 0.15, size=len(xs))
    ax.scatter(xj[~is_p5], ys[~is_p5], c="lightgray", s=8, alpha=0.5,
               label="cells (any FCS period)")
    ax.scatter(xj[is_p5], ys[is_p5], c="tab:blue", s=14, alpha=0.85,
               label="FCS strict P5")
    ax.plot([0, 13], [0, 13], "k--", alpha=0.4, label="y = x (perfect)")
    ax.plot([0, 13], [0, 4 * 13], "tab:orange", alpha=0.6,
            linestyle="--", label="y = 4x (~observed slope)")
    ax.axhline(4, color="tab:green", linestyle=":", alpha=0.7,
               label="target T_pred = 4")
    ax.set_xlim(0.5, 12.5); ax.set_ylim(0, 30)
    ax.set_xlabel("FCS-measured period of A (ticks)")
    ax.set_ylabel(r"$T_{\rm pred} = 2\pi / |\rm Im(\lambda)|$ (ticks)")
    ax.set_title("Phase 2: predicted vs measured period across the grid",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_pdf = RESULTS / "T_pred_vs_FCS_period.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    plt.close(fig)

    # Tolerance sweep curve.
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.plot(tols, j_curve, "o-", color="tab:purple", markersize=3)
    ax.axvline(TOL_DEFAULT, color="gray", linestyle="--",
               alpha=0.5, label=f"tol = {TOL_DEFAULT}")
    ax.axvline(best_tol, color="tab:green", linestyle="--",
               alpha=0.5, label=f"best tol = {best_tol:.2f}")
    ax.axhline(best_j, color="tab:green", linestyle=":",
               alpha=0.5, label=f"best J = {best_j:.3f}")
    ax.set_xlabel("|T_pred − 4| tolerance (ticks)")
    ax.set_ylabel("Jaccard vs FCS strict P5")
    ax.set_title("Phase 2: Jaccard vs tolerance sweep", fontsize=10)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out_pdf = RESULTS / "tol_sweep.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    plt.close(fig)

    print()
    print("Phase 2 verdict:")
    if abs(T_def / 4 - 1) < 0.3:
        print(f"  PASS: default-cell T_pred ({T_def:.1f}) matches FCS period 4 "
              "within 30%.")
    else:
        print(f"  Default-cell T_pred ({T_def:.1f}) is "
              f"{T_def/4:.1f}× the FCS measured period 4.")
        print(f"  Single-pole H(ω) systematically over-estimates the ringing")
        print(f"  period — likely because the FCS 5-tap windowed integrator")
        print(f"  introduces faster effective dynamics than a single-pole")
        print(f"  low-pass with τ_m=2.35 captures. Best tolerance {best_tol:.2f}")
        print(f"  gives Jaccard {best_j:.3f}, vs strict-tol Jaccard "
              f"{j_default:.3f}.")


if __name__ == "__main__":
    main()
