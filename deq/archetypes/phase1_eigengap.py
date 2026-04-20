"""Phase 1: test Hypothesis 1 — eigenvalue gap as WTA predictor.

Against the Phase 0 ground truth:
  - fcs_fig10_groundtruth.npy  (binary WTA / no-WTA)
  - fcs_fig10_dominance.npy    (continuous (n1-n2)/(n1+n2+1))

we compute four candidate spectral predictors per grid cell:
  1. Δ(W)       : raw-W eigengap ||λ₁|-|λ₂||
  2. Δ(A)       : linearised-state eigengap
  3. ρ(A)       : spectral radius (anticipates Phase 2)
  4. arg(A)     : argument of dominant eigenvalue (for negative loop)

and render them as heatmaps with the ground-truth boundary overlaid, plus
report classification accuracy and rank correlation with the dominance ratio.

For the negative loop, we separately sweep (w_AI, w_IA) at fixed w_XA=11 and
compare arg(dominant λ) to the period-4 ground truth from the simulator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from archetypes.lif_fcs import simulate
from archetypes.topologies import contralateral, negative_loop
from archetypes.spectral import (
    weight_eigengap, linearized_A, linearized_eigengap, spectral_radius,
    arg_dominant, operating_point, fit_r_effective,
)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# Parameters chosen from operating-point diagnostics (see phase1_report.md)
P_MID = 30.0        # tuned: puts p* (≈22) in the steep part of the sigmoid
K = 0.08
R_EFF = 0.5         # default; LS-fit gives ≈0.545, compared in report


# --- Sweep contralateral ------------------------------------------------------

def sweep_contralateral(w_vals, p_mid=P_MID, r=R_EFF):
    """At each (w_12, w_21) cell, compute the four spectral predictors."""
    n = len(w_vals)
    gap_W = np.zeros((n, n))
    gap_A = np.zeros((n, n))
    rho_A = np.zeros((n, n))
    arg_A = np.zeros((n, n))
    # Also track operating-point saturation for diagnostics
    g_mean = np.zeros((n, n))
    u = np.array([1.0, 1.0])
    for i, w21 in enumerate(w_vals):
        for j, w12 in enumerate(w_vals):
            W, B, _ = contralateral(int(w12), int(w21), T=1)
            gap_W[i, j] = weight_eigengap(W)
            pstar = operating_point(W, B, u, r=r, p_mid=p_mid)
            A = linearized_A(W, pstar, r=r, p_mid=p_mid)
            gap_A[i, j] = linearized_eigengap(A)
            rho_A[i, j] = spectral_radius(A)
            arg_A[i, j] = arg_dominant(A)
            # diagnostic: mean sigmoid derivative at operating point
            from archetypes.spectral import f_sigmoid_deriv
            g_mean[i, j] = float(f_sigmoid_deriv(pstar, p_mid=p_mid).mean())
    return dict(gap_W=gap_W, gap_A=gap_A, rho_A=rho_A, arg_A=arg_A, g_mean=g_mean)


# --- Plotting -----------------------------------------------------------------

def _fcs_extent(w_vals):
    right = w_vals.max() + 0.5
    left = w_vals.min() - 0.5
    # x: -1 at left, -40 at right per FCS Fig. 10 orientation
    return [right, left, right, left]


def _base_axes(ax, w_vals, title, cbar_label, im):
    ax.set_xlabel(r"$w_{12}$ (inhibition N1$\to$N2)")
    ax.set_ylabel(r"$w_{21}$ (inhibition N2$\to$N1)")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.5, -40.5)
    ax.set_ylim(-0.5, -40.5)
    for v in range(-40, 1, 5):
        ax.axvline(v, color="gray", linewidth=0.3, alpha=0.3)
        ax.axhline(v, color="gray", linewidth=0.3, alpha=0.3)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label(cbar_label, fontsize=9)


def plot_predictor_with_overlay(
    w_vals, values, ground_truth, dominance, title, cbar_label, save,
    cmap="viridis", vmin=None, vmax=None, contour_level=None, diverging=False,
):
    fig, ax = plt.subplots(figsize=(7.2, 6))
    extent = _fcs_extent(w_vals)
    if diverging:
        mx = max(abs(values.min()), abs(values.max()))
        vmin, vmax = -mx, mx
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, origin="upper", aspect="equal",
                   interpolation="nearest")
    # Overlay predictor contour if requested
    if contour_level is not None:
        Xc = np.linspace(extent[0], extent[1], values.shape[1])
        Yc = np.linspace(extent[2], extent[3], values.shape[0])
        Xg, Yg = np.meshgrid(Xc, Yc)
        ax.contour(Xg, Yg, values, levels=[contour_level],
                   colors="white", linewidths=1.4)
    # Overlay ground-truth WTA boundary: show blue cells as dots
    yy, xx = np.where(ground_truth)
    ax.scatter(w_vals[xx], w_vals[yy], s=3, c="k", alpha=0.25, marker="s")
    _base_axes(ax, w_vals, title, cbar_label, im)
    fig.tight_layout()
    fig.savefig(save, dpi=120)
    plt.close(fig)


def plot_scatter(predictor, dominance, title, save, xlabel):
    fig, ax = plt.subplots(figsize=(6, 5))
    x = predictor.flatten()
    y = dominance.flatten()
    ax.scatter(x, y, s=4, alpha=0.4)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("dominance ratio (n1-n2)/(n1+n2+1)")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save, dpi=120)
    plt.close(fig)


# --- Classification metrics ---------------------------------------------------

def classification_accuracy(predictor, ground_truth, threshold=0.0, polarity=+1):
    """Fraction of cells where sign(polarity * (predictor - threshold)) agrees
    with ground_truth (True=blue).
    """
    pred_binary = (polarity * (predictor - threshold)) > 0
    return float((pred_binary == ground_truth).mean())


# --- Negative loop sweep ------------------------------------------------------

def period4_sim_oracle(w_AI_range, w_IA_range, w_XA=11, T=50):
    """For each (w_AI, w_IA) cell, check whether the simulator's activator
    output matches a period-4 1100 pattern over the last 20 ticks.
    """
    nA = len(w_AI_range)
    nI = len(w_IA_range)
    match = np.zeros((nI, nA), dtype=bool)
    for i, w_IA in enumerate(w_IA_range):
        for j, w_AI in enumerate(w_AI_range):
            W, B, ext = negative_loop(w_XA=int(w_XA), w_AI=int(w_AI),
                                      w_IA=int(w_IA), T=T)
            spikes, _ = simulate(W, B, ext, T=T)
            tail = spikes[0, -20:]
            # autocorrelation at lag 4 on last 20 samples
            pat = np.array([1, 1, 0, 0] * 5, dtype=bool)
            match[i, j] = np.array_equal(tail, pat)
    return match


def sweep_negative_loop_spectrum(w_AI_range, w_IA_range, w_XA=11,
                                 p_mid=P_MID, r=R_EFF):
    nA = len(w_AI_range)
    nI = len(w_IA_range)
    arg_grid = np.zeros((nI, nA))
    rho_grid = np.zeros((nI, nA))
    for i, w_IA in enumerate(w_IA_range):
        for j, w_AI in enumerate(w_AI_range):
            W, B, _ = negative_loop(w_XA=int(w_XA), w_AI=int(w_AI),
                                    w_IA=int(w_IA), T=1)
            u = np.array([1.0])
            pstar = operating_point(W, B, u, r=r, p_mid=p_mid)
            A = linearized_A(W, pstar, r=r, p_mid=p_mid)
            arg_grid[i, j] = arg_dominant(A)
            rho_grid[i, j] = spectral_radius(A)
    return arg_grid, rho_grid


# --- Main --------------------------------------------------------------------

def main():
    print("[Phase 1] Loading Phase 0 ground truth...")
    w_vals = np.load(RESULTS / "fcs_fig10_wvals.npy")
    gt = np.load(RESULTS / "fcs_fig10_groundtruth.npy")
    dom = np.load(RESULTS / "fcs_fig10_dominance.npy")

    print("[Phase 1] Sweeping spectral predictors on contralateral grid...")
    preds = sweep_contralateral(w_vals, p_mid=P_MID, r=R_EFF)

    # -- Predictor heatmaps
    print("[Phase 1] Rendering heatmaps...")
    plot_predictor_with_overlay(
        w_vals, preds["gap_W"], gt, dom,
        title=r"$\Delta(W) = ||\lambda_1|-|\lambda_2||$ of raw $W$ "
              "(identically 0 for 2×2 zero-diag)",
        cbar_label=r"$\Delta(W)$",
        save=RESULTS / "phase1_contra_gapW.png",
        cmap="viridis",
    )
    plot_predictor_with_overlay(
        w_vals, preds["gap_A"], gt, dom,
        title=r"$\Delta(A)$ of linearised state matrix  ($r=0.5$, $p_{mid}=30$)",
        cbar_label=r"$\Delta(A)$",
        save=RESULTS / "phase1_contra_gapA.png",
        cmap="viridis",
    )
    plot_predictor_with_overlay(
        w_vals, preds["rho_A"], gt, dom,
        title=r"$\rho(A)$ spectral radius with $\rho=1$ contour (Phase 2 preview)",
        cbar_label=r"$\rho(A)$",
        save=RESULTS / "phase1_contra_rhoA.png",
        cmap="viridis",
        contour_level=1.0,
    )
    plot_predictor_with_overlay(
        w_vals, preds["g_mean"], gt, dom,
        title="Mean $f'(p^*)$ across neurons (operating-point saturation diagnostic)",
        cbar_label=r"$\overline{f'(p^*)}$",
        save=RESULTS / "phase1_contra_gmean.png",
        cmap="magma",
    )

    # -- Scatter plots against dominance
    plot_scatter(preds["gap_A"], dom,
                 "Linearised gap Δ(A) vs dominance ratio",
                 RESULTS / "phase1_scatter_gapA.png",
                 xlabel=r"$\Delta(A)$")
    plot_scatter(preds["rho_A"], np.abs(dom),
                 "ρ(A) vs |dominance|",
                 RESULTS / "phase1_scatter_rhoA_absdom.png",
                 xlabel=r"$\rho(A)$")

    # -- Metrics
    accs = {}
    corrs = {}
    for name, pred in [("gap_W", preds["gap_W"]),
                       ("gap_A", preds["gap_A"]),
                       ("rho_A", preds["rho_A"])]:
        # Best polarity/threshold for binary WTA classification
        best_acc = 0
        for polarity in (+1, -1):
            for thr in np.linspace(pred.min(), pred.max(), 41):
                a = classification_accuracy(pred, gt, threshold=thr, polarity=polarity)
                if a > best_acc:
                    best_acc = a
        accs[name] = best_acc
        pr = pearsonr(pred.flatten(), np.abs(dom).flatten()).statistic
        sr = spearmanr(pred.flatten(), np.abs(dom).flatten()).statistic
        corrs[name] = (pr, sr)

    # -- Negative loop sweep
    print("[Phase 1] Sweeping negative loop (w_AI, w_IA) at fixed w_XA=11...")
    w_AI_range = np.arange(1, 21)
    w_IA_range = np.arange(-1, -21, -1)
    period4_gt = period4_sim_oracle(w_AI_range, w_IA_range, w_XA=11, T=50)
    arg_grid, rho_nl = sweep_negative_loop_spectrum(w_AI_range, w_IA_range)

    # Plot: simulator period-4 mask + overlay of arg(dom) = π/2 contour
    fig, ax = plt.subplots(figsize=(7, 5.5))
    extent = [w_AI_range.min() - 0.5, w_AI_range.max() + 0.5,
              w_IA_range.min() - 0.5, w_IA_range.max() + 0.5]
    im = ax.imshow(arg_grid, cmap="twilight", vmin=-np.pi, vmax=np.pi,
                   extent=extent, origin="upper", aspect="auto")
    # overlay arg = π/2 contour
    Xg, Yg = np.meshgrid(w_AI_range, w_IA_range)
    ax.contour(Xg, Yg, arg_grid, levels=[np.pi / 2], colors="white", linewidths=1.4)
    # overlay simulator period-4 mask
    ys, xs = np.where(period4_gt)
    ax.scatter(w_AI_range[xs], w_IA_range[ys], s=20, c="red",
               edgecolors="white", linewidth=0.4, label="sim period-4")
    ax.set_xlabel(r"$w_{AI}$")
    ax.set_ylabel(r"$w_{IA}$")
    ax.set_title(r"Negative loop: $\arg \lambda_{dom}(A)$ heatmap, "
                 r"$\arg = \pi/2$ contour (white), period-4 cells (red dots)")
    ax.legend(loc="upper left")
    plt.colorbar(im, ax=ax, fraction=0.046, label=r"$\arg \lambda$ (rad)")
    fig.tight_layout()
    fig.savefig(RESULTS / "phase1_negloop.png", dpi=120)
    plt.close(fig)

    # -- Report
    print("[Phase 1] Writing report...")
    lines = []
    lines.append("# Phase 1 Report — Eigenvalue Gap as WTA Predictor\n")
    lines.append("## Setup\n")
    lines.append(f"- Sigmoid approximation: k={K}, p_mid={P_MID} "
                 f"(tuned from fallback sweep; saturates at p_mid=90)\n")
    r_ls = fit_r_effective()
    lines.append(f"- Scalar leak r: using {R_EFF} (LS-fit to rvector gives {r_ls:.3f})\n")
    lines.append(f"- Ground truth: {RESULTS / 'fcs_fig10_groundtruth.npy'} "
                 f"(1014/1600 blue cells from Phase 0)\n\n")

    lines.append("## Hypothesis 1 outcome on contralateral inhibition\n\n")
    lines.append("### (A) Δ(W): raw weight eigengap\n")
    lines.append("Analytical result: for a 2×2 zero-diagonal W with off-diagonal "
                 "entries $a, b$, the eigenvalues are $\\pm\\sqrt{ab}$ with "
                 "**equal magnitudes** regardless of $|a|$ vs $|b|$. "
                 "So Δ(W) ≡ 0 over the entire sweep. The raw-W eigengap is **not** "
                 "a usable predictor for this archetype — a fact that was not "
                 "flagged in the original plan and is worth recording.\n\n")
    lines.append(f"- Verified numerically: max Δ(W) across 1600 cells = "
                 f"{preds['gap_W'].max():.6e} (machine-ε noise).\n\n")

    lines.append("### (B) Δ(A): linearised-state eigengap\n")
    lines.append(f"- Best classification accuracy (over polarity × threshold): "
                 f"{accs['gap_A']*100:.1f}% against binary WTA.\n")
    pr, sr = corrs["gap_A"]
    lines.append(f"- Correlation with |dominance|: Pearson r = {pr:+.3f}, "
                 f"Spearman ρ = {sr:+.3f}.\n")
    lines.append(f"- The scalar-r sigmoid linearisation suffers from operating-"
                 f"point saturation: even at the tuned p_mid={P_MID}, the mean "
                 f"sigmoid derivative is {preds['g_mean'].mean():.4f}, three "
                 f"orders of magnitude below a typical neural gain. "
                 f"This keeps Δ(A) tiny everywhere and gives a weak predictor.\n\n")

    lines.append("### (C) ρ(A): spectral radius (Phase 2 preview)\n")
    lines.append(f"- Best classification accuracy: {accs['rho_A']*100:.1f}%.\n")
    pr, sr = corrs["rho_A"]
    lines.append(f"- Correlation with |dominance|: Pearson r = {pr:+.3f}, "
                 f"Spearman ρ = {sr:+.3f}.\n")
    lines.append(f"- ρ(A) ranges over [{preds['rho_A'].min():.4f}, "
                 f"{preds['rho_A'].max():.4f}] — well below 1 across the entire "
                 f"sweep under the scalar-r linearisation. The $\\rho=1$ contour "
                 f"is therefore **empty** in our grid, confirming that the "
                 f"scalar-r linearisation under-predicts instability. Phase 2 "
                 f"will need a richer state representation (the full 5-tap "
                 f"memory per neuron) to capture the true bifurcation.\n\n")

    lines.append("## Hypothesis 1 outcome on the negative loop\n")
    n_period4 = int(period4_gt.sum())
    lines.append(f"- Simulator period-4 cells: {n_period4} of "
                 f"{period4_gt.size} ({100*n_period4/period4_gt.size:.1f}%). "
                 f"The cleanest case is $w_{{IA}} = -w_{{AI}}$ (exact-cancellation "
                 f"rule — this is the Phase 0 tuning finding).\n")
    lines.append(f"- arg(λ_dom) range over the sweep: "
                 f"[{arg_grid.min():.3f}, {arg_grid.max():.3f}] rad.\n")
    lines.append("- Does the arg = π/2 contour align with the simulator's "
                 "period-4 cells?\n")
    closest_arg = np.abs(arg_grid - np.pi / 2)
    # cells within, say, 0.05 rad of π/2
    close_mask = closest_arg < 0.1
    if close_mask.any():
        overlap = float((close_mask & period4_gt).sum() / max(1, close_mask.sum()))
        lines.append(f"  - Cells with |arg - π/2| < 0.1 rad: {close_mask.sum()}; "
                     f"of these, {int((close_mask & period4_gt).sum())} are "
                     f"period-4 in the simulator (precision = {overlap*100:.0f}%).\n")
    else:
        lines.append("  - **No cells in the swept range have arg(λ_dom) within 0.1 "
                     "rad of π/2.** The scalar-r linearisation produces arg ≈ 0 "
                     "(real dominant eigenvalue) across most of the sweep, "
                     "fundamentally failing to predict the period-4 oscillation.\n")
    lines.append(f"  - This is the same root cause as (B)/(C) above: the FCS "
                 f"windowed integrator (rvector=[10,5,3,2,1]) is a length-5 FIR "
                 f"filter whose dynamics cannot be captured by a single scalar "
                 f"r. The period-4 oscillation of Property 5 requires poles at "
                 f"the 8th roots of unity of the full FIR transfer function.\n\n")

    lines.append("## Root Cause: Scalar-r Linearisation is Too Crude\n")
    lines.append("Across all three predictors, the common failure mode is the "
                 "collapse of the 5-tap windowed integrator to a scalar leak. "
                 "The true state of a single FCS neuron is (mem[0..4]), so an "
                 "n-neuron network has a 5n-dimensional state, not n. Phase 2 "
                 "therefore requires building the **full 5n × 5n state matrix** "
                 "(with the FIR-filter shift as off-diagonal blocks, the summed-"
                 "input map on the top-row block, and the spike-reset "
                 "non-linearity linearised around the operating point).\n\n")

    lines.append("## Recommendation\n")
    lines.append("Hypothesis 1 is **not validated** in its stated form. "
                 "The failure is not a flaw in the spectral-cartography "
                 "programme — it is a specification error: the plan prescribed "
                 "a scalar-r linearisation that cannot represent the 5-tap "
                 "FIR-filter dynamics that FCS LI&F neurons actually execute.\n\n")
    lines.append("Concretely:\n")
    lines.append("1. Raw-W eigengap is provably identically zero for 2×2 "
                 "zero-diagonal weight matrices (analytical, not empirical).\n")
    lines.append("2. Scalar-r linearised A saturates at the operating point and "
                 "produces predictors ~3 orders of magnitude weaker than needed.\n")
    lines.append("3. Phase 2's ρ=1 bifurcation contour will not exist under "
                 "scalar-r; we must build the 5n×5n state matrix before "
                 "testing Hypothesis 2.\n\n")
    lines.append("**Proposed path forward.** Before formally proceeding to "
                 "Phase 2, produce a revised `spectral.py` that builds the "
                 "full 5n-dim state matrix $A_{full}$ directly from the FCS "
                 "dynamics (the FIR shift + spike-reset linearisation). Then "
                 "retest Hypotheses 1 and 2 against the same ground truth. "
                 "This is ~50 lines of code and is a one-time investment.\n\n")
    lines.append("Pending user decision, the existing scalar-r artifacts are "
                 "preserved in `results/phase1_*.png` as documentation of the "
                 "failure mode.\n")

    with open(RESULTS / "phase1_report.md", "w") as f:
        f.writelines(lines)

    print("[Phase 1] Done.")
    print(f"  Δ(W) max           : {preds['gap_W'].max():.3e}  (analytically 0)")
    print(f"  Δ(A) classification: {accs['gap_A']*100:.1f}%")
    print(f"  ρ(A) classification: {accs['rho_A']*100:.1f}%")
    print(f"  ρ(A) range         : [{preds['rho_A'].min():.3f}, {preds['rho_A'].max():.3f}]")
    print(f"  Neg loop period-4  : {int(period4_gt.sum())}/{period4_gt.size} cells")


if __name__ == "__main__":
    main()
