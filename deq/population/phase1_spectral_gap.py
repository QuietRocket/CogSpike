"""Phase 1 - Spectral gap as behavioral proxy (Hypothesis A).

Sweeps the contralateral-inhibition archetype on a 50x50 grid of
(w12, w21) in [0, 5]^2 under constant drive I = [1.5, 1.5]. For each grid
cell we compute:

  1. The symmetric fixed point (from guess [0.5, 0.5]).
  2. The dominant-eigenvalue real part at that fixed point -- the spectral
     predictor of WTA onset.
  3. The simulation-based WTA verdict: starting from two mirror-image
     perturbations of the symmetric state, does the system commit to one
     population winning in both runs?

We then overlay the analytical pitchfork curve computed by fine-grained
bisection against the empirical WTA boundary, report classification
accuracy, and emit phase1_report.pdf with a PASS / FAIL verdict.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# Unbuffered stdout so progress lines appear promptly through tee / pipes.
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=sys.stdout.encoding, line_buffering=True)
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bifurcation import pitchfork_curve_contralateral, pitchfork_diagonal  # noqa: E402
from ground_truth import wta_contralateral  # noqa: E402
from linearization import jacobian, spectrum  # noqa: E402
from topologies import contralateral_inhibition  # noqa: E402
from wilson_cowan import Sigmoid, find_saddle_contralateral  # noqa: E402

SEED = 20260420
np.random.seed(SEED)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
FIG_DIR = RESULTS / "phase1"
FIG_DIR.mkdir(exist_ok=True)

# --- Parameter sweep spec -----------------------------------------------------
DRIVE = 1.5
TAU = 1.0
W_MIN, W_MAX = 0.0, 5.0
N_GRID = 50  # -> Delta w = 0.1

# --- Ground-truth WTA knobs ---------------------------------------------------
T_FINAL = 50.0
PERTURB = 0.05
MARGIN = 0.3


def banner(msg: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{msg}\n{bar}")


def save_fig(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")


def run_sweep(sigmoid: Sigmoid):
    w_grid = np.linspace(W_MIN, W_MAX, N_GRID)
    dom_real = np.full((N_GRID, N_GRID), np.nan)
    gap = np.full((N_GRID, N_GRID), np.nan)
    wta = np.zeros((N_GRID, N_GRID), dtype=bool)
    fp_found = np.zeros((N_GRID, N_GRID), dtype=bool)

    n_fp_grid = np.zeros((N_GRID, N_GRID), dtype=int)
    t0 = time.time()
    for i, w12 in enumerate(w_grid):
        for j, w21 in enumerate(w_grid):
            W, I = contralateral_inhibition(float(w12), float(w21), drive=DRIVE)

            # Use the middle-branch (saddle) fixed point: past the
            # pitchfork fsolve from (0.5, 0.5) drifts to an asymmetric
            # stable fixed point. The saddle is what carries the stability
            # information relevant to WTA onset.
            rho_star, n_fp = find_saddle_contralateral(float(w12), float(w21), DRIVE, sigmoid)
            n_fp_grid[i, j] = n_fp
            fp_found[i, j] = n_fp > 0 and not np.any(np.isnan(rho_star))
            if not fp_found[i, j]:
                continue

            J = jacobian(W, I, rho_star, TAU, sigmoid)
            eigvals, _ = spectrum(J)
            dom_real[i, j] = float(eigvals[0].real)
            if eigvals.size >= 2:
                gap[i, j] = float(eigvals[0].real - eigvals[1].real)

            verdict, _ = wta_contralateral(
                float(w12), float(w21), DRIVE, TAU, sigmoid,
                t_final=T_FINAL, perturbation=PERTURB, margin=MARGIN,
            )
            wta[i, j] = verdict

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  row {i + 1}/{N_GRID} in {elapsed:.1f}s", flush=True)

    elapsed = time.time() - t0
    print(f"Sweep complete in {elapsed:.1f}s", flush=True)
    return w_grid, dom_real, gap, wta, fp_found, n_fp_grid


def compute_boundary_cells(wta: np.ndarray) -> np.ndarray:
    """Return boolean mask of cells adjacent to a WTA / non-WTA transition."""
    n, m = wta.shape
    boundary = np.zeros_like(wta, dtype=bool)
    for i in range(n):
        for j in range(m):
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < m:
                    if wta[i, j] != wta[ni, nj]:
                        boundary[i, j] = True
                        break
    return boundary


def curve_to_grid_distance(
    curve_w12: np.ndarray, curve_w21: np.ndarray, boundary_pts: np.ndarray
) -> np.ndarray:
    """For each empirical boundary cell (w12, w21), min distance to the curve."""
    dists = np.zeros(len(boundary_pts))
    for k, (w12, w21) in enumerate(boundary_pts):
        d = np.sqrt((curve_w12 - w12) ** 2 + (curve_w21 - w21) ** 2)
        dists[k] = d.min()
    return dists


def make_plots(
    w_grid: np.ndarray,
    dom_real: np.ndarray,
    gap: np.ndarray,
    wta: np.ndarray,
    curve_w12: np.ndarray,
    curve_w21: np.ndarray,
    w_diag: float,
):
    extent = (w_grid[0], w_grid[-1], w_grid[0], w_grid[-1])

    # --- Panel (a) ground truth WTA map
    fig_a, ax_a = plt.subplots(figsize=(4.5, 4))
    # wta[i, j] corresponds to w12 = w_grid[i] (row) and w21 = w_grid[j] (col).
    # imshow expects array[row=y, col=x]. We plot with w12 on x and w21 on y:
    ax_a.imshow(
        wta.T,
        origin="lower",
        extent=extent,
        cmap="Greys",
        vmin=0,
        vmax=1,
        aspect="equal",
        interpolation="nearest",
    )
    ax_a.set_xlabel(r"$w_{12}$")
    ax_a.set_ylabel(r"$w_{21}$")
    ax_a.set_title("(a) Empirical WTA (black = WTA)")
    save_fig(fig_a, "panel_a_wta_ground_truth")
    plt.close(fig_a)

    # --- Panel (b) spectral gap heatmap with zero-contour of dominant real part
    fig_b, ax_b = plt.subplots(figsize=(5, 4))
    im = ax_b.imshow(
        gap.T,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        aspect="equal",
        interpolation="nearest",
    )
    cs = ax_b.contour(
        w_grid,
        w_grid,
        dom_real.T,
        levels=[0.0],
        colors="black",
        linewidths=1.5,
    )
    ax_b.clabel(cs, fmt={0.0: r"$\mathrm{Re}(\lambda_{\max}) = 0$"}, inline=True, fontsize=8)
    ax_b.set_xlabel(r"$w_{12}$")
    ax_b.set_ylabel(r"$w_{21}$")
    ax_b.set_title(r"(b) Spectral gap $\Delta = \mathrm{Re}(\lambda_1 - \lambda_2)$")
    fig_b.colorbar(im, ax=ax_b, label=r"$\Delta$")
    save_fig(fig_b, "panel_b_spectral_gap")
    plt.close(fig_b)

    # --- Panel (c) analytical bifurcation curve over ground truth
    fig_c, ax_c = plt.subplots(figsize=(4.5, 4))
    ax_c.imshow(
        wta.T,
        origin="lower",
        extent=extent,
        cmap="Greys",
        vmin=0,
        vmax=1,
        aspect="equal",
        interpolation="nearest",
        alpha=0.6,
    )
    ax_c.plot(curve_w12, curve_w21, "r-", linewidth=2, label="analytical curve")
    ax_c.plot(w_diag, w_diag, "yo", markersize=8, label=f"diagonal $w^*={w_diag:.3f}$")
    ax_c.set_xlim(W_MIN, W_MAX)
    ax_c.set_ylim(W_MIN, W_MAX)
    ax_c.set_xlabel(r"$w_{12}$")
    ax_c.set_ylabel(r"$w_{21}$")
    ax_c.set_title("(c) Analytical curve vs empirical WTA")
    ax_c.legend(loc="lower right", framealpha=0.9)
    save_fig(fig_c, "panel_c_curve_overlay")
    plt.close(fig_c)


def render_report(
    accuracy: float,
    w_diag: float,
    rho_at_diag: float,
    g_at_diag: float,
    median_boundary_err: float,
    max_boundary_err: float,
    grid_spacing: float,
    gap_crosses_zero_at_boundary: bool,
    overall_pass: bool,
) -> None:
    verdict = "PASS" if overall_pass else "FAIL"
    typ = HERE / "phase1_report.typ"
    pdf = HERE / "phase1_report.pdf"
    content = f"""#set page(paper: "a4", margin: 2cm)
#set text(size: 10pt)
#set par(justify: true)

#align(center)[
  #text(size: 14pt, weight: "bold")[Phase 1 report -- Spectral gap as behavioral proxy]
  #v(0.2em)
  Verdict: *{verdict}*
]

= Setup

Contralateral inhibition archetype with drive $I = [1.5, 1.5]$, time
constant $tau = 1$, and sigmoid $f(x) = 1 / (1 + exp(-4(x-1)))$. The
weight grid is $w_(12), w_(21) in [0, 5]$ with a 50 times 50 mesh
($Delta w = 0.1$).

For each cell, three quantities are computed:

- the middle-branch fixed point of the scalar reduction
  $rho_1 = f(I - w_(21) f(I - w_(12) rho_1))$, i.e. the analytical
  analogue of the symmetric saddle in the asymmetric 2D system,
- the dominant real part $"Re"(lambda_1)$ of the Jacobian at that
  fixed point and the spectral gap $Delta = "Re"(lambda_1) - "Re"(lambda_2)$,
- a simulation-based WTA verdict over $t in [0, 50]$, starting from
  the mirror-image perturbations $rho^* plus.minus (0.05, -0.05)$, with
  WTA declared when both runs end with $|rho_1 - rho_2| > 0.3$ *and*
  with opposite signs of $rho_1 - rho_2$. The sign-opposition clause
  tightens the plan's stated criterion (plan §3.3), which by its bare
  letter would also flag strongly skew single-FP regimes as WTA; the
  parenthetical in the same section ('the system commits to one
  population dominating [which population depends on the initial
  condition]') makes clear that bistable choice is the intended meaning,
  and the tightening is the faithful reading.

= Closed-form diagonal result

On the diagonal $w_(12) = w_(21) = w$, the linearization of the WC
system at the symmetric fixed point reduces to a matrix with off-diagonal
entries $-w g$, where $g = f'(I - w rho^*)$ is the sigmoid slope at the
fixed-point input. Eigenvalues are $(-1 plus.minus w g) slash tau$ and
the pitchfork condition $w g = 1$ yields

$ w^* = {w_diag:.6f}, quad rho^*(w^*) = {rho_at_diag:.6f}, quad g(w^*) = {g_at_diag:.6f}. $

= Classification accuracy

Binarising the spectral predictor at the sign of $"Re"(lambda_1)$ (WTA
when the dominant real part is positive) and comparing cell-by-cell
against the simulation-based ground truth yields an accuracy of
*{accuracy*100:.2f} %* (acceptance threshold: $gt.eq$ 95 %).

The analytical bifurcation curve was traced by bisection along radial
slices in the positive quadrant (181 angles); its distance from the
empirical boundary cells is summarized by

- median: {median_boundary_err:.4f} weight units ($= {median_boundary_err/grid_spacing:.2f}$ grid cells),
- maximum: {max_boundary_err:.4f} weight units ($= {max_boundary_err/grid_spacing:.2f}$ grid cells).

The plan's acceptance threshold is within one grid cell
($Delta w = {grid_spacing}$) at every boundary point.

The spectral gap changes sign (i.e. $"Re"(lambda_1)$ crosses zero)
precisely at the empirical WTA boundary:
{"*yes*" if gap_crosses_zero_at_boundary else "*NO*"}.

= Figures

#figure(image("results/phase1/panel_a_wta_ground_truth.pdf", width: 70%),
  caption: [Panel (a). Empirical WTA map on the 50 times 50 weight grid.
  Black cells commit to winner-take-all from both symmetry-breaking
  initial conditions within $t = 50$; white cells remain symmetric.])

#figure(image("results/phase1/panel_b_spectral_gap.pdf", width: 80%),
  caption: [Panel (b). Spectral gap $Delta = "Re"(lambda_1) - "Re"(lambda_2)$
  at the symmetric fixed point. The black contour marks the zero-level of
  the dominant real part, i.e. the linear-stability boundary.])

#figure(image("results/phase1/panel_c_curve_overlay.pdf", width: 70%),
  caption: [Panel (c). Fine-grained analytical bifurcation curve (red)
  obtained by bisection on the dominant eigenvalue, overlaid on the
  empirical WTA map (grey). The yellow marker locates the closed-form
  diagonal pitchfork point $w^* = w_(12) = w_(21)$.])

= Verdict

Accuracy $gt.eq$ 95 %: {"yes" if accuracy >= 0.95 else "NO"}.
Analytical curve within 1 grid cell at every boundary point:
{"yes" if max_boundary_err <= grid_spacing else "NO"} (max = {max_boundary_err/grid_spacing:.2f} cells).
Dominant real part crosses zero at the boundary:
{"yes" if gap_crosses_zero_at_boundary else "NO"}.

*Overall verdict: {verdict}.*
"""
    typ.write_text(content)
    subprocess.run(["typst", "compile", str(typ), str(pdf)], check=True, cwd=str(HERE))


def main() -> int:
    sigmoid = Sigmoid(k=4.0, theta=1.0)

    banner("Phase 1  spectral gap vs WTA on contralateral inhibition")
    w_grid, dom_real, gap, wta, fp_found, n_fp_grid = run_sweep(sigmoid)
    np.save(RESULTS / "n_fp_contralateral.npy", n_fp_grid)

    np.save(RESULTS / "ground_truth_contralateral.npy", wta)
    np.save(RESULTS / "spectral_gap_contralateral.npy", gap)
    np.save(RESULTS / "dominant_real_contralateral.npy", dom_real)
    np.save(RESULTS / "w_grid.npy", w_grid)

    # --- Classification accuracy: predict WTA iff dom_real > 0.
    predictor = dom_real > 0.0
    valid = fp_found & ~np.isnan(dom_real)
    accuracy = float(np.mean(predictor[valid] == wta[valid]))
    print(f"Classification accuracy: {accuracy*100:.2f}%")

    # --- Diagonal closed-form pitchfork.
    w_diag, rho_at_diag, g_at_diag = pitchfork_diagonal(DRIVE, TAU, sigmoid)
    print(f"Diagonal pitchfork: w* = {w_diag:.6f}, rho* = {rho_at_diag:.6f}, g = {g_at_diag:.6f}")

    # --- Fine-grained bifurcation curve via radial bisection.
    curve_w12, curve_w21 = pitchfork_curve_contralateral(DRIVE, TAU, sigmoid, w_max=W_MAX, n_angles=361)
    print(f"Bifurcation curve: {len(curve_w12)} points")
    np.save(RESULTS / "bifurcation_curve.npy", np.column_stack([curve_w12, curve_w21]))

    # --- Boundary displacement metric.
    boundary_mask = compute_boundary_cells(wta)
    boundary_i, boundary_j = np.where(boundary_mask)
    boundary_pts = np.column_stack([w_grid[boundary_i], w_grid[boundary_j]])
    if boundary_pts.size == 0:
        median_err = max_err = float("nan")
    else:
        dists = curve_to_grid_distance(curve_w12, curve_w21, boundary_pts)
        median_err = float(np.median(dists))
        max_err = float(dists.max())
    grid_spacing = float(w_grid[1] - w_grid[0])
    print(f"Boundary displacement: median {median_err:.4f} ({median_err/grid_spacing:.2f} cells), max {max_err:.4f} ({max_err/grid_spacing:.2f} cells)")

    # --- Spectral gap sign crossing at boundary: check that adjacent cells
    # across any transition bracket zero in dom_real.
    sign_crossing = True
    n = wta.shape[0]
    for i in range(n):
        for j in range(n):
            if not boundary_mask[i, j]:
                continue
            neighbours = []
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n and wta[i, j] != wta[ni, nj]:
                    neighbours.append((ni, nj))
            for ni, nj in neighbours:
                a, b = dom_real[i, j], dom_real[ni, nj]
                if np.isnan(a) or np.isnan(b):
                    continue
                if a * b > 0:  # same sign both sides of a WTA transition
                    sign_crossing = False
    print(f"Dominant-real sign changes at every boundary transition: {sign_crossing}")

    make_plots(w_grid, dom_real, gap, wta, curve_w12, curve_w21, w_diag)

    overall_pass = (
        accuracy >= 0.95
        and (max_err <= grid_spacing if not np.isnan(max_err) else False)
        and sign_crossing
    )

    render_report(
        accuracy=accuracy,
        w_diag=w_diag,
        rho_at_diag=rho_at_diag,
        g_at_diag=g_at_diag,
        median_boundary_err=median_err,
        max_boundary_err=max_err,
        grid_spacing=grid_spacing,
        gap_crosses_zero_at_boundary=sign_crossing,
        overall_pass=overall_pass,
    )

    banner(f"Phase 1 verdict: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
