"""Phase 0: simulator validation and FCS Fig. 10 ground-truth sweep.

Deliverables (all written into ./results/):
  phase0_property5_trace.png      — activator/inhibitor spike trains for neg loop
  phase0_property7_examples.png   — 4 contralateral traces at selected weights
  fcs_fig10_groundtruth.npy       — (40, 40) bool grid (A.7 WTA criterion)
  phase0_fig10_reproduction.png   — blue/red staircase (FCS Fig. 10 orientation)
  phase0_report.md                — text summary & sanity-check results

The simulator in lif_fcs.py is the authoritative oracle; everything in later
phases will be validated against its output.
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from archetypes.lif_fcs import simulate, spike_sequence_to_str
from archetypes.topologies import negative_loop, contralateral

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# --- Property 5 ---------------------------------------------------------------

def property5_sanity_check(T=30):
    """Verify the activator of the negative loop produces 0,1,1,0,0,1,1,... exactly."""
    W, B, ext = negative_loop(w_XA=11, w_AI=11, w_IA=-11, T=T)
    spikes, local = simulate(W, B, ext, T=T)
    got = spike_sequence_to_str(spikes[0])
    expected = "0" + "".join("1100"[i % 4] for i in range(T - 1))
    ok = got == expected
    return ok, got, expected, spikes, local


def plot_property5(spikes, save):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    T = spikes.shape[1]
    ticks = np.arange(T)
    # Render as raster
    for i, (name, row) in enumerate(zip(["Activator A", "Inhibitor I"], spikes)):
        ax.vlines(ticks[row], i + 0.1, i + 0.9, colors=["C0", "C3"][i], linewidth=2.5)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["A", "I"])
    ax.set_xlabel("tick t")
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(-0.1, 2.1)
    ax.set_title("FCS Property 5: negative loop (w_XA=11, w_AI=11, w_IA=-11)")
    ax.grid(axis="x", alpha=0.3)
    # Annotate expected pattern
    expected = "0" + "".join("1100"[i % 4] for i in range(T - 1))
    ax.text(
        0.01, 0.98,
        "Activator:  " + spike_sequence_to_str(spikes[0]) + "\n"
        "Expected:   " + expected,
        transform=ax.transAxes, va="top", ha="left",
        family="monospace", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    fig.tight_layout()
    fig.savefig(save, dpi=120)
    plt.close(fig)


# --- Property 7 / Fig. 10 -----------------------------------------------------

def wta_criterion(spikes, T=50):
    """Appendix A.7 criterion for 'blue' (WTA within 4 ticks).

    (a) By tick 4, N1 and N2 outputs differ.
    (b) In ticks 5..T-1, one neuron emits >=40 spikes and the other emits 0.
    """
    early = spikes[:, :5]
    if np.array_equal(early[0], early[1]):
        return False
    late = spikes[:, 5:T]
    counts = late.sum(axis=1)
    n1, n2 = counts
    expected_high = (T - 5) - 10  # allow some slack; A.7 says >=40 for T=50 (T-5=45)
    expected_high = max(expected_high, 40)
    if (n1 >= expected_high and n2 == 0) or (n2 >= expected_high and n1 == 0):
        return True
    return False


def fig10_sweep(T=50, w_range=range(-1, -41, -1)):
    """Sweep (w_12, w_21) on the FCS Fig. 10 integer grid.

    Returns:
      w_vals: ordered array of swept weight values.
      grid:   (n, n) bool array — True = WTA reached (blue) per A.7 criterion.
      count_diff: (n, n) int — (N1 spikes) - (N2 spikes) in ticks 5..T-1.
      dominance: (n, n) float — (n1 - n2) / (n1 + n2 + 1), in (-1, 1).
                 +1 = N1 fully dominates; -1 = N2 fully dominates; 0 = tied.

    Axes: rows indexed by w_21 (y-axis, FCS Fig. 10 convention), cols by w_12 (x-axis).
    """
    w_vals = np.array(list(w_range), dtype=np.int64)
    n = len(w_vals)
    grid = np.zeros((n, n), dtype=bool)
    count_diff = np.zeros((n, n), dtype=np.int64)
    dominance = np.zeros((n, n), dtype=np.float64)
    for i, w21 in enumerate(w_vals):
        for j, w12 in enumerate(w_vals):
            W, B, ext = contralateral(int(w12), int(w21), T=T)
            spikes, _ = simulate(W, B, ext, T=T)
            grid[i, j] = wta_criterion(spikes, T=T)
            n1 = int(spikes[0, 5:T].sum())
            n2 = int(spikes[1, 5:T].sum())
            count_diff[i, j] = n1 - n2
            dominance[i, j] = (n1 - n2) / (n1 + n2 + 1)
    return w_vals, grid, count_diff, dominance


def plot_dominance(w_vals, dominance, save):
    """Diverging heatmap of (n1 - n2) / (n1 + n2 + 1). Matches FCS Fig. 10 axes."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    right = w_vals.max() + 0.5
    left = w_vals.min() - 0.5
    vmax = float(np.abs(dominance).max()) or 1.0
    im = ax.imshow(
        dominance,
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        extent=[right, left, right, left],
        origin="upper",
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xlabel(r"$w_{12}$ (inhibition N1$\to$N2)")
    ax.set_ylabel(r"$w_{21}$ (inhibition N2$\to$N1)")
    ax.set_title("Dominance ratio $(n_1 - n_2) / (n_1 + n_2 + 1)$  (ticks 5–49)")
    ax.set_xlim(-0.5, -40.5)
    ax.set_ylim(-0.5, -40.5)
    for v in range(-40, 1, 5):
        ax.axvline(v, color="gray", linewidth=0.3, alpha=0.3)
        ax.axhline(v, color="gray", linewidth=0.3, alpha=0.3)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label(r"$+1$: $N_1$ wins     $0$: tied     $-1$: $N_2$ wins")
    fig.tight_layout()
    fig.savefig(save, dpi=120)
    plt.close(fig)


def plot_fig10(w_vals, grid, save):
    """Render with FCS Fig. 10 orientation: y=w_21, x=w_12, both negative axes.

    Top-left = 0 (small |w|). Bottom-right = -40 (large |w|).
    """
    fig, ax = plt.subplots(figsize=(6.5, 6))
    # extent: left, right, bottom, top
    left = w_vals.min() - 0.5
    right = w_vals.max() + 0.5
    # imshow with origin='upper' and extent: we want y axis to go 0 at top, -40 at bottom
    # data row 0 corresponds to w_vals[0] = -1 (near top). Use origin='upper' and flip y-axis extent.
    im = ax.imshow(
        grid.astype(int),
        cmap="RdBu",
        vmin=0, vmax=1,
        extent=[right, left, right, left],  # x: -1..-40, y: -1..-40
        origin="upper",
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xlabel("$w_{12}$ (inhibition N1→N2)")
    ax.set_ylabel("$w_{21}$ (inhibition N2→N1)")
    ax.set_title("FCS Fig. 10 reproduction — blue=WTA in 4 ticks, red=not")
    # Invert so 0 is top-left and -40 is bottom-right
    ax.set_xlim(-0.5, -40.5)
    ax.set_ylim(-0.5, -40.5)
    # Grid lines
    for v in range(-40, 1, 5):
        ax.axvline(v, color="gray", linewidth=0.3, alpha=0.3)
        ax.axhline(v, color="gray", linewidth=0.3, alpha=0.3)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["red (no WTA)", "blue (WTA)"])
    fig.tight_layout()
    fig.savefig(save, dpi=120)
    plt.close(fig)


def plot_property7_examples(save, T=50):
    """Show 4 raster traces of contralateral inhibition at representative weights."""
    examples = [
        (-1, -1, "symmetric weak"),
        (-2, -15, "asymmetric strong inhib from N2"),
        (-20, -3, "asymmetric strong inhib from N1"),
        (-30, -30, "symmetric strong"),
    ]
    fig, axes = plt.subplots(len(examples), 1, figsize=(10, 1.8 * len(examples)),
                             sharex=True)
    for ax, (w12, w21, label) in zip(axes, examples):
        W, B, ext = contralateral(w12, w21, T=T)
        spikes, _ = simulate(W, B, ext, T=T)
        wta = wta_criterion(spikes, T=T)
        ticks = np.arange(T)
        for i, (name, color) in enumerate(zip(["N1", "N2"], ["C0", "C3"])):
            ax.vlines(ticks[spikes[i]], i + 0.1, i + 0.9, colors=color, linewidth=2)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["N1", "N2"])
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylim(-0.1, 2.1)
        tag = "blue (WTA)" if wta else "red (not WTA)"
        ax.set_title(f"w_12={w12}, w_21={w21}  [{label}]  →  {tag}", fontsize=10)
    axes[-1].set_xlabel("tick t")
    fig.tight_layout()
    fig.savefig(save, dpi=120)
    plt.close(fig)


# --- Main ---------------------------------------------------------------------

def main():
    report_lines = []
    report_lines.append("# Phase 0 Report — Simulator + Ground Truth\n")

    # Property 5
    print("[Phase 0] Sanity-check: Property 5 exact pattern match...")
    ok, got, expected, p5_spikes, p5_local = property5_sanity_check(T=30)
    report_lines.append("## Property 5 (negative loop)\n")
    report_lines.append(f"- Weights used: `w_XA=11, w_AI=11, w_IA=-11` (tuned)\n")
    report_lines.append(f"- Expected activator sequence: `{expected}`\n")
    report_lines.append(f"- Got activator sequence:      `{got}`\n")
    report_lines.append(f"- **Exact match:** {ok}\n")
    report_lines.append("\nNote: FCS Appendix A.3 suggests `w_IA=-20` as a starting "
                        "point. Analytical tracing of the Lustre semantics shows that "
                        "-20 overshoots (period 5), while -11 (exact cancellation) "
                        "reproduces the Property 5 period-4 pattern exactly. The "
                        "negative loop is therefore used with `w_IA=-11` henceforth.\n")
    plot_property5(p5_spikes, RESULTS / "phase0_property5_trace.png")
    assert ok, f"Property 5 sanity check failed:\n  expected {expected}\n  got      {got}"

    # Property 7 examples
    print("[Phase 0] Plotting Property 7 examples...")
    plot_property7_examples(RESULTS / "phase0_property7_examples.png")

    # Fig. 10 sweep
    print("[Phase 0] Sweeping FCS Fig. 10 grid (40x40)...")
    w_vals, grid, count_diff, dominance = fig10_sweep()
    np.save(RESULTS / "fcs_fig10_groundtruth.npy", grid)
    np.save(RESULTS / "fcs_fig10_countdiff.npy", count_diff)
    np.save(RESULTS / "fcs_fig10_dominance.npy", dominance)
    np.save(RESULTS / "fcs_fig10_wvals.npy", w_vals)
    plot_fig10(w_vals, grid, RESULTS / "phase0_fig10_reproduction.png")
    plot_dominance(w_vals, dominance, RESULTS / "phase0_fig10_dominance.png")

    n_blue = int(grid.sum())
    n_total = grid.size
    report_lines.append("## Property 7 / FCS Fig. 10\n")
    report_lines.append(f"- Grid: {len(w_vals)}×{len(w_vals)}, "
                        f"w_12, w_21 ∈ [{w_vals.min()}, {w_vals.max()}]\n")
    report_lines.append(f"- T=50, WTA criterion per plan Appendix A.7\n")
    report_lines.append(f"- Blue cells (WTA reached): {n_blue}/{n_total} "
                        f"({100*n_blue/n_total:.1f}%)\n")

    # Diagnostic: diagonal cells
    diag_blue = int(np.trace(grid))
    report_lines.append(f"- Diagonal (symmetric weights) blue cells: {diag_blue}/{len(w_vals)} "
                        f"(expected 0 by symmetry)\n")

    # Quick-look: where is the boundary?
    # Count blue in "small |w|" region (|w12|, |w21| <= 5) and "large |w|" region
    small = grid[:5, :5]
    large = grid[-5:, -5:]
    report_lines.append(f"- Upper-left 5×5 (|w|≤5): {int(small.sum())}/25 blue\n")
    report_lines.append(f"- Bottom-right 5×5 (|w|≥36): {int(large.sum())}/25 blue\n")

    # Dominance ratio stats (continuous ground truth for Phase 1 regression)
    dom_max = float(np.abs(dominance).max())
    dom_std = float(dominance.std())
    off_diag_mask = ~np.eye(len(w_vals), dtype=bool)
    dom_off_abs_mean = float(np.abs(dominance[off_diag_mask]).mean())
    report_lines.append(f"- Dominance ratio range: [{float(dominance.min()):+.3f}, "
                        f"{float(dominance.max()):+.3f}], std={dom_std:.3f}\n")
    report_lines.append(f"- Mean |dominance| off-diagonal: {dom_off_abs_mean:.3f} "
                        "(Phase 1 will regress Δ against this continuous signal)\n")

    # Qualitative check
    report_lines.append("\n### Qualitative comparison to FCS Fig. 10\n")
    report_lines.append(
        "**FCS Fig. 10 description (plan Appendix A.4):** blue occupies top-and-left "
        "edges (small |w|); red occupies bottom-right (large |w|); boundary is a "
        "staircase from upper-right to lower-left.\n")
    report_lines.append(
        "\n**Our reproduction** shows a qualitatively different structure:\n"
        "- Symmetric weights (diagonal) are uniformly red — the simulator is fully\n"
        "  deterministic under integer arithmetic, so N1 and N2 receive identical\n"
        "  external drive and produce identical spike trains when w_12 = w_21.\n"
        "- A *central red block* (roughly |w| ∈ [13, 29] on both axes, extending\n"
        "  to moderately asymmetric weights): both neurons synchronize into a\n"
        "  shared rhythm whose period increases with |w|, but neither falls silent.\n"
        "- Two triangular *blue wings* off the diagonal: when |w_12| and |w_21|\n"
        "  differ by enough to break the shared rhythm, one neuron captures and\n"
        "  the other goes silent.\n"
        "- Two *corner red regions* (very-small × very-large) where the asymmetry\n"
        "  is large but one side's inhibition is too weak to reach threshold at all.\n")
    report_lines.append(
        "\n**Source of discrepancy.** FCS Fig. 10 is produced by Kind2 model-checking "
        "an LTL property — Kind2 searches all reachable states and declares 'blue' "
        "whenever *some* trajectory reaches WTA. Our simulator, starting from zero "
        "state with symmetric constant input, explores a single trajectory. Under "
        "symmetric weights that single trajectory is tied, so no WTA emerges. "
        "Under moderately asymmetric weights that still produce synchronised "
        "spiking, the symmetry between simultaneous spike emissions prevents WTA "
        "from breaking out within our finite 50-tick window.\n")
    report_lines.append(
        "\n**Implications for Phases 1–3.** The ground truth from this simulator is "
        "the authoritative oracle for the planned spectral predictors. The structure "
        "observed here (synchronised-rhythm red block + asymmetry-driven blue wings) "
        "is itself a non-trivial 2D classification target — arguably *more* "
        "informative than FCS Fig. 10 because it directly reflects the LI&F "
        "dynamics rather than a reachability property. Phase 1 will test whether "
        "the eigenvalue gap Δ = ||λ₁|-|λ₂|| of the raw inhibitory W tracks the "
        "*boundary of the blue wings* (i.e., the transition from synchronised tie "
        "to asymmetric capture).\n")
    report_lines.append(
        "\n**Stop-and-report rule.** Per plan §8, Phase 0 halts here pending user "
        "review. The simulator Property-5 exact match confirms FCS semantic "
        "fidelity. The Fig. 10 ground-truth structure is well-defined and ready to "
        "serve as the Phase 1 target, even though it differs visually from the "
        "FCS staircase description. User may wish to: (a) approve proceeding to "
        "Phase 1 using this ground truth, (b) attempt to reproduce FCS's "
        "reachability-based blue/red coloring by adding a small symmetry-breaking "
        "bias to one neuron's initial state, or (c) refine the WTA criterion.\n")

    # Write report
    with open(RESULTS / "phase0_report.md", "w") as f:
        f.writelines(report_lines)

    print(f"[Phase 0] Done. Results in {RESULTS}")
    print(f"  Blue cells: {n_blue}/{n_total}  ({100*n_blue/n_total:.1f}%)")
    print(f"  Diagonal blue: {diag_blue}/{len(w_vals)}  (expected 0 by symmetry)")


if __name__ == "__main__":
    main()
