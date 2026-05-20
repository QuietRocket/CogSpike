"""Phase 4 step 2: Mechanism decomposition of QR-vs-FCS agreement.

Decomposes QR's Jaccard agreement with FCS (and the gain over Siegert)
into per-cell contributions classified by the cell's intrinsic FCS
synchrony-period block on the diagonal staircase.

Two mechanism claims to test:

  Q1 (best at N=50). Is QR-N=50's +0.021 J-gain over Siegert
      concentrated on the diagonal-staircase cells (period-2 / 3 / 4
      blocks), or is it accidental cancellation?

  Q2 (monotone in N). Why does J(QR vs FCS) decline as N grows from
      50 to 20000? The hypothesis is that sqrt(A/N) noise sets a
      dissolution probability per staircase cell; this should decay
      monotonically in 1/N, with the period-N block (deepest lock)
      dissolving last.

Per-cell classification:
  - 'blue'         : fcs_labels = 1
  - 'period-N'     : fcs_labels = 0 AND smallest period p with
                     spikes synchronously locked at offset p in the
                     post-warmup window is p in {1,2,3,4}
  - 'red-other'    : fcs_labels = 0 AND no period <= 12 fits

The period detection runs the FCS oracle directly on the FCS-red
cells (~586 cells, ~30 s).

Outputs (under results/phase4/):
  mechanism_grid.npz
  mechanism_blocks.pdf
  mechanism_dissolution_vs_N.pdf
  mechanism_flip_maps.pdf
  mechanism_jaccard_decomp.pdf
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from deq.archetypes.lif_fcs import simulate  # noqa: E402
from deq.archetypes.topologies import contralateral  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "phase4"
RESULTS.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Cell classification: synchrony-period inference
# ----------------------------------------------------------------------

T_MAX = 50
T_WARMUP = 4
PERIOD_CANDIDATES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
TOP_PERIODS = (1, 2, 3, 4)         # the "staircase" cell classes
CLASS_NAMES = ("blue", "period-1", "period-2", "period-3", "period-4",
               "red-other")
CLASS_COLORS = {
    "blue":      "tab:blue",
    "period-1":  "tab:cyan",
    "period-2":  "tab:purple",
    "period-3":  "tab:red",
    "period-4":  "tab:orange",
    "red-other": "tab:gray",
}


def infer_period(spike_row: np.ndarray) -> int:
    """Smallest p in PERIOD_CANDIDATES with spikes periodic mod p.

    Tests whether spike_row[i + p] == spike_row[i] for all valid i in
    the post-warmup window. Returns the smallest such p, or 0 if none.
    """
    n = len(spike_row)
    for p in PERIOD_CANDIDATES:
        if p >= n:
            break
        ok = True
        for i in range(n - p):
            if spike_row[i] != spike_row[i + p]:
                ok = False
                break
        if ok:
            return p
    return 0


def classify_cells(W_VALUES: np.ndarray, fcs_labels: np.ndarray) -> np.ndarray:
    """Return (nW, nW) int array with class index per cell.

    Class indices follow CLASS_NAMES order: blue=0, period-1=1,
    period-2=2, period-3=3, period-4=4, red-other=5.
    """
    nW = len(W_VALUES)
    cls = np.zeros((nW, nW), dtype=int)
    name_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}

    t0 = time.time()
    n_red = int((1 - fcs_labels).sum())
    print(f"  Period inference on {n_red} FCS-red cells...")
    for i, w_21 in enumerate(W_VALUES):
        for j, w_12 in enumerate(W_VALUES):
            if fcs_labels[i, j] == 1:
                cls[i, j] = name_to_idx["blue"]
                continue
            W, B, ext = contralateral(int(w_12), int(w_21), T=T_MAX)
            spikes, _ = simulate(W, B, ext, T=T_MAX)
            post0 = spikes[0, T_WARMUP:]
            post1 = spikes[1, T_WARMUP:]
            p0 = infer_period(post0)
            p1 = infer_period(post1)
            sync = np.array_equal(post0, post1)
            if sync and p0 in TOP_PERIODS:
                cls[i, j] = name_to_idx[f"period-{p0}"]
            elif (not sync) and p0 in TOP_PERIODS and p1 in TOP_PERIODS:
                # Asymmetric red: e.g., one neuron locks at period-p and
                # the other at a different period. Use max(p0, p1) as
                # the dominant period; assign to that class.
                cls[i, j] = name_to_idx[f"period-{max(p0, p1)}"]
            else:
                cls[i, j] = name_to_idx["red-other"]
    elapsed = time.time() - t0
    print(f"  Period inference done in {elapsed:.1f}s")
    return cls


# ----------------------------------------------------------------------
# Decomposition: per-N gain/loss vs Siegert by cell class
# ----------------------------------------------------------------------


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    inter = (a & b).sum(); union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else 1.0


def decompose(cls: np.ndarray, fcs: np.ndarray, sieg: np.ndarray,
              qr_per_N: dict, N_SWEEP: list) -> dict:
    """Per-N per-class counts of QR-red and QR-blue cells.

    For each cell class c and each N, count:
      n_qr_red_c   : cells in class c that QR labels red
      n_qr_blue_c  : cells in class c that QR labels blue
      n_cells_c    : total cells in class c (constant)

    Also returns the Siegert reference: n_sieg_red_c per class.
    """
    n_classes = len(CLASS_NAMES)
    counts = {
        "n_cells": np.zeros(n_classes, dtype=int),
        "n_sieg_red": np.zeros(n_classes, dtype=int),
        "n_qr_red": np.zeros((len(N_SWEEP), n_classes), dtype=int),
        "j_vs_fcs": np.zeros(len(N_SWEEP)),
        "j_vs_sieg": np.zeros(len(N_SWEEP)),
    }
    for c in range(n_classes):
        mask = (cls == c)
        counts["n_cells"][c] = int(mask.sum())
        counts["n_sieg_red"][c] = int((mask & (sieg == 0)).sum())

    for k, N in enumerate(N_SWEEP):
        qr = qr_per_N[N]
        for c in range(n_classes):
            mask = (cls == c)
            counts["n_qr_red"][k, c] = int((mask & (qr == 0)).sum())
        counts["j_vs_fcs"][k] = jaccard(qr, fcs)
        counts["j_vs_sieg"][k] = jaccard(qr, sieg)
    return counts


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------


def plot_blocks(cls: np.ndarray, W_VALUES: np.ndarray, out_pdf: Path):
    """Annotated FCS Fig. 10 with cells colored by synchrony period class."""
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    name_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}
    for i, w_21 in enumerate(W_VALUES):
        for j, w_12 in enumerate(W_VALUES):
            name = CLASS_NAMES[cls[i, j]]
            color = CLASS_COLORS[name]
            ax.scatter(int(w_12), int(w_21), c=color, s=22, edgecolor="none")
    handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=8,
                          markerfacecolor=CLASS_COLORS[n],
                          label=f"{n} ({int((cls == name_to_idx[n]).sum())})")
               for n in CLASS_NAMES]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlabel(r"$w_{12}$ (FCS scaled)")
    ax.set_ylabel(r"$w_{21}$ (FCS scaled)")
    ax.set_title(
        "Per-cell synchrony classification of FCS Property 7 grid\n"
        "(blue: WTA; period-p: synchronous lock at integer-tick period p)",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    plt.close(fig)


def plot_dissolution_vs_N(counts: dict, N_SWEEP: list, fcs: np.ndarray,
                          out_pdf: Path):
    """frac_diss_p(N) per period block, plus boundary broadening cost."""
    name_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    Ns = np.array(N_SWEEP)

    for name in ("period-1", "period-2", "period-3", "period-4", "red-other"):
        c = name_to_idx[name]
        n_cells = counts["n_cells"][c]
        if n_cells == 0:
            continue
        frac_red = counts["n_qr_red"][:, c] / n_cells
        # "dissolution" = fraction of FCS-red cells in this class that
        # QR also calls red.
        ax.plot(Ns, frac_red, 'o-', label=f"{name} ({n_cells} cells)",
                color=CLASS_COLORS[name])

    # Boundary broadening: fraction of FCS-blue cells QR labels red.
    c_blue = name_to_idx["blue"]
    n_blue = counts["n_cells"][c_blue]
    frac_broad = counts["n_qr_red"][:, c_blue] / max(n_blue, 1)
    ax.plot(Ns, frac_broad, 's--', label=f"blue→red (broadening, {n_blue} cells)",
            color="black")

    ax.set_xscale("log")
    ax.set_xlabel("N (population size)")
    ax.set_ylabel("fraction of class labelled red by QR")
    ax.set_title(
        "QR red-call fraction per FCS-class vs N\n"
        "(higher on staircase = good dissolution; "
        "higher on blue = boundary broadening cost)",
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    plt.close(fig)


def plot_flip_maps(cls: np.ndarray, W_VALUES: np.ndarray,
                   qr_per_N: dict, sieg: np.ndarray, N_SWEEP: list,
                   out_pdf: Path):
    """Cells flipped from Siegert-blue to QR-red at each N, by class."""
    n_panels = len(N_SWEEP)
    ncol = 4
    nrow = (n_panels + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow))
    if nrow == 1:
        axes = np.array([axes])

    for k, N in enumerate(N_SWEEP):
        ax = axes[k // ncol, k % ncol]
        qr = qr_per_N[N]
        flipped = (sieg == 1) & (qr == 0)
        for i, w_21 in enumerate(W_VALUES):
            for j, w_12 in enumerate(W_VALUES):
                if not flipped[i, j]:
                    ax.scatter(int(w_12), int(w_21), c="lightgray", s=8,
                               edgecolor="none", alpha=0.5)
                else:
                    name = CLASS_NAMES[cls[i, j]]
                    color = CLASS_COLORS[name]
                    ax.scatter(int(w_12), int(w_21), c=color, s=22,
                               edgecolor="none")
        n_flip = int(flipped.sum())
        ax.set_title(f"N={N}: {n_flip} Siegert-blue → QR-red flips",
                     fontsize=10)
        ax.set_xlabel(r"$w_{12}$", fontsize=9)
        ax.set_ylabel(r"$w_{21}$", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    for k in range(n_panels, nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.suptitle(
        "Where QR's noise flips Siegert-blue cells to red, by FCS class\n"
        "(colored = flip; gray = unchanged. Q1 evidence: flips concentrate "
        "on staircase classes.)",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    plt.close(fig)


def plot_jaccard_decomp(counts: dict, cls: np.ndarray, fcs: np.ndarray,
                        sieg: np.ndarray, qr_per_N: dict, N_SWEEP: list,
                        out_pdf: Path):
    """Per-N waterfall: gain from each staircase class minus broadening cost.

    Net change in correctly-labelled red-cells over Siegert:
        gain[c]  = | (cls == c) AND (qr == 0) AND (sieg == 1) |  (true flips)
        cost[c]  = | (cls == c) AND (qr == 1) AND (sieg == 0) |  (false un-flips,
                    usually negligible)

    For the FCS-blue class:
        cost     = | (cls == 0) AND (qr == 0) AND (sieg == 1) |  (broadening)
        gain     = | (cls == 0) AND (qr == 1) AND (sieg == 0) |  (negligible)
    """
    name_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: net correct-red flips per class
    bar_labels = []
    for name in ("period-1", "period-2", "period-3", "period-4", "red-other"):
        c = name_to_idx[name]
        per_N = []
        for k, N in enumerate(N_SWEEP):
            qr = qr_per_N[N]
            mask = (cls == c)
            true_flips = int((mask & (qr == 0) & (sieg == 1)).sum())
            per_N.append(true_flips)
        axes[0].plot(N_SWEEP, per_N, 'o-', color=CLASS_COLORS[name],
                     label=f"{name}")
    # Boundary cost
    c = name_to_idx["blue"]
    per_N_cost = []
    for k, N in enumerate(N_SWEEP):
        qr = qr_per_N[N]
        cost = int(((cls == c) & (qr == 0) & (sieg == 1)).sum())
        per_N_cost.append(cost)
    axes[0].plot(N_SWEEP, per_N_cost, 's--', color="black",
                 label="cost: blue→red")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("cell count")
    axes[0].set_title("Staircase true-flips vs broadening cost per N",
                      fontsize=10)
    axes[0].legend(fontsize=9, loc="best")
    axes[0].grid(alpha=0.3, which="both")

    # Panel B: Jaccard vs Siegert and FCS, per N
    j_fcs = counts["j_vs_fcs"]
    j_sieg = counts["j_vs_sieg"]
    j_fcs_sieg = jaccard(sieg, fcs)
    axes[1].plot(N_SWEEP, j_fcs, 'o-', color="tab:red", label="J(QR vs FCS)")
    axes[1].plot(N_SWEEP, j_sieg, 's-', color="tab:blue",
                 label="J(QR vs Siegert)")
    axes[1].axhline(j_fcs_sieg, color="gray", linestyle=":",
                    label=f"Siegert reference J(vs FCS)={j_fcs_sieg:.3f}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("Jaccard")
    axes[1].set_title("Jaccard agreement: QR vs FCS and vs Siegert",
                      fontsize=10)
    axes[1].legend(fontsize=9, loc="center right")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].set_ylim(0.55, 1.02)

    fig.suptitle(
        "Mechanism decomposition: where the QR gain over Siegert comes from",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    print("=" * 70)
    print("Phase 4 mechanism analysis")
    print("=" * 70)

    p0 = np.load(HERE / "results" / "phase0" / "fcs_grid.npz",
                 allow_pickle=True)
    W_VALUES = p0["W_VALUES"]
    fcs_labels = p0["fcs_labels"]
    nW = len(W_VALUES)
    print(f"\nGrid: {nW}x{nW} over (w_12, w_21) in [{int(W_VALUES[0])}, "
          f"{int(W_VALUES[-1])}]^2")
    print(f"FCS blue: {int(fcs_labels.sum())}/{nW*nW}; "
          f"red: {nW*nW - int(fcs_labels.sum())}/{nW*nW}")

    p1 = np.load(HERE / "results" / "phase1" / "siegert_grid.npz",
                 allow_pickle=True)
    sieg_labels = p1["sieg_labels"]

    p4 = np.load(HERE / "results" / "phase4" / "qr_grid_extended.npz",
                 allow_pickle=True)
    N_SWEEP = list(p4["N_SWEEP"])
    qr_labels_arr = p4["qr_labels"]
    qr_per_N = {int(N): qr_labels_arr[k] for k, N in enumerate(N_SWEEP)}
    print(f"\nExtended-N QR sweep loaded: N in {N_SWEEP}")

    print("\nStep 1: classify FCS cells by synchrony period")
    cls = classify_cells(W_VALUES, fcs_labels)

    print("\n  Class counts:")
    for c, name in enumerate(CLASS_NAMES):
        n = int((cls == c).sum())
        print(f"    {name:11s}: {n:4d} cells")

    print("\nStep 2: per-N decomposition")
    counts = decompose(cls, fcs_labels, sieg_labels, qr_per_N,
                       [int(N) for N in N_SWEEP])

    print("\n  Per-N stats:")
    print(f"  {'N':>6s}  {'J(FCS)':>7s}  {'J(Sieg)':>8s}  "
          f"{'p1 red':>7s}  {'p2 red':>7s}  {'p3 red':>7s}  "
          f"{'p4 red':>7s}  {'other':>7s}  {'broad':>7s}")
    name_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}
    for k, N in enumerate(N_SWEEP):
        N = int(N)
        row = []
        for name in ("period-1", "period-2", "period-3", "period-4",
                     "red-other"):
            c = name_to_idx[name]
            n_red = counts["n_qr_red"][k, c]
            n_tot = counts["n_cells"][c]
            row.append(f"{n_red}/{n_tot}" if n_tot > 0 else "—")
        # broadening: FCS-blue that QR called red
        c_blue = name_to_idx["blue"]
        broad_red = counts["n_qr_red"][k, c_blue]
        broad_tot = counts["n_cells"][c_blue]
        row.append(f"{broad_red}/{broad_tot}")
        print(f"  {N:>6d}  {counts['j_vs_fcs'][k]:>7.3f}  "
              f"{counts['j_vs_sieg'][k]:>8.3f}  "
              f"{row[0]:>7s}  {row[1]:>7s}  {row[2]:>7s}  "
              f"{row[3]:>7s}  {row[4]:>7s}  {row[5]:>7s}")

    print("\nStep 3: plotting")
    plot_blocks(cls, W_VALUES, RESULTS / "mechanism_blocks.pdf")
    plot_dissolution_vs_N(counts, [int(N) for N in N_SWEEP], fcs_labels,
                          RESULTS / "mechanism_dissolution_vs_N.pdf")
    plot_flip_maps(cls, W_VALUES, qr_per_N, sieg_labels,
                   [int(N) for N in N_SWEEP],
                   RESULTS / "mechanism_flip_maps.pdf")
    plot_jaccard_decomp(counts, cls, fcs_labels, sieg_labels, qr_per_N,
                        [int(N) for N in N_SWEEP],
                        RESULTS / "mechanism_jaccard_decomp.pdf")

    print("\nStep 4: save mechanism grid")
    np.savez(
        RESULTS / "mechanism_grid.npz",
        W_VALUES=W_VALUES,
        N_SWEEP=np.array(N_SWEEP),
        cls=cls,
        class_names=np.array(CLASS_NAMES),
        n_cells=counts["n_cells"],
        n_sieg_red=counts["n_sieg_red"],
        n_qr_red=counts["n_qr_red"],
        j_vs_fcs=counts["j_vs_fcs"],
        j_vs_sieg=counts["j_vs_sieg"],
    )
    print(f"  wrote {RESULTS / 'mechanism_grid.npz'}")

    # Summary verdict
    print("\n" + "=" * 70)
    print("Mechanism verdict")
    print("=" * 70)
    j_sieg_vs_fcs = jaccard(sieg_labels, fcs_labels)
    print(f"  Siegert baseline: J(vs FCS) = {j_sieg_vs_fcs:.3f}")
    best_k = int(np.argmax(counts["j_vs_fcs"]))
    best_N = int(N_SWEEP[best_k])
    best_j = float(counts["j_vs_fcs"][best_k])
    print(f"  Best QR: N={best_N} at J(vs FCS) = {best_j:.3f} "
          f"(+{best_j - j_sieg_vs_fcs:.3f} vs Siegert)")
    # Where did the gain come from at best N?
    qr_best = qr_per_N[best_N]
    flips_in_staircase = 0
    flips_outside_staircase = 0
    cost_blue_broaden = 0
    for c, name in enumerate(CLASS_NAMES):
        mask = (cls == c)
        true_flips = int((mask & (qr_best == 0) & (sieg_labels == 1)).sum())
        false_flips = int((mask & (qr_best == 0) & (sieg_labels == 0)).sum())
        if name == "blue":
            cost_blue_broaden = true_flips  # these are bad (blue → red)
        elif name in ("period-1", "period-2", "period-3", "period-4"):
            flips_in_staircase += true_flips
        else:
            flips_outside_staircase += true_flips
    print(f"  At N={best_N}:")
    print(f"    staircase true-flips (FCS-red, Siegert-blue → QR-red): "
          f"{flips_in_staircase}")
    print(f"    other-red true-flips: {flips_outside_staircase}")
    print(f"    broadening cost (FCS-blue, Siegert-blue → QR-red): "
          f"{cost_blue_broaden}")
    print(f"    net correct flips - cost: "
          f"{flips_in_staircase + flips_outside_staircase - cost_blue_broaden}")


if __name__ == "__main__":
    main()
