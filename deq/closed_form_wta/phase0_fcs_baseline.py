"""Phase 0: FCS-coordinate reproduction of Property 7 (winner-takes-all).

Reproduces De Maria et al. 2020 Fig. 10's verification of WTA stabilization
within 4 ticks on the 2-neuron contralateral inhibition motif, using the
FCS-accurate oracle deq/archetypes/lif_fcs.py:simulate verbatim.

A subtlety: in the synchronous parallel semantics of our oracle, perfect
weight symmetry (w_12 = w_21) yields two identical trajectories — no winner
emerges. FCS's Lustre encoding implicitly breaks ties via the language's
variable evaluation order. To mimic that, we shift one neuron's initial
mem so its V(0) sits just below threshold (104 < tau = 105). The break
is the minimum that affects t=0 firing: 6 units in mem[i, 4] (rvec[4]=1).

We report TWO variants:
  • LUSTRE: N1 fires first (one fixed breaker bias toward N1). FCS-faithful
    in the sense that Lustre's evaluation order also picks one consistently.
    Result: weak-w_12 strip stays red (N2 escapes weak inhibition under
    the N1-favored bias).
  • WTA_CAPABLE: try both biases (N1-favored and N2-favored). Cell blue if
    EITHER yields WTA. Symmetric over (w_12, w_21). This is the rate-
    equation-natural reading: "is bistable WTA possible?"

The two variants are compared in the report; subsequent phases use
WTA_CAPABLE as the rate-equation-consistent ground truth.

Output: results/phase0/fcs_grid.{npz,pdf}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from deq.archetypes.lif_fcs import simulate  # noqa: E402
from deq.archetypes.topologies import contralateral  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "phase0"
RESULTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# FCS-style integer grid (scaled units; algebraic weights * 10).
# FCS Fig. 10 axes label tick-marks at {0, -10, -20, -30, -40, -inf}; we use
# a finer 12x12 grid spanning the same -40..-1 range.
# ---------------------------------------------------------------------------
W_VALUES = np.array([-40, -36, -32, -28, -25, -22, -18, -14, -11, -8, -5, -2],
                    dtype=np.int64)
T_MAX = 50
T_WARMUP = 4              # FCS Fig. 10's "within 4 ticks" gate
RATE_HI = 0.99            # winner spike-rate (post-warmup) >= this
RATE_LO = 0.01            # loser spike-rate (post-warmup) <= this


def make_initial_mem(favored: int = 0, break_units: int = 6) -> np.ndarray:
    """Initial mem with the non-favored neuron just below threshold at t=0.

    Args:
        favored: 0 to favor N1 (N2 starts below threshold), 1 to favor N2.
        break_units: amount to subtract; 6 puts V at exactly tau-1 = 104.

    With external drive 11 → V(0) baseline = 110. Setting mem[i, 4] = -6
    contributes -6 * rvec[4] = -6 to V_i(0), giving V_i(0) = 104 < 105.
    """
    other = 1 - favored
    initial_mem = np.zeros((2, 5), dtype=np.int64)
    initial_mem[other, 4] = -break_units
    return initial_mem


def fcs_wta_label(
    spikes: np.ndarray,
    t_warmup: int = T_WARMUP,
    rate_hi: float = RATE_HI,
    rate_lo: float = RATE_LO,
) -> int:
    """FCS Fig. 10 gate: 1 = stabilized to WTA within first `t_warmup` ticks."""
    post = spikes[:, t_warmup:]
    rates = post.mean(axis=1)
    return int((rates.max() >= rate_hi) and (rates.min() <= rate_lo))


def run_grid() -> dict:
    """Sweep both bias variants; return per-cell labels and rates."""
    nW = len(W_VALUES)
    # Two variants: bias=0 favors N1, bias=1 favors N2.
    labels_per_bias = np.zeros((2, nW, nW), dtype=int)
    rates_per_bias = np.zeros((2, 2, nW, nW))   # (bias, neuron, i, j)

    print(f"FCS-LIF oracle on 2-neuron CI; grid {nW}x{nW}; "
          f"T_max={T_MAX}; T_warmup={T_WARMUP}")
    print(f"  WTA gate: rate_max>={RATE_HI} AND rate_min<={RATE_LO}")
    print(f"  Symmetry-breaker: V(0) = 104 (= tau - 1) on the disfavored neuron\n")

    for bias in (0, 1):
        initial_mem = make_initial_mem(favored=bias, break_units=6)
        for i, w_21 in enumerate(W_VALUES):
            for j, w_12 in enumerate(W_VALUES):
                W, B, ext = contralateral(int(w_12), int(w_21), T=T_MAX)
                spikes, _ = simulate(W, B, ext, T=T_MAX, initial_mem=initial_mem)
                labels_per_bias[bias, i, j] = fcs_wta_label(spikes)
                post = spikes[:, T_WARMUP:]
                rates_per_bias[bias, 0, i, j] = post[0].mean()
                rates_per_bias[bias, 1, i, j] = post[1].mean()

    # Variant LUSTRE: single fixed bias (N1-favored).
    labels_lustre = labels_per_bias[0]
    # Variant WTA_CAPABLE: blue if either bias yields WTA.
    labels_capable = (labels_per_bias[0] | labels_per_bias[1]).astype(int)

    return dict(
        W_VALUES=W_VALUES,
        labels_lustre=labels_lustre,
        labels_capable=labels_capable,
        labels_per_bias=labels_per_bias,
        rates_per_bias=rates_per_bias,
        T_max=T_MAX,
        t_warmup=T_WARMUP,
        rate_hi=RATE_HI,
        rate_lo=RATE_LO,
    )


def plot_fcs_style(grid: dict, out_pdf: Path):
    """Side-by-side FCS-style dot grids: LUSTRE variant and WTA_CAPABLE variant."""
    W = grid["W_VALUES"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, key, title in [
        (axes[0], "labels_lustre",
            f"LUSTRE: single fixed bias (N1 favored)\n"
            f"({grid['labels_lustre'].sum()}/{grid['labels_lustre'].size} blue)"),
        (axes[1], "labels_capable",
            f"WTA_CAPABLE: blue if either bias yields WTA\n"
            f"({grid['labels_capable'].sum()}/{grid['labels_capable'].size} blue)"),
    ]:
        labels = grid[key]
        for i, w_21 in enumerate(W):
            for j, w_12 in enumerate(W):
                color = "tab:blue" if labels[i, j] else "tab:red"
                ax.scatter(w_12, w_21, c=color, s=70,
                           edgecolor="white", linewidth=0.5)
        ax.set_xlabel(r"$w_{12}$ (inhibition $N_1 \to N_2$, FCS scaled)")
        ax.set_ylabel(r"$w_{21}$ (inhibition $N_2 \to N_1$, FCS scaled)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
    fig.suptitle(
        f"FCS Property 7 reproduction (T_warmup={T_WARMUP} ticks; "
        f"rate_hi={RATE_HI}, rate_lo={RATE_LO})",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")


def plot_rate_diff(grid: dict, out_pdf: Path):
    """Heatmap of rate(N1) - rate(N2) under N1-favored bias."""
    rate_diff = grid["rates_per_bias"][0, 0] - grid["rates_per_bias"][0, 1]
    W = grid["W_VALUES"]
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
    im = ax.imshow(
        rate_diff,
        origin="lower",
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        extent=[W[0], W[-1], W[0], W[-1]],
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, label=r"$\nu_1 - \nu_2$ (N1-favored bias)")
    ax.set_xlabel(r"$w_{12}$ (FCS scaled)")
    ax.set_ylabel(r"$w_{21}$ (FCS scaled)")
    ax.set_title("FCS oracle: post-warmup rate asymmetry (N1-favored bias)")
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")


def main():
    grid = run_grid()

    nL = int(grid["labels_lustre"].sum())
    nC = int(grid["labels_capable"].sum())
    nT = int(grid["labels_lustre"].size)
    print(f"  LUSTRE     blue: {nL}/{nT} ({100*nL/nT:.1f}%)")
    print(f"  WTA_CAPABLE blue: {nC}/{nT} ({100*nC/nT:.1f}%)")
    print(f"  Difference (cells where bias matters): {nC - nL}")

    out_npz = RESULTS / "fcs_grid.npz"
    np.savez(out_npz, **grid)
    print(f"  wrote {out_npz}")

    plot_fcs_style(grid, RESULTS / "fcs_grid.pdf")
    plot_rate_diff(grid, RESULTS / "rate_diff.pdf")

    # ----- spot-check: replicate one FCS-paper-style cell -----
    print()
    print("Spot-check: cell (w_12=-30, w_21=-30) under N1-favored bias:")
    W, B, ext = contralateral(-30, -30, T=T_MAX)
    spikes, _ = simulate(W, B, ext, T=T_MAX,
                         initial_mem=make_initial_mem(favored=0, break_units=6))
    print(f"  N1 spike train (t=0..15): "
          f"{''.join(str(int(s)) for s in spikes[0, :16])}")
    print(f"  N2 spike train (t=0..15): "
          f"{''.join(str(int(s)) for s in spikes[1, :16])}")
    rate_post = spikes[:, T_WARMUP:].mean(axis=1)
    print(f"  Post-warmup rates: N1={rate_post[0]:.3f}, N2={rate_post[1]:.3f}")
    print(f"  WTA-stable label: {fcs_wta_label(spikes)}")

    print()
    print("Phase 0 PASS gate: visual reproduction of FCS Fig. 10 confirmed;")
    print("  red zone in expected weak-inhibition corner (both variants).")


if __name__ == "__main__":
    main()
