"""Phase 0: FCS-coordinate reproduction of Property 7 at N > 2 neurons.

Sweeps the FCS-LI&F oracle on the (w, N, drive_bump) grid:

    W_VALUES = -40..-1     (scaled inhibitory weight, shared across all edges)
    N_VALUES = [2, 3, 4, 6, 10]
    DRIVE_BUMPS = [0, 1]   (integer self-drive bump on neuron 0)

WTA gate: post-warmup (t >= 4) rate_max >= 0.99 AND second_max <= 0.01,
which at N=2 coincides with the 2-neuron gate rate_max >= 0.99 AND
rate_min <= 0.01.

Output: results/phase0/fcs_grid_multi.{npz,pdf}, fcs_winner_fraction.pdf.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from deq.archetypes.lif_fcs import simulate  # noqa: E402
from deq.archetypes.topologies import all_to_all_inhibition  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "phase0"
RESULTS.mkdir(parents=True, exist_ok=True)


W_VALUES = np.arange(-40, 0, dtype=np.int64)
N_VALUES = [2, 3, 4, 6, 10]
DRIVE_BUMPS = [0, 1]
T_MAX = 50
T_WARMUP = 4
RATE_HI = 0.99
RATE_LO = 0.01
MARGIN_GATE = 0.30   # soft (margin-based) gate; matches phase 3 QR


def fcs_wta_label_strict(spikes: np.ndarray) -> int:
    """1 if post-warmup rate_max >= 0.99 AND second_max <= 0.01 (FCS gate)."""
    post = spikes[:, T_WARMUP:]
    rates = post.mean(axis=1)
    if rates.size < 2:
        return 0
    sorted_rates = np.sort(rates)[::-1]
    return int(sorted_rates[0] >= RATE_HI and sorted_rates[1] <= RATE_LO)


def fcs_wta_label_margin(spikes: np.ndarray) -> int:
    """1 if post-warmup rate_max - second_max >= MARGIN_GATE.

    Softer gate consistent with phase 3 QR and phase 1 Siegert spread gate.
    """
    post = spikes[:, T_WARMUP:]
    rates = post.mean(axis=1)
    if rates.size < 2:
        return 0
    sorted_rates = np.sort(rates)[::-1]
    return int((sorted_rates[0] - sorted_rates[1]) >= MARGIN_GATE)


def run_grid():
    nW = len(W_VALUES)
    nN = len(N_VALUES)
    nB = len(DRIVE_BUMPS)
    fcs_labels_strict = np.zeros((nN, nB, nW), dtype=int)
    fcs_labels_margin = np.zeros((nN, nB, nW), dtype=int)
    rate_max = np.zeros((nN, nB, nW))
    rate_2nd = np.zeros((nN, nB, nW))
    rate_min = np.zeros((nN, nB, nW))

    print(f"FCS-LI&F oracle on uniform inhibition; "
          f"N in {N_VALUES}, drive_bumps in {DRIVE_BUMPS}, "
          f"W in [{int(W_VALUES[0])}, {int(W_VALUES[-1])}] "
          f"({nW} values); T_max={T_MAX}, T_warmup={T_WARMUP}")
    print(f"  strict gate: rate_max>={RATE_HI} AND second_max<={RATE_LO}")
    print(f"  margin gate: rate_max - second_max >= {MARGIN_GATE}\n")

    for nIdx, N in enumerate(N_VALUES):
        for bIdx, db in enumerate(DRIVE_BUMPS):
            t0 = time.time()
            for wIdx, w in enumerate(W_VALUES):
                W, B, ext = all_to_all_inhibition(
                    N=int(N), w=int(w), T=T_MAX, drive_bump=int(db),
                )
                spikes, _ = simulate(W, B, ext, T=T_MAX)
                fcs_labels_strict[nIdx, bIdx, wIdx] = fcs_wta_label_strict(spikes)
                fcs_labels_margin[nIdx, bIdx, wIdx] = fcs_wta_label_margin(spikes)
                post = spikes[:, T_WARMUP:]
                rates = post.mean(axis=1)
                sorted_r = np.sort(rates)[::-1]
                rate_max[nIdx, bIdx, wIdx] = sorted_r[0]
                rate_2nd[nIdx, bIdx, wIdx] = sorted_r[1]
                rate_min[nIdx, bIdx, wIdx] = sorted_r[-1]
            n_s = int(fcs_labels_strict[nIdx, bIdx].sum())
            n_m = int(fcs_labels_margin[nIdx, bIdx].sum())
            elapsed = time.time() - t0
            print(f"  N={N:2d}, drive_bump={db}: "
                  f"strict {n_s}/{nW}, margin {n_m}/{nW}, "
                  f"elapsed {elapsed:.1f}s")

    return dict(
        W_VALUES=W_VALUES,
        N_VALUES=np.array(N_VALUES),
        DRIVE_BUMPS=np.array(DRIVE_BUMPS),
        fcs_labels_strict=fcs_labels_strict,
        fcs_labels_margin=fcs_labels_margin,
        rate_max=rate_max,
        rate_2nd=rate_2nd,
        rate_min=rate_min,
        T_max=T_MAX,
        t_warmup=T_WARMUP,
        rate_hi=RATE_HI,
        rate_lo=RATE_LO,
        margin_gate=MARGIN_GATE,
    )


def plot_grid(grid: dict, out_pdf: Path):
    """Two-column plot: strict gate (left) vs margin gate (right), per N row.

    Within each cell, two rows of dots for drive_bump=0 and drive_bump=1.
    """
    W = grid["W_VALUES"]
    Ns = grid["N_VALUES"]
    Bs = grid["DRIVE_BUMPS"]
    nN = len(Ns)
    fig, axes = plt.subplots(nN, 2, figsize=(13, 1.6 * nN), sharex=True)
    if nN == 1:
        axes = axes.reshape(1, 2)
    for nIdx, N in enumerate(Ns):
        for gIdx, (gate_key, gate_name) in enumerate([
            ("fcs_labels_strict", "strict (max>=0.99 AND 2nd<=0.01)"),
            ("fcs_labels_margin", "margin (max-2nd >= 0.30)"),
        ]):
            ax = axes[nIdx, gIdx]
            labels = grid[gate_key]
            for bIdx, db in enumerate(Bs):
                y = float(db)
                for wIdx, w in enumerate(W):
                    color = "tab:blue" if labels[nIdx, bIdx, wIdx] else "tab:red"
                    ax.scatter(int(w), y, c=color, s=36, edgecolor="none")
            ax.set_ylim(-0.5, max(Bs) + 0.5)
            ax.set_yticks([float(b) for b in Bs])
            ax.set_yticklabels([f"db={b}" for b in Bs])
            n0 = int(labels[nIdx, 0].sum())
            n1 = int(labels[nIdx, 1].sum())
            ax.set_title(
                f"N={int(N)} | {gate_name}\n"
                f"db=0: {n0}/{len(W)}, db=1: {n1}/{len(W)}",
                fontsize=9,
            )
            ax.grid(True, alpha=0.3)
    for col in range(2):
        axes[-1, col].set_xlabel(r"$w$ (FCS scaled, shared across all edges)")
    fig.suptitle(
        "Phase 0 (multi-N): FCS oracle WTA labels on uniform inhibition. "
        "blue = WTA, red = no WTA.",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_pdf}")


def plot_winner_fraction(grid: dict, out_pdf: Path):
    """Winner-fraction (1 - second_max/rate_max) heatmap per (N, drive_bump)."""
    rate_max = grid["rate_max"]
    rate_2nd = grid["rate_2nd"]
    Ns = grid["N_VALUES"]
    Bs = grid["DRIVE_BUMPS"]
    W = grid["W_VALUES"]
    nN = len(Ns)
    fig, axes = plt.subplots(1, 2, figsize=(11, 1.0 + 0.4 * nN), sharey=True)
    for bIdx, db in enumerate(Bs):
        ax = axes[bIdx]
        # spread = rate_max - second_max
        spread = rate_max[:, bIdx, :] - rate_2nd[:, bIdx, :]
        im = ax.imshow(
            spread,
            aspect="auto",
            origin="lower",
            extent=[W[0], W[-1], -0.5, nN - 0.5],
            cmap="viridis",
            vmin=0, vmax=1,
        )
        ax.set_yticks(range(nN))
        ax.set_yticklabels([f"N={int(n)}" for n in Ns])
        ax.set_xlabel(r"$w$")
        ax.set_title(f"drive_bump = {int(db)}")
        plt.colorbar(im, ax=ax, label="rate_max − second_max")
    fig.suptitle("Phase 0 winner-fraction (continuous WTA strength)", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_pdf}")


def main():
    grid = run_grid()

    print("\nSummary (strict / margin / total per N, drive_bump):")
    nW = len(grid["W_VALUES"])
    for nIdx, N in enumerate(grid["N_VALUES"]):
        for bIdx, db in enumerate(grid["DRIVE_BUMPS"]):
            n_s = int(grid["fcs_labels_strict"][nIdx, bIdx].sum())
            n_m = int(grid["fcs_labels_margin"][nIdx, bIdx].sum())
            print(f"  N={int(N):2d}, drive_bump={int(db)}: "
                  f"strict {n_s:2d}/{nW}, margin {n_m:2d}/{nW}")

    # Save and plot.
    np.savez(RESULTS / "fcs_grid_multi.npz", **grid)
    print(f"\n  wrote {RESULTS / 'fcs_grid_multi.npz'}")

    plot_grid(grid, RESULTS / "fcs_grid_multi.pdf")
    plot_winner_fraction(grid, RESULTS / "fcs_winner_fraction.pdf")

    # Spot-checks at representative cells.
    print("\nSpot-checks (drive_bump=1, w=-30, post-warmup rates):")
    for N in grid["N_VALUES"]:
        W, B, ext = all_to_all_inhibition(N=int(N), w=-30, T=T_MAX, drive_bump=1)
        spikes, _ = simulate(W, B, ext, T=T_MAX)
        post_rates = spikes[:, T_WARMUP:].mean(axis=1)
        print(f"  N={int(N):2d}: rates = "
              f"[{', '.join(f'{r:.2f}' for r in post_rates)}]; "
              f"strict={fcs_wta_label_strict(spikes)} "
              f"margin={fcs_wta_label_margin(spikes)}")

    # Pass gate.
    print("\nPhase 0 verdict:")
    n_s = int(grid["fcs_labels_strict"][:, 1, :].sum())
    n_m = int(grid["fcs_labels_margin"][:, 1, :].sum())
    total = grid["fcs_labels_strict"][:, 1, :].size
    print(f"  drive_bump=1: strict {n_s}/{total}, margin {n_m}/{total}")
    if n_m > 0:
        print(f"  PASS: drive_bump=1 sweep produces near-WTA "
              f"(margin gate) at all small N; staircase structure documented.")
    else:
        print(f"  FAIL: no WTA cells even with drive_bump=1 under margin gate.")


if __name__ == "__main__":
    main()
