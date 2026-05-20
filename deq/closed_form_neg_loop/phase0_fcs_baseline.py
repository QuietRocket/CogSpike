"""Phase 0: FCS-coordinate reproduction of Property 5 (negative loop oscillation).

Reproduces De Maria et al. 2020, §6.2.5, Property 5: when a sequence of 1s
is fed to the negative-loop motif, the activator A fires the periodic
pattern `1100` and the inhibitor I echoes one tick later. Runs the
FCS-accurate oracle `deq/archetypes/lif_fcs.py:simulate` verbatim over a
2-D scaled-integer grid in (w_IA, w_XA), with w_AI fixed at 11.

Two labels per cell (post-warmup window):
  - strict_p5: A's spike train matches the 1100 cyclic pattern (any of
    the 4 phase shifts) with period exactly 4.
  - broad_osc: A's train is non-trivially oscillatory (period in [2, 12],
    at least one 1 and one 0 per cycle, regular).

Output: results/phase0/fcs_grid.{npz,pdf}, period_map.pdf.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from deq.archetypes.lif_fcs import simulate, spike_sequence_to_str  # noqa: E402
from deq.archetypes.topologies import negative_loop  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "phase0"
RESULTS.mkdir(parents=True, exist_ok=True)


# (w_IA, w_XA) grid. w_AI fixed at 11 (the FCS default; A->I excitatory).
W_IA_VALUES = np.arange(-40, 0, dtype=np.int64)   # -40..-1
W_XA_VALUES = np.arange(1, 41, dtype=np.int64)    # 1..40
W_AI = 11
T_MAX = 64                # enough to lock a period and observe many cycles
T_WARMUP = 16             # discard initial transient
PROP5_TEMPLATE = "1100"   # A's exact firing pattern under Property 5


def _all_phase_shifts(s: str) -> list:
    """All cyclic rotations of `s`."""
    return [s[i:] + s[:i] for i in range(len(s))]


PROP5_PATTERNS = _all_phase_shifts(PROP5_TEMPLATE)


def detect_period(seq: np.ndarray, max_period: int = 12) -> int:
    """Return the smallest period p in [1, max_period] for which seq is
    eventually periodic with period p, else 0.
    """
    n = len(seq)
    if n < 2 * max_period:
        return 0
    s = seq.astype(np.int64)
    for p in range(1, max_period + 1):
        # check that the last 2*p ticks repeat with period p
        tail = s[-2 * p:]
        if np.array_equal(tail[:p], tail[p:]):
            return p
    return 0


def strict_property5(seq: np.ndarray, t_warmup: int = T_WARMUP) -> int:
    """1 iff seq[t_warmup:] matches some cyclic rotation of `1100` repeated.

    We require: (i) length post-warmup is a multiple of 4 segment, (ii) every
    period-4 chunk equals one fixed rotation of `1100` (the same rotation
    throughout).
    """
    post = seq[t_warmup:]
    n = len(post) - (len(post) % 4)
    if n < 8:
        return 0
    post = post[:n]
    # build the string of post
    s = "".join("1" if x else "0" for x in post.astype(int))
    for pat in PROP5_PATTERNS:
        cyc = (pat * (n // 4 + 1))[:n]
        if s == cyc:
            return 1
    return 0


def broad_oscillation(seq: np.ndarray, t_warmup: int = T_WARMUP) -> int:
    """1 iff seq[t_warmup:] is regularly periodic with period in [2, 12],
    has at least one spike and one silence per cycle.
    """
    post = seq[t_warmup:]
    p = detect_period(post, max_period=12)
    if p < 2:
        return 0
    cycle = post[-p:]
    n_ones = int(cycle.sum())
    if n_ones == 0 or n_ones == p:
        return 0  # all-silent or all-firing, not an oscillation
    return 1


def run_grid() -> dict:
    nIA = len(W_IA_VALUES)
    nXA = len(W_XA_VALUES)
    strict = np.zeros((nIA, nXA), dtype=int)
    broad = np.zeros((nIA, nXA), dtype=int)
    period = np.zeros((nIA, nXA), dtype=int)
    rate_A = np.zeros((nIA, nXA))
    rate_I = np.zeros((nIA, nXA))

    print(f"FCS-LIF oracle on 2-neuron negative loop; "
          f"grid {nIA} x {nXA} (w_IA x w_XA); w_AI fixed at {W_AI}")
    print(f"  T_max={T_MAX}, T_warmup={T_WARMUP}")
    print(f"  strict_p5: A's spikes match cyclic rotation of '{PROP5_TEMPLATE}'")
    print(f"  broad_osc: any regular period in [2, 12] with mixed 1/0")
    print()

    for i, w_IA in enumerate(W_IA_VALUES):
        for j, w_XA in enumerate(W_XA_VALUES):
            W, B, ext = negative_loop(
                w_XA=int(w_XA), w_AI=W_AI, w_IA=int(w_IA), T=T_MAX,
            )
            spikes, _ = simulate(W, B, ext, T=T_MAX)
            seq_A = spikes[0]
            seq_I = spikes[1]
            strict[i, j] = strict_property5(seq_A)
            broad[i, j] = broad_oscillation(seq_A)
            period[i, j] = detect_period(seq_A[T_WARMUP:], max_period=12)
            rate_A[i, j] = seq_A[T_WARMUP:].mean()
            rate_I[i, j] = seq_I[T_WARMUP:].mean()

    return dict(
        W_IA_VALUES=W_IA_VALUES,
        W_XA_VALUES=W_XA_VALUES,
        W_AI=W_AI,
        strict_p5=strict,
        broad_osc=broad,
        period=period,
        rate_A=rate_A,
        rate_I=rate_I,
        T_max=T_MAX,
        t_warmup=T_WARMUP,
    )


def plot_boolean_grid(grid: dict, key: str, title: str, out_pdf: Path):
    labels = grid[key]
    W_IA = grid["W_IA_VALUES"]
    W_XA = grid["W_XA_VALUES"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    for i, w_IA in enumerate(W_IA):
        for j, w_XA in enumerate(W_XA):
            color = "tab:blue" if labels[i, j] else "tab:red"
            ax.scatter(int(w_XA), int(w_IA), c=color, s=20, edgecolor="none")
    # Mark the FCS default (w_IA=-11, w_XA=11)
    ax.scatter([11], [-11], facecolors="none", edgecolors="gold",
               s=170, linewidths=2.0, zorder=5,
               label="FCS default (-11, 11)")
    ax.set_xlabel(r"$w_{XA}$ (external drive into A, FCS scaled)")
    ax.set_ylabel(r"$w_{IA}$ (inhibition $I \to A$, FCS scaled)")
    ax.set_title(
        f"{title}\n"
        f"({int(labels.sum())}/{labels.size} blue; "
        f"w_AI fixed at {W_AI})",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    plt.close(fig)


def plot_period_map(grid: dict, out_pdf: Path):
    period = grid["period"]
    W_IA = grid["W_IA_VALUES"]
    W_XA = grid["W_XA_VALUES"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    # Display: period 0 (none/saturated) as grey, otherwise discrete colormap.
    masked = np.ma.masked_where(period == 0, period)
    cmap = plt.cm.get_cmap("viridis", 12)
    im = ax.imshow(
        masked, origin="lower",
        cmap=cmap, vmin=1, vmax=12,
        extent=[W_XA[0] - 0.5, W_XA[-1] + 0.5,
                W_IA[0] - 0.5, W_IA[-1] + 0.5],
        aspect="auto",
    )
    cbar = plt.colorbar(im, ax=ax, ticks=range(1, 13))
    cbar.set_label("FCS-measured period (ticks)")
    ax.scatter([11], [-11], facecolors="none", edgecolors="gold",
               s=170, linewidths=2.0, zorder=5)
    ax.set_xlabel(r"$w_{XA}$")
    ax.set_ylabel(r"$w_{IA}$")
    ax.set_title(
        "Phase 0: FCS-measured period of A on negative-loop grid\n"
        "(grey = no regular period in [2,12])",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    plt.close(fig)


def main():
    grid = run_grid()

    n_strict = int(grid["strict_p5"].sum())
    n_broad = int(grid["broad_osc"].sum())
    n_total = grid["strict_p5"].size
    print(f"  strict_p5: {n_strict}/{n_total} ({100*n_strict/n_total:.1f}%)")
    print(f"  broad_osc: {n_broad}/{n_total} ({100*n_broad/n_total:.1f}%)")

    # Period distribution.
    print("  period histogram:")
    for p in range(0, 13):
        c = int((grid["period"] == p).sum())
        if c:
            print(f"    period={p:2d}: {c:4d} cells")

    # Sanity gate: default cell (w_IA=-11, w_XA=11) must be strict_p5.
    i_def = int(np.where(grid["W_IA_VALUES"] == -11)[0][0])
    j_def = int(np.where(grid["W_XA_VALUES"] == 11)[0][0])
    default_strict = int(grid["strict_p5"][i_def, j_def])
    default_period = int(grid["period"][i_def, j_def])
    print(f"\n  FCS default cell (w_IA=-11, w_XA=11): "
          f"strict_p5={default_strict}, period={default_period}")

    out_npz = RESULTS / "fcs_grid.npz"
    np.savez(out_npz, **grid)
    print(f"\n  wrote {out_npz}")

    plot_boolean_grid(
        grid, "strict_p5",
        f"FCS Property 5 (strict): A matches cyclic '{PROP5_TEMPLATE}'",
        RESULTS / "prop5_strict.pdf",
    )
    plot_boolean_grid(
        grid, "broad_osc",
        "Broad oscillation: regular period in [2,12], mixed firing",
        RESULTS / "osc_broad.pdf",
    )
    plot_period_map(grid, RESULTS / "period_map.pdf")

    # Spot checks.
    print()
    spot_cells = [(-11, 11, "FCS default"),
                  (-20, 11, "Appendix A.3 suggestion"),
                  (-5, 11, "weak inhibition"),
                  (-40, 40, "strong both"),
                  (-1, 1, "very weak both")]
    for w_IA, w_XA, label in spot_cells:
        W, B, ext = negative_loop(
            w_XA=w_XA, w_AI=W_AI, w_IA=w_IA, T=24,
        )
        spikes, _ = simulate(W, B, ext, T=24)
        sA = spike_sequence_to_str(spikes[0])
        sI = spike_sequence_to_str(spikes[1])
        p = detect_period(spikes[0][T_WARMUP:])
        print(f"  ({w_IA:3d}, {w_XA:2d}) [{label:24s}]: "
              f"A={sA}, I={sI}, period={p}")

    print()
    if default_strict and default_period == 4:
        print("Phase 0 PASS gate: FCS default cell reproduces Property 5 "
              "(period-4 '1100' pattern).")
    else:
        print("Phase 0 FAIL gate: FCS default cell did not reproduce "
              f"Property 5 (got strict_p5={default_strict}, "
              f"period={default_period}; expected strict_p5=1, period=4).")


if __name__ == "__main__":
    main()
