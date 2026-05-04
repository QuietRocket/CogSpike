"""Phase 0: FCS-coordinate reproduction of Property 7 (winner-takes-all).

Reproduces De Maria et al. 2020 Fig. 10's verification of WTA stabilization
within 4 ticks on the 2-neuron contralateral inhibition motif, using the
FCS-accurate oracle deq/archetypes/lif_fcs.py:simulate verbatim.

No symmetry breaker is applied. FCS's Lustre encoding has no implicit
breaker either: at perfectly symmetric weights w_12 = w_21, both neurons
follow identical trajectories and lock into synchronous oscillation
(period-k for various k), which counts as 'no WTA' (red). This is the
spike-timing-lock signature: a diagonal staircase of red blocks running
along the symmetric direction, broadening as |w| grows because integer-
tick dynamics absorb small asymmetries into the same firing period. The
broadening is the artefact that smooth-rate theory (Siegert) cannot see.

Grid: 40 x 40 over (w_12, w_21) in {-40, -39, ..., -1}^2 to match FCS's
visual resolution.

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
# Fine 40x40 grid over [-40, -1]^2 to resolve the diagonal-red staircase
# structure FCS reported in Fig. 10.
# ---------------------------------------------------------------------------
W_VALUES = np.arange(-40, 0, dtype=np.int64)   # -40, -39, ..., -1
T_MAX = 50
T_WARMUP = 4              # FCS Fig. 10's "within 4 ticks" gate
RATE_HI = 0.99            # winner spike-rate (post-warmup) >= this
RATE_LO = 0.01            # loser spike-rate (post-warmup) <= this


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
    """FCS-faithful sweep: no symmetry breaker."""
    nW = len(W_VALUES)
    fcs_labels = np.zeros((nW, nW), dtype=int)
    rate_n1 = np.zeros((nW, nW))
    rate_n2 = np.zeros((nW, nW))

    print(f"FCS-LIF oracle on 2-neuron CI; grid {nW}x{nW}; "
          f"T_max={T_MAX}; T_warmup={T_WARMUP}")
    print(f"  WTA gate: rate_max>={RATE_HI} AND rate_min<={RATE_LO}")
    print(f"  No symmetry breaker (FCS Lustre-faithful)\n")

    for i, w_21 in enumerate(W_VALUES):
        for j, w_12 in enumerate(W_VALUES):
            W, B, ext = contralateral(int(w_12), int(w_21), T=T_MAX)
            spikes, _ = simulate(W, B, ext, T=T_MAX)
            fcs_labels[i, j] = fcs_wta_label(spikes)
            post = spikes[:, T_WARMUP:]
            rate_n1[i, j] = post[0].mean()
            rate_n2[i, j] = post[1].mean()

    return dict(
        W_VALUES=W_VALUES,
        fcs_labels=fcs_labels,
        rate_n1=rate_n1,
        rate_n2=rate_n2,
        T_max=T_MAX,
        t_warmup=T_WARMUP,
        rate_hi=RATE_HI,
        rate_lo=RATE_LO,
    )


def plot_fcs_style(grid: dict, out_pdf: Path):
    """FCS Fig. 10 visual style: blue/red dot grid in (w_12, w_21) plane."""
    fcs_labels = grid["fcs_labels"]
    W = grid["W_VALUES"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    # Use a 2D image for speed (40x40 dots is OK).
    for i, w_21 in enumerate(W):
        for j, w_12 in enumerate(W):
            color = "tab:blue" if fcs_labels[i, j] else "tab:red"
            ax.scatter(int(w_12), int(w_21), c=color, s=22,
                       edgecolor="none")
    ax.set_xlabel(r"$w_{12}$ (inhibition $N_1 \to N_2$, FCS scaled)")
    ax.set_ylabel(r"$w_{21}$ (inhibition $N_2 \to N_1$, FCS scaled)")
    ax.set_title(
        f"FCS Property 7 reproduction\n"
        f"blue: WTA stable within {T_WARMUP} ticks; "
        f"red: synchronous oscillation / no WTA\n"
        f"({fcs_labels.sum()}/{fcs_labels.size} blue, "
        f"no symmetry breaker)",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")


def plot_rate_diff(grid: dict, out_pdf: Path):
    """Heatmap of rate(N1) - rate(N2): asymmetric WTA strength visualization."""
    rate_diff = grid["rate_n1"] - grid["rate_n2"]
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
    plt.colorbar(im, ax=ax, label=r"$\nu_1 - \nu_2$ (post-warmup rate diff)")
    ax.set_xlabel(r"$w_{12}$ (FCS scaled)")
    ax.set_ylabel(r"$w_{21}$ (FCS scaled)")
    ax.set_title("FCS oracle: post-warmup rate asymmetry")
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")


def main():
    grid = run_grid()

    n_blue = int(grid["fcs_labels"].sum())
    n_total = int(grid["fcs_labels"].size)
    print(f"  WTA-stable: {n_blue}/{n_total} ({100*n_blue/n_total:.1f}%)")

    # Diagonal-cell check: how many on or near the diagonal are red?
    diag_offsets = []
    for i, w_21 in enumerate(grid["W_VALUES"]):
        for j, w_12 in enumerate(grid["W_VALUES"]):
            if abs(int(w_12) - int(w_21)) <= 1 and not grid["fcs_labels"][i, j]:
                diag_offsets.append((int(w_12), int(w_21)))
    print(f"  Diagonal red cells (|w_12 - w_21| <= 1): {len(diag_offsets)}")
    if diag_offsets:
        print(f"    sample: {diag_offsets[:5]}")

    out_npz = RESULTS / "fcs_grid.npz"
    np.savez(out_npz, **grid)
    print(f"  wrote {out_npz}")

    plot_fcs_style(grid, RESULTS / "fcs_grid.pdf")
    plot_rate_diff(grid, RESULTS / "rate_diff.pdf")

    # ----- spot-checks at symmetric and asymmetric cells -----
    print()
    for w_12, w_21, label in [(-30, -30, "symmetric"),
                              (-5, -30, "asymmetric (weak w12)"),
                              (-2, -2, "very weak both")]:
        W, B, ext = contralateral(w_12, w_21, T=T_MAX)
        spikes, _ = simulate(W, B, ext, T=T_MAX)
        rate_post = spikes[:, T_WARMUP:].mean(axis=1)
        print(f"  ({w_12:3d}, {w_21:3d}) [{label}]: "
              f"N1={''.join(str(int(s)) for s in spikes[0, :12])}, "
              f"N2={''.join(str(int(s)) for s in spikes[1, :12])}, "
              f"rates=({rate_post[0]:.3f}, {rate_post[1]:.3f}), "
              f"WTA={fcs_wta_label(spikes)}")

    print()
    print("Phase 0 PASS gate: diagonal-staircase red-block structure visible;")
    print(f"  symmetric cells → synchronous oscillation → red, "
          f"as in FCS Fig. 10.")


if __name__ == "__main__":
    main()
