"""Phase 1: Siegert (static) reading of FCS Property 7.

Lifts calibration constants from deq/closed_form/results/phase1_grid.npz
(alpha, beta, tau_m, tau_ref locked at the operating point p_thin=0.7) and
applies find_all_fixed_points_contralateral on the same integer (w_12, w_21)
grid Phase 0 scanned with the FCS oracle. Per-cell label:

  WTA-capable iff
    (>=2 fixed points with |nu_a - nu_b| >= ASYM_BISTABLE)  OR
    (1 fixed point with |nu_1 - nu_2| >= ASYM_MONOSTABLE)

Compares Phase 1 labels against Phase 0's WTA_CAPABLE ground truth via
Jaccard agreement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from deq.closed_form.siegert import (  # noqa: E402
    Siegert,
    find_all_fixed_points_contralateral,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "phase1"
RESULTS.mkdir(parents=True, exist_ok=True)


# Operating point from prior closed_form thread (do not recalibrate).
DRIVE = 11.0           # FCS scaled self_drive
P_THIN = 0.7
ASYM_BISTABLE = 0.30   # min |nu_a - nu_b| separation for "asymmetric bistable"
ASYM_MONOSTABLE = 0.30 # min |nu_1 - nu_2| for monostable WTA call


def load_calibration() -> dict:
    """Lift the locked calibration constants from the prior thread."""
    src = ROOT / "deq" / "closed_form" / "results" / "phase1_grid.npz"
    d = np.load(src, allow_pickle=True)
    return dict(
        alpha=float(d["calib_alpha"]),
        beta=float(d["calib_beta"]),
        tau_m=float(d["calib_tau_m"]),
        tau_ref=float(d["calib_tau_ref"]),
        r2=float(d["calib_r2"]),
    )


def siegert_wta_label(fps: list,
                      asym_bistable: float = ASYM_BISTABLE,
                      asym_monostable: float = ASYM_MONOSTABLE) -> int:
    """1 if the FP structure supports WTA, 0 otherwise."""
    if len(fps) >= 2:
        # Bistable: check that two FPs are sufficiently asymmetric.
        spreads = [abs(fp[0] - fp[1]) for fp in fps]
        return int(max(spreads) >= asym_bistable)
    if len(fps) == 1:
        return int(abs(fps[0][0] - fps[0][1]) >= asym_monostable)
    return 0


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else 1.0


def main():
    calib = load_calibration()
    print(f"Calibration (locked from closed_form/phase1):")
    for k, v in calib.items():
        print(f"  {k} = {v:.6f}")
    print()

    siegert = Siegert(
        V_th=1.0, V_r=0.0,
        tau_m=calib["tau_m"],
        tau_ref=calib["tau_ref"],
    )

    # Lift Phase 0 grid spec.
    p0 = np.load(HERE / "results" / "phase0" / "fcs_grid.npz",
                 allow_pickle=True)
    W_VALUES = p0["W_VALUES"]
    fcs_capable = p0["labels_capable"]
    fcs_lustre = p0["labels_lustre"]
    nW = len(W_VALUES)

    print(f"Siegert FP enumeration on {nW}x{nW} grid; p_thin={P_THIN}, "
          f"drive={DRIVE}\n")

    sieg_labels = np.zeros((nW, nW), dtype=int)
    n_fps = np.zeros((nW, nW), dtype=int)
    spread_grid = np.zeros((nW, nW))

    for i, w_21 in enumerate(W_VALUES):
        for j, w_12 in enumerate(W_VALUES):
            fps = find_all_fixed_points_contralateral(
                w12=float(w_12), w21=float(w_21),
                drive=DRIVE, p_thin=P_THIN,
                siegert=siegert,
                alpha=calib["alpha"], beta=calib["beta"],
            )
            n_fps[i, j] = len(fps)
            sieg_labels[i, j] = siegert_wta_label(fps)
            if fps:
                spread_grid[i, j] = max(abs(fp[0] - fp[1]) for fp in fps)

    j_capable = jaccard(sieg_labels, fcs_capable)
    j_lustre = jaccard(sieg_labels, fcs_lustre)

    print(f"Siegert WTA-capable cells: {sieg_labels.sum()}/{sieg_labels.size}")
    print(f"  Jaccard vs Phase 0 WTA_CAPABLE: {j_capable:.3f}  (gate >= 0.70)")
    print(f"  Jaccard vs Phase 0 LUSTRE:      {j_lustre:.3f}")
    print()
    print(f"FP-count distribution: "
          f"0 FPs={int((n_fps == 0).sum())}, "
          f"1 FP={int((n_fps == 1).sum())}, "
          f"2 FPs={int((n_fps == 2).sum())}, "
          f"3+ FPs={int((n_fps >= 3).sum())}")

    np.savez(
        RESULTS / "siegert_grid.npz",
        W_VALUES=W_VALUES,
        sieg_labels=sieg_labels,
        n_fps=n_fps,
        spread_grid=spread_grid,
        jaccard_capable=j_capable,
        jaccard_lustre=j_lustre,
        **calib,
        drive=DRIVE, p_thin=P_THIN,
    )

    # Side-by-side plot.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, labels, title in [
        (axes[0], fcs_capable,
            f"FCS oracle (Phase 0 WTA_CAPABLE)\n"
            f"{int(fcs_capable.sum())}/{fcs_capable.size} blue"),
        (axes[1], sieg_labels,
            f"Siegert FP enumeration\n"
            f"{int(sieg_labels.sum())}/{sieg_labels.size} blue\n"
            f"Jaccard = {j_capable:.3f}"),
        (axes[2], (sieg_labels != fcs_capable).astype(int),
            f"Disagreement cells\n"
            f"{int((sieg_labels != fcs_capable).sum())} mismatches"),
    ]:
        for i, w_21 in enumerate(W_VALUES):
            for j, w_12 in enumerate(W_VALUES):
                color = "tab:blue" if labels[i, j] else "tab:red"
                if title.startswith("Disagreement") and labels[i, j]:
                    color = "black"
                elif title.startswith("Disagreement") and not labels[i, j]:
                    color = "lightgray"
                ax.scatter(int(w_12), int(w_21), c=color, s=70,
                           edgecolor="white", linewidth=0.5)
        ax.set_xlabel(r"$w_{12}$ (FCS scaled)")
        ax.set_ylabel(r"$w_{21}$ (FCS scaled)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
    fig.suptitle(
        "Phase 1: Siegert (static) reading of FCS Property 7", fontsize=11,
    )
    plt.tight_layout()
    out_pdf = RESULTS / "siegert_vs_fcs.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")

    # Spread heatmap (continuous WTA strength).
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
    im = ax.imshow(
        spread_grid, origin="lower", cmap="viridis",
        vmin=0, vmax=1,
        extent=[W_VALUES[0], W_VALUES[-1], W_VALUES[0], W_VALUES[-1]],
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, label=r"max $|\nu_1^* - \nu_2^*|$ over FPs")
    ax.set_xlabel(r"$w_{12}$ (FCS scaled)")
    ax.set_ylabel(r"$w_{21}$ (FCS scaled)")
    ax.set_title("Siegert WTA spread (continuous)")
    plt.tight_layout()
    out_pdf = RESULTS / "siegert_spread.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")

    print()
    if j_capable >= 0.70:
        print(f"Phase 1 PASS: Jaccard {j_capable:.3f} >= 0.70 gate.")
    else:
        print(f"Phase 1 FAIL: Jaccard {j_capable:.3f} < 0.70 gate.")


if __name__ == "__main__":
    main()
