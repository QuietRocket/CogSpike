"""Phase 2: H(omega) (dynamic) reading of FCS's 4-tick gate.

Predicts decision latency at each (w_12, w_21) cell from the rate-equation
Jacobian eigenvalues at the Siegert fixed points. Hypothesis:

    FCS-blue (stable WTA in <= 4 ticks)  iff  |Re(lambda_dom)| > 1/T_FCS = 1/4

where lambda_dom is the dominant eigenvalue of A = (1/tau_m)(-I + diag(g)*J),
J the contralateral connectivity in Siegert units (alpha * W_FCS), g_i =
dPhi/dmu evaluated at the operating point. Three regimes per cell:

  • bistable (3 FPs): saddle middle FP has Re(lambda) > 0 (unstable).
    Decision speed = saddle escape rate. We evaluate at the symmetric
    FP (ties for the saddle).
  • mono-stable asymmetric (1 FP, |nu_1 - nu_2| >= 0.30): convergence
    speed = -max Re(lambda) at the FP.
  • mono-stable symmetric (1 FP, |nu_1 - nu_2| < 0.30): no WTA possible,
    cell labeled red regardless of eigenvalue.

The H(omega) gate's expected effect: it shrinks Phase 1's Siegert-blue
envelope by removing slow-decay cells. Diagonal-staircase cells (small
asymmetry) lie near the bifurcation, so |Re(lambda)| is small there - the
gate excludes them, recovering FCS-red. This closes the spike-timing-lock
gap that smooth-rate fixed-point structure alone could not see.
"""

from __future__ import annotations

import math
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
from deq.closed_form.transfer import (  # noqa: E402
    dphi_dmu,
    jacobian_eigenvalues,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "phase2"
RESULTS.mkdir(parents=True, exist_ok=True)


DRIVE = 11.0
P_THIN = 0.7
T_FCS = 4
LAMBDA_GATE = 1.0 / T_FCS    # |Re(lambda_dom)| threshold
ASYM_MONO = 0.30


def load_calibration() -> dict:
    src = ROOT / "deq" / "closed_form" / "results" / "phase1_grid.npz"
    d = np.load(src, allow_pickle=True)
    return dict(
        alpha=float(d["calib_alpha"]),
        beta=float(d["calib_beta"]),
        tau_m=float(d["calib_tau_m"]),
        tau_ref=float(d["calib_tau_ref"]),
    )


def fp_inputs(fp: np.ndarray, w_12: float, w_21: float,
              alpha: float, beta: float) -> tuple:
    """At fixed point fp = (nu_1*, nu_2*), compute (mu_i*, sigma_i*) for i=1,2."""
    mus, sigmas = [], []
    for i in range(2):
        j = 1 - i
        w_ji = w_21 if i == 0 else w_12
        mean_in = DRIVE * P_THIN + w_ji * fp[j]
        var_in = (DRIVE ** 2) * P_THIN * (1 - P_THIN) + (w_ji ** 2) * fp[j] * (1 - fp[j])
        mus.append(alpha * mean_in)
        sigmas.append(math.sqrt(max(beta * var_in, 0.0)))
    return np.array(mus), np.array(sigmas)


def cell_lambda(fps: list, w_12: float, w_21: float,
                siegert: Siegert, alpha: float, tau_m: float) -> tuple:
    """Return (lambda_dom_re, regime) where regime in {'bistable','mono_asym',
    'mono_sym','no_fp'}."""
    if not fps:
        return 0.0, "no_fp"

    # J in Siegert units. alpha * W_FCS, where W_FCS[i, j] is weight from
    # neuron j to neuron i.
    J = alpha * np.array([[0.0, w_21], [w_12, 0.0]])

    spreads = [abs(fp[0] - fp[1]) for fp in fps]
    n = len(fps)

    if n >= 3:
        # Bistable: pick the most-symmetric FP (the saddle).
        idx = int(np.argmin(spreads))
        regime = "bistable"
    elif n == 1:
        if spreads[0] >= ASYM_MONO:
            idx = 0
            regime = "mono_asym"
        else:
            return 0.0, "mono_sym"   # no WTA possible
    else:  # n == 2 (degenerate fold)
        idx = int(np.argmax(spreads))
        regime = "fold"

    fp = fps[idx]
    mus, sigmas = fp_inputs(fp, w_12, w_21, alpha,
                            siegert.tau_ref)  # beta passed elsewhere
    # Actually we need beta in fp_inputs; pass it explicitly via closure:
    return None  # unreachable, replaced by the closure version below


def cell_lambda_full(fps: list, w_12: float, w_21: float,
                     siegert: Siegert, alpha: float, beta: float,
                     tau_m: float) -> tuple:
    """Return (lambda_re_dom, regime, fp_chosen)."""
    if not fps:
        return 0.0, "no_fp", None

    J = alpha * np.array([[0.0, w_21], [w_12, 0.0]])
    spreads = [abs(fp[0] - fp[1]) for fp in fps]
    n = len(fps)

    if n >= 3:
        idx = int(np.argmin(spreads))
        regime = "bistable"
    elif n == 1:
        if spreads[0] >= ASYM_MONO:
            idx = 0
            regime = "mono_asym"
        else:
            return -1.0, "mono_sym", fps[0]   # negative => no WTA
    else:  # n == 2
        idx = int(np.argmax(spreads))
        regime = "fold"

    fp = fps[idx]
    # Compute (mu, sigma) at this FP for both populations.
    mus, sigmas = [], []
    for i in range(2):
        j = 1 - i
        w_ji = w_21 if i == 0 else w_12
        mean_in = DRIVE * P_THIN + w_ji * fp[j]
        var_in = (DRIVE ** 2) * P_THIN * (1 - P_THIN) + (w_ji ** 2) * fp[j] * (1 - fp[j])
        mus.append(alpha * mean_in)
        sigmas.append(math.sqrt(max(beta * var_in, 0.0)))
    mus = np.array(mus)
    sigmas = np.array(sigmas)

    gains = np.array([dphi_dmu(siegert, mus[i], sigmas[i]) for i in range(2)])
    eigs = jacobian_eigenvalues(J, gains, tau_m)
    # Dominant: eigenvalue with largest Re (most positive / least negative).
    re_dom = float(np.max(np.real(eigs)))
    return re_dom, regime, fp


def main():
    calib = load_calibration()
    print("Calibration (locked):")
    for k, v in calib.items():
        print(f"  {k} = {v:.6f}")
    print()

    siegert = Siegert(V_th=1.0, V_r=0.0,
                      tau_m=calib["tau_m"], tau_ref=calib["tau_ref"])

    p0 = np.load(HERE / "results" / "phase0" / "fcs_grid.npz", allow_pickle=True)
    W_VALUES = p0["W_VALUES"]
    fcs_labels = p0["fcs_labels"]
    nW = len(W_VALUES)

    # Phase 1's spread-based labels (the smooth-rate "envelope").
    p1 = np.load(HERE / "results" / "phase1" / "siegert_grid.npz",
                 allow_pickle=True)
    sieg_labels = p1["sieg_labels"]

    print(f"H(omega) latency-gate reading on {nW}x{nW} grid; "
          f"|Re(lambda)| gate = 1/{T_FCS} = {LAMBDA_GATE:.3f}\n")

    re_dom_grid = np.zeros((nW, nW))
    regime_grid = np.empty((nW, nW), dtype=object)

    for i, w_21 in enumerate(W_VALUES):
        for j, w_12 in enumerate(W_VALUES):
            fps = find_all_fixed_points_contralateral(
                w12=float(w_12), w21=float(w_21),
                drive=DRIVE, p_thin=P_THIN,
                siegert=siegert,
                alpha=calib["alpha"], beta=calib["beta"],
            )
            re_dom, regime, _ = cell_lambda_full(
                fps, float(w_12), float(w_21),
                siegert, calib["alpha"], calib["beta"], calib["tau_m"],
            )
            re_dom_grid[i, j] = re_dom
            regime_grid[i, j] = regime

    # H-gate: cell blue iff |Re(lambda_dom)| > 1/T_FCS AND regime supports WTA.
    # For mono_sym: red regardless. For others: gate on |Re(lambda)|.
    h_labels = np.zeros((nW, nW), dtype=int)
    for i in range(nW):
        for j in range(nW):
            if regime_grid[i, j] in ("bistable", "mono_asym", "fold"):
                if abs(re_dom_grid[i, j]) > LAMBDA_GATE:
                    h_labels[i, j] = 1

    # Sweep gate threshold to report the best.
    print("Gate sweep (|Re(lambda)| threshold vs Jaccard with FCS):")
    best = (0.0, 0.0)
    for thresh in np.linspace(0.0, 2.0, 41):
        labels = np.zeros((nW, nW), dtype=int)
        for i in range(nW):
            for j in range(nW):
                if regime_grid[i, j] in ("bistable", "mono_asym", "fold"):
                    if abs(re_dom_grid[i, j]) > thresh:
                        labels[i, j] = 1
        inter = (labels.astype(bool) & fcs_labels.astype(bool)).sum()
        union = (labels.astype(bool) | fcs_labels.astype(bool)).sum()
        jac = inter / union if union > 0 else 1.0
        if jac > best[1]:
            best = (thresh, jac)
    print(f"  best: thresh={best[0]:.3f}, jaccard={best[1]:.3f}")

    # Use the FCS-prescribed threshold = 1/T_FCS.
    j_h = (h_labels.astype(bool) & fcs_labels.astype(bool)).sum() / max(
        (h_labels.astype(bool) | fcs_labels.astype(bool)).sum(), 1)

    print(f"\nH-gate at threshold {LAMBDA_GATE:.3f} (1/{T_FCS}):")
    print(f"  H-gate blue cells: {int(h_labels.sum())}/{h_labels.size}")
    print(f"  Jaccard vs FCS: {j_h:.3f}  (gate >= 0.70)")
    print(f"  Phase 1 Siegert Jaccard was: {p1['jaccard_fcs']:.3f}")
    improvement = j_h - float(p1["jaccard_fcs"])
    print(f"  Improvement over Phase 1: {improvement:+.3f}")

    np.savez(
        RESULTS / "h_grid.npz",
        W_VALUES=W_VALUES,
        h_labels=h_labels,
        re_dom_grid=re_dom_grid,
        regime_grid=regime_grid.astype(str),
        jaccard_h=j_h,
        best_thresh=best[0],
        best_jaccard=best[1],
        lambda_gate=LAMBDA_GATE,
    )

    # ----- plotting -----
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    panel_specs = [
        (axes[0], fcs_labels, "boolean",
            f"FCS oracle\n{int(fcs_labels.sum())}/{fcs_labels.size} blue"),
        (axes[1], sieg_labels, "boolean",
            f"Phase 1: Siegert FP\n{int(sieg_labels.sum())}/{sieg_labels.size} blue, "
            f"J={float(p1['jaccard_fcs']):.3f}"),
        (axes[2], h_labels, "boolean",
            f"Phase 2: |Re(λ)|>{LAMBDA_GATE:.2f} gate\n"
            f"{int(h_labels.sum())}/{h_labels.size} blue, J={j_h:.3f}"),
        (axes[3], np.clip(re_dom_grid, -2, 2), "continuous",
            r"Re($\lambda_{\rm dom}$) (clipped to $\pm 2$)"),
    ]
    for ax, data, kind, title in panel_specs:
        if kind == "boolean":
            for i, w_21 in enumerate(W_VALUES):
                for j, w_12 in enumerate(W_VALUES):
                    color = "tab:blue" if data[i, j] else "tab:red"
                    ax.scatter(int(w_12), int(w_21), c=color, s=22,
                               edgecolor="none")
        else:
            im = ax.imshow(
                data, origin="lower", cmap="RdBu_r",
                vmin=-2, vmax=2,
                extent=[W_VALUES[0], W_VALUES[-1], W_VALUES[0], W_VALUES[-1]],
                aspect="equal",
            )
            plt.colorbar(im, ax=ax, label=r"Re($\lambda_{\rm dom}$)")
        ax.set_xlabel(r"$w_{12}$")
        ax.set_ylabel(r"$w_{21}$")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
    fig.suptitle(
        "Phase 2: H(omega) eigenvalue gate vs FCS Property 7", fontsize=11,
    )
    plt.tight_layout()
    out_pdf = RESULTS / "h_gate_vs_fcs.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\n  wrote {out_pdf}")

    print()
    if j_h >= 0.70 or improvement >= 0.05:
        print(f"Phase 2 PASS (J={j_h:.3f}, improvement {improvement:+.3f} "
              f"over Phase 1 Siegert).")
    else:
        print(f"Phase 2 PARTIAL (J={j_h:.3f}, improvement {improvement:+.3f}); "
              f"latency gate alone insufficient.")


if __name__ == "__main__":
    main()
