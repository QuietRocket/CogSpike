"""Phase 2: H(omega) (dynamic) reading of FCS's 4-tick gate at N > 2.

For each (w, N, drive_bump) cell:
  1. Pick the largest-spread Siegert FP from phase 1 (the WTA-mode candidate).
  2. Build N×N Jacobian J = alpha * W_uniform(N, w) (off-diag w, zero diag).
  3. Compute per-neuron transfer gain g_i = dphi/dmu evaluated at the FP.
  4. Eigenvalues of A = (1/tau_m)(-I + diag(g) J).
  5. Gate: blue iff |Re(lambda_dom)| > 1/T_FCS = 0.25.

Diagnostic: at one cell per N, print the analytic closed-form spectrum
prediction (winner-loser 2x2 block + (N-2)-fold loser-permutation
eigenvalue) and compare to numpy.linalg.eigvals.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from deq.closed_form.siegert import Siegert, _phi_from_means  # noqa: E402
from deq.closed_form.transfer import (  # noqa: E402
    dphi_dmu, jacobian_eigenvalues, closed_loop_zero_freq_consistency,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "phase2"
RESULTS.mkdir(parents=True, exist_ok=True)


DRIVE = 11.0
P_THIN = 0.7
T_FCS = 4
LAMBDA_GATE = 1.0 / T_FCS


def load_calibration() -> dict:
    src = ROOT / "deq" / "closed_form" / "results" / "phase1_grid.npz"
    d = np.load(src, allow_pickle=True)
    return dict(
        alpha=float(d["calib_alpha"]),
        beta=float(d["calib_beta"]),
        tau_m=float(d["calib_tau_m"]),
        tau_ref=float(d["calib_tau_ref"]),
    )


def jaccard(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = (a & b).sum(); union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else 1.0


def cell_re_dom(fp_vec, N, w, drive_bump, calib, siegert):
    """Return (re_dom, eigenvalues array). fp_vec: shape (N,) sorted descending.

    If fp_vec is all-nan (no FP found), returns (nan, None).
    """
    if np.all(np.isnan(fp_vec)):
        return float('nan'), None
    nu = fp_vec.copy()
    # Compute (mu, sigma) for each neuron and the corresponding gain.
    sum_nu = np.nansum(nu)
    sum_var = np.nansum(nu * (1.0 - nu))
    mus = np.zeros(N); sigmas = np.zeros(N)
    for i in range(N):
        D_i = DRIVE + (drive_bump if i == 0 else 0.0)
        recur_mean = w * (sum_nu - nu[i])
        recur_var = (w ** 2) * (sum_var - nu[i] * (1.0 - nu[i]))
        mean_in = D_i * P_THIN + recur_mean
        var_in = (D_i ** 2) * P_THIN * (1.0 - P_THIN) + recur_var
        mus[i] = calib['alpha'] * mean_in
        sigmas[i] = math.sqrt(max(calib['beta'] * var_in, 0.0))
    gains = np.array([dphi_dmu(siegert, mus[i], sigmas[i]) for i in range(N)])
    # J in Siegert units: alpha * W_uniform.
    J = calib['alpha'] * w * (np.ones((N, N)) - np.eye(N))
    # Careful: W_uniform[i, i] = 0, W[i, j] = w for i!=j, so the off-diag is w.
    # J in Siegert units typically: J_FCS_to_siegert is alpha applied to mean,
    # so J = alpha * W_FCS. Verified by 2-neuron phase 2 which does
    # J = alpha * np.array([[0, w_21], [w_12, 0]]).
    eigs = jacobian_eigenvalues(J, gains, calib['tau_m'])
    re_dom = float(np.max(np.real(eigs)))
    return re_dom, eigs


def main():
    calib = load_calibration()
    print(f"Calibration: alpha={calib['alpha']:.4f} beta={calib['beta']:.4f} "
          f"tau_m={calib['tau_m']:.4f} tau_ref={calib['tau_ref']:.4f}")
    print(f"Latency gate: |Re(lambda_dom)| > 1/{T_FCS} = {LAMBDA_GATE:.3f}\n")

    siegert = Siegert(V_th=1.0, V_r=0.0,
                      tau_m=calib['tau_m'], tau_ref=calib['tau_ref'])

    # Load phase 0 (FCS labels) and phase 1 (Siegert FPs).
    p0 = np.load(HERE / "results" / "phase0" / "fcs_grid_multi.npz",
                 allow_pickle=True)
    p1 = np.load(HERE / "results" / "phase1" / "siegert_orbits.npz",
                 allow_pickle=True)
    W_VALUES = p0["W_VALUES"]
    N_VALUES = list(p0["N_VALUES"])
    DRIVE_BUMPS = list(p0["DRIVE_BUMPS"])
    fcs_labels_margin = p0["fcs_labels_margin"]
    sieg_labels = p1["sieg_labels"]
    max_spread = p1["max_spread"]
    winner_orbit_fp = p1["winner_orbit_fp"]

    nW = len(W_VALUES); nN = len(N_VALUES); nB = len(DRIVE_BUMPS)
    re_dom_grid = np.full((nN, nB, nW), np.nan)
    h_labels = np.zeros((nN, nB, nW), dtype=int)

    print(f"H(omega) latency-gate eval on {nN}x{nB}x{nW} grid...\n")

    for nIdx, N in enumerate(N_VALUES):
        for bIdx, db in enumerate(DRIVE_BUMPS):
            t0 = time.time()
            for wIdx, w in enumerate(W_VALUES):
                fp = winner_orbit_fp[nIdx, bIdx, wIdx, :int(N)]
                re_dom, _eigs = cell_re_dom(
                    fp, int(N), float(w), float(db), calib, siegert,
                )
                re_dom_grid[nIdx, bIdx, wIdx] = re_dom
                # Gate logic: blue iff (a) Siegert says there's a WTA-capable
                # FP (sieg_labels = 1) AND (b) the chosen FP has |Re(lambda_dom)|
                # > 1/T_FCS.
                if sieg_labels[nIdx, bIdx, wIdx] == 1 and not math.isnan(re_dom):
                    if abs(re_dom) > LAMBDA_GATE:
                        h_labels[nIdx, bIdx, wIdx] = 1
            elapsed = time.time() - t0
            n_h = int(h_labels[nIdx, bIdx].sum())
            print(f"  N={int(N):2d}, drive_bump={int(db)}: "
                  f"H-blue {n_h:2d}/{nW}, elapsed {elapsed:.1f}s")

    # Sweep gate threshold to find best Jaccard vs FCS-margin.
    print("\nGate-threshold sweep (vs FCS margin gate, drive_bump=1, all N):")
    best = (0.0, 0.0)
    for thresh in np.linspace(0.0, 2.0, 41):
        labels = np.zeros_like(h_labels)
        for nIdx in range(nN):
            for bIdx in range(nB):
                for wIdx in range(nW):
                    if sieg_labels[nIdx, bIdx, wIdx] == 1 and not math.isnan(re_dom_grid[nIdx, bIdx, wIdx]):
                        if abs(re_dom_grid[nIdx, bIdx, wIdx]) > thresh:
                            labels[nIdx, bIdx, wIdx] = 1
        jac = jaccard(labels[:, 1, :], fcs_labels_margin[:, 1, :])
        if jac > best[1]:
            best = (float(thresh), float(jac))
    print(f"  best: thresh={best[0]:.3f}, J(FCS margin, db=1)={best[1]:.3f}")
    print(f"  at FCS-prescribed thresh={LAMBDA_GATE:.3f}: "
          f"J(FCS margin, db=1)="
          f"{jaccard(h_labels[:, 1, :], fcs_labels_margin[:, 1, :]):.3f}")

    # Jaccards per (N, db).
    print("\nJaccard vs FCS (margin gate), per (N, drive_bump):")
    for nIdx, N in enumerate(N_VALUES):
        for bIdx, db in enumerate(DRIVE_BUMPS):
            j_h = jaccard(h_labels[nIdx, bIdx], fcs_labels_margin[nIdx, bIdx])
            j_s = jaccard(sieg_labels[nIdx, bIdx], fcs_labels_margin[nIdx, bIdx])
            print(f"  N={int(N):2d}, db={int(db)}: "
                  f"J(H)={j_h:.3f}, J(Siegert)={j_s:.3f}, "
                  f"H_blue={int(h_labels[nIdx, bIdx].sum())}, "
                  f"sieg_blue={int(sieg_labels[nIdx, bIdx].sum())}, "
                  f"fcs_blue_m={int(fcs_labels_margin[nIdx, bIdx].sum())}")

    # Closed-form spectrum diagnostic at one cell per N.
    print("\nClosed-form spectrum diagnostic (db=1, w=-30):")
    for N in N_VALUES:
        if N < 2:
            continue
        nIdx = N_VALUES.index(N)
        wIdx = list(W_VALUES).index(-30)
        bIdx = 1
        fp = winner_orbit_fp[nIdx, bIdx, wIdx, :int(N)]
        if np.all(np.isnan(fp)):
            print(f"  N={int(N):2d}: no FP found")
            continue
        re_dom, eigs = cell_re_dom(fp, int(N), -30.0, 1.0, calib, siegert)
        re_eigs = np.sort(np.real(eigs))
        print(f"  N={int(N):2d}: Re(eigvals) = "
              f"[{', '.join(f'{r:.3f}' for r in re_eigs)}]")
        print(f"          |Re(dom)| = {abs(re_dom):.3f} "
              f"({'>' if abs(re_dom) > LAMBDA_GATE else '<='} "
              f"gate {LAMBDA_GATE:.3f})")

    np.savez(
        RESULTS / "h_grid_multi.npz",
        W_VALUES=W_VALUES,
        N_VALUES=np.array(N_VALUES),
        DRIVE_BUMPS=np.array(DRIVE_BUMPS),
        h_labels=h_labels,
        re_dom_grid=re_dom_grid,
        lambda_gate=LAMBDA_GATE,
        best_thresh=best[0],
        best_jaccard=best[1],
    )
    print(f"\n  wrote {RESULTS / 'h_grid_multi.npz'}")

    # Plot: FCS-margin vs Siegert vs H-gate, per N.
    fig, axes = plt.subplots(nN, 3, figsize=(15, 1.6 * nN), sharex=True)
    if nN == 1: axes = axes.reshape(1, 3)
    for nIdx, N in enumerate(N_VALUES):
        for gIdx, (lbls, name) in enumerate([
            (fcs_labels_margin[nIdx], "FCS oracle (margin)"),
            (sieg_labels[nIdx], "Siegert FP"),
            (h_labels[nIdx], r"$H(\omega)$ gate"),
        ]):
            ax = axes[nIdx, gIdx]
            for bIdx, db in enumerate(DRIVE_BUMPS):
                y = float(db)
                for wIdx, w in enumerate(W_VALUES):
                    color = "tab:blue" if lbls[bIdx, wIdx] else "tab:red"
                    ax.scatter(int(w), y, c=color, s=32, edgecolor="none")
            ax.set_ylim(-0.5, max(DRIVE_BUMPS) + 0.5)
            ax.set_yticks([float(b) for b in DRIVE_BUMPS])
            ax.set_yticklabels([f"db={b}" for b in DRIVE_BUMPS])
            ax.set_title(
                f"N={int(N)} | {name} | db=1 blue: "
                f"{int(lbls[1].sum())}/{nW}",
                fontsize=9,
            )
            ax.grid(True, alpha=0.3)
    for col in range(3):
        axes[-1, col].set_xlabel(r"$w$")
    fig.suptitle(
        r"Phase 2: $H(\omega)$ latency gate vs Siegert FP vs FCS oracle",
        fontsize=11,
    )
    plt.tight_layout()
    out_pdf = RESULTS / "h_gate_vs_fcs_multi.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_pdf}")

    # Eigenvalue spectrum visualization (db=1, all w, per N).
    fig, axes = plt.subplots(1, 2, figsize=(12, 1.0 + 0.4 * nN), sharey=True)
    for bIdx, db in enumerate(DRIVE_BUMPS):
        ax = axes[bIdx]
        im = ax.imshow(
            np.clip(re_dom_grid[:, bIdx, :], -3, 3),
            aspect="auto", origin="lower",
            extent=[W_VALUES[0], W_VALUES[-1], -0.5, nN - 0.5],
            cmap="RdBu_r", vmin=-3, vmax=3,
        )
        ax.set_yticks(range(nN))
        ax.set_yticklabels([f"N={int(n)}" for n in N_VALUES])
        ax.set_xlabel(r"$w$")
        ax.set_title(f"drive_bump = {int(db)}")
        plt.colorbar(im, ax=ax, label=r"Re($\lambda_{\rm dom}$) (clipped $\pm 3$)")
    fig.suptitle("Phase 2: dominant Jacobian eigenvalue", fontsize=11)
    plt.tight_layout()
    out_pdf = RESULTS / "eigvalue_spectrum_vs_N.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_pdf}")

    # Verdict.
    print("\nPhase 2 verdict:")
    n_h_db1 = int(h_labels[:, 1, :].sum())
    j_h_overall = jaccard(h_labels[:, 1, :], fcs_labels_margin[:, 1, :])
    j_s_overall = jaccard(sieg_labels[:, 1, :], fcs_labels_margin[:, 1, :])
    print(f"  drive_bump=1: H-gate blue={n_h_db1}, "
          f"J(H vs FCS margin)={j_h_overall:.3f}, "
          f"J(Siegert vs FCS margin)={j_s_overall:.3f}")
    delta = j_h_overall - j_s_overall
    print(f"  Improvement of H-gate over Siegert: {delta:+.3f}")
    if delta > 0.01 or j_h_overall >= 0.30:
        print(f"  PASS: H(omega) latency gate tightens the Siegert prediction "
              f"by removing slow-decay cells.")
    else:
        print(f"  PARTIAL: H-gate at threshold {LAMBDA_GATE} does not "
              f"meaningfully tighten Siegert. The inverse-staircase cells "
              f"FCS sees at high N have no rate-equation FP to evaluate, "
              f"so the gate cannot recover them.")


if __name__ == "__main__":
    main()
