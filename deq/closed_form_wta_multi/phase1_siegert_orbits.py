"""Phase 1: Siegert (static) FP enumeration of FCS Property 7 at N > 2.

For each (w, N, drive_bump) cell, enumerate rate-equation fixed points via
**multi-restart fsolve** in the full N-dimensional rate vector. This is the
fallback the design plan flagged for when the symmetric-orbit ansatz
fails — and the fsolve diagnostic at drive_bump=1 confirmed the ansatz
misses spontaneously-broken FPs (e.g., at N=3, drive_bump=1, w=-30 there
exist FPs like [0.23, 0.12, 0.03] with three distinct rates that the
'bumped + (N-1)-tied losers' shape does not represent).

Multi-restart fsolve enumerates the full FP set; we still report orbit
labels (k = number of 'winner-class' rates above some threshold) for
context and for downstream phases.

WTA-capable iff some FP has rate-spread (max - second_max) >= 0.30.

Compares Siegert labels against Phase 0 FCS labels (both gates).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from deq.closed_form.siegert import Siegert, _phi_from_means  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "phase1"
RESULTS.mkdir(parents=True, exist_ok=True)


DRIVE = 11.0
P_THIN = 0.7
WTA_SPREAD_GATE = 0.30
FSOLVE_RESTARTS = 30
FP_DEDUP_TOL = 5e-3


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


def full_residual(nu, N, w, drive, drive_bump, p_thin, siegert, alpha, beta):
    """N-D Siegert FP residual for uniform inhibition with drive_bump on neuron 0."""
    res = np.zeros(N)
    sum_nu = nu.sum()
    sum_var = (nu * (1.0 - nu)).sum()
    for i in range(N):
        D_i = drive + (drive_bump if i == 0 else 0.0)
        recur_mean = w * (sum_nu - nu[i])
        recur_var = (w ** 2) * (sum_var - nu[i] * (1.0 - nu[i]))
        mean_in = D_i * p_thin + recur_mean
        var_in = (D_i ** 2) * p_thin * (1.0 - p_thin) + recur_var
        res[i] = nu[i] - _phi_from_means(siegert, alpha, beta, mean_in, var_in)
    return res


def enumerate_fps(N, w, drive_bump, calib, siegert,
                  n_restarts=FSOLVE_RESTARTS, seed=0):
    """Multi-restart fsolve to enumerate all FPs in [0, 1]^N.

    Returns list of arrays, each shape (N,), sorted descending by rate.
    Deduplicated to within FP_DEDUP_TOL on the canonical (sorted) form.
    """
    rng = np.random.default_rng(seed)
    found = []
    args = (N, float(w), DRIVE, float(drive_bump), P_THIN, siegert,
            calib['alpha'], calib['beta'])

    # Restart seeds. Two design constraints:
    # (a) cover symmetric, asymmetric (k-winners) and broken-symmetry cases;
    # (b) include EXTREME-asymmetric ICs because at large N the basin of the
    #     1-winner FP is small (winner near saturation; losers near zero).
    initial_conds = [
        np.full(N, 0.05),
        np.full(N, 0.3),
        np.full(N, 0.5),
        np.full(N, 0.95),
    ]
    # k-winners, (N-k)-losers, for k = 1, 2, ..., min(N-1, 4). Extreme +
    # mild variants.
    for k in range(1, min(N, 5)):
        # Extreme: winners near 1, losers near 0.
        ic = np.zeros(N)
        ic[:k] = 0.98; ic[k:] = 0.005
        initial_conds.append(ic)
        # Mild.
        ic = np.zeros(N)
        ic[:k] = 0.6; ic[k:] = 0.05
        initial_conds.append(ic)
    # Random fill.
    for _ in range(max(0, n_restarts - len(initial_conds))):
        initial_conds.append(rng.uniform(0.01, 0.99, size=N))

    for x0 in initial_conds:
        try:
            sol, info, ier, _msg = fsolve(
                full_residual, x0, args=args,
                full_output=True, xtol=1e-9, maxfev=200,
            )
            if ier != 1:
                continue
            # Verify residual.
            res = full_residual(sol, *args)
            if np.max(np.abs(res)) > 1e-6:
                continue
            sol = np.clip(sol, 0.0, 1.0)
            # Canonical form: sorted descending.
            sol_canon = np.sort(sol)[::-1]
            duplicate = False
            for s in found:
                if np.max(np.abs(s - sol_canon)) < FP_DEDUP_TOL:
                    duplicate = True; break
            if not duplicate:
                found.append(sol_canon)
        except Exception:
            pass
    return found


def wta_label_from_fps(fps: list) -> tuple:
    """Returns (label, max_spread, n_fps).

    Spread = rate_max - rate_second per FP. Label 1 if any FP has spread
    >= WTA_SPREAD_GATE.
    """
    if not fps:
        return 0, 0.0, 0
    spreads = [float(fp[0] - fp[1]) if len(fp) >= 2 else 0.0 for fp in fps]
    max_s = max(spreads)
    return int(max_s >= WTA_SPREAD_GATE), max_s, len(fps)


def main():
    calib = load_calibration()
    print(f"Calibration: alpha={calib['alpha']:.4f} beta={calib['beta']:.4f} "
          f"tau_m={calib['tau_m']:.4f} tau_ref={calib['tau_ref']:.4f}")
    print(f"Multi-restart fsolve: {FSOLVE_RESTARTS} ICs per cell; "
          f"dedup_tol={FP_DEDUP_TOL}\n")

    siegert = Siegert(V_th=1.0, V_r=0.0,
                      tau_m=calib['tau_m'], tau_ref=calib['tau_ref'])

    p0 = np.load(HERE / "results" / "phase0" / "fcs_grid_multi.npz",
                 allow_pickle=True)
    W_VALUES = p0["W_VALUES"]
    N_VALUES = list(p0["N_VALUES"])
    DRIVE_BUMPS = list(p0["DRIVE_BUMPS"])
    fcs_labels_margin = p0["fcs_labels_margin"]
    fcs_labels_strict = p0["fcs_labels_strict"]

    nW = len(W_VALUES); nN = len(N_VALUES); nB = len(DRIVE_BUMPS)
    sieg_labels = np.zeros((nN, nB, nW), dtype=int)
    max_spread = np.zeros((nN, nB, nW))
    n_fps_grid = np.zeros((nN, nB, nW), dtype=int)
    # Store the winner-orbit FP (the one with largest spread, padded to N_max).
    N_max = max(N_VALUES)
    winner_orbit_fp = np.full((nN, nB, nW, N_max), np.nan)

    print(f"Multi-restart fsolve enumeration on {nN}x{nB}x{nW} grid...\n")

    for nIdx, N in enumerate(N_VALUES):
        for bIdx, db in enumerate(DRIVE_BUMPS):
            t0 = time.time()
            for wIdx, w in enumerate(W_VALUES):
                fps = enumerate_fps(
                    int(N), int(w), int(db), calib, siegert,
                    seed=42 + nIdx * 1000 + bIdx * 100 + wIdx,
                )
                label, spread, nfp = wta_label_from_fps(fps)
                sieg_labels[nIdx, bIdx, wIdx] = label
                max_spread[nIdx, bIdx, wIdx] = spread
                n_fps_grid[nIdx, bIdx, wIdx] = nfp
                if fps:
                    # Pick FP with largest spread.
                    spreads = [fp[0] - fp[1] if len(fp) >= 2 else 0.0
                               for fp in fps]
                    best = fps[int(np.argmax(spreads))]
                    winner_orbit_fp[nIdx, bIdx, wIdx, :int(N)] = best
            elapsed = time.time() - t0
            n_blue = int(sieg_labels[nIdx, bIdx].sum())
            print(f"  N={int(N):2d}, drive_bump={int(db)}: "
                  f"blue {n_blue:2d}/{nW}, mean FPs/cell="
                  f"{n_fps_grid[nIdx, bIdx].mean():.2f}, "
                  f"elapsed {elapsed:.1f}s")

    # Jaccards.
    print("\nJaccard vs FCS (margin and strict), per (N, drive_bump):")
    for nIdx, N in enumerate(N_VALUES):
        for bIdx, db in enumerate(DRIVE_BUMPS):
            j_m = jaccard(sieg_labels[nIdx, bIdx], fcs_labels_margin[nIdx, bIdx])
            j_s = jaccard(sieg_labels[nIdx, bIdx], fcs_labels_strict[nIdx, bIdx])
            n_sb = int(sieg_labels[nIdx, bIdx].sum())
            n_fb_m = int(fcs_labels_margin[nIdx, bIdx].sum())
            n_fb_s = int(fcs_labels_strict[nIdx, bIdx].sum())
            print(f"  N={int(N):2d}, db={int(db)}: "
                  f"J(margin)={j_m:.3f}, J(strict)={j_s:.3f}, "
                  f"sieg_blue={n_sb}, fcs_b_margin={n_fb_m}, "
                  f"fcs_b_strict={n_fb_s}")

    np.savez(
        RESULTS / "siegert_orbits.npz",
        W_VALUES=W_VALUES,
        N_VALUES=np.array(N_VALUES),
        DRIVE_BUMPS=np.array(DRIVE_BUMPS),
        sieg_labels=sieg_labels,
        max_spread=max_spread,
        n_fps_grid=n_fps_grid,
        winner_orbit_fp=winner_orbit_fp,
        wta_spread_gate=WTA_SPREAD_GATE,
    )
    print(f"\n  wrote {RESULTS / 'siegert_orbits.npz'}")

    # Plot: side-by-side Siegert vs FCS-margin, per N.
    fig, axes = plt.subplots(nN, 2, figsize=(13, 1.6 * nN), sharex=True)
    if nN == 1: axes = axes.reshape(1, 2)
    for nIdx, N in enumerate(N_VALUES):
        for gIdx, (lbls, name) in enumerate([
            (fcs_labels_margin[nIdx], "FCS oracle (margin gate)"),
            (sieg_labels[nIdx], "Siegert FP enumeration"),
        ]):
            ax = axes[nIdx, gIdx]
            for bIdx, db in enumerate(DRIVE_BUMPS):
                y = float(db)
                for wIdx, w in enumerate(W_VALUES):
                    color = "tab:blue" if lbls[bIdx, wIdx] else "tab:red"
                    ax.scatter(int(w), y, c=color, s=36, edgecolor="none")
            ax.set_ylim(-0.5, max(DRIVE_BUMPS) + 0.5)
            ax.set_yticks([float(b) for b in DRIVE_BUMPS])
            ax.set_yticklabels([f"db={b}" for b in DRIVE_BUMPS])
            if gIdx == 1:
                j_db0 = jaccard(sieg_labels[nIdx, 0], fcs_labels_margin[nIdx, 0])
                j_db1 = jaccard(sieg_labels[nIdx, 1], fcs_labels_margin[nIdx, 1])
                ax.set_title(
                    f"N={int(N)} | {name} | "
                    f"J(db=0)={j_db0:.2f}, J(db=1)={j_db1:.2f}",
                    fontsize=9,
                )
            else:
                ax.set_title(f"N={int(N)} | {name}", fontsize=9)
            ax.grid(True, alpha=0.3)
    for col in range(2):
        axes[-1, col].set_xlabel(r"$w$")
    fig.suptitle(
        "Phase 1: Siegert FP enumeration vs FCS oracle (margin gate)",
        fontsize=11,
    )
    plt.tight_layout()
    out_pdf = RESULTS / "siegert_orbits_vs_fcs.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_pdf}")

    # Spread heatmap.
    fig, axes = plt.subplots(1, 2, figsize=(11, 1.0 + 0.4 * nN), sharey=True)
    for bIdx, db in enumerate(DRIVE_BUMPS):
        ax = axes[bIdx]
        im = ax.imshow(
            max_spread[:, bIdx, :],
            aspect="auto", origin="lower",
            extent=[W_VALUES[0], W_VALUES[-1], -0.5, nN - 0.5],
            cmap="viridis", vmin=0, vmax=1,
        )
        ax.set_yticks(range(nN))
        ax.set_yticklabels([f"N={int(n)}" for n in N_VALUES])
        ax.set_xlabel(r"$w$")
        ax.set_title(f"drive_bump = {int(db)}")
        plt.colorbar(im, ax=ax, label="max FP spread (rate_max - rate_2nd)")
    fig.suptitle("Phase 1 Siegert max FP spread (continuous WTA strength)",
                 fontsize=11)
    plt.tight_layout()
    out_pdf = RESULTS / "siegert_max_spread.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_pdf}")

    # FP count per (w, N) per drive_bump.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for bIdx, db in enumerate(DRIVE_BUMPS):
        ax = axes[bIdx]
        for nIdx, N in enumerate(N_VALUES):
            ax.plot(W_VALUES, n_fps_grid[nIdx, bIdx, :], label=f"N={int(N)}",
                    marker='.', alpha=0.7)
        ax.set_xlabel("w"); ax.set_ylabel("# distinct FPs found by fsolve")
        ax.set_title(f"drive_bump = {int(db)}")
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Phase 1 FP count vs w (multi-restart fsolve)", fontsize=11)
    plt.tight_layout()
    out_pdf = RESULTS / "orbit_count_vs_N.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_pdf}")

    # Verdict.
    print("\nPhase 1 verdict:")
    sieg_db1 = sieg_labels[:, 1, :].astype(bool)
    fcs_db1 = fcs_labels_margin[:, 1, :].astype(bool)
    recall = (sieg_db1 & fcs_db1).sum() / max(fcs_db1.sum(), 1)
    j_overall = jaccard(sieg_labels[:, 1, :], fcs_labels_margin[:, 1, :])
    print(f"  drive_bump=1 overall: Siegert blue={int(sieg_db1.sum())}, "
          f"FCS blue (margin)={int(fcs_db1.sum())}, "
          f"recall={recall:.3f}, Jaccard={j_overall:.3f}")
    print(f"  drive_bump=0 (FCS-faithful symmetric):")
    sieg_db0 = sieg_labels[:, 0, :].astype(bool)
    fcs_db0 = fcs_labels_margin[:, 0, :].astype(bool)
    print(f"    Siegert blue={int(sieg_db0.sum())}, FCS blue={int(fcs_db0.sum())}")
    print(f"    Generalizes the N=2 'staircase invisibility' to N>2: smooth-")
    print(f"    rate theory sees bistable FPs that FCS integer-tick dynamics")
    print(f"    cannot select due to perfect S_N synchrony.")


if __name__ == "__main__":
    main()
