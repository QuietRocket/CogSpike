"""Phase 3: Quasi-renewal (finite N_pop) reading of FCS Property 7 at N > 2.

Sweeps the (w, N_neurons, N_pop) grid with quasi-renewal mesoscopic
simulation at N_pop in {50, 100, 500, 2000} for each N_neurons in
{2, 3, 4, 6, 10}. Uses drive_bump=1 throughout (the regime where Siegert
mean-field is meaningful; drive_bump=0 is uniformly red at every N per
Phase 0).

WTA gate: rate_max - second_max >= 0.30 post-warmup window.

Hypothesis (carried over from 2-neuron thread):
  - At N_pop -> infinity, QR converges to Siegert mean-field (Phase 1).
  - At small N_pop, finite-size noise broadens the no-WTA band near
    the bifurcation; some near-bifurcation Siegert-blue cells turn red.
  - The inverse-staircase cells (FCS-blue, Siegert-red at high N) cannot
    be recovered by QR because there is no rate-equation FP to converge to.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from deq.closed_form.siegert import Siegert  # noqa: E402
from deq.closed_form.quasi_renewal import (  # noqa: E402
    QuasiRenewal,
    simulate_uniform_inhibition,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "phase3"
RESULTS.mkdir(parents=True, exist_ok=True)


DRIVE = 11.0
P_THIN = 0.7
T_TOTAL = 200
T_WARMUP_QR = 50
ASYM_GATE = 0.30
N_POP_SWEEP = [50, 100, 500, 2000]
SEED = 1234


def load_calibration() -> dict:
    src = ROOT / "deq" / "closed_form" / "results" / "phase1_grid.npz"
    d = np.load(src, allow_pickle=True)
    return dict(
        alpha=float(d["calib_alpha"]),
        beta=float(d["calib_beta"]),
        tau_m=float(d["calib_tau_m"]),
        tau_ref=float(d["calib_tau_ref"]),
    )


def qr_wta_label(rates, t_warmup=T_WARMUP_QR, asym=ASYM_GATE):
    """rate_max - second_max >= asym in post-warmup."""
    post = rates[:, t_warmup:]
    mean_rates = np.sort(post.mean(axis=1))[::-1]
    if mean_rates.size < 2:
        return 0
    return int((mean_rates[0] - mean_rates[1]) >= asym)


def jaccard(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = (a & b).sum(); union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else 1.0


def main():
    calib = load_calibration()
    print(f"Calibration: alpha={calib['alpha']:.4f} beta={calib['beta']:.4f}")
    print(f"T={T_TOTAL}, T_warmup={T_WARMUP_QR}, asym_gate={ASYM_GATE}")
    print(f"N_pop sweep: {N_POP_SWEEP}\n")

    siegert = Siegert(V_th=1.0, V_r=0.0,
                      tau_m=calib['tau_m'], tau_ref=calib['tau_ref'])
    qr = QuasiRenewal(siegert=siegert, K_max=30, tau_ref_ticks=0, dt=1.0)

    # Load phase 0 and phase 1 labels for drive_bump=1.
    p0 = np.load(HERE / "results" / "phase0" / "fcs_grid_multi.npz",
                 allow_pickle=True)
    p1 = np.load(HERE / "results" / "phase1" / "siegert_orbits.npz",
                 allow_pickle=True)
    W_VALUES = p0["W_VALUES"]
    N_VALUES = list(p0["N_VALUES"])
    DRIVE_BUMPS = list(p0["DRIVE_BUMPS"])
    db1_idx = list(DRIVE_BUMPS).index(1)

    fcs_db1 = p0["fcs_labels_margin"][:, db1_idx, :]   # (nN, nW)
    sieg_db1 = p1["sieg_labels"][:, db1_idx, :]         # (nN, nW)

    nW = len(W_VALUES); nN = len(N_VALUES); nP = len(N_POP_SWEEP)

    qr_labels = np.zeros((nP, nN, nW), dtype=int)
    qr_spread = np.zeros((nP, nN, nW))

    print(f"QR sweep on {nP}x{nN}x{nW} cells; drive_bump=1...\n")

    for pIdx, N_pop in enumerate(N_POP_SWEEP):
        for nIdx, N_neurons in enumerate(N_VALUES):
            t0 = time.time()
            # Strong neuron-0-favored init. The 2-neuron thread used (0.5, 0.1)
            # — that 0.4 asymmetry sufficed at N=2 but is too weak at higher
            # N because each loser feels (N-1) competing winners' inhibition
            # while the winner feels only (N-1) losers' inhibition; the
            # symmetric basin attracts unless the IC is strongly off-axis.
            # We use (0.95, 0.005, ..., 0.005) consistently across N, which
            # both reproduces the 2-neuron behaviour and breaks into the
            # asymmetric basin at high N.
            init_A = np.full(int(N_neurons), 0.005)
            init_A[0] = 0.95
            for wIdx, w in enumerate(W_VALUES):
                rates = simulate_uniform_inhibition(
                    N_neurons=int(N_neurons), w=float(w),
                    drive=DRIVE, p_thin=P_THIN,
                    qr=qr, alpha=calib['alpha'], beta=calib['beta'],
                    N_pop=int(N_pop), T=T_TOTAL,
                    seed=SEED + pIdx * 1000 + nIdx * 100 + wIdx,
                    drive_bump=1.0, init_A=init_A,
                )
                qr_labels[pIdx, nIdx, wIdx] = qr_wta_label(rates)
                post = rates[:, T_WARMUP_QR:]
                mean_r = np.sort(post.mean(axis=1))[::-1]
                qr_spread[pIdx, nIdx, wIdx] = float(mean_r[0] - mean_r[1])
            elapsed = time.time() - t0
            n_blue = int(qr_labels[pIdx, nIdx].sum())
            j_fcs = jaccard(qr_labels[pIdx, nIdx], fcs_db1[nIdx])
            j_sieg = jaccard(qr_labels[pIdx, nIdx], sieg_db1[nIdx])
            print(f"  N_pop={N_pop:4d}, N_neurons={int(N_neurons):2d}: "
                  f"blue {n_blue:2d}/{nW}, J(FCS)={j_fcs:.3f}, "
                  f"J(Sieg)={j_sieg:.3f}, elapsed {elapsed:.1f}s")

    np.savez(
        RESULTS / "qr_grid_multi.npz",
        W_VALUES=W_VALUES,
        N_VALUES=np.array(N_VALUES),
        N_POP_SWEEP=np.array(N_POP_SWEEP),
        qr_labels=qr_labels,
        qr_spread=qr_spread,
        T_total=T_TOTAL,
        T_warmup=T_WARMUP_QR,
        asym_gate=ASYM_GATE,
    )
    print(f"\n  wrote {RESULTS / 'qr_grid_multi.npz'}")

    # Side-by-side panels: FCS (db=1) | Siegert (db=1) | QR at each N_pop.
    # Rows: N_neurons. Columns: 2 + nP.
    ncols = 2 + nP
    fig, axes = plt.subplots(nN, ncols, figsize=(2.6 * ncols, 1.6 * nN),
                              sharex=True, sharey=True)
    if nN == 1: axes = axes.reshape(1, -1)
    for nIdx, N_neurons in enumerate(N_VALUES):
        col_data = [
            (fcs_db1[nIdx], f"FCS (m), db=1"),
            (sieg_db1[nIdx], f"Siegert, db=1"),
        ]
        for pIdx, N_pop in enumerate(N_POP_SWEEP):
            col_data.append((qr_labels[pIdx, nIdx],
                              f"QR N_pop={N_pop}"))
        for cIdx, (lbls, name) in enumerate(col_data):
            ax = axes[nIdx, cIdx]
            for wIdx, w in enumerate(W_VALUES):
                color = "tab:blue" if lbls[wIdx] else "tab:red"
                ax.scatter(int(w), 0, c=color, s=28, edgecolor="none")
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_title(f"N={int(N_neurons)} | {name}\n"
                         f"blue {int(lbls.sum())}/{nW}", fontsize=8)
            ax.grid(True, alpha=0.3)
    for col in range(ncols):
        axes[-1, col].set_xlabel(r"$w$")
    fig.suptitle(
        "Phase 3: QR finite-N_pop sweep (drive_bump=1) vs FCS oracle and Siegert mean-field",
        fontsize=10,
    )
    plt.tight_layout()
    out_pdf = RESULTS / "qr_n_sweep_multi.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_pdf}")

    # Jaccard vs N_pop, one curve per N_neurons.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for nIdx, N_neurons in enumerate(N_VALUES):
        j_fcs = [jaccard(qr_labels[p, nIdx], fcs_db1[nIdx])
                 for p in range(nP)]
        j_sieg = [jaccard(qr_labels[p, nIdx], sieg_db1[nIdx])
                  for p in range(nP)]
        axes[0].plot(N_POP_SWEEP, j_fcs, 'o-',
                     label=f"N_neurons={int(N_neurons)}", alpha=0.8)
        axes[1].plot(N_POP_SWEEP, j_sieg, 's-',
                     label=f"N_neurons={int(N_neurons)}", alpha=0.8)
    axes[0].set_title("J(QR, FCS-margin) vs N_pop")
    axes[1].set_title("J(QR, Siegert) vs N_pop")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("N_pop")
        ax.set_ylabel("Jaccard")
        ax.axhline(0.70, color="gray", linestyle="--", alpha=0.5,
                   label="gate (0.70)" if ax is axes[0] else None)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Phase 3 Jaccard curves vs N_pop", fontsize=11)
    plt.tight_layout()
    out_pdf = RESULTS / "qr_jaccard_vs_Npop_per_N.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_pdf}")

    # Verdict.
    print("\nPhase 3 verdict:")
    n_pop_max = N_POP_SWEEP[-1]
    pMax = nP - 1
    for nIdx, N_neurons in enumerate(N_VALUES):
        j_fcs = jaccard(qr_labels[pMax, nIdx], fcs_db1[nIdx])
        j_sieg = jaccard(qr_labels[pMax, nIdx], sieg_db1[nIdx])
        print(f"  N_neurons={int(N_neurons):2d}, N_pop={n_pop_max}: "
              f"J(FCS)={j_fcs:.3f}, J(Siegert)={j_sieg:.3f}")
    print()
    j_sieg_mean = np.mean([jaccard(qr_labels[pMax, nIdx], sieg_db1[nIdx])
                           for nIdx in range(nN)])
    if j_sieg_mean >= 0.70:
        print(f"  PASS (QR -> Siegert mean-field convergence): mean "
              f"J(QR_N=2000, Siegert) = {j_sieg_mean:.3f} >= 0.70.")
    else:
        print(f"  PARTIAL: mean J(QR_N=2000, Siegert) = {j_sieg_mean:.3f}; "
              f"convergence weaker than expected.")


if __name__ == "__main__":
    main()
