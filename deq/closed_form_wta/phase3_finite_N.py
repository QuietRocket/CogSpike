"""Phase 3: Quasi-renewal (finite-N) reading of FCS Property 7.

Sweeps the 40x40 (w_12, w_21) FCS grid with quasi-renewal mesoscopic
simulation at N in {50, 100, 500, 2000}. Per-cell label: WTA-stable iff
post-warmup mean rates show clean asymmetry (rate_max >= 0.95,
rate_min <= 0.05).

Three hypotheses:
  • At N -> infinity, the boundary should converge to Phase 1 Siegert
    mean-field labels (the rate-equation envelope).
  • At small N, finite-size sqrt(A/N) noise broadens the no-WTA band
    near the rate-equation bifurcation boundary; some Siegert-blue cells
    near the boundary lose their commitment and turn red.
  • The diagonal-staircase cells (FCS-red, Siegert-blue) are NOT near
    the rate-equation bifurcation - they are deep in the rate-equation
    bistable basin. Quasi-renewal cannot recover FCS-red labels there
    because noise alone does not undo integer-tick synchronous locking.
    The 'spike-timing-lock' is genuinely beyond rate-equation theory.

Output: results/phase3/qr_grid.npz, qr_n_sweep.pdf, qr_jaccard_vs_N.pdf.
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
    simulate_contralateral,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "phase3"
RESULTS.mkdir(parents=True, exist_ok=True)


DRIVE = 11.0
P_THIN = 0.7
T_TOTAL = 200
T_WARMUP_QR = 50         # post-warmup window starts here
ASYM_GATE = 0.30         # rate-equation WTA: rate_max - rate_min >= ASYM_GATE
                         # (rate-eq winner saturates at ~0.5 not 1.0, unlike
                         #  deterministic LIF which fires every tick)
N_SWEEP = [50, 100, 500, 2000]
SEED = 1234
INIT_A = (0.5, 0.1)      # mild N1-favored seed (same as prior thread)


def load_calibration() -> dict:
    src = ROOT / "deq" / "closed_form" / "results" / "phase1_grid.npz"
    d = np.load(src, allow_pickle=True)
    return dict(
        alpha=float(d["calib_alpha"]),
        beta=float(d["calib_beta"]),
        tau_m=float(d["calib_tau_m"]),
        tau_ref=float(d["calib_tau_ref"]),
    )


def qr_wta_label(rates_2xT: np.ndarray, t_warmup: int = T_WARMUP_QR,
                 asym: float = ASYM_GATE) -> int:
    """Rate-asymmetry gate: rate_max - rate_min >= asym in post-warmup window."""
    post = rates_2xT[:, t_warmup:]
    mean_rates = post.mean(axis=1)
    return int((mean_rates.max() - mean_rates.min()) >= asym)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    inter = (a & b).sum(); union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else 1.0


def main():
    calib = load_calibration()
    print("Calibration (locked):", calib)
    print()

    siegert = Siegert(V_th=1.0, V_r=0.0,
                      tau_m=calib["tau_m"], tau_ref=calib["tau_ref"])
    qr = QuasiRenewal(siegert=siegert, K_max=30, tau_ref_ticks=0, dt=1.0)

    p0 = np.load(HERE / "results" / "phase0" / "fcs_grid.npz", allow_pickle=True)
    W_VALUES = p0["W_VALUES"]
    fcs_labels = p0["fcs_labels"]
    nW = len(W_VALUES)

    p1 = np.load(HERE / "results" / "phase1" / "siegert_grid.npz",
                 allow_pickle=True)
    sieg_labels = p1["sieg_labels"]

    print(f"Quasi-renewal sweep on {nW}x{nW} grid; "
          f"N sweep: {N_SWEEP}; T={T_TOTAL}; init_A={INIT_A}")
    print(f"WTA gate: rate_max - rate_min >= {ASYM_GATE} "
          f"(post-warmup window [t={T_WARMUP_QR}, T={T_TOTAL}])\n")

    qr_labels = {}     # N -> (nW, nW) int array
    qr_rates_n1 = {}
    qr_rates_n2 = {}

    for N in N_SWEEP:
        t0 = time.time()
        labels = np.zeros((nW, nW), dtype=int)
        r1g = np.zeros((nW, nW))
        r2g = np.zeros((nW, nW))
        for i, w_21 in enumerate(W_VALUES):
            for j, w_12 in enumerate(W_VALUES):
                rates = simulate_contralateral(
                    w12=float(w_12), w21=float(w_21),
                    drive=DRIVE, p_thin=P_THIN,
                    qr=qr, alpha=calib["alpha"], beta=calib["beta"],
                    N=N, T=T_TOTAL, seed=SEED + i*nW + j,
                    init_A=INIT_A,
                )
                labels[i, j] = qr_wta_label(rates)
                post = rates[:, T_WARMUP_QR:]
                r1g[i, j] = post[0].mean()
                r2g[i, j] = post[1].mean()
        qr_labels[N] = labels
        qr_rates_n1[N] = r1g
        qr_rates_n2[N] = r2g
        elapsed = time.time() - t0
        n_blue = int(labels.sum())
        j_fcs = jaccard(labels, fcs_labels)
        j_sieg = jaccard(labels, sieg_labels)
        print(f"  N={N:4d}: blue={n_blue:4d}/{nW*nW}, "
              f"J(FCS)={j_fcs:.3f}, J(Siegert)={j_sieg:.3f}, "
              f"elapsed {elapsed:.1f}s")

    # Save.
    np.savez(
        RESULTS / "qr_grid.npz",
        W_VALUES=W_VALUES,
        N_SWEEP=np.array(N_SWEEP),
        qr_labels=np.stack([qr_labels[N] for N in N_SWEEP]),
        qr_rates_n1=np.stack([qr_rates_n1[N] for N in N_SWEEP]),
        qr_rates_n2=np.stack([qr_rates_n2[N] for N in N_SWEEP]),
        T_total=T_TOTAL, T_warmup=T_WARMUP_QR,
    )

    # ----- Plots -----
    n_panels = len(N_SWEEP) + 2  # FCS + Siegert + each N
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    panel_specs = [
        (axes[0], fcs_labels,
            f"FCS oracle\n{int(fcs_labels.sum())}/{fcs_labels.size}"),
        (axes[1], sieg_labels,
            f"Siegert mean field\n{int(sieg_labels.sum())}/{sieg_labels.size}"),
    ]
    for k, N in enumerate(N_SWEEP):
        panel_specs.append((
            axes[2 + k], qr_labels[N],
            f"QR N={N}\n{int(qr_labels[N].sum())}/{qr_labels[N].size}\n"
            f"J(FCS)={jaccard(qr_labels[N], fcs_labels):.3f}",
        ))
    for ax, labels, title in panel_specs:
        for i, w_21 in enumerate(W_VALUES):
            for j, w_12 in enumerate(W_VALUES):
                color = "tab:blue" if labels[i, j] else "tab:red"
                ax.scatter(int(w_12), int(w_21), c=color, s=18,
                           edgecolor="none")
        ax.set_xlabel(r"$w_{12}$")
        ax.set_ylabel(r"$w_{21}$")
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
    fig.suptitle(
        "Phase 3: Quasi-renewal at finite N vs FCS oracle and Siegert mean field",
        fontsize=11,
    )
    plt.tight_layout()
    out_pdf = RESULTS / "qr_n_sweep.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\n  wrote {out_pdf}")

    # Jaccard vs N curve.
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    j_fcs_vec = [jaccard(qr_labels[N], fcs_labels) for N in N_SWEEP]
    j_sieg_vec = [jaccard(qr_labels[N], sieg_labels) for N in N_SWEEP]
    ax.plot(N_SWEEP, j_fcs_vec, 'o-', label="J(QR vs FCS)", color="tab:red")
    ax.plot(N_SWEEP, j_sieg_vec, 's-', label="J(QR vs Siegert mean-field)",
            color="tab:blue")
    ax.axhline(0.70, color="gray", linestyle="--", alpha=0.5,
               label="gate (0.70)")
    ax.set_xscale("log")
    ax.set_xlabel("N (population size)")
    ax.set_ylabel("Jaccard")
    ax.set_title("QR Jaccard vs FCS and Siegert across N")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_pdf = RESULTS / "qr_jaccard_vs_N.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")

    print()
    print("Phase 3 verdict:")
    j_fcs_n2000 = j_fcs_vec[-1]
    j_sieg_n2000 = j_sieg_vec[-1]
    print(f"  At N=2000 (~mean field): J(FCS)={j_fcs_n2000:.3f}, "
          f"J(Siegert)={j_sieg_n2000:.3f}")
    if j_sieg_n2000 >= 0.85:
        print(f"  PASS (mean-field convergence): "
              f"QR at N=2000 matches Siegert mean-field at "
              f"J={j_sieg_n2000:.3f} >= 0.85.")
    else:
        print(f"  PARTIAL: QR-vs-Siegert convergence weaker than expected.")


if __name__ == "__main__":
    main()
