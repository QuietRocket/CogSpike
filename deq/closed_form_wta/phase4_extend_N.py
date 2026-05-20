"""Phase 4 step 1: Extended-N quasi-renewal sweep.

Re-runs the Phase 3 quasi-renewal contralateral sweep on the same 40x40
(w_12, w_21) FCS grid, but at the wider population set
  N in {10, 20, 50, 100, 500, 2000, 20000}.

At N=10, 20 we probe the deep-noise regime where the staircase should
dissolve most aggressively (and the boundary should broaden most).
At N=20000 we probe the deterministic mean-field limit; this is the
strict Siegert envelope.

The four-N subset {50, 100, 500, 2000} reproduces Phase 3 exactly under
the same RNG seed scheme (per-cell seed = SEED + i*nW + j), giving a
correctness check.

Output: results/phase4/qr_grid_extended.npz with the same layout as
results/phase3/qr_grid.npz.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from deq.closed_form.siegert import Siegert  # noqa: E402
from deq.closed_form.quasi_renewal import (  # noqa: E402
    QuasiRenewal,
    simulate_contralateral,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "phase4"
RESULTS.mkdir(parents=True, exist_ok=True)


DRIVE = 11.0
P_THIN = 0.7
T_TOTAL = 200
T_WARMUP_QR = 50
ASYM_GATE = 0.30
N_SWEEP = [10, 20, 50, 100, 500, 2000, 20000]
SEED = 1234
INIT_A = (0.5, 0.1)


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

    print(f"Quasi-renewal extended-N sweep on {nW}x{nW} grid")
    print(f"  N sweep: {N_SWEEP}; T={T_TOTAL}; init_A={INIT_A}")
    print(f"  WTA gate: rate_max - rate_min >= {ASYM_GATE} "
          f"(post-warmup window [t={T_WARMUP_QR}, T={T_TOTAL}])\n")

    qr_labels = {}
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
        print(f"  N={N:5d}: blue={n_blue:4d}/{nW*nW}, "
              f"J(FCS)={j_fcs:.3f}, J(Siegert)={j_sieg:.3f}, "
              f"elapsed {elapsed:.1f}s")

    np.savez(
        RESULTS / "qr_grid_extended.npz",
        W_VALUES=W_VALUES,
        N_SWEEP=np.array(N_SWEEP),
        qr_labels=np.stack([qr_labels[N] for N in N_SWEEP]),
        qr_rates_n1=np.stack([qr_rates_n1[N] for N in N_SWEEP]),
        qr_rates_n2=np.stack([qr_rates_n2[N] for N in N_SWEEP]),
        T_total=T_TOTAL, T_warmup=T_WARMUP_QR,
    )
    print(f"\n  wrote {RESULTS / 'qr_grid_extended.npz'}")


if __name__ == "__main__":
    main()
