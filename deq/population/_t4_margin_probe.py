"""Probe: how does the t=4 WTA region depend on the margin threshold?

The default Phase 1 classifier (margin=0.3) commits at moderate
asymmetry; FCS Fig. 10 effectively requires a sharper commitment to
declare success within four ticks. This script sweeps a few margin
values to see how the WTA region shrinks. Used to choose the panel that
best matches FCS Fig. 10's geometry.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ground_truth import wta_contralateral
from wilson_cowan import Sigmoid

DRIVE = 1.5
TAU = 1.0
N_GRID = 50
W_MIN, W_MAX = 0.0, 5.0


def _job(args):
    i, j, w12, w21, t_final, margin = args
    sig = Sigmoid(k=4.0, theta=1.0)
    v, _ = wta_contralateral(
        w12, w21, DRIVE, TAU, sig,
        t_final=t_final, perturbation=0.05, margin=margin,
    )
    return i, j, bool(v)


def sweep(t_final: float, margin: float):
    w_grid = np.linspace(W_MIN, W_MAX, N_GRID)
    tasks = [
        (i, j, float(w_grid[i]), float(w_grid[j]), t_final, margin)
        for i in range(N_GRID) for j in range(N_GRID)
    ]
    out = np.zeros((N_GRID, N_GRID), dtype=bool)
    n_workers = max(1, min(12, (os.cpu_count() or 2) - 2))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for fut in as_completed([ex.submit(_job, t) for t in tasks]):
            i, j, v = fut.result()
            out[i, j] = v
    return out


def main() -> int:
    for margin in (0.3, 0.5, 0.7, 0.85):
        wta = sweep(t_final=4.0, margin=margin)
        print(f"t=4, margin={margin}: {wta.sum()}/{N_GRID*N_GRID} cells WTA", flush=True)
        np.save(HERE / "results" / f"ground_truth_contralateral_t4_m{int(margin*100):02d}.npy", wta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
