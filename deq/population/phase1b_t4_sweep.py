"""Phase 1b - 4-tick WTA sweep on the contralateral archetype.

Mirrors the Phase 1 (asymptotic, t=50) ground-truth sweep but with the
classifier's integration horizon shortened to t=4. This is the
apples-to-apples continuous-side analogue of FCS Fig. 10's "stabilises
within 4 time units" criterion (FCS §6.3.4): the FCS verification
declares red whenever the contralateral archetype has not committed to a
winner within four ticks, while the existing Phase 1 sweep declares WTA
whenever the system commits eventually.

The classifier `wta_contralateral` already takes `t_final` as a kwarg, so
this is a single-parameter change. We save to a separate .npy so the
v2 note can show both panels side by side without re-running Phase 1.

Parallelised across cores via ProcessPoolExecutor (mirroring Phase 4).
"""

from __future__ import annotations

import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=sys.stdout.encoding, line_buffering=True)

from ground_truth import wta_contralateral  # noqa: E402
from wilson_cowan import Sigmoid  # noqa: E402

SEED = 20260420
np.random.seed(SEED)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# Match Phase 1 grid exactly so the panels are directly comparable.
DRIVE = 1.5
TAU = 1.0
W_MIN, W_MAX = 0.0, 5.0
N_GRID = 50

T_FINAL = 4.0          # the only parameter that differs from Phase 1
PERTURB = 0.05
MARGIN = 0.3

# Sigmoid parameters fixed at module level so workers can rebuild it
# without inheriting unpicklable state.
SIGMOID_K = 4.0
SIGMOID_THETA = 1.0


def _classify_cell(args: tuple[int, int, float, float]) -> tuple[int, int, bool]:
    i, j, w12, w21 = args
    sigmoid = Sigmoid(k=SIGMOID_K, theta=SIGMOID_THETA)
    verdict, _ = wta_contralateral(
        w12, w21, DRIVE, TAU, sigmoid,
        t_final=T_FINAL, perturbation=PERTURB, margin=MARGIN,
    )
    return i, j, bool(verdict)


def main() -> int:
    w_grid = np.linspace(W_MIN, W_MAX, N_GRID)
    wta = np.zeros((N_GRID, N_GRID), dtype=bool)

    tasks = [
        (i, j, float(w_grid[i]), float(w_grid[j]))
        for i in range(N_GRID) for j in range(N_GRID)
    ]
    n_workers = max(1, min(12, (os.cpu_count() or 2) - 2))
    print(f"Phase 1b: {len(tasks)} cells across {n_workers} workers (t_final={T_FINAL})", flush=True)
    t0 = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for fut in as_completed([ex.submit(_classify_cell, t) for t in tasks]):
            i, j, v = fut.result()
            wta[i, j] = v
            completed += 1
            if completed % 200 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = (len(tasks) - completed) / rate if rate > 0 else float("nan")
                print(f"  {completed}/{len(tasks)} cells in {elapsed:.1f}s (eta {eta:.0f}s)", flush=True)

    print(f"Sweep complete in {time.time()-t0:.1f}s", flush=True)
    np.save(RESULTS / "ground_truth_contralateral_t4.npy", wta)

    # Sanity stats so the build log shows the qualitative comparison.
    wta_t50 = np.load(RESULTS / "ground_truth_contralateral.npy")
    n_wta_t4 = int(wta.sum())
    n_wta_t50 = int(wta_t50.sum())
    print(f"WTA cells at t=4:  {n_wta_t4}/{N_GRID*N_GRID}")
    print(f"WTA cells at t=50: {n_wta_t50}/{N_GRID*N_GRID}")
    print(f"t=4 region is a subset of t=50: {bool(((wta & ~wta_t50).sum()) == 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
