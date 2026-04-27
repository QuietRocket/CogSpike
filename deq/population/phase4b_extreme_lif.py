"""Phase 4b - Extreme-weight LI&F sweep.

Phase 4 swept the integer LI&F weight grid out to |w^LIF| <= 40 and
showed that the discrete bistable region is rectangular --- two
axis-aligned strips at |w12| >= w_c OR |w21| >= w_c with w_c about 6 ---
which agrees with the continuous WC pitchfork wedge at the symmetric
corner but diverges in the arms. The natural follow-up question is the
scaling motive of the v2 note: does the rectangular-strip structure
*persist* at higher weights, or do new features emerge once weights are
far past the WC saturation regime?

This script repeats the Phase 4 classifier on a 200x200 integer grid
covering |w^LIF| in [1, 200]. The result feeds the v2 §7 figure that
extends Fig. 5 outward.
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

ARCHETYPES_DIR = HERE.parent / "archetypes"
sys.path.insert(0, str(ARCHETYPES_DIR))
from lif_fcs import simulate as lif_oracle  # noqa: E402

SEED = 20260420
np.random.seed(SEED)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

LIF_GRID = 200            # 200 x 200 grid; |w| in [1, 200]
LIF_T = 50
LIF_SELF_DRIVE = 11
INIT_DELAY = 2
WTA_TAIL = 20
WTA_RATIO = 8


def _run_one(w12: int, w21: int, favor_neuron: int, T: int) -> tuple[int, int]:
    W = np.array([[0, w21], [w12, 0]], dtype=np.int64)
    B = np.array([[LIF_SELF_DRIVE, 0], [0, LIF_SELF_DRIVE]], dtype=np.int64)
    external = np.ones((2, T), dtype=np.int64)
    other = 1 - favor_neuron
    external[other, :INIT_DELAY] = 0
    spikes, _ = lif_oracle(W, B, external, T=T)
    tail = spikes[:, -WTA_TAIL:]
    return int(tail[0].sum()), int(tail[1].sum())


def _winner(s0: int, s1: int) -> int:
    lo, hi = min(s0, s1), max(s0, s1)
    if hi == 0:
        return -1
    if lo == 0:
        return 0 if s0 > s1 else 1
    if hi >= WTA_RATIO * lo:
        return 0 if s0 > s1 else 1
    return -1


def _classify_cell(args):
    i, j, w12, w21 = args
    s0_A, s1_A = _run_one(w12, w21, favor_neuron=0, T=LIF_T)
    s0_B, s1_B = _run_one(w12, w21, favor_neuron=1, T=LIF_T)
    w_A = _winner(s0_A, s1_A)
    w_B = _winner(s0_B, s1_B)
    bistable = (w_A >= 0 and w_B >= 0 and w_A != w_B)
    return i, j, bool(bistable), s0_A, s1_A, s0_B, s1_B


def main() -> int:
    w_vals = np.arange(1, LIF_GRID + 1)
    wta = np.zeros((LIF_GRID, LIF_GRID), dtype=bool)
    s0A = np.zeros((LIF_GRID, LIF_GRID), dtype=int)
    s1A = np.zeros((LIF_GRID, LIF_GRID), dtype=int)
    s0B = np.zeros((LIF_GRID, LIF_GRID), dtype=int)
    s1B = np.zeros((LIF_GRID, LIF_GRID), dtype=int)

    tasks = [
        (i, j, int(-int(w_vals[i])), int(-int(w_vals[j])))
        for i in range(LIF_GRID) for j in range(LIF_GRID)
    ]
    n_workers = max(1, min(12, (os.cpu_count() or 2) - 2))
    print(f"Phase 4b: {len(tasks)} cells across {n_workers} workers (|w| up to {LIF_GRID})", flush=True)
    t0 = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for fut in as_completed([ex.submit(_classify_cell, t) for t in tasks]):
            i, j, is_wta, a0, a1, b0, b1 = fut.result()
            wta[i, j] = is_wta
            s0A[i, j] = a0; s1A[i, j] = a1
            s0B[i, j] = b0; s1B[i, j] = b1
            completed += 1
            if completed % 4000 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = (len(tasks) - completed) / rate if rate > 0 else float("nan")
                print(f"  {completed}/{len(tasks)} cells in {elapsed:.1f}s (eta {eta:.0f}s)", flush=True)
    print(f"Sweep complete in {time.time()-t0:.1f}s", flush=True)

    np.save(RESULTS / "lif_extreme_wta_map.npy", wta)
    np.save(RESULTS / "lif_extreme_w_vals.npy", w_vals)
    np.save(RESULTS / "lif_extreme_n0_A.npy", s0A)
    np.save(RESULTS / "lif_extreme_n1_A.npy", s1A)
    np.save(RESULTS / "lif_extreme_n0_B.npy", s0B)
    np.save(RESULTS / "lif_extreme_n1_B.npy", s1B)

    n_bistable = int(wta.sum())
    print(f"Bistable cells in 200x200 sweep: {n_bistable}/{LIF_GRID*LIF_GRID}", flush=True)

    # Quick geometry check: at every column j, find the smallest |w12|
    # magnitude that makes the cell bistable (i.e. the row index of the
    # boundary in row direction).
    threshold_per_col = np.full(LIF_GRID, -1, dtype=int)
    for j in range(LIF_GRID):
        col = wta[:, j]
        if col.any():
            threshold_per_col[j] = int(np.argmax(col))  # first True row
    # The rectangular conjecture: threshold_per_col is roughly constant
    # across j (the boundary in w12 doesn't depend on w21 once w21 is
    # itself past w_c). Report median + spread.
    valid = threshold_per_col[threshold_per_col >= 0]
    if len(valid) > 0:
        print(f"Boundary in |w12|: median={np.median(valid)+1}, p10={np.percentile(valid,10)+1:.0f}, p90={np.percentile(valid,90)+1:.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
