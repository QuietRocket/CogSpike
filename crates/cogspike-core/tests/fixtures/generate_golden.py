#!/usr/bin/env python3
"""Generate the byte-faithful golden fixtures for the Rust coder parity tests.

Dumps the exact numpy/spikecoder outputs (constants + deterministic test vectors)
that crates/cogspike-core/tests/coder_parity.rs asserts the Rust port reproduces
to <=1e-12 (deterministic) / 5e-5 (named constants).

This is the human-run `parity --update` step referenced in the build plan; it is
NOT run by CI. Run from the repo root with any interpreter that has numpy + the
spikecoder package importable, e.g.:

    deq/.venv/bin/python crates/cogspike-core/tests/fixtures/generate_golden.py
    # or
    uv run --project research/compression/nengo_experiments python \
        crates/cogspike-core/tests/fixtures/generate_golden.py

Only the pure-numpy spikecoder modules are imported (config/source/information/
latency) -- nengo is NOT required.
"""

import json
import os
import subprocess
import sys

import numpy as np

REPO = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
).stdout.strip()
SC = os.path.join(REPO, "research", "compression", "nengo_experiments")
sys.path.insert(0, SC)

from spikecoder import config as cfg  # noqa: E402
from spikecoder import information as info  # noqa: E402
from spikecoder import latency as lat  # noqa: E402
from spikecoder.source import (  # noqa: E402
    LABELS,
    PI,
    S,
    momentum_chain,
    sample_stream,
)

P = momentum_chain(PI, S)

# Deterministic test inputs (chosen to cover the working band + the ceiling edge).
QS = [0.5, 0.25, 0.125, 0.85, 0.9, 0.0375, 0.98, 0.99, 0.95]
LOGITS = [
    [1.0, 0.0, -1.0, -2.0],
    [0.0, 0.0, 0.0, 0.0],
    [2.0, 1.0, 0.5, 0.0],
    [-3.0, 4.0, 0.0, 1.5],
]
# (p, q) pairs for entropy / cross-entropy / KL.
PQ = [
    (PI.tolist(), PI.tolist()),
    (P[0].tolist(), P[0].tolist()),
    (P[0].tolist(), PI.tolist()),
    (P[1].tolist(), momentum_chain(PI, 0.4)[1].tolist()),
]


def sha():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()


fixtures = {
    "_provenance": {
        "generator": "crates/cogspike-core/tests/fixtures/generate_golden.py",
        "sources": [
            "research/compression/nengo_experiments/spikecoder/{config,source,information,latency}.py",
            "research/compression/validate.py",
            "research/compression/learn_validate.py",
        ],
        "git_sha": sha(),
        "numpy_version": np.__version__,
        "seed": cfg.SEED,
        "tolerances": {"deterministic": 1e-12, "named_constants": 5e-5},
        "note": "PCG64 stream fields are reference-only; Rust asserts statistical convergence, not exact stream equality.",
    },
    "labels": LABELS,
    "constants": {
        "LN2": info.LN2,
        "S": S,
        "PI": PI.tolist(),
        "H_MARGINAL": info.H_MARGINAL,
        "H_RATE": info.H_RATE,
        "MUTUAL_INFO": info.MUTUAL_INFO,
        "LEARNED_REF": info.LEARNED_REF,
        "TAU_RC": cfg.TAU_RC,
        "TAU_REF": cfg.TAU_REF,
        "LAMBDA": cfg.LAMBDA,
        "ALPHA": cfg.ALPHA,
        "THETA": cfg.THETA,
        "RHEOBASE_CEILING": cfg.RHEOBASE_CEILING,
        "Q_CLIP_LO": cfg.Q_CLIP_LO,
        "Q_CLIP_HI": cfg.Q_CLIP_HI,
        "DT": cfg.DT,
        "DT_FINE": cfg.DT_FINE,
        "SEED": cfg.SEED,
    },
    "P": P.tolist(),
    "cond_entropies": [info.entropy_bits(P[i]) for i in range(4)],
    "softmax_inputs": LOGITS,
    "softmax_outputs": [info.softmax(z).tolist() for z in LOGITS],
    "info_pq": [
        {
            "p": p,
            "q": q,
            "entropy_p": info.entropy_bits(p),
            "cross_entropy": info.cross_entropy_bits(p, q),
            "kl": info.kl_bits(p, q),
        }
        for (p, q) in PQ
    ],
    "cross_entropy_rate": {
        "perfect": info.cross_entropy_rate(P, PI, P),
        "memoryless": info.cross_entropy_rate(P, PI, np.tile(PI, (4, 1))),
        "wrong_s04": info.cross_entropy_rate(P, PI, momentum_chain(PI, 0.4)),
    },
    "latency": {
        "qs": QS,
        "alpha_of": lat.alpha_of(),
        "analytic": [float(lat.analytic_latency_ideal(q)) for q in QS],
        "calibration_drive": [float(lat.calibration_drive(q)) for q in QS],
        "nengo_first_spike_of_drive": [
            float(lat.nengo_first_spike_time(lat.calibration_drive(q))) for q in QS
        ],
        "q_for_drive_roundtrip": [
            float(lat.q_for_drive(lat.calibration_drive(q))) for q in QS
        ],
    },
    "q_max_for_ceiling": float(lat.q_max_for_ceiling(cfg.RHEOBASE_CEILING)),
    "t_min_for_ceiling": float(lat.t_min_for_ceiling(cfg.RHEOBASE_CEILING)),
    "drive_table": [list(r) for r in lat.drive_table()],
    # PCG64 reference (NOT for exact Rust equality -- statistical convergence only):
    "stream_seed7_first64": sample_stream(P, PI, 64, seed=7).tolist(),
    "stream_seed7_200k_marginal": (
        np.bincount(sample_stream(P, PI, 200_000, seed=7), minlength=4) / 200_000
    ).tolist(),
}

out = os.path.join(REPO, "crates", "cogspike-core", "tests", "fixtures", "coder_golden.json")
with open(out, "w") as fh:
    json.dump(fixtures, fh, indent=2)
print(f"wrote {out}")
print(f"  H_MARGINAL={info.H_MARGINAL:.6f} H_RATE={info.H_RATE:.6f} "
      f"MUTUAL_INFO={info.MUTUAL_INFO:.6f} ALPHA={cfg.ALPHA:.6f} "
      f"q_max={fixtures['q_max_for_ceiling']:.4f}")
