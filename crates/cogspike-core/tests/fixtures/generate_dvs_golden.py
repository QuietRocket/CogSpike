#!/usr/bin/env python3
"""Generate the DVS (event-camera) golden fixture for the Rust parity test.

Byte-faithful to research/compression/nengo_experiments/e16_event_stream/run.py for the
SOURCE + PREDICTOR + RESIDUAL IDENTITY. We do NOT run Nengo here: e16 proved (9/9) that
its LIF soma bank with J_ON=6 >> theta=1 and a per-frame reset fires exactly when
`raw - predicted > 0`, i.e. the residual is the pure integer identity
`resid = raw AND NOT predicted`. This script computes that identity with numpy so the Rust
port (crates/cogspike-core/src/dvs.rs) can be asserted byte-exact against numpy's array ops
(bar brightness, edge events, np.roll prediction, integer residual, the jitter sweep).

Regenerate:  deq/.venv/bin/python crates/cogspike-core/tests/fixtures/generate_dvs_golden.py
(numpy only; uses the deq venv which already has numpy).
"""

import json
from pathlib import Path

import numpy as np

P = 24
W = 4
V = 1

HERE = Path(__file__).resolve().parent


def bar_brightness(positions, p, w):
    n = len(positions)
    b = np.zeros((n, p), dtype=int)
    for k, pos in enumerate(positions):
        if pos < 0:
            continue
        idx = (np.arange(w) + int(pos)) % p
        b[k, idx] = 1
    return b


def events_from_brightness(b):
    n, p = b.shape
    prev = np.zeros((n, p), dtype=int)
    prev[1:] = b[:-1]
    diff = b - prev
    return (diff > 0).astype(int), (diff < 0).astype(int)


def predict_events(on, off, v_hat):
    n, p = on.shape
    v = np.full(n, int(v_hat)) if np.isscalar(v_hat) else np.asarray(v_hat, int)
    on_pred = np.zeros_like(on)
    off_pred = np.zeros_like(off)
    for k in range(1, n):
        on_pred[k] = np.roll(on[k - 1], int(v[k]))
        off_pred[k] = np.roll(off[k - 1], int(v[k]))
    return on_pred, off_pred


def residual(raw, pred):
    return ((raw == 1) & (pred == 0)).astype(int)


def build_schedule():
    pos, vel, novel = [], [], []
    for _ in range(3):
        pos.append(-1)
        vel.append(0)
    novel.append(("onset", len(pos)))
    pos.append(0)
    vel.append(0)
    cur = 0
    for _ in range(15):
        cur = (cur + V) % P
        pos.append(cur)
        vel.append(V)
    novel.append(("reversal", len(pos)))
    for _ in range(12):
        cur = (cur - V) % P
        pos.append(cur)
        vel.append(-V)
    novel.append(("jump", len(pos)))
    for _ in range(10):
        cur = (cur + 2 * V) % P
        pos.append(cur)
        vel.append(2 * V)
    return np.array(pos), novel, np.array(vel)


def lagging(vel):
    v = np.zeros_like(vel)
    v[1:] = vel[:-1]
    return v


def jittered_schedule(rho, seed, n_steady=40):
    rg = np.random.default_rng(seed)
    pos = np.empty(n_steady, dtype=int)
    cur = 0
    pos[0] = cur
    for k in range(1, n_steady):
        step = V
        if rg.random() < rho:
            step += int(rg.choice([-1, 1]))
        cur = (cur + step) % P
        pos[k] = cur
    return pos


# --- headline: canonical schedule with the adaptive one-frame-lag predictor ---
positions, novel, vel = build_schedule()
v_hat = lagging(vel)
b = bar_brightness(positions, P, W)
on_raw, off_raw = events_from_brightness(b)
on_pred, off_pred = predict_events(on_raw, off_raw, v_hat)
on_resid = residual(on_raw, on_pred)
off_resid = residual(off_raw, off_pred)
raw_total = int(on_raw.sum() + off_raw.sum())
resid_total = int(on_resid.sum() + off_resid.sum())
compression = raw_total / max(resid_total, 1)

# --- jitter sweep: fixed +V predictor, stored positions for exact Rust replay ---
rho_grid = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
sweep = []
for i, rho in enumerate(rho_grid):
    pj = jittered_schedule(rho, seed=100 + i)
    bj = bar_brightness(pj, P, W)
    onj, offj = events_from_brightness(bj)
    onp, offp = predict_events(onj, offj, V)
    raw_j = int(onj.sum() + offj.sum())
    resid_j = int(residual(onj, onp).sum() + residual(offj, offp).sum())
    hits = int((onj & onp).sum() + (offj & offp).sum())
    sweep.append(
        {
            "rho": rho,
            "positions": pj.tolist(),
            "raw": raw_j,
            "resid": resid_j,
            "ratio": raw_j / max(resid_j, 1),
            "predictability": hits / max(raw_j, 1),
        }
    )
corr = float(
    np.corrcoef([s["predictability"] for s in sweep], [s["ratio"] for s in sweep])[0, 1]
)

out = {
    "_provenance": {
        "source": "research/compression/nengo_experiments/e16_event_stream/run.py",
        "note": "integer residual identity resid = raw AND NOT pred (J_ON>>theta); numpy oracle, no Nengo",
        "P": P,
        "W": W,
        "V": V,
    },
    "positions": positions.tolist(),
    "vel_true": vel.tolist(),
    "v_hat": v_hat.tolist(),
    "novel": [[lbl, int(k)] for lbl, k in novel],
    "on_raw": on_raw.tolist(),
    "off_raw": off_raw.tolist(),
    "on_resid": on_resid.tolist(),
    "off_resid": off_resid.tolist(),
    "raw_total": raw_total,
    "resid_total": resid_total,
    "compression_ratio": compression,
    "rho_grid": rho_grid,
    "sweep": sweep,
    "corr_pred_ratio": corr,
}

dst = HERE / "dvs_golden.json"
dst.write_text(json.dumps(out, indent=1))
print(f"wrote {dst}")
print(f"  raw_total={raw_total} resid_total={resid_total} compression={compression:.2f}x")
print(f"  jitter sweep corr(predictability, ratio) = {corr:+.3f}")
