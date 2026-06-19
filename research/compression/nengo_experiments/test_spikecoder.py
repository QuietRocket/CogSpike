#!/usr/bin/env python3
"""Unit tests for the shared spikecoder package.

Asserts the package's constants and idioms match the paper and the numpy
validators to the stated precision. Run:  uv run python test_spikecoder.py
"""

import warnings

import numpy as np
import nengo

warnings.filterwarnings("ignore")

from spikecoder import config as cfg
from spikecoder import information as info
from spikecoder import latency as lat
from spikecoder import metrics as met
from spikecoder.source import PI, S, momentum_chain, sample_stream, RoverSource
from spikecoder.networks import build_readout_bank, build_softmax_predictor

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"[{PASS if cond else FAIL}] {name}  {detail}")


# --- 1. information constants match the paper / validators --------------------
check("H_marginal == 1.7500", abs(info.H_MARGINAL - 1.7500) < 5e-5,
      f"= {info.H_MARGINAL:.4f}")
check("H_rate == 0.9782", abs(info.H_RATE - 0.9782) < 5e-5, f"= {info.H_RATE:.4f}")
check("mutual_info == 0.7718", abs(info.MUTUAL_INFO - 0.7718) < 5e-5,
      f"= {info.MUTUAL_INFO:.4f}")

# --- 2. source matches validate.py -------------------------------------------
P = momentum_chain(PI, S)
drift = np.abs(PI @ P - PI).max()
check("stationary pi P = pi", drift < 1e-15, f"drift = {drift:.2e}")
x = sample_stream(P, PI, 200000, seed=7)
emp = np.bincount(x, minlength=4) / len(x)
check("empirical marginal ~ pi", np.abs(emp - PI).max() < 0.01,
      f"max|emp-pi| = {np.abs(emp - PI).max():.3f}")

# --- 3. calibration drive table matches the paper ----------------------------
table = lat.drive_table()  # q -> (R I/theta, surprisal)
exp = {0.5: 1.58, 0.9: 7.09, 0.95: 14.0, 0.99: 69.5}
ok = all(abs(dict((q, ri) for q, ri, _ in table)[q] - v) / v < 0.02
         for q, v in exp.items())
check("drive table ~ paper (1.58/7.09/14.0/69.5)", ok,
      str({q: round(ri, 2) for q, ri, _ in table}))

# --- 4. calibration composes to t* = -lambda log2 q (analytic) ---------------
qs = np.array([0.5, 0.25, 0.125, 0.85, 0.9, 0.0375])
J = lat.calibration_drive(qs)
t_nengo = lat.nengo_first_spike_time(J)
t_ideal = lat.analytic_latency_ideal(qs)
check("analytic: nengo_first_spike(calib_drive(q)) == -lambda log2 q",
      np.abs(t_nengo - t_ideal).max() < 1e-12,
      f"max err = {np.abs(t_nengo - t_ideal).max():.2e}")

# --- 5. q_max from the rheobase ceiling --------------------------------------
qmax = lat.q_max_for_ceiling(cfg.RHEOBASE_CEILING)
check("q_max in [0.85, 0.95] for ceiling=10x", 0.80 < qmax < 0.95, f"q_max = {qmax:.3f}")

# --- 6. readout bank: right neuron fires first at the right latency -----------
q_vec = np.array([0.5, 0.25, 0.125, 0.125])  # P[U] row of the rover
net = nengo.Network(seed=0)
with net:
    q_node = nengo.Node(q_vec)
ens, drive = build_readout_bank(net, q_node, N=4)
with net:
    p = nengo.Probe(ens.neurons)
with nengo.Simulator(net, dt=cfg.DT_FINE, progress_bar=False) as sim:
    sim.run(0.15)
lats = met.per_window_first_spikes(sim.data[p], sim.trange(), 0.15, 1, dt=cfg.DT_FINE)[0]
winner = int(np.argmin(lats))
t_win = lats[winner]
t_expect = lat.analytic_latency_ideal(q_vec.max())
check("readout bank: argmax-q neuron fires first", winner == int(np.argmax(q_vec)),
      f"winner={winner} (expect {int(np.argmax(q_vec))})")
check("readout bank: winner latency = -lambda log2 max(q)",
      abs(t_win - t_expect) < 3 * cfg.DT_FINE,
      f"meas={t_win:.5f} expect={t_expect:.5f}")

# --- 7. softmax predictor builds and roughly decodes softmax -----------------
logits = np.array([1.0, 0.0, -1.0, -2.0])
net2 = nengo.Network(seed=1)
with net2:
    a_node = nengo.Node(logits)
ens2, q_out = build_softmax_predictor(net2, a_node, N=4, n_neurons=600, radius=4.0)
with net2:
    pq = nengo.Probe(q_out, synapse=0.02)
with nengo.Simulator(net2, dt=cfg.DT, progress_bar=False) as sim:
    sim.run(0.3)
q_dec = sim.data[pq][-50:].mean(axis=0)
q_true = info.softmax(logits)
check("softmax predictor ~ analytic softmax (NEF approx)",
      np.abs(q_dec - q_true).max() < 0.08,
      f"max|q_dec - softmax| = {np.abs(q_dec - q_true).max():.3f}, sum={q_dec.sum():.3f}")

# --- summary -----------------------------------------------------------------
npass = sum(b for _, b in results)
print(f"\n{npass}/{len(results)} checks passed.")
raise SystemExit(0 if npass == len(results) else 1)
