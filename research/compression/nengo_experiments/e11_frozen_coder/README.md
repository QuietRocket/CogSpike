# e11 — The full end-to-end spiking entropy coder (frozen perfect predictor)

The flagship reproduction (paper claim #3 + the three-stage circuit, composed and run
whole). The three verified pieces — context (previous symbol) → **frozen perfect**
predictor `q = P[c_t]` (the true rover row; `softmax(log P) = P`) → calibrated readout
race (`build_readout_bank`, stage 2) → first-spike-takes-all decode
(`metrics.decode_first_spike`, stage 3) — are wired into ONE coder and run on the rover
stream (s=0.7, seed 7) with per-window reset. We run it two honest ways: a genuine
continuous Nengo pipeline over 400 windows (stage 1→2→3, emitted-only drive, blank
reset) and a lookup-assembled mean over n=4000 symbols (each distinct predictor row
measured once in real spikes, then assembled by (context, outcome) lookup), and check
the two agree.

**Headline:** the perfect predictor's mean per-symbol first-spike time / λ →
**0.9790 bits/symbol**, on the entropy-rate floor **H_rate = 0.9782** (+0.73 mbit), and
the decode is **lossless** (0 errors end-to-end, for both the perfect predictor and a
uniform q=1/4 which is slow at 1.936 bits but never wrong). The three-baseline table
reproduces validate.py B2 in spikes — perfect **0.9790**, memoryless q=π **1.7521**
(≈ H_marginal 1.7500), wrong-momentum s'=0.4 **1.1142** (≈ 1.1133) — with the efficiency
ordering perfect < wrong < memoryless and the stupidity-tax gaps **0.773** (≈ mutual
info 0.7718) and **0.135** (≈ KL 0.1351). The +0.73-mbit overhead above the floor is
decomposed by ablation: **dt rounding** (+1.4 mbit fine → +17.6 mbit coarse, vanishing
as dt→0), finite-stream sampling, and **context error** (+20.4 mbit per 1% attractor
misread, but ≈0 at e06's measured ring fidelity). 13/13 acceptance checks pass.
