# e12 — The learned end-to-end spiking entropy coder (the money plot)

The Tier-4 capstone that composes everything. e11 ran the full coder with a **frozen
perfect** predictor; here the predictor is the **e09 PES learner**, starting from the
**uniform** predictor `q = 1/4` (so the coder's bill starts at log₂4 = 2.0 bits/symbol)
and learning the rover law **online** from the realized stream. Its **learned** `q̂(·|c)`
then **drives the calibrated readout bank** (stage 2) → first-spike-takes-all decode
(stage 3) — so as the circuit learns, its own spike-time bill falls. We snapshot the
learner at 9 training checkpoints and, at each, drive **fresh real Nengo readout banks**
(dt=1e-4) with the learned, clamped rows and measure the mean per-symbol cost in **bits
from actual first-spike latencies** (`latency/λ`), plus the model's own `-log₂q̂` and the
decode top-1 accuracy, over the realized n=4000 stream. The learned `q̂` is clamped to
`(1e-4, q_max=0.9296]` before the drive so the closed loop stays finite through the
noisy early transient.

**Headline (the money plot):** the coder's **measured** spike-time bill descends
**1.985 → 1.026 bits/symbol** — from the uniform init (log₂4=2.0), past the marginal
**1.7500** (by checkpoint 25), into a noise ball **0.05 bits above** the entropy-rate
floor **0.9782** — **monotonically**, while decode top-1 accuracy rises **0.379 → 0.803**,
landing **exactly on the Bayes-optimal ceiling** `∑_c π_c max_j P[c,j] = 0.8031` (a
stochastic rover cannot be predicted better). Bits and accuracy anti-correlate at
**r = −0.98**: compression, prediction, and learning are **one descent on one number**.
The real spike clock equals the learned model's surprisal to a **+1.68 mbit** dt tax;
the genuine continuous learned pipeline is **lossless** (decode error 0) and matches the
lookup to 6.48 mbit. The 0.05-bit residual is the honest e09 constant-lr noise ball
plus the e11 dt tax — direction exact, residual the price of physicality. 12/12
acceptance checks pass.
