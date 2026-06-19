# e04 — The calibrated readout race

**Tier 1.** The paper's claim #2 (circuit stage 2): N=4 calibrated LIF readouts race
to threshold; readout j is driven by `RI(q_j)`, so the highest-q readout fires first
at latency `−λ log₂ max_j q_j` — the MAP symbol and its surprisal, read off as a
single spike time. Verified in real Nengo spikes: argmax-q wins for every context and
the winner latency matches theory to ≤0.45 dt. Over a 4000-symbol rover stream the
π-weighted mean spike-time/λ reproduces `validate.py` part B2: 0.9790 bits for the
perfect predictor q=P (on the entropy-rate floor 0.9782), 1.7521 for memoryless q=π
(≈H_marginal=1.7500), 1.1142 for wrong-momentum s'=0.4 (≈1.1133). Two honest costs:
a **signed O(dt) timing-resolution overhead** (+1.4–3.7 mbits, the integer-bit
penalty's spiking analogue) and a discrete-clock readout tie that bites only when two
model probabilities are nearly equal — the natural rover (top-2 gaps ≥46 ms) never
ties, but an engineered 0.30-ms-gap distribution resolves at dt=1e-4 and fuses at
dt=1e-3. 10/10 checks pass.

Run: `uv run python run.py` → `results/` + `report.pdf`.
