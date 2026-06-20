# e14 — Eligibility traces for memory beyond the previous symbol

Tests the paper's "Deeper memory" outlook + FIX G / BIO-06: the local rule
`ΔW_ij = η c_i (y_j − q_j)` credits the residual to the context active at the **same**
step — right for a first-order source, not enough when structure spans several past
symbols. The fix is an eligibility trace `e_ij ← γ e_ij + c_i`, `ΔW_ij = η e_ij r_j`,
which credits a late error to recently-active synapses. We build a **second-order
"ping-pong vs run" rover** (next symbol depends on the ordered pair `(x_{t−2}, x_{t−1})`,
an XOR-like source) with closed-form floors `H0 = 2.0000 > H1 = 1.8365 > H2 = 1.0545`,
so a first-order predictor must leave **I₂ = H1 − H2 = 0.7820 bits/symbol** on the table.

**Headline:** a first-order learner (numpy delta rule `E1 = 1.9033`; spiking PES
`E1 = 2.1017`) is stuck in the order-1 band, a full bit above `H2`, recovering **none**
of I₂. A **lag-tagged eligibility-trace** learner (lags kept as separate channels)
drops below H1 to `1.1190` — recovering **100%** of I₂ toward H2 — gated entirely on
`γ > 0` (γ = 0 recovers 0%). The trace dynamics `e ← γ e + c` are realized on a spiking
leaky-integrator population (`γ̂ = 0.748`). **Honest strains:** the spiking PES noise
ball sits above the (uniform) marginal on this weak-order-1 source, and the spiking
trace's quantitative γ carries NEF integrator drift — so the core learning result uses
the numpy trace while the spiking arms demonstrate stuck-ness and the mechanism. A
**collapsed (lag-blind) trace fails** (energy 2.0018, not below E1): the trace must tag
which lag each synapse fired at — the real content of **FIX G**. Capacity: order-2 needs
16 vs 4 states (4 vs 2 bits), inside e06's measured ring budget (C_isi = 7.9 bits) but
at a 4×-tighter half-cell margin (11.2° vs 45°) that order-3 would breach. **11/11
acceptance checks pass.**
