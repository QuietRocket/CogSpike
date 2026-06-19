# e01 — Raw LIF first-spike latency law

**Tier 0.** Does Nengo's real spiking LIF obey the paper's charging/latency law
`t*(I) = τ ln(RI/(RI−θ))`? Yes — to within one timestep — once the initial voltage
is pinned to rest. Surfaces two spiking-reality facts the numpy validator never
sees: (1) Nengo randomizes initial voltage (must pin V(0)=0 for a latency code),
and (2) the refractory period delays the *inter-spike interval*, not the first
spike from rest — so the latency code is refractory-immune. Recovers τ_rc=20.018 ms.

Run: `uv run python run.py` → `results/` + `report.pdf`.
