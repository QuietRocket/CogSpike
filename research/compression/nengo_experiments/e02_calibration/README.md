# e02 — Latency calibration: spike time is surprisal

**Tier 0.** The paper's central exact identity (Theorem 1): driving a readout with
`RI(q)=θ/(1−q^α)` makes its first-spike latency exactly `t*(q)=−λ log₂ q`. Verified
in a real Nengo LIF to ≤1 dt across q∈[0.0375, 0.98], with the drive table matching
the paper (7.09× rheobase at q=0.9). Turns the paper's three §4 honesty checks into
measured curves: finite drive caps q_max=0.93 (10× ceiling) with a saturation floor,
and the rare-symbol tail is noise-dominated (latency CV grows 0.008→0.110 as q→0).

Run: `uv run python run.py` → `results/` + `report.pdf`.
