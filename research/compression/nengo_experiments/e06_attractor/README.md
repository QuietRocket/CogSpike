# e06 — A ring attractor holds the running context c_t

**Tier 2.** The recurrent-predictor stage: the running context `c_t` (which previous
symbol) is held as a bump of graded persistent activity on a ring attractor (Seung
1996; Ben-Yishai 1995). Symbol `s` is the angle `θ=s·2π/4` on a 2-D NEF ring; a
clear-then-write load wipes the old bump and writes the new one from rest (an
additive write cancels antipodal bumps through the origin — observed and fixed). The
ring holds an arbitrary written angle to ≤5.7° over 300 ms (a genuine continuous ring
of fixed points), settles a new symbol in ≤35 ms (≪200 ms ISI), and decodes a
6-symbol stream with 0 errors. Drift is bounded (1.48° rms per ISI, 4.90° after 1 s),
giving an effective capacity `C=log₂(SNR)` of 7.93 bits at the ISI timescale (6.20
after a long hold) — comfortably above the `log₂(4)=2` bits a first-order source needs.
This is FIX G / BIO-06 made measurable: capacity is finite, noise-limited, and
substrate-dependent, but holds a first-order context with bits to spare.

Run: `uv run python run.py` → `results/` + `report.pdf`.
