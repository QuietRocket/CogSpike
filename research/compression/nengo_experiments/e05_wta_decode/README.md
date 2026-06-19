# e05 — Temporal winner-take-all decoder (first-spike-takes-all)

**Tier 1.** The paper's structural-safety claim: the decoder commits to the *first*
readout to cross threshold, so the code is *lossless* (`decoded == emitted`) for any
predictor. Verified — over a 60-symbol rover stream the calibrated readout bank
decodes with **zero** errors under both the perfect predictor (mean latency 17.9 ms)
*and* a uniform `q=1/4` predictor (38.7 ms = the 2-bit floor): "a bad model is slow,
never wrong." Under membrane noise (σ=0.5) decode errors **concentrate entirely at
small latency-gap margin** (0.333 at 0.58 ms → 0 by 5 ms; 100% of errors in the
small-margin half), exactly the predicted failure mode. The dynamical WTA layer
latches a single correct winner in the emitted-only regime (settling time 7→63 ms,
monotone in surprisal) and gives an always-correct commit-time decode in the full
race (4/4); the free-run multi-winner steady state — the integration transient the
global "≤1 always" claim mistakes for an invariant (the settled `F G` form is the
real property) — is documented honestly. **8/8 checks pass.**

Run: `uv run python run.py` → `results/` + `report.pdf`.
