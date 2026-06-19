# e03 — NEF softmax: the exact simplex normalizer, bought with neurons

**Tier 1.** Tests paper claim #7: the predicted `q = softmax(W c)` is the *exact*
simplex normalizer (`Σ q = 1` identically), while Carandini–Heeger divisive
normalization `r = a/(σ + Σa)` sums to *strictly less than 1* and is only the
`σ→0` approximation. In `float64` the partition of unity is free; in the NEF the
softmax must be decoded from a finite spiking LIF population, so `Σ q = 1` becomes
*representational*. The decode RMSE falls as `1/√N` (fitted log-log slope −0.522,
NEF ideal −0.5), is 0.0229 at n=400 (< 0.05 target), and the normalization defect
shrinks from 0.018 to 0.0011 over 50→1600 neurons — i.e. partition-of-unity is
bought with neurons at 5.83→9.83 bits of precision. Divisive norm sums to 0.800
(σ=1) rising to 0.9988 (σ=0.005), exactly its analytic `Σa/(σ+Σa)`. 8/8 checks
pass; the most reliable rung in the suite.

Run: `uv run python run.py` → `results/` + `report.pdf`.
