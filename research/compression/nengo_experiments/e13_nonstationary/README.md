# e13 — Tracking a drifting source: the constant-η noise ball as a FEATURE

Tests the paper's sharpest learning-theory claim: the e09/e10 convergence theorem
assumed a *fixed* source and a *decreasing* (Robbins–Monro) step (`V → 0` a.s.), but a
*drifting* world wants a *constant* step — trading almost-sure convergence for a
steady-state `O(η)` noise ball that *is* the adaptation mechanism. The momentum rover is
a perfectly controlled non-stationarity: switching the stickiness `s` holds the marginal
`π = (1/2,1/4,1/8,1/8)` — and so the memoryless cost `H(π) = 1.7500` — **fixed**, moving
only the conditional structure (and the entropy-rate floor: `H_rate(0.70)=0.9782`,
`H_rate(0.40)=1.4852`, `H_rate(0.85)=0.5959`). We switch `s: 0.70 → 0.40 → 0.85 → 0.70`
and chase the moving floor with a constant-`η` spiking PES learner (e09 idiom), with an
exact float64 delta-rule twin carrying the denoised time-constant / lr-sweep / freeze.

**Headline.** The constant-η spiking learner **re-tracks every moving floor**: per-segment
converged energy `0.9856 / 1.4997 / 0.6259 / 1.0066` against floors `0.9782 / 1.4852 /
0.5959 / 0.9782` (excess `0.007`–`0.030`), all below the **invariant** marginal `1.7500`.
Each switch re-descends with a finite tracking **time-constant** `τ = 458 / 552 / 1106`
symbols (numpy twin), settling into a steady-state ball `V∞ ≈ 0.002`. The steady-state
tracking error **scales as `O(η)`** — `5e−3→0.00069, 8e−3→0.00094, 1.2e−2→0.00140,
1.6e−2→0.00186, 2.4e−2→0.00281`, log-log slope **0.91**. The Robbins–Monro **decreasing**
step (`η₀=0.08, t₀=800`) — almost-surely optimal on a *fixed* source — **freezes** on the
late `0.40→0.85` sharpen (excess `0.0409` vs the constant learner's `0.0030`, a **13.6×**
gap), and a tiny constant step `η=2e−4` is the over-frozen extreme (never tracks any
segment, excess `0.29`–`0.84`). The noise ball is not a bug — on a drifting source it is
the adaptation mechanism. 8/8 acceptance checks pass.
