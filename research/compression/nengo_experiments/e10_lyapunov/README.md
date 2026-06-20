# e10 — Excess energy is a Lyapunov function: descent + the noise ball

Tests paper claim #11: the excess energy `V(W) = H(p,q) − H(p) = Σ_i π_i D_KL(P_i‖q_i)`
is a Lyapunov function for learning — the averaged flow `Ẇ = −η∇V` gives
`V̇ = −η‖∇V‖² ≤ 0` (monotone descent), the loss is convex with the unique fixed point
`q = P`, a decreasing Robbins–Monro step converges to the floor, and a constant step
settles into an `O(η)` noise ball. We anchor the deterministic theory with two exact
numpy audit checks and make the noise ball physical with the e09 spiking PES learner.

**Headline.** The two adversarial audit anchors reproduce exactly: sum-mode
conservation `Σ_j ΔW_ij = 0` holds to machine ε (per-step **9.2e−17**, row-sum drift
**7.3e−14**, from a nonzero init), and the Lyapunov-rate constant is
`k = 0.693131 = ln2` (NOT η; |k−ln2| = 1.6e−5) — the corrected bits-vs-nats factor.
The averaged flow dissipates `V` monotonically to the floor (frac non-increasing 1.0,
max increase 0.0). The spiking learner descends the energy from the uniform init 2.0
bits below the marginal 1.7500 (by step 12) along a monotone cummin envelope. The
constant-η noise ball has **two faces of one O(η) trade-off**: the *asymptotic* face,
shown converged in numpy (smaller η → smaller ball: 1.6e−2→0.0021, 1.2e−2→0.0016,
8e−3→0.0012, 5e−3→0.0011), and the *descent-speed* face, shown physical in the spiking
PES residual at a fixed 8000-symbol budget (larger η descends faster → smaller residual:
1e−3→0.0222, 5e−4→0.0346, 2e−4→0.0438, 1e−4→0.0470). A spiking run long enough to
converge the slow rates (>1e6 symbols) is infeasible, so the spiking sweep honestly
reads the descent-speed face while the exact numpy twin reads the asymptotic ball — one
trade-off, two budgets. The spiking decoder's representational partition defect
`|Σ_j q_j − 1| = 7.8e−4` (e03) sits ~1e10× above the exact float64 invariant. 9/9
acceptance checks pass.
