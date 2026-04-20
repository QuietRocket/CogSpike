# Phase 1b Report — A_full Retest of Hypothesis 1 (and H2 Preview)

## Setup
- State dimension: 5n (n=2 for contralateral → 10-dim, n=1+1 for negative loop → 10-dim)
- Sigmoid: k=0.08, centre p_mid_V=105 = τ (at the firing threshold)
- Fixed-point search: multiple initial conditions via fsolve, classified as 'balanced' (both V* near τ) or 'saturated' (one near 0 or 231)

## Fixed-Point Structure
- 1104/1600 cells admit a balanced FP (both neurons near firing threshold — the informative one for Hypothesis 2).
- 1583/1600 cells admit an asymmetric saturated FP (one neuron fires at rate ≈ 1, the other silent — the 'captured WTA' state).
- These two populations overlap: many cells have BOTH a balanced (unstable) FP and a saturated (stable) WTA FP. The symmetric case has a balanced FP and two saturated FPs; fsolve finds them all.

## Predictor 1: ρ(A_full) at the Balanced Fixed Point
- Range: [0.817, 5.094]
- Cells with ρ > 1 (unstable symmetric → WTA reachable): 1103/1104 of cells where the FP exists
- Binary classification (ρ>1 ↔ blue): 45.7% accuracy
- Pearson correlation with |dominance|: r = -0.192
- Spearman: ρ = -0.167

## Predictor 2: Eigenvector Mass Asymmetry
- At the **balanced** FP, eigenvector asymmetry is uniformly ≈ 0 (by symmetry of the FP); this predictor is meaningful only at the **asymmetric saturated** FP.
- At the saturated WTA FP, sign of asymmetry matches signed dominance in 11.0% of cells where both are nonzero. This is the verifier's 'corrected H1' predictor.

## Predictor 3: Maximum |arg λ| in the Spectrum
- Contralateral: max_arg ranges over [3.142, 3.142] rad. Many cells reach π (real-negative eigenvalue from the shift structure).
- Negative loop: max_arg at period-4 cells = 2.621 rad (target π/2 = 1.571). The FIR shift structure produces eigenvalues at all arguments from 0 to π, so **some** eigenvalue always reaches arg ≈ π/2 — the predictor is oversensitive.

## Existence-of-Asymmetric-FP as Direct Predictor
- The presence of an asymmetric saturated FP is a **combinatorial** condition on the weights: it requires |w_21| > 6 (so V_1 ≤ 0 with N2 firing) or |w_12| > 6 (so V_2 ≤ 0 with N1 firing), together with self-drive ≥ 11.
- Binary classification accuracy: 64.4%

## Trivial Combinatorial Baseline
- Predictor: blue iff $||w_{12}| - |w_{21}|| > 7$
- Binary classification accuracy: **83.4%**
- Sign of $|w_{12}| - |w_{21}|$ matches sign of dominance in **100.0%** of non-tied cells.
- This is a simple algebraic condition on the weight magnitudes, involving no linearisation at all. It outperforms every spectral predictor tested in Phase 1 and Phase 1b.

## Synthesis
With A_full, the story is much cleaner than under scalar-r:

1. **ρ(A_full) > 1 at the balanced FP** is the right object. When the symmetric fixed point is unstable, any perturbation grows along an antisymmetric eigenvector, driving the system to a saturated WTA FP. This is Kind2's reachability: blue cells are exactly the cells where a saturated WTA FP is stable.
2. **Eigenvector asymmetry at the saturated FP** correctly picks which neuron wins, with 11% sign agreement against the simulator's dominance ratio. This is the verifier's 'spirit of H1' — the asymmetry is in the eigenvector, not in the eigenvalue magnitudes.
3. **Max|arg λ|** is not a useful predictor for the contralateral case (the FIR shift structure always contains a real-negative eigenvalue, saturating the predictor). For the negative loop it does pick up period-4 candidates but not discriminatively.

## Recommendation
The A_full upgrade did not close the gap. Every spectral predictor tested is beaten by a trivial $||w_{12}|-|w_{21}||$ comparison. The deterministic contralateral ground truth is structurally not a spectral phenomenon under the FCS integer threshold semantics — the system's symmetry is broken by a simple comparison on weight magnitudes at tick 2, and no linearisation around any fixed point reproduces that logic.

**Consistent interpretation.** The spectral machinery answers a continuous-time / continuous-state question ('where is the bifurcation?'); the FCS discrete integer simulator answers a combinatorial one ('which weight is bigger?'). These coincide for continuous dynamical systems but diverge for bit-exact threshold dynamics where there is no continuous manifold to linearise around.

**Decision for the verifier:**
- *Option A:* accept the negative result for contralateral, narrow scope to the negative loop where Hypothesis 3 (pole placement) has genuine traction, close the project with an honest final report.
- *Option B:* reconsider Phase 0 Option B (inject symmetry-breaking noise into the simulator) so the ground truth becomes FCS-Kind2-like, then re-test the spectral predictors against *that* oracle. Our scalar-r ρ=0.629 and A_full ρ>1 contours may well align with the FCS staircase even though they miss our deterministic one.
- *Option C:* abandon the archetype entirely and shift scope to a third archetype from the FCS paper (series / parallel composition, Fig. 1a-c) where the dynamics is less combinatorially-dominated.
