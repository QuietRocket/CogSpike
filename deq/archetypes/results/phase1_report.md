# Phase 1 Report — Eigenvalue Gap as WTA Predictor
## Setup
- Sigmoid approximation: k=0.08, p_mid=30.0 (tuned from fallback sweep; saturates at p_mid=90)
- Scalar leak r: using 0.5 (LS-fit to rvector gives 0.545)
- Ground truth: /Users/quietrocket/Documents/PhD/CogSpike/deq/archetypes/results/fcs_fig10_groundtruth.npy (1014/1600 blue cells from Phase 0)

## Hypothesis 1 outcome on contralateral inhibition

### (A) Δ(W): raw weight eigengap
Analytical result: for a 2×2 zero-diagonal W with off-diagonal entries $a, b$, the eigenvalues are $\pm\sqrt{ab}$ with **equal magnitudes** regardless of $|a|$ vs $|b|$. So Δ(W) ≡ 0 over the entire sweep. The raw-W eigengap is **not** a usable predictor for this archetype — a fact that was not flagged in the original plan and is worth recording.

- Verified numerically: max Δ(W) across 1600 cells = 1.421085e-14 (machine-ε noise).

### (B) Δ(A): linearised-state eigengap
- Best classification accuracy (over polarity × threshold): 68.9% against binary WTA.
- Correlation with |dominance|: Pearson r = -0.245, Spearman ρ = -0.250.
- The scalar-r sigmoid linearisation suffers from operating-point saturation: even at the tuned p_mid=30.0, the mean sigmoid derivative is 0.0128, three orders of magnitude below a typical neural gain. This keeps Δ(A) tiny everywhere and gives a weak predictor.

### (C) ρ(A): spectral radius (Phase 2 preview)
- Best classification accuracy: 68.9%.
- Correlation with |dominance|: Pearson r = -0.245, Spearman ρ = -0.250.
- ρ(A) ranges over [0.5178, 0.9292] — well below 1 across the entire sweep under the scalar-r linearisation. The $\rho=1$ contour is therefore **empty** in our grid, confirming that the scalar-r linearisation under-predicts instability. Phase 2 will need a richer state representation (the full 5-tap memory per neuron) to capture the true bifurcation.

## Hypothesis 1 outcome on the negative loop
- Simulator period-4 cells: 16 of 400 (4.0%). The cleanest case is $w_{IA} = -w_{AI}$ (exact-cancellation rule — this is the Phase 0 tuning finding).
- arg(λ_dom) range over the sweep: [0.021, 0.466] rad.
- Does the arg = π/2 contour align with the simulator's period-4 cells?
  - **No cells in the swept range have arg(λ_dom) within 0.1 rad of π/2.** The scalar-r linearisation produces arg ≈ 0 (real dominant eigenvalue) across most of the sweep, fundamentally failing to predict the period-4 oscillation.
  - This is the same root cause as (B)/(C) above: the FCS windowed integrator (rvector=[10,5,3,2,1]) is a length-5 FIR filter whose dynamics cannot be captured by a single scalar r. The period-4 oscillation of Property 5 requires poles at the 8th roots of unity of the full FIR transfer function.

## Root Cause: Scalar-r Linearisation is Too Crude
Across all three predictors, the common failure mode is the collapse of the 5-tap windowed integrator to a scalar leak. The true state of a single FCS neuron is (mem[0..4]), so an n-neuron network has a 5n-dimensional state, not n. Phase 2 therefore requires building the **full 5n × 5n state matrix** (with the FIR-filter shift as off-diagonal blocks, the summed-input map on the top-row block, and the spike-reset non-linearity linearised around the operating point).

## Recommendation
Hypothesis 1 is **not validated** in its stated form. The failure is not a flaw in the spectral-cartography programme — it is a specification error: the plan prescribed a scalar-r linearisation that cannot represent the 5-tap FIR-filter dynamics that FCS LI&F neurons actually execute.

Concretely:
1. Raw-W eigengap is provably identically zero for 2×2 zero-diagonal weight matrices (analytical, not empirical).
2. Scalar-r linearised A saturates at the operating point and produces predictors ~3 orders of magnitude weaker than needed.
3. Phase 2's ρ=1 bifurcation contour will not exist under scalar-r; we must build the 5n×5n state matrix before testing Hypothesis 2.

**Proposed path forward.** Before formally proceeding to Phase 2, produce a revised `spectral.py` that builds the full 5n-dim state matrix $A_{full}$ directly from the FCS dynamics (the FIR shift + spike-reset linearisation). Then retest Hypotheses 1 and 2 against the same ground truth. This is ~50 lines of code and is a one-time investment.

Pending user decision, the existing scalar-r artifacts are preserved in `results/phase1_*.png` as documentation of the failure mode.
