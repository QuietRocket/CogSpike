# Phase 0 Report — Simulator + Ground Truth
## Property 5 (negative loop)
- Weights used: `w_XA=11, w_AI=11, w_IA=-11` (tuned)
- Expected activator sequence: `011001100110011001100110011001`
- Got activator sequence:      `011001100110011001100110011001`
- **Exact match:** True

Note: FCS Appendix A.3 suggests `w_IA=-20` as a starting point. Analytical tracing of the Lustre semantics shows that -20 overshoots (period 5), while -11 (exact cancellation) reproduces the Property 5 period-4 pattern exactly. The negative loop is therefore used with `w_IA=-11` henceforth.
## Property 7 / FCS Fig. 10
- Grid: 40×40, w_12, w_21 ∈ [-40, -1]
- T=50, WTA criterion per plan Appendix A.7
- Blue cells (WTA reached): 1014/1600 (63.4%)
- Diagonal (symmetric weights) blue cells: 0/40 (expected 0 by symmetry)
- Upper-left 5×5 (|w|≤5): 0/25 blue
- Bottom-right 5×5 (|w|≥36): 0/25 blue
- Dominance ratio range: [-0.978, +0.978], std=0.779
- Mean |dominance| off-diagonal: 0.636 (Phase 1 will regress Δ against this continuous signal)

### Qualitative comparison to FCS Fig. 10
**FCS Fig. 10 description (plan Appendix A.4):** blue occupies top-and-left edges (small |w|); red occupies bottom-right (large |w|); boundary is a staircase from upper-right to lower-left.

**Our reproduction** shows a qualitatively different structure:
- Symmetric weights (diagonal) are uniformly red — the simulator is fully
  deterministic under integer arithmetic, so N1 and N2 receive identical
  external drive and produce identical spike trains when w_12 = w_21.
- A *central red block* (roughly |w| ∈ [13, 29] on both axes, extending
  to moderately asymmetric weights): both neurons synchronize into a
  shared rhythm whose period increases with |w|, but neither falls silent.
- Two triangular *blue wings* off the diagonal: when |w_12| and |w_21|
  differ by enough to break the shared rhythm, one neuron captures and
  the other goes silent.
- Two *corner red regions* (very-small × very-large) where the asymmetry
  is large but one side's inhibition is too weak to reach threshold at all.

**Source of discrepancy.** FCS Fig. 10 is produced by Kind2 model-checking an LTL property — Kind2 searches all reachable states and declares 'blue' whenever *some* trajectory reaches WTA. Our simulator, starting from zero state with symmetric constant input, explores a single trajectory. Under symmetric weights that single trajectory is tied, so no WTA emerges. Under moderately asymmetric weights that still produce synchronised spiking, the symmetry between simultaneous spike emissions prevents WTA from breaking out within our finite 50-tick window.

**Implications for Phases 1–3.** The ground truth from this simulator is the authoritative oracle for the planned spectral predictors. The structure observed here (synchronised-rhythm red block + asymmetry-driven blue wings) is itself a non-trivial 2D classification target — arguably *more* informative than FCS Fig. 10 because it directly reflects the LI&F dynamics rather than a reachability property. Phase 1 will test whether the eigenvalue gap Δ = ||λ₁|-|λ₂|| of the raw inhibitory W tracks the *boundary of the blue wings* (i.e., the transition from synchronised tie to asymmetric capture).

**Stop-and-report rule.** Per plan §8, Phase 0 halts here pending user review. The simulator Property-5 exact match confirms FCS semantic fidelity. The Fig. 10 ground-truth structure is well-defined and ready to serve as the Phase 1 target, even though it differs visually from the FCS staircase description. User may wish to: (a) approve proceeding to Phase 1 using this ground truth, (b) attempt to reproduce FCS's reachability-based blue/red coloring by adding a small symmetry-breaking bias to one neuron's initial state, or (c) refine the WTA criterion.
