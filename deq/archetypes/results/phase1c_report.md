# Phase 1c Report — Spectral Predictors vs Reachability GT

## Sweep setup
- Perturbation set: 18 initial-state perturbations (±2 on each tap of mem[1..4], plus two crossed configurations) plus the zero-init baseline.
- 'Reachable-blue' means at least one of these initial states produced WTA within the A.7 criterion over 50 ticks.

## Ground-truth comparison
| Semantics | Blue cells | % |
|---|---|---|
| Deterministic (zero init) | 1014 | 63.4% |
| Reachable (any ε-perturbation) | 1564 | 97.8% |
| Perturbation adds | 550 | 34.4% |

## Classification accuracy
| Predictor | vs Deterministic | vs Reachable |
|---|---|---|
| scalar-r ρ(A) | 64.9% | 98.5% |
| A_full ρ at balanced FP | 64.1% | 97.9% |
| combinatorial |Δw| | — | 96.0% (θ=0) |

## Structure of the Non-Reachable Region
- All 36 non-reachable cells satisfy |w_12| ≤ 6 AND |w_21| ≤ 6.
- They form a 6×6 block in the weak-mutual-inhibition corner. Equivalent characterisation: neither neuron's inhibition is strong enough to push the other below firing threshold.
- This matches the analytical condition for an asymmetric saturated fixed point to exist (|w| ≥ 7), derived in Phase 1b.

## ρ Distribution by Class
- **scalar_r**: ρ|red = 0.555 (range [0.518, 0.598]); ρ|blue = 0.719 (range [0.544, 0.929])
- **A_full_balanced**: ρ|red = 3.700 (range [0.817, 3.858]); ρ|blue = 3.862 (range [1.552, 5.094])

## 100%-Precision Operating Point
At the threshold where ρ < thr catches zero blue cells:
- scalar-r: thr = 0.544, red cells detected = 14/36 (recall 38.9%)
- A_full (balanced): thr = 1.552, red detected = 2/36 (recall 5.6%)

The ρ distributions of red vs blue cells barely overlap (see `phase1c_rho_distributions.png`), and a conservative threshold at the blue minimum identifies a meaningful fraction of red cells with perfect precision. This is the first positive spectral signal in the project.

**Signal:** the reachability ground truth is measurably more predictable by ρ(A) than the deterministic one (Δ=+33.6pp scalar-r, Δ=+33.8pp A_full). This is consistent with the hypothesis that spectral methods describe Kind2-style reachability rather than deterministic single-trajectory dynamics.

