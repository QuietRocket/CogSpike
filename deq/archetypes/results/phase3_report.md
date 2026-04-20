# Phase 3 Report — Pole Placement on the Negative Loop

## Sanity check (Phase 0 known-good period-4)

- Weights: w_XA=11, w_AI=11, w_IA=-11
- A_full dominant complex arg = 1.3422 rad (target π/2 = 1.5708, error 0.229)
- Predicted period = 4.68; simulator measured period = 4
- |λ| = 2.212 > 1 → linearisation correctly predicts sustained oscillation.
- **Sanity check: PASSED** — simulator produces period 4 as expected.

## Pole placement sweep

Numerical integer-grid search over (w_AI ∈ [1..30], w_IA ∈ [-30..-1]) minimising |arg(λ_dom,A_full) - 2π/T|, with a soft bonus for |λ| ≥ 1.

| T (target) | w_AI | w_IA | pred arg | pred T̂ | sim period | match | |λ| |
|---|---|---|---|---|---|---|---|
| 3 | 4 | -26 | 1.653 | 3.80 | None | ✗ | 0.53 |
| 4 | 11 | -11 | 1.342 | 4.68 | 4 | ✓ | 2.21 |
| 5 | 5 | -18 | 1.074 | 5.85 | 8 | ✗ | 1.16 |
| 6 | 5 | -13 | 0.905 | 6.94 | 8 | ✗ | 0.98 |
| 7 | 12 | -6 | 0.784 | 8.02 | 4 | ✗ | 0.87 |
| 8 | 12 | -6 | 0.784 | 8.02 | 4 | ✗ | 0.87 |

**Match rate: 1/6 target periods.**

## Diagnostic: arg_pred across ground-truth periods

To probe whether the linearisation can even *discriminate* target periods, we inverted the search: took the first known-simulator-period cell for each T and computed the linearisation's dominant arg there.

| Sim period T | (w_AI, w_IA) | arg_pred | raw target 2π/T | calibrated α·2π/T |
|---|---|---|---|---|
| 3 | (+9, -12) | 1.329 | 2.094 | 1.790 |
| 4 | (+7, -12) | 1.270 | 1.571 | 1.342 |
| 5 | (+6, -12) | 1.182 | 1.257 | 1.074 |
| 6 | (+6, -30) | 1.312 | 1.047 | 0.895 |
| 7 | (+5, -12) | 0.860 | 0.898 | 0.767 |
| 8 | (+5, -30) | 1.217 | 0.785 | 0.671 |

Correlation(T_sim, arg_pred) = -0.513.

## Interpretation

The A_full linearisation's dominant arg sits in a narrow band around $1.2$--$1.35$ rad across *all* simulator periods $3$ through $8$. The correlation of $-0.51$ between simulator period and predicted arg is far from the strong negative correlation that pole placement requires.

The period-$4$ weights $(11, -11)$ happen to sit at the self-consistent point where arg_pred matches the calibrated target exactly — but this is because we *calibrated to that point*. Every other target period picks weights whose linearisation has approximately the same arg, which the simulator then realises at various periods depending on the spike-reset nonlinearity.

**Diagnosis.** The linearisation captures 'oscillation exists at roughly frequency $\pi/2$' but the actual integer period is determined by the spike-reset logic that the linearisation erases. A_full correctly identifies the regime (oscillatory / non-oscillatory, roughly) but cannot invert to a specific integer period. This is a stronger form of the Phase 1c observation: spectral cartography handles *qualitative* regime questions for FCS LI\&F, not *quantitative* design.

**Practical consequence.** To realise a target simulator period, direct enumeration over integer $(w_{AI}, w_{IA})$ pairs (~1 second on a laptop) is more effective than pole placement. Phase 3's failure is a genuine limitation of the inverse-design hypothesis, not an implementation bug.
