# `deq/closed_form_neg_loop/` — Three closed-form lenses on FCS Property 5

Same three lenses as `deq/closed_form_wta/` (Siegert mean-field /
Richardson H(ω) / Naud–Gerstner quasi-renewal), now applied to the
**negative-loop archetype** (FCS Fig. 1d, §6.2.5, Property 5).

Property 5 (De Maria et al. 2020, p. 12) is the oscillation property:
with two delayer-like neurons in a negative loop and constant input,
the activator A fires the period-4 pattern `1100` and the inhibitor I
echoes it one tick later. Unlike contralateral inhibition's
winner-takes-all (a stable bistability), the negative loop is
intrinsically **oscillatory**, so the three lenses get reinterpreted:

| Lens         | WTA reading (`closed_form_wta`)                  | Negative-loop reading                                              |
| ------------ | ------------------------------------------------ | ------------------------------------------------------------------ |
| Siegert FP   | enumerate FPs, ν*₁−ν*₂ ≥ 0.30 → blue              | FP exists & Jacobian has Im(λ) ≠ 0 → spiral (sustained ringing)    |
| H(ω) Jacobian | slowest mode \|Re(λ)\| > 1/T_FCS = 0.25 → blue   | predicted ringing period T_pred = 2π/\|Im(λ)\|; compare to FCS-4   |
| Quasi-renewal | post-warmup Δν ≥ 0.30 → blue                    | period & 1100-template score on A(t) at N ∈ {50,100,500,2000}      |

For single-pole low-pass H(ω) and the negative-loop weight matrix
W = [[0, w_IA], [w_AI, 0]], the rate-equation Jacobian eigenvalues are
λ = (−1 ± √(g_A g_I w_AI w_IA))/τ_m. Because w_IA < 0 and w_AI > 0,
the radicand is negative ⇒ a complex-conjugate pair with Re(λ) =
−1/τ_m < 0 — the FP is always a **stable spiral**, never Hopf-unstable.
Rate theory therefore predicts decaying ringing, not sustained
oscillation; the discrete-tick FCS semantics promote that ringing to
a true limit cycle. The interesting prediction is the *ringing
frequency*: T_pred ≈ 2π / \|Im(λ)\|, which we check against FCS's
exact period 4.

## Layout

- `phase0_fcs_baseline.py` — FCS LIF oracle over (w_IA, w_XA) grid;
  labels `strict_p5` (exact 1100 pattern) and `broad_osc` (any
  oscillation), plus measured period.
- `phase1_siegert_hopf.py` — Siegert FP + Jacobian eigenvalues.
- `phase2_freq_gate.py` — H(ω) predicted period T_pred = 2π/\|Im(λ)\|.
- `phase3_finite_N.py` — quasi-renewal mesoscopic at N ∈ {50, 100,
  500, 2000}.
- `note/closed_form_neg_loop_note.typ` — final integrating note.
- `results/phase{0..3}/` — per-phase numerical outputs and plots.
- `make_all.sh` — runs all phases end-to-end.

## Reuse (no reimplementation)

- FCS LI&F oracle: [`deq/archetypes/lif_fcs.py:simulate`](../archetypes/lif_fcs.py)
- Negative-loop topology: [`deq/archetypes/topologies.py:13`](../archetypes/topologies.py)
- Siegert φ: [`deq/closed_form/siegert.py`](../closed_form/siegert.py)
- Jacobian eigenvalues + dφ/dμ: [`deq/closed_form/transfer.py`](../closed_form/transfer.py)
- Quasi-renewal stepper: [`deq/closed_form/quasi_renewal.py`](../closed_form/quasi_renewal.py)
- Calibration constants (α, β, τ_m, τ_ref): locked at the
  [`closed_form/results/phase1_grid.npz`](../closed_form/results/phase1_grid.npz)
  operating point (p_thin = 0.7), same as `closed_form_wta`.

## Grid

- `(w_IA, w_XA) ∈ {-40..-1} × {1..40}`, `w_AI = 11` fixed.
- 1600 cells per phase.
- FCS default `(w_IA, w_XA) = (-11, 11)` lives at the centre as the
  sanity-gate Property 5 cell.

## Run

```sh
./make_all.sh
```

Final PDF: `note/closed_form_neg_loop_note.pdf`.
