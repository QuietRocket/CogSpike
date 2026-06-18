# research/compression — Spiking Entropy Coder

A two-note series porting the *compression ⟺ prediction ⟺ intelligence* thesis
(Shannon 1948/1950) into recurrent spiking neural circuits, in the CogSpike
archetype-and-verification idiom. **Note I** builds the coder; **Note II**
derives the learning rule that makes it good.

## Files

| file | what it is |
|---|---|
| `spiking_entropy_coder.typ` | **Note I** — the coder: spike-time = cross-entropy |
| `learning_to_be_unsurprised.typ` | **Note II** — local plasticity that descends the energy |
| `validate.py` | numerical validation for Note I (numpy only) |
| `learn_validate.py` | numerical validation for Note II (numpy only) |
| `spiking_entropy_coder.pctl` | PRISM-style property templates (safety + optimality) |

Reproduce the numbers:

```
deq/.venv/bin/python research/compression/validate.py        # Note I
deq/.venv/bin/python research/compression/learn_validate.py  # Note II
```

Build the notes (if `typst` is on PATH):

```
typst compile research/compression/spiking_entropy_coder.typ
typst compile research/compression/learning_to_be_unsurprised.typ
```

## The one idea

A spiking neuron pre-charged to threshold by its *expected* input is silent; a
neuron whose input violates its recurrent context's prediction fires. So a
recurrent circuit that subtracts its own prediction emits, per symbol, a spike
cost equal to the surprisal `−log₂ q(xₜ | context)`. Total spike-time over a
stream = **cross-entropy** of the predictor against the source. The entropy
rate is the floor; the excess is the KL divergence, paid in joules. The loop is
"intelligent" exactly to the degree it closes that gap — and every clause is a
checkable property.

## Design decisions (and why)

The note (`spiking_entropy_coder.typ`) is fully self-contained — it builds the
neural dynamics, information theory, coding theory, Markov source, and recurrent
motifs from scratch. Three decisions shape its direction:

1. **Latency, proved exact — not rate, not a hedge.** Derives the exact LIF
   drive `R·I(q) = θ/(1−qᵅ)` that makes first-spike latency *equal* the Shannon
   codeword length `−λ log₂ q` (note Thm 1; `validate.py` confirms to ODE step
   size). Time-to-first-spike is the encoding that is exact, prefix-free for
   free, and costs one spike per symbol.

2. **A continuous-time advantage discrete codes can't have.** Spike *time* is a
   real number, so the latency code escapes the integer-codeword penalty
   `H ≤ L < H+1`; per-symbol cost equals exact surprisal with no block-length
   limit (note Thm 2). Stated honestly: the combinatorial `+1`-bit penalty is
   *traded* for an analog timing-resolution penalty, not abolished.

3. **Safety separated from optimality.** Losslessness is *structural* (WTA mutual
   exclusion) and holds for any predictor, however bad; near-entropy cost is
   *quantitative* and predictor-dependent. A bad model is slow, never wrong.
   This split is the formal-methods contribution and drives the §11 / `.pctl`
   two-class property layout.

The source is a principled one-parameter **momentum rover** `P = sI + (1−s)·1πᵀ`
that provably preserves the marginal `π = (½,¼,⅛,⅛)` for all `s`, so the *only*
thing the cycle changes is the conditional structure. At `s=0.7` the memory is
worth `I(xₜ;xₜ₋₁) = 1.75 − 0.978 = 0.772` bits/symbol (44%) — fixed in closed
form before any circuit is built.

## Note II — the learning rule (`learning_to_be_unsurprised.typ`)

Note I assumed a good predictor `q ≈ p`; Note II **derives the rule that gets
there**, and the punchline is an identity, not an add-on:

> The residual **error spike** the encoder emits as a symbol's *cost* is — when
> correlated with the context that predicted it — exactly the gradient that
> lowers the cost. **Cost and learning are the same spikes.**

- **The rule.** Gradient descent on the per-symbol energy `−log₂ q(xₜ|c)` is the
  local three-factor Hebbian update `ΔWᵢⱼ = η · cᵢ · (yⱼ − qⱼ)` (pre × residual ×
  gate). The residual `yⱼ − qⱼ` is the predictive-subtraction error spike. The
  gradient identity is confirmed numerically to `1e-10`.
- **Energy is the Lyapunov function.** The excess energy `= avg KL` (Note I Thm 6)
  is convex in the weights and dissipates monotonically under the averaged flow;
  the unique fixed point is `q = p`. So "spend less energy" and "predict better"
  are *one* monotone descent. `learn_validate.py` takes the circuit from `2.0`
  → `0.979` bits/symbol (the entropy-rate floor), KL → `0.0004`.
- **Structural bonus.** `Σⱼ ΔWᵢⱼ = η(1−1) = 0`: learning lives entirely in the
  **difference modes**; the **sum-mode** normalization invariant (`Σqⱼ = 1`) is
  preserved at *every* step — directly reusing Note I's mode decomposition.
- **It re-derives four known objects** (delta / Rescorla–Wagner / predictive
  coding / free-energy principle), regrounded as descent on spiking energy.
- **New verification obligations**: locality, sum-mode invariance,
  weight-boundedness (safety); energy descent, convergence to the floor (limit).
- **Research programme (Note II §10)**: eligibility traces / TD(λ) for deeper
  memory, constant-η tracking of non-stationary sources (regret bounds), learned
  lossy quantization, on-chip neuromorphic plasticity, learned calibration.

## How it lands in the existing program

- The decode head **is** the contralateral-inhibition WTA archetype already
  verified; its "always eventually exactly one winner" pair = lossless +
  unambiguous decoding.
- The sum/difference **mode decomposition** certifies correctness: difference
  mode = symbol selection (pitchfork = decision boundary); sum mode under
  divisive normalization = `Σqᵢ = 1` (the partition-of-unity / interval-tiling
  invariant).
- The reachable compression floor is an **attractor-capacity** question — the
  `closed_form_wta_multi` / population toolkits.

## Open threads

- ✅ ~~The plasticity rule that performs the energy descent is assumed, not
  derived.~~ **Resolved in Note II** (local three-factor Hebbian rule + Lyapunov
  proof).
- Lossy / rate–distortion via graded WTA (bump width = distortion knob) — now
  with a *learnable* quantizer (Note II §10).
- Realistic anchor: neuromorphic **event-camera** streams (data already spikes).
- **Channel** coding, where recurrence becomes provably necessary (LDPC belief
  propagation = recurrent message passing; attractor settling = MAP decoder).
- Eligibility traces / TD(λ) for deeper-than-first-order memory; non-stationary
  tracking with regret bounds (Note II §10).
