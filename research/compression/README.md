# research/compression — Spikes as Bits, Learned

One self-contained paper porting the *compression ⟺ prediction ⟺ intelligence*
thesis (Shannon 1948/1950) into recurrent spiking neural circuits, in the
CogSpike archetype-and-verification idiom. It **builds the coder**, **derives the
local plasticity rule that makes it good**, and **folds in a meticulous critical
self-evaluation** — every fixable defect found in review is corrected in-text,
and the residue is flagged as open problems.

It supersedes the original two-note series (now in [`archive/`](archive/)).

## Files

| file | what it is |
|---|---|
| `unified_spiking_compression.typ` | **the canonical paper** (69 pp): coder + learning rule + critical evaluation |
| `refs_compression.bib` | bibliography (Shannon, Kraft, Widrow–Hoff, Rescorla–Wagner, Rao–Ballard, Friston, Carandini–Heeger, Thorpe, Rissanen, …) |
| `validate.py` | numerical validation for the coder (numpy only) |
| `learn_validate.py` | numerical validation for the learning rule (numpy only) |
| `critique_checks.py` | two checks added during the audit: sum-mode invariance, the Lyapunov constant |
| `spiking_entropy_coder.pctl` | PRISM-style property templates (structural/safety + quantitative/optimality) |
| `archive/` | the superseded Note I + Note II (frozen, for provenance) |

Reproduce every number in the paper:

```
deq/.venv/bin/python research/compression/validate.py
deq/.venv/bin/python research/compression/learn_validate.py
deq/.venv/bin/python research/compression/critique_checks.py
```

Build the paper (if `typst` is on PATH):

```
typst compile research/compression/unified_spiking_compression.typ
```

## The one idea

A spiking neuron pre-charged to threshold by its *expected* input is silent; a
neuron whose input violates its recurrent context's prediction fires. So a
recurrent circuit that subtracts its own prediction emits, per symbol, a
first-spike *latency* equal to the surprisal `−λ·log₂ q(xₜ | context)` — proved
exactly via the calibration drive `R·I(q) = θ/(1−qᵅ)`. Total spike-**time** over a
stream is the **cross-entropy** of the predictor against the source; the entropy
rate is the floor and the excess is the KL divergence. Gradient descent on that
per-symbol cost *is* a local three-factor Hebbian rule
`ΔWᵢⱼ = η·cᵢ·(yⱼ−qⱼ)`, whose excess energy is a Lyapunov function that descends to
`q = p`. Compressing, predicting, spending less, and learning are one monotone
descent on one number — checkable property by property.

## How this paper was produced

The two original notes were put through a meticulous multi-perspective critique
(a multi-agent workflow: a claim ledger over both notes, seven critical lenses —
proof rigor, information/coding theory, dimensional/constant checks, learning
theory, neuroscience plausibility, formal-methods verifiability, novelty —
followed by adversarial verification of every finding and a numerics
reproduction pass). Of 70 findings raised, **50 were confirmed** (13 major, 36
minor, 1 cosmetic) and 20 dismissed by adversarial verification. A second
synthesis workflow merged the notes and applied every confirmed fix in-text. The
full audit is documented in the paper's **Critical Evaluation** section.

### Headline corrections folded in

- **Lyapunov constant.** The original `V̇ = −η‖∇V‖²` dropped a `ln 2` factor and
  mixed two flow definitions. Fixed: the averaged flow is `Ẇ = −η∇V` (so
  `V̇ = −η‖∇V‖²` exactly), related to the rule's expected step; `critique_checks.py`
  CHECK 2 confirms the unscaled-flow constant is `ln 2 = 0.693`.
- **Cost-spike = teaching signal**, now stated *conditionally*: the gradient
  equals `cᵢ(yⱼ−qⱼ)` (proved); the residual is *realized* by a two-channel
  ON/OFF error population — a **distinct** observable from the latency code, not
  "the same spikes".
- **Time vs energy.** The exact, proven observable is spike-**time** (`λ`, s/bit);
  the **energy** reading (`κ`, J/bit) is a model-dependent corollary, never
  silently swapped.
- **Calibration realism.** `R·I/θ ≈ 7.1×` at `q=0.9`, `≈70×` at `q=0.99`, pinning
  `q_max ≈ 0.85–0.95`; predictive subtraction restated as feedback inhibition.
- **Coherence win from merging.** The predictor Note I left unspecified is exactly
  Note II's `q = softmax(W·c_t)`; the same softmax (not divisive normalization,
  which sums to `< 1`) carries the partition-of-unity invariant.
- **WTA safety** restated in settled `F G` form; **weight-boundedness** gauge-fixed;
  **citations** added throughout; channel-coding necessity softened to a conjecture.

## How it lands in the existing program

- The decode head **is** the contralateral-inhibition WTA archetype already
  verified; "eventually exactly one settled winner" = lossless + unambiguous decode.
- The sum/difference **mode decomposition** certifies correctness; the reachable
  compression floor is an **attractor-capacity** question — the
  `closed_form_wta_multi` / population toolkits.
