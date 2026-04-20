// Phase 1b Report — A_full retest of Hypothesis 1
// CogSpike / LI&F Archetypes — April 2026

#set document(
  title: "Phase 1b: A_full Retest — and a Combinatorial Surprise",
  author: "CogSpike Research Team",
  date: datetime.today(),
)

#set page(paper: "a4", margin: (x: 2.5cm, y: 2.5cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#let finding(body) = block(
  width: 100%, inset: 8pt,
  fill: rgb("#f0fff0"),
  stroke: (left: 2pt + rgb("#2e8b57")),
  [*Finding.* #body],
)

#let negresult(body) = block(
  width: 100%, inset: 8pt,
  fill: rgb("#fff0f0"),
  stroke: (left: 2pt + rgb("#b22222")),
  [*Negative result.* #body],
)

#let decision(body) = block(
  width: 100%, inset: 8pt,
  fill: rgb("#fde7f3"),
  stroke: (left: 2pt + rgb("#a83279")),
  [*Decision requested.* #body],
)

#let intuition(body) = block(
  width: 100%, inset: 8pt,
  fill: rgb("#f0f7ff"),
  stroke: (left: 2pt + rgb("#4a90d9")),
  [*Intuition:* #body],
)

#let remark(body) = block(
  width: 100%, inset: 8pt,
  stroke: (left: 2pt + orange),
  [*Remark.* #body],
)

#align(center)[
  #text(size: 18pt, weight: "bold")[
    Phase 1b --- Full 5n-Dim Linearisation \
    Retest (and a Combinatorial Surprise)
  ]
  #v(0.3em)
  #text(size: 12pt)[CogSpike Research Team --- April 2026]
  #v(0.3em)
  #text(size: 10pt, style: "italic")[
    The scalar-$r$ fix was correct but insufficient: the contralateral ground
    truth is fundamentally not a spectral phenomenon.
  ]
]

#v(1em)

*Abstract.* We replaced the scalar-$r$ linearisation with the full
$5 n$-dimensional state matrix $A_"full"$ that preserves the FCS windowed
integrator and retested all three predictors the verifier specified: spectral
radius $rho(A_"full")$ at the balanced fixed point, dominant-eigenvector
neuron-mass asymmetry, and maximum $|arg lambda|$ across the spectrum. All
three underperform the scalar-$r$ results from Phase 1a. Worse: a *trivial
algebraic predictor* --- blue iff $||w_(1 2)| - |w_(2 1)|| > 7$ --- achieves
$83.4%$ classification accuracy, and the sign of $|w_(1 2)| - |w_(2 1)|$
perfectly predicts ($100%$) which neuron wins whenever a winner exists. The
deterministic contralateral ground truth is therefore a *combinatorial*
phenomenon (integer threshold arithmetic), not a spectral one. The $A_"full"$
upgrade is technically correct but targets the wrong object: there is no
continuous manifold to linearise around when a bit-exact threshold comparison
at tick 2 is what breaks the symmetry. We request a strategic decision ---
narrow scope to the negative loop, pivot to FCS-Kind2 reachability semantics,
or change archetype.

= Setup

State dimension: $5 n$ (contralateral: $10$; negative loop: $10$). Sigmoid
centre $p_"mid,V" = tau = 105$ (at the firing threshold, where $f'$ is
largest). Fixed-point search uses multiple initial conditions and classifies
each solution as either *balanced* (both $V^star$ near threshold) or
*saturated* (one near $0$, other near $231$ = fully-firing).

Across the $40 times 40$ grid: $1104$ cells admit a balanced fixed point;
$1583$ admit an asymmetric saturated fixed point (most cells admit both plus
the symmetric WTA pair). The balanced FP is the informative one for
Hypothesis 2: its stability determines whether the symmetric trajectory is an
attractor or a repeller under continuous perturbation.

= The Three Corrected Predictors

== $rho(A_"full")$ at the Balanced Fixed Point

#figure(
  image("results/phase1b_rho_balanced.png", width: 95%),
  caption: [$rho(A_"full")$ at the balanced FP with $rho = 1$ contour
    (white) and ground-truth blue cells (black dots). White pixels are grid
    cells with no balanced FP. Where the balanced FP exists, $rho > 1$
    *everywhere* (range $[0.82, 5.09]$) --- the $rho = 1$ contour is empty.
    The linearisation declares the symmetric trajectory unstable wherever
    it exists, but the deterministic simulator does not follow the
    instability because integer symmetry is preserved tick-by-tick.],
)

#negresult[
  Binary classification $rho > 1 <==>$ blue: *$45.7%$* accuracy --- worse
  than the $63.4%$ majority baseline. The $A_"full"$ linearisation
  captures an instability that does not manifest in the deterministic
  ground truth because the two are answering different questions.
]

== Dominant Eigenvector Neuron-Mass Asymmetry

#figure(
  image("results/phase1b_asym_saturated.png", width: 95%),
  caption: [Eigenvector asymmetry at the asymmetric saturated FP (where
    such a FP exists). Sign agreement with dominance ground truth:
    $11%$ --- worse than random. At a saturated FP with one neuron at
    $V^star = 231$ (firing at rate $approx 1$) and the other at
    $V^star approx -300$ (silent), the sigmoid derivative $f'(V^star)$ is
    essentially zero for both neurons, so $A_"full"$ degenerates to the
    pure nilpotent FIR shift. The "dominant eigenvector" is then an
    artefact of the shift structure, not of the network's coupling.],
)

#negresult[
  At the balanced FP the asymmetry is uniformly zero (by symmetry of the
  FP). At the saturated FP the linearisation has vanishing gain on both
  neurons, making the eigenvector uninformative. The verifier's suggested
  "corrected H1 predictor" is well-motivated in a continuous-dynamics
  framework but has nothing to attach to at the FCS operating points.
]

== Maximum $|arg lambda|$ in the Spectrum

#figure(
  image("results/phase1b_maxarg.png", width: 95%),
  caption: [$max_k |arg lambda_k|$ of $A_"full"$ across the contralateral
    grid. The FIR shift matrix always contains a real-negative eigenvalue
    (the nilpotent shift has eigenvalues at the $5$th roots of $0$; with
    finite coupling the spread covers most of the unit circle), so
    $max |arg|$ saturates at $pi$ nearly everywhere. Uninformative.],
)

#figure(
  image("results/phase1b_negloop.png", width: 95%),
  caption: [Negative loop: $max |arg lambda|$ with $= pi/2$ contour (white)
    and simulator period-4 cells (cyan). Period-4 cells at $w_"AI" in {9, 10}$,
    $w_"IA" in [-20, -13]$ show $max |arg| = 2.62 plus.minus 0.00$ rad ---
    essentially the same value ($2.73 plus.minus 0.10$) as non-period-4
    cells. The FIR structure provides eigenvalues at every argument; the
    predictor cannot discriminate.],
)

= The Combinatorial Baseline

Motivated by the absence of any spectral signal, we tested the simplest
possible algebraic predictor:

$ "blue"(w_(1 2), w_(2 1)) <==> ||w_(1 2)| - |w_(2 1)|| > theta $

for integer threshold $theta$.

#figure(
  image("results/phase1b_combinatorial.png", width: 95%),
  caption: [Combinatorial magnitude-asymmetry heatmap $||w_(1 2)|-|w_(2 1)||$
    with the $theta = 7$ contour (white) and blue ground-truth cells (black
    dots). The contour traces the boundary of the blue wings almost exactly.],
)

#finding[
  - *Binary classification* blue $<==> ||w_(1 2)|-|w_(2 1)|| > 7$:
    *$83.4%$* accuracy --- well above the $68.9%$ that any spectral
    predictor (scalar-$r$ or $A_"full"$) achieved.
  - *Sign of winner*: $"sign"(|w_(1 2)| - |w_(2 1)|)$ matches $"sign"("dom")$
    in *$100%$* of non-tied cells.
]

#intuition[
  At tick $2$ of the contralateral simulator, both neurons have just fired
  (their first spike from the external drive at tick $1$) and now
  inhibit each other. Neuron $i$'s inhibition at tick $2$ equals the
  incoming weight $w_(j i)$. In the next tick, each neuron's V is
  $"rvector"[0] dot ("drive" + w_(j i))$, which crosses threshold or not
  depending on the single integer comparison "which weight is larger in
  magnitude". The whole subsequent dynamics flow from this bit-exact
  tick-2 decision. No linearisation around a continuous manifold can
  reach inside that comparison, because there is no manifold --- only
  a disjunction.
]

= Why Spectral Cartography Does Not Apply Here

Spectral analysis presumes a continuous state space with a smooth vector
field; it extracts poles of the linearised Jacobian around a continuous
manifold of fixed points. Under FCS integer threshold semantics neither
presumption holds:

- *State space is discrete.* $"mem"[e]$ is an integer in $[-380, 380]$
  (bounded by weight magnitudes times rvector magnitudes). The "continuous
  state" we constructed for linearisation is a projection, not the real
  state.
- *Dynamics is piecewise constant with jumps.* The threshold comparison
  $V >= tau$ is a Heaviside function, not a sigmoid. Our sigmoid
  approximation softens the jump, but the softened Jacobian has no
  relation to the true tick-by-tick evolution near the threshold.
- *No continuous symmetry breaking.* In a continuous system, a symmetric
  equilibrium with $rho > 1$ breaks symmetry under infinitesimal
  perturbation. In the integer simulator, perfectly symmetric initial
  conditions remain symmetric forever; the only way to break symmetry is
  to make the weights asymmetric *to begin with*.

#remark[
  This is consistent with Phase 0's semantic-discrepancy finding.
  Kind2's reachability semantics allow arbitrary initial perturbations,
  which makes the FCS system behave like a continuous one for
  verification purposes. Our deterministic integer simulator does not,
  and spectral predictors designed for continuous dynamics cannot bridge
  that gap.
]

= What Remains Viable

Two parts of the research programme survive the contralateral negative
result intact:

+ *FCS-Kind2 reachability semantics.* Under Phase 0's Option B
  (inject $epsilon$-perturbation into the initial state), the deterministic
  ground truth would shift to match FCS's blue/red staircase. The
  scalar-$r$ $rho(A)$ contour from Phase 1a might then align with the
  shifted boundary, because the question would no longer be "does the
  bit-exact system break symmetry" but "can a nearby perturbation reach a
  WTA state." This is an untested hypothesis; running it is cheap.

+ *Negative-loop pole placement (Hypothesis 3).* The negative loop is
  structurally different: its dynamics is *oscillatory*, not a bistable
  symmetry-breaking, and the period-4 pattern is a genuine dynamical
  property (not a combinatorial one). Pole placement via sympy on the
  $2 times 2$ characteristic polynomial of either the scalar-$r$ or
  $A_"full"$ matrix might still yield weights that realise prescribed
  oscillation periods in the simulator. Whether the linearisation is
  accurate *enough* to hit an integer-threshold period is the open
  question, and it is answerable with a few hundred lines of code.

= Decision Points for the Verifier

#decision[
  *Option A (recommended).* Accept the negative result for contralateral,
  narrow the programme to the negative loop, and run Hypothesis 3 (pole
  placement) as the final experiment. Close the project with an honest
  final report whose headline is: *spectral cartography works for
  oscillatory archetypes (negative loop) but not for bistable symmetry-
  breaking archetypes (contralateral) under bit-exact FCS semantics*.
  Cost: 1--2 days for pole placement + final writeup.
]

#decision[
  *Option B.* Before closing, re-run Phase 1 under Phase 0 Option B
  (symmetry-breaking perturbation injected into the simulator), to check
  whether the spectral predictors agree with FCS-Kind2-style ground truth.
  If they do, the final report has a positive headline for *both* the
  reachability interpretation and the oscillatory case. Cost: one day.
]

#decision[
  *Option C.* Pivot to a different FCS archetype (series, parallel,
  positive loop; Fig. 1a--c) where the dynamics is not combinatorially
  dominated. The research question "can spectral cartography replace
  exhaustive Kind2 sweeps" is then tested on a new target. Cost:
  substantial --- probably one week of reimplementing the Phase 0
  pipeline for a new topology.
]

The author's recommendation is Option A with Option B as a cheap add-on.
The negative result for contralateral under deterministic semantics is
already a publishable finding (it clarifies exactly where spectral methods
do and do not apply to discrete-threshold neural models), and the negative
loop remains an untested positive case that would round out the story.

#v(0.8cm)
#line(length: 100%)
#text(size: 9pt, style: "italic")[
  Generated from `deq/archetypes/phase1b_afull.py`. Reproduce via
  `.venv/bin/python3 archetypes/phase1b_afull.py`. All numerics in
  `deq/archetypes/results/phase1b_*`.
]
