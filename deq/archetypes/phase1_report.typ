// Phase 1 Report — Eigenvalue Gap as WTA Predictor
// CogSpike / LI&F Archetypes — April 2026

#set document(
  title: "Phase 1: Eigenvalue Gap as WTA Predictor",
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

#let remark(body) = block(
  width: 100%, inset: 8pt,
  stroke: (left: 2pt + orange),
  [*Remark.* #body],
)

#let intuition(body) = block(
  width: 100%, inset: 8pt,
  fill: rgb("#f0f7ff"),
  stroke: (left: 2pt + rgb("#4a90d9")),
  [*Intuition:* #body],
)

#align(center)[
  #text(size: 18pt, weight: "bold")[
    Phase 1 --- Eigenvalue Gap \
    as WTA Predictor
  ]
  #v(0.3em)
  #text(size: 12pt)[CogSpike Research Team --- April 2026]
  #v(0.3em)
  #text(size: 10pt, style: "italic")[
    Hypothesis 1 --- a negative result, with a clean diagnosis and proposed fix
  ]
]

#v(1em)

*Abstract.* Hypothesis 1 (the eigenvalue gap $Delta = ||lambda_1| - |lambda_2||$
of the weight matrix $W$, or of the linearised state matrix $A = r I + W "diag"(f'(p^star))$,
predicts WTA on the FCS contralateral grid) is *not validated*. The failure
mode is clean and fully diagnosed: it stems from a specification error in the
working plan rather than a problem with spectral cartography itself. Raw-$W$
eigengap is provably identically zero for $2 times 2$ zero-diagonal inhibitory
matrices; the scalar-$r$ linearisation of $A$ saturates because the FCS
neuron's 5-tap windowed integrator cannot be collapsed to a single leak
coefficient. The path forward is to replace the scalar-$r$ linearisation with
the full $5n$-dimensional state matrix that preserves the FIR-filter structure
before testing Hypothesis 2. Negative-loop results reinforce the same
diagnosis: the simulator's period-4 oscillation (Property 5) cannot be
captured under scalar-$r$, which predicts $arg lambda_"dom"$ no larger than
$0.47$ rad (target $pi/2 approx 1.57$).

= Setup and Predictors

Against the Phase 0 ground truth (`fcs_fig10_groundtruth.npy` binary, plus
`fcs_fig10_dominance.npy` continuous) we compute four candidate spectral
predictors per grid cell on the same $40 times 40$ weight grid:

- $Delta(W)$ --- raw-weight eigengap $||lambda_1|-|lambda_2||$
- $Delta(A)$ --- linearised-state eigengap
- $rho(A)$ --- spectral radius (Phase 2 preview)
- $arg lambda_"dom"(A)$ --- argument of the dominant eigenvalue (negative-loop sweep)

Sigmoid firing-rate approximation uses $k = 0.08$ and $p_"mid" = 30$ after
the fallback sweep specified in the working plan (at $p_"mid" = 90$ the
operating point saturates; at $p_"mid" in {60, 75}$ it is still degenerate;
$p_"mid" = 30$ is the smallest value that places the operating point in the
steep sigmoid region). Scalar leak $r = 0.5$ is the plan default; the
least-squares fit of `rvector = [10, 5, 3, 2, 1]` to $r^e dot 10$ gives
$r approx 0.545$, but this difference is not the cause of the result below
and is reported only as a secondary measurement.

= Hypothesis 1 on Contralateral Inhibition

== Raw-$W$ Eigengap: Analytically Zero

#finding[
  For any $2 times 2$ matrix $W = mat(0, a; b, 0)$ with zero diagonal, the
  eigenvalues are $plus.minus sqrt(a b)$ with *equal magnitudes* regardless
  of $|a|$ vs $|b|$. So $Delta(W) equiv 0$ over the entire contralateral
  sweep --- verified numerically to machine epsilon ($max = 1.4 times 10^(-14)$).
]

The working plan's claim (Phase 1 spec, Appendix A.4) that $Delta = 0$ iff
$|a| = |b|$ is simply wrong for this matrix family. The raw-$W$ eigengap
cannot distinguish the blue-wing asymmetry structure of the ground truth
because the underlying spectrum has no asymmetry to distinguish.

== Linearised $A$ Predictors: Saturated, Weak Signal

#figure(
  image("results/phase1_contra_rhoA.png", width: 95%),
  caption: [$rho(A)$ heatmap (scalar-$r$ linearisation, $p_"mid" = 30$). Blue
    WTA cells from Phase 0 shown as black dots. The $rho = 1$ contour
    specified for Phase 2 does not exist: $rho(A) in [0.52, 0.93]$ across
    the whole grid --- the scalar-$r$ linearisation never reaches the
    bifurcation boundary.],
)

#figure(
  image("results/phase1_scatter_rhoA_absdom.png", width: 75%),
  caption: [$rho(A)$ against $|n_1 - n_2|/(n_1 + n_2 + 1)$. The continuous
    dominance signal is bimodal (either $approx 0$ tied or $approx 0.98$
    captured), and $rho(A)$ does not discriminate between the two bands:
    both modes span $rho in [0.55, 0.90]$.],
)

Best classification accuracy over all (polarity, threshold) choices:

#table(
  columns: (auto, auto, auto, auto),
  inset: 6pt,
  align: center,
  stroke: 0.5pt,
  [*Predictor*], [*Best binary accuracy*], [*Pearson $r$ vs $|"dom"|$*], [*Spearman $rho$ vs $|"dom"|$*],
  [$Delta(W)$], [63.4% (majority baseline)], [--], [--],
  [$Delta(A)$], [68.9%], [see scatter], [see scatter],
  [$rho(A)$], [68.9%], [see scatter], [see scatter],
)

The baseline (predict all cells as blue) is 63.4%; the two linearised-state
predictors improve over that by only $5.5$ percentage points. That is
"signal, but weak" --- consistent with $rho(A)$ correlating loosely with
$sqrt(|w_(1 2) w_(2 1)|)$, but too symmetric in $(w_(1 2), w_(2 1))$ to
distinguish the two blue wings.

#figure(
  image("results/phase1_contra_gapA.png", width: 95%),
  caption: [$Delta(A)$ heatmap. The ridge from top-right to bottom-left
    roughly follows $sqrt(|w_(1 2) w_(2 1)|)$, peaking where both weights
    are large. The blue-wing structure of the ground truth (which lies on
    the *off-diagonal*) does not align with this ridge.],
)

#negresult[
  Under scalar-$r$ linearisation, neither $Delta(A)$ nor $rho(A)$ is sensitive
  to the asymmetry between $w_(1 2)$ and $w_(2 1)$ that drives WTA. Both
  quantities are functions of the eigenvalue magnitudes of $A$, which for
  the $2 times 2$ zero-diagonal structure are $|r plus.minus g sqrt(w_(1 2) w_(2 1))|$
  --- symmetric in the weight product, blind to the ratio.
]

= Hypothesis 1 on the Negative Loop

We fix $w_"XA" = 11$ and sweep $(w_"AI", w_"IA") in [1, 20] times [-1, -20]$
on an integer grid, comparing simulator-derived period-4 output
(autocorrelation on the last 20 ticks) to $arg lambda_"dom"(A)$ from the
same scalar-$r$ linearisation.

#figure(
  image("results/phase1_negloop.png", width: 95%),
  caption: [Negative loop: $arg lambda_"dom"(A)$ heatmap (dark = small arg).
    Red dots mark cells where the simulator produces an exact period-4
    $1100$ pattern. The $arg = pi/2$ target contour does not appear ---
    $arg$ ranges only over $[0.021, 0.467]$ rad across the sweep.],
)

Of the 400 cells swept, 16 produce the exact period-4 pattern in the
simulator. The eigenvalue-argument predictor places $arg lambda_"dom"$ in
a narrow band near zero (all predicted periods much greater than 4 ticks). The scalar-$r$
linearisation does not see the FCS neuron's length-5 FIR filter and therefore
cannot produce the high-frequency pole locations that the discrete-time
system actually realises.

#remark[
  Phase 0's tuning finding ($w_"IA" = -w_"AI"$ exactly cancels the drive and
  reproduces Property 5) is visible in the simulator ground truth: all 16
  period-4 cells lie in a narrow vertical stripe at $w_"AI" in {9, 10}$
  with $w_"IA" in [-20, -13]$. The near-cancellation invariant is real;
  it simply is not captured by the scalar-$r$ linearisation.
]

= Root Cause: Scalar-$r$ is Too Crude

Every failure mode above has a single common cause. The FCS neuron's
"potential" is not a scalar --- it is the 5-tap windowed integrator
$V(t) = sum_(e=0)^4 "rvector"[e] dot "mem"[e](t)$. The true state of an
$n$-neuron network has $5n$ components, not $n$. Collapsing the FIR filter
to a scalar leak discards four of the five poles of the single-neuron
transfer function --- including the ones responsible for period-4
oscillation and for the sensitivity of $rho$ to weight asymmetry.

#intuition[
  The scalar-$r$ model answers "how does $"mem"[0]$ decay if we stop stimulating?"
  The true model answers "how does the full length-5 memory window evolve under
  the shift-and-reset-on-spike rule?" The second system can oscillate at any
  frequency the FIR filter supports; the first can only do geometric decay.
  Hypothesis 1 was tested with a linearisation that cannot represent the
  phenomenon it was asked to predict.
]

= Proposed Fix: Full $5n$-Dimensional State Matrix

The corrected linearisation is straightforward. For each neuron $i$, the
state vector is $("mem"_i [0], "mem"_i [1], ..., "mem"_i [4])$. At the non-spiking
operating point $p^star$:

$ "mem"_i [0](t+1) &= sum_j W_(i j) dot f'(V_j^star) dot sum_(e=0)^4 r_e dot "mem"_j [e](t) + "const" \
  "mem"_i [k](t+1) &= "mem"_i [k-1](t) quad "for" k = 1 .. 4 $

where $r_e = "rvector"[e]$.

This gives a $5n times 5n$ block matrix $A_"full"$ with the top row of each
block implementing the weighted FIR-sum coupling and the lower rows
implementing the shift. Hypothesis 1 and Hypothesis 2 should be retested
against this matrix before any Phase 2 / Phase 3 work. Approximately 50
lines of code, one-time investment.

Under $A_"full"$ we expect:

+ $rho(A_"full") = 1$ contour to actually exist in the contralateral sweep,
  tracking the boundary of the synchronous-rhythm red block.
+ Complex-conjugate eigenvalue pairs of $A_"full"$ for the negative loop
  with $arg lambda$ capable of reaching $pi/2$ --- predicting Property 5's
  period 4 and Phase 3's inverse-design targets.
+ Modal eigenvectors that are genuinely asymmetric between $N_1$ and $N_2$
  when the weights are asymmetric, restoring the ability to distinguish the
  two blue wings.

= Files Generated

- `deq/archetypes/spectral.py` --- scalar-$r$ linearisation helpers
  (quarantined; will be extended, not replaced, by the $5n$ version)
- `deq/archetypes/phase1_eigengap.py` --- the sweep orchestrator
- `deq/archetypes/results/phase1_contra_gapW.png`, `phase1_contra_gapA.png`,
  `phase1_contra_rhoA.png`, `phase1_contra_gmean.png`,
  `phase1_scatter_gapA.png`, `phase1_scatter_rhoA_absdom.png`,
  `phase1_negloop.png`
- `deq/archetypes/results/phase1_report.md` --- markdown version of this note

= Decision Point for the Verifier

#decision[
  *Option A (recommended).* Approve building the full $5n$-dimensional
  state matrix $A_"full"$ before proceeding to Phase 2. This is a clean
  technical pivot: the research question is unchanged, only the linearisation
  is upgraded to one that can represent the FCS dynamics. Retest Hypotheses
  1 and 2 against $A_"full"$; if they validate there, Phase 3 inverse design
  follows as planned. *Cost:* ~50 lines in `spectral.py` plus re-running
  Phase 1 sweep; half a day.
]

#decision[
  *Option B.* Declare Hypothesis 1 a closed negative result and proceed
  directly to a Phase-2-style $rho(A_"full") = 1$ test, skipping the
  eigenvalue-gap formulation entirely. The gap framing is arguably the
  weakest of the three hypotheses (it assumes a magnitude asymmetry that
  $2 times 2$ contralateral $W$ structurally cannot produce), so cutting
  it simplifies the narrative. *Cost:* same ~50 lines, but the report
  structure collapses Phase 1 into Phase 2.
]

#decision[
  *Option C.* Abandon the contralateral archetype as unsuited to spectral
  cartography and restrict the remaining work to the negative loop, where
  Hypothesis 3 (pole placement for target period) has clearer theoretical
  grounding. Honest but narrower scope; cedes the Fig. 10 comparison.
]

The author's recommendation is Option A. The scalar-$r$ failure is
instructive but not structural --- it is a correctable specification error,
and the $A_"full"$ replacement is the obvious right object to study. The
Phase 0 ground-truth map still stands (it is the real dynamics, not an
artefact of the linearisation), and the bifurcation question remains
scientifically open.

#v(1cm)
#line(length: 100%)
#text(size: 9pt, style: "italic")[
  Generated from `deq/archetypes/phase1_eigengap.py`. Reproduce via
  `.venv/bin/python3 archetypes/phase1_eigengap.py` from the `deq/` directory.
  Phase 0 artifacts in `deq/archetypes/results/phase0_*`; Phase 1 artifacts
  in `deq/archetypes/results/phase1_*`.
]
