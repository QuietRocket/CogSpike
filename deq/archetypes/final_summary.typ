// Final summary — Spectral Cartography of LI&F Archetype Parameter Spaces
// CogSpike Research Team — April 2026

#set document(
  title: "Spectral Cartography of LI&F Archetypes — Final Summary",
  author: "CogSpike Research Team",
  date: datetime.today(),
)

#set page(paper: "a4", margin: (x: 2.3cm, y: 2.3cm), numbering: "1")
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

#let headline(body) = block(
  width: 100%, inset: 10pt,
  fill: rgb("#f0f7ff"),
  stroke: (left: 3pt + rgb("#4a90d9")),
  text(weight: "semibold", body),
)

#let intuition(body) = block(
  width: 100%, inset: 8pt,
  fill: rgb("#f0f7ff"),
  stroke: (left: 2pt + rgb("#4a90d9")),
  [*Intuition:* #body],
)

#align(center)[
  #text(size: 20pt, weight: "bold")[
    Spectral Cartography of \
    LI\&F Archetype Parameter Spaces
  ]
  #v(0.4em)
  #text(size: 13pt)[Final Summary]
  #v(0.3em)
  #text(size: 11pt)[CogSpike Research Team --- April 2026]
  #v(0.3em)
  #text(size: 10pt, style: "italic")[
    A three-phase experiment on the De Maria et al. 2020 FCS contralateral
    inhibition and negative loop archetypes.
  ]
]

#v(0.8em)

#headline[
  *Central result.* Spectral cartography of FCS LI\&F archetypes splits
  cleanly into two regimes along semantic lines. Under *deterministic
  single-trajectory* semantics, winner-take-all is a combinatorial
  phenomenon decided by an integer comparison on weight magnitudes; no
  linearisation reaches into that comparison. Under *Kind2-style
  reachability* semantics, the same system is spectral: a single scalar
  -$r$ linearisation's spectral radius cleanly separates reachable from
  non-reachable parameter regions with $98.5%$ classification accuracy.
  The prior CogSpike spectral framework is therefore in-scope for
  formal-verification questions (which Kind2 answers) and out-of-scope
  for bit-exact trajectory prediction.
]

#v(0.4em)

= The Three-Panel Picture

#figure(
  image("results/final_triptych.png", width: 100%),
  caption: [Central finding visualised. *(a)* Phase 0 deterministic ground
    truth on the $40 times 40$ $(w_(1 2), w_(2 1))$ grid: blue iff a
    winner emerges within 4 ticks from zero init, $63.4%$ of cells.
    *(b)* The combinatorial predictor $||w_(1 2)|-|w_(2 1)|| > 7$
    (black contour) captures the deterministic structure at $83.4%$
    accuracy --- sign of the asymmetry perfectly predicts which neuron
    wins ($100%$ in all non-tied cells). No spectral quantity reaches
    this accuracy. *(c)* The Phase 1c reachability ground truth
    (any $epsilon$-perturbation of initial state counts) has $97.8%$
    blue cells; the scalar-$r$ spectral radius contour $rho(A) > 0.544$
    (black curve) demarcates the non-reachable region at $98.5%$
    accuracy. Perfect precision is possible at $rho < 0.544$
    ($11$ of $36$ unreachable cells detected with zero false
    positives).],
)

= What We Built

A self-contained pipeline in `deq/archetypes/` that reproduces the FCS
§6.2 Lustre LI\&F semantics verbatim (length-5 windowed integrator with
`rvector = [10, 5, 3, 2, 1]`, integer-scaled weights, reset-after-spike,
one-tick spike-emission delay), then applies spectral-cartography tooling
to the two canonical archetypes:

- *Negative loop* (Property 5): an activator/inhibitor pair that under
  constant input produces the period-4 pattern $0 1 1 0 0 1 1 dots$
  exactly. Our tuning finding: the "exact cancellation rule"
  $w_"IA" = -w_"AI"$ is the minimal analytical condition for the
  period-4 pattern; the FCS-stated $w_"IA" = -20$ overshoots into a
  period-5 regime.

- *Contralateral inhibition* (Property 7): two mutually inhibiting
  neurons with constant self-drive. We mapped two distinct ground
  truths --- the deterministic map (one trajectory per cell) and the
  reachability map (any $epsilon$-perturbation counts) --- and
  tested eight spectral predictors against both.

= Phase-by-Phase Scorecard

#table(
  columns: (auto, 1fr, 1.4fr),
  inset: 6pt, align: (left, left, left), stroke: 0.5pt,
  [*Phase*], [*Question tested*], [*Outcome*],
  [0], [Reproduce FCS semantics; generate deterministic ground truth.],
  [✓ Property 5 exact match; Fig. 10 reproduction produces a
   richer structure (central red block + asymmetry wings + corner red
   regions) than the FCS Kind2 staircase.],
  [1a], [Does $Delta(W)$ or $Delta(A)$ with scalar-$r$ linearisation
    predict WTA?],
  [✗ $Delta(W) equiv 0$ analytically for $2 times 2$ zero-diag $W$;
   $Delta(A)$ and $rho(A)$ give $64$--$69%$ classification (baseline
   $63.4%$). Negative loop $arg lambda$ never reaches $pi/2$ under
   scalar-$r$.],
  [1b], [Does the $5n$-dim $A_"full"$ linearisation fix the Phase 1a
    failures?],
  [✗ Predictors at balanced FP give $46%$ (worse than baseline);
   eigenvector asymmetry sign-matches dominance in $11%$ of cells
   (worse than random). Trivial combinatorial predictor
   $||w_(1 2)|-|w_(2 1)||>7$ achieves $83.4%$.],
  [1c], [Under Kind2-style reachability semantics ($epsilon$-perturbed
    init), do the same predictors align with the reachability oracle?],
  [✓ scalar-$r$ $rho(A)$ achieves $98.5%$ classification accuracy;
   clean separation of $rho$ distributions between reachable
   ($[0.544, 0.929]$) and non-reachable ($[0.518, 0.598]$) cells.
   First positive spectral signal.],
  [2], [Does the delayer-augmented topology (FCS Fig. 11) reproduce
    the asymmetric red-zone growth, and does $rho(A_"full")$ on the
    15-dim delayed state matrix predict it?],
  [✓ Asymmetry reproduced: $N_2$ wins $1136$ cells vs $N_1$'s $448$
   (FCS §6.3.4 "contrary to expectation" observation). Spectral
   prediction via $rho(A_"full")$ achieves $95.8%$ on reachability
   GT (matches baseline), $53%$ on deterministic (matches baseline).
   Same split as contralateral without delayer.],
  [3], [Inverse pole placement: solve for weights that realise a target
    period in the simulator.],
  [✗ Sanity check at period-4 known-good weights passes only after a
   single-point calibration; inverse design for other periods misses
   $5/6$ targets. $A_"full"$'s dominant arg is a weak discriminator
   ($r = -0.51$) of simulator period. Direct enumeration is faster
   and more accurate.],
)

= Six Specific Contributions

+ *Tuning finding (Phase 0).* The period-4 invariant for the negative
  loop is $w_"IA" = -w_"AI"$ (exact cancellation of drive by feedback).
  FCS Appendix A.3's suggested starting value $w_"IA" = -20$ overshoots.
  This invariant is not stated in the FCS paper and is the minimal
  analytical description of Property 5.

+ *Two-ground-truth reformulation (Phase 0 + 1c).* The FCS contralateral
  archetype has *two* meaningful WTA maps, not one: the deterministic
  single-trajectory map ($63.4%$ blue, asymmetry-wing structure) and
  the Kind2-style reachability map ($97.8%$ blue, corner red block).
  These answer different questions and should not be conflated.

+ *Combinatorial invariant (Phase 1b).* The deterministic contralateral
  ground truth is *not* a spectral phenomenon. The predictor
  $||w_(1 2)|-|w_(2 1)||>7$ reaches $83.4%$ classification accuracy,
  and $"sign"(|w_(1 2)|-|w_(2 1)|)$ is a perfect ($100%$) predictor of
  which neuron wins. The FCS paper does not state this invariant;
  readers can use it directly as a back-of-envelope check.

+ *Spectral-reachability correspondence (Phase 1c).* The scalar-$r$
  $rho(A)$ from the prior CogSpike note's framework aligns with
  reachability semantics: cells with $rho(A) > 0.544$ admit WTA under
  small perturbation, with perfect precision at that threshold. This
  positions the earlier framework as correct in-scope for formal
  verification.

+ *Scalar-$r$ adequacy (Phase 1a vs 1b vs 1c).* The full $5 n$-dim
  $A_"full"$ linearisation is not a uniform improvement over the
  scalar-$r$ version. For the contralateral case, scalar-$r$ is
  cleaner both as a classifier and as a distribution-separator.
  $A_"full"$ matters for negative-loop oscillation diagnostics but
  adds noise for bistable symmetry-breaking questions.

+ *Pole-placement limitation (Phase 3).* Inverse design via pole
  placement fails on FCS negative loops because the spike-reset
  nonlinearity determines the integer period, not the linearised
  dynamics. The linearisation's dominant complex arg sits in a
  narrow band ($[0.86, 1.33]$) across simulator periods $3$ through
  $8$. Direct enumeration (~1 second on a laptop) is more effective.

= Where Spectral Cartography Works and Where it Does Not

#finding[
  Spectral cartography of FCS LI\&F is *in-scope* for:
  - Kind2-style reachability: does WTA exist in the attractor
    structure? ($rho(A) > "threshold"$ on scalar-$r$ works; $98.5%$)
  - Regime classification: oscillatory vs. non-oscillatory, stable
    vs. unstable near fixed points.
]

#negresult[
  Spectral cartography is *out-of-scope* for:
  - Deterministic single-trajectory outcomes under integer threshold
    arithmetic: the tick-2 comparison is combinatorial, not spectral.
  - Inverse design of specific integer periods: the spike-reset
    nonlinearity dominates the integer period in a way no
    linearisation captures.
]

#intuition[
  The spike-reset rule takes the trajectory off any linear manifold
  every time $V$ crosses $tau$. The linearisation describes what
  happens *between* resets but not *when* resets happen. For
  qualitative questions ("is there any trajectory leading to WTA?")
  between-reset dynamics are enough; for quantitative questions
  ("which exact period?") the reset schedule is the essential
  information, and it is not encoded in the linearised spectrum.
]

= Reproducibility and Files

All experiments are deterministic and reproducible from the archived
code:

```
deq/archetypes/
  lif_fcs.py                    # FCS-accurate discrete simulator
  topologies.py                 # negative_loop, contralateral (+delayed)
  spectral.py                   # scalar-r + A_full linearisation helpers
  phase0_groundtruth.py         # Property 5 + Fig. 10 sweep
  phase1_eigengap.py            # H1 scalar-r retest
  phase1b_afull.py              # H1 A_full retest + combinatorial baseline
  phase1c_perturbed.py          # H2 preview under reachability semantics
  phase2_delayer.py             # FCS Fig. 11 delayer reproduction
  phase3_pole_placement.py      # H3 inverse design
  final_summary.py              # triptych figure generator
  results/                      # .npy / .png / .md / .pdf artifacts
```

Reproduce end-to-end:
```
cd deq
.venv/bin/python3 archetypes/phase0_groundtruth.py
.venv/bin/python3 archetypes/phase1_eigengap.py
.venv/bin/python3 archetypes/phase1b_afull.py
.venv/bin/python3 archetypes/phase1c_perturbed.py
.venv/bin/python3 archetypes/phase2_delayer.py
.venv/bin/python3 archetypes/phase3_pole_placement.py
.venv/bin/python3 archetypes/final_summary.py
```

Each phase produces a typst-formatted PDF report alongside its markdown
output. The phase 0, 1a, 1b, 1c, 3 report PDFs and this summary together
constitute the project's deliverables.

= FCS Fig. 11 Winner Asymmetry

#figure(
  image("results/phase2_winner_map.png", width: 60%),
  caption: [Phase 2 delayer-augmented winner map: $N_2$ wins ($1136$
    cells, red) dominates $N_1$ wins ($448$ cells, blue). The delayer
    on the $N_1 arrow N_2$ branch gives $N_2$ a one-tick head start
    that biases the tick-$2$/$3$ symmetry-breaking in its favour,
    even in many cells where $|w_(1 2)| > |w_(2 1)|$. This is the
    quantitative form of FCS §6.3.4's "contrary to expectation"
    observation.],
)

= Outlook

One natural extension suggests itself:

- *Analyse FCS Figs. 1a--c (series, parallel, positive loop)* where the
  dynamics is not bistable-symmetry-breaking. Spectral methods may
  simply work there without the caveats the contralateral case
  required. If so, the final positive-result footprint grows.

The primary scientific question has been answered:
spectral cartography replaces exhaustive Kind2 sweeps for reachability
questions but not for deterministic outcomes or inverse-design questions.
Phase 2 confirms this split transfers unchanged to the Fig. 11
delayer-augmented topology.

#v(0.8cm)
#line(length: 100%)
#text(size: 9pt, style: "italic")[
  Generated from `deq/archetypes/final_summary.py`. All code, data, and
  PDFs in `deq/archetypes/`.
]
