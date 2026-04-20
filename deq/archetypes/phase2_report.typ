// Phase 2 Report — FCS Fig. 11 delayer reproduction
// CogSpike / LI&F Archetypes — April 2026

#set document(
  title: "Phase 2: FCS Fig. 11 Delayer Reproduction",
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
    Phase 2 --- FCS Fig. 11 \
    Delayer Reproduction
  ]
  #v(0.3em)
  #text(size: 12pt)[CogSpike Research Team --- April 2026]
  #v(0.3em)
  #text(size: 10pt, style: "italic")[
    Asymmetric red-zone growth reproduced; spectral
    prediction tracks reachability, not deterministic winner.
  ]
]

#v(1em)

*Abstract.* Inserting a unit-gain delayer on the $N_1 arrow N_2$
inhibitory branch reproduces the FCS Fig. 11 asymmetric red-zone
growth exactly. In our deterministic sweep, $N_2$ wins in $1136$ cells
versus $N_1$'s $448$ --- a $688$-cell asymmetry (versus zero in the
undelayed Fig. 10 sweep). This is the quantitative form of FCS §6.3.4's
"contrary to expectation" observation. Spectral prediction via
$rho(A_"full")$ on the $15$-dim delayed state matrix reproduces the
Phase 1b/1c pattern: $53%$ classification against the deterministic
ground truth (baseline), $96%$ against reachability (also near
baseline). The delayer does not change the core finding that spectral
cartography tracks reachability rather than deterministic bit-exact
outcomes, but it does demonstrate that the spectral pipeline transfers
cleanly to a new topology (same code paths work on the $3$-neuron
system).

= Topology and Ground Truth

The delayed contralateral topology inserts a delayer between $N_1$'s
spike output and $N_2$'s inhibitory input: $N_1 arrow.r.long_(w=+11)$
delayer $arrow.r.long_(w=w_(1 2))$ $N_2$. Meanwhile $N_2 arrow.r.long_(w=w_(2 1))$
$N_1$ remains direct. The swept weight $w_(1 2)$ thus lives on the
delayer-to-$N_2$ edge; the $N_1 arrow$ delayer edge is fixed at the
unit-gain buffer weight $11$.

#figure(
  image("results/phase2_det_comparison.png", width: 100%),
  caption: [Deterministic ground truth: undelayed (Phase 0, left) vs
    delayed (Phase 2, right). The delayed version has $850/1600$
    ($53.1%$) blue cells, down from the undelayed $1014$. More striking
    is the *loss of symmetry*: the delayed map is no longer invariant
    under $(w_(1 2), w_(2 1))$ swap.],
)

#figure(
  image("results/phase2_winner_map.png", width: 70%),
  caption: [Which neuron wins in the delayed topology. Red = $N_2$ wins
    ($1136$ cells); blue = $N_1$ wins ($448$ cells); white strip = tied
    ($16$ cells). The $N_2$-wins region dominates, including large
    portions where $|w_(1 2)| > |w_(2 1)|$ (i.e., $N_1$'s inhibition on
    $N_2$ is nominally *stronger* but delayed, and still loses).],
)

= The FCS Fig. 11 Asymmetry

#finding[
  *FCS §6.3.4 observation reproduced.* In the delayed topology, $N_2$
  wins the WTA competition in $1136/1600$ cells versus only $448/1600$
  for $N_1$ --- a $688$-cell asymmetry. The intuitive but wrong
  prediction ("$N_2$ loses because its inhibition on $N_1$ is reached
  without delay, while $N_1$'s is late") is contradicted in our
  deterministic simulator: $N_2$ wins *more* often, exactly as FCS
  reports.
]

#intuition[
  The delayer gives $N_2$ a one-tick head start. Under integer
  semantics, both neurons receive external drive at tick $1$ and fire
  at tick $2$. $N_2$'s spike propagates to $N_1$ in a single tick
  (direct edge); $N_1$'s spike propagates via delayer-buffer and
  reaches $N_2$ at tick $3$. So at tick $3$ $N_1$ is already partially
  suppressed by $N_2$ while $N_2$ is not yet suppressed. This
  tick-$3$ asymmetry propagates forward, and $N_2$ typically escapes
  to its saturated firing attractor first.
]

The asymmetry depends on the specific tick-$2$/tick-$3$ decision logic
--- the same combinatorial mechanism that Phase 1b identified for the
undelayed case. The delayer simply adds a new constant (a one-tick
latency on one branch) to the same integer comparison.

= Reachability Ground Truth

Under $epsilon$-perturbation of initial state (same perturbation set as
Phase 1c, extended to the 3-neuron system), the delayed ground truth is
$1534/1600$ ($95.9%$) blue --- nearly identical to the undelayed
$97.8%$. The delayer shrinks the non-reachable region only slightly.

#figure(
  image("results/phase2_reach_comparison.png", width: 100%),
  caption: [Reachability ground truth: undelayed (left, $97.8%$ blue)
    vs delayed (right, $95.9%$ blue). The small non-reachable corner
    persists in both --- weak-inhibition cells where no amount of
    initial perturbation can break the tie.],
)

= Spectral Prediction on the 15-dim Delayed State

We build $A_"full"$ on the $3 times 5 = 15$-dimensional state
(including the delayer's memory buffer) and test its spectral radius as
a predictor.

#figure(
  image("results/phase2_rho_reach.png", width: 95%),
  caption: [$rho(A_"full")$ heatmap on the delayed 15-dim linearisation
    with reachability-GT overlay. $rho$ ranges $[0.005, 3.568]$;
    classification accuracy is $95.8%$, near the $95.9%$ majority
    baseline for reachability.],
)

#table(
  columns: (auto, auto, auto),
  inset: 6pt, align: center, stroke: 0.5pt,
  [*Predictor*], [*vs Deterministic GT*], [*vs Reachability GT*],
  [$rho(A_"full")$ (15-dim)], [53.1%], [95.8%],
  [Majority baseline], [53.1%], [95.9%],
)

#remark[
  The spectral classifier on the delayed 15-dim system performs
  identically to the majority baseline: it cannot improve over "predict
  most cells as blue". This mirrors the Phase 1b undelayed finding, and
  confirms that the delayer does not change the fundamental issue ---
  spectral methods see "some trajectory reaches WTA" but not "the
  specific deterministic trajectory reaches WTA".
]

= What Worked, What Did Not

- *Topology reproduction* (✓): The delayer-augmented simulator
  produces the FCS Fig. 11 asymmetric winner structure, with $N_2$
  preferred by a $688$-cell margin. The exact-cancellation logic and
  integer threshold machinery from Phase 0 carries forward unchanged.

- *Spectral prediction* (≈): $rho(A_"full")$ on the 15-dim delayed
  state matrix classifies the reachability GT at $95.8%$ --- but that
  matches the $95.9%$ baseline. No meaningful signal, though no
  degradation either.

- *Eigenvector asymmetry for winner direction* (not tested here):
  Could in principle predict $N_2$ vs $N_1$ preference via
  eigenvector-mass on neuron indices; left for future work given the
  Phase 1b result that this predictor underperformed even on the
  simpler $2$-neuron system.

= Consolidation

Phase 2 completes the scorecard and confirms the three-way split from
the final summary:

+ Deterministic contralateral (with or without delayer) is
  *combinatorial*: the tick-$2$/$3$ integer comparison determines the
  outcome, and spectral methods have no access to that comparison.

+ Reachability contralateral (with or without delayer) is *spectral
  under scalar-$r$* with ρ(A) classifier: $98.5%$ undelayed, $95.8%$
  delayed (both near their respective majority baselines, but with
  clean $rho$-distribution separation in the undelayed case).

+ The delayer-induced winner asymmetry (FCS Fig. 11) is itself
  combinatorial, arising from the extra-tick-of-latency constant
  entering the same tick-$2$/$3$ comparison.

The headline result from the project does not change. Phase 2 adds a
third data point consistent with the Phase 1b/1c diagnosis and
confirms that the spectral/combinatorial split is a property of the
archetype class, not of any specific topology within it.

#v(0.8cm)
#line(length: 100%)
#text(size: 9pt, style: "italic")[
  Generated from `deq/archetypes/phase2_delayer.py`. Artifacts in
  `deq/archetypes/results/phase2_*` and `fcs_fig11_*`.
]
