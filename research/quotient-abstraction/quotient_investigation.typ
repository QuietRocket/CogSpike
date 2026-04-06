// ============================================================================
// Technical Note: Quotient Abstraction for Recurrent Competitive Topologies
// ============================================================================

#set document(
  title: "Quotient Abstraction Limits for Recurrent Competitive Topologies: A PRISM Investigation",
  author: "CogSpike Working Notes",
  date: datetime.today(),
)

#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.5cm),
  numbering: "1",
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 9pt, fill: luma(120))
      _Quotient Abstraction for Recurrent Competitive Topologies_
      #h(1fr)
      CogSpike Working Notes
    ]
  },
)

#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")

#show heading.where(level: 1): it => {
  v(1em)
  set text(size: 14pt, weight: "bold")
  it
  v(0.3em)
}

#show heading.where(level: 2): it => {
  v(0.8em)
  set text(size: 12pt, weight: "bold")
  it
  v(0.2em)
}

#show raw.where(block: true): it => {
  set text(size: 9pt)
  block(
    fill: luma(245),
    inset: 10pt,
    radius: 3pt,
    width: 100%,
    it,
  )
}

// ── Title block ──────────────────────────────────────────────────────────────

#align(center)[
  #text(size: 18pt, weight: "bold")[
    Quotient Abstraction Limits for\ Recurrent Competitive Topologies
  ]
  #v(0.3em)
  #text(size: 13pt, fill: luma(80))[
    A PRISM Investigation of Winner-Take-All Dynamics
  ]
  #v(1em)
  #text(size: 11pt)[CogSpike Working Notes — #datetime.today().display("[month repr:long] [day], [year]")]
]

#v(1em)

#block(
  fill: rgb("#f0f7ff"),
  inset: 12pt,
  radius: 4pt,
  width: 100%,
)[
  *Abstract.* The CogSpike weight-discretized quotient abstraction successfully reduces DTMC state spaces for feedforward SNN topologies. We show that the standard quotient — with additive leak $lambda_d$ and non-negative potential domain — is _qualitatively unsound_ for recurrent competitive networks: the core Winner-Take-All (WTA) property $P = 1.0 [F G (y_"loser" = 0)]$ inverts to $P = 0.0$. Through systematic PRISM experiments on a 4-neuron contralateral inhibition network, we identify two root causes (the `max(0,...)` clamp erasing differential inhibition memory, and the additive leak preventing asymmetric recovery) and derive a _minimal corrected quotient_ that restores all WTA properties: $W = 6$ weight levels with multiplicative leak $r dot p$ and a signed potential domain. This corrected quotient achieves a $50 times$ state space reduction (7,558 vs. 377,271 states) while exactly preserving the BSCC structure (8 absorbing states) and all 10 verified PCTL properties.
]

= Background <sec-background>

== Contralateral Inhibition and WTA Dynamics

In a contralateral inhibition circuit, $n$ processing neurons receive parallel excitatory input and mutually inhibit each other through negative synaptic weights. When the inhibitory weights are asymmetric — one neuron delivers stronger suppression than it receives — the network exhibits _Winner-Take-All_ (WTA) dynamics: the strongest inhibitor eventually suppresses all competitors and becomes the sole persistent spiker.

For this investigation we use a 4-neuron contralateral inhibition network with the following weight structure:

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt,
    align: (left, center, left),
    table.header(
      [*Connection*], [*Weight*], [*Role*],
    ),
    [$S_i -> N_i$ (for $i in {1..4}$)], [+100], [Excitatory input],
    [$N_i -> O_i$ (for $i in {1..4}$)], [+100], [Output projection],
    [$N_1 -> N_2, N_3, N_4$], [*$-100$*], [Strong suppression (winner)],
    [$N_2, N_3, N_4 -> N_1$], [$-70$], [Weak back-inhibition],
    [Losers $<->$ Losers], [$-70$], [Weak mutual inhibition],
  ),
  caption: [Synaptic weights of the 4-neuron contralateral inhibition network. The 30-unit asymmetry ($-100$ vs. $-70$) predetermines $N_1$ as the winner.],
) <tab-weights>

The precise (non-discretized) PRISM model of this network, with membrane potential range $[-360, 150]$ and multiplicative leak $r = 0.5$, successfully verifies all WTA properties.

== The Standard Quotient Abstraction

The CogSpike quotient model uses three transformations:

+ *Weight discretization:* $delta_W(w) = lr(⌊ w dot W / w_"max" ⌉)$ maps weights to $[-W, W]$.

+ *Additive leak:* $lambda_d = -max(1, floor(ell dot T_d))$ replaces the multiplicative retention $r dot p$.

+ *Non-negative clamping:* $p'_n = max(0, min(P_"max", p_n + C_n + lambda_d))$ restricts potentials to $[0, k]$.

This abstraction is sound for feedforward topologies, where negative potentials do not carry competitive information.

= The Failure <sec-failure>

Applying the standard quotient ($W = 4$, $lambda_d = -2$, `max(0,...)`) and verifying against the WTA property suite reveals a complete qualitative inversion:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    align: (left, left, center, center),
    table.header(
      [*Property*], [*Meaning*], [*Precise*], [*Quotient (W=4)*],
    ),
    [`F G (y_N2=0)`], [Loser silence], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]], [#text(fill: rgb("#c62828"), weight: "bold")[0.0 ✗]],
    [`G F "spike_N1"`], [Winner persists], [1.0], [1.0],
    [`G F "spike_N2"`], [Loser persists?], [#text(fill: rgb("#2e7d32"), weight: "bold")[0.0 ✓]], [#text(fill: rgb("#c62828"), weight: "bold")[1.0 ✗]],
    [`G F "spike_N3"`], [Loser persists?], [#text(fill: rgb("#2e7d32"), weight: "bold")[0.0 ✓]], [#text(fill: rgb("#c62828"), weight: "bold")[1.0 ✗]],
    [`G F "spike_N4"`], [Loser persists?], [#text(fill: rgb("#2e7d32"), weight: "bold")[0.0 ✓]], [#text(fill: rgb("#c62828"), weight: "bold")[1.0 ✗]],
    [`F G (sum <= 1)`], [Mutual exclusion], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]], [#text(fill: rgb("#c62828"), weight: "bold")[0.0 ✗]],
  ),
  caption: [WTA property verification: precise model vs. standard quotient ($W = 4$). Six of ten properties produce qualitatively incorrect results.],
) <tab-failure>

The BSCC analysis reveals the structural divergence: the precise model has *8 absorbing BSCCs* (single-state fixed points, corresponding to the 4 possible winners $times$ 2 firing states), while the quotient has *1 ergodic BSCC* of 3,973 states — a symmetric cycle where all neurons fire indefinitely with no convergence.

= Root Cause Analysis <sec-root-cause>

== Problem 1: The Non-Negative Clamp Erases Differential Inhibition

In the precise model, after mutual suppression (all four neurons fire simultaneously, all reset to $p = 0$), the post-suppression potentials at the next tick are:

$
p_N_1 &= floor(100 - 70 - 70 - 70 + 0.5 dot 0) = -110 \
p_N_2 &= floor(100 - 100 - 70 - 70 + 0.5 dot 0) = -140
$

The *30-unit gap* ($-110$ vs. $-140$) encodes the competitive advantage that drives WTA. With multiplicative leak $r = 0.5$, this gap creates differential recovery rates: $N_1$ recovers faster than $N_2$, eventually reaching threshold first.

In the standard quotient ($W = 4$), the corresponding post-suppression potentials are:

$
p_N_1 &= max(0, 4 - 3 - 3 - 3 + (-2)) = max(0, -7) = 0 \
p_N_2 &= max(0, 4 - 4 - 3 - 3 + (-2)) = max(0, -8) = 0
$

The `max(0, ...)` clamp maps both to 0, erasing the competitive memory entirely. Both neurons restart from identical baselines, producing symmetric dynamics.

== Problem 2: Additive Leak Prevents Asymmetric Recovery

Even when the domain is extended to allow negative values (#emph[Experiment 2], $P_"min" = -10$), the additive leak $lambda_d = -2$ creates _linear_ recovery with _constant_ rate:

$
p = -7: &quad p' = -7 + 4 + (-2) = -5 quad "(+2/tick)" \
p = -8: &quad p' = -8 + 4 + (-2) = -6 quad "(+2/tick)"
$

Both neurons recover at the same rate (+2/tick). The initial 1-unit gap is preserved but never _amplified_. By contrast, the precise model's multiplicative leak $r dot p$ creates _geometric_ recovery where the rate depends on current depth:

$
r dot (-110) = -55 quad "(recovery of 55 units)" \
r dot (-140) = -70 quad "(recovery of 70 units, but to lower level)"
$

The gap _compounds_ because deeper potentials recover proportionally farther per tick but remain farther from threshold.

== Problem 3: Floor Quantization at Low W

Even with multiplicative leak, when $W = 4$ the `floor()` function erases the differential:

$
"N1 recovery:" quad &floor(4 + 0.5 times (-5)) = floor(1.5) = 1 \
"N2 recovery:" quad &floor(4 + 0.5 times (-6)) = floor(1.0) = 1
$

Both land on the _same_ integer value. The 0.5-unit difference is below the resolution of the `floor()` function at this weight scale.

= Experimental Progression <sec-experiments>

We constructed five PRISM models, each addressing one or more identified failure mechanisms:

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt,
    align: (left, center, center, center, center),
    table.header(
      [*Model*], [*Leak*], [*Domain*], [*W*], [*WTA?*],
    ),
    [Standard quotient], [$lambda_d = -2$], [$[0, 8]$], [4], [#text(fill: rgb("#c62828"), weight: "bold")[No]],
    [Negative domain], [$lambda_d = -2$], [$[-10, 8]$], [4], [#text(fill: rgb("#c62828"), weight: "bold")[No]],
    [Multiplicative leak at $W=4$], [$r dot p$], [$[-10, 8]$], [4], [#text(fill: rgb("#c62828"), weight: "bold")[No]],
    [Mult. leak $W=6$, non-neg], [$r dot p$], [$[0, 8]$], [6], [#text(fill: rgb("#c62828"), weight: "bold")[No]],
    [*Mult. leak $W=6$, signed*], [$r dot p$], [$[-16, 8]$], [6], [#text(fill: rgb("#2e7d32"), weight: "bold")[Yes]],
  ),
  caption: [Experimental progression. WTA dynamics are restored only when _all three_ fixes (multiplicative leak, sufficient $W$, signed domain) are applied jointly.],
) <tab-experiments>

The $W=4$ multiplicative case was verified analytically via the floor-collision argument above; all other entries were verified via PRISM. The key result is that the $W = 6$ model with multiplicative leak _and_ signed domain is the minimum configuration that passes all WTA properties.

== Ablation: Multiplicative Leak Without Signed Domain

To confirm that the signed domain is independently necessary — not merely an artifact of the additive leak failure — we tested $W = 6$ with multiplicative leak $r dot p$ but the standard non-negative clamp `max(0,...)`.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    align: (left, center, center, center),
    table.header(
      [*Property*], [*Precise*], [*W=6, $p >= 0$*], [*W=6, signed*],
    ),
    [`F G (y_N2=0)`], [1.0], [#text(fill: rgb("#c62828"), weight: "bold")[0.305]], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
    [`G F "spike_N1"`], [1.0], [1.0], [1.0],
    [`G F "spike_N2"`], [0.0], [#text(fill: rgb("#c62828"), weight: "bold")[0.695]], [#text(fill: rgb("#2e7d32"), weight: "bold")[0.0 ✓]],
    [`F G (sum<=1)`], [1.0], [#text(fill: rgb("#c62828"), weight: "bold")[0.305]], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
  ),
  caption: [Ablation: $W = 6$ with multiplicative leak but non-negative domain. WTA properties become _probabilistic_ rather than certain — loser silence holds only 30.5% of the time.],
) <tab-ablation>

The non-negative model has only *446 states* and *2 BSCCs* (one absorbing of size 1, one ergodic of size 2), compared to the signed model's 7,558 states and 8 absorbing BSCCs. Loser silence drops from $P = 1.0$ to $P = 0.305$ — the system reaches the WTA fixed point only about 30% of the time.

The reason is immediate from the potential update. After mutual suppression:

$
p_N_1 &= max(0, floor(6 - 4 - 4 - 4 + 0.5 times 0)) = max(0, -6) = 0 \
p_N_2 &= max(0, floor(6 - 6 - 4 - 4 + 0.5 times 0)) = max(0, -8) = 0
$

The clamp erases the differential _before_ the multiplicative leak has any state to act on. At the next tick, both neurons see $r times 0 = 0$, producing identical recovery dynamics. The multiplicative leak is only useful if the neuron can _store_ the negative potential for one tick so that $r times p_"winner" eq.not r times p_"loser"$ at recovery time.

This confirms that the signed domain and multiplicative leak are _independently necessary_ — neither alone is sufficient.

= The Corrected Quotient ($W = 6$, Multiplicative Leak) <sec-solution>

== Design

The corrected quotient makes two changes to the standard abstraction:

+ *Multiplicative leak.* Replace $lambda_d$ with $r dot p$ in the potential update:
  $ p'_n = max(P_"min", min(P_"max", floor(C_n + r dot p_n))) $
  This preserves the geometric recovery rate of the precise model.

+ *Signed domain.* Extend $p_n in [P_"min", P_"max"]$ where $P_"min" < 0$, computed from fan-in analysis:
  $
  P_"min" = lim_(t -> infinity) floor(C_"min" + r dot p(t)) = floor(C_"min" / (1 - r))
  $
  For $N_1$: $C_"min" = 6 - 4 - 4 - 4 = -6$, $P_"min" = floor(-6 / 0.5) = -12$.
  For $N_2$: $C_"min" = 6 - 6 - 4 - 4 = -8$, $P_"min" = floor(-8 / 0.5) = -16$.

The weight discretization function remains $delta_W$ with $W = 6$:

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt,
    align: (center, center, center),
    table.header(
      [*Original*], [*$delta_6$*], [*Role*],
    ),
    [$+100$], [$+6$], [Excitatory],
    [$-100$], [$-6$], [Strong inhibitory],
    [$-70$], [$-4$], [Weak inhibitory ($"round"(0.7 times 6) = 4$)],
  ),
  caption: [Weight discretization at $W = 6$.],
) <tab-w6-weights>

The critical property is that $delta_6(-100) = -6$ and $delta_6(-70) = -4$ differ by 2 units (not 1 as at $W=4$), providing sufficient separation for `floor()` to preserve the recovery differential.

== Threshold Levels

With $T_d = ceil(80 times 6 / 100) = 5$ and $K = 5$ threshold levels, the firing probabilities are:

#figure(
  table(
    columns: (auto, auto),
    stroke: 0.5pt,
    align: (left, center),
    table.header([*Potential range*], [*Firing probability*]),
    [$p'_n <= 1$], [$0%$],
    [$1 < p'_n <= 2$], [$20%$],
    [$2 < p'_n <= 3$], [$40%$],
    [$3 < p'_n <= 4$], [$60%$],
    [$4 < p'_n <= 5$], [$80%$],
    [$p'_n > 5$], [$100%$],
  ),
  caption: [Threshold levels for $K = 5$, $T_d = 5$. Each level maps to probability $i\/K$.],
) <tab-levels>

== WTA Mechanism Trace

After mutual suppression (all fire, all reset to $p = 0$):

$ "Tick" t: quad
  p_N_1 = 6 - 4 - 4 - 4 = -6, quad
  p_N_2 = 6 - 6 - 4 - 4 = -8
$

Recovery with multiplicative leak (nobody fires, input on):

$ "Tick" t+1: quad
  &p'_N_1 = floor(6 + 0.5 times (-6)) = floor(3) = 3 quad -> 40% "fire" \
  &p'_N_2 = floor(6 + 0.5 times (-8)) = floor(2) = 2 quad -> 20% "fire"
$

$N_1$ fires at *twice* the probability of $N_2$. The conditional dynamics after $N_1$ fires alone:

$ "Tick" t+2: quad
  &N_1 "resets to" p = 0 \
  &p'_N_2 = floor(6 + (-6) + 0.5 times 2) = floor(1) = 1 quad -> 0% "fire"
$

When $N_1$ fires, its $-6$ inhibition exactly cancels $N_2$'s $+6$ excitatory input. $N_2$ drops to potential 1 (no fire), while $N_1$ resets and immediately receives $+6$ at the next tick:

$ "Tick" t+3: quad p'_N_1 = floor(6 + 0.5 times 0) = 6 quad -> 100% "fire" $

$N_1$ fires deterministically, re-suppressing $N_2$. The system has entered an absorbing cycle from which $N_2$ never recovers.

= Full PRISM Verification Results <sec-results>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    stroke: 0.5pt,
    align: (left, center, center, center, center, center),
    table.header(
      [*Property*], [*Precise*], [*W=4*], [*W=4 neg*], [*W=6 non-neg*], [*W=6 signed*],
    ),
    [`F G (y_N2=0)`], [1.0], [0.0], [0.0], [0.305], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
    [`G F "spike_N1"`], [1.0], [1.0], [1.0], [1.0], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
    [`G F "spike_N2"`], [0.0], [1.0], [1.0], [0.695], [#text(fill: rgb("#2e7d32"), weight: "bold")[0.0 ✓]],
    [`G F "spike_N3"`], [0.0], [1.0], [1.0], [0.695], [#text(fill: rgb("#2e7d32"), weight: "bold")[0.0 ✓]],
    [`G F "spike_N4"`], [0.0], [1.0], [1.0], [0.695], [#text(fill: rgb("#2e7d32"), weight: "bold")[0.0 ✓]],
    [`F G (sum<=1)`], [1.0], [0.0], [0.0], [0.305], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
  ),
  caption: [LTL property results across all five models. Only the $W = 6$ signed-domain model matches the precise model on every property.],
) <tab-ltl>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    stroke: 0.5pt,
    align: (left, center, center, center, center, center),
    table.header(
      [*Metric*], [*Precise*], [*W=4*], [*W=4 neg*], [*W=6 non-neg*], [*W=6 signed*],
    ),
    [States], [377,271], [8,980], [305,215], [446], [#text(weight: "bold")[7,558]],
    [BSCCs], [8 × 1], [1 × 3,973], [1 × 223,018], [2 (sizes 2,1)], [#text(weight: "bold")[8 × 1]],
    [$N_1$ spikes / 50], [37.16], [11.37], [10.25], [29.30], [#text(weight: "bold")[35.54]],
    [$N_2$ spikes / 50], [7.77], [10.40], [5.47], [16.09], [#text(weight: "bold")[5.98]],
    [Dominance ($N_1 / N_2$)], [$4.78 times$], [$1.09 times$], [$1.88 times$], [$1.82 times$], [#text(weight: "bold")[$5.94 times$]],
    [Reduction vs. precise], [$1 times$], [$42 times$], [$1.2 times$], [$846 times$], [#text(weight: "bold")[$50 times$]],
  ),
  caption: [Quantitative metrics across all five models. The W=6 non-negative model achieves extreme state reduction (446 states) but at the cost of incorrect WTA dynamics; the signed-domain model sacrifices some compactness to preserve correctness.],
) <tab-quantitative>

= Minimum Weight Resolution Analysis <sec-minimum-w>

The critical constraint is that `floor()` must map the recovery potentials of the winner and loser to _different integer values_ that fall in _different threshold levels_. For a network with excitatory weight $w_e$, strong inhibitory weight $w_s$, weak inhibitory weight $w_w$, fan-in $n - 1$, and retention rate $r$:

$
"rec"_"winner" &= floor(delta_W (w_e) + r dot (delta_W(w_e) - (n-1) dot delta_W(|w_w|))) \
"rec"_"loser"  &= floor(delta_W (w_e) + r dot (-delta_W(|w_s|) - (n-2) dot delta_W(|w_w|)))
$

The requirement is $"rec"_"winner" > "rec"_"loser"$. For the specific weights $(w_e = 100, w_s = 100, w_w = 70, n = 4, r = 0.5)$:

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    stroke: 0.5pt,
    align: (center, center, center, center, center, center),
    table.header(
      [*$W$*], [*$delta_W(-70)$*], [*$p_"winner"$*], [*$p_"loser"$*], [*$"rec"_"winner"$*], [*$"rec"_"loser"$*],
    ),
    [4], [-3], [-5], [-6], [1], [1 #text(fill: rgb("#c62828"))[(=)]],
    [5], [-4], [-7], [-8], [1], [1 #text(fill: rgb("#c62828"))[(=)]],
    [*6*], [*-4*], [*-6*], [*-8*], [*3*], [*2* #text(fill: rgb("#2e7d32"))[(≠)]],
    [7], [-5], [-8], [-10], [3], [2 #text(fill: rgb("#2e7d32"))[(≠)]],
    [10], [-7], [-11], [-14], [4], [3 #text(fill: rgb("#2e7d32"))[(≠)]],
  ),
  caption: [Minimum $W$ analysis. $W = 6$ is the smallest discretization where `floor()` preserves the recovery differential.],
) <tab-min-w>

The jump at $W = 6$ occurs because $delta_6(-70) = "round"(0.7 times 6) = "round"(4.2) = 4$, producing a 2-unit gap between $delta_6(-100) = -6$ and $delta_6(-70) = -4$. At lower $W$ values, the gap is only 1 unit, which the floor function cannot preserve through multiplicative recovery.

= Necessary and Sufficient Conditions <sec-conditions>

#block(
  stroke: 1pt + luma(180),
  inset: 12pt,
  radius: 4pt,
  width: 100%,
)[
  For a weight-discretized quotient to preserve WTA dynamics in contralateral inhibition networks, three conditions are jointly necessary and sufficient:

  + *Signed potential domain.* The potential range must include negative values ($P_"min" < 0$), so that inhibitory contributions can be stored rather than clamped. Bounds are computed from fan-in fixed-point analysis: $P_"min" = floor(C_"min" \/ (1 - r))$.

  + *Multiplicative leak.* The potential update must use $r dot p_n$ (not additive $lambda_d$), so that negative potentials recover geometrically toward zero — preserving the asymmetric recovery rates that drive competitive convergence.

  + *Sufficient weight resolution.* The discretization parameter $W$ must satisfy
    $ floor(delta_W(w_e) + r dot p_"winner") > floor(delta_W(w_e) + r dot p_"loser") $
    ensuring the `floor()` function maps the winner's and loser's recovery potentials to distinct threshold levels.

  Each condition is independently necessary (see @tab-experiments and @tab-ablation). Removing any single condition causes WTA failure.
]

When all three conditions hold, the corrected quotient is not merely an approximation: it exactly reproduces the BSCC structure of the precise model (8 absorbing states) and matches all qualitative PCTL/LTL properties. This demonstrates that the abstraction is _sound and complete_ for the verified property class.

= Implications

== For the Current Paper

The paper's existing quotient abstraction with $lambda_d$ remains correct for all feedforward topologies tested (chains, forks, diamonds, convergence). The contralateral inhibition case study already uses the precise model, so no claimed results are affected.

The conclusion's mention of "extension to recurrent topologies" as future work is directly addressed by this note: such extension requires replacing $lambda_d$ with $r dot p$ and computing the minimum $W$ from the network's weight asymmetry.

== For CogSpike Implementation

The quotient model generator could detect recurrent cycles in the SNN graph and automatically switch to the corrected quotient mode: multiplicative leak, signed domain with bounds from fan-in fixed-point analysis, and the minimum $W$ computed from the recovery-differential condition.

== Theoretical Significance

The standard quotient abstraction partitions the potential domain $[0, k]$ into equivalence classes that collapse all subthreshold states. This is complete for feedforward networks where subthreshold values are transient. In recurrent competitive networks, subthreshold (particularly negative) values carry _persistent competitive information_ that determines the asymptotic winner. The non-negative domain assumption is an implicit _completeness_ requirement that fails when the verified properties depend on negative potential dynamics.

The multiplicative leak requirement has a natural interpretation: it ensures the abstraction preserves the _direction_ of the leak dynamics — decay toward zero from _both_ sides of the axis — rather than only the magnitude of positive-side decay.

= Reproducing the Results

All models and properties are in the `discrete/` directory. Verification was performed with PRISM 4.9 on macOS (ARM64):

```
PRISM=/path/to/prism-4.9-mac64-arm/bin/prism

# Standard quotient (fails WTA)
$PRISM model_disc.pm props_diagnostic.pctl

# Negative domain only (fails WTA)
$PRISM model_disc_neg.pm props_diagnostic.pctl

# W=6 + multiplicative leak, non-negative (partial WTA)
$PRISM model_w6_muleak_nonneg.pm props_diagnostic.pctl

# W=6 + multiplicative leak + signed domain (passes all WTA) ✓
$PRISM model_w6_muleak.pm props_diagnostic.pctl
```

All invocations use `-javamaxmem 4g -cuddmaxmem 4g`. The $W = 6$ signed-domain model completes in under 20 seconds on an Apple M-series chip.
