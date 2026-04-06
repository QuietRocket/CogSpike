// ============================================================================
// Technical Note: Transfer Module Removal — Formal Proof via PRISM
// ============================================================================

#set document(
  title: "Transfer Module Removal in PRISM SNN Models: A Formal Proof of Correctness",
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
      _Transfer Module Removal — Formal Proof_
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
    Transfer Module Removal in\ PRISM SNN Models
  ]
  #v(0.3em)
  #text(size: 13pt, fill: luma(80))[
    A Formal Proof of Correctness via Model Checking
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
  *Abstract.* The CogSpike PRISM code generator previously used _transfer modules_—per-synapse auxiliary variables that copy a source neuron's spike output—to implement synaptic delay. We formally prove, via three PRISM model checking scenarios verified against PRISM 4.9, that these modules are not only redundant but introduce _spurious latency_: each transfer module adds one extra tick of propagation delay per synapse. Removing transfer modules restores correct 1-tick-per-synapse delay isomorphic with the simulation engine, while reducing the theoretical state space by a factor of $2^(|E_"int"|)$.
]

= Background

== The Original Transfer Module Pattern

In the original PRISM code generator, each neuron-to-neuron synapse (edge $e = (A, B)$) was accompanied by a _transfer module_ containing a binary variable $z_(A,B)$ that copied the source neuron's spike output at each tick:

```
module Transfer_A_B
  z_A_B : [0..1] init 0;
  [tick] true -> (z_A_B' = y_A);
endmodule
```

The downstream neuron's membrane potential formula referenced the transfer variable rather than the source spike output directly:

$ "newPotential"_B = max(P_min, min(P_max, floor(w_(A,B) dot z_(A,B) + r dot p_B))) $

This pattern was introduced to ensure that spike signals "transferred" reliably between neurons with a 1-tick synaptic delay, consistent with biological axonal propagation.

== The Change Under Investigation

In commits `2355673` (precise generator) and `42890e7` (discretized generator), transfer modules were removed. The potential formula now reads the source neuron's spike output $y_A$ directly:

$ "newPotential"_B = max(P_min, min(P_max, floor(w_(A,B) dot y_A + r dot p_B))) $

This note formally verifies that this change is correct.

= PRISM Synchronous Composition Semantics

The correctness argument rests on a fundamental property of PRISM's synchronous module composition:

#block(
  stroke: 1pt + luma(180),
  inset: 12pt,
  radius: 4pt,
  width: 100%,
)[
  *PRISM Synchronous Evaluation Rule.* When multiple modules synchronize on a shared action label (e.g., `[tick]`), all guard expressions are evaluated on the _pre-transition_ (current) state. All primed variable updates take effect simultaneously _after_ guard evaluation. #footnote[PRISM Manual, §4.3: "the combined transitions are computed by synchronising over common action labels."]
]

Therefore, when neuron $B$'s guard references $y_A$ in the formula `newPotential_B`, it reads the value of $y_A$ that was set during the _previous_ tick—because $y_A$'s update (primed assignment) from the current tick has not yet taken effect. This is precisely the 1-tick delay semantics that transfer modules were intended to provide.

= Proof Methodology

We construct three PRISM models covering the three fundamental topologies through which spikes propagate: chains (serial), forks (fan-out), and convergence (fan-in with coincidence detection). All models use deterministic thresholds (1 level, $P_"rth" = 100$, weight $= 110 > P_"rth"$) to eliminate stochastic noise and enable exact timing verification.

Each model was verified using PRISM 4.9 (macOS ARM64) with the symbolic (BDD) engine. Properties are expressed in Probabilistic Computation Tree Logic (PCTL).

= Proof 1: Chain Topology — Serial Propagation Delay <sec-chain>

== Model

A 3-neuron chain with a pulse input:

#align(center)[
  #text(size: 12pt)[
    $"Input"$ $arrow.r$ $A$ $arrow.r$ $B$ $arrow.r$ $C$
  ]
]

All weights are 110 (exceeding threshold 100), leak rate $r = 0.95$, no refractory periods. The input fires a single pulse at step 0.

== Properties and Results

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    align: (left, left, center, center),
    table.header(
      [*ID*], [*PCTL Property*], [*Expected*], [*Result*],
    ),
    [P1], [`P=? [ F "output_fires" ]`], [1.0], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
    [P2], [`P=? [ F "A_fires" ]`], [1.0], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
    [P3], [`P=? [ F "B_fires" ]`], [1.0], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
    [P4], [`P=? [ X X "A_fires" ]`], [1.0], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
    [P5], [`P=? [ X X X "B_fires" ]`], [1.0], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
    [P6], [`P=? [ X X X X "output_fires" ]`], [1.0], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
    [P7], [`P=? [ X X X !(y_C=1) ]`], [1.0], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
    [P8], [`P=? [ F<=5 "output_fires" ]`], [1.0], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
    [P9], [`P=? [ F<=3 "output_fires" ]`], [0.0], [#text(fill: rgb("#2e7d32"), weight: "bold")[0.0 ✓]],
  ),
  caption: [Chain topology verification results (no transfer modules). All 9 properties pass.],
) <tab-chain>

Properties P4–P6 are the critical timing proofs: $A$ fires at step 2, $B$ at step 3, $C$ at step 4—exactly 1 tick per synapse. P7 confirms $C$ has _not_ fired at step 3 (before $B$'s spike reaches it). P8–P9 bound the propagation: reachable within 5 steps but not within 3.

== Comparison: Same Properties on Transfer Module Model

Running the identical properties against a model _with_ transfer modules reveals a critical difference:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    align: (left, left, center, center),
    table.header(
      [*Property*], [*No Transfer*], [*With Transfer*], [*Verdict*],
    ),
    [`P=? [ F "output_fires" ]`], [1.0], [1.0], [Both reach],
    [`P=? [ X X X "B_fires" ]`], [1.0], [#text(fill: rgb("#c62828"), weight: "bold")[0.0]], [#text(fill: rgb("#c62828"))[B is late!]],
    [`P=? [ X X X X "output_fires" ]`], [1.0], [#text(fill: rgb("#c62828"), weight: "bold")[0.0]], [#text(fill: rgb("#c62828"))[C is late!]],
    [`P=? [ F<=5 "output_fires" ]`], [1.0], [#text(fill: rgb("#c62828"), weight: "bold")[0.0]], [#text(fill: rgb("#c62828"))[Needs >5 steps]],
  ),
  caption: [Timing comparison: without vs. with transfer modules. Transfer modules add 1 extra tick of latency per synapse.],
) <tab-comparison>

== Execution Trace Analysis

The PRISM `-simpath` command produces the full state trace, confirming the timing divergence:

#figure(
  grid(
    columns: 2,
    gutter: 12pt,
    [
      #text(weight: "bold", size: 10pt)[Without Transfer Modules]
      #table(
        columns: (auto, auto, auto, auto, auto),
        stroke: 0.5pt,
        align: center,
        table.header([*Step*], [$x_"Inp"$], [$y_A$], [$y_B$], [$y_C$]),
        [0], [0], [0], [0], [0],
        [1], [*1*], [0], [0], [0],
        [2], [0], [*1*], [0], [0],
        [3], [0], [0], [*1*], [0],
        [4], [0], [0], [0], [*1*],
      )
    ],
    [
      #text(weight: "bold", size: 10pt)[With Transfer Modules]
      #table(
        columns: (auto, auto, auto, auto, auto, auto, auto),
        stroke: 0.5pt,
        align: center,
        table.header([*Step*], [$x_"Inp"$], [$y_A$], [$z_(A,B)$], [$y_B$], [$z_(B,C)$], [$y_C$]),
        [0], [0], [0], [0], [0], [0], [0],
        [1], [*1*], [0], [0], [0], [0], [0],
        [2], [0], [*1*], [0], [0], [0], [0],
        [3], [0], [0], [*1*], [0], [0], [0],
        [4], [0], [0], [0], [*1*], [0], [0],
        [5], [0], [0], [0], [0], [*1*], [0],
        [6], [0], [0], [0], [0], [0], [*1*],
      )
    ],
  ),
  caption: [State traces from PRISM `-simpath`. Without transfer modules: 4 steps to output. With transfer modules: 6 steps — each $z$ variable adds 1 tick of latency.],
) <fig-traces>

*Diagnosis.* The transfer module `[tick] true -> (z_A_B' = y_A)` copies $y_A$ into $z_(A,B)$ during the _same_ tick that $A$ fires. However, the primed assignment $z'_(A,B) = y_A$ only takes effect _after_ the tick completes—so $B$ cannot see $z_(A,B) = 1$ until the _following_ tick. This creates a 2-tick chain: tick $t$: $A$ fires ($y_A arrow.l 1$); tick $t+1$: transfer copies ($z_(A,B) arrow.l 1$); tick $t+2$: $B$ reads $z_(A,B) = 1$ and fires. Without the transfer module, $B$ reads $y_A$ directly at tick $t+1$—a single 1-tick delay.

= Proof 2: Fork Topology — Simultaneous Fan-Out Delivery

== Model

One source neuron fans out to two targets:

#align(center)[
  #text(size: 12pt)[
    #grid(
     columns: 1,
     gutter: 0pt,
     align(center)[$"Input" arrow.r A arrow.r B_1$],
     align(center)[$#h(5.85em) arrow.br B_2$],
    )
  ]
]

== Results

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    align: (left, left, center, center),
    table.header(
      [*ID*], [*PCTL Property*], [*Expected*], [*Result*],
    ),
    [P1], [`P=? [ F "both_fire" ]`], [1.0], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
    [P2], [`P=? [ F "B1_fires" ]`], [1.0], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
    [P3], [`P=? [ F "B2_fires" ]`], [1.0], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
    [P4], [`P=? [ X X X "both_fire" ]`], [1.0], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
    [P5], [`P=? [ G ("B1_fires" => "B2_fires") ]`], [1.0], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
    [P6], [`P=? [ G ("B2_fires" => "B1_fires") ]`], [1.0], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
  ),
  caption: [Fork topology verification results. P5–P6 prove that $B_1$ and $B_2$ always fire in lockstep.],
) <tab-fork>

Properties P5–P6 are the strongest: $bold(G)(y_(B_1)=1 arrow.r.double y_(B_2)=1)$ holds with probability 1.0, meaning that _at every tick along every possible path_, whenever $B_1$ fires, $B_2$ fires at the same instant (and vice versa). Fan-out delivery is perfectly synchronized without transfer modules.

= Proof 3: Convergence Topology — Coincidence Detection

== Model

Two sources with _subthreshold_ weights converge on one target:

#align(center)[
  #text(size: 12pt)[
    $"Input"_1 arrow.r A #h(0.2em) arrow.br$
    $#h(7.1em) C$
    $"Input"_2 arrow.r B #h(0.2em) arrow.tr$
  ]
]

Each edge $A arrow.r C$ and $B arrow.r C$ has weight 55, while the threshold is 100. Neuron $C$ can only fire when _both_ $A$ and $B$ spike simultaneously ($55 + 55 = 110 > 100$). Leak rate is set to $r = 0.0$ (no leak) to isolate the coincidence detection mechanism.

== Results

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    align: (left, left, center, center),
    table.header(
      [*ID*], [*PCTL Property*], [*Expected*], [*Result*],
    ),
    [P1], [`P=? [ F "output_fires" ]`], [1.0], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
    [P2], [`P=? [ X X "both_sources" ]`], [1.0], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
    [P3], [`P=? [ X X X "output_fires" ]`], [1.0], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
    [P4], [`P=? [ X X !(y_C=1) ]`], [1.0], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
    [P5], [`R{"spikes_C"}=? [ C<=10 ]`], [1.0], [#text(fill: rgb("#2e7d32"), weight: "bold")[1.0 ✓]],
  ),
  caption: [Convergence topology verification results. P3 confirms coincidence detection at step 3.],
) <tab-convergence>

Property P3 is the most demanding: the potential formula $"newPotential"_C = 55 dot y_A + 55 dot y_B$ must see _both_ $y_A = 1$ and $y_B = 1$ in the same evaluation to produce $110 > 100$. Without transfer modules, both values are read from the pre-transition state at step 3, where both $A$ and $B$ fired at step 2. P5 confirms exactly 1 cumulative spike over 10 ticks, ruling out spurious repeated firing.

= State Space Impact

Each transfer module adds a binary variable $z_(i,j) in {0, 1}$, contributing a multiplicative factor of 2 to the theoretical state space. For a network with $|E_"int"|$ internal (neuron-to-neuron) edges, the total state space overhead is:

$ Delta_"transfer" = 2^(|E_"int"|) $

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt,
    align: (left, center, center),
    table.header(
      [*Topology*], [$|E_"int"|$], [*State factor saved*],
    ),
    [Chain (3 neurons)], [2], [$4 times$],
    [Fork (3 neurons)], [2], [$4 times$],
    [Convergence (3 neurons)], [2], [$4 times$],
    [Contralateral inhibition (4 neurons)], [8], [$256 times$],
    [Fully connected (5 neurons)], [20], [$1,048,576 times$],
  ),
  caption: [Theoretical state space reduction from removing transfer modules.],
) <tab-statespace>

For the contralateral inhibition topology studied in the main CogSpike paper, removing transfer modules eliminates $256 times$ overhead—a critical reduction that helps avoid the Out-Of-Memory failures observed at larger network sizes.

= Conclusion

The formal PRISM verification across three canonical topologies establishes that:

+ *Transfer modules are redundant.* PRISM's synchronous composition semantics guarantee that guard expressions read pre-transition values, providing exactly the 1-tick synaptic delay that transfer modules were intended to achieve.

+ *Transfer modules introduce spurious latency.* Each transfer variable adds one extra tick of propagation delay per synapse, making the PRISM model _less_ isomorphic with the simulation engine (_cf._ @fig-traces).

+ *Removal improves isomorphism.* The simulation engine implements 1-tick delay via double-buffered "Phase 4" updates. The no-transfer PRISM model matches this exactly: guards at tick $t$ read the spike output $y$ that was set at tick $t - 1$.

+ *Removal reduces state space.* The $2^(|E_"int"|)$ factor from transfer variables is eliminated entirely, enabling verification of larger networks within PRISM's memory limits.

All 20 PCTL properties across two model variants and three topologies were verified using PRISM 4.9. The PRISM model files and property specifications are available in the `transfer_proof/` directory of the CogSpike repository.
