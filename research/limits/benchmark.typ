// Empirical OOM Benchmark: Maximum Verifiable SNN Sizes
// Extension to "Topology-Dependent Scaling Limits of SNN Verification"
// MARCH 2026

#set document(
  title: "Empirical OOM Benchmark for SNN Verification via PRISM",
  author: "CogSpike Research Team",
  date: datetime.today(),
)

#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.5cm),
  numbering: "1",
)

#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

// Custom environments (consistent with the main research note)
#let definition(body) = block(
  width: 100%,
  inset: 8pt,
  stroke: (left: 2pt + gray),
  [*Definition.* #body],
)

#let remark(body) = block(
  width: 100%,
  inset: 8pt,
  stroke: (left: 2pt + orange),
  [*Remark.* #body],
)

#let intuition(body) = block(
  width: 100%,
  inset: 8pt,
  fill: rgb("#f0f7ff"),
  stroke: (left: 2pt + rgb("#4a90d9")),
  [💡 *Intuition:* #body],
)

#let finding(body) = block(
  width: 100%,
  inset: 8pt,
  fill: rgb("#f0fff0"),
  stroke: (left: 2pt + rgb("#2e8b57")),
  [🔬 *Finding:* #body],
)

// ============================================================================
// DOCUMENT
// ============================================================================

#align(center)[
  #text(size: 18pt, weight: "bold")[
    Empirical OOM Benchmark \
    for SNN Verification via PRISM
  ]
  #v(0.5em)
  #text(size: 12pt)[CogSpike Research Team — March 2026]
  #v(0.3em)
  #text(size: 10pt, style: "italic")[
    Companion to: _Topology-Dependent Scaling Limits of SNN Verification_
  ]
]

#v(1em)

*Abstract.* We empirically determine the maximum number of neurons verifiable
in PRISM before out-of-memory (OOM) errors occur, for two canonical SNN
topologies: _chains_ (depth-scaling) and _forks_ (width-scaling). We test
across three model configurations (Deterministic, Fast, Full), in both precise
and weight-discretized modes (with $W in {2, 3, 5}$), under fixed PRISM
resource budgets. The results provide concrete, reproducible scaling limits
that complement the theoretical state-space analysis of our prior work and
quantify the practical impact of the weight discretization scheme.

= Introduction

Our companion note @main-note derived closed-form expressions for the DTMC
state space as a function of SNN topology and model configuration, and
validated them empirically on seven canonical topologies. That analysis
identified the key scaling parameters but left open the question:

#align(center)[
  _For a concrete PRISM resource budget, what is the maximal topology size_
  _that can be verified before OOM?_
]

This note answers that question via systematic binary-search probing of two
topology families, under fixed and reproducible resource constraints.

= Experimental Setup

== Topologies Under Test

We study two complementary topology families that isolate _depth_ and _width_
scaling:

#definition[
  *Arrangement 1 — Chain.* A linear sequence of $N$ processing neurons:
  $ "In" -> N_1 -> N_2 -> dots.c -> N_N $
  This isolates the effect of _depth_: each additional neuron multiplies the
  state space by the per-neuron factor, with fan-in = fan-out = 1.
]

#definition[
  *Arrangement 2 — Fork.* A single hub neuron with $B$ parallel branches:
  $ "In" -> N_1 -> {N_2, N_3, dots, N_(B+1)} $
  This isolates the effect of _width_: the hub neuron feeds $B$ independent
  leaf neurons. The total processing neuron count is $B + 1$.
]

All synaptic weights are set to $w = 80$ (excitatory), and input neurons
use the `AlwaysOn` pattern, consistent with the main research note.

== Model Configurations

We test the same three presets as the main note, each in both _precise_ and
_discretized_ mode:

#figure(
  table(
    columns: 6,
    stroke: 0.5pt,
    [*Preset*], [*Thresholds ($k$)*], [*ARP*], [*RRP*], [*Model Type*], [*Weight Levels ($W$)*],
    [Deterministic], [1], [Off], [Off], [Precise], [---],
    [Deterministic], [1], [Off], [Off], [Discretized], [2, 3, 5],
    [Fast], [4], [Off], [Off], [Precise], [---],
    [Fast], [4], [Off], [Off], [Discretized], [2, 3, 5],
    [Full], [10], [On (ARP=2)], [On (RRP=4)], [Precise], [---],
    [Full], [10], [On (ARP=2)], [On (RRP=4)], [Discretized], [2, 3, 5],
  ),
  caption: "Configuration matrix: 3 presets × (1 precise + 3 discretized) = 12 configurations.",
) <tab-configs>

== PRISM Resource Budget

All experiments use identical PRISM resources to ensure fair comparison.
These are the *PRISM defaults* that any researcher would encounter out of
the box, maximizing reproducibility:

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    [*Parameter*], [*Value*], [*PRISM Flag*],
    [Java Heap], [1 GB], [`-javamaxmem 1g`],
    [Java Stack], [4 MB], [`-javastack 4m`],
    [CUDD Memory], [1 GB], [`-cuddmaxmem 1g`],
    [Engine], [Hybrid (default)], [---],
    [Time Bound ($T_"max"$)], [20], [Model constant],
    [Timeout], [120 s], [`-timeout 120`],
  ),
  caption: "Fixed PRISM resource constants for all benchmark experiments.",
) <tab-resources>

#remark[
  The choice of default PRISM values (1 GB each for JVM heap and CUDD BDD
  memory) is deliberate. While larger budgets would increase absolute limits,
  the _proportional_ differences between configurations are invariant with
  respect to the memory budget. A researcher with 4 GB CUDD can scale
  our chain limits by approximately $log_(f_n)(4) slash log_(f_n)(1)$ where
  $f_n$ is the per-neuron factor, but the ratio between e.g.\ a precise fast
  chain and a discretized $W$=3 fast chain remains constant.
]

== Methodology

For each of the 12 configurations × 2 topologies = 24 experimental
combinations, we perform a *binary search* over the size parameter
$N$ (chain length) or $B$ (fork branch count) in the range $[1, 30]$.
At each probe point, PRISM is invoked with `-exporttrans` and
`-exportstates`; the model build phase is the OOM-sensitive step.

- *Success*: PRISM completes and exports state/transition files.
  We record the reachable state count $|S|$, transition count $|T|$,
  and wall-clock time.
- *Failure*: PRISM exits with a non-zero code, or its output contains
  `OutOfMemoryError`, `CUDD`, or similar error indicators.

Binary search converges in $ceil(log_2(30)) = 5$ probes per configuration,
yielding $approx 120$ total PRISM invocations.

= Results

== Chain Topology — Maximum Depth Before OOM

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    [*Configuration*], [*Max Neurons*], [*States at Max*], [*Transitions*], [*Time (s)*],
    [Det. Precise], [8], [131], [131], [8],
    [Det. Disc. $W$=2], [10], [7], [7], [11],
    [Det. Disc. $W$=3], [10], [8], [8], [13],
    [Det. Disc. $W$=5], [9], [19], [19], [4],
    [Fast Precise], [6], [20 902 972], [101 459 903], [121],
    [Fast Disc. $W$=2], [7], [3 035], [11 951], [7],
    [Fast Disc. $W$=3], [6], [993], [3 375], [3],
    [Fast Disc. $W$=5], [6], [16 366], [93 728], [5],
    [Full Precise], [4], [362 289], [885 160], [20],
    [Full Disc. $W$=2], [5], [104], [133], [7],
    [Full Disc. $W$=3], [5], [116], [152], [15],
    [Full Disc. $W$=5], [4], [509], [664], [2],
  ),
  caption: [Maximum chain length (number of processing neurons) before OOM for each
    configuration. Resource budget: 1 GB JVM heap + 1 GB CUDD.],
) <tab-chain>

== Fork Topology — Maximum Width Before OOM

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    [*Configuration*], [*Max Branches*], [*States at Max*], [*Transitions*], [*Time (s)*],
    [Det. Precise], [9], [12], [12], [15],
    [Det. Disc. $W$=2], [11], [7], [7], [19],
    [Det. Disc. $W$=3], [11], [8], [8], [26],
    [Det. Disc. $W$=5], [10], [15], [15], [9],
    [Fast Precise], [6], [17 843], [97 847], [19],
    [Fast Disc. $W$=2], [6], [134], [327], [5],
    [Fast Disc. $W$=3], [6], [135], [329], [22],
    [Fast Disc. $W$=5], [5], [13 579], [158 168], [4],
    [Full Precise], [3], [45 966], [98 301], [14],
    [Full Disc. $W$=2], [4], [163], [210], [5],
    [Full Disc. $W$=3], [4], [164], [212], [13],
    [Full Disc. $W$=5], [3], [729], [952], [3],
  ),
  caption: [Maximum fork branch count before OOM. Each fork has 1 hub neuron plus $B$ leaf
    neurons (total $B + 1$ processing neurons). Resource budget: 1 GB JVM + 1 GB CUDD.],
) <tab-fork>

== Discretization Sensitivity ($W$ Effect)

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    [*$W$*], [*Chain Max (Fast)*], [*Fork Max (Fast)*], [*Chain Max (Full)*], [*Fork Max (Full)*],
    [Precise], [6], [6], [4], [3],
    [2], [7], [6], [5], [4],
    [3], [6], [6], [5], [4],
    [5], [6], [5], [4], [3],
  ),
  caption: [Effect of weight level $W$ on the maximum verifiable size. Lower $W$ reduces
    per-neuron state domain, allowing larger networks.],
) <tab-w-effect>

= Analysis

== Depth vs Width

#intuition[
  Chains (depth) and forks (width) should exhibit different scaling because:
  - In a *chain*, each neuron sees increasingly rich input distributions from
    its predecessor. The per-neuron state expansion accelerates with depth.
  - In a *fork*, all leaf neurons see the _same_ hub neuron's output, so their
    state spaces are structurally identical (symmetric). PRISM's BDD engine
    can exploit this symmetry.

  We therefore expect the OOM limit for forks to be _higher_ than for chains
  of equivalent processing neuron count.
]

The results partially confirm this intuition. For *deterministic* models, forks
consistently support 1 more branch than chains support neurons (9 vs 8 precise,
11 vs 10 disc. $W$=2). However, for *Fast* and *Full* models the limits are
identical or nearly so (both 6 for Fast Precise, 3-4 for Full).

#finding[
  The depth-vs-width advantage is most pronounced in low-complexity
  (deterministic) models. As per-neuron state complexity increases ($k = 4$ or
  $k = 10$), the per-neuron factor dominates and the structural advantage of
  BDD symmetry in forks diminishes. In practice, chains and forks hit
  comparable limits for the Fast and Full presets.
]

== Impact of Discretization

The weight discretization scheme @weight-disc reduces the per-neuron potential
domain from $R_n approx 121$ (precise, with $w = 80$) to
$P_"max"^d + 1 = T_d + delta_W(w) + 1$:

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    [*$W$*], [$T_d$], [$delta_W(80)$], [$P^d_"max" + 1$],
    [2], [$ceil(100 dot 2 / 100) = 2$], [$"round"(80 dot 2 / 100) = 2$], [5],
    [3], [$ceil(100 dot 3 / 100) = 3$], [$"round"(80 dot 3 / 100) = 2$], [6],
    [5], [$ceil(100 dot 5 / 100) = 5$], [$"round"(80 dot 5 / 100) = 4$], [10],
  ),
  caption: [Discretized potential domain for single fan-in ($w = 80$) at different $W$.],
) <tab-disc-domain>

#remark[
  Smaller $W$ values provide greater state space reduction per neuron, but at
  the cost of precision: the Soundness Theorem @weight-disc guarantees no
  spurious spikes at any $W$, but coarser discretization loses information
  about the exact membrane potential trajectory. The choice of $W$ is a
  trade-off between verifiable network size and per-neuron behavioral fidelity.
]

== Memory Proportionality

#finding[
  The absolute OOM limits reported here are specific to the 1 GB CUDD + 1 GB
  JVM budget. However, the *ratios* between configurations are
  budget-invariant. If a chain in Fast Precise mode reaches $N_"max" = n_1$
  neurons and in Fast Disc. $W$=3 mode reaches $N_"max" = n_2$, then
  doubling the CUDD budget would increase both limits, but the improvement
  factor $n_2 / n_1$ remains approximately constant.

  This is because the state space grows as $f^N$ where $f$ is the per-neuron
  base. Increasing memory by a factor $M$ adds $log_f(M)$ additional neurons
  to the limit — an _additive_ shift that preserves the multiplicative ratio
  between configurations.
]

= Conclusions

+ *Depth limits*: Chains in precise mode reach $N = 6$ neurons (Fast, 20.9M
  states) and $N = 4$ neurons (Full, 362K states) before OOM.

+ *Width limits*: Forks accommodate $B = 6$ branches (Fast Precise, 17.8K
  states) and $B = 4$ branches (Full Disc. $W$=3, 164 states).

+ *Discretization multiplier*: For the Full preset, discretization with
  $W in {2, 3}$ extends the chain limit from 4 to 5 (+25%) and the fork
  limit from 3 to 4 (+33%). The reduction in state count is dramatic:
  362K $arrow.r$ 116 states (3 100× reduction) at the same depth.

+ *The $W$ trade-off*: Moving from $W = 2$ to $W = 5$ reduces the Full chain
  limit from 5 to 4 neurons. For Fast chains, $W = 2$ adds 1 neuron (7 vs 6)
  while $W = 3$ and $W = 5$ provide no additional reach. The sweet spot is
  $W = 2$ for maximum verification reach and $W = 3$ for a balanced
  precision-scalability trade-off.

+ *Depth vs Width*: Forks and chains reach comparable limits for Fast and
  Full presets. The fork advantage only manifests in deterministic models
  (9 vs 8 precise, 11 vs 10 disc.), where BDD symmetry can exploit the
  identical leaf neuron structure.

+ *Deterministic models scale best*: With $k = 1$ and no refractory periods,
  chains reach $N = 8$ (precise) and $N = 10$ (disc. $W = 2$) — 2× more
  neurons than the Full preset. This confirms that threshold levels are the
  dominant cost factor per neuron.

#bibliography("references.bib", style: "ieee")
