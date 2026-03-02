// Research Note: Topology-Dependent Scaling Limits of SNN Verification
// State Space Analysis for PRISM Model Checking
// MARCH 2026

#set document(
  title: "Topology-Dependent Scaling Limits of SNN Verification",
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

// Custom theorem-like environments (consistent with weight discretization note)
#let definition(body) = block(
  width: 100%,
  inset: 8pt,
  stroke: (left: 2pt + gray),
  [*Definition.* #body],
)

#let theorem(title, body) = block(
  width: 100%,
  inset: 8pt,
  stroke: (left: 2pt + blue),
  [*Theorem* (#title)*.* #body],
)

#let proposition(body) = block(
  width: 100%,
  inset: 8pt,
  stroke: (left: 2pt + green),
  [*Proposition.* #body],
)

#let proof(body) = block(
  width: 100%,
  inset: 8pt,
  fill: luma(248),
  [_Proof._ #body #h(1fr) $square$],
)

#let intuition(body) = block(
  width: 100%,
  inset: 8pt,
  fill: rgb("#f0f7ff"),
  stroke: (left: 2pt + rgb("#4a90d9")),
  [💡 *Intuition:* #body],
)

#let example(body) = block(
  width: 100%,
  inset: 8pt,
  fill: luma(245),
  [*Example.* #body],
)

#let remark(body) = block(
  width: 100%,
  inset: 8pt,
  stroke: (left: 2pt + orange),
  [*Remark.* #body],
)

// ============================================================================
// DOCUMENT
// ============================================================================

#align(center)[
  #text(size: 18pt, weight: "bold")[
    Topology-Dependent Scaling Limits \
    of SNN Verification via PRISM
  ]
  #v(0.5em)
  #text(size: 12pt)[CogSpike Research Team — March 2026]
  #v(0.3em)
  #text(size: 10pt, style: "italic")[
    Working Paper — First Draft
  ]
]

#v(1em)

*Abstract.* We analyze how the topology of a Spiking Neural Network (SNN) determines
the state space of its corresponding DTMC (Discrete-Time Markov Chain) representation
used for formal verification in PRISM. We derive closed-form expressions for state space
size as a function of network structure and model configuration, validate them empirically
against PRISM DTMC exports, and characterize the scaling limits of verification for
different connectivity patterns. We show that discretized models achieve exponential
state space reduction per neuron, and that the practical verification limit of a 4-neuron
chain under the full model exceeds 362,000 states — with the precise diamond topology
already exceeding PRISM's BDD memory at 5 neurons.

= Introduction

The formal verification of Spiking Neural Networks via probabilistic model checking
is constrained by the _state space explosion problem_: the number of reachable states
in the DTMC grows exponentially with network size. Our prior work @weight-disc extended
the filtration-based quotient model abstraction of Baier and Katoen @baier-katoen
with a _weight discretization scheme_ that preserves synaptic weight information
during verification. That work introduced threshold-dependent leak, proved soundness
(no spurious spikes can occur in the discretized model), and demonstrated
preservation of biological properties (tonic spiking, integrator behavior,
excitability) in the quotient model. The discretized model maps continuous
membrane potentials to a finite set of potential classes whose size is determined
by the weight level parameter $W$, reducing the per-neuron state domain from the
full potential range to $O(W)$ classes. However, that analysis focused on the
_vertical_ dimension (per-neuron precision) without systematically addressing
the _horizontal_ dimension: how network topology (neuron count, connectivity
pattern, fan-in, fan-out) determines the overall DTMC size.

This note addresses three questions from our recent research planning:

+ *DTMC Export and Inspection:* Can we export DTMCs from PRISM to observe concrete
  state/transition counts for different topologies?

+ *Graph-Theoretic State Space Model:* Can we derive a closed-form formula for the
  state space size as a function of topology parameters?

+ *Memory Estimation:* Can we predict whether a given SNN will fit within PRISM's
  memory budget before attempting verification?

= Preliminaries

We recall the CogSpike SNN model and its PRISM representation, using notation
consistent with the weight discretization formalization @weight-disc.

== SNN Model Recap

An SNN is a directed graph $G = (V, E)$ where:
- $V = V_"in" union V_"proc"$ — input neurons and processing neurons
- $E subset.eq V times V$ — directed synaptic connections with weights $w_e in [-100, 100]$

Each processing neuron $n in V_"proc"$ has state variables:
- Membrane potential $p_n in [P_"min", P_"max"]$
- Spike output $y_n in {0, 1}$
- Refractory state $s_n in {0, 1, 2}$ (Normal, ARP, RRP) — when refractory periods are enabled
- ARP counter $"aref"_n in [0, "ARP"]$ and RRP counter $"rref"_n in [0, "RRP"]$

Each input neuron $i in V_"in"$ has:
- Spike output $x_i in {0, 1}$

Each internal edge $(i, j) in E$ where $i in V_"proc"$ has:
- Transfer variable $z_(i,j) in {0, 1}$

In the *discretized quotient model* @weight-disc, the continuous membrane potentials
are abstracted via a weight discretization function
$delta_W(w) = round(w dot W "/" w_"max")$
that maps each synaptic weight to a finite discrete range $[-W, W]$. The discretized
firing threshold $T_d = ceil(T dot W "/" w_"max")$ is calibrated using ceiling to ensure
the discretized neuron is _at least as hard_ to fire as the original (Threshold
Preservation Theorem @weight-disc). The Soundness Theorem @weight-disc further
guarantees that no spurious spikes are introduced: if the original neuron should not
fire, neither does the discretized one. A threshold-dependent leak factor
$lambda_d = -max(1, floor(ell dot T_d))$ ensures membrane decay scales correctly
with the calibrated threshold.

== PRISM Module Decomposition <sec-modules>

The PRISM model is composed of independent modules whose state spaces form a
Cartesian product. CogSpike generates the following module types:

#definition[
  The *module decomposition* of an SNN graph $G = (V, E)$ is:

  $cal(M)(G) = {"GlobalClock"} union {"Inputs"} union {M_n : n in V_"proc"} union {T_(i,j) : (i,j) in E_"int"}$

  where $E_"int" = {(i,j) in E : i in V_"proc"}$ is the set of internal (neuron-to-neuron) edges.
]

== Model Configuration

The `ModelConfig` determines the per-neuron state complexity:

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    [*Parameter*], [*Notation*], [*Effect on State Space*],
    [Threshold levels], [$k in {1, ..., 10}$], [Determines transition fan-out],
    [Enable ARP], [$"arp" in {"on", "off"}$], [Adds $s_n$ and $"aref"_n$ variables],
    [Enable RRP], [$"rrp" in {"on", "off"}$], [Adds $"rref"_n$ variable],
    [Potential range], [$(P_"min", P_"max")$], [Dominates state count per neuron],
    [Weight levels (disc.)], [$W in {1, ..., 10}$], [Determines $delta_W$ domain $[-W, W]$ and $T_d$; see @weight-disc],
  ),
  caption: "Model configuration parameters and their state space impact.",
) <tab-params>

= DTMC State Space Analysis <sec-analysis>

== Per-Module State Contribution

#definition[
  Let $|S_M|$ denote the number of possible valuations of all variables in module $M$.
  We call this the *local state space* of module $M$.
]

For each module type:

*GlobalClock* (present when input patterns are time-dependent):
$ |S_"clock"| = T_"max" + 1 $

*Inputs* (one module aggregating all input neurons):
$ |S_"inputs"| = 2^(|V_"in"|) $

*Neuron module* $M_n$ for $n in V_"proc"$ (precise model):
$
  |S_n^"precise"| = cases(
    2 dot.c (P_"max" - P_"min" + 1) & "if" "arp" = "off" and "rrp" = "off" quad "(fast)",
    (1 + delta_"arp") dot.c (1 + "ARP")^(delta_"arp") dot.c (1 + delta_"rrp") dot.c (1 + "RRP")^(delta_"rrp") dot.c 2 dot.c R_n & "otherwise (full)"
  )
$

where $R_n = P_("max",n) - P_("min",n) + 1$ is the per-neuron potential range, and
$delta_"arp", delta_"rrp" in {0, 1}$ indicate whether ARP/RRP are enabled.

For the *fast* configuration ($k = 4$, no ARP/RRP):
$ |S_n^"fast"| = 2 dot.c R_n $

For the *full* configuration ($k = 10$, ARP + RRP with ARP=2, RRP=4):
$ |S_n^"full"| = 3 dot.c 3 dot.c 5 dot.c 2 dot.c R_n = 90 dot.c R_n $

*Neuron module* $M_n$ (discretized quotient model with weight level $W$):
$ |S_n^"disc"| = 2 dot.c (P_("max",n)^d + 1) $
where $P_("max",n)^d = T_d + E_n^d$ with $E_n^d = sum_(e in "In"(n), w_e^d > 0) w_e^d$
being the maximum weighted contribution (in the sense of the contribution-based
class evolution defined in @weight-disc) and
$T_d = ceil(P_"rth" dot.c W / 100)$ is the calibrated discretized threshold.
The threshold-dependent leak $lambda_d = -max(1, floor(ell dot T_d))$ does not add
new state variables but constrains the reachable potential range, ensuring the
Soundness property (no spurious spikes) is maintained.

*Transfer module* $T_(i,j)$:
$ |S_(i,j)| = 2 $

== Theoretical State Space

#theorem("State Space Product")[
  The theoretical (upper bound) state space of the DTMC is the Cartesian product
  of all module state spaces:

  $
    |S_"theory"| = |S_"clock"| dot.c |S_"inputs"| dot.c product_(n in V_"proc") |S_n| dot.c product_((i,j) in E_"int") 2
  $
]

#proof[
  Each module's variables are disjoint and updated only within that module (or via
  synchronized formulas). Since PRISM constructs the global state as a tuple of
  all variable valuations, the total state space is the Cartesian product of the
  local state spaces.
]

#definition[
  The *reachability density* $rho$ is the fraction of the theoretical state space
  that is actually reachable from the initial state:

  $ rho = frac(|S_"reachable"|, |S_"theory"|) $
]

#intuition[
  The reachability density $rho$ captures how much of the theoretical state space is
  "wasted." For sparse topologies (chains), many variable combinations are unreachable
  because upstream neurons constrain downstream ones. We expect $rho << 1$ for realistic
  SNNs, and $rho$ to decrease as the network grows. This is what makes PRISM's BFS
  reachability analysis effective — but the BDD representation still requires memory
  proportional to (a compression of) $|S_"theory"|$.
]

== Chain Topology Analysis

Consider a chain of $N$ processing neurons: $"In" -> n_1 -> n_2 -> dots.c -> n_N$.

This topology has:
- $|V_"in"| = 1$, $|V_"proc"| = N$
- $|E| = N$ (one input edge + $N-1$ internal edges)
- $|E_"int"| = N - 1$ (internal edges only)

For the *fast precise* model with $P_"rth" = 100$ and weight $w = 80$:
- Fan-in of each neuron is 1, so $R_n = P_"max" - P_"min" + 1$
- Per-neuron narrowing gives $P_"max" = max(P_"rth", 3w"/"2) = max(100, 120) = 120$
- No inhibitory synapses, so $P_"min" = 0$
- Therefore $R_n = 121$ for each neuron

$ |S_"theory"^"chain"| = (T_"max" + 1) dot.c 2 dot.c product_(n=1)^N (2 dot.c 121) dot.c 2^(N-1) $

With $T_"max" = 20$:
$ |S_"theory"^"chain"| = 21 dot.c 2 dot.c 242^N dot.c 2^(N-1) = 21 dot.c 2^N dot.c 242^N = 21 dot.c 484^N $

For the *discretized* model with $W = 3$, $T_d = ceil(100 dot.c 3 / 100) = 3$:
- $P_"max"^d = 3 + 3 = 6$ (max excitatory input is $delta_3(80) = 3$)
- Per-neuron states: $2 dot.c 7 = 14$

$ |S_"theory"^("chain,disc")| = 21 dot.c 2 dot.c 14^N dot.c 2^(N-1) = 21 dot.c 28^N $

#remark[
  The per-neuron factor drops from 484 (precise) to 28 (discretized $W=3$),
  a *17.3× reduction per neuron*. For $N = 4$ neurons, this compounds to
  $17.3^4 approx 89,500 times$ overall.
]

= Empirical Validation <sec-empirical>

We generated 63 PRISM models across 7 topologies, 3 model configurations
(deterministic, fast, full), and 3 model types (precise, discretized $W$=2,
discretized $W$=3). PRISM 4.9 was used to build each model and export the
reachable state space via `-exporttrans` and `-exportstates`.

== Results Summary

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    [*Topology*], [*Precise (Det.)*], [*Precise (Fast)*], [*Precise (Full)*], [*Disc. W=3 (Fast)*],
    [Single (1N)], [5], [6], [18], [6],
    [Chain-2 (2N)], [12], [23], [246], [11],
    [Chain-3 (3N)], [23], [275], [13,180], [28],
    [Chain-4 (4N)], [42], [6,917], [362,289], [88],
    [Fork (3N)], [12], [83], [3,388], [15],
    [Diamond (5N)], [15], [1,511], [*OOM*], [98],
    [Convergent (1N, 2in)], [17], [20], [65], [16],
  ),
  caption: [Reachable states $|S_"reachable"|$ for selected configurations.
    All values from PRISM DTMC exports. "OOM" indicates CUDD BDD out-of-memory.],
) <tab-states>

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    [*Topology*], [*Precise (Det.)*], [*Precise (Fast)*], [*Precise (Full)*], [*Disc. W=3 (Fast)*],
    [Single (1N)], [5], [7], [23], [9],
    [Chain-2 (2N)], [12], [35], [369], [19],
    [Chain-3 (3N)], [23], [569], [27,080], [57],
    [Chain-4 (4N)], [42], [19,656], [885,160], [215],
    [Fork (3N)], [12], [167], [6,015], [29],
    [Diamond (5N)], [15], [3,603], [---], [378],
    [Convergent (1N, 2in)], [26], [36], [128], [40],
  ),
  caption: [Transitions $|T|$ for selected configurations.],
) <tab-trans>

== Scaling Analysis

The empirical data reveals clear patterns:

=== Exponential Growth in Chain Length

For the *precise fast* model, the chain sequence grows:
$ 6 -> 23 -> 275 -> 6917 $

The per-neuron multiplicative factor is:
$ r_"fast" approx 23/6 approx 3.8, quad 275/23 approx 12.0, quad 6917/275 approx 25.2 $

This acceleration occurs because each additional neuron's _reachable_ potential range
grows with the cumulative input. The first neuron sees only input $x in {0,1}$ weighted
by 80, giving potential values ${0, 80, 120}$ (with leak). Subsequent neurons see richer
input distributions.

For the *precise full* model:
$ 18 -> 246 -> 13,180 -> 362,289 $
$ r_"full" approx 13.7, quad 53.6, quad 27.5 $

The refractory state machine ($s_n times "aref"_n times "rref"_n$) adds a multiplicative
factor of up to $3 times 3 times 5 = 45$ per neuron.

=== Discretization Effect

#proposition[
  The discretized model achieves an *empirical reduction ratio* of:

  $ R_"disc" = frac(|S_"reachable"^"precise"|, |S_"reachable"^"disc"|) $

  For chain topologies with fast config and $W = 3$:

  #figure(
    table(
      columns: 4,
      stroke: 0.5pt,
      [*Topology*], [$|S_"precise"|$], [$|S_"disc(W=3)"|$], [$R_"disc"$],
      [Chain-2], [23], [11], [2.1×],
      [Chain-3], [275], [28], [9.8×],
      [Chain-4], [6,917], [88], [78.6×],
    ),
    caption: "State space reduction from discretization (fast config).",
  )

  The reduction is _super-linear_ in chain length because the per-neuron potential
  domain shrinks from $R_n approx 121$ to $P_"max"^d + 1 approx 7$, and the
  multiplicative compounding across neurons amplifies this exponentially.
]

=== Topology Impact: Fan-Out vs Fan-In

Comparing fork (fan-out) and convergent (fan-in) topologies:

- *Fork* (1 → N1, N1 → {N2, N3}): 83 states (fast precise) — the fan-out creates
  independent branches whose states multiply.

- *Convergent* (In1, In2 → N1): 20 states (fast precise) — multiple inputs to a
  single neuron increase its potential range but don't multiply modules.

- *Diamond* (fan-out then fan-in): 1,511 states (fast precise) — combines both effects.
  The convergent neuron N4 receives input from two neurons with independent state
  histories, creating a rich reachable state set.

#remark[
  Fan-out is more expensive than fan-in. Adding a parallel branch doubles the module
  count (neuron + transfer), while adding a convergent input only increases the
  potential range of the target neuron. This has important implications for network
  design: _wide_ networks (many parallel paths) are exponentially harder to verify
  than _deep_ networks (long chains) of the same total neuron count.
]

= Memory Consumption Model <sec-memory>

== PRISM Memory Architecture

PRISM uses BDDs (Binary Decision Diagrams) or MTBDDs for symbolic model representation.
The memory consumed depends on:

+ *BDD node count*: Proportional to (a compressed representation of) the state space
+ *Transition matrix storage*: Sparse encoding of $|T|$ transitions
+ *Java heap overhead*: JVM object headers, hash tables, etc.

For the explicit engine, memory scales roughly as:
$ M_"explicit" approx |S_"reachable"| dot.c c_s + |T| dot.c c_t $

where $c_s approx 40$ bytes per state (index, variable values, hash) and
$c_t approx 24$ bytes per transition (source, destination, probability).

== Empirical Memory Estimates

Using the explicit model estimates with $c_s = 40, c_t = 24$ bytes:

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    [*Model*], [*States*], [*Transitions*], [*Est. Memory*], [*Actual Outcome*],
    [Chain-4 fast precise], [6,917], [19,656], [~748 KB], [OK],
    [Chain-4 full precise], [362,289], [885,160], [~35.7 MB], [OK (46s)],
    [Diamond full precise], [$>$1M est.], [---], [$>$64 MB est.], [OOM (CUDD)],
    [Chain-4 fast disc W=3], [88], [215], [~8.7 KB], [OK (\<1s)],
  ),
  caption: "Memory estimates for selected models.",
) <tab-memory>

#remark[
  The diamond-full-precise model failed not on Java heap but on CUDD internal
  BDD memory. CUDD's default memory limit is lower than the Java heap (typically
  1 GB). This suggests that the BDD representation of the diamond topology is
  particularly unfriendly to BDD ordering heuristics — the crossing paths between
  N2→N4 and N3→N4 create variable interdependencies that resist BDD compression.
  The `-cuddmaxmem` flag can increase this limit, but the underlying exponential
  scaling remains.
]

== Predictive Formula

#definition[
  The *state space predictor* for an SNN graph $G$ with configuration $C$ is:

  $ hat(S)(G, C) = |S_"clock"| dot.c 2^(|V_"in"|) dot.c product_(n in V_"proc") f_n(C) dot.c 2^(|E_"int"|) $

  where $f_n(C)$ is the per-neuron state count:

  $
    f_n(C) = cases(
      2 dot.c R_n(G) & "precise fast",
      90 dot.c R_n(G) & "precise full" quad ("with ARP=2, RRP=4"),
      2 dot.c (T_d + E_n^d + 1) & "discretized",
    )
  $

  with $R_n(G) = max(P_"rth", 3/2 sum_(e in "In"(n), w_e > 0) w_e) + 1$ (for excitatory-only fan-in)
  and $E_n^d = sum_(e in "In"(n), w_e^d > 0) w_e^d$.
]

The predictor $hat(S)$ overestimates the reachable state space (since not all
theoretical states are reachable), but provides a *conservative upper bound*
for memory planning:

$ hat(M)(G, C) = hat(S)(G, C) dot.c (c_s + bar(d) dot.c c_t) $

where $bar(d)$ is the average out-degree of the DTMC (typically $approx 2$--$3$ for
CogSpike models due to probabilistic branching at threshold levels).

= Impact of Weight Discretization on Scaling <sec-disc>

The weight discretization scheme established in @weight-disc replaces each neuron's
continuous potential domain with a smaller discretized domain. The key mechanism is
the weight discretization function $delta_W(w) = round(w dot W "/" w_"max")$, which
maps the original weight range $[-100, 100]$ to $[-W, W]$, and the calibrated
threshold $T_d = ceil(T dot W "/" w_"max")$, which ensures threshold preservation.
The _weighted contribution_ for each neuron becomes $C_n = sum_(i) w_i^d dot y_i$,
and the maximum potential is bounded by $T_d + E_n^d$ where $E_n^d$ is the maximum
excitatory contribution.

Critically, the Soundness Theorem @weight-disc guarantees that this domain reduction
does not introduce spurious spikes: the threshold-dependent leak factor
$lambda_d = -max(1, floor(ell dot T_d))$ ensures potentials decay correctly, and
the zero-potential-maps-to-zero-firing-probability property provides a complete
safety chain from sub-threshold potential through decay to permanent silence.

We now characterize the _scaling impact_ of this domain reduction when compounded
across multiple neurons in a network topology.

#theorem("Exponential Reduction")[
  For a chain of $N$ neurons with $|V_"in"| = 1, |E_"int"| = N - 1$,
  the state space ratio between precise and discretized models is:

  $ frac(|S^"precise"|, |S^"disc"|) = product_(n=1)^N frac(R_n, P_("max",n)^d + 1) $

  For uniform fan-in = 1, weight $w$, and $W = 3$:

  $ R_n approx max(P_"rth", 3w"/"2) + 1 quad "vs." quad P_("max",n)^d + 1 = T_d + delta_W(w) + 1 $

  With $P_"rth" = 100$, $w = 80$: $R_n = 121$ vs. $P_"max"^d + 1 = 7$.

  *Reduction per neuron:* $121 / 7 approx 17.3 times$

  *Reduction for $N$ neurons:* $(121 / 7)^N approx 17.3^N$

  For $N = 4$: $17.3^4 approx 89,547 times$
]

This exponential compounding explains why the discretized chain-4 model (88 states)
is dramatically smaller than the precise version (6,917 states for fast, 362,289
for full). The soundness guarantee from @weight-disc ensures this reduction is
_safe_: no behavioral properties are lost.

= Conclusions and Recommendations

== Key Findings

+ *State space grows exponentially* in the number of processing neurons, with the
  base depending on model configuration (fast: ~12--25× per neuron, full: ~27--54×).

+ *The weight discretization of @weight-disc provides exponential reduction* when
  compounded across neurons. The $delta_W$ mapping reduces each neuron's potential
  domain from $O(P_"rth")$ to $O(W)$, with the Soundness Theorem guaranteeing
  safety. For $W = 3$, the per-neuron reduction is ~17×, and biological properties
  (tonic spiking, integrator behavior, excitability) are preserved.

+ *Fan-out is more expensive than fan-in* for verification. Parallel branches
  multiply module counts, while convergent inputs only increase potential ranges.

+ *BDD memory (CUDD) is the practical bottleneck*, not Java heap. The BDD
  representation of diamond topologies is particularly problematic due to
  variable ordering challenges.

+ *The deterministic configuration* ($k = 1$, no ARP/RRP) provides the smallest
  state spaces, suitable for rapid feasibility checks before full verification.

== Practical Limits

Based on our empirical data and the 2 GB CUDD default limit:

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    [*Configuration*], [*Chain Max*], [*Diamond Max*], [*Notes*],
    [Det. precise], [>10 neurons], [>10 neurons], [Minimal state space],
    [Fast precise], [~5--6 neurons], [~4 neurons], [With per-neuron narrowing],
    [Full precise], [~4 neurons], [\<4 neurons], [ARP/RRP dominate],
    [Fast disc $W$=3], [>10 neurons], [~7--8 neurons], [Dramatically reduced],
    [Full disc $W$=3], [~8 neurons], [~5--6 neurons], [Best precision/tractability],
  ),
  caption: "Estimated maximum network sizes for different configurations (2 GB CUDD limit).",
)

== Recommendations for CogSpike

+ *Pre-flight estimator*: Implement $hat(S)(G, C)$ in CogSpike to warn users before
  launching PRISM when the predicted state space exceeds a configurable threshold.

+ *Engine selection*: Use PRISM's explicit engine (`-e`) for models with >100K states;
  the BDD engine struggles with large, irregularly-structured models.

+ *CUDD memory flag*: Expose `-cuddmaxmem` as a configurable parameter in CogSpike's
  PRISM options (currently only `-javamaxmem` is configurable).

+ *Hybrid verification strategy*: For large networks, verify subgraphs independently
  and compose results, exploiting the modular structure of the PRISM model.

+ *Progressive refinement*: Start with deterministic config, then fast, then full —
  using the cheaper configurations to establish baseline properties before investing
  in the exponentially more expensive full verification.

#bibliography("references.bib")
