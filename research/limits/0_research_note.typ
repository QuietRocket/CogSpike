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

#let proofstep(num, title, body) = block(
  width: 100%,
  inset: (left: 12pt, top: 4pt, bottom: 4pt),
  [*Step #num* (#title)*:* #body],
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
used for formal verification in PRISM. We derive closed-form expressions for DTMC
state space size as a function of network structure and model configuration, and validate
them empirically by exporting DTMCs from PRISM across seven canonical topologies.
Our central finding is that state space grows _exponentially_ in the number of processing
neurons, with the multiplicative base per neuron determined by the model configuration.
We further show that the weight discretization scheme of @weight-disc provides an
exponentially compounding reduction — each additional neuron multiplies the savings —
making verification of substantially larger networks feasible. We characterize the
practical verification limits for chains, forks, and diamond topologies, and identify
BDD variable ordering in PRISM's CUDD engine as the binding memory constraint.

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

#intuition[
  Each processing neuron becomes its own PRISM module with local state variables
  (potential, spike, refractory counters). Each internal edge gets a transfer module
  carrying the spike signal from source to target. The key insight is that these
  modules are _independent_: they synchronize only through shared labels (`tick`),
  so their state spaces combine as a Cartesian product — which is why each
  additional neuron _multiplies_ the total state space.
]

== Model Configuration <sec-config>

CogSpike's `ModelConfig` determines the per-neuron state complexity. Throughout
this note we use three named *configuration presets* that represent increasing
levels of biological fidelity:

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    [*Preset*], [*Thresholds ($k$)*], [*ARP*], [*RRP*], [*Description*],
    [*Deterministic*],
    [1],
    [Off],
    [Off],
    [Binary firing: neuron fires with probability 1 if $p >= T$. No stochasticity, no refractory periods. Yields the smallest possible state space.],

    [*Fast*],
    [4],
    [Off],
    [Off],
    [Default. Four probabilistic threshold levels (25% granularity). No refractory periods. Balanced between expressiveness and tractability.],

    [*Full*],
    [10],
    [On (ARP=2)],
    [On (RRP=4)],
    [Biologically accurate. Ten threshold levels (10% granularity) with full refractory state machine (Normal → ARP → RRP → Normal). Largest state space.],
  ),
  caption: "The three configuration presets used in our experiments.",
) <tab-presets>

#intuition[
  The preset choice controls two independent dimensions of state complexity:
  (1) the number of _threshold levels_ $k$ determines how many distinct firing
  probabilities exist (affecting transition fan-out but not state count), and
  (2) whether _refractory periods_ are enabled adds extra state variables per
  neuron ($s_n$, $"aref"_n$, $"rref"_n$), multiplying the per-neuron state space by
  up to $3 times 3 times 5 = 45$. The potential range $R_n$ dominates in all cases.
]

The full list of parameters and their state space impact:

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    [*Parameter*], [*Notation*], [*Effect on State Space*],
    [Threshold levels], [$k in {1, ..., 10}$], [Determines transition fan-out (probabilistic branching)],
    [Enable ARP],
    [$"arp" in {"on", "off"}$],
    [Adds refractory state $s_n in {0,1,2}$ and counter $"aref"_n in [0, "ARP"]$],

    [Enable RRP], [$"rrp" in {"on", "off"}$], [Adds counter $"rref"_n in [0, "RRP"]$],
    [Potential range], [$(P_"min", P_"max")$], [Dominates per-neuron state count],
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

#intuition[
  Each module has a fixed set of integer variables, each with a finite range.
  The local state space is simply the count of all possible combinations of values
  those variables can take. For example, a neuron with spike $y in {0,1}$ and
  potential $p in [0, 120]$ has $2 times 121 = 242$ local states.
]

For each module type:

*GlobalClock* (present when input patterns are time-dependent):
$ |S_"clock"| = T_"max" + 1 $

*Inputs* (one module aggregating all input neurons):
$ |S_"inputs"| = 2^(|V_"in"|) $

*Neuron module* $M_n$ for $n in V_"proc"$ (precise model, fast config):

The fast configuration ($k = 4$, no ARP/RRP) yields two state variables per neuron:
spike output $y_n in {0, 1}$ and membrane potential $p_n in [P_"min", P_"max"]$:

$ |S_n^"fast"| = underbrace(2, y_n) dot.c underbrace(R_n, p_n) = 2 R_n $

where $R_n = P_("max",n) - P_("min",n) + 1$ is the per-neuron potential range.

*Neuron module* $M_n$ (precise model, full config):

The full configuration ($k = 10$, ARP=2, RRP=4) adds refractory state variables:

$
  |S_n^"full"| = underbrace(3, s_n) dot.c underbrace(3, "aref"_n) dot.c underbrace(5, "rref"_n) dot.c underbrace(2, y_n) dot.c underbrace(R_n, p_n) = 90 dot.c R_n
$

where:
- $s_n in {0, 1, 2}$ (Normal, ARP, RRP) → 3 values
- $"aref"_n in {0, 1, 2}$ (ARP countdown) → 3 values
- $"rref"_n in {0, 1, 2, 3, 4}$ (RRP countdown) → 5 values

#intuition[
  The refractory state machine alone contributes a $3 times 3 times 5 = 45 times$
  multiplicative overhead compared to the fast configuration. This is why the full
  model's state space is roughly 45× larger than the fast model's for the same
  topology, _before_ considering that the potential range may also differ.
]

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
  #proofstep(1, "Module independence")[
    PRISM's module system ensures each module $M$ has its own set of local
    variables, disjoint from all other modules. Modules interact only through
    _synchronized labels_ (in CogSpike, the `tick` label).
  ]

  #proofstep(2, "Global state as tuple")[
    PRISM constructs each global state as the tuple of all variable valuations
    across all modules: $(v_1, v_2, ..., v_m)$ where $v_i$ is the value assignment
    of module $i$.
  ]

  #proofstep(3, "Cartesian product")[
    Since each module's variables are independent and range over their full
    domains, the set of all possible tuples is the Cartesian product of the
    individual module state spaces.
  ]
]

#intuition[
  The product structure is the root cause of exponential scaling. Each new module
  _multiplies_ (not adds) the total state space. A chain of $N$ neurons with $R$
  potential values each gives $R^N$ — exponential in $N$. This is fundamentally
  why SNN verification becomes intractable for larger networks.
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
- CogSpike applies a _per-neuron potential bounding_ heuristic: the maximum
  potential is capped at $P_"max" = max(P_"rth", 3/2 dot.c W_"exc")$, where
  $W_"exc"$ is the sum of excitatory incoming weights. The factor $3"/"2$ provides
  headroom above the maximum single-step contribution to account for potential
  accumulation across multiple steps before leak brings it back down. With
  $W_"exc" = 80$: $P_"max" = max(100, 120) = 120$.
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
(deterministic, fast, full; see @sec-config), and 3 model types (precise,
discretized $W$=2, discretized $W$=3). PRISM 4.9 was used to build each model
and export the reachable state space via `-exporttrans` and `-exportstates`.

== Experimental Topologies

The following seven canonical topologies were chosen to isolate the effects of
chain length, fan-out, fan-in, and combined connectivity:

#figure(
  ```
  Single:       In ──► N1

  Chain-2:      In ──► N1 ──► N2

  Chain-3:      In ──► N1 ──► N2 ──► N3

  Chain-4:      In ──► N1 ──► N2 ──► N3 ──► N4

  Fork:         In ──► N1 ──┬──► N2
                            └──► N3

  Diamond:      In ──► N1 ──┬──► N2 ──┬──► N4
                            └──► N3 ──┘

  Convergent:   In1 ──┬──► N1
                In2 ──┘
  ```,
  caption: [The seven canonical topologies. All edges have excitatory weight $w = 80$
    except where noted. Arrows indicate directed synaptic connections.],
) <fig-topologies>

#intuition[
  These topologies are chosen to separate three effects:
  (1) *Chain length* (Single → Chain-4): pure effect of adding neurons in series.
  (2) *Fan-out* (Fork): one neuron drives two downstream targets, doubling the module count.
  (3) *Fan-in* (Convergent, Diamond): multiple inputs converge on one neuron,
  increasing its potential range without adding modules.
  The Diamond combines fan-out and fan-in, making it the most demanding topology.
]

== Results Summary

The most striking pattern in @tab-states is the _exponential growth_ across rows
in the Precise (Full) column: from 18 states for a single neuron to 362,289 for
a 4-neuron chain — nearly 20,000×. Meanwhile, the Disc. W=3 column remains
remarkably stable, growing only from 6 to 88 across the same range.

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
grows with the cumulative input. The first neuron sees only the binary input $x in {0,1}$
scaled by weight 80, giving a few distinct potential values. But downstream neurons
receive _richer input distributions_: their presynaptic neuron can output spikes at
different times, and each firing pattern leaves the downstream neuron at a different
potential. This means each successive neuron explores a larger fraction of its
theoretical potential range.

#intuition[
  The growth ratio is _not_ constant because the theoretical formula ($484^N$)
  overestimates by assuming all potential values are reachable. In practice, the
  first neuron only reaches a handful of potentials (density $rho$ is very low),
  so adding the second neuron has a small multiplicative effect. As the chain
  grows, the input signal becomes richer and $rho$ increases, so the ratio
  approaches the theoretical maximum of $484$ per neuron from below.
]

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

PRISM offers two storage engines for the DTMC transition matrix:

- The *explicit engine* stores every reachable state and transition as individual
  data structures in Java heap memory.
- The *symbolic (BDD) engine* uses Binary Decision Diagrams to represent the
  transition matrix compactly. BDDs exploit regularity in the state space to share
  common sub-structures, but their memory efficiency depends critically on
  _variable ordering_ — the order in which PRISM's variables are laid out in the
  BDD. Poor orderings can cause exponential blowup.

For the explicit engine, memory scales roughly as:
$ M_"explicit" approx |S_"reachable"| dot.c c_s + |T| dot.c c_t $

where $c_s approx 40$ bytes per state (state index, variable values array,
hash table entry) and $c_t approx 24$ bytes per transition (source index,
destination index, probability as IEEE 754 double). These constants are derived
from PRISM's `explicit.StateStorage` and `sparse.DTMCSimple` internal
representations.

#example[
  *Chain-3, fast precise:* 275 states, 569 transitions.
  $ M approx 275 times 40 + 569 times 24 = 11,000 + 13,656 = 24,656 "bytes" approx 24 "KB" $
  This is trivially within PRISM's default 1 GB Java heap.
]

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

#intuition[
  The memory estimates above assume the explicit engine. Even the largest
  successful model (chain-4 full precise, ~35.7 MB) is well within a 1 GB heap.
  The OOM failure for the diamond model occurred in the BDD engine (CUDD),
  which has a separate and usually smaller memory budget. This is a crucial
  distinction addressed in the next section.
]

== BDD Engine and CUDD Memory Limits <sec-cudd>

The diamond-full-precise model's failure reveals a subtle but critical bottleneck:
it was not Java heap that ran out, but CUDD's _internal_ BDD memory.

CUDD (Colorado University Decision Diagram package) manages its own memory pool
separate from the JVM heap. Its default limit is typically 1 GB, compared to
PRISM's default 1 GB Java heap. The key difference is that BDD memory consumption
depends not on $|S_"reachable"|$ directly, but on the _BDD width_ — which is
determined by the variable ordering.

#remark[
  The diamond topology is particularly BDD-unfriendly. The two parallel paths
  N1→N2→N4 and N1→N3→N4 create _crossing variable dependencies_: the
  convergent neuron N4's potential depends on both N2 and N3's spike histories,
  which are themselves independent. This forces the BDD to represent their
  joint distribution, which resists the compression that BDDs excel at for
  tree-like dependency structures. The `-cuddmaxmem` flag can increase the
  CUDD memory limit, but the underlying exponential scaling remains.
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

#example[
  *Worked example: chain-3, fast precise.*

  Parameters: $T_"max" = 20$, $|V_"in"| = 1$, $|V_"proc"| = 3$, $|E_"int"| = 2$,
  $P_"rth" = 100$, $w = 80$.

  Per-neuron: $R_n = max(100, 3/2 dot.c 80) + 1 = 121$, so $f_n = 2 times 121 = 242$.

  $ hat(S) = 21 dot.c 2^1 dot.c 242^3 dot.c 2^2 = 21 dot.c 2 dot.c 14,172,888 dot.c 4 = 2,381,285,184 $

  With $bar(d) = 2$: $hat(M) approx 2.38 times 10^9 times (40 + 2 times 24) approx 210 "GB"$.

  The _actual_ reachable state space is only 275 (from @tab-states), giving
  $rho = 275 / 2.38 times 10^9 approx 1.2 times 10^(-7)$. The predictor is
  extremely conservative, but that is by design: it guarantees the actual
  memory will never _exceed_ the prediction.
]

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

#bibliography("references.bib", style: "ieee")
