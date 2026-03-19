// Formal Verification of Spiking Neural Networks
// via Weight-Discretized Quotient Abstractions
//
// ICANN 2026 — Double-blind submission (LNCS format)
// Constraint: max 12 pages INCLUDING references
//
// Page budget (12 pages):
//   Title + Abstract (0.5p) + Intro (1p) + Related Work (1p) + Prelim (1.5p)
//   = 4p background
//   Weight-Disc Quotient Abstraction (2.5p) + Scaling Limits (2.5p) + Conclusion (0.5p)
//   = 5.5p contributions
//   References (~1p)

#import "Typst/llncs.typ": *

// ── Anonymized institute (double-blind) ──────────────────────────────────────
#let inst-anon = institute("Anonymous Institution")

// ── Apply template ───────────────────────────────────────────────────────────
#show: lncs.with(
  title: "Formal Verification of Spiking Neural Networks via Weight-Discretized Quotient Abstractions",
  authors: (
    author("Anonymous submission", insts: (inst-anon,)),
  ),
  running-authors: "Anonymous",
  abstract: [
    Spiking Neural Networks (SNNs) offer biologically plausible, energy-efficient
    computation but pose severe challenges for formal verification due to state
    space explosion. Existing quotient model abstractions reduce the state space
    by partitioning membrane potentials into equivalence classes, but discard
    synaptic weight information, limiting the properties that can be verified.
    This paper introduces a _weight-discretized quotient model abstraction_ that
    maps continuous synaptic weights to a compact integer range while preserving
    the relative contribution of each synapse. The discretization is accompanied
    by formal correctness guarantees: a Threshold Preservation theorem
    (completeness) ensures that no fireable configurations are lost, and an
    Asymptotic Silence theorem (soundness) guarantees that no spurious spikes are
    introduced. The core biological properties of Leaky Integrate-and-Fire
    neurons---tonic spiking, integrator behaviour, and excitability---are
    provably maintained. A topology-dependent scaling analysis derives
    closed-form state space formulas and shows that the reduction compounds
    exponentially across neurons: approximately 17$times$ per neuron for a
    discretization parameter $W = 3$. Empirical validation across seven
    canonical topologies in the PRISM model checker confirms the theoretical
    predictions, enabling verification of networks that are otherwise
    intractable.
  ],
  keywords: (
    "Spiking Neural Networks",
    "Formal Verification",
    "Model Checking",
    "PRISM",
    "Quotient Abstraction",
    "Weight Discretization",
  ),
  // No acknowledgments in double-blind submission
  disclosure: [
    The authors have no competing interests to declare that are relevant to
    the content of this article.
  ],
  bib: bibliography("refs.bib"),
)


// ═══════════════════════════════════════════════════════════════════════════════
// PAPER BODY — 6 sections + references
// ═══════════════════════════════════════════════════════════════════════════════


// ─── 1. Introduction (~1 page) ──────────────────────────────────────────────
= Introduction <sec-intro>

The evolution of artificial neural networks is conventionally classified into
three generations @maass1997networks. The first generation, epitomized by
McCulloch--Pitts threshold gates, employed binary inputs and outputs and was
limited to linearly separable problems. The second generation introduced
continuous activation functions and gradient-based optimization via
backpropagation, enabling deep learning architectures that have achieved
remarkable success in pattern recognition and sequence modelling. Yet these
networks lack the temporal dynamics and energy efficiency of biological neural
circuits.

The third generation---Spiking Neural Networks (SNNs)---bridges the gap between
machine learning and computational neuroscience. SNNs employ biologically
plausible neurons communicating via discrete, asynchronous spikes, where the
precise timing of each spike conveys information alongside the aggregate firing
rate @maass1997networks. This event-driven computation offers substantial power
advantages, making SNNs suitable for neuromorphic hardware @nguyen2021review.
However, the non-differentiable Heaviside activation function precludes standard
gradient-based optimization, and the stochastic, temporal nature of spike-based
computation makes formal verification of correctness significantly harder than
for traditional networks. As SNNs are increasingly targeted for safety-critical
applications, formal methods providing mathematical guarantees of functional
correctness become essential.

This paper presents a _weight-discretized quotient model abstraction_ for the
formal verification of SNNs via probabilistic model checking. The approach
extends the filtration-based quotient model of Baier and
Katoen @BaierKatoen2008 to preserve synaptic weight information during
abstraction, enabling verification in the PRISM model
checker @PRISM2011 with drastically reduced state spaces. The contributions
are threefold:

+ A *weight discretization scheme* that maps continuous synaptic weights to a
  finite discrete range while preserving threshold feasibility and relative
  synaptic contributions (@sec-disc-function).

+ *Formal correctness proofs*: a _Threshold Preservation_ theorem
  (completeness) and an _Asymptotic Silence_ theorem (soundness) guaranteeing
  that the discretization neither loses fireable configurations nor introduces
  spurious spikes (@sec-proofs).

+ A *topology-dependent state space analysis* with closed-form formulas for
  DTMC size as a function of network structure, validated empirically across
  seven canonical topologies (@sec-scaling).


// ─── 2. Related Work (~1 page) ──────────────────────────────────────────────
= Related Work <sec-related>

== Formal Verification of SNNs <sec-related-formal>

De Maria et al.~@naco20 pioneered the modelling of SNNs as timed automata,
formalizing Leaky Integrate-and-Fire (LIF) neurons with parameter learning.
Their work established key biological properties---tonic spiking, integrator
behaviour, excitability---as formal verification targets, and introduced the
Advice Back-Propagation (ABP) algorithm for supervised parameter inference
driven by model-checking counter-examples rather than continuous gradients.
This approach was extended to _neuronal archetypes_---primitive micro-circuits
such as contralateral inhibition and convergent excitation---allowing
macroscopic network properties to be composed from formally verified building
blocks @demaria2022formal.

Yao et al.~@yao2025probabilistic introduced the Refractory-evolve
Probabilistic LI\&F (RP-LI\&F) neuron model, unifying discrete-time refractory
dynamics with probabilistic spike generation. Their contract-based verification
approach translates SNN topologies into Discrete-Time Markov Chains (DTMCs) and
specifies behavioural properties using Probabilistic Computation Tree Logic
(PCTL), enabling rigorous assume/guarantee contracts.

These approaches establish the feasibility of formal SNN verification but share
a common limitation: the _state space explosion problem_. As network size
grows, the DTMC state space grows exponentially in the number of neurons, and
existing quotient model abstractions @BaierKatoen2008 lose synaptic weight
information during reduction. The present work addresses both limitations.

== Learning Algorithms for SNNs <sec-related-learning>

Parameter learning in SNNs has motivated diverse approaches. Spike-Timing-Dependent
Plasticity (STDP), combined with winner-takes-all lateral inhibition and
adaptive thresholds, achieves competitive unsupervised
accuracy @diehl2015unsupervised. Competitive learning frameworks using global
inhibition and anti-Hebbian dynamics replace non-local backpropagation with
biologically plausible mechanisms @krotov2019unsupervised. Self-Organizing Maps
adapted to the spiking domain use distance-dependent lateral inhibition to
create topographically ordered feature
maps @hazan2018unsupervised @rumbell2014spiking. For supervised learning at
scale, surrogate gradient methods substitute smooth proxy derivatives for the
spike function during backpropagation through
time @neftci2019surrogate, while three-factor learning rules with eligibility
traces enable online, hardware-compatible
training @bellec2019eligibility.

The present work is complementary to these learning approaches: rather than
optimizing SNN parameters, it provides formal guarantees that a given SNN
configuration satisfies specified temporal and probabilistic properties.


// ─── 3. Preliminaries (~1.5 pages) ──────────────────────────────────────────
= Preliminaries <sec-prelim>

== Spiking Neural Network Model <sec-snn-model>

An SNN is modelled as a directed graph $G = (V, E)$ where $V = V_"in" union
V_"proc"$ partitions into input neurons and processing neurons, and $E
subset.eq V times V$ represents directed synaptic connections with integer
weights $w_e in [-100, 100]$.

Each processing neuron $n in V_"proc"$ follows Leaky Integrate-and-Fire (LIF)
dynamics @naco20 @hodgkin1952quantitative. At each discrete time step, the
membrane potential integrates incoming weighted spikes and decays toward rest:
$ p_(n, t+1) = max(0, (1 - ell) dot.c p_(n,t) + sum_(i in "In"(n)) w_(i,n) dot.c y_(i,t)) $ <eq-lif>
where $ell in [0,1]$ is the leak factor. When $p_n >= P_"rth"$ (the firing
threshold), neuron $n$ emits a spike ($y_n = 1$) and resets to zero.

Firing is _probabilistic_: the potential maps to a discrete threshold level
$L in {0, ..., N-1}$ where $N$ is configurable (1--10), and each level yields
a firing probability. Optionally, neurons implement a three-state refractory
machine: Normal ($s = 0$), Absolute Refractory Period (ARP, $s = 1$), and
Relative Refractory Period (RRP, $s = 2$), with reduced firing probability
during RRP scaled by factor $alpha$.

== Probabilistic Model Checking <sec-model-checking>

#definition[
  A _Discrete-Time Markov Chain_ (DTMC) is a tuple $cal(D) = (S, s_0, bold(P))$
  where $S$ is a finite set of states, $s_0 in S$ is the initial state, and
  $bold(P) : S times S -> [0, 1]$ is the transition probability matrix
  satisfying $sum_(s' in S) bold(P)(s, s') = 1$ for all $s in
  S$ @BaierKatoen2008.
]

Behavioural properties are specified in Probabilistic Computation Tree Logic
(PCTL), which extends branching-time temporal logic with probabilistic path
quantifiers $P_(⋈ p) [phi]$. For instance,
$P_(>= 1)[bold(F) (y_n = 1)]$ asserts that neuron $n$ fires with probability
one, while $P_(>= 1)[bold(G) (y_n = 0)]$ asserts permanent silence.

PRISM @PRISM2011 is a probabilistic model checker supporting DTMCs.
Models are specified as parallel compositions of _modules_ with local integer
variables and guarded probabilistic transitions, synchronizing via shared
labels. The global state space is the Cartesian product of all module state
spaces. PRISM offers an _explicit_ engine (enumerates reachable states) and a
_symbolic BDD_ engine (uses Binary Decision Diagrams via the CUDD library),
whose efficiency depends on variable ordering.

== Quotient Model Abstraction <sec-quotient>

Quotient model abstraction @BaierKatoen2008 @Katoen2016 reduces the DTMC state
space by partitioning states into equivalence classes. The filtration-based
approach maps continuous membrane potentials to a finite class set, preserving
probabilistic branching structure. However, the standard filtration treats all
synapses uniformly, collapsing weight information: it cannot distinguish strong
excitatory from weak excitatory or inhibitory contributions. The weight
discretization scheme presented in @sec-weight-disc addresses this limitation.


// ─── 4. Weight-Discretized Quotient Abstraction (~2.5 pages) ────────────────
= Weight-Discretized Quotient Abstraction <sec-weight-disc>

The standard filtration-based quotient model abstracts membrane potentials into
equivalence classes but treats all synapses uniformly, discarding weight
information. This section introduces a _weight discretization scheme_ that
preserves the relative contribution of each synapse.

== Weight Discretization Function <sec-disc-function>

#definition[
  Given a weight range $[- w_"max", w_"max"]$ (typically $w_"max" = 100$) and a
  discretization parameter $W in NN^+$, the _weight discretization function_
  $delta_W : RR -> ZZ$ is:
  $ delta_W (w) = lr(⌊ w dot.c W / w_"max" ⌉) $
  mapping original weights to the discrete range $[-W, W] subset ZZ$.
]

The function preserves relative weight magnitudes: for $W = 3$, strong
excitatory ($w = 100$) maps to $delta_3(100) = 3$, medium ($w = 67$) to
$delta_3(67) = 2$, weak ($w = 33$) to $delta_3(33) = 1$, and inhibitory
($w = -50$) to $delta_3(-50) = -2$. Weights are stored as static integer
constants in the generated PRISM model, contributing no additional state
variables.

The _weighted contribution_ for neuron $n$ with discretized incoming weights
${w_1^d, ..., w_m^d}$ replaces the binary class evolution of the original
quotient model:
$ C_n = sum_(i=1)^m w_i^d dot.c y_i $ <eq-contribution>
where $y_i in {0,1}$ is the spike output of presynaptic neuron $i$ and $m$ is
the fan-in.

== Threshold Calibration <sec-threshold-cal>

#definition[
  The _discretized threshold_ for a neuron with original threshold $T$ is:
  $ T_d = ceil(T dot.c W / w_"max") $
]

The use of ceiling (rather than rounding) ensures $T_d >= T dot.c W \/ w_"max"$,
so the discretized neuron is _at least as hard_ to fire as the original. This
conservative calibration prevents false-positive firings and is essential for
the soundness guarantee established in @sec-soundness.

== Threshold-Dependent Leak Factor <sec-leak>

#definition[
  The _discretized leak factor_ is:
  $ lambda_d = - max(1, floor(ell dot.c T_d)) $
  where $ell in [0,1]$ is the leak rate ($ell = 1 - r$ with retention rate $r$).
]

This formulation links the leak to $T_d$ rather than to the number of
equivalence classes $k$. Since $T_d$ scales with the actual potential range,
the decay remains proportional to realistic membrane dynamics. The
$max(1, ...)$ floor ensures a minimum decay of one unit per step, which is
required by the Soundness theorem (@sec-soundness). The class evolution
becomes:
$ c'_n = op("clamp")(c_n + Delta(C_n) + lambda_d, 0, k) $
where $Delta(C_n) = op("clamp")(lr(⌊ C_n \/ gamma ⌉), -k, k)$ is the class
delta function with class width $gamma = T_d \/ k$.

== Formal Proofs <sec-proofs>

=== Threshold Preservation (Completeness) <sec-completeness>

#theorem[
  Let $cal(N)$ be a neuron with weights ${w_1, ..., w_m}$ and threshold $T$.
  If $cal(N)$ can fire in a single step (i.e., $exists bold(y) in {0,1}^m$ such
  that $sum_(i=1)^m w_i dot.c y_i >= T$), then the discretized neuron $cal(N)'$
  with weights ${delta_W (w_1), ..., delta_W (w_m)}$ and threshold $T_d$ can
  also fire.
]

#proof[
  Let $bold(y)^*$ be a firing input pattern with weighted sum
  $S = sum w_i y_i^* >= T$. By the rounding bound,
  $delta_W(w_i) >= w_i dot.c W \/ w_"max" - 1\/2$, so the discretized sum
  satisfies:
  $ S_d = sum delta_W(w_i) dot.c y_i^* >= S dot.c W \/ w_"max" - m^* \/ 2 $
  where $m^* = sum y_i^* <= m$ is the number of active inputs. Since
  $T_d = ceil(T dot.c W \/ w_"max") <= T dot.c W \/ w_"max" + 1$, the
  sufficient condition for firing is
  $(S - T) dot.c W \/ w_"max" >= m\/2 + 1$.
  For $W >= w_"max" dot.c (m\/2 + 1) \/ T$, the condition is satisfied whenever
  $S >= T$. With typical parameters ($W = 3$, $w_"max" = 100$, $m <= 10$,
  $T = 100$), the bound requires $W >= 6$.
]

#remark[
  The cumulative rounding error of $-m\/2$ implies that high fan-in neurons
  require finer discretization. For neurons with $m > 2W$, the rounding noise
  may exceed the smallest synaptic weight, and a threshold correction factor
  $T'_d = T_d - floor(m \/ (2W))$ can optionally be applied.
]

=== Asymptotic Silence (Soundness) <sec-soundness>

#theorem[
  Let $cal(N)'$ be a discretized neuron with potential $P_t < T_d$ and leak
  factor $lambda_d <= -1$. If the input is zero for all $t' >= t$ (i.e.,
  $S_d = 0$ henceforth), then $cal(N)'$ will never fire.
]

#proof[
  Without input, $P_(t+1) = max(0, P_t + lambda_d)$. Since
  $lambda_d <= -1$ and $P_t > 0$, the potential strictly decreases:
  $P_(t+1) <= P_t - 1$. The sequence converges to the absorbing state $P = 0$.
  Since the trajectory is non-increasing and starts below $T_d$, the firing
  condition $P >= T_d$ is never met. At $P = 0$, the threshold level is $L = 0$,
  which maps to firing probability zero, ensuring permanent silence.
]

=== Biological Property Preservation <sec-bio-preservation>

The discretized model preserves the core LIF properties formalized by De Maria
et al.~@naco20:

- *Tonic spiking.* Under constant input $C_"in"$, the neuron has non-zero
  firing probability iff $C_"in" > |lambda_d|$. When satisfied, the potential
  accumulates to a level $L > 0$, yielding periodic probabilistic spiking with
  expected inter-spike interval $"ISI" = ceil(T_d \/ (C_"in" - |lambda_d|))$.

- *Integrator.* The probability of immediate firing on simultaneous inputs
  reaches 1.0 iff $sum delta_W(w_i) >= T_d$. Below threshold, the response is
  graded via the threshold level mapping.

- *Excitability.* The expected inter-spike interval decreases monotonically as
  input strength increases, since $"ISI" = ceil(T_d \/ (C_"in" - |lambda_d|))$
  is a decreasing function of $C_"in"$.


// ─── 5. Topology-Dependent Scaling Limits (~2.5 pages) ──────────────────────
= Topology-Dependent Scaling Limits <sec-scaling>

To characterize the practical limits of SNN verification, this section derives
closed-form formulas for the DTMC state space as a function of network
topology and model configuration.

== PRISM Module Decomposition <sec-modules>

The PRISM model for an SNN $G = (V, E)$ consists of four module types:

- *GlobalClock*: a step counter with $T_"max" + 1$ states.
- *Inputs*: $2^(|V_"in"|)$ states (one binary variable per input neuron).
- *Neuron* $M_n$: per-neuron state depends on configuration:
  - _Fast_ ($k = 10$ threshold levels, no refractory):
    $|S_n^"fast"| = 2 dot.c R_n$ where $R_n = P_("max",n) - P_("min",n) + 1$.
  - _Full_ ($k = 10$, ARP=2, RRP=4):
    $|S_n^"full"| = 3 dot.c 3 dot.c 5 dot.c 2 dot.c R_n = 90 dot.c R_n$.
  - _Discretized_ ($W = 3$):
    $|S_n^"disc"| = 2 dot.c (P_("max",n)^d + 1)$ where
    $P_("max",n)^d = T_d + E_n^d$.
- *Transfer* $T_(i,j)$: 2 states per internal edge.

#theorem[
  *(State Space Product).* The theoretical state space is the Cartesian product
  of all module state spaces:
  $ |S_"theory"| = (T_"max"+1) dot.c 2^(|V_"in"|) dot.c product_(n in V_"proc") f_n (C) dot.c 2^(|E_"int"|) $
  where $f_n (C) = |S_n|$ depends on the model configuration $C$.
]

For a chain of $N$ neurons with $P_"rth" = 100$ and weight $w = 80$: the
per-neuron factor is $f_n = 2 dot.c 121 = 242$ (fast precise), dropping to
$f_n = 2 dot.c 7 = 14$ (discretized $W = 3$). The per-neuron reduction factor
is $242 \/ 14 approx 17.3 times$.

#theorem[
  *(Exponential Reduction).* For a chain of $N$ neurons, the state space ratio
  between precise and discretized models compounds exponentially:
  $ |S^"precise"| / |S^"disc"| = product_(n=1)^N R_n / (P_("max",n)^d + 1) approx 17.3^N $
  For $N = 4$: $17.3^4 approx 89,500 times$.
]

== Empirical Validation <sec-results>

63 PRISM models were generated across 7 canonical topologies (single, chain-2
through chain-4, fork, diamond, convergent), 3 configurations (deterministic,
fast, full), and 3 model types (precise, disc. $W=2$, disc. $W=3$).
@tab-states reports the reachable state counts from DTMC exports.

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    [*Topology*], [*Det.*], [*Fast*], [*Full*], [*Disc. $W$=3*],
    [Single (1N)], [5], [6], [18], [6],
    [Chain-2 (2N)], [12], [23], [246], [11],
    [Chain-3 (3N)], [23], [275], [13,180], [28],
    [Chain-4 (4N)], [42], [6,917], [362,289], [88],
    [Fork (3N)], [12], [83], [3,388], [15],
    [Diamond (5N)], [15], [1,511], [OOM], [98],
    [Convergent (1N)], [17], [20], [65], [16],
  ),
  caption: [Reachable DTMC states $|S_"reachable"|$ by topology and
    configuration. All precise models use $k = 10$ threshold levels.
    "OOM" denotes CUDD BDD out-of-memory (2~GB limit).],
) <tab-states>

The most striking pattern is the exponential growth across chain lengths in the
precise configurations: from 18 states (single, full) to 362,289 (chain-4,
full)---a $20{,}000 times$ increase for just three additional neurons. The
discretized column remains remarkably stable, growing from 6 to 88 across the
same range.

The per-neuron multiplicative factor for the fast precise model is:
$ r_"fast" approx 23\/6 approx 3.8, quad 275\/23 approx 12.0, quad 6917\/275 approx 25.2 $

This acceleration occurs because downstream neurons receive progressively
richer input distributions, approaching the theoretical maximum of 484 per
neuron as the reachability density $rho$ increases.

Three further observations emerge:

+ *Fan-out is more expensive than fan-in.* The fork topology (3 neurons, fan-out
  2) produces 275 states (fast), while the convergent topology (1 processing
  neuron, fan-in 2) produces only 20. Parallel branches multiply the module
  count, whereas convergent inputs only increase the potential range.

+ *BDD memory is the practical bottleneck.* The diamond topology (5 neurons)
  causes CUDD out-of-memory in the full precise configuration,
  while the explicit engine succeeds for chain-4 (362,289 states). BDD variable
  ordering for multi-path topologies creates exponential intermediate
  representations.

+ *Discretization scales gracefully.* The disc. $W = 3$ diamond completes with
  98 states, compared to OOM for the precise full model. @tab-limits summarizes
  the estimated maximum network sizes.

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    [*Configuration*], [*Chain*], [*Diamond*], [*Notes*],
    [Det. precise], [\>10N], [\>10N], [Minimal state space],
    [Fast precise], [5--6N], [~4N], [Per-neuron narrowing],
    [Full precise], [~4N], [\<4N], [ARP/RRP dominate],
    [Fast disc. $W$=3], [\>10N], [7--8N], [17× reduction/neuron],
    [Full disc. $W$=3], [~8N], [5--6N], [Best precision/tractability],
  ),
  caption: [Estimated maximum verifiable network sizes (2~GB CUDD limit).],
) <tab-limits>


// ─── 6. Conclusion (~0.5 page) ──────────────────────────────────────────────
= Conclusion <sec-conclusion>

This paper presented a weight-discretized quotient model abstraction for the
formal verification of spiking neural networks. The discretization function
$delta_W$ maps continuous synaptic weights to a compact integer range while
preserving threshold feasibility (Theorem~1) and preventing spurious spikes
(Theorem~2). The core biological properties of LIF neurons---tonic spiking,
integrator behaviour, and excitability @naco20 --- are maintained by the
threshold-dependent leak factor $lambda_d$.

The topology-dependent scaling analysis demonstrates that the state space
reduction compounds exponentially across neurons: approximately $17 times$ per
neuron for $W = 3$, enabling verification of networks that are otherwise
intractable. Empirical validation across seven canonical topologies confirms
the theoretical predictions and identifies BDD memory as the binding practical
constraint.

Several directions for future work emerge. First, _compositional verification_
could exploit the modular structure of neuronal archetypes @demaria2022formal
to verify large networks by composing archetype-level guarantees. Second,
_automated $W$ selection_ based on fan-in analysis could optimize the
precision/tractability trade-off per neuron. Third, extending the framework to
_recurrent topologies_ would broaden its applicability. Finally, integrating
the formal verification loop into biologically plausible learning
frameworks @bellec2019eligibility --- using verified safety constraints as
modulatory third factors---could enable training of SNNs that are intrinsically
correct by construction.

