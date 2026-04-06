// Formal Verification of Spiking Neural Networks
// via Weight-Discretized Quotient Abstractions
//
// ICANN 2026 — Double-blind submission (LNCS format)
// Constraint: max 12 pages INCLUDING references
//
// Page budget (12 pages):
//   Title + Abstract (0.5p) + Intro (1p) + Related Work (0.5p) + Prelim (1.5p)
//   = 3.5p background
//   Weight-Disc Quotient Abstraction (2.5p) + CogSpike (0.75p) + Case Study (2p) + Conclusion (0.5p)
//   = 5.75p contributions
//   References (~1p)

#import "llncs.typ": *

// ── Anonymized institute (double-blind) ──────────────────────────────────────
#let inst-anon = institute("Anonymous Institution")

// ── Apply template ───────────────────────────────────────────────────────────
#show: lncs.with(
  title: "A tool for Formal Verification of\n Probabilistic Spiking Neural Networks\nvia Weight-Discretized Quotient Abstractions",
  authors: (
    author("Anonymous submission", insts: (inst-anon,)),
  ),
  running-title: "Formal Verification of SNNs via Weight-Discretized Quotient Abstractions",
  running-authors: "Anonymous",
  abstract: [
    Spiking Neural Networks (SNNs) are known for modelling biological neural dynamics more
    faithfully than classical artificial networks. Introducing stochastic,
    event-driven computation in SNNs demands probabilistic models for which
    deterministic abstractions are mathematically inadequate. Formal
    verification of such models via probabilistic model checking faces a
    fundamental barrier: the _state space explosion problem_, where the
    underlying transition system grows exponentially with the
    number of neurons. General-purpose quotient model
    abstractions can in principle mitigate this growth by
    partitioning membrane potentials into equivalence classes, but a naïve
    application to SNNs discards synaptic weight information, limiting the
    properties that can be verified. This paper introduces a _weight-discretized quotient model
    abstraction_ that maps continuous synaptic weights to a compact integer
    range while preserving the relative contribution of each synapse, and
    presents CogSpike, a unified workbench that integrates probabilistic SNN design,
    simulation, and PRISM-based formal verification within a single
    tool chain. The discretization is accompanied by formal
    correctness guarantees: a Threshold Preservation theorem
    ensures that no fireable configurations are lost, and an Asymptotic
    Silence theorem guarantees that no spurious spikes are
    introduced. A case study on contralateral inhibition demonstrates the
    full design--simulate--verify workflow and confirms that the
    discretized model preserves Winner-Takes-All dynamics with a
    $9.3 times$ state space reduction.
  ],
  keywords: (
    "Spiking Neural Networks",
    "Formal Verification",
    "Model Checking",
    "PRISM",
    "Quotient Abstraction",
    "Weight Discretization",
  ),
  bib: bibliography("refs.bib"),
)


// ═══════════════════════════════════════════════════════════════════════════════
// PAPER BODY — 6 sections + references
// ═══════════════════════════════════════════════════════════════════════════════


// ─── 1. Introduction (~1 page) ──────────────────────────────────────────────
= Introduction <sec-intro>

Spiking Neural Networks (SNNs), the third generation of artificial neural
networks @maass1997networks, are modelled as directed graphs whose nodes
represent neurons and whose edges represent synaptic connections that can be
either _excitatory_ (positive weight) or _inhibitory_ (negative weight).
Unlike rate-coded deep networks, SNNs communicate via discrete, asynchronous
spikes whose precise timing encodes information alongside aggregate firing
rates, making them a natural candidate for studying how real neural circuits
process and transmit information. Among SNN neuron models, the Leaky
Integrate-and-Fire (LIF) formulation provides analytical
tractability, while biophysically detailed models such as
Hodgkin--Huxley @hodgkin1952quantitative and computationally efficient
alternatives such as Izhikevich neurons @izhikevich2003simple reproduce a
wider repertoire of biological firing patterns at higher computational cost.

Understanding how the brain computes requires studying the _temporal dynamics_
of neural circuits, that is, how spikes propagate, interact, and give rise to emergent
behaviours in small but functionally relevant network topologies such as
chains, convergent motifs, and recurrent loops. However, modelling these
dynamics demands a delicate balance. On one hand, biological neurons are
inherently stochastic: ion-channel noise, unreliable synaptic vesicle release,
and variable axonal delays introduce randomness at every stage of signal
transmission @hodgkin1952quantitative @nguyen2021review, making probabilistic
models mathematically necessary.

One promising avenue encodes the network as
a Discrete-Time Markov Chain (DTMC): probabilistic model checking can then
determine whether the DTMC satisfies the behavioural properties expected by
neuroscientists, such as tonic spiking under sustained input or silence
without stimulation @naco20. However, this approach faces a fundamental
barrier: the _state space explosion problem_, where the DTMC state space
grows exponentially with network size, rendering verification intractable
beyond a handful of neurons.

This paper addresses that challenge. We propose CogSpike, a unified
tool for probabilistic spiking neural networks that integrates three tightly
coupled capabilities within a single framework: (i)~_simulation_ of probabilistic
LIF-based SNN dynamics, (ii)~_formal modelling_ of the same networks as DTMCs
for the PRISM model checker @PRISM2011, and (iii)~_automated model checking_
of behavioural properties expressed in Probabilistic Computation Tree Logic
(PCTL) @hansson1994logic. The underlying neuron model employs a weight-discretized quotient
abstraction that overcomes the limitations of naïve quotient
models @BaierKatoen2008, which discard synaptic weight information when
partitioning states into equivalence classes.

Concretely, the contributions are threefold:
(1)~a _weight discretization scheme_ that maps continuous synaptic weights to a finite discrete range while preserving threshold feasibility and relative synaptic contributions;// (@sec-disc-function);
(2) a _Threshold Preservation_ theorem ensuring that every input pattern triggering a spike in the original model also triggers one in the discretized model, and an _Asymptotic Silence_ theorem guaranteeing that the discretized model introduces no spurious spikes absent from the original;// (@sec-proofs); and
(3)~*CogSpike*, a unified workbench integrating SNN design, simulation, and formal verification, whose code generator produces a PRISM representation isomorphic to the simulation engine, enabling automated formal modelling and model checking.// (@sec-cogspike).
The complete formal proofs and extended derivations will be made available in a research report once the anonymity requirement is lifted.
The remainder of the paper is organized as follows: @sec-related surveys related work, @sec-prelim introduces the necessary background, @sec-weight-disc presents the weight-discretized quotient abstraction, @sec-cogspike describes the CogSpike workbench, and @sec-casestudy presents a contralateral inhibition case study demonstrating the full design--simulate--verify workflow and the state space reduction achieved by weight discretization.


// ─── 2. Related Work (~1 page) ──────────────────────────────────────────────
= Related Work <sec-related>

Biological neurons exhibit significant trial-to-trial variability even under
identical stimulation @gerstner2002spiking. Classical approaches capture this
stochasticity through three mechanisms: _escape noise_, which introduces a
probabilistic firing threshold; _diffusive noise_, modelling stochastic spike
arrivals via synaptic bombardment; and _slow noise_, which adds fluctuations
to neuronal parameters @gerstner2002spiking. These formulations underpin
large-scale analyses of noisy integrate-and-fire
networks @brunel1999fast, but they operate in continuous state spaces and do
not yield discrete-state models amenable to exhaustive formal verification.
Among existing simulators, Brian~2 @stimberg2019brian2 supports stochastic
firing thresholds via escape noise (SDE integration), while
Nengo @bekolay2014nengo, NEST @gewaltig2007nest, and
BindsNET @hazan2018bindsnet are limited to noise injection at the input level.
Crucially, none of these platforms support probabilistic model checking.

Our model takes a different route: rather than adding continuous noise to
a differential equation, we discretize the membrane potential into threshold
levels and assign each level an explicit firing probability, yielding a
finite-state probabilistic model that maps directly onto a DTMC. This
enables exhaustive formal verification via model checking, a capability
that is fundamentally unavailable with continuous-noise formulations.

Concerning the use of formal verification for Spiking Neural Networks, De Maria et al.~@naco20 pioneered the modelling of SNNs as timed automata,
formalizing Leaky Integrate-and-Fire (LIF) neurons with parameter learning.
Their work established key biological properties, e.g., tonic spiking, integrator
behaviour, and excitability, as formal verification targets, and introduced the
Advice Back-Propagation (ABP) algorithm for supervised parameter inference
driven by model-checking counter-examples rather than continuous gradients.
A formal approach to model and verify _neuronal archetypes_, which are primitive micro-circuits
such as contralateral inhibition or parallel composition, was introduced in @demaria2022formal, allowing
macroscopic network properties to be composed from formally verified building
blocks.
More recently, Yao et al.~@yao2025probabilistic introduced the
Refractory-evolve Probabilistic LI\&F (RP-LI\&F) neuron model, unifying
discrete-time refractory dynamics with probabilistic spike generation. Their
contract-based verification approach translates SNN topologies into
Discrete-Time Markov Chains (DTMCs) and specifies behavioural properties
using Probabilistic Computation Tree Logic (PCTL), enabling rigorous
assume/guarantee contracts.

These approaches establish the feasibility of formal SNN verification but
share a common limitation: the _state space explosion problem_. As network
size grows, the transition system state space grows exponentially, rendering verification
intractable beyond small networks. General-purpose quotient
abstractions @BaierKatoen2008 can mitigate this growth in principle, but
a naïve application to SNNs discards synaptic weight information, limiting
the properties that can be verified. The present work addresses this
limitation through a weight-discretized quotient abstraction that preserves
synaptic contributions while reducing the state space exponentially. Moreover, no existing tool unifies probabilistic SNN simulation and formal verification. The present work integrates both capabilities.


// ─── 3. Preliminaries (~1.5 pages) ──────────────────────────────────────────
= Preliminaries <sec-prelim>
In this section, we introduce important background on the LIF model, probabilistic model checking, and quotient model abstraction.

== Spiking Neural Network Model <sec-snn-model>

An SNN is modelled as a directed graph $G = (V, E)$, where directed edges
represent unidirectional synaptic connections and $V = V_"in" union
V_"proc" union V_"out"$ partitions into input, processing, and output neurons, and $E
subset.eq V times V$ represents directed synaptic connections with integer
weights $w_e in [-100, 100]$.

Each neuron $n$ follows Leaky Integrate-and-Fire (LIF)
dynamics @naco20 @hodgkin1952quantitative. Let $r in [0,1]$ be the leak
factor and $P_"rth"$ be the firing threshold. At each discrete time step $t$,
the membrane potential _p_ integrates incoming weighted spikes and decays toward
rest:
$ p_n (t+1) = max(0, r dot.c p_n (t) + sum_(i in "In"(n)) w_(i,n) dot.c x_i (t)) $ <eq-lif>
where $x_i (t) in {0,1}$ is the spike event of presynaptic neuron $i$ at time $t$.
When $p_n (t+1) >= P_"rth"$, neuron $n$ emits a spike //($y_n (t+1) = 1$)
and resets to zero.

Firing is _probabilistic_: the membrane potential maps to a discrete integer threshold level
$L in {0, ..., N-1}$ where $N$ is configurable (1--10), and each level yields
a firing probability. Optionally, neurons implement a three-state refractory
period model: Normal ($s = 0$), Absolute Refractory Period (ARP, $s = 1$), and
Relative Refractory Period (RRP, $s = 2$), with reduced firing probability
during RRP scaled by factor $alpha$.

== Model Checking and Temporal Logics <sec-model-checking>

#definition[
  A _Discrete-Time Markov Chain_ (DTMC) is a tuple $cal(D) = (S, s_0, bold(P))$
  where $S$ is a finite set of states, $s_0 in S$ is the initial state, and
  $bold(P) : S times S -> [0, 1]$ is the transition probability matrix
  satisfying $sum_(s' in S) bold(P)(s, s') = 1$ for all $s in
  S$ @BaierKatoen2008.
]

Behavioural properties over state-transition systems are often expressed using
_Computation Tree Logic_ (CTL) @BaierKatoen2008, a branching-time temporal
logic built from four path operators: $bold(X) phi$ (_neXt_---$phi$ holds in
the immediate successor state), $phi_1 bold(U) phi_2$ (_Until_---$phi_1$
holds along a path until $phi_2$ becomes true), $bold(F) phi$
(_Finally_---$phi$ eventually holds, syntactic sugar for
$top bold(U) phi$), and $bold(G) phi$
(_Globally_---$phi$ holds at every state along the path). _Probabilistic Computation Tree Logic_ (PCTL) @hansson1994logic extends CTL to stochastic
systems by replacing the universal/existential path quantifiers with a
probabilistic operator $P_(⋈ p) [psi]$, which asserts that the
probability of satisfying path formula $psi$ meets the bound
$⋈ p$ where $⋈ in {>,>=, =, <, <=}$. For instance,
$P_(>= 1)[bold(F) (y_n = 1)]$ asserts that neuron $n$ fires with probability
one, while $P_(>= 1)[bold(G) (y_n = 0)]$ asserts permanent silence.

The role of a _probabilistic model checker_ is to compute, given a DTMC
$cal(D)$ and a PCTL property $phi$, the exact probability with which $phi$
is satisfied from the initial state. In the context of SNN verification, this
serves two purposes: (i)~_validating model correctness_, i.e., confirming that
the formal DTMC encoding faithfully reproduces expected neural behaviours
(e.g., tonic spiking under sustained input, silence without input); and
(ii)~_studying the temporal dynamics_ of small but functionally relevant
neural configurations, such as quantifying spike propagation probabilities
across chains or characterizing inhibitory gating in convergent motifs.
PRISM @PRISM2011 is a probabilistic model checker supporting DTMCs.
Models are specified as parallel compositions of _modules_ with local integer
variables and guarded probabilistic transitions, synchronizing via shared
labels. The global state space is the Cartesian product of all module state
spaces. PRISM offers an _explicit_ engine (enumerates reachable states) and a
_symbolic BDD_ engine (uses Binary Decision Diagrams via the CUDD library),
whose efficiency depends on variable ordering.

== Quotient Model Abstraction <sec-quotient>

Quotient model abstraction @Katoen2016 reduces the DTMC state
space by partitioning _probabilistically bisimilar_ states---those yielding
identical firing probabilities and, under every input, transitioning to
equivalent successor classes---into equivalence classes. The resulting quotient
reduces the per-neuron state
space from $|P_"max" - P_"min" + 1|$ values to $k + 1$ classes (where $k$ is
the number of threshold levels). However, a naïve application to SNNs treats
all synapses uniformly, collapsing weight information: it cannot distinguish
strong excitatory from weak excitatory or inhibitory contributions. The weight
discretization scheme in @sec-weight-disc addresses this limitation.


// ─── 4. Weight-Discretized Quotient Abstraction (~2.5 pages) ────────────────
= Weight-Discretized Quotient Abstraction <sec-weight-disc>

The quotient model of @sec-quotient abstracts membrane potentials into
equivalence classes but treats all synapses uniformly. This section introduces
a _weight discretization scheme_ that resolves this limitation while
preserving the relative contribution of each synapse.

== Weight Discretization Function <sec-disc-function>

#definition[
  Given a weight range $[- w_"max", w_"max"]$ (typically $w_"max" = 100$) and a
  discretization parameter $W in NN^+$, the _weight discretization function_

  $delta_W : RR -> ZZ$ is: $delta_W (w) = op("round")(w dot.c W / w_"max") = lr(⌊ w dot.c W / w_"max" ⌉)$ mapping original weights to the discrete range $[-W, W] subset ZZ$.
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
$C_n = sum_(i=1)^m w_i^d dot.c y_i$ <eq-contribution>
where $y_i in {0,1}$ is the spike output of presynaptic neuron $i$ and $m$ is
the fan-in.

== Threshold Calibration <sec-threshold-cal>

To ensure that the discretized neuron preserves the same firing difficulty as
the original, the threshold must be recalibrated to the discrete weight domain.

#definition[
  The _discretized threshold_ for a neuron with original threshold $T$ is:
  $T_d = ceil(T dot.c W / w_"max")$
]

The use of the ceiling function $op("ceil")$ (rather than rounding) ensures $T_d >= T dot.c W \/ w_"max"$,
so the discretized neuron is _at least as hard_ to fire as the original. This
conservative calibration prevents false-positive firings and is essential for
the Asymptotic Silence guarantee (Theorem~2).

== Multiplicative Leak <sec-leak>

In the discretized model, the membrane potential decays via the same
multiplicative leak as the original LIF dynamics (@eq-lif): the discretized
update rule is
$ p'_n = floor(r dot.c p_n) + C_n $
where $C_n$ is the weighted contribution from @eq-contribution and $r in [0,1]$
is the leak factor. Since $r < 1$, the floor operation guarantees that
$floor(r dot.c p_n) < p_n$ for all $p_n > 0$, ensuring strict decay in the
absence of input. This preserves strict isomorphism between the simulation
engine and the PRISM model.

== Key Properties of the Discretization <sec-proofs>

This subsection presents the two main formal guarantees. //Complete derivations will be provided in a supplementary research report, to be released upon acceptance.

The following theorem ensures that discretization does not suppress any firing configuration that was possible in the original model.

#theorem[
  *(Threshold Preservation.)*
  Let $cal(N)$ be a neuron with weights ${w_1, ..., w_m}$ and threshold $T$.
  If $cal(N)$ can fire in a single step (i.e., $exists bold(x) in {0,1}^m$ such
  that $sum_(i=1)^m w_i dot.c x_i >= T$), then the discretized neuron $cal(N)'$
  with weights ${delta_W (w_1), ..., delta_W (w_m)}$ and threshold $T_d$ can
  also fire.
]

Conversely, the next theorem provides a safety guarantee: discretization does not introduce spurious spikes.

#theorem[
  *(Asymptotic Silence.)*
  Let $cal(N)$ be a neuron with weights ${w_1, ..., w_m}$ and threshold $T$.
  If $cal(N)$ does not fire in a single step (i.e., $forall bold(x) in {0,1}^m$, $sum_(i=1)^m w_i dot.c x_i < T$), then the discretized neuron $cal(N)'$
  with weights ${delta_W (w_1), ..., delta_W (w_m)}$ and threshold $T_d$ does not fire (i.e.,  $sum_(i=1)^m delta_W (w_i) dot.c x_i' < T_d$).
]

== Biological Property Preservation <sec-bio-preservation>

The discretized model preserves the core LIF properties formalized by De Maria
et al.~@naco20:

- *Tonic spiking.* Under constant input $C_"in"$, the neuron has non-zero
  firing probability iff the net gain per step overcomes the multiplicative
  decay, i.e., $C_"in" > T_d dot.c (1 - r)$.

- *Integrator.* The probability of immediate firing on simultaneous inputs
  reaches 1.0 iff $sum delta_W(w_i) >= T_d$.

- *Excitability.* The expected inter-spike interval decreases monotonically as
  input strength increases, since stronger input yields higher net accumulation
  per step.


// ─── 5. The CogSpike Workbench (~0.75 pages) ────────────────────────────────
= The CogSpike Workbench <sec-cogspike>

Existing SNN simulation platforms offer varying levels of stochastic
modelling: Brian~2 @stimberg2019brian2 supports stochastic firing thresholds
via escape noise, while Nengo @bekolay2014nengo, NEST @gewaltig2007nest, and
BindsNET @hazan2018bindsnet are limited to noise injection at the input
level, but none integrate probabilistic model checking for formal
verification.

CogSpike bridges this gap by unifying _probabilistic_ SNN _design_,
_simulation_, and _verification_ in a single desktop workbench. The tool is
implemented in Rust with an immediate-mode GUI (egui), and its core design
principle is strict isomorphism: the PRISM code generator produces a DTMC
representation that is isomorphic to the simulation engine, that is, both share the
same mathematical model, namely the LIF dynamics of @eq-lif, the three-state
refractory period model, and the probabilistic firing logic, so that verification
results faithfully analyse simulation behaviour.

The workbench provides:

+ A *visual graph editor* for constructing SNN topologies as directed graphs
  with configurable synaptic weights, input spike generators (periodic,
  Poisson, burst, custom), and multi-generator combination modes (OR, AND,
  XOR).

+ An *isomorphic simulation engine* that executes the LIF dynamics with
  configurable model complexity: three presets---Deterministic (1 threshold
  level, no refractory), Fast (4 levels, no refractory), and Full (10 levels,
  ARP/RRP enabled)---allow trading biological fidelity for computational
  tractability. Results are visualized via raster plots, membrane potential
  traces, and aggregate firing statistics.

+ *Automated PRISM code generation* that translates the SNN graph into a
  DTMC model. The generator supports both _precise_ and _weight-discretized
  quotient_ abstraction modes (@sec-weight-disc), and synthesizes PCTL
  properties for reachability, safety, and liveness verification.
  Per-neuron potential bounds are computed from fan-in analysis to minimize
  the state space.

+ A *verification bridge* that invokes the PRISM model checker as a
  background process with configurable engine selection (explicit, sparse,
  MTBDD), JVM memory limits, and solver options (Gauss--Seidel, topological
  value iteration). Results are parsed and displayed inline alongside the
  simulation output.

// ─── 6. Case Study: Contralateral Inhibition ────────────────────────────────
= Case Study: Contralateral Inhibition <sec-casestudy>

To demonstrate the complete design--simulate--verify workflow, we apply
CogSpike to a _contralateral inhibition_ network, a neuronal archetype in
which competing neurons mutually suppress each other until a single winner
emerges @demaria2022formal. @fig-cogspike shows the 9-neuron topology
constructed in the CogSpike workbench: 3 input neurons (S1--S3) providing
constant excitation ($w = +100$), 3 competing processing neurons (N1--N3)
connected by mutual inhibitory synapses, and 3 output neurons (O1--O3).
The inhibitory weights are _asymmetric_: N1 delivers $w = -100$ to N2 and N3
while receiving only $w = -70$ in return, predetermining N1 as the winner.
The model uses $k = 4$ threshold levels, $P_"rth" = 80$, and leak factor
$r = 0.5$.

#figure(
  image("cogspike.png", width: 100%),
  caption: [The CogSpike workbench showing the contralateral inhibition
    topology: input neurons (orange, S1--S3), competing processing neurons
    (blue, N1--N3) with mutual inhibitory connections (red edges), and output
    neurons (purple, O1--O3).],
) <fig-cogspike>

== State Space Reduction

To quantify the impact of weight discretization on this topology, we compare
the reachable state space of the _precise_ PRISM model against the
_weight-discretized_ model ($W = 6$), both using the non-refractory
configuration ($k = 4$ threshold levels, no ARP/RRP).

The precise model yields 3,603 reachable states and 9,917 transitions, while
the $W = 6$ discretized model reduces this to just 387 states and 1,000
transitions, that is, a $bold(9.3 times)$ _state_ reduction and a $bold(9.9 times)$
_transition_ reduction. This substantial compression seems especially significant
given the recurrent inhibitory connections of the network.

== Formal Verification of Winner-Takes-All Dynamics

@tab-wta reports the PCTL verification results on both models. Three
properties characterize classical Winner-Takes-All behaviour: the
predetermined winner (N1) must fire infinitely often, and both losers (N2, N3)
must eventually become permanently silent.

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    [*PCTL Property*], [*Precise*], [*Disc. $W$=6*], [*Interpretation*],
    [$P_(=?) [bold(G) bold(F) (y_"N1" = 1)]$], [1.0], [1.0], [Winner fires infinitely often],
    [$P_(=?) [bold(F) bold(G) (y_"N2" = 0)]$], [1.0], [1.0], [Loser N2 eventually silent],
    [$P_(=?) [bold(F) bold(G) (y_"N3" = 0)]$], [1.0], [1.0], [Loser N3 eventually silent],
  ),
  caption: [PCTL verification results on both the precise model (3,603
    states, 9,917 transitions) and the discretized $W = 6$ model (387
    states, 1,000 transitions).],
) <tab-wta>

Both models formally verify all three WTA properties with identical
probabilities: the asymmetric weight advantage guarantees N1 as the persistent
winner (probability 1.0 of firing infinitely often), while both losers are
driven to permanent silence (probability 1.0 of eventually ceasing all
activity). Critically, the discretized model preserves these properties
exactly despite using only $387$ states, a $9.3 times$ reduction from the
precise model's $3,603$ states, demonstrating that the weight-discretized
quotient abstraction achieves substantial state space compression without
sacrificing verification fidelity.


// ─── 8. Conclusion (~0.5 page) ──────────────────────────────────────────────
= Conclusion <sec-conclusion>

Formal verification of spiking neural networks requires balancing
faithfulness to the underlying SNN model against the combinatorial explosion
inherent in exhaustive state-space exploration. In this paper we presented
a weight-discretized quotient model abstraction that addresses this challenge.
The discretization function maps continuous synaptic weights to a
compact integer range while preserving threshold feasibility  and
preventing spurious spikes. The core biological properties of LIF
neurons, i.e., tonic spiking, integrator behaviour, and excitability, are
maintained.
We also introduced CogSpike, a unified workbench that integrates probabilistic SNN design,
simulation, and PRISM-based formal verification.
//The topology-dependent scaling analysis demonstrates that the state space
//reduction compounds exponentially across neurons: approximately $17 times$ per
//neuron for $W = 3$, enabling verification of networks that are otherwise
//intractable. Empirical validation across seven canonical topologies confirms
//the theoretical predictions and identifies BDD memory as the binding practical
//constraint.
A case study on contralateral inhibition demonstrated the full
design--simulate--verify workflow, formally proving Winner-Takes-All dynamics
in a 9-neuron network. The $W = 6$ discretized model preserves all verified
properties with identical probabilities while reducing the state space from
3,603 to 387 states---a $9.3 times$ reduction.

//Future directions include _compositional verification_ exploiting neuronal
//archetypes @demaria2022formal, _automated $W$ selection_ based on fan-in
//analysis, extension to _recurrent topologies_, and integrating verification
//into biologically plausible learning frameworks @bellec2019eligibility as
//modulatory safety constraints.
As for future work, we intend to focus on automated $W$ selection based on fan-in analysis, and to extend CogSpike with a parameter learning facility. The idea is to leverage probabilistic temporal logic formulae to guide the parameter search process. In particular, the violation of specific classes of properties would trigger targeted corrective updates in the network, and the procedure would iterate until the probability of satisfying key specifications converges close to 1.

// #include "appendix.typ"
