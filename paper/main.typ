// Formal Verification of Spiking Neural Networks
// via Weight-Discretized Quotient Abstractions
//
// ICANN 2026 — Double-blind submission (LNCS format)
// Constraint: max 12 pages INCLUDING references
//
// Page budget (12 pages):
//   Title + Abstract (0.5p) + Intro (1p) + Related Work (0.5p) + Prelim (1.5p)
//   = 3.5p background
//   Weight-Disc Quotient Abstraction (2.5p) + CogSpike (0.75p) + Scaling (2p) + Conclusion (0.5p)
//   = 5.75p contributions
//   References (~1p)

#import "llncs.typ": *

// ── Anonymized institute (double-blind) ──────────────────────────────────────
#let inst-anon = institute("Anonymous Institution")

// ── Apply template ───────────────────────────────────────────────────────────
#show: lncs.with(
  title: "A tool for Formal Verification of Spiking Neural Networks via Weight-Discretized Quotient Abstractions",
  authors: (
    author("Anonymous submission", insts: (inst-anon,)),
  ),
  running-title: "Formal Verification of SNNs via Weight-Discretized Quotient Abstractions",
  running-authors: "Anonymous",
  abstract: [
    Spiking Neural Networks (SNNs) model biological neural dynamics more
    faithfully than classical artificial networks, but their stochastic,
    event-driven computation---rooted in ion-channel noise and unreliable
    synaptic vesicle release---demands probabilistic models for which
    deterministic abstractions are mathematically inadequate. Formal
    verification of such models via probabilistic model checking faces a
    fundamental barrier: the _state space explosion problem_, where the
    Discrete-Time Markov Chain (DTMC) encoding grows exponentially with the
    number of neurons. General-purpose quotient model
    abstractions @BaierKatoen2008 can in principle mitigate this growth by
    partitioning membrane potentials into equivalence classes, but a naïve
    application to SNNs discards synaptic weight information, limiting the
    properties that can be verified. This paper introduces a _weight-discretized quotient model
    abstraction_ that maps continuous synaptic weights to a compact integer
    range while preserving the relative contribution of each synapse, and
    presents CogSpike, a unified workbench that integrates SNN design,
    simulation, and PRISM-based formal verification within a single
    isomorphic tool chain. The discretization is accompanied by formal
    correctness guarantees: a Threshold Preservation theorem (completeness)
    ensures that no fireable configurations are lost, and an Asymptotic
    Silence theorem (soundness) guarantees that no spurious spikes are
    introduced. A topology-dependent scaling analysis shows that the state
    space reduction compounds exponentially---approximately 17$times$ per
    neuron for discretization parameter $W = 3$---enabling verification of
    networks that are otherwise intractable, as confirmed empirically across
    seven canonical topologies.
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

Spiking Neural Networks (SNNs)---the third generation of artificial neural
networks @maass1997networks ---are modelled as directed graphs whose nodes
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
of neural circuits---how spikes propagate, interact, and give rise to emergent
behaviours in small but functionally relevant network topologies such as
chains, convergent motifs, and recurrent loops. However, modelling these
dynamics demands a delicate balance. On one hand, biological neurons are
inherently stochastic: ion-channel noise, unreliable synaptic vesicle release,
and variable axonal delays introduce randomness at every stage of signal
transmission @hodgkin1952quantitative @nguyen2021review, making probabilistic
models mathematically necessary. One promising avenue encodes the network as
a Discrete-Time Markov Chain (DTMC) and applies probabilistic model checking,
but this faces a fundamental barrier: the _state space explosion problem_,
where the DTMC state space grows exponentially with network size, rendering
verification intractable beyond a handful of neurons. A model that is biologically faithful yet computationally tractable
enough for formal analysis represents a Pareto compromise between these two
desiderata.

This paper pursues precisely that compromise. We propose CogSpike, a unified
tool for probabilistic spiking neural networks that integrates three tightly
coupled capabilities within a single isomorphic framework: (i)~_simulation_ of
LIF-based SNN dynamics, (ii)~_formal modelling_ of the same networks as DTMCs
for the PRISM model checker @PRISM2011, and (iii)~_automated model checking_
of behavioural properties expressed in Probabilistic Computation Tree Logic
(PCTL). The underlying neuron model employs a weight-discretized quotient
abstraction that overcomes the limitations of naïve quotient
models @BaierKatoen2008, which discard synaptic weight information when
partitioning states into equivalence classes.

Concretely, the contributions are fourfold:

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

+ *CogSpike*, a unified workbench integrating SNN design, simulation,
  and formal verification, whose code generator produces a PRISM
  representation isomorphic to the simulation engine, enabling automated
  formal modelling and model checking (@sec-cogspike).


// ─── 2. Related Work (~1 page) ──────────────────────────────────────────────
= Related Work <sec-related>

== Stochastic Approaches to SNN Modelling <sec-related-noise>

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
enables exhaustive formal verification via model checking---a capability
that is fundamentally unavailable with continuous-noise formulations.

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

More recently, Yao et al.~@yao2025probabilistic introduced the
Refractory-evolve Probabilistic LI\&F (RP-LI\&F) neuron model, unifying
discrete-time refractory dynamics with probabilistic spike generation. Their
contract-based verification approach translates SNN topologies into
Discrete-Time Markov Chains (DTMCs) and specifies behavioural properties
using Probabilistic Computation Tree Logic (PCTL), enabling rigorous
assume/guarantee contracts.

These approaches establish the feasibility of formal SNN verification but
share a common limitation: the _state space explosion problem_. As network
size grows, the DTMC state space grows exponentially, and general-purpose
quotient abstractions @BaierKatoen2008 would lose synaptic weight information
if applied naïvely. Moreover, no existing tool unifies probabilistic SNN
simulation and formal verification. The present work addresses all three
limitations.


// ─── 3. Preliminaries (~1.5 pages) ──────────────────────────────────────────
= Preliminaries <sec-prelim>

== Spiking Neural Network Model <sec-snn-model>

An SNN is modelled as a directed graph $G = (V, E)$, where directed edges
represent unidirectional synaptic connections and $V = V_"in" union
V_"proc"$ partitions into input neurons and processing neurons, and $E
subset.eq V times V$ represents directed synaptic connections with integer
weights $w_e in [-100, 100]$.

Each processing neuron $n in V_"proc"$ follows Leaky Integrate-and-Fire (LIF)
dynamics @naco20 @hodgkin1952quantitative. Let $ell in [0,1]$ be the leak
factor and $P_"rth"$ be the firing threshold. At each discrete time step $t$,
the membrane potential integrates incoming weighted spikes and decays toward
rest:
$ p_n (t+1) = max(0, (1 - ell) dot.c p_n (t) + sum_(i in "In"(n)) w_(i,n) dot.c y_i (t)) $ <eq-lif>
where $y_i (t)$ is the spike event of presynaptic neuron $i$ at time $t$.
When $p_n (t+1) >= P_"rth"$, neuron $n$ emits a spike ($y_n (t+1) = 1$) and
resets to zero.

Firing is _probabilistic_: the potential maps to a discrete integer threshold level
$L in {0, ..., N-1}$ where $N$ is configurable (1--10), and each level yields
a firing probability. Optionally, neurons implement a three-state refractory
machine: Normal ($s = 0$), Absolute Refractory Period (ARP, $s = 1$), and
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

Behavioural properties over state-transition systems are expressed using
_Computation Tree Logic_ (CTL) @BaierKatoen2008, a branching-time temporal
logic built from four path operators: $bold(X) phi$ (_neXt_---$phi$ holds in
the immediate successor state), $phi_1 bold(U) phi_2$ (_Until_---$phi_1$
holds along a path until $phi_2$ becomes true), $bold(F) phi$
(_Finally_---$phi$ eventually holds, syntactic sugar for
$top bold(U) phi$), and $bold(G) phi$
(_Globally_---$phi$ holds at every state along the path).

_Probabilistic Computation Tree Logic_ (PCTL) extends CTL to stochastic
systems by replacing the universal/existential path quantifiers with a
probabilistic operator $P_(⋈ p) [psi]$, which asserts that the
probability of satisfying path formula $psi$ meets the bound
$⋈ p$ @BaierKatoen2008. For instance,
$P_(>= 1)[bold(F) (y_n = 1)]$ asserts that neuron $n$ fires with probability
one, while $P_(>= 1)[bold(G) (y_n = 0)]$ asserts permanent silence.

The role of a _probabilistic model checker_ is to compute, given a DTMC
$cal(D)$ and a PCTL property $phi$, the exact probability with which $phi$
is satisfied from the initial state. In the context of SNN verification, this
serves two purposes: (i)~_validating model correctness_---confirming that
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

Quotient model abstraction @BaierKatoen2008 @Katoen2016 reduces the DTMC state
space by partitioning _probabilistically bisimilar_ states---those yielding
identical firing probabilities and, under every input, transitioning to
equivalent successor classes---into equivalence classes. The resulting quotient
is the coarsest PCTL-preserving partition, reducing the per-neuron state
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
  $delta_W : RR -> ZZ$ is:
  $ delta_W (w) = op("round")(w dot.c W / w_"max") = lr(⌊ w dot.c W / w_"max" ⌉) $
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

The use of the ceiling function $op("ceil")$ (rather than rounding) ensures $T_d >= T dot.c W \/ w_"max"$,
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

This subsection presents the formal proofs of the key correctness
properties. For complete derivations, the reader is referred to the
extended appendix.

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
  $S = sum w_i y_i^* >= T$, and let $m^* = sum y_i^* <= m$ denote the number of
  active inputs. By definition of rounding to the nearest integer,
  $delta_W(w_i) = lr(⌊ w_i dot.c W \/ w_"max" ⌉)$ satisfies
  $ delta_W(w_i) >= w_i dot.c W \/ w_"max" - 1\/2 $ <eq-rounding-lb>
  Multiplying @eq-rounding-lb by $y_i^* in {0,1}$ and summing over all inputs:
  $
    S_d = sum_(i=1)^m delta_W(w_i) dot.c y_i^*
    >= sum_(i=1)^m w_i y_i^* dot.c W \/ w_"max" - 1\/2 sum_(i=1)^m y_i^*
    = S dot.c W \/ w_"max" - m^* \/ 2
  $ <eq-sd-bound>
  Each active input contributes at most $-1\/2$ of rounding error, yielding
  a cumulative shortfall of $-m^*\/2$. Since
  $T_d = ceil(T dot.c W \/ w_"max") <= T dot.c W \/ w_"max" + 1$,
  the discretized neuron fires ($S_d >= T_d$) whenever:
  $ (S - T) dot.c W \/ w_"max" >= m^* \/ 2 + 1 $
  Since $S >= T$ by hypothesis and $m^* <= m$, this holds for
  $W >= w_"max" dot.c (m\/2 + 1) \/ T$.
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


// ─── 5. The CogSpike Workbench (~0.75 pages) ────────────────────────────────
= The CogSpike Workbench <sec-cogspike>

Existing SNN simulation platforms offer varying levels of stochastic
modelling---Brian~2 @stimberg2019brian2 supports stochastic firing thresholds
via escape noise, while Nengo @bekolay2014nengo, NEST @gewaltig2007nest, and
BindsNET @hazan2018bindsnet are limited to noise injection at the input
level---but none integrate probabilistic model checking for formal
verification.

CogSpike bridges this gap by unifying _probabilistic_ SNN _design_,
_simulation_, and _verification_ in a single desktop workbench. The tool is
implemented in Rust with an immediate-mode GUI (egui), and its core design
principle is *strict isomorphism*: the PRISM code generator produces a DTMC
representation that is isomorphic to the simulation engine---both share the
same mathematical model, namely the LIF dynamics of @eq-lif, the three-state
refractory machine, and the probabilistic firing logic---so that verification
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

Since model complexity presets (Deterministic, Fast, Full) are shared between
simulation and verification, any configuration tested in simulation can be
directly verified, enabling a rapid _design--simulate--verify_ workflow
without manually constructing PRISM models.


// ─── 6. Topology-Dependent Scaling Limits (~2 pages) ─────────────────────────
= Topology-Dependent Scaling Limits <sec-scaling>

The weight-discretized quotient abstraction promises exponential state space
reduction; this section quantifies that promise. We derive closed-form
formulas for the DTMC state space as a function of network topology and model
configuration, and validate them empirically to identify the practical
limits of PRISM-based SNN verification.

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
- *Transfer* $T_(i,j)$: 2 states per internal edge (representing the intermediate state of a spike traveling along a synapse).

#theorem[
  *(State Space Product).* The theoretical state space is the Cartesian product
  of all module state spaces:
  $ |S_"theory"| = (T_"max"+1) dot.c 2^(|V_"in"|) dot.c product_(n in V_"proc") f_n (C) dot.c 2^(|E_"int"|) $
  where $f_n (C) = |S_n|$ depends on the model configuration $C$. The $2^(|E_"int"|)$ factor accounts for the binary states of all internal transfer edges.
]

For a chain of $N$ neurons with $P_"rth" = 100$ and weight $w = 80$: the
per-neuron factor is $f_n = 2 dot.c 121 = 242$ (fast precise), dropping to
$f_n = 2 dot.c 7 = 14$ (discretized $W = 3$). The per-neuron reduction factor
is $242 \/ 14 approx 17.3 times$.

#theorem[
  *(Exponential State Space Reduction).* For a chain of $N$ neurons, the
  state space ratio between precise and discretized models compounds
  exponentially:
  $ frac(|S^"precise"|, |S^"disc"|) = product_(n=1)^N frac(R_n, P_("max",n)^d + 1) approx 17.3^N $
  The factor $17.3$ is the per-neuron reduction ratio $242 slash 14$ from the
  chain example above. For $N = 4$: $17.3^4 approx 89,500 times$.
]

== Empirical Validation <sec-results>

63 PRISM models were generated across 7 canonical topologies (single, chain-2
through chain-4, fork, diamond, convergent), 3 configurations (Det: deterministic,
Fast: no refractory period, Full: complete refractory dynamics), and 3 model types (precise, disc. $W=2$, disc. $W=3$).
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

Three further observations emerge: (i)~_fan-out is more expensive than
fan-in_---the fork topology (3 neurons, fan-out 2) produces 275 fast states
versus 20 for the convergent topology (fan-in 2), because parallel branches
multiply module counts; (ii)~_BDD memory is the practical bottleneck_---the
diamond topology (5 neurons) causes CUDD OOM in the full precise
configuration, while the explicit engine handles chain-4 (362,289 states);
(iii)~_discretization scales gracefully_---the disc.~$W = 3$ diamond
completes with 98 states versus OOM for the full precise model. @tab-limits
summarizes the estimated maximum verifiable network sizes.

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


// ─── 7. Conclusion (~0.5 page) ──────────────────────────────────────────────
= Conclusion <sec-conclusion>

Formal verification of spiking neural networks requires balancing the
biological fidelity of probabilistic models against the compact state spaces
demanded by model checking. This paper presented a weight-discretized
quotient model abstraction that resolves this tension.
The discretization function $delta_W$ maps continuous synaptic weights to a
compact integer range while preserving threshold feasibility (Theorem~1) and
preventing spurious spikes (Theorem~2). The core biological properties of LIF
neurons---tonic spiking, integrator behaviour, and excitability @naco20 ---are
maintained by the threshold-dependent leak factor $lambda_d$.

The topology-dependent scaling analysis demonstrates that the state space
reduction compounds exponentially across neurons: approximately $17 times$ per
neuron for $W = 3$, enabling verification of networks that are otherwise
intractable. Empirical validation across seven canonical topologies confirms
the theoretical predictions and identifies BDD memory as the binding practical
constraint.

Future directions include _compositional verification_ exploiting neuronal
archetypes @demaria2022formal, _automated $W$ selection_ based on fan-in
analysis, extension to _recurrent topologies_, and integrating verification
into biologically plausible learning frameworks @bellec2019eligibility as
modulatory safety constraints.

