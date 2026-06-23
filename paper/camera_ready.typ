// Formal Verification of Spiking Neural Networks
// via Weight-Discretized Quotient Abstractions
//
// ICANN 2026 — CAMERA-READY (LNCS format). Max 12 pages INCLUDING references.
//
// Derived from the accepted submission (shorter.typ). Review responses are
// highlighted for the internal review pass. Compile:
//   typst compile camera_ready.typ camera_ready_review.pdf                (marked-up)
//   typst compile --input review=false camera_ready.typ camera_ready.pdf  (clean final)
// Full proofs + the 7-topology scaling study live in the extended version
// (main.typ -> arXiv), cited here as @cogspikeExtended.

#import "llncs.typ": *

// ── Review markup ────────────────────────────────────────────────────────────
// REVIEW=true highlights each passage added/changed for a reviewer comment (Rn).
// Pass `--input review=false` on the CLI for the clean camera-ready PDF.
#let REVIEW = sys.inputs.at("review", default: "true") == "true"
#let resp(id, body) = if REVIEW {
  highlight(fill: rgb("#fff2a8"))[#body#h(0.05em)#super(text(size: 6pt, fill: rgb("#cc0000"), weight: "bold")[#id])]
} else { body }

// Extended version cited as @cogspikeExtended (see refs.bib).

// ── Institutes ───────────────────────────────────────────────────────────────
#let inst-uns = institute("Université Côte d'Azur, CNRS, I3S, France")

// ── Apply template ───────────────────────────────────────────────────────────
#show: lncs.with(
  title: "A tool for Formal Verification of\n Probabilistic Spiking Neural Networks\nvia Weight-Discretized Quotient Abstractions",
  authors: (
    author("Nikan Zandian Jazi", insts: (inst-uns,)),
    author("Elisabetta De Maria", insts: (inst-uns,)),
    author("Christopher Leturc", insts: (inst-uns,)),
  ),
  running-title: "Formal Verification of SNNs via Weight-Discretized Quotient Abstractions",
  running-authors: "N. Zandian Jazi, E. De Maria, C. Leturc",
  abstract: [
    Spiking Neural Networks (SNNs) model biological neural dynamics more
    faithfully than classical artificial networks, but their stochastic,
    event-driven computation demands probabilistic models for which
    deterministic abstractions are inadequate. Formal verification via
    probabilistic model checking then faces a fundamental barrier: the
    _state space explosion problem_, where the transition system grows
    exponentially with the number of neurons. General-purpose quotient model
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
    ensures that firing configurations with sufficient margin are preserved,
    and an Asymptotic Silence theorem guarantees that spurious firing cannot
    persist. A case study on contralateral inhibition demonstrates the
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
  acknowledgments: none,
  disclosure: [The authors have no competing interests to declare that are relevant to the content of this article.],
  bib: bibliography("refs.bib"),
)

// ── Review legend (review pass only) ─────────────────────────────────────────
#if REVIEW {
  block(width: 100%, fill: rgb("#fff2a8"), inset: 7pt, radius: 3pt, stroke: 0.5pt + rgb("#cc0000"))[
    #text(size: 8pt)[*Review pass — not the camera-ready PDF.* Highlighted spans answer reviewer
    comments: #text(fill: rgb("#cc0000"))[R1] proofs / scalability / comparison;
    #text(fill: rgb("#cc0000"))[R3] §5 repetition + spelling;
    #text(fill: rgb("#cc0000"))[R4] proof sketch, choosing $W$, fan-in, module synchronisation;
    #text(fill: rgb("#cc0000"))[R5] PRISM intro, $P_"rth"$/$T$ notation, "asymptotic".
    Reviewer #2's review was for a different paper. Compile with `--input review=false` for the clean PDF.]
  ]
}


// ═══════════════════════════════════════════════════════════════════════════════
// PAPER BODY — 7 sections + references
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
of neural circuits---how spikes propagate, interact, and give rise to emergent
behaviours in small but functionally relevant topologies such as chains,
convergent motifs, and recurrent loops. Modelling these dynamics demands a
delicate balance: biological neurons are inherently stochastic---ion-channel
noise, unreliable vesicle release, and variable axonal delays inject randomness
at every stage @hodgkin1952quantitative @nguyen2021review, so probabilistic
models are necessary.

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
for #resp("R5", [the PRISM model checker @PRISM2011 (a tool that computes the probability with which a temporal-logic property holds over a DTMC, up to floating-point precision)]), and (iii)~_automated model checking_
of behavioural properties expressed in Probabilistic Computation Tree Logic
(PCTL) @hansson1994logic. The underlying neuron model employs a weight-discretized quotient
abstraction that overcomes the limitations of naïve quotient
models @BaierKatoen2008, which discard synaptic weight information when
partitioning states into equivalence classes.

Concretely, the contributions are threefold:
(1)~a _weight discretization scheme_ that maps continuous synaptic weights to a finite discrete range while preserving threshold feasibility and relative synaptic contributions;// (@sec-disc-function);
(2) a _Threshold Preservation_ theorem ensuring that firing patterns with sufficient margin are preserved by the discretization, and an _Asymptotic Silence_ theorem guaranteeing that the discretized model cannot sustain spurious firing;// (@sec-proofs); and
(3)~*CogSpike*#footnote[All code and experiments are available at #link("https://github.com/QuietRocket/CogSpike").], a unified workbench integrating SNN design, simulation, and formal verification, whose code generator produces a PRISM representation isomorphic to the simulation engine, enabling automated formal modelling and model checking.// (@sec-cogspike).
#resp("R1", [The complete formal proofs, additional derivations, and an empirical scaling study across seven canonical topologies are provided in the extended version of this paper @cogspikeExtended.])
The remainder surveys related work (@sec-related) and background (@sec-prelim), presents the weight-discretized abstraction (@sec-weight-disc) and the CogSpike workbench (@sec-cogspike), and reports a contralateral-inhibition case study (@sec-casestudy).


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
share the _state space explosion problem_: the state space grows exponentially
with network size. General-purpose quotient abstractions @BaierKatoen2008 can
mitigate this in principle, but a naïve application to SNNs discards synaptic
weight information; moreover, no existing tool unifies probabilistic SNN
simulation and formal verification. The present work addresses both through a
weight-discretized quotient abstraction integrated into a single workbench.


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
factor and #resp("R5", $T$) be the firing threshold. At each discrete time step $t$,
the membrane potential _p_ integrates incoming weighted spikes and decays toward
rest:
$ p_n (t+1) = max(0, r dot.c p_n (t) + sum_(i in "In"(n)) w_(i,n) dot.c y_i (t)) $ <eq-lif>
where $y_i (t) in {0,1}$ is the spike event of presynaptic neuron $i$ at time $t$.
When $p_n (t+1) >= T$, neuron $n$ emits a spike ($y_n (t+1) = 1$)
and resets to zero.

Firing is _probabilistic_: the potential maps to a discrete threshold level
$L in {0, ..., k-1}$ ($k$ configurable, 1--10), each level yielding a firing
probability. Optionally, neurons implement a three-state refractory machine---Normal,
Absolute (ARP), and Relative (RRP)---with firing probability during RRP scaled by
$alpha$.

== Model Checking and Temporal Logics <sec-model-checking>

#definition[
  A _Discrete-Time Markov Chain_ (DTMC) is a tuple $cal(D) = (S, s_0, bold(P))$
  where $S$ is a finite set of states, $s_0 in S$ is the initial state, and
  $bold(P) : S times S -> [0, 1]$ is the transition probability matrix
  satisfying $sum_(s' in S) bold(P)(s, s') = 1$ for all $s in
  S$ @BaierKatoen2008.
]

Behavioural properties are expressed in _Computation Tree Logic_ (CTL)
@BaierKatoen2008, built from the path operators $bold(X)$ (_next_),
$bold(U)$ (_until_), $bold(F)$ (_finally_), and $bold(G)$ (_globally_).
_Probabilistic CTL_ (PCTL) @hansson1994logic replaces CTL's path quantifiers
with a probabilistic operator $P_(⋈ p) [psi]$, asserting that the probability
of satisfying $psi$ meets the bound $⋈ p$, $⋈ in {>,>=,=,<,<=}$. For instance,
$P_(>= 1)[bold(F) (y_n = 1)]$ asserts that neuron $n$ fires with probability one
and $P_(>= 1)[bold(G) (y_n = 0)]$ permanent silence.

The role of a _probabilistic model checker_ is to compute, given a DTMC
$cal(D)$ and a PCTL property $phi$, the probability with which $phi$
is satisfied from the initial state, up to floating-point precision. In the context of SNN verification, this
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
spaces. #resp("R4", [In our encoding every neuron module synchronises on a single global step ("tick") action: on each step a module reads its presynaptic neighbours' current spike outputs and updates simultaneously, so a spike crosses one synapse per step.]) PRISM offers an _explicit_ engine (enumerates reachable states) and a
_symbolic BDD_ engine (uses Binary Decision Diagrams via the CUDD library),
whose efficiency depends on variable ordering.

== Quotient Model Abstraction <sec-quotient>

Quotient model abstraction @BaierKatoen2008 @Katoen2016 reduces the DTMC state space by
partitioning _probabilistically bisimilar_ states---those yielding identical
firing probabilities and, under every input, transitioning to equivalent
successor classes. This reduces the per-neuron state space from
$|P_"max" - P_"min" + 1|$ values to $k + 1$ threshold classes. However, a naïve
application to SNNs treats all synapses uniformly, collapsing weight
information: it cannot distinguish strong from weak excitatory or inhibitory
contributions. The weight discretization scheme in @sec-weight-disc addresses
this limitation.


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
$ C_n = sum_(i=1)^m w_i^d dot.c y_i $ <eq-contribution>
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
conservative calibration underlies the Asymptotic Silence guarantee
(Theorem~2).

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
  If $cal(N)$ can fire with margin (i.e., $exists bold(y) in {0,1}^m$ such
  that $sum_(i=1)^m w_i dot.c y_i >= T + w_"max"(m\/2+1)\/W$), then the discretized
  neuron $cal(N)'$ with weights ${delta_W (w_1), ..., delta_W (w_m)}$ and
  threshold $T_d$ can also fire.
]

#resp("R4", [The proof is constructive and yields a practical rule for _choosing_ $W$: a single-step firing with margin $gamma = sum_i w_i y_i - T$ is preserved by any $W >= w_"max" (m\/2 + 1) \/ gamma$, where $m$ is the fan-in. Weights within $w_"max" m \/ (2 W)$ of threshold may flip---the gray zone that $W$ trades against the state-space reduction.])

Conversely, the next theorem provides a safety guarantee: the discretization cannot sustain spurious firing.

#theorem[
  *(Asymptotic Silence.)*
  Let $cal(N)'$ be a discretized neuron with potential $p_t < T_d$ and leak
  factor $r < 1$. If the input is zero for all $t' >= t$ (i.e., $C_n = 0$
  henceforth), then $cal(N)'$ will never fire.
]

#resp("R5", [The guarantee is _asymptotic_ rather than single-step: $delta_W$ rounds each weight by up to $1\/2$, so for fan-in $m >= 2$ the discretized contribution can exceed its scaled original by up to $m\/2$, and an isolated subthreshold step is not guaranteed silent. Once the supra-threshold drive ceases, however, the strict decay $floor(r dot.c p) < p$ ($r < 1$) sends the potential monotonically to the absorbing value $0 < T_d$, so the neuron stays silent for the entire remaining trajectory; inhibitory input only hastens this.])

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

#resp("R3", [Building on the abstraction of @sec-weight-disc, CogSpike turns the
design--simulate--verify loop into a single tool. Whereas the simulators surveyed in
@sec-related provide stochastic dynamics but no path to formal verification, CogSpike]) unifies
_probabilistic_ SNN _design_, _simulation_, and _verification_ in a single desktop workbench. The tool is
implemented in Rust with an immediate-mode GUI (egui), and its core design
principle is strict isomorphism: the PRISM code generator produces a DTMC
representation that is isomorphic to the simulation engine, that is, both share the
same mathematical model, namely the LIF dynamics of @eq-lif, the three-state
refractory machine, and the probabilistic firing logic, so that verification
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
  Per-neuron potential bounds are computed from #resp("R4", [_fan-in analysis_---bounding each neuron's reachable potential from the signs and magnitudes of its incoming weights---]) to minimize
  the state space.

+ A *verification bridge* that invokes PRISM as a background process with
  configurable engines (explicit, sparse, MTBDD) and solver options, parsing
  results inline alongside the simulation output.

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
The model uses $k = 4$ threshold levels, threshold $T = 80$, and leak factor
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
_transition_ reduction. This compression is especially notable given the
recurrent inhibitory connections of the network. #resp("R1", [The saving is not specific to this topology: across seven canonical feedforward motifs the per-neuron reduction compounds at roughly $17 times$ per neuron (for $W = 3$), growing exponentially with network size. The full topology-by-topology measurements appear in the extended version @cogspikeExtended.])

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

Both models verify all three WTA properties with identical probabilities: the
asymmetric weight advantage makes N1 the persistent winner, while both losers
are driven to permanent silence. The discretized model preserves these
properties exactly with only $387$ states---a $9.3 times$ reduction from
$3,603$---demonstrating substantial compression without loss of verification
fidelity.


// ─── 7. Conclusion (~0.5 page) ──────────────────────────────────────────────
= Conclusion <sec-conclusion>

Formal verification of spiking neural networks must balance faithfulness to the
SNN model against the combinatorial explosion of exhaustive state-space
exploration. We presented a weight-discretized quotient abstraction that maps
continuous synaptic weights to a compact integer range while preserving
threshold feasibility and preventing spurious sustained firing, maintaining the
core LIF properties of tonic spiking, integrator behaviour, and excitability.
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
Future work will pursue automated $W$ selection from fan-in analysis and a PCTL-guided parameter-learning facility, in which property violations trigger targeted corrective updates until the key specifications are satisfied with probability close to~1.

// #include "appendix.typ"
