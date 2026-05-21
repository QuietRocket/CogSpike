// ============================================================================
// Continuous and Closed-Form Methods for LI&F Archetype Analysis.
// A methodology compendium consolidating seven DEQ research threads.
// deq/final/synthesis.typ  —  Nikan Zandian, May 2026.
// ============================================================================

#set document(
  title: "Continuous and Closed-Form Methods for LI&F Archetype Analysis",
  author: "Nikan Zandian",
  date: datetime.today(),
)

#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, first-line-indent: 0pt)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")
#show heading.where(level: 1): set text(size: 13pt)
#show heading.where(level: 2): set text(size: 11.5pt)
#show heading.where(level: 3): set text(size: 11pt, style: "italic")

// ------------------------------ callout helpers ----------------------------
#let finding(body) = block(
  width: 100%, inset: 8pt,
  fill: rgb("#f0fff0"), stroke: (left: 2pt + rgb("#2e8b57")),
  [*Finding.* #body],
)
#let negresult(body) = block(
  width: 100%, inset: 8pt,
  fill: rgb("#fff0f0"), stroke: (left: 2pt + rgb("#b22222")),
  [*Negative result.* #body],
)
#let partial(body) = block(
  width: 100%, inset: 8pt,
  fill: rgb("#fffbe6"), stroke: (left: 2pt + rgb("#c0a020")),
  [*Partial result.* #body],
)
#let headline(body) = block(
  width: 100%, inset: 10pt,
  fill: rgb("#f0f7ff"), stroke: (left: 3pt + rgb("#4a90d9")),
  text(weight: "semibold", body),
)
#let intuition(body) = block(
  width: 100%, inset: 8pt,
  fill: rgb("#f0f7ff"), stroke: (left: 2pt + rgb("#4a90d9")),
  [*Plain-language reading.* #body],
)
#let plain(body) = block(
  width: 100%, inset: 8pt,
  fill: luma(247), stroke: (left: 2pt + luma(160)),
  [*What this method is.* #body],
)

// ------------------------------ title block --------------------------------
#align(center)[
  #text(size: 17pt, weight: "bold")[
    Continuous and Closed-Form Methods \
    for Leaky Integrate-and-Fire \
    Archetype Analysis
  ]
  #v(0.4em)
  #text(size: 11pt)[A Methodology Compendium]
  #v(0.3em)
  #text(size: 10pt, style: "italic")[
    What differential-equation lenses see in spiking-network archetypes,
    and what only discrete verification reaches
  ]
  #v(0.5em)
  #text(size: 11pt)[Nikan Zandian]
  #v(0.2em)
  #text(size: 10pt, style: "italic")[CogSpike research note --- May 2026]
]

#v(0.8em)

#block(
  width: 100%, inset: 10pt,
  fill: rgb("#f6f6f6"), stroke: (left: 3pt + luma(160)),
)[
  *Abstract.* The CogSpike programme verifies spiking neural networks
  *discretely* --- the prior paper @cogspike_wd encodes them as Markov chains
  and model-checks them in PRISM, and the De Maria et al. "FCS" line
  @DeMaria2020 encodes leaky integrate-and-fire (LI\&F) neurons in Lustre and
  verifies temporal-logic properties with Kind2. Discrete verification is
  exact but exponential, and it certifies a property at a parameter point
  without describing the *shape* of the region where it holds. Over seven
  research threads we asked whether a *continuous* lens --- differential
  equations, eigenvalues, transfer functions, bifurcation theory --- could see
  structure the discrete lens cannot. This compendium consolidates all seven.
  The answer is a clean two-sided one. Continuous methods do recover shape and
  scale: a single eigenvalue predicts the winner-take-all boundary at
  $99.96%$ in mean-field; the contralateral pitchfork and the
  activator--inhibitor Hopf are exact closed-form curves; the Siegert formula
  gives a $99.6%$-recall envelope of the FCS winner-take-all region. But every
  continuous method systematically *misses the integer-tick spike-timing
  physics* --- the exact staircase of synchronous-lock cells, the exact
  oscillation period, the binary `1100` waveform, deterministic lock-in. One
  nonlinearity, the *spike-reset rule*, is responsible for the entire boundary
  between what these methods reach and what they do not, and whether it
  appears in the property being verified is a decidable, a-priori test. The
  six methods form a *ladder* of increasing fidelity and decreasing
  tractability --- linearization, transfer functions, Wilson--Cowan
  mean-field, the Siegert closed form, quasi-renewal mesoscopics, and finally
  the discrete FCS oracle itself. This document is organised by that ladder.
  It reports the successes and the dead ends with equal weight, because the
  dead ends turned out to be the sharpest result: they measure exactly how
  much of a verified property is irreducibly discrete.
]

#v(0.5em)

#outline(depth: 2, indent: auto)

#pagebreak()

// ════════════════════════════════════════════════════════════════════════════
= Introduction and the central finding <sec-intro>
// ════════════════════════════════════════════════════════════════════════════

== The starting point: the discrete programme <sec-start>

The CogSpike project studies *spiking neural networks* (SNNs): networks of
neurons that communicate by discrete, all-or-nothing spikes rather than
continuous signals. The neuron model throughout is the *leaky
integrate-and-fire* (LI\&F) neuron --- a unit that sums weighted input into a
membrane potential, leaks a fraction of it away each step, and emits a spike
(then resets) when the potential crosses a threshold.

The programme's prior work is *discrete* end to end. The published CogSpike
paper @cogspike_wd encodes a probabilistic SNN as a Discrete-Time Markov Chain
and verifies behavioural properties with the PRISM model checker, introducing
a weight-discretization scheme to fight the *state-space explosion problem*
(the chain grows exponentially with neuron count). The body of work it builds
on --- De Maria et al., which we abbreviate *FCS* throughout after the journal
of its archetypes paper @DeMaria2020 --- encodes LI\&F neurons in the
synchronous language Lustre and verifies temporal-logic properties with the
Kind2 model checker. Both are *sound and exact*: they give a machine-checked
yes/no answer. Both share two limitations. They are *exponential* in network
size, and they verify a property *at a point* in parameter space --- one
integer weight assignment at a time --- without telling you the shape of the
region of weights where the property holds, or why.

#intuition[
  A model checker is like testing a circuit by trying every input
  combination. It is exhaustive and trustworthy, but it scales badly, and
  when it finishes you know *that* the circuit works, not *why*, and not what
  margin you have before it stops working.
]

== The question <sec-question>

A spiking network is a *dynamical system*: a state that evolves in steps. The
mature mathematics of dynamical systems --- differential equations,
eigenvalues, transfer functions, bifurcation theory --- was built for
*continuous* systems, and it answers exactly the questions a model checker
does not: where is the boundary of a behaviour in parameter space, how fast
does the system settle, what happens just past the boundary. The seven-thread
research programme consolidated here asked one question:

#headline[
  Can a continuous, differential-equation lens reveal structure in LI\&F
  spiking-network archetypes that the discrete model-checking lens cannot ---
  can we see these systems "as if continuous" and read off shapes invisible to
  exhaustive verification?
]

The obstruction is one specific feature of a spiking neuron: the *reset*. The
membrane potential integrates smoothly, then --- at a spike --- jumps
discontinuously back down. Calculus, eigenvalues, and every tool above
presuppose a *smooth* system. The reset is not smooth. The seven threads are,
in effect, seven different ways of getting around the reset, and seven
measurements of what that costs.

== The answer <sec-answer>

The answer is two-sided and consistent across all seven threads.

#headline[
  *Continuous methods recover the shape and scale of behaviour --- boundaries,
  envelopes, bifurcation curves, oscillation existence, reachability --- but
  systematically miss the integer-tick spike-timing physics: the exact
  staircase of synchronous-lock cells, the exact integer period, the binary
  spike waveform, deterministic lock-in.* The two are complementary, not
  competing. The shapes the project hoped to see are real and useful; the
  discrete lens nonetheless retains a residual "positive content" that no
  continuous theory reaches.
]

Both halves are quantified later, but the headline numbers are worth stating
now. On the positive side: a single eigenvalue predicts the winner-take-all
boundary at $99.96%$ accuracy in the mean-field description (#ref(<sec-wta>));
the winner-take-all and oscillation onset curves are derivable in *closed
form* and match simulation to machine precision (#ref(<sec-rung-wc>)); the
Siegert formula gives an *envelope* of the FCS winner-take-all region with
$99.6%$ recall, usable as a verification pre-filter (#ref(<sec-rung-siegert>)).
On the negative side: no rate-equation method exceeds a *Jaccard agreement of
roughly $0.70$* with the discrete FCS winner-take-all oracle --- the residual
$approx 0.30$ is the integer-tick lock that only discrete verification
resolves (#ref(<sec-wta>)); and the predicted oscillation period is wrong by a
*structural factor of four* that no recalibration can fix
(#ref(<sec-oscillation>)).

The single mechanism behind every line of that paragraph is the spike-reset
rule. This is the *diagnostic principle* of the whole programme, and it
recurs in every thread:

#intuition[
  The spike-reset is the one nonlinearity that every continuous method erases.
  Whether the reset *matters* for the property you are verifying decides
  whether continuous methods apply. Properties about the *existence and
  stability of attractors* --- can a winner emerge, is the network stable,
  does it oscillate at all --- do not depend on the exact reset schedule, and
  continuous methods handle them. Properties about the *exact integer schedule
  of spikes* --- which precise period, which binary pattern, deterministic
  lock-in --- are the reset schedule, and continuous methods cannot reach them.
]

== How to read this document <sec-howto>

The six methods used across the seven threads form a *ladder*, ordered from
the cheapest and coarsest to the most faithful and most expensive. The ladder
is the spine of this document:

- #ref(<sec-ladder>) climbs the ladder rung by rung. Each rung gets a
  plain-language introduction --- what the method is, why it exists, an
  analogy to something a formal-methods reader already knows --- *before* any
  equation. No prior background in differential equations or control theory is
  assumed. #ref(<sec-primer>) is a one-page primer and a discrete-to-continuous
  translation table that the rest of the document leans on.

- #ref(<sec-phenomena>) crosses the ladder the other way. It takes the three
  phenomena the threads studied --- winner-take-all, oscillation, reachability
  --- and walks each one *down* the ladder, so the reader sees every method
  applied to the same target. This is where the thread-by-thread results live.

- #ref(<sec-matrix>) is the index: a capability matrix of method $times$
  phenomenon, with the verdict and the one number that supports it in each
  cell, and an explicit list of what each method *cannot* do.

- #ref(<sec-deadends>) collects the dead ends --- the hypotheses that failed
  --- and argues that each failure, because it falls exactly where the
  diagnostic principle predicts, is itself a result.

- #ref(<sec-publication>) connects the compendium back to the discrete CogSpike
  paper and forward to a publication.

Three appendices follow: a thread-by-thread chronology (#ref(<sec-chrono>)),
per-thread phase scorecards (#ref(<sec-scorecards>)), and a glossary of every
technical term (#ref(<sec-glossary>)). A reader who meets an unfamiliar word
should expect it to be defined in the glossary and introduced in full at its
first ladder rung.

// ════════════════════════════════════════════════════════════════════════════
= The methodology ladder <sec-ladder>
// ════════════════════════════════════════════════════════════════════════════

The seven threads use six analysis methods. They are not six competitors for
the same job: they form a *ladder*. Each rung makes one more concession to the
discrete spike physics, gains fidelity, and loses tractability. Rung 1 is a
napkin calculation; rung 6 is the exponential model checker the whole
programme is trying to avoid. The art is knowing which rung a given question
needs.

This section climbs the ladder. Every rung opens with a plain-language box ---
what the method is and why it exists --- and only then states the equations,
each followed by a sentence of intuition. The phenomena and the numbers come
in #ref(<sec-phenomena>); this section is the vocabulary.

== A reader's primer <sec-primer>

A *dynamical system* is a rule that advances a *state* one step at a time. For
an SNN the state is the vector of membrane potentials, and the rule is
"integrate weighted input, leak, and spike-and-reset on threshold crossing".
Everything in this document is an attempt to analyse that rule.

The rule has two parts. The *subthreshold* part --- integrate and leak --- is
*linear*: doubling the input doubles the response. Linear systems are
completely understood. The *spiking* part --- the threshold test and the reset
--- is *nonlinear* and, worse, *discontinuous*: at a spike the state jumps. The
six methods differ only in how they handle that discontinuity, from "ignore
it" (rung 1) to "simulate it exactly" (rung 6).

Five ideas are used throughout. Each is developed properly at its rung; this
is the one-paragraph version.

- *Linearization.* Near an operating point, replace the rule by its best
  linear approximation. The error is small for small deviations.

- *Eigenvalue.* For a linear rule $x arrow.r M x$, an eigenvalue $lambda$ is a
  number such that some direction is simply rescaled by $lambda$ each step.
  $|lambda| < 1$ means that direction shrinks (stable); $|lambda| > 1$ means it
  grows. The *spectral radius* $rho(M) = max |lambda|$ is the single number
  that decides whether the whole system settles.

- *Fixed point.* A state the rule maps to itself --- an equilibrium. A network
  "at rest" sits at a fixed point; *which* fixed point, and whether it is
  stable, is the whole game.

- *Frequency domain.* Instead of asking how the system relaxes on its own, ask
  how it responds when *driven* by an input that oscillates at frequency
  $omega$. The answer is a single complex number $H(omega)$ per frequency.

- *Bifurcation.* As a parameter (a synaptic weight) is varied, a fixed point
  can change its stability or split in two. The parameter value where this
  happens is a *bifurcation*, and it is the *boundary* of a behaviour --- the
  shape the discrete lens cannot draw.

Because the threads move between the discrete FCS world and the continuous
world, every continuous symbol has a discrete counterpart. #ref(<tbl-bridge>)
is the translation table; it is used without further comment from here on.

#figure(
  table(
    columns: (auto, 1.45fr, 1.5fr, 1.9fr),
    inset: 6pt, align: (left, left, left, left), stroke: 0.5pt,
    table.header(
      [*Object*], [*Discrete (FCS) world*], [*Continuous world*],
      [*One-line intuition*],
    ),
    [Time], [integer tick $t$], [continuous time, or a step of a smooth map],
      [The FCS neuron advances in indivisible ticks; the continuous lens
       smears them into a flow.],
    [Activity], [Boolean spike $y in {0,1}$], [firing rate $nu in [0,1]$],
      [One neuron either spikes or not; a *population* of them has a
       continuous fraction firing.],
    [Memory / leak], [windowed integrator, $r$-vector $[10,5,3,2,1]$],
      [time constant $tau$],
      [How long past input lingers. The FCS $5$-tap window is the discrete
       form of an exponential leak.],
    [Threshold], [integer $tau_("th") = 105$, hard],
      [a smooth gain function $f$ or $Phi$],
      [A single hard threshold becomes a smooth input--output curve once
       averaged over many neurons.],
    [Weight], [integer $w$ (scaled $times 10$)], [real weight $w$ or matrix $J$],
      [The wiring. Unchanged in meaning; only the arithmetic differs.],
    [The reset], [`mem` taps zeroed after a spike --- *exact*],
      [averaged away, or linearized away],
      [The defining nonlinearity. Rungs 1--5 each lose it in a different way;
       rung 6 keeps it.],
    [Eigenvalue $lambda$], [---], [mode growth/decay factor],
      [No discrete analogue: this is new information the continuous lens adds.],
    [Fixed point], [an absorbing set / a periodic lock],
      [a root of $x = F(x)$],
      [Where the system comes to rest, or cycles.],
    [Bifurcation], [the edge of a verified region on the weight grid],
      [a parameter value where stability changes],
      [The continuous lens draws this edge as a curve; the discrete lens only
       samples it cell by cell.],
  ),
  caption: [*The discrete--continuous bridge.* Every continuous symbol used in
    this document, with its FCS counterpart and a plain reading. The last
    three rows have no discrete analogue --- they are precisely the new
    structure the continuous lens contributes.],
) <tbl-bridge>

One FCS-specific detail recurs and is worth stating once. An FCS neuron keeps a
length-$5$ memory buffer of recent weighted inputs, combines them with the
fixed weights $r = [10,5,3,2,1]$ into $V(t) = sum_e r_e dot "mem"[e](t)$, fires
when $V(t) >= 105$, exports the spike one tick later, and --- the crucial part
--- *zeroes the shifted memory taps on the tick after a spike*. That zeroing is
the reset. A neuron driven by constant input of weight $>= 11$ fires on the
first tick and is called a *delayer*. The two archetypes studied throughout are
*contralateral inhibition* (two neurons, each inhibiting the other --- produces
winner-take-all) and the *negative loop* (an activator exciting an inhibitor
that feeds back --- produces oscillation).

== Rung 1 --- Linearization, eigenvalues, spectral radius <sec-rung-lin>

#plain[
  A leaky neuron's subthreshold behaviour is a *linear recurrence*: next
  potential $=$ leak factor $times$ current potential $+$ weighted input. The
  whole network, ignoring spikes for a moment, is one linear map iterated:
  $x arrow.r M x + "input"$. Linear maps are solved completely by their
  *eigenvalues*. An eigenvalue $lambda$ is a number attached to a direction
  that the map merely rescales; after $t$ steps that direction has been scaled
  by $lambda^t$. So $|lambda| < 1$ directions die out, $|lambda| > 1$
  directions take over, and the largest $|lambda|$ --- the *spectral radius*
  $rho$ --- decides the network's fate. This is rung 1: linearize, take
  eigenvalues, read off stability.
]

The LI\&F membrane recurrence, with leak $r$, weight matrix $W$, spike outputs
$y$, and external drive $u$, is
$ p_i (t+1) = r dot p_i (t) + sum_j W_(i j) y_j (t) + B_i u_i (t). $ <eq-lif>
This says each neuron keeps a fraction $r$ of its potential and adds the
weighted spikes it received. Strip the spike nonlinearity and @eq-lif is a
textbook linear recurrence. To bring the spikes back in *approximately*,
replace the Boolean spike $y_j$ by a smooth firing-rate function $f(p_j)$ and
linearize around an operating point $p^star$. The deviation
$delta p = p - p^star$ then obeys
$ delta p(t+1) = A dot delta p(t), quad
  A = r I + W dot "diag"(f'(p^star)), $ <eq-jacobian>
where $A$ is the *Jacobian* --- the best linear approximation of the dynamics
at $p^star$. Its eigenvalues are the whole story: the system is stable exactly
when $rho(A) = max_i |lambda_i(A)| < 1$.

#intuition[
  A formal-methods reader already knows this theorem in another guise.
  $rho(A) < 1$ is the convergence test for a fixed-point iteration --- the same
  condition that decides whether $x^((k+1)) = M x^((k)) + c$ converges. And the
  *eigenvalue gap* (how far the top eigenvalue stands above the rest) is the
  spiking-network analogue of the *spectral gap* of a Markov chain, which
  controls how fast a random walk mixes. Eigendecomposition of the weight
  matrix is, quite literally, PageRank applied to the wiring diagram: the
  dominant eigenvector names the neuron that wins.
]

*What it costs.* An eigendecomposition is $O(n^3)$ --- polynomial, cheap,
done in milliseconds for the small archetypes here. That is the entire appeal:
rung 1 is the napkin calculation that a model checker's exponential cost is
being traded against.

*Where it appears.* The foundational threads (#ref(<sec-chrono>),
threads 1--2) established rung 1. On a contrived $4$-neuron winner-take-all
network the dominant eigenvector of $W$ correctly names the winner and the
eigenvalue gap of $20.9$ quantifies its margin; the linearized $rho(A) = 0.629$
correctly predicts settling in $approx 6$ steps. The honest test came when
this was carried to the real FCS archetypes (#ref(<sec-wta>),
#ref(<sec-reachability>)), where it both succeeded and failed --- in a way that
turned out to be the project's first major finding.

== Rung 2 --- Transfer functions and frequency response <sec-rung-tf>

#plain[
  Eigenvalues describe how a system relaxes when *left alone*. A *transfer
  function* describes how it responds when *driven*. Feed a linear system an
  input that wiggles at frequency $omega$; it responds at the same frequency,
  but rescaled and delayed. The complex number $H(omega)$ records that rescaling
  (its size) and that delay (its angle) at every frequency. The one idea to
  keep: a leaky neuron is a *low-pass filter* --- it follows slow inputs
  faithfully and ignores fast ones, because the leak averages rapid
  fluctuations away.
]

Taking the $Z$-transform of the linear recurrence @eq-lif turns a single
neuron into the transfer function $H(z) = 1 \/ (z - r)$ --- a first-order
low-pass filter with its "pole" at $z = r$. For a network around a fixed point,
the linear-response transfer function of each population (the Richardson
single-pole form @Richardson2007) is
$ H_i (omega) = g_i \/ (1 + i omega tau_m), $ <eq-richardson>
where $g_i$ is the neuron's gain (how strongly its rate responds to input) and
$tau_m$ its time constant. The numerator is the gain; the denominator is the
low-pass roll-off. Closing the loop over a network with connectivity $J$, the
network's natural frequencies are the roots of $det(I - H(omega) J) = 0$ ---
classical control theory, unchanged.

#intuition[
  $|H(omega)|$ plotted against $omega$ is a *Bode plot*. Reading it is reading
  how nimble the network is: a wide plot means it tracks fast input, a narrow
  one means it is sluggish. The *phase* of $H(omega)$ says how delayed the
  response is, and a loop whose phase reaches a half-turn while its gain still
  exceeds one will oscillate --- the Barkhausen criterion of electronics,
  which reappears literally in #ref(<sec-rung-wc>).
]

*What it costs.* Same as rung 1 --- it is the same linearization, read in the
frequency domain instead of the time domain. The extra information is dynamic:
ringing frequencies, bandwidths, stability margins.

#figure(
  image("figs/bode.pdf", width: 58%),
  caption: [A Bode plot (rung 2): the closed-loop frequency response of the
    negative-loop archetype. The magnitude falls off as the driving frequency
    rises --- the network is a low-pass filter --- with a small resonant rise
    near its natural ringing rate.],
)

*Where it appears.* Transfer functions diagnose *oscillation*
(#ref(<sec-oscillation>)). They succeed at saying *whether and roughly how* a
negative loop rings, and they fail --- structurally, unfixably --- at saying
*at what integer period*. That failure (#ref(<sec-deadends>), dead end 4) is
one of the cleanest results in the programme.

== Rung 3 --- Wilson--Cowan mean-field <sec-rung-wc>

#plain[
  Rungs 1--2 quietly replaced the hard spike threshold with a smooth curve.
  Rung 3 says *where that smooth curve comes from*. The trick is a change of
  scale. Instead of one neuron with one hard threshold, consider a *population*
  of many similar neurons whose thresholds are slightly spread out. Ask not
  "did this neuron spike" but "what fraction of the population spiked". Because
  the thresholds are spread, that fraction is a *smooth* function of the input
  --- the reset discontinuity of any single neuron is averaged away by the
  crowd. The population's firing rate then obeys a smooth differential
  equation, the *Wilson--Cowan equation* @WilsonCowan1972, and a smooth
  equation is exactly what bifurcation theory needs.
]

Replace each node of an archetype with a population of $N$ LI\&F neurons whose
firing thresholds are drawn from a distribution. The fraction firing at input
$x$ is the fraction whose threshold is below $x$ --- the cumulative
distribution, a smooth S-shaped *gain function* $f$. The population rate
$rho_i$ then relaxes according to
$ tau dot(rho)_i = -rho_i + f(sum_j W_(i j) rho_j + I_i), $ <eq-wc>
which reads: each population decays toward the rate its current total input
calls for. This is a smooth ODE. Linearizing it (rung 1, now on solid ground)
and locating where an eigenvalue crosses zero gives the *bifurcation curves* ---
the analytic boundaries of behaviour.

#intuition[
  The single most useful consequence: mutual inhibition between two
  populations is a feedback loop whose *loop gain* is the product
  $w_(12) w_(21) g_1 g_2$ of the two weights and the two gains. The
  winner-take-all transition happens exactly when that product crosses $1$ ---
  the *Barkhausen criterion* again. Because perturbations *multiply* around a
  loop, the boundary is a curve of constant *product* (a hyperbola), not a
  straight line. That is why the winner-take-all region has the shape it has,
  and the shape falls out of two lines of algebra.
]

*What it costs.* The same $O(n^3)$ linear algebra, plus a one-time symbolic
derivation per archetype. The gain is qualitative completeness: pitchfork,
Hopf, and saddle-node bifurcations are all in reach, and the FCS archetype
taxonomy (contralateral, negative loop, series, parallel, positive loop) maps
cleanly onto them.

*Where it appears.* The population thread (#ref(<sec-chrono>), thread 3) is
rung 3. Its bifurcation curves are *exact* --- machine-precision agreement with
simulation --- which is the strongest positive result in the compendium
(#ref(<sec-wta>)). Its honest limit is that the smooth average has, by
construction, discarded the reset; #ref(<sec-deadends>) dead end 5 is exactly
the price.

== Rung 4 --- The Siegert closed form <sec-rung-siegert>

#plain[
  Rung 3 still needs a gain function $f$, and so far $f$ was a hand-tuned
  S-curve with two free knobs. Rung 4 removes the knobs. Siegert's $1951$
  formula @Siegert1951 derives the gain function *from the neuron's own
  physics*: if a neuron integrates noisy input and fires on threshold
  crossing, its firing rate is a specific closed-form expression in just the
  *mean* and the *variance* of its input. Nothing is fitted to behaviour ---
  every quantity in the formula is a measurable property of the LI\&F neuron.
]

The Siegert firing rate is
$ nu = Phi(mu, sigma) = 1 / (tau_("ref") + tau_m sqrt(pi)
  integral_((V_r - mu)/sigma)^((V_("th") - mu)/sigma)
  "erfcx"(-u) dif u), $ <eq-siegert>
where $mu$ and $sigma$ are the mean and standard deviation of the input and
$"erfcx"$ is a standard special function. The formula looks heavy; the reading
is light: it is the smooth gain curve of rung 3, but now its shape is *dictated*
by the neuron rather than chosen. The reset enters honestly, as the lower limit
of the integral (the potential resets to $V_r$). A network's resting state is
then found by solving the *self-consistency* condition
$nu_i^star = Phi(mu_i (nu^star), sigma_i (nu^star))$ --- each population's rate
must be consistent with the input the other rates produce.

#intuition[
  Rung 4 buys *physical honesty*. Where rung 3's S-curve could be accused of
  being chosen to fit, rung 4's curve is a prediction. Concretely, enumerating
  the self-consistent fixed points of @eq-siegert tells you whether an
  archetype is *bistable* (two stable rates --- a winner and a loser exist) or
  *monostable* (one rate --- no winner), with no tuning. To make the FCS
  neuron's input genuinely noisy enough for the diffusion picture to hold, the
  threads inject controlled randomness --- small threshold jitter and random
  input thinning --- which is a modelling choice, not a fit.
]

*What it costs.* A one-time calibration of four physical constants against the
neuron's measured input--output curve, then root-finding --- still cheap.

*Where it appears.* The closed-form thread and its winner-take-all sibling
(#ref(<sec-chrono>), threads 4--5) are rung 4. Siegert beats the hand-tuned
curve and yields the $99.6%$-recall *envelope* of the FCS winner-take-all
region (#ref(<sec-wta>)). Its limit is that it is *static* --- a resting-rate
theory with no notion of time --- and that its envelope is exactly an
envelope: it over-includes the staircase.

== Rung 5 --- Quasi-renewal mesoscopics <sec-rung-qr>

#plain[
  Mean-field theory (rungs 3--4) assumes *infinitely many* neurons, so its
  averages are exact only in that limit. A real, finite population is noisy:
  the fraction firing fluctuates from tick to tick, and the size of that
  fluctuation scales as $1\/sqrt(N)$ for $N$ neurons. The *quasi-renewal*
  equations @NaudGerstner2012 add this finite-size noise back, and --- more
  importantly --- track the population's *age distribution*: how long ago each
  neuron last fired. Tracking age is how the reset gets back in: a neuron that
  just fired is young and cannot fire again immediately. Rung 5 is the first
  rung that *carries the reset*, at the price of being a stochastic rather than
  a deterministic description.
]

The mesoscopic update is the single-integral form
$ A(t) = sum_k m_k (t-1) thin h(k; mu(t)) + sqrt(A(t) \/ N) thin xi(t), $ <eq-qr>
where $m_k$ is the fraction of the population that last fired $k$ ticks ago,
$h$ is the hazard (the Siegert rate, gated by how long since the last spike),
and $xi$ is unit noise. The first term is "of the neurons eligible to fire,
this fraction does"; the second is the $1\/sqrt(N)$ finite-size jitter. As
$N arrow infinity$ the noise vanishes and @eq-qr collapses back to the
mean-field rate equation --- rung 5 *contains* rungs 3--4 as its large-$N$
limit.

#intuition[
  Two things rung 5 recovers that rung 4 cannot. First, *time*: because it
  tracks age and reset explicitly, it produces an actual oscillation period,
  and --- remarkably --- the *correct* one (#ref(<sec-oscillation>)). Second,
  *finite-size effects*: at small $N$ the noise blurs sharp boundaries, and
  that blur is sometimes exactly the correction needed to better match the
  discrete oracle. The reset is back in the description; what is still missing
  is that the description remains a *rate*, so it cannot produce a *binary*
  spike train.
]

*What it costs.* A stochastic simulation, swept over population size $N$ ---
more expensive than rungs 1--4, still far below the model checker.

*Where it appears.* Quasi-renewal is the finite-size rung in every closed-form
thread (#ref(<sec-chrono>), threads 4--7). It recovers the oscillation period
exactly (#ref(<sec-oscillation>)) and partially dissolves the winner-take-all
staircase (#ref(<sec-wta>)); its honest ceiling is the $approx 0.70$ Jaccard
floor and the inability to produce the binary waveform.

== Rung 6 --- The discrete FCS oracle <sec-rung-fcs>

#plain[
  The bottom rung is the system itself. The FCS LI\&F neuron, simulated or
  model-checked *exactly*: every integer tick, every threshold test, every
  reset computed with no averaging and no linearization. This is rung 6. It
  answers any question about the network --- exact spike trains, exact periods,
  bit-exact verdicts --- and it is what the prior CogSpike paper @cogspike_wd
  and the FCS programme @DeMaria2020 use. Its cost is the state-space
  explosion: exhaustive verification is exponential in neuron count. Rung 6 is
  the reference against which rungs 1--5 are measured, and the cost of rung 6
  is the reason rungs 1--5 exist.
]

There is nothing to approximate at rung 6, so there are no equations to
introduce --- only the FCS LI\&F definition already given in
#ref(<sec-primer>), run forward tick by tick (the deterministic *oracle* used
as ground truth in every thread) or compiled to Lustre and discharged to the
Kind2 model checker (the verification path of @DeMaria2020).

#intuition[
  Rung 6 is the only rung that sees the binary `1100` spike pattern, the exact
  integer period, and the deterministic lock-in --- because rung 6 is the only
  rung that did not throw the reset away. Everything the continuous ladder
  *cannot* do is, definitionally, what rung 6 is *for*. The compendium's
  recurring phrase "the positive content of formal verification" means
  precisely the set of questions that need rung 6.
]

== The ladder as a whole, and the diagnostic principle <sec-ladder-whole>

#figure(
  table(
    columns: (auto, 1.15fr, 1.5fr, auto),
    inset: 6pt, align: (left, left, left, left), stroke: 0.5pt,
    table.header([*Rung*], [*Method*], [*What it computes*], [*Cost*]),
    [1], [Linearization / eigenvalues], [stability, convergence rate,
      reachability of attractors], [$O(n^3)$],
    [2], [Transfer functions $H(omega)$], [frequency response, ringing
      direction, stability margins], [$O(n^3)$],
    [3], [Wilson--Cowan mean-field], [bifurcation curves: winner-take-all and
      oscillation *boundaries*], [$O(n^3)$ + symbolic],
    [4], [Siegert closed form], [physically-grounded fixed points; the
      winner-take-all *envelope*], [calibration + root-finding],
    [5], [Quasi-renewal mesoscopics], [finite-size corrections; the
      oscillation *period*], [stochastic sim, swept in $N$],
    [6], [Discrete FCS oracle], [everything, bit-exact: spike trains, integer
      periods, verdicts], [exponential],
  ),
  caption: [*The methodology ladder.* Fidelity increases and tractability
    decreases down the rungs. Each rung restores one more piece of the spike
    physics that the rung above discarded; the boundary between rungs 1--5 and
    rung 6 is the spike-reset rule.],
) <tbl-ladder>

Climbing from rung 1 to rung 6, each step puts back one piece of the spike
physics. Rung 1 ignores the reset entirely. Rung 3 averages it into a smooth
curve. Rung 5 tracks it through the age distribution. Rung 6 computes it
exactly. The single thing the rungs disagree about is the reset --- and that
is the diagnostic principle, now stated in full:

#headline[
  *The diagnostic principle.* The spike-reset rule is the essential
  nonlinearity that continuous analysis erases. Whether it appears in the
  formal property being verified determines whether continuous methods apply.
  Properties about the *existence and stability of attractors* --- reachability,
  bifurcation, oscillatory-vs-not classification --- do not depend on the exact
  reset schedule, and rungs 1--5 reach them. Properties about the *exact integer
  schedule of spike emissions* --- deterministic bit-exact trajectories,
  specific integer periods, binary waveforms, deterministic lock-in --- *are*
  the reset schedule, and only rung 6 reaches them.
]

This principle is not a slogan attached after the fact. It was rediscovered
independently in four threads, on three mathematical frameworks (eigenvalues,
transfer functions, quasi-renewal) and two phenomena (winner-take-all,
oscillation). #ref(<sec-phenomena>) is the evidence; #ref(<sec-matrix>) is the
ledger.

// ════════════════════════════════════════════════════════════════════════════
= The three phenomena <sec-phenomena>
// ════════════════════════════════════════════════════════════════════════════

#ref(<sec-ladder>) introduced the six methods. This section applies them. The
threads studied three phenomena of FCS LI\&F archetypes --- *winner-take-all*,
*oscillation*, and *reachability* --- and each subsection below takes one
phenomenon and walks it down the ladder, so the reader sees the same target hit
by every method in turn. The thread-level numbers live here; the per-thread
chronology and scorecards are in #ref(<sec-chrono>) and #ref(<sec-scorecards>).

== Winner-take-all <sec-wta>

*The phenomenon.* In the *contralateral inhibition* archetype, two neurons
each receive constant external drive and each inhibit the other. *Winner-take-all*
(WTA) is the outcome where one neuron fires steadily and the other falls
silent. FCS verifies this as Property 7, and its Figure 10 plots, over a
$40 times 40$ grid of the two inhibitory weights $(w_(12), w_(21))$, whether
WTA stabilises within four ticks.

*The discrete ground truth (rung 6).* Reproducing FCS's oracle gives
$1014\/1600 = 63.4%$ WTA-stable ("blue") cells. The striking structure is
*three diagonal blocks of red* in a sea of blue: bands of weights where, instead
of a winner, both neurons fall into a *synchronous lock* --- they fire in
identical lockstep and never break symmetry. Classifying the red cells by the
period of that lock gives a clean partition: period-$2$ lock at weak weights
($144$ cells), period-$3$ at medium weights ($361$ cells), period-$4$ at strong
weights ($81$ cells). These three blocks are the *staircase*, and they are the
recurring antagonist of this section: every continuous method must somehow
account for them, and none fully does.

#figure(
  image("figs/fcs_staircase.pdf", width: 62%),
  caption: [The discrete winner-take-all ground truth (rung 6): the FCS
    Property 7 / Figure 10 grid, our reproduction. $63.4%$ of cells are
    WTA-stable (blue); the rest fall into the three diagonal *staircase* blocks
    of synchronous lock (red). Every continuous method below is judged against
    this map.],
) <fig-staircase>

=== Rung 1 on WTA: a two-regime split <sec-wta-rung1>

The first honest test of the eigenvalue method was: does the spectral radius,
or the eigenvalue gap, of the linearized contralateral system predict the blue
region of #ref(<fig-staircase>)? The answer split cleanly in two, along a
semantic line, and the split is the foundational thread's central result.

It depends on *which question* "winner-take-all holds here" is asking. Two
formalisations coincide for smooth systems but diverge for a bit-exact discrete
one:

- *Deterministic semantics:* starting from the zero initial state, does the
  one trajectory the simulator follows reach WTA?
- *Reachability semantics:* does *some* trajectory, from *some* small
  perturbation of the initial state, reach WTA?

Kind2 model checking searches over reachable states --- it verifies the
*second*. A physical network from fixed initial conditions runs the *first*.

#finding[
  Under *deterministic* semantics, WTA is *combinatorial*, not spectral. No
  eigenvalue quantity beats $64$--$69%$ classification accuracy against the
  $63.4%$ blue baseline. The eigenvalue gap of the raw weight matrix is in fact
  *identically zero* for any $2 times 2$ zero-diagonal matrix (a structural
  feature of mutual inhibition), and the linearized spectral radius is
  symmetric under swapping the two weights, hence blind to the asymmetry that
  decides the winner. What *does* decide it is an integer comparison: the
  simulator breaks symmetry at tick $2$ via the predicate
  $(|w_(12)| > 12) "xor" (|w_(21)| > 12)$, and the winner is the neuron
  receiving weaker inhibition. The plain rule $||w_(12)| - |w_(21)|| > 7$
  classifies the deterministic map at $83.4%$ --- better than any spectral
  quantity --- and the sign of $|w_(12)| - |w_(21)|$ names the winner with
  $100%$ accuracy.
]

The same eigenvalue method, asked the *reachability* question, succeeds:

#finding[
  Under *reachability* semantics --- the semantics Kind2 actually verifies ---
  WTA *is* spectral. The reachability ground truth has $97.8%$ blue cells
  (a $6 times 6$ corner of weak weights is the only unreachable region), and
  the scalar linearization's spectral radius separates the classes cleanly:
  unreachable cells have $rho(A) in [0.518, 0.598]$, reachable cells
  $rho(A) in [0.544, 0.929]$. The predicate $rho(A) < 0.544$ is a *sound*
  certificate of non-reachability --- every cell it flags is genuinely
  non-reachable --- and overall classification reaches $98.5%$.
]

This two-regime split #ref(<fig-triptych>) is the cleanest illustration of the
diagnostic principle in the whole programme. The reset is the mechanism that
*chooses* the winner under deterministic semantics, so deterministic WTA is out
of reach of linearization; under reachability semantics the reset's effect is
averaged over perturbations, so reachability WTA is in reach.

#figure(
  image("figs/triptych.png", width: 100%),
  caption: [*The two-regime split.* (a) Deterministic ground truth, $63.4%$
    blue. (b) The combinatorial predictor $||w_(12)|-|w_(21)|| > 7$ traces the
    deterministic boundary at $83.4%$ --- no spectral quantity reaches this.
    (c) Reachability ground truth, $97.8%$ blue; the spectral-radius contour
    $rho(A) = 0.544$ (black) bounds the unreachable region at $98.5%$. Panels
    (b) and (c) are the finding: the deterministic map is combinatorial, the
    reachability map is spectral.],
) <fig-triptych>

=== Rung 3 on WTA: the bifurcation curve <sec-wta-rung3>

Lifting the contralateral archetype to a Wilson--Cowan population pair makes
the WTA boundary a *bifurcation curve* that can be written down. Linearizing
@eq-wc and decomposing into sum and difference modes shows the symmetric
("no winner") state loses stability exactly when the loop gain
$w_(12) w_(21) g_1 g_2$ crosses $1$ --- a *pitchfork* bifurcation. The dominant
eigenvalue of the mean-field Jacobian then classifies WTA on a $50 times 50$
grid at $99.96%$ accuracy, and the pitchfork curve itself, traced symbolically
from the loop-gain-unity condition, agrees with the numerical bifurcation trace
to machine precision.

#finding[
  In the mean-field description, winner-take-all is an *exactly solvable*
  bifurcation. The boundary is the loop-gain-unity hyperbola
  $w_(12) w_(21) g_1 g_2 = 1$; a single eigenvalue predicts which side of it a
  cell is on at $99.96%$. This is the strongest positive result in the
  compendium: it is the "shape invisible to the discrete lens" the programme
  set out to find. The whole FCS archetype taxonomy follows the same pattern
  --- contralateral inhibition gives a pitchfork, the negative loop a Hopf, the
  positive loop a saddle-node, and the sign of the loop's weight product
  selects which (Thomas's rule).
]

But the mean-field WTA boundary is not the *discrete* WTA boundary. Validated
against the LI\&F simulator on the same grid, the smooth pitchfork hyperbola
agrees only at the symmetric corner. The discrete bistable region is a pair of
*rectangular strips* (a winner emerges once *either* weight exceeds $approx 7$),
because once either neuron fires first, its inhibition can lock the other below
threshold --- a *spike-timing lock-in* that the smooth average has no term for.

#partial[
  The Wilson--Cowan pitchfork is a *lower bound* on the discrete winner-take-all
  region, not its envelope. The smooth average captures one mechanism of
  bistability --- the symmetric state losing stability --- and misses a second,
  *spike-timing lock-in*, which is intrinsically discrete. The two are
  complementary mechanisms at two scales, not a right and a wrong answer; but a
  reader who needs the discrete boundary cannot get it from rung 3.
]

#figure(
  grid(columns: 2, column-gutter: 8pt,
    image("figs/wc_wta_panel.pdf"),
    image("figs/wc_lif_overlay.pdf"),
  ),
  caption: [Rung 3 on WTA. *Left:* the mean-field winner-take-all region with
    the symbolic pitchfork curve overlaid; the dominant eigenvalue classifies
    it at $99.96%$. *Right:* the discrete LI\&F bistable region (grey) against
    the same pitchfork curve (red) --- they agree at the symmetric corner and
    diverge in the arms, where spike-timing lock-in, invisible to the smooth
    average, dominates.],
) <fig-wc-wta>

=== Rung 4 on WTA: the Siegert envelope <sec-wta-rung4>

The closed-form thread replaces rung 3's hand-tuned gain curve with the Siegert
formula @eq-siegert and re-reads the FCS staircase grid. Enumerating the
self-consistent fixed points and marking a cell WTA-capable when a bistable
pair of rates exists gives a map that relates to the discrete oracle in a
specific, useful way:

#finding[
  The Siegert fixed-point map is a high-recall *envelope* of the FCS
  winner-take-all region: it recovers $99.6%$ of the FCS-blue cells. Its errors
  are almost all one-sided --- it labels the staircase blocks blue when FCS
  calls them red. The reading is that rate-equation bistability is *necessary*
  for WTA but not *sufficient*: the staircase cells are bistable in the
  rate description, but the integer-tick dynamics never commit to either
  branch. Because the errors are one-sided, the envelope is a sound
  *pre-filter*: any cell *outside* it is guaranteed non-WTA and need not be
  model-checked --- a $7.4%$ reduction in the Kind2 workload on this grid.
]

The Siegert curve also beats the hand-tuned curve on its own terms: calibrated
against the neuron's measured input--output curve, it agrees with the discrete
WTA region at Jaccard $0.843$ versus the hand-tuned curve's $0.796$. Rung 4's
limit is structural --- it is a *static* theory, so its overall Jaccard with
the staircase-bearing FCS oracle sits at $0.680$, the staircase being exactly
the cells it over-includes.

#figure(
  image("figs/siegert_envelope.pdf", width: 100%),
  caption: [Rung 4 on WTA. Left: the FCS oracle. Centre: the Siegert
    fixed-point map. Right: the disagreement --- black cells are Siegert-blue,
    FCS-red, i.e. exactly the diagonal staircase. Siegert recovers $99.6%$ of
    FCS-blue cells but adds the staircase as one-sided false positives,
    which is what makes it a sound pre-filter envelope.],
) <fig-siegert-envelope>

=== Rung 2 on WTA: a negative result <sec-wta-rung2>

It is natural to hope that the staircase --- where the discrete system fails to
*commit* within four ticks --- is a slow-dynamics effect, so that a transfer-function
latency gate ("is the slowest mode fast enough?") would separate the staircase
from the genuine WTA cells. It does not.

#negresult[
  The transfer-function latency gate is *orthogonal* to the staircase.
  Gating cells by $|"Re"(lambda_("dom"))| > 1\/4$ changes the Jaccard agreement
  with FCS by $-0.003$ --- statistically nothing. The eigenvalue distributions
  of staircase and non-staircase cells overlap almost completely: the staircase
  cells are *not* slow-decay cells. They have well-separated fixed points and
  respectable decay rates. Their redness is integer-tick determinism, which has
  no signature in any linear timescale.
]

This is a useful negative: it rules out the most plausible rung-2 rescue and
sharpens the diagnosis. The staircase is not "slow"; it is *discrete*.

=== Rung 5 on WTA: partial dissolution <sec-wta-rung5>

Quasi-renewal adds finite-size noise. Since the staircase is a fragile
synchronous lock, noise should occasionally knock a locked cell off its orbit
and let it commit to a winner --- partially dissolving the staircase and
improving agreement with FCS. It does, partially.

#partial[
  Finite-size noise partially dissolves the staircase. Agreement with the FCS
  oracle is best at *small* population size --- Jaccard $0.701$ at $N = 50$,
  rising to a peak of $0.717$ at $N approx 20$ --- because more noise dissolves
  more lock. But the agreement is *unimodal* in $N$, not monotone: below
  $N approx 20$ the same noise also destabilises genuine WTA cells, and the two
  effects trade off. The improvement over the static Siegert envelope is real
  but small ($+0.04$ Jaccard), and it is concentrated on the period-$3$ and
  period-$4$ blocks. As $N arrow infinity$ the noise vanishes and quasi-renewal
  converges back to the Siegert mean-field (Jaccard $0.963$), confirming it
  sits correctly above rung 4 on the ladder.
]

#figure(
  image("figs/qr_staircase_jaccard.pdf", width: 62%),
  caption: [Rung 5 on the winner-take-all staircase: agreement with the
    discrete oracle against population size $N$. Agreement is *unimodal* ---
    best near $N approx 20$, where finite-size noise dissolves the most
    synchronous lock before it begins to destabilise genuine winner-take-all
    cells --- and converges back to the static Siegert envelope as
    $N arrow infinity$.],
)

=== The Jaccard floor, and the multi-neuron inverse staircase <sec-wta-floor>

Across rungs 1--5, no rate-equation method exceeds a Jaccard agreement of
*about $0.70$* with the discrete FCS winner-take-all oracle. This ceiling is
structural --- it is not closed by more noise, a better gain curve, or a
finer linearization.

#headline[
  The $approx 0.70$ Jaccard floor is the *positive content of formal
  verification*. The residual $approx 0.30$ disagreement between every
  rate-equation method and the discrete oracle is the integer-tick synchronous
  lock --- the staircase --- and it lies on a genuine feature of the dynamics
  (deterministic spike-timing lock with phase memory) that no rate-and-hazard
  theory can express. It is, precisely, what only rung 6 can verify.
]

Extending the contralateral archetype from two neurons to $N > 2$ (uniform
all-to-all inhibition) does not just preserve this gap --- it adds a *second*,
opposite failure. The two-neuron Siegert envelope had $99.6%$ recall; at
$N > 2$ recall *collapses* with $N$ (from $0.90$ at $N = 2$ to $0.00$ at
$N = 10$). And at $N = 10$ the discrete oracle shows clean WTA at specific
integer weights where Siegert finds *no asymmetric fixed point at all*.

#negresult[
  The high-recall Siegert envelope is a two-neuron luxury. At $N > 2$ a new
  phenomenon appears --- the *inverse staircase*: the discrete FCS dynamics
  produce clean winner-take-all *below* the smooth-rate bifurcation threshold,
  in cells where the rate equations admit only the symmetric no-winner fixed
  point. At $N = 2$ the continuous lens *over*-predicts WTA (the staircase); at
  $N = 10$ it *under*-predicts it (the inverse staircase). The continuous lens
  fails in both directions, and neither failure is closed by finite-size noise.
]

#figure(
  image("figs/inverse_staircase.pdf", width: 100%),
  caption: [The multi-neuron inverse staircase. Siegert fixed-point enumeration
    (right of each pair) versus the FCS oracle (left), per neuron count $N$. At
    large $N$ the discrete oracle shows winner-take-all where Siegert finds
    only the symmetric fixed point --- the continuous lens now *under*-predicts,
    the mirror image of the two-neuron staircase.],
) <fig-inverse>

== Oscillation <sec-oscillation>

*The phenomenon.* In the *negative loop* archetype, an activator $A$ excites an
inhibitor $I$, which inhibits $A$ back. Under constant drive the activator
settles into a periodic spike pattern. FCS verifies this as Property 5: with
the default weights the activator fires the *period-$4$* pattern `1100`
repeating, and the inhibitor echoes it one tick later. Reproducing the oracle
over the weight grid, $27.8%$ of cells produce that strict `1100` pattern and
$59.1%$ produce some regular oscillation.

#figure(
  image("figs/prop5_trace.png", width: 88%),
  caption: [The discrete oscillation ground truth (rung 6): activator (blue)
    and inhibitor (red) spike rasters of the negative loop at the default
    weights. The activator fires the exact FCS Property 5 pattern `1100`
    repeating, period $4$; the inhibitor is the activator delayed one tick.],
) <fig-prop5>

Oscillation exposes the ladder differently from winner-take-all, because here
the *mean-field theory cannot even produce the phenomenon*.

=== Rungs 3--4 on oscillation: a stable spiral everywhere <sec-osc-mf>

For the negative loop the rate-equation Jacobian has eigenvalues
$lambda_(plus.minus) = (-1 plus.minus sqrt(g_A g_I w_(A I) w_(I A))) \/ tau_m$.
Because the loop is inhibitory, the term under the square root is negative, so
$lambda$ is a complex pair with *negative* real part $-1\/tau_m$ --- a *stable
spiral*. The rate equations describe a system that *rings and then settles*,
never one that oscillates forever.

#negresult[
  Mean-field theory cannot create the FCS oscillation. The negative loop's
  rate-equation fixed point is a *stable spiral for every weight choice* --- the
  real part of the eigenvalue is pinned negative. Property 5's sustained
  oscillation is therefore *not* a Hopf bifurcation of the rate equations; it
  lies strictly beyond mean-field reach. The Siegert "spiral envelope" (cells
  whose fixed point has a complex eigenvalue) covers $90%$ of the grid but
  agrees with the strict `1100` region at Jaccard only $0.31$ --- it marks
  where ringing is *possible*, not where the integer-tick limit cycle locks.
]

This is a sharper limit than rung 3's winner-take-all limit. There, mean-field
got a *lower bound*; here it gets the wrong *type* of object --- a decaying
spiral instead of a sustained cycle.

=== Rung 2 on oscillation: the right direction, the wrong period <sec-osc-tf>

The transfer function turns the spiral's imaginary part into a predicted
ringing period $T_("pred") = 2 pi \/ |"Im"(lambda)|$. This gets the *qualitative*
behaviour right --- it correctly says the loop rings, and that the ringing rate
grows with the inhibition strength. But the *period* is wrong, and wrong in a
very particular way.

#negresult[
  The transfer-function period prediction is too long by a *structural factor
  of about four*. Across the entire FCS period-$4$ region the predicted period
  lands on the line $T_("pred") = 4 T_("FCS")$ --- a constant multiple, not
  scatter. The obvious fix --- the calibration is off, so recalibrate the time
  constant --- was tried (followup A of the negative-loop thread) and
  *falsified*: refitting the time constant against the neuron's dynamic
  response made the gap *wider*, not narrower. The reason is structural: at the
  oscillation frequency the neuron's linear response magnitude is
  $approx 0.075$ --- essentially flat. The closed-loop oscillation lives deep
  in the rolled-off regime of the single-neuron filter, where no single-pole
  model at any time constant has a resonance to offer. The factor of four is
  not a calibration error; it is the linear-response framework reaching past
  its domain.
]

This is the cleanest *epistemic* result in the programme: a tempting
explanation (recalibration) was stated as a hypothesis, tested, and killed with
a mechanism. The factor-of-four is reclassified from "a bug to fix" to "a
provable limit": no single-pole transfer function can predict the period.

#figure(
  grid(columns: 2, column-gutter: 8pt,
    image("figs/negloop_Hw_period.pdf"),
    image("figs/negloop_qr_period.pdf"),
  ),
  caption: [Oscillation period: rung 2 versus rung 5. *Left:* the
    transfer-function prediction lands on the line $T_("pred") = 4 T_("FCS")$
    (orange) --- a structural factor-of-four overshoot. *Right:* the
    quasi-renewal prediction lands on the diagonal $T = T_("FCS")$ --- the
    correct integer-tick period.],
) <fig-negloop-period>

=== Rung 5 on oscillation: the period recovered <sec-osc-qr>

Quasi-renewal, which carries the reset through the age distribution, succeeds
exactly where the transfer function fails.

#finding[
  Quasi-renewal recovers the oscillation period. Using the *same* calibration
  the transfer function used, it predicts a period of $4.05$ ticks at the
  default cell --- essentially exact agreement with the FCS period of $4$ ---
  and across the grid its predictions land on the correct diagonal rather than
  the factor-of-four line. The difference from rung 2 is not the parameters; it
  is the *mechanism*: the age-distribution stepper represents spike-and-reset
  explicitly, and "reset" is exactly what a single-pole filter cannot express.
  The sustained oscillation itself is finite-size: its amplitude decays from
  $0.20$ at $N = 50$ to $0.05$ at $N = 2000$, confirming that the cycle is
  noise-driven --- the deterministic mean field still only spirals inward.
]

What rung 5 still cannot deliver is the *binary waveform*. Its output is a
smooth rate, so even when its period is exact its match to the binary `1100`
template is only $approx 0.28$. A companion experiment confirmed this is
structural: a deterministic single-neuron renewal predictor also recovers the
period ($4.00$) exactly but, being a rate, cannot sharpen into a binary spike
train. The waveform is rung 6's alone.

A three-neuron negative loop closes the section cleanly: the period scales as
$2(n+1)$ in the number of delayer stages $n$, quasi-renewal tracks that scaling
($6.38$ predicted versus $6$ exact for the three-neuron loop), and the transfer
function stays structurally wrong. The ladder's verdict on oscillation
generalises.

#figure(
  image("figs/negloop_3neuron.pdf", width: 72%),
  caption: [The three-neuron negative loop. The discrete period is $6$ ticks
    --- the rule $2(n+1)$ for $n$ delayer stages --- and quasi-renewal tracks
    it ($6.38$ predicted); the transfer function remains structurally wrong.
    The ladder's verdict on oscillation is not a two-neuron artefact.],
)

== Reachability <sec-reachability>

*The phenomenon.* *Reachability* asks: can the system reach a target behaviour
from *some* admissible initial state? It is not a separate archetype --- it is
the question a model checker actually answers, since Kind2 searches over
reachable states. It is treated as its own phenomenon here because it is the
one place the continuous lens delivers a clean, *sound* positive result, and
because it pins down exactly which formalisation of a property the continuous
methods match.

The reachability story was told in #ref(<sec-wta-rung1>): under reachability
semantics, winner-take-all *is* spectral. The scalar linearization's spectral
radius separates reachable from non-reachable contralateral cells at $98.5%$,
and the predicate $rho(A) < 0.544$ is a *sound* non-reachability certificate.
Two points complete the picture.

#figure(
  image("figs/rho_distributions.png", width: 72%),
  caption: [Why reachability is the ladder's cleanest success: the linearized
    spectral radius $rho(A)$ cleanly separates reachable from non-reachable
    contralateral cells. The non-reachable cells occupy a narrow low-$rho$
    band, so $rho(A) < 0.544$ flags them with no false positives --- a *sound*
    certificate.],
)

#finding[
  Reachability is the continuous ladder's cleanest success, and it is sound.
  The predicate $rho(A) < 0.544$ never flags a reachable cell as
  non-reachable, so it can be trusted as a *pre-filter*: a cell it flags is
  provably not winner-take-all under the semantics Kind2 checks, and need not
  be model-checked at all. This composes the polynomial-cost rung 1 in front of
  the exponential-cost rung 6 --- exactly the multi-scale workflow the ladder is
  built for. The saving on the contralateral grid is modest (a one-sided
  certificate), but the *principle* --- a sound spectral certificate of a
  model-checking outcome --- is the engineering payoff of the whole programme.
]

#finding[
  The two-regime split is a property of the *archetype class*, not of one
  topology. Inserting a delayer neuron into the contralateral loop (FCS
  Figure 11) reproduces FCS's "contrary to expectation" asymmetric winner map
  --- the neuron whose incoming inhibition is delayed wins more often, $1136$
  cells to $448$ --- and the spectral radius still predicts the reachability
  ground truth ($95.8%$) while missing the deterministic one. The diagnostic
  principle transfers unchanged.
]

#figure(
  image("figs/winner_map.png", width: 56%),
  caption: [The delayer-augmented contralateral topology (FCS Figure 11). The
    neuron whose incoming inhibition is delayed wins in $1136$ cells against
    $448$ --- FCS's "contrary to expectation" asymmetry, reproduced
    quantitatively. The two-regime split of #ref(<sec-wta-rung1>) holds here
    unchanged.],
)

Why does reachability succeed where deterministic winner-take-all and exact
oscillation fail? The diagnostic principle answers it directly. Reachability is
a question about the *existence* of a trajectory to an attractor --- a
topological fact about the basins, which the linearized spectrum sees. The
exact deterministic trajectory, and the exact period, are facts about the
*reset schedule*, which it does not. Reachability is in scope because, by
construction, it averages the reset over perturbations.

// ════════════════════════════════════════════════════════════════════════════
= Capability and limitations matrix <sec-matrix>
// ════════════════════════════════════════════════════════════════════════════

#ref(<sec-ladder>) is the methods; #ref(<sec-phenomena>) is the results. This
section is the index that crosses them. #ref(<tbl-capability>) gives, for every
rung and every phenomenon, the verdict and the one number that supports it. It
is the document's quick-reference card.

#figure(
  table(
    columns: (auto, 1.32fr, 1.32fr, 1.2fr),
    inset: 6pt, align: (left, left, left, left), stroke: 0.5pt,
    table.header(
      [*Rung*], [*Winner-take-all*], [*Oscillation (neg. loop)*],
      [*Reachability*],
    ),
    [1 — Linearization / $rho(A)$],
      [*fail* deterministic ($64$--$69%$ vs $63.4%$ baseline)],
      [*fail* — period indistinct, args cluster across periods $3$--$8$],
      [*pass* — $rho(A) < 0.544$ sound; $98.5%$ overall],
    [2 — Transfer function $H(omega)$],
      [*fail* — latency gate orthogonal to staircase ($Delta J = -0.003$)],
      [*partial* — ringing direction right; period $approx 4 times$ too long
       (structural)],
      [— addressed at rung 1],
    [3 — Wilson--Cowan mean-field],
      [*partial* — gap $99.96%$ in mean-field; a lower bound on the discrete
       region],
      [*fail* — fixed point is a stable spiral; no sustained cycle],
      [*pass* — bifurcation-free regions certifiable],
    [4 — Siegert closed form],
      [*partial* — $99.6%$-recall envelope; staircase = false positives;
       recall collapses for $N > 2$],
      [*fail* — static; no dynamics],
      [*pass* — fixed-point enumeration as pre-filter],
    [5 — Quasi-renewal mesoscopics],
      [*partial* — partial dissolution; $approx 0.70$ Jaccard floor],
      [*pass* — period $4.05$ exact; sustained at finite $N$],
      [*pass* — finite-size-corrected boundary],
    [6 — Discrete FCS oracle],
      [*pass* — bit-exact staircase], [*pass* — exact `1100`, period $4$],
      [*pass* — the reference oracle],
  ),
  caption: [*Capability matrix.* Verdict of each ladder rung on each
    phenomenon, with the supporting number. The pattern is the diagnostic
    principle made visible: rungs $1$--$5$ pass on the "shape and existence"
    questions (reachability, bifurcation curves, envelopes) and fail on the
    "exact integer schedule" questions (the staircase, the period, the
    waveform); only rung $6$ passes everywhere, at exponential cost.],
) <tbl-capability>

The matrix reads diagonally. The top-left-to-bottom-right tendency is not an
accident: it is the diagnostic principle. Questions about *shape and existence*
--- is there a winner, where is the boundary, can the target be reached --- are
answered by cheap rungs. Questions about the *exact integer schedule* --- which
staircase cell, which period, which waveform --- are answered only by rung 6.

Stated as a list, here is the one hard limit each rung carries --- not as an
apology, but as the precise edge of its domain:

- *Rung 1* cannot reach an integer comparison. Deterministic winner-take-all is
  decided by a tick-$2$ integer test; no spectral quantity sees inside it.
- *Rung 2* cannot reach an absolute period. At the oscillation frequency the
  single-neuron response is flat, so no single-pole model resonates there ---
  the factor-of-four is structural.
- *Rung 3* cannot see spike-timing lock-in. Its winner-take-all region is a
  provable lower bound on the discrete one; a second, discrete bistability
  mechanism is invisible to the smooth average.
- *Rung 4* cannot see the staircase, and its high-recall envelope is a
  two-neuron luxury --- recall collapses, and inverts, for $N > 2$.
- *Rung 5* cannot produce a binary waveform. It recovers the period exactly but
  its output is a smooth rate; the `1100` pattern is beyond it.
- *Rung 6* cannot scale. It is bit-exact and answers everything, at a cost
  exponential in network size --- which is the entire reason rungs $1$--$5$
  exist.

// ════════════════════════════════════════════════════════════════════════════
= Dead ends, honestly <sec-deadends>
// ════════════════════════════════════════════════════════════════════════════

A compendium that reported only the successes would misrepresent the research
and, worse, waste the most useful part of it. Several specific hypotheses were
stated, pursued, and *failed*. They are collected here --- each with what was
hoped, what was tried, why it failed, and what the failure taught --- because,
read together, they are not a list of mistakes but a *map of the boundary*.
Every one of them fails exactly where the diagnostic principle predicts; each
is therefore a confirmation of the principle from its negative side.

#negresult[
  *Dead end 1 --- the eigenvalue gap predicts deterministic winner-take-all.*
  Hoped: the eigenvalue gap of the weight matrix, or of the linearized
  Jacobian, classifies the FCS WTA grid. Tried: every spectral quantity, on
  both the scalar and the full windowed-integrator linearization. Failed: best
  accuracy $64$--$69%$, barely above the $63.4%$ baseline; the raw-weight gap is
  *identically zero* for $2 times 2$ mutual inhibition, and the Jacobian
  spectral radius is symmetric in the two weights and so blind to the asymmetry
  that picks the winner. Taught: deterministic WTA is decided by an integer
  comparison at tick $2$, not by any continuous quantity --- and this is what
  first forced the deterministic-versus-reachability distinction.
]

#negresult[
  *Dead end 2 --- a fuller linearization fixes dead end 1.* Hoped: the scalar
  leak approximation was too crude; the full linearization that preserves the
  $5$-tap windowed integrator would recover the signal. Tried: the
  $5n$-dimensional state matrix, with three predictors. Failed: $46%$ accuracy
  at the symmetric fixed point --- *worse* than the baseline --- and an
  eigenvector-asymmetry predictor at $11%$, worse than chance. Taught: the
  missing information is not linear. More linear detail does not help, because
  the quantity that decides the outcome is not in the linear dynamics at all.
  The plain combinatorial rule $||w_(12)|-|w_(21)|| > 7$ reaches $83.4%$ ---
  no linearization came close.
]

#negresult[
  *Dead end 3 --- pole placement designs an oscillation period.* Hoped: if the
  spectrum predicts dynamics, invert it --- place the dominant poles to realise
  a target integer period. Tried: solving for weights that put the poles at the
  target angle, in both the archetypes thread and the population thread.
  Failed: the dominant pole angle clusters in a narrow band across simulator
  periods $3$ through $8$, so it cannot discriminate them; inverse design
  missed $5$ of $6$ targets. (In the smooth Wilson--Cowan setting the *linear*
  placement is exact, but the realised nonlinear limit cycle still missed
  $2$ of $8$ near a degenerate bifurcation.) Taught: the integer period is set
  by the reset schedule, which the linear spectrum does not encode. Direct
  enumeration --- a one-second brute-force sweep --- subsumes pole placement
  outright.
]

#figure(
  image("figs/poleplacement.png", width: 78%),
  caption: [Dead end 3 made visible: the dominant pole angle of the linearized
    negative loop, against the integer period the simulator actually produces.
    The angle clusters in a narrow band across periods $3$--$8$, so it cannot
    discriminate them --- the integer period is set by the reset schedule, not
    the linear spectrum.],
)

#negresult[
  *Dead end 4 --- recalibration fixes the oscillation period.* Hoped: the
  transfer-function period is a structural $4 times$ too long because the time
  constant was calibrated on static data; refit it on dynamic data and the gap
  closes. Tried: a Bode-style sweep of the FCS neuron's dynamic response, then
  a refit. Failed: the refitted time constant made the prediction *worse* ---
  the gap widened. The mechanism: at the oscillation frequency the neuron's
  linear-response magnitude is $approx 0.075$, essentially flat, so no
  single-pole model at any time constant has a resonance there. Taught: the
  factor-of-four is *structural*, not parametric. This converts a "method has a
  $4 times$ error" into the stronger statement "*no* single-pole transfer
  function can predict this period, and here is why" --- a negative result
  upgraded to an impossibility result.
]

#figure(
  image("figs/expA_bode.pdf", width: 64%),
  caption: [Why dead end 4 is structural. The measured frequency response of
    the FCS neuron is essentially flat ($approx 0.075$) at the oscillation
    frequency: the closed-loop oscillation lives deep in the filter's
    rolled-off regime, where no single-pole model at any time constant has a
    resonance to offer.],
)

#negresult[
  *Dead end 5 --- the Wilson--Cowan pitchfork is the discrete winner-take-all
  boundary.* Hoped: the exactly-solvable mean-field pitchfork curve *is* the
  LI\&F winner-take-all boundary. Tried: cross-validating the symbolic
  pitchfork against the discrete simulator on the same grid, out to extreme
  weights. Failed: the discrete bistable region is a pair of rectangular strips
  (a winner emerges once *either* weight exceeds $approx 7$); the pitchfork is a
  hyperbola; they coincide only at the symmetric corner. Taught: the pitchfork
  is a *lower bound*. The smooth average captures one bistability mechanism and
  misses a second, spike-timing lock-in, which is intrinsically discrete. This
  is the most productive dead end --- it reframed a quantitative "failure" as
  the discovery of *two complementary mechanisms at two scales*.
]

#negresult[
  *Dead end 6 --- a latency gate separates the winner-take-all staircase.*
  Hoped: FCS's "stabilise within four ticks" gate is a speed test, so a
  transfer-function latency gate would carve the staircase out of the WTA
  region. Tried: gating cells by the slowest-mode decay rate. Failed: the gate
  changed agreement with FCS by $-0.003$; the eigenvalue distributions of
  staircase and non-staircase cells overlap almost entirely. Taught: the
  staircase cells are not slow --- they have respectable decay rates. Their
  redness is integer-tick determinism, orthogonal to any linear timescale. This
  ruled out the most plausible rung-2 rescue and sharpened the staircase
  diagnosis.
]

#negresult[
  *Dead end 7 --- the Siegert envelope generalises beyond two neurons.* Hoped:
  the $99.6%$-recall two-neuron WTA envelope carries to $N > 2$. Tried: Siegert
  fixed-point enumeration for $N = 2, 3, 4, 6, 10$. Failed: recall degrades
  monotonically with $N$ --- $0.90$ at $N = 2$ down to $0.00$ at $N = 10$ ---
  and at large $N$ the discrete oracle shows winner-take-all where *no*
  asymmetric fixed point exists (the *inverse staircase*). Taught: the
  high-recall envelope is a small-$N$ luxury. At larger $N$ the continuous lens
  fails in the *opposite* direction, under-predicting WTA; the continuous lens
  is wrong on *both* sides of the discrete boundary.
]

#headline[
  Every dead end above fails exactly where #ref(<sec-ladder-whole>)'s
  diagnostic principle predicts: each one tried to extract, from a continuous
  description, a fact about the integer reset schedule. The seven failures, on
  three mathematical frameworks and two phenomena, are not seven accidents ---
  they are seven confirmations of a single predictive principle from its
  negative side. Reported together, they *measure* the boundary of continuous
  analysis instead of merely bumping into it.
]

// ════════════════════════════════════════════════════════════════════════════
= Connection to the discrete programme, and the path to publication <sec-publication>
// ════════════════════════════════════════════════════════════════════════════

== Back to the discrete CogSpike paper <sec-back>

The prior CogSpike paper @cogspike_wd and this compendium are two responses to
the *same* problem --- the state-space explosion of exhaustive SNN verification
--- pulling in opposite directions. The discrete paper keeps the model exact
and *shrinks the chain*: its weight-discretized quotient abstraction collapses
bisimilar states, cutting a contralateral-inhibition model from $3603$ to $387$
states while preserving every verified winner-take-all property. This
compendium keeps the cost low and *leaves the chain*: it replaces exhaustive
enumeration with continuous approximation, accepting that some properties (the
staircase, the period, the waveform) are lost in exchange for polynomial cost
and analytic boundaries.

The two are genuinely complementary, and the same case study --- contralateral
inhibition, winner-take-all --- runs through both, which makes the pairing
concrete rather than rhetorical. The capability matrix (#ref(<tbl-capability>))
is, in effect, the contract between them: it states which questions the
continuous companion answers cheaply and which must still go to the discrete
model checker. The diagnostic principle is the clause that makes the contract
*decidable in advance* --- before running anything, one can say whether a given
property is in continuous scope.

One subtlety worth recording: the discrete paper's neuron is a *probabilistic*
LI\&F (threshold levels with firing probabilities, a Markov chain), while the
FCS oracle used as ground truth here is the *deterministic* FCS LI\&F (windowed
integrator, integer threshold). They are two LI\&F variants in one programme;
the continuous ladder was developed and validated against the deterministic
one.

== Forward to a paper <sec-forward>

This compendium is the internal, comprehensive record. A companion paper
draft, `paper/continuous_lens.typ`, distils it to a publishable narrative whose
thesis is the diagnostic principle: continuous analysis recovers the shape and
scale of LI\&F archetype behaviour, and the spike-reset rule is a decidable,
a-priori boundary on which formal properties it can certify. The paper leads
with the two-regime winner-take-all split as its case study, keeps the dead
ends as primary evidence rather than caveats, and closes on the methodology
ladder and the pre-filter workflow.

The single most publication-ready result is the *sound spectral pre-filter*: a
polynomial-cost certificate ($rho(A) < 0.544$) that provably skips
model-checking cells, composed in front of the exponential model checker. It is
modest in saving on the small archetypes measured here, but it is sound, and it
is a concrete instance of the continuous and discrete lenses working together.

== Open follow-ups <sec-followups>

Consolidating the open directions named across the seven threads, in
decreasing order of confidence:

- *Hybrid-system spectral theory.* Pole placement failed (dead ends 3--4)
  because the linear spectrum has no term for the reset. The principled fix is
  a linearization that *includes* the reset as a jump map --- saltation
  matrices, or Floquet theory of periodic-reset systems --- whose spectrum
  would carry the reset frequency explicitly. This is the natural route to a
  rung between $5$ and $6$ that *could* reach the integer period.
- *Dynamic recalibration done right.* Followup A falsified a naive
  recalibration; a calibration that fits the time constant against
  impulse-response data, separately from the static input--output curve, would
  close the transfer-function *magnitude* bias even though it cannot close the
  period gap.
- *The full age-structured mesoscopic theory.* Quasi-renewal here uses the
  simplest single-integral form. The full age-structured equations
  @Schwalger2017, which carry phase information beyond a single hazard, would
  sharpen the staircase prediction --- though the diagnostic principle predicts
  they still cannot close the $approx 0.30$ floor entirely.
- *The pre-filter as a delivered tool.* The sound $rho(A)$ certificate
  (#ref(<sec-reachability>)) is currently a result, not a tool. Composing it
  against a concrete Lustre encoding, so it actually shortens a Kind2 sweep,
  is a self-contained engineering deliverable.

#pagebreak()

// ════════════════════════════════════════════════════════════════════════════
= Appendix A --- Thread chronology <sec-chrono>
// ════════════════════════════════════════════════════════════════════════════

The compendium is organised by method, not by history. For the record, here is
the research as it actually unfolded: seven threads over roughly seven weeks,
each a phase-gated pipeline with its own advisor-facing note.

*Thread 1 --- the research notes and the analysis toolkit.* The opening move:
recognising that an LI\&F neuron's subthreshold dynamics are a first-order
linear recurrence, hence open to the classical engineering toolkit ---
eigenvalues, $Z$-domain transfer functions, Bode plots, fixed-point analysis.
Two notes were written, one for an engineering audience and one recasting the
same ideas for computer scientists (eigendecomposition as PageRank, spectral
radius as iterative-method convergence, fixed points as Markov-chain absorbing
sets). Validated on a contrived $4$-neuron winner-take-all network: the
dominant eigenvector named the winner, the spectral radius predicted settling
time, and every prediction was confirmed against PRISM and Monte Carlo. This is
rungs $1$--$2$, established. Its weakness --- never tested on a real FCS
archetype --- is what thread 2 addressed.

*Thread 2 --- archetypes / spectral cartography.* The honest test: do rung-$1$
spectral methods predict behaviour on the actual FCS contralateral and
negative-loop archetypes? The answer was the two-regime split
(#ref(<sec-wta-rung1>)): deterministic winner-take-all is combinatorial,
reachability winner-take-all is spectral, and the spike-reset rule is what
divides them. Pole placement for oscillation periods failed (dead end 3). This
thread produced the diagnostic principle and the central two-regime figure.

*Thread 3 --- population / Wilson--Cowan.* The lift to rung $3$: replace each
neuron with a population, average the reset into a smooth gain function, and
recover the full bifurcation toolkit. The winner-take-all boundary became an
exact pitchfork curve ($99.96%$ classification), oscillation onset an exact
Hopf curve, and the framework generalised to series, parallel, and positive-loop
archetypes. Cross-validation against the discrete simulator exposed dead end 5
--- the pitchfork is a lower bound, missing spike-timing lock-in. The note was
rewritten (v2) for an FCS-native audience, with the discrete-to-continuous
bridge table that #ref(<tbl-bridge>) descends from.

*Thread 4 --- closed-form (Siegert, $H(omega)$, quasi-renewal).* Rungs $4$--$5$:
replace the hand-tuned gain curve with three physically-derived objects --- the
Siegert static rate, the Richardson transfer function, the Naud--Gerstner
quasi-renewal mesoscopic. Siegert beat the hand-tuned curve; the transfer
function's self-consistency was exact at zero frequency; quasi-renewal recovered
the finite-size structure. The cross-validation against discrete spike trains
exposed the transfer-function *magnitude* bias --- a calibration limitation that
foreshadowed dead end 4.

*Thread 5 --- closed-form winner-take-all.* The three rung-$4$/$5$ lenses
brought onto the exact FCS Property 7 / Figure 10 staircase grid. Siegert is the
$99.6%$-recall envelope; the transfer-function latency gate is orthogonal to
the staircase (dead end 6); quasi-renewal partially dissolves it, with a
unimodal $N$-dependence quantified in a dedicated mechanism note. This thread
produced the $approx 0.70$ Jaccard floor.

*Thread 6 --- multi-neuron winner-take-all.* The same lenses extended from two
neurons to $N > 2$. The two-neuron envelope did not survive: recall collapses
with $N$, and the *inverse staircase* appears --- discrete winner-take-all below
the smooth-rate threshold (dead end 7). The continuous lens fails on both sides
of the discrete boundary.

*Thread 7 --- closed-form negative loop.* The three lenses on FCS Property 5
oscillation. Mean-field cannot produce the oscillation at all (it is a stable
spiral); the transfer function gets the direction but the period is a
structural $4 times$ too long (dead end 4, with the falsified-recalibration
followup); quasi-renewal recovers the period exactly. Followups extended the
result to a single-neuron renewal predictor and a three-neuron loop.

// ════════════════════════════════════════════════════════════════════════════
= Appendix B --- Per-thread phase scorecards <sec-scorecards>
// ════════════════════════════════════════════════════════════════════════════

Each thread was a phase-gated pipeline; each phase tested one falsifiable
hypothesis and recorded a verdict. The scorecards below put every verdict on one
page. "Pass / partial / fail" are the threads' own gate verdicts.

#let sc(title, rows) = [
  #text(weight: "bold", title)
  #table(
    columns: (auto, 1fr, auto),
    inset: 5pt, align: (left, left, center), stroke: 0.5pt,
    table.header([*Phase*], [*Hypothesis tested*], [*Verdict*]),
    ..rows,
  )
  #v(0.3em)
]

#sc("Thread 2 — archetypes / spectral cartography", (
  [0], [Reproduce FCS semantics; build deterministic ground truth], [pass],
  [1a], [Eigenvalue gap (scalar linearization) predicts WTA], [fail],
  [1b], [Full windowed-integrator linearization fixes 1a], [fail],
  [1c], [Spectral radius predicts WTA under reachability semantics], [pass],
  [2], [Two-regime split transfers to the delayer-augmented topology], [pass],
  [3], [Pole placement designs oscillation periods], [fail],
))

#sc("Thread 3 — population / Wilson–Cowan", (
  [0], [Mean-field infrastructure: solver, Jacobian, fixed points], [pass],
  [1], [Spectral gap predicts the WTA boundary ($99.96%$)], [pass],
  [2], [Pitchfork and Hopf loci derivable in closed form], [pass],
  [3], [Pole placement designs a target oscillation frequency], [partial],
  [4], [The WC pitchfork is the discrete LI\&F WTA boundary], [fail],
  [5], [The framework generalises to series / parallel / positive loop], [pass],
))

#sc("Thread 4 — closed-form (Siegert / H(ω) / quasi-renewal)", (
  [0], [Stochastic-LI\&F bridge reaches the diffusion regime], [pass],
  [1], [Siegert beats the hand-tuned gain curve], [pass],
  [2], [Transfer-function self-consistency exact at zero frequency], [pass],
  [3], [Transfer function predicts driven-response magnitude and phase],
    [partial],
  [4], [Quasi-renewal recovers finite-size bistability structure], [pass],
))

#sc("Thread 5 — closed-form winner-take-all", (
  [0], [Reproduce the FCS Property 7 staircase grid], [pass],
  [1], [Siegert is a high-recall WTA envelope], [pass],
  [2], [Transfer-function latency gate separates the staircase], [partial],
  [3], [Quasi-renewal finite-size noise dissolves the staircase], [partial],
))

#sc("Thread 6 — multi-neuron winner-take-all", (
  [0], [Reproduce FCS Property 7 at $N > 2$], [pass],
  [1], [The Siegert envelope generalises beyond two neurons], [partial],
  [2], [The transfer-function gate rescues the multi-neuron failure], [partial],
  [3], [Quasi-renewal converges to mean-field at every $N$], [pass],
))

#sc("Thread 7 — closed-form negative loop", (
  [0], [Reproduce FCS Property 5 oscillation], [pass],
  [1], [Siegert fixed point predicts where oscillation occurs], [partial],
  [2], [Transfer function predicts the oscillation period], [partial],
  [3], [Quasi-renewal recovers the oscillation period], [pass],
  [A], [Dynamic recalibration closes the period gap], [fail],
  [B], [A single-neuron renewal predictor recovers the binary waveform],
    [partial],
  [C], [The three-lens machinery generalises to a three-neuron loop], [pass],
))

#text(size: 9.5pt, style: "italic")[
  Thread 1 (the foundational research notes and analysis toolkit) was not
  phase-gated; it is summarised in #ref(<sec-chrono>).
]

// ════════════════════════════════════════════════════════════════════════════
= Appendix C --- Glossary <sec-glossary>
// ════════════════════════════════════════════════════════════════════════════

Plain-language definitions of every technical term, for a reader without a
differential-equations background. The section where each term is introduced in
full is given in brackets.

/ Bifurcation: a parameter value at which a system's qualitative behaviour
  changes --- a fixed point gains or loses stability, or splits. Bifurcations
  are the *boundaries* of behaviours in parameter space. (#ref(<sec-rung-wc>))
/ Bode plot: a plot of the transfer-function magnitude $|H(omega)|$ against
  driving frequency $omega$; it shows how nimbly a system tracks fast input.
  (#ref(<sec-rung-tf>))
/ Diffusion approximation: the assumption that a neuron's input is noisy enough
  to be treated as a mean plus Gaussian fluctuations; the basis of the Siegert
  formula. (#ref(<sec-rung-siegert>))
/ Eigenvalue: for a linear map, a number $lambda$ by which some direction is
  rescaled each step; $|lambda| < 1$ shrinks, $|lambda| > 1$ grows.
  (#ref(<sec-rung-lin>))
/ Eigenvalue gap: the margin by which the dominant eigenvalue stands above the
  rest; the spiking-network analogue of a Markov chain's spectral gap.
  (#ref(<sec-rung-lin>))
/ Fixed point: a state the dynamics map to itself --- an equilibrium. A network
  can have several; which one is stable matters. (#ref(<sec-primer>))
/ Hopf bifurcation: the bifurcation at which a stable fixed point gives way to
  a sustained oscillation; the boundary of oscillatory behaviour.
  (#ref(<sec-rung-wc>))
/ Jaccard index: an agreement score between two yes/no maps --- the size of
  their overlap divided by the size of their union; $1$ is perfect agreement.
  (#ref(<sec-wta>))
/ Jacobian: the matrix of the best linear approximation of a (nonlinear)
  dynamics at a fixed point; its eigenvalues decide local stability.
  (#ref(<sec-rung-lin>))
/ Linearization: replacing a nonlinear rule by its best linear approximation
  near an operating point. (#ref(<sec-primer>))
/ Mean-field: a description that replaces a population of units by their
  average, exact in the limit of infinitely many units. (#ref(<sec-rung-wc>))
/ Pitchfork bifurcation: the bifurcation at which one stable state splits into
  two --- the winner-take-all boundary of mutual inhibition.
  (#ref(<sec-rung-wc>))
/ Pole placement: choosing weights so that the linearized system has
  prescribed eigenvalues ("poles") --- inverse design. (dead end 3,
  #ref(<sec-deadends>))
/ Quasi-renewal: a finite-population theory that tracks the age distribution
  (time since last spike) and adds $1\/sqrt(N)$ noise; rung $5$.
  (#ref(<sec-rung-qr>))
/ Reachability: whether a target behaviour is attainable from *some* admissible
  initial state --- the question a model checker answers. (#ref(<sec-reachability>))
/ Siegert formula: the closed-form expression for a noisy integrate-and-fire
  neuron's firing rate, given its input mean and variance; rung $4$.
  (#ref(<sec-rung-siegert>))
/ Spectral radius $rho$: the largest eigenvalue magnitude; $rho < 1$ is the
  condition for a linear system to settle. (#ref(<sec-rung-lin>))
/ Spike-reset rule: the discontinuous jump of a neuron's state after it fires;
  the one nonlinearity that every continuous method erases, and the subject of
  the diagnostic principle. (#ref(<sec-intro>), #ref(<sec-ladder-whole>))
/ Staircase: the three diagonal blocks of synchronous-lock cells in the FCS
  winner-take-all grid --- the integer-tick structure no rate theory reproduces.
  (#ref(<sec-wta>))
/ Transfer function $H(omega)$: the complex number giving a linear system's
  gain and phase delay in response to input at frequency $omega$; rung $2$.
  (#ref(<sec-rung-tf>))
/ Wilson--Cowan equation: the smooth differential equation for a neural
  population's firing rate; the basis of rung $3$. (#ref(<sec-rung-wc>))
/ Winner-take-all (WTA): the outcome where one neuron in a competing pair fires
  steadily and the other falls silent. (#ref(<sec-wta>))

#v(1em)
#line(length: 100%)
#v(0.3em)
#text(size: 9pt, style: "italic")[
  Compendium of the CogSpike DEQ research programme, threads 1--7
  (`deq/research_note`, `deq/archetypes`, `deq/population`, `deq/closed_form`,
  `deq/closed_form_wta`, `deq/closed_form_wta_multi`,
  `deq/closed_form_neg_loop`). Figures are reproduced from the per-thread
  result directories; see `deq/final/README.md` for provenance. Companion
  paper draft: `paper/continuous_lens.typ`.
]

#bibliography("refs.bib", style: "ieee")
