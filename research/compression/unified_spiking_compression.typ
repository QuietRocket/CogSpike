// Unified research paper: Spikes as Bits, Learned
// A self-contained merge of the spiking-entropy-coder note and the
// learning-to-be-unsurprised note, with all confirmed critique fixes applied
// in-text and a folded-in critical self-evaluation.
// CogSpike / research/compression, June 2026

#set document(
  title: "Spikes as Bits, Learned",
  author: "CogSpike Research Team",
  date: datetime.today(),
)
#set page(paper: "a4", margin: (x: 2.5cm, y: 2.5cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")
// Numbering convention: a display equation is numbered iff it carries a label,
// i.e. iff the text refers back to it. Recalls, reminders, and proof-internal
// steps are shown unnumbered, so every visible number names an equation that is
// actually cited. Unlabeled block equations are reconstructed with no number and
// their counter step is undone, keeping the numbered equations contiguous.
#show math.equation: it => {
  if it.block and it.numbering != none and not it.has("label") {
    counter(math.equation).update(n => n - 1)
    math.equation(it.body, block: true, numbering: none)
  } else { it }
}

#let definition(body) = block(width: 100%, inset: 8pt,
  stroke: (left: 2pt + gray), [*Definition.* #body])
#let theorem(title, body) = block(width: 100%, inset: 8pt,
  stroke: (left: 2pt + blue), [*Theorem* (#title)*.* #body])
#let proposition(body) = block(width: 100%, inset: 8pt,
  stroke: (left: 2pt + green), [*Proposition.* #body])
#let proof(body) = block(width: 100%, inset: 8pt, fill: luma(248),
  [_Proof._ #body #h(1fr) $square$])
#let intuition(body) = block(width: 100%, inset: 8pt, fill: rgb("#f0f7ff"),
  stroke: (left: 2pt + rgb("#4a90d9")), [💡 *Intuition:* #body])
#let remark(body) = block(width: 100%, inset: 8pt,
  stroke: (left: 2pt + orange), [*Remark.* #body])
#let background(title, body) = block(width: 100%, inset: 8pt,
  fill: rgb("#f1faf3"), stroke: (left: 2pt + rgb("#3a9d5d")),
  [📘 *Background — #title.* #body])
#let honest(body) = block(width: 100%, inset: 8pt, fill: rgb("#fff7f0"),
  stroke: (left: 2pt + rgb("#d98a4a")), [⚖️ *Honesty check:* #body])
#let openproblem(body) = block(width: 100%, inset: 8pt, fill: rgb("#f5f0ff"),
  stroke: (left: 2pt + rgb("#8a6ad9")), [🔭 *Open problem:* #body])

// Multi-letter math operators used across the merged sections.
#let softmax = math.op("softmax")
#let argmax = math.op("argmax")
#let sign = math.op("sign")
#let repair(name, ids, was, fixed) = block(width: 100%, breakable: true,
  stroke: (left: 1.5pt + luma(200)), inset: (left: 8pt, y: 3pt))[
  *#name* (#ids). #linebreak()
  _Was:_ #was #linebreak()
  _Fixed:_ #fixed
]

#align(center)[
  #text(size: 17pt, weight: "bold")[Spikes as Bits, Learned]
  #v(0.2em)
  #text(size: 12pt)[A Verifiable Spiking Entropy Coder, Its Local Energy-Descending Plasticity Rule, and a Critical Self-Evaluation]
  #v(0.4em)
  #text(size: 10pt, style: "italic")[
    `research/compression/`, June 2026 · CogSpike research note (canonical; supersedes the two-note series)
  ]
]
#v(0.5em)

#block(width: 100%, inset: 10pt, fill: luma(247), radius: 4pt)[
  *Abstract.* We construct a recurrent spiking neural circuit whose spiking *time* is, exactly, the information cost of its own predictions, derive the local plasticity rule that makes those predictions good, and fold in a meticulous critical self-evaluation of both results. The construction rests on one calibration: a leaky integrate-and-fire readout driven by $R I(q) = theta slash (1 - q^alpha)$ fires at first-spike latency $t^*(q) = -lambda log_2 q$, so a symbol's surprisal is its spike time exactly, with no integer-bit rounding penalty (a *real*-valued, continuous-time advantage we prove and bound below by timing resolution). Summed over a stream, total spike-time equals the *cross-entropy* of the circuit's predictions against the source; Shannon's entropy rate is the floor, and the excess is the Kullback–Leibler divergence between the circuit's internal model $q$ and the world's law $p$. The predictor is made concrete: $q(dot mid(|) c_t) = "softmax"(W c_t)$, where the running context $c_t$ is held on a line/ring attractor and the softmax — *not* divisive normalization, which is only its biophysical $sigma arrow 0$ approximation — is what enforces $sum_j q_j = 1$ exactly. We then prove that stochastic gradient descent on the per-symbol spike cost is a local three-factor Hebbian rule $Delta W_(i j) = eta thin c_i (y_j - q_j)$, whose post factor is the signed residual carried by a *distinct* two-channel error population — distinct from the latency code, not the same spikes. *Conditional on that realization*, the cost the circuit pays is its own gradient. The excess energy is a Lyapunov function for the averaged flow $dot(W) = -eta nabla V$; the loss is convex in the logits and the unique fixed point (up to softmax gauge) is $q = p$, so the circuit learns the true conditional law and descends to the entropy-rate floor. We separate the time result, which is *exact and proven*, from the energy reading, which we recast as a *model-dependent corollary* under an explicit per-spike biophysical cost ($lambda$ in seconds-per-bit is not $kappa$ in joules-per-bit). Every claim is a checkable property; we keep the structural/safety obligations (predictor-independent losslessness, settled mutual exclusion, sum-mode invariance, gauge-fixed weight boundedness) separate from the quantitative/optimality ones (near-floor cost, monotone energy descent), and discharge them in the model-checking language the CogSpike program already uses. A dedicated Critical Evaluation section accounts for all fifty confirmed critique findings — fixed in text or flagged open — with a reproduced-numerics table. The unifying thesis: on a structured source, compressing better, predicting better, spending less energy, and learning are one monotone descent on one number, performed by a rule the substrate can physically run.
]

= Notation
#table(
  columns: (auto, 1fr, auto),
  align: (left, left, left),
  stroke: 0.5pt + luma(180),
  table.header[*Symbol*][*Meaning*][*Units*],
  [$t$], [Continuous time; also discrete stream step index $t = 1, dots, n$], [s; (index) —],
  [$V(t)$], [Membrane potential of a LIF neuron], [V (or dimensionless, scaled)],
  [$dot(V)$], [Time derivative $d V slash d t$], [V/s],
  [$tau$], [Membrane time constant (LIF leak timescale)], [s],
  [$R$], [Input resistance of the LIF neuron], [$Omega$],
  [$I$, $I(t)$], [Input (drive) current; $R I$ is the equilibrium target of $V$], [A; $R I$ in V],
  [$theta$], [Firing threshold on $V$], [V (same as $V$)],
  [$t^*$, $t^*(q)$], [First-spike latency; calibrated to $-lambda log_2 q$], [s],
  [$t_min$], [Minimum latency from refractory period / current saturation], [s],
  [$t_("settle")$], [WTA settling time: bound on the decode-window transient before a unique winner latches], [s],
  [$delta t$], [Timing resolution (clock period) of the latency readout], [s],
  [$lambda$], [Time-per-bit calibration constant (latency code); *exact, proven*], [s/bit],
  [$alpha$], [Calibration exponent $alpha = lambda slash (tau ln 2)$], [dimensionless],
  [$kappa$], [Energy-per-bit substrate constant (energy corollary); *model-dependent*], [J/bit],
  [$cal(X)$], [Finite symbol alphabet (rover: ${U, D, L, R}$)], [—],
  [$x_t$], [Realized symbol emitted at step $t$], [—],
  [$p(x)$, $p$], [True source symbol law], [dimensionless prob.],
  [$q(x)$, $q$, $q_j$], [Circuit's internal model probability; $q_j = q(j mid(|) c_t)$], [dimensionless prob.],
  [$q_(i j)$], [Predicted prob. of symbol $j$ in context $i$ (row $i$ of model)], [dimensionless prob.],
  [$q_max$], [Max representable prob. given finite drive ($approx 0.85$–$0.95$)], [dimensionless prob.],
  [$q_min$], [Noise-dominated lower edge of the exactly-representable probability band], [dimensionless prob.],
  [$pi$, $pi_i$], [Stationary distribution of the Markov source ($pi P = pi$)], [dimensionless prob.],
  [$P$, $P_(i j)$], [Markov transition matrix; $P_(i j) = Pr(x_(t+1)=j mid(|) x_t=i)$], [dimensionless prob.],
  [$s$], [Momentum-rover stickiness (memory dial), $s in [0,1)$], [dimensionless],
  [$I_q(x)$], [Surprisal (information content) $-log_2 q(x)$], [bits],
  [$H(p)$], [Entropy of the source (marginal)], [bits/symbol],
  [$overline(H)(p)$, $overline(H)(P)$], [Entropy *rate* of the source (conditional floor)], [bits/symbol],
  [$H(p,q)$, $overline(H)(p,q)$], [Cross-entropy / cross-entropy *rate* of model vs source], [bits/symbol],
  [$D_("KL")$, $overline(D)_("KL")(p||q)$], [Kullback–Leibler divergence / divergence rate, $gt.eq 0$], [bits/symbol],
  [$I(x_t; x_(t-1))$], [Mutual information between consecutive symbols ($= H(pi) - overline(H)(P)$)], [bits/symbol],
  [$ell$, $ell_i$, $ell(x)$], [Codeword length (coding theory); ideal $ell_i = -log_2 p_i$], [bits],
  [$L$, $L^*$], [Expected / optimal expected code length], [bits/symbol],
  [$C(x)$], [Binary codeword assigned to symbol $x$], [bit string],
  [$P("sequence")$], [Probability of an entire sequence (arithmetic coding interval width)], [dimensionless prob.],
  [$ell_t$], [Per-symbol energy/loss $-log_2 q(x_t mid(|) c_t)$ (surprisal of realized symbol)], [bits],
  [$c_t$, $c_i$], [Context state (attractor bump); $c_i$ = activity of context line $i$], [dimensionless (activity)],
  [$W$, $W_(i j)$], [Lateral weight from context $i$ to readout $j$ (a logit / log-odds)], [dimensionless],
  [$W_max$], [Bound on the gauge-fixed (centered) weight], [dimensionless],
  [$a_j$], [Logit (net drive) of readout $j$, $a_j = sum_i W_(i j) c_i$], [dimensionless],
  [$y_j$], [One-hot outcome indicator $bb(1)[x_t = j]$], [dimensionless ${0,1}$],
  [$r_j$], [Signed residual error $y_j - q_j in [-1, 1]$ (the teaching signal)], [dimensionless],
  [$r_j^+$, $r_j^-$], [Rectified ON / OFF error channels, $r_j = r_j^+ - r_j^-$], [dimensionless $gt.eq 0$],
  [$eta$, $eta_t$], [Learning rate / global three-factor gate; schedule $eta_t = eta_0 slash (1 + t slash t_0)$], [dimensionless],
  [$e_(i j)$, $gamma$], [Eligibility trace at synapse $(i,j)$ and its decay factor (future work)], [dimensionless],
  [$V(W)$], [Lyapunov function = excess energy $overline(H)(p,q) - overline(H)(p) = sum_i pi_i D_("KL")(P_(i dot) || q_(i dot))$], [bits/symbol],
  [$nabla V$, $dot(V)$], [Gradient of $V$ in $W$; its time derivative along the flow], [bits/symbol per $W$; bits/symbol/s],
  [$nu_1, nu_2$, $nu_i$], [Activities of competing WTA readout units], [dimensionless (activity)],
  [$u$, $d$], [Sum mode $nu_1 + nu_2$ and difference mode $nu_1 - nu_2$], [dimensionless (activity)],
  [$sigma$], [Divisive-normalization semi-saturation constant ($r_i = a_i slash (sigma + sum_j a_j)$)], [dimensionless (activity)],
  [$epsilon$], [Tolerance band above the entropy-rate floor in cost obligations], [bits/symbol],
  [$n$, $N$], [Stream length; alphabet size], [—],
)

// ===== S1-orientation =====
= Orientation: the claim, what we build, and what we check

A long-standing idea binds three words that, on their face, sound unrelated:

#block(inset: (left: 10pt, top: 4pt, bottom: 4pt))[
  *compression* $space arrow.l.r space$ *prediction* $space arrow.l.r space$
  *intelligence.*
]

The middle equivalence is the exact one, and it is old. To compress a stream you
must predict it: a coder that knows the probability of the next symbol can spend
few bits on likely symbols and many on rare ones, and Shannon's source-coding
theorem fixes the floor it is racing toward at the source's entropy
@shannon1948. The converse holds too — anything that compresses well must, inside
it, *be* predicting well: it must carry an accurate model of the data's
structure. That two-way street is the content of the *minimum description length*
(MDL) principle, which makes "the best model is the one that compresses the data
most" a formal criterion rather than a slogan @rissanen1978. The flanking word
*intelligence* is the informal wager that *building such a model is what
understanding is.* This paper's aim is to make all three words mean something
concrete and *checkable* in a *spiking neural network* (SNN), and in particular to
pin down what the SNN's defining feature — that it is a *circuit with cycles*, not
a feedforward pipeline — actually buys.

We assume a reader fluent in *formal methods*: probabilistic model checking,
discrete- and continuous-time Markov models, temporal logic, the safety/liveness
distinction, and Lyapunov stability as it is used in verification. Everything else
— neural dynamics, information theory, coding theory, plasticity, dynamical
systems — we build from first principles as we need it. The compression-as-
modeling thesis itself, and its neural-network reading through MDL and the
bits-back description-length account of network weights, are inherited from the
literature @shannon1948 @rissanen1978 @hintonvancamp1993; our contribution is a
*spiking, recurrent, verifiable* realization of it, together with the learning
rule that closes it and an unusually thorough audit of where the realization
holds and where it strains.

== What this paper does, in three movements

This document is a self-contained merger of two research notes. It does three
things, in order, and a reader should hold the boundaries between them in mind
because the paper's honesty depends on it.

#block(inset: (left: 6pt))[
  *(a) It builds the coder.* We construct a recurrent spiking circuit whose
  per-symbol cost is, exactly, the information-theoretic cost of its own
  predictions. The construction rests on one observation: a leaky neuron driven
  precisely to threshold by an *expected* input can be made to fire *late* in
  proportion to how surprising the input is, so that a calibrated readout's
  first-spike *time* equals the surprisal $-log_2 q(x_t mid(|) c_t)$ of the
  realized symbol under the circuit's internal model $q$. Summed over a stream,
  the total spike-time equals the *cross-entropy* of the model against the source;
  the entropy rate is the floor and the excess is the Kullback–Leibler divergence
  between model and world. This calibration is *exact and time-valued*: the proven
  observable is a spike time measured in seconds, scaled by a calibration constant
  $lambda$ (s/bit). We keep this distinct, throughout, from any reading of the
  cost in *joules*, which is a separate and model-dependent corollary governed by
  a different constant $kappa$ (J/bit) — the two are never silently swapped.

  *(b) It derives the learning rule.* The first note left exactly one thing
  assumed rather than built: the learning that makes the predictions good. We
  supply it. Gradient descent on the per-symbol surprisal is shown to be a
  *local, three-factor Hebbian rule* $Delta W_(i j) = eta thin c_i (y_j - q_j)$ —
  presynaptic context $c_i$, times a postsynaptic residual $y_j - q_j$, times a
  global gate $eta$. The gradient *algebra* is unconditional and standard (the
  delta / Widrow–Hoff rule @widrowhoff1960); the genuine — and *conditional* —
  novelty is the physical reading, that the residual the circuit must *emit* is a
  teaching signal. We prove descent with the excess energy as a Lyapunov function
  and certify convergence to the true conditional law $q = p$.

  *(c) It folds in a meticulous critical self-evaluation.* The two notes, written
  separately, clashed in places and overclaimed in others. Rather than quietly
  paper over this, the paper carries a dedicated *Critical Evaluation* section
  that accounts for every confirmed finding of an adversarial critique — which we
  *fixed in-text* and which we *flag open* — and reproduces the validators'
  numbers in a single table. The self-audit is part of the contribution, not an
  afterthought: a claim we cannot check, or can only check under a stated
  hypothesis, is reported as such.
]

== Four load-bearing reconciliations the reader should carry

Merging the two notes forces four corrections that recur everywhere downstream.
We state them once here so the later sections can cite them rather than
relitigate them; each is justified in place.

#block(inset: (left: 6pt))[
  *(i) The simplex-normalizer is the softmax; divisive normalization is only its
  biophysical approximation.* The predicted distribution is
  $q_j = "softmax"_j(W c_t)$ with logits $a_j = sum_i W_(i j) c_i$, and the
  partition-of-unity invariant $sum_j q_j = 1$ holds *exactly* because of the
  softmax. Carandini–Heeger divisive normalization
  $r_i = a_i slash (sigma + sum_j a_j)$ @carandiniheeger2012 sums to strictly less
  than one; it is the gain-control realization of the softmax, valid only as
  $sigma arrow 0$, not the thing that enforces normalization.

  *(ii) The predictor the first note left unspecified is the one the second note
  supplies.* The map $c_t arrow q(dot mid(|) c_t)$ — the largest gap in the coder
  as first written — is exactly $q_j = "softmax"_j(W c_t)$; with a one-hot context
  $c = e_i$, each row $W_(i dot)$ is the belief "what follows symbol $i$." The
  calibration drive encodes precisely this $q_j$. Stated once, the gap closes.

  *(iii) Time is exact and proven; energy is a model-dependent corollary.* The
  latency identity $t^* = -lambda log_2 q$ (seconds) and the spike-time =
  cross-entropy result are exact. The energy reading ($kappa dot$ surprisal,
  joules) is a *separate* corollary under an explicit minimal biophysical model,
  not an identity — we flag where it fails (a long-latency rare symbol may draw
  less instantaneous power yet accumulate more leak).

  *(iv) The latency code and the error population are distinct observables.* The
  coder of movement (a) carries surprisal in spike *time*: one race-to-threshold
  readout per symbol. The learning residual $r_j = y_j - q_j$ of movement (b) is a
  *separate* signal, realized by two rectified ON/OFF error channels — *not* the
  same spikes. When we say "the cost spike is the teaching signal," we prove only
  the gradient algebra unconditionally and state the physical step as an explicit
  hypothesis imported from the predictive-subtraction architecture.
]

== The plan, and the order in which we supply background

The merged sections proceed from first principles to the assembled, learning,
audited circuit:

#block(inset: (left: 6pt))[
  We first introduce *spiking neurons* from a single differential equation, and
  the one trick — *predictive subtraction*, realized as feedback inhibition — that
  genuinely needs a cycle. We then build the *information measures* (surprisal,
  entropy, cross-entropy, divergence) and just enough *coding theory* (prefix
  codes, the Kraft inequality @kraft1949 @mcmillan1956, the integer-bit penalty
  and its escape via arithmetic coding @rissanenlangdon1979) to say what "optimal"
  means. With those in hand we prove the central *latency calibration* exactly,
  connecting it to time-to-first-spike coding @thorpe1996 @thorpe2001. We
  introduce *sources with memory* (Markov chains, entropy rate @shannon1948) and
  fix a one-parameter running example whose memory is worth a closed-form amount.
  We supply the *recurrent ingredients* — line/ring attractors for graded memory
  @seung1996 @benyishai1995, winner-take-all selection, and divisive gain control
  @carandiniheeger2012 — then *assemble the circuit* and certify it through a
  sum/difference mode decomposition that cleanly separates *unconditional
  correctness* from *earned optimality*. We then *derive the learning rule*,
  grounding it in the delta rule @widrowhoff1960, surprise-driven conditioning
  @rescorlawagner1972, predictive coding @raoballard1999, three-factor plasticity
  @fremauxgerstner2016, and — as a degenerate, point-estimate special case only —
  the free-energy principle @friston2010, and prove its descent and convergence
  under a decreasing Robbins–Monro schedule @robbinsmonro1951. Finally we write the
  *verification obligations* in probabilistic temporal logic, gather the
  *intelligence claim* into one precise sentence, and close with the *Critical
  Evaluation* and outlook — the latter pointing, with appropriate hedging, at the
  channel-coding regime where loopy belief propagation makes cyclic computation
  genuinely advantageous @yedidia2005 @gallager1962.
]

#intuition[
  The thread to hold onto: $H(p, q) = H(p) + D_("KL")(p || q)$. The cross-entropy
  the circuit pays splits into what the *world* costs (irreducible) and what
  *being wrong* costs (avoidable). Movement (a) makes that bill a literal spike-
  time; movement (b) shows a local rule drains the avoidable part to zero;
  movement (c) checks, line by line, that we are entitled to say so. "Compress
  better," "predict better," and "learn" turn out to be one monotone descent on
  one number — and the discipline of this paper is to keep claiming exactly that,
  and nothing more.
]

// ===== S2-neuron =====
= Spiking neurons, from one differential equation <sec-neurons>

#background("what a spiking neural network is")[
  An SNN is a network of *neurons* that communicate by *spikes* — instantaneous,
  identical electrical pulses — rather than by continuous real numbers. Each
  neuron maintains an internal scalar, its *membrane potential* $V$, which
  integrates incoming spikes; when $V$ reaches a *threshold* the neuron emits a
  spike of its own and resets. Information is carried by *which* neurons spike
  and *when*. A *feedforward* network wires neurons in layers, signals flowing
  one way; a *recurrent* network additionally has cycles — a neuron's output can
  loop back and influence its own future input. SNNs are inherently recurrent,
  circuit-like objects, and that is the feature we will exploit.
]

We need exactly one differential equation, and we build it from the leaky-bucket
picture.

#definition[
  A *leaky integrate-and-fire* (LIF) neuron has membrane potential $V(t)$ that
  decays toward a rest level while integrating input current $I(t)$:
  $ tau dot(V)(t) = -V(t) + R I(t), $ <eq-lif>
  where $tau > 0$ is the membrane time constant (in seconds), $R$ the input
  resistance, and $dot(V) = d V slash d t$ the time derivative. On reaching a
  threshold $theta$ the neuron emits a spike and $V$ resets to $0$. Between
  spikes this is a linear first-order ordinary differential equation (ODE).
]

#background("reading a first-order ODE")[
  An ODE relates a quantity to its own rate of change. Here $dot(V) = (-V + R
  I) slash tau$ says: the rate at which $V$ changes equals (target $-$ current
  value), scaled by $1 slash tau$. Whenever the right-hand side is zero the
  quantity stops moving — that is an *equilibrium* (here $V = R I$). The
  constant $tau$ sets the timescale: large $tau$ means sluggish, small $tau$
  means snappy. "Solving" the ODE means finding $V(t)$ as an explicit function
  of time given a starting value $V(0)$. First-order means only the first
  derivative $dot(V)$ appears, so one initial condition $V(0)$ fixes the whole
  trajectory; linear means the right-hand side is an affine function of $V$, so
  the solution is a single decaying exponential plus a constant — no oscillation,
  no chaos, just monotone relaxation toward equilibrium.
]

#intuition[
  Read $tau dot(V) = -V + R I$ as a leaky bucket. $R I$ is the inflow rate; the
  $-V$ term is a leak proportional to how full the bucket already is. Left alone
  with constant inflow, the level settles where inflow balances leak, at $V = R
  I$. A spike is the bucket brimming over a rim at height $theta$; resetting to
  $0$ empties it. Everything we need is "how long until the next overflow."
]

For constant drive $I$ from rest $V(0) = 0$, the ODE @eq-lif solves to
$ V(t) = R I (1 - e^(-t slash tau)), $ <eq-charge>
a curve rising monotonically toward its equilibrium $R I$. (One verifies
@eq-charge by substitution: $tau dot(V) = R I e^(-t slash tau) = R I - V$, and
$V(0) = 0$.) If $R I > theta$ the curve crosses threshold; setting $V(t^*) =
theta$ and solving for $t^*$ gives the *first-spike latency*
$ t^*(I) = tau ln ( (R I) / (R I - theta) ). $ <eq-latency>

This single formula is the heart of everything downstream. Note two ways to read
a neuron's output, both of which the construction will use:

- *Count reading.* A neuron driven harder overflows more often; spike *count*
  over a window grows with the drive. A surprising input means a large drive
  means a burst of spikes.
- *Latency reading.* A neuron driven just barely above threshold fills slowly
  and fires *late*; a strongly driven neuron fires *early*. The first-spike
  *time* carries the message, and the spike *delimits its own symbol* — the
  instant it happens, the symbol is over. Coding a value in the *timing* of the
  first spike is the *time-to-first-spike* (TTFS), or rank-order latency,
  scheme @thorpe2001 @thorpe1996; it is the reading the exact identity later
  rests on.

== Predictive subtraction: the one operation that needs a cycle

The latency law @eq-latency reacts to whatever current $I(t)$ arrives at the
neuron. The mechanism that makes a spiking circuit a *compressor*, rather than a
mere transducer, is to arrange that the current reaching the soma is not the raw
input but the input *minus what the circuit expected*. That subtraction is the
one operation a feedforward stage cannot perform across time — it requires a
path carrying a prediction back to the input — and it is the reason an SNN's
defining cyclic structure is load-bearing rather than incidental.

#background("predictive coding and prediction-error units")[
  *Predictive coding* is the hypothesis that a neural circuit represents not its
  raw input but the *error* between the input and an internally generated
  prediction of it @raoballard1999. A higher stage sends down a prediction; a
  lower stage subtracts it and forwards only the *residual* — the part of the
  signal the prediction failed to explain. A perfectly predicted input produces
  no residual and so costs nothing to transmit. The idea has a long lineage in
  surprise-driven learning, where only the *unexpected* part of a stimulus drives
  adaptation @rescorlawagner1972. We borrow exactly the residual-forwarding
  picture and read its cost in spikes.
]

The naive statement of this trick — "a feedback loop charges the neuron with
subthreshold depolarization equal to the input it expects, so the neuron sees
only $"input" - "prediction"$" — has the *wrong sign*, and we correct it here
because the sign is what makes the mechanism biophysically real rather than a
convenient fiction.

#remark[
  *The sign of cancellation (fix F).* The drive the circuit wants to cancel is
  *excitatory*: an expected input depolarizes the soma, pushing $V$ *up* toward
  threshold. To cancel an excitatory drive one must apply an *opposing*,
  *inhibitory* current — feedback that hyperpolarizes, or that shunts (clamps the
  membrane toward rest by transiently raising its conductance). Adding a *second*
  depolarization "equal to the prediction" would *double* the drive, not cancel
  it. So predictive subtraction is realized physically as *feedback inhibition* /
  *dendritic shunting*: the recurrent loop drives an inhibitory pathway whose
  current is tuned to oppose the expected excitatory input, leaving the soma to
  integrate only the *unpredicted residual*.
]

Write the soma's effective drive as the excitatory input $I_"in"(t)$ minus an
inhibitory prediction current $I_"pred"(t)$ supplied by the loop:
$ I_"eff"(t) = I_"in"(t) - I_"pred"(t). $ <eq-predsub>

#honest[
  *Exact additive subtraction is an idealization (fix F).* Equation @eq-predsub
  treats inhibition as a clean *subtractive* (hyperpolarizing) current, so the
  residual is exactly $I_"in" - I_"pred"$. Real synaptic inhibition is partly
  *shunting* (*divisive*): opening inhibitory conductance $g_"inh"$ adds a term
  $-g_"inh"(V - E_"inh")$ to @eq-lif, which scales the membrane's response rather
  than subtracting a fixed current, and drives $V$ toward the (near-rest)
  reversal potential $E_"inh"$ rather than below it. The exact additive form
  @eq-predsub is recovered in the regime where the inhibitory reversal sits at
  rest and the conductance change is small relative to the leak, so that the
  shunt acts, to first order, as a subtractive current matched to $I_"pred"$.
  Outside that regime the cancellation is approximate; the residual still
  *shrinks* with a good prediction, but is no longer the exact difference. We
  carry the idealized additive form @eq-predsub forward as the analyzed model and
  flag this as the gap between it and a conductance-based implementation.
]

#intuition[
  *Why the cycle is the active ingredient.* With predictive subtraction in place,
  a perfectly predicted input produces $I_"eff" approx 0$: near-zero residual,
  near-zero (or maximally delayed) firing. A surprising input produces a large
  residual and an early, strong spike. The emitted spike train therefore stops
  being a copy of the symbol stream and becomes a *surprise* stream — exactly the
  quantity the next sections make numerical. Crucially, $I_"pred"(t)$ must be
  computed from the *past* and routed *back* to the input of the very neuron it
  modulates; a strictly feedforward pipeline has no such return path. This is the
  single place where the recurrent, cyclic nature of an SNN does work that no
  finite feedforward unrolling reproduces, and it is the mechanism the rest of
  the note quantifies.
]

We have what this section owes the construction: a single ODE @eq-lif, its closed
solution @eq-charge, the latency law @eq-latency that turns drive into spike
*time*, and the predictive-subtraction loop @eq-predsub — stated with the correct
inhibitory sign — that turns that spike time into a measure of *surprise*. The
exact calibration of drive to surprisal, and the specific predictor
$q(dot mid(|) c_t)$ whose value the drive encodes, are supplied in the later
sections; here we have only fixed the neuron and the cyclic operation it needs.

// ===== S3-info-coding =====
= Information and codes, measured in bits <sec-info>

We now make "surprise" numerical, and then fix the benchmark — coding theory — against which the spiking coder of the next sections will be called near-optimal. Both stories are due to @shannon1948; we build them from first principles for the formal-methods reader.

Let a source emit symbols from a finite alphabet $cal(X)$, the true symbol law being $p(x)$; let $q(x)$ be the probabilities the circuit's *internal model* assigns. We will keep $p$ (world) and $q$ (model) rigorously distinct: the gap between them is the whole subject.

== Surprisal, entropy, cross-entropy, divergence

#definition[
  The *information content* (or *surprisal*) of a symbol $x$ under model $q$ is
  $ I_q (x) = -log_2 q(x) = log_2 (1 slash q(x)) "bits". $ <eq-surprisal>
  The *entropy* of the source is the expected surprisal under the *true* law,
  $ H(p) = -sum_x p(x) log_2 p(x), $ <eq-entropy>
  the *cross-entropy* is the expected surprisal a model $q$ pays on data drawn from $p$,
  $ H(p, q) = -sum_x p(x) log_2 q(x), $ <eq-crossent>
  and their difference is the *Kullback–Leibler (KL) divergence*
  $ D_("KL")(p || q) = H(p, q) - H(p) = sum_x p(x) log_2 (p(x)) / (q(x)) >= 0, $ <eq-kl>
  which is zero iff $q = p$.
]

#background("why the logarithm, and why these are the right measures")[
  Two demands shape the surprisal. First, surprise should *decrease* with probability: a certain event ($q arrow 1$) carries no information ($I arrow 0$), an impossible one ($q arrow 0$) infinite information. Second, the surprise of *independent* events should *add*: learning two unrelated facts is twice the news. But independent probabilities *multiply*, and among continuous functions that turn products into sums, $f(a b) = f(a) + f(b)$, the logarithm is the canonical choice — and, under a mild monotonicity or measurability side-condition, the only one up to the choice of base @shannon1948. We adopt $I_q = -log q$ on this basis; base $2$ merely names the unit "bit" (one bit = the information in one fair coin flip). We do not claim the logarithm is forced by the additivity equation *alone* — pathological non-measurable solutions of $f(a b) = f(a) + f(b)$ exist; it is forced once one also asks for monotonicity (more probable, less surprising). *Entropy* @eq-entropy is then the average surprise of the source — its irreducible uncertainty per symbol. *Cross-entropy* @eq-crossent is the average surprise your *model* registers, which is at least the entropy and equals it only when the model is exactly right. The excess gap is the *KL divergence* @eq-kl: a non-negative number measuring how wrong $q$ is, in bits.
]

#intuition[
  The decomposition $H(p, q) = H(p) + D_("KL")(p || q)$ is the entire thesis in one line. $H(p)$ is what the *world* costs — irreducible. $D_("KL")$ is what *being wrong* costs — the model's avoidable waste. A learner can only lower its cross-entropy by shrinking $D_("KL")$, i.e. by making $q$ resemble the true $p$ — which is to say, by *coming to understand the source.* Downstream, $H(p, q)$ will literally be the circuit's spike-time bill, and $D_("KL")$ its excess.
]

#remark[
  Non-negativity of the divergence @eq-kl is Gibbs' inequality, a one-line consequence of the concavity of $log$ (Jensen): $-D_("KL")(p||q) = sum_x p(x) log_2 (q(x) slash p(x)) <= log_2 sum_x p(x) (q(x) slash p(x)) = log_2 sum_x q(x) = 0$. Equality holds iff $q = p$. This inequality is exactly the statement that no model beats the truth, and it is the floor every later optimality claim rests on.
]

== Codes, prefix-freeness, and the price of whole bits

To say the circuit is "near-optimal" we must say optimal *against what*. The benchmark is the theory of lossless codes — which is also what makes the continuous-time advantage of the spiking coder legible later.

#background("codes, prefix-freeness, and the Kraft inequality")[
  A *binary code* assigns each symbol $x$ a finite bit string $C(x)$ of length $ell(x)$; a message is sent by concatenating codewords. A code is *prefix-free* (equivalently *instantaneous*) if no codeword is an initial segment of another. Then a receiver reading left to right knows a codeword has ended the moment it completes one — no delimiters, no lookahead. Geometrically, reserving a codeword of length $ell$ claims a sub-interval of $[0, 1)$ of width $2^(-ell)$ (all infinite strings beginning with that codeword), and prefix-freeness says the claimed intervals must not overlap. This is the *Kraft inequality* @kraft1949: a prefix-free binary code with lengths ${ell_i}$ exists iff
  $ sum_i 2^(-ell_i) <= 1. $ <eq-kraft>
  Remarkably, the same inequality @eq-kraft is necessary for *every uniquely decodable* code, not just prefix-free ones @mcmillan1956 — so demanding instantaneous decodability costs nothing in achievable length.
]

The Kraft inequality @eq-kraft turns code design into a constrained optimization, and that optimization is what pins the entropy as a floor.

#proposition[
  Minimizing the expected length $L = sum_i p_i ell_i$ over real lengths subject to the Kraft constraint @eq-kraft is solved by the *ideal lengths* $ell_i = -log_2 p_i$, attaining $L = H(p)$. With the integrality constraint $ell_i in bb(Z)_(>= 0)$ restored, the optimum $L^*$ obeys *Shannon's bound*
  $ H(p) <= L^* < H(p) + 1. $ <eq-shannon-bound>
]

#proof[
  Relaxing integrality, a Lagrange multiplier on $sum_i 2^(-ell_i) = 1$ (the constraint is tight at the optimum) gives $partial_(ell_i) (sum_j p_j ell_j - mu sum_j 2^(-ell_j)) = p_i + mu (ln 2) 2^(-ell_i) = 0$, so $2^(-ell_i) prop p_i$; normalizing forces $2^(-ell_i) = p_i$, i.e. $ell_i = -log_2 p_i$ and $L = sum_i p_i (-log_2 p_i) = H(p)$. This is the lower bound: by Gibbs' inequality no Kraft-feasible integer code beats it. For the upper bound, the *Shannon code* $ell_i = ceil(-log_2 p_i)$ is Kraft-feasible (since $2^(-ceil(-log_2 p_i)) <= 2^(log_2 p_i) = p_i$ summing to $<= 1$) and pays $L = sum_i p_i ceil(-log_2 p_i) < sum_i p_i (-log_2 p_i + 1) = H(p) + 1$.
]

The "$+1$" in @eq-shannon-bound is the *integer-bit penalty*: the cost of spending a whole bit where a fraction would do, because $ceil(dot)$ rounds the ideal $-log_2 p_i$ up to an integer. A symbol whose ideal cost is $3.737$ bits is charged $4$.

#background("escaping the integer penalty: arithmetic coding")[
  *Arithmetic coding* @rissanenlangdon1979 removes the per-symbol penalty by refusing to encode symbols one at a time. It represents an entire *sequence* as a single sub-interval of $[0, 1)$ whose width equals the sequence's probability $P("sequence")$, then transmits a point inside it. Pinning a point to enough precision costs $approx -log_2 P("sequence")$ bits *total*, so the rounding penalty is paid *once for the whole stream* rather than once per symbol: amortized over $n$ symbols the overhead is $O(1 slash n) arrow 0$. Arithmetic coding is thus the *discrete-time optimum* — it reaches the entropy rate in the long-sequence limit — and it is the construction the continuous-time spiking coder will *sidestep entirely*: spike *time* is a real number, so it spends a symbol's exact fractional bit-cost with no rounding and no block to amortize over (developed in the latency-code section).
]

Keep two facts in hand for the rest of the paper: (i) the entropy $H(p)$ is the floor on expected lossless code length @eq-shannon-bound, attained only by ideal real-valued lengths $ell_i = -log_2 p_i$; and (ii) discrete prefix codes pay up to one extra bit per symbol for the indivisibility of bits, escapable only by amortizing over long blocks (arithmetic coding's interval trick). The spiking substrate will later sidestep (ii) *per symbol*, precisely because time is not quantized into whole bits.

#intuition[
  Discrete prefix codes live on a *binary tree*: every codeword is a whole number of left/right turns, so one can only ever spend an integer number of bits. Arithmetic coding escapes the tree by working on the *continuum* $[0,1)$ but must batch a whole sequence to do so. The spiking coder will inherit the continuum — the real time axis — *without* batching: each symbol's surprisal is laid down as a real-valued latency, one symbol at a time. That is the structural advantage the coding-theory benchmark exists to make precise.
]

// ===== S4-exact-identity =====
= The exact identity: spike-time is surprisal <sec-identity>

Here is the move that turns "spikes are bits" from slogan into theorem. We
*calibrate* a readout neuron's drive so that its first-spike latency outputs
precisely a symbol's surprisal. We then show that — unlike a discrete prefix code
— the continuous time axis pays no integer-bit penalty, and we are scrupulous
about the two physical limits (finite drive, finite timing resolution) that bound
the idealization. The exact, proven currency throughout this section is *time*:
$lambda$ seconds per bit. The reading of that time as metabolic *energy* is a
separate, model-dependent corollary developed later, and we never silently swap
the time constant $lambda$ (s/bit, exact) for the energy constant $kappa$ (J/bit,
model-dependent).

Recall the first-passage latency of the leaky integrate-and-fire (LIF) neuron:
under constant drive from rest, $V(t) = R I (1 - e^(-t slash tau))$, and setting
$V(t^*) = theta$ gives the first-spike latency
$ t^*(I) = tau ln ( (R I) / (R I - theta) ). $
We now choose the model probability $q in (0, 1)$ that a readout assigns to its
own symbol — supplied by the recurrent loop as $q_j(c_t) = "softmax"_j(W c_t)$,
the predictor stated once in the circuit section — and design a drive that makes
@eq-latency read out exactly the surprisal $-log_2 q$.

== Latency calibration

#theorem("Latency calibration")[
  Fix a time-per-bit constant $lambda > 0$ and set the calibration exponent
  $alpha = lambda slash (tau ln 2)$. Drive a readout neuron whose model
  probability for its symbol is $q in (0, 1)$ with the current
  $ R I(q) = theta / (1 - q^alpha). $ <eq-calib>
  Then its first-spike latency is exactly
  $ t^*(q) = -lambda log_2 q = lambda dot I_q (x), $ <eq-tstar>
  i.e. proportional to the surprisal $I_q (x) = -log_2 q$, with proportionality
  constant $lambda$ in seconds per bit.
]

#proof[
  Substitute the calibrated drive @eq-calib into the latency law
  @eq-latency. Since $R I = theta slash (1 - q^alpha)$, the argument of the
  logarithm becomes, after clearing the common factor $theta$,
  $ (R I) / (R I - theta)
    = (theta slash (1 - q^alpha)) / (theta slash (1 - q^alpha) - theta)
    = (1 slash (1 - q^alpha)) / (1 slash (1 - q^alpha) - 1)
    = 1 / (1 - (1 - q^alpha))
    = q^(-alpha). $
  Hence, using $ln(q^(-alpha)) = -alpha ln q$ and the change of base
  $ln q = (ln 2)(log_2 q)$,
  $ t^* = tau ln(q^(-alpha)) = -alpha tau ln q
    = -(alpha tau)(ln 2)(log_2 q). $
  By the definition $alpha = lambda slash (tau ln 2)$ we have
  $(alpha tau)(ln 2) = lambda$, so $t^* = -lambda log_2 q$, which is
  @eq-tstar. The identity is *exact*: no approximation, no limit, no
  block-length amortization enters the algebra.
]

#intuition[
  The calibration absorbs every nuisance constant of the neuron ($tau$, $theta$,
  $R$) into a single knob $alpha$, leaving one clean time-per-bit constant
  $lambda$. A *certain* symbol ($q arrow 1$) demands $R I arrow infinity$ and
  fires instantly — zero bits, zero time. An *impossible* symbol ($q arrow 0$)
  demands $R I arrow theta^+$ (barely the rheobase, the minimum current that
  fires at all) and fires arbitrarily late — unbounded surprisal. Everything
  between interpolates smoothly and exactly: the time axis *is* the bit tape.
]

The calibration is the genuinely new piece here. Time-to-first-spike latency
coding — early spikes for strong drive — is itself classical
@thorpe1996 @thorpe2001; what @eq-calib adds is the *exact* dictionary
$"latency" = lambda dot "surprisal"$ realized by a *single* time constant
$lambda$, so that the LIF first-passage law and the Shannon codeword length
@shannon1948 coincide on the nose rather than monotonically. A numerical
integration of the LIF ODE under @eq-calib (companion script `validate.py`)
reproduces $t^*(q) = -log_2 q$ to the integrator's step size (max absolute error
$1.56 times 10^(-5)$ with Euler step $delta t = 10^(-5)$, i.e. pure
discretization error) across $q in {0.0375, dots, 0.98}$.

=== Streaming: total spike-time is the cross-entropy

Suppose the recurrent loop presents, at each step $t$, a conditional model
$q(dot mid(|) c_t)$ where $c_t$ is the loop's context state, and the realized
symbol is $x_t$. The readout for $x_t$ fires at latency
$-lambda log_2 q(x_t mid(|) c_t)$. Summing over the stream gives the total
transmission time; taking expectations gives the cross-entropy *rate*.

#theorem("Spike-time equals cross-entropy")[
  The total first-spike time to emit a length-$n$ stream is
  $ T_n = lambda sum_(t = 1)^n (-log_2 q(x_t mid(|) c_t))
    = lambda dot ("total surprisal of the stream under" q). $ <eq-Tn>
  If, moreover, $q(dot mid(|) c_t)$ is itself a first-order conditional model
  matched to the source's context, then taking expectations over a stationary,
  ergodic source $p$ with the loop holding the true contexts, the expected
  per-symbol time converges to the *cross-entropy rate*
  $ lim_(n arrow infinity) 1 / n EE[T_n] = lambda dot overline(H)(p, q)
    = lambda (overline(H)(p) + overline(D)_("KL")(p || q)), $ <eq-xrate>
  equal to the entropy-rate floor $lambda dot overline(H)(p)$ iff $q = p$, and
  exceeding it by $lambda dot overline(D)_("KL")(p || q) gt.eq 0$ otherwise.
]

#proof[
  Equation @eq-Tn is @eq-tstar summed over $t$, with no hypothesis beyond
  Theorem 1 applying at each step. For @eq-xrate, write
  $(1 slash n) EE[T_n] = lambda dot (1 slash n) sum_t EE[-log_2 q(x_t mid(|)
  c_t)]$. Under a stationary ergodic source with the loop tracking the true
  context, the per-step terms share the stationary law, and the time average
  converges to the stationary expectation $lambda dot overline(H)(p, q)$ by the
  ergodic theorem. The decomposition $overline(H)(p,q) = overline(H)(p) +
  overline(D)_("KL")(p || q)$ and $overline(D)_("KL") gt.eq 0$ with equality iff
  $q = p$ are the chain-rule and Gibbs' inequality applied to the
  conditional laws @shannon1948.
]

#remark[
  The matched-structure hypothesis is load-bearing for the word *rate*. If $q$ is
  *not* a first-order conditional model of the same order as the source, the
  limit in @eq-xrate is still a well-defined per-symbol average surprisal, but it
  need not equal the cross-entropy *rate* $overline(H)(p, q)$; it is then merely
  the average of $-log_2 q(x_t mid(|) c_t)$ along the trajectory. We state the
  hypothesis explicitly rather than fold it silently into the limit.
]

The companion script confirms @eq-Tn on a $2 times 10^6$-symbol stream (the
momentum-rover source, defined later): a perfect predictor spends $0.9760$
time-units per symbol, a memoryless predictor $1.7477$, and a wrong-parameter
predictor $1.1114$, against the theoretical cross-entropy rates $0.9782$,
$1.7500$, and $1.1133$ respectively. The empirical means track the
cross-entropy rates to sampling error. We flag one subtlety for honesty: the
perfect-predictor sample mean $0.9760$ sits slightly *below* the entropy-rate
floor $0.9782$. This is not a violation of the source-coding theorem — it is an
ordinary finite-sample fluctuation of a sample mean about its expectation; over
a longer stream it relaxes up to $0.9782$ from below.

== A price the binary tree pays and the time axis does not

Discrete prefix codes obey Shannon's bound $H(p) lt.eq L^* < H(p) + 1$
@shannon1948 @kraft1949: the slack comes from rounding fractional ideal lengths
$-log_2 p_i$ up to whole bits, and is removable only by amortizing over long
blocks (arithmetic coding @rissanenlangdon1979). Spike *time* carries no such
constraint, and we prove this in one line — together with the finiteness
hypothesis under which the statement is honest.

#theorem("Continuous time has no integer penalty")[
  Assume the idealized regime of unbounded drive and arbitrarily fine timing
  resolution (so that any positive real latency is both realizable and
  resolvable). Then the calibrated latency $t^*(q) = -lambda log_2 q$ takes a
  *real* value, not constrained to any integer multiple of a fixed quantum.
  Consequently the calibrated spiking coder achieves the per-symbol cost
  $lambda(-log_2 q(x_t mid(|) c_t))$ *exactly*, for *every single symbol* — not
  merely in a block limit. With $q = p$ its expected per-symbol cost is exactly
  $lambda dot overline(H)(p)$, the floor, with *zero* slack.
]

#proof[
  By Theorem 1 the latency is $t^*(q) = -lambda log_2 q in (0, infinity)$, a
  continuous function of $q in (0,1)$ with range all of $(0, infinity)$. Under
  the idealization hypothesis this value is realizable (the required drive
  @eq-calib is finite for $q > 0$ and unbounded only in the limit $q arrow
  1$) and resolvable (the readout clock is infinitely fine). No step of the
  construction rounds $t^*$ to a lattice: the integer constraint of prefix codes
  arises only because a codeword is a whole number of bit-symbols, and a latency
  is not a string of bit-symbols. Hence the per-symbol cost equals
  $-lambda log_2 q$ with no ceiling operation, and summing and taking the $q = p$
  expectation gives $lambda dot overline(H)(p)$ exactly by Theorem 2.
]

#intuition[
  Discrete prefix codes live on a *binary tree*: every codeword is a whole
  number of left/right turns, so a symbol whose ideal cost is $3.737$ bits gets
  charged $4$. The spiking coder lives on the *real time axis*: it spends
  $3.737$ bits' worth of *duration*, no rounding. The Kraft integer ceilings
  @kraft1949 relax, in continuous time, into ordinary real inequalities — met
  with equality, per symbol, the way arithmetic coding @rissanenlangdon1979 only
  achieves in the long-block amortized limit.
]

#remark[
  *Honest contrast with arithmetic coding.* Arithmetic coding @rissanenlangdon1979
  already escapes the integer penalty — by encoding a whole *sequence* as one
  sub-interval and amortizing the rounding over its length. The spiking coder's
  difference is *where* the escape happens, not *that* it happens: arithmetic
  coding amortizes a discrete cost over a block to reach the floor in the limit,
  whereas the latency code spends a *physically real-valued* duration per symbol
  and meets the floor symbol-by-symbol with no block. This is a substrate
  distinction (analog time vs. amortized bits), not a new compression *ratio*:
  both reach $overline(H)(p)$; neither beats it. We do not claim the spiking
  coder compresses *better* than arithmetic coding — only that it dispenses with
  the integer penalty without buffering a block.
]

== What physics costs: bounding the idealization honestly

The integer-penalty advantage is bounded *below* by physics, not combinatorics,
and we state the bound rather than hide it. There are two distinct limits.

*(a) Finite drive caps the representable probability.* The calibrated drive
@eq-calib blows up well *inside* the working range, not merely at the
$q arrow 1$ endpoint. With $tau = lambda = 1$ (so $alpha = 1 slash ln 2 approx
1.4427$), the drive-to-threshold ratio $R I slash theta = 1 slash (1 - q^alpha)$
grows steeply:

#figure(
  table(
    columns: (auto, auto, auto),
    align: (center, center, left),
    stroke: 0.5pt + luma(180),
    table.header[$q$][$R I slash theta = (1 - q^alpha)^(-1)$][surprisal
      $-log_2 q$],
    [$0.10$], [$1.04 times$], [$3.32$ bits],
    [$0.50$], [$1.58 times$], [$1.00$ bits],
    [$0.85$], [$4.79 times$], [$0.234$ bits],
    [$0.90$], [$7.09 times$], [$0.152$ bits],
    [$0.95$], [$14.0 times$], [$0.074$ bits],
    [$0.99$], [$69.5 times$], [$0.0145$ bits],
  ),
  caption: [Calibration drive vs. model probability at $alpha approx 1.4427$.
    Encoding a $q = 0.9$ symbol already demands $approx 7.1 times$ rheobase;
    $q = 0.99$ demands $approx 70 times$. A real neuron's drive saturates a
    few-fold above rheobase, which pins a maximum representable probability.],
) <fig-drive>

A biophysical neuron cannot supply unbounded current: post-spike refractoriness
and synaptic/conductance saturation cap $R I$ at a few-fold rheobase. From
@fig-drive, a ceiling of, say, $5$–$15 times$ rheobase pins a *maximum
representable probability* $q_max approx 0.85$–$0.95$. Two consequences follow.
First, near-certain symbols ($q > q_max$) cannot be driven to fire arbitrarily
early; they incur a small *floor cost* — a minimum latency $t_min$ that
over-charges the most confident, lowest-surprisal symbols (precisely the symbols
the code would most like to make nearly free). Second, this is the one place the
exact identity @eq-tstar must be read as $t^*(min(q, q_max))$: the code is exact
on the open interval $(0, q_max]$ and saturates above it.

*(b) The rare-symbol end is noise-dominated.* As $q arrow 0$ the drive approaches
rheobase from above, $R I arrow theta^+$, and the latency $t^*$ diverges. Near
rheobase the membrane sits for a long time barely below threshold, where
membrane noise dominates the deterministic drift: the first-passage *time*
becomes a broad, heavy-tailed random variable rather than a sharp value. So the
$q arrow 0$ (high-surprisal) regime is exactly where the timing code is *least*
reliable — rare symbols are encoded by long, noise-jittered latencies. The
exactly representable band is therefore an *interior* window of $q$, bounded
above by drive saturation and below by noise.

*(c) Finite timing resolution replaces — does not abolish — the integer penalty.*
A real readout reads $t^*$ with a clock of period $delta t$ and against a noise
floor. We can make the recoverable precision quantitative rather than assert it.
Suppose admissible latencies span $[t_min, t_max]$ and are read to resolution
$delta t$; the number of *distinguishable* latency bins is $(t_max - t_min) slash
delta t$, so the latency channel carries at most
$ log_2 ((t_max - t_min) / (delta t)) "bits" approx log_2 (t^* / (delta t)) $
of resolvable information per spike, the approximation holding when $t_min ≪ t^*
approx t_max$. (This is a heuristic counting bound, not a channel-capacity
theorem: it counts resolvable bins and ignores the latency-dependent shape of
the noise; a tighter statement would integrate the first-passage-time density
against the clock, which we leave as future work.) The honest summary:

#honest[
  The spiking coder trades the combinatorial $+1$-bit rounding penalty of prefix
  codes for an *analog* timing-resolution penalty of order $log_2(t^* slash delta
  t)$ bits per spike. It does not abolish the penalty — it changes its character.
  The trade is favourable exactly when the clock is fast relative to the firing
  latencies ($delta t ≪ t^*$), which is the regime neuromorphic hardware
  targets, and unfavourable for the rarest symbols, whose long noise-jittered
  latencies (limit b) and the confident symbols' saturation floor (limit a)
  together carve the *exactly representable* probabilities down to an interior
  band $q in (q_min, q_max]$ with $q_max approx 0.85$–$0.95$. Within that band
  the per-symbol identity $t^* = -lambda log_2 q$ is exact; outside it, the code
  degrades gracefully but is no longer the clean Shannon codeword length.
]

#remark[
  *Where the novelty sits, stated precisely.* Three ingredients of this section
  are classical and we attribute them: time-to-first-spike latency coding
  @thorpe1996 @thorpe2001; the source-coding floor and the integer penalty
  @shannon1948 @kraft1949; and arithmetic coding's amortized escape from that
  penalty @rissanenlangdon1979. What is *new here* is narrow and exact: a *single*
  membrane time constant $lambda$ that makes LIF first-passage latency
  *equal* — not merely monotone in — the Shannon surprisal $-log_2 q$ via the
  closed-form calibration @eq-calib, so that a stream's total spike-*time* is
  its cross-entropy (Theorem 2) symbol-by-symbol in continuous time. The energy
  reading of that time, and the verification split that makes correctness
  independent of code optimality, are developed in later sections; the
  conditional cost-spike$=$gradient identity is the paper's other genuine
  novelty and is foregrounded where the learning rule is built.
]

// ===== S5-source =====
= Sources with memory, and one that forces the cycle to work

So far the loop's prediction $q(dot mid(|) c_t)$ could depend on context, but we
have not said why context *helps*. It helps only if the source has *temporal
structure* — if the next symbol is statistically tied to the past. A
*memoryless* source (successive symbols independent) gives a cycle nothing to
do: a fixed feedforward code is already optimal. So the case study must begin by
choosing a source whose past genuinely predicts its future, and by
*quantifying* the help up front — *before* any circuit is built — so that the
recurrent advantage is a number we can later check the circuit against.

#background("Markov chains, stationary law, entropy rate, and mutual information")[
  A *(first-order) Markov chain* over a finite alphabet $cal(X)$ is specified by
  *transition probabilities* $P_(i j) = Pr(x_(t+1) = j mid(|) x_t = i)$: the
  next symbol depends on the current one and on no earlier history. The matrix
  $P$ has nonnegative entries and each row sums to one ($sum_j P_(i j) = 1$), so
  it maps probability distributions to probability distributions. A distribution
  $pi$ over states is *stationary* if one step leaves it unchanged, $pi P = pi$
  — i.e. $pi$ is a left eigenvector of $P$ with eigenvalue $1$. For an
  irreducible aperiodic chain $pi$ is unique and equals the long-run fraction of
  time spent in each state, independent of where the chain started.

  Two distinct entropies now appear, and the gap between them is the whole point
  of having memory. The *marginal* entropy
  $ H(pi) = -sum_i pi_i log_2 pi_i $
  treats the symbols as if they were independent draws with frequencies $pi$; it
  is what a *memoryless* code pays, blind to order. The *entropy rate* instead
  conditions on the immediate predecessor,
  $ overline(H)(P) = sum_i pi_i H(P_(i dot)), quad
    H(P_(i dot)) = -sum_j P_(i j) log_2 P_(i j), $ <eq-entropy-rate>
  averaging each row's residual uncertainty against the long-run frequency
  $pi_i$ of being in that row. It is the true per-symbol floor for a predictor
  that is *allowed to use the last symbol* — the irreducible uncertainty that
  remains even with perfect knowledge of the chain @shannon1948. Because
  conditioning never increases entropy, $overline(H)(P) <= H(pi)$, and the
  difference is exactly the *mutual information* between consecutive symbols,
  $ I(x_t ; x_(t-1)) = H(pi) - overline(H)(P) >= 0. $ <eq-mi>
  This is, precisely, the number of bits per symbol that knowing the past lets a
  predictor recover — the bits a memory can pay back, and not one more.
]

#intuition[
  The marginal entropy $H(pi)$ is the bill a coder pays if it knows *how often*
  each symbol occurs but nothing about *what tends to follow what*. The entropy
  rate $overline(H)(P)$ is the bill once it also knows the local transition
  rule. Their difference @eq-mi is the recurrent circuit's entire reason to
  exist on this stream: it is the compression that a cycle — and *only* a cycle,
  since the dependence is across time — can deliver. If that difference is zero,
  no amount of recurrence helps; if it is large, a feedforward coder is leaving
  exactly that many bits on the table every symbol.
]

We now pick a one-parameter source engineered so the arithmetic stays in closed
form and the *only* thing the dial changes is the memory — the marginal
statistics are held fixed, so any improvement a predictor shows is unambiguously
attributable to using the past.

#definition[
  *Momentum rover.* A rover moves on the alphabet $cal(X) = {U, D, L, R}$ with
  long-run move frequencies $pi = (1 slash 2, 1 slash 4, 1 slash 8, 1 slash 8)$,
  but with *inertia*: with *stickiness* $s in [0, 1)$ it repeats its last move,
  and otherwise (probability $1 - s$) it draws a fresh move from $pi$. Its
  transition matrix is therefore
  $ P_(i j) = s dot bb(1)[i = j] + (1 - s) pi_j, $ <eq-rover>
  a convex blend of the identity (pure momentum) and the rank-one matrix
  $bb(1) pi^top$ (pure memorylessness), with $s$ dialling between them.
]

#proposition[
  For every $s in [0, 1)$ the momentum rover @eq-rover has stationary
  distribution exactly $pi = (1 slash 2, 1 slash 4, 1 slash 8, 1 slash 8)$.
]

#proof[
  Compute the $j$-th component of $pi P$ directly from @eq-rover, using $sum_i
  pi_i = 1$:
  $ (pi P)_j = sum_i pi_i (s dot bb(1)[i = j] + (1 - s) pi_j)
    = s pi_j + (1 - s) pi_j sum_i pi_i
    = s pi_j + (1 - s) pi_j = pi_j. $
  The first term keeps only $i = j$; the second factors $pi_j$ out of the sum,
  which collapses to $1$. Hence $pi P = pi$ for every $s$. (The chain is
  irreducible and aperiodic for $s in [0, 1)$ — every state still has positive
  probability $(1 - s) pi_j > 0$ of being entered from any other — so this
  stationary law is the unique one and is genuinely approached from any start.)
]

The construction does exactly what we wanted. The *marginal* statistics — and
hence a memoryless code's cost — are pinned at
$ H(pi) = -(1/2 log_2 1/2 + 1/4 log_2 1/4 + 1/8 log_2 1/8 + 1/8 log_2 1/8)
  = bold(1.7500) " bits/symbol" $ <eq-Hpi>
for *all* $s$ (the move U costs $1$ bit, D costs $2$, and L, R cost $3$ each,
weighted by $pi$). Only the *conditional* structure $P_(i dot)$ moves with $s$.
We fix $s = 0.7$ as the worked operating point (reproduced in the companion
`validate.py`). The four context rows of @eq-rover then carry the
self-prediction probabilities $P_(i i) = s + (1 - s) pi_i$ and the conditional
entropies @eq-entropy-rate tabulated below.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    stroke: 0.5pt + luma(180),
    table.header[context $i$][$P_(i i)$ (repeat its own move)][$H(P_(i dot))$],
    [after U], [$0.850$], [$0.835$ bits],
    [after D], [$0.775$], [$1.051$ bits],
    [after L], [$0.7375$], [$1.192$ bits],
    [after R], [$0.7375$], [$1.192$ bits],
  ),
  caption: [Per-context conditional entropies of the momentum rover at $s =
    0.7$. The chain is most self-predictive right after the common move U (low
    residual uncertainty) and least so after the rare moves L, R.],
) <fig-rover-rows>

Averaging the conditional entropies against the stationary frequencies $pi$
gives the entropy rate @eq-entropy-rate:
$ overline(H)(P) = 0.5(0.835) + 0.25(1.051) + 0.125(1.192) + 0.125(1.192)
  = bold(0.9782) " bits/symbol". $ <eq-rover-rate>
Set against the marginal floor $H(pi) = 1.7500$ bits, the memory is worth
exactly the mutual information @eq-mi between a symbol and its predecessor,
$ I(x_t ; x_(t-1)) = H(pi) - overline(H)(P) = 1.7500 - 0.9782
  = bold(0.7718) " bits/symbol". $ <eq-rover-mi>

This single number is the case study's reason to exist, and — crucially — it is
fixed *before* any circuit is designed: it is a property of the source alone.
Relative to the *marginal* baseline of $H(pi) = 1.7500$ bits/symbol, recovering
$I(x_t ; x_(t-1)) = 0.7718$ bits is a $0.7718 slash 1.7500 approx 44%$ reduction
in per-symbol cost — bits that a memoryless spike code discards and that a
recurrent predictor, *if it learns the conditional law*, can reclaim. We
emphasize the baseline explicitly because two different denominators are in play
throughout this paper: the $44%$ figure is computed against the *marginal*
entropy $H(pi) = 1.7500$ (the memoryless cost), whereas the entropy *rate*
$overline(H)(P) = 0.9782$ is the *floor* a memory-using predictor descends
toward — and never below.

#honest[
  The reduction @eq-rover-mi is the *information-theoretic* ceiling on what
  recurrence can buy here; it is what a predictor with the *exact* conditional
  law would save. It is not what any particular learned circuit will save. A
  finitely trained predictor sits slightly *above* the floor: the converged
  learned per-symbol energy in the companion run is $0.9787$ bits — above
  $overline(H)(P) = 0.9782$ by a small finite-learning residual, not at the
  floor — so $0.9782$, never $0.979$ rounded, is the quantity to call "the
  floor." Conversely a finite *sample* mean can dip slightly below the
  expectation by ordinary sampling fluctuation (a length-$n$ stream from the
  perfect predictor measures $approx 0.9760$ bits/symbol against the expectation
  $0.9782$); this is finite-sample noise, not a violation of the entropy-rate
  bound.
]

#remark[
  *The advantage is a dial with a closed form.* Sweeping the stickiness traces
  the entire spectrum. As $s arrow 0$ the rover degenerates to a memoryless
  source ($P_(i j) arrow pi_j$, every row identical), the conditional structure
  vanishes, and $I(x_t ; x_(t-1)) arrow 0$: memory is worthless and a
  feedforward code is already optimal. As $s arrow 1$ the rover approaches
  deterministic repetition ($P arrow II$), each conditional entropy
  $H(P_(i dot)) arrow 0$, and the mutual information rises toward the full
  marginal $H(pi) = 1.7500$ bits: the entire stream becomes predictable from its
  predecessor. The recurrent advantage is thus a tunable quantity with the
  explicit form $I(x_t; x_(t-1))(s) = H(pi) - sum_i pi_i H(P_(i dot)(s))$, and
  $s = 0.7$ is merely a representative interior point.
]

#remark[
  *A note on matched structure (used in @sec-identity).* For the
  per-symbol time of a calibrated predictor to converge to the cross-entropy
  *rate* $overline(H)(p, q)$ — and not merely to a per-symbol average of
  surprisals — the circuit's model $q(dot mid(|) c_t)$ must itself be a
  *first-order conditional* model, i.e. its context $c_t$ must summarize the
  immediately preceding symbol (and no more is needed here, since the rover is
  first-order). When the model's memory order matches the source's, the
  expected per-symbol cost is the cross-entropy rate, whose minimum over models
  is exactly the entropy rate @eq-rover-rate, attained iff $q(dot mid(|) c_t) =
  P_(c_t, dot)$. A model with *less* memory than the source cannot reach the
  rate floor (it is stuck at or above the marginal $H(pi)$); a model with *more*
  memory than a first-order source gains nothing — the finite, noise-limited
  context capacity of the attractor (introduced next) is therefore matched, not
  wasted, on this source. The Markov chain is also the native object of the
  probabilistic model checkers we target later, which is why we keep the source
  exactly in this form.
]

// ===== S6-stream =====
= A stream spike-time is its cross-entropy

The calibration theorem made one symbol's latency exact: a readout driven by
$R I(q) = theta slash (1 - q^alpha)$ fires its first spike at time $t^*(q) =
-lambda log_2 q$, the symbol's surprisal in seconds (with $lambda$ the
*exact, proven* time-per-bit constant of @eq-calib). We now run the
coder on a *stream* and sum these times. The result is the central identity of
the source-coding half of this paper: the circuit's total spike-time is, exactly,
the cross-entropy of its predictions against the source, and its per-symbol
average is the cross-entropy *rate*. Shannon's entropy rate is the floor; the
excess above it is the average Kullback--Leibler divergence between the circuit's
model and the world @shannon1948.

We emphasise once, and never silently swap it: the proven, summed observable here
is a *time* (seconds), built from the exact constant $lambda$ (s/bit). It is not
energy. The metabolic reading --- joules per bit --- is a separate,
model-dependent corollary developed later under an explicit biophysical model;
it is never an identity, and $lambda$ (s/bit, exact) is never substituted by the
substrate constant $kappa$ (J/bit, model-dependent).

== The setup: a conditional predictor reading a context

Let the recurrent loop present, at each stream step $t = 1, dots, n$, a
*conditional* model $q(dot mid(|) c_t)$ over the alphabet $cal(X)$, where $c_t$
is the loop's context state and $q_j(c_t) = "softmax"_j (W c_t)$ is the
predicted probability of symbol $j$ (the predictor map fixed once in the circuit
section; it is the same $q$ the calibration drive encodes). The source emits the
realized symbol $x_t$. The calibrated readout for that symbol fires its first
spike at latency $-lambda log_2 q(x_t mid(|) c_t)$, and the temporal
winner-take-all decoder reads it off and resets the integration clock for step
$t + 1$. Each symbol therefore contributes exactly its own surprisal, in
seconds, to the running total.

#intuition[
  Each step is one calibrated race-to-threshold, already proven exact for a
  single symbol. Streaming changes nothing about that exactness --- it only
  *accumulates* it. The loop's job between steps is to update the context $c_t$,
  hence the prediction $q(dot mid(|) c_t)$; the readout's job is to convert that
  prediction into a latency. The total time the circuit spends spiking is just
  the sum of per-symbol surprisals, with no cross-symbol bookkeeping and no
  rounding (continuous time carries no integer-bit penalty).
]

== Total spike-time equals total surprisal

The first statement is an exact, *deterministic* identity for any realized
stream --- no probabilistic hypothesis is needed, because it is a sum of
per-symbol exact times.

#theorem("Total spike-time is total surprisal")[
  For any realized stream $x_1, dots, x_n$ and any sequence of contexts
  $c_1, dots, c_n$ the loop supplies, the total first-spike time the calibrated
  coder spends to emit the stream is exactly
  $ T_n = sum_(t = 1)^n t^*(q(x_t mid(|) c_t)) = lambda sum_(t = 1)^n
  (-log_2 q(x_t mid(|) c_t)) = lambda dot "(total surprisal)". $ <eq-total-time>
]

#proof[
  By the calibration theorem (@eq-calib), the readout for the realized
  symbol $x_t$ in context $c_t$ has first-spike latency exactly
  $-lambda log_2 q(x_t mid(|) c_t)$, a real number with no rounding. The decoder
  resets the clock at each spike, so the per-symbol latencies are disjoint
  durations that add. Summing over $t = 1, dots, n$ gives @eq-total-time. The
  identity is termwise exact and holds for every individual stream, not merely on
  average. #h(1fr) $square$
]

This is the slogan "spikes are bits" made literal over a whole message: read off
the time axis and you have read off $lambda$ times the surprisal the circuit's
model assigned to exactly what happened. A good model spends little time; a bad
model spends a lot; neither is ever *wrong*, only slow --- correctness rests on
the decode rule, not on the predictor.

== Expected per-symbol time is the cross-entropy rate

To turn the per-stream identity into a *rate*, we average over the source. Two
things must line up: the source must have a long-run statistical regularity to
average against, and the circuit's model must be the *same kind of object* as
the source's conditional law, or the per-symbol average will not be a
cross-entropy *rate* at all. We state the first as a one-line hypothesis and the
second as an explicit matched-structure condition (fix A2/IT-04).

#block(inset: (left: 6pt))[
  *Hypothesis (stationary, ergodic source).* The source is a stationary, ergodic
  finite-state Markov chain with transition matrix $P$ and stationary law $pi$
  ($pi P = pi$); every context $i$ with $pi_i > 0$ is visited infinitely often.
  Stationarity makes the per-symbol expectation well defined; ergodicity makes
  the long-run *time average* equal that expectation almost surely
  @shannon1948.
]

We keep this deliberately light. The momentum-rover source of the case study is
manifestly stationary and ergodic ($P_(i j) > 0$ for all $i, j$ at any
stickiness $s in [0, 1)$), so the hypothesis is discharged by inspection rather
than by a heavy theorem; it earns its place only to name what the averaging
relies on.

#background("matched structure: why \"rate\" needs a conditional model")[
  The *cross-entropy rate* of a model $q$ against a source $P$ is the long-run
  expected surprisal per symbol *when both condition on the same past*. For that
  per-symbol average to deserve the word "rate," the circuit's predictor
  $q(dot mid(|) c_t)$ must itself be a *first-order conditional* model --- its
  context $c_t$ must carry (at least) the previous symbol, matching the
  first-order memory of $P$. If instead $q$ were a fixed *marginal* (a memoryless
  code, ignoring $c_t$), the same average would still be a perfectly good
  per-symbol mean cost, but it would be the marginal cross-entropy
  $H(pi, q)$ --- *not* the conditional cross-entropy rate $overline(H)(p, q)$, and
  it could not reach the conditional floor. The identity below is therefore
  stated for a context that resolves the source's predecessor; this is the
  hypothesis that makes "rate" the right word.
]

#theorem("Expected per-symbol spike-time is the cross-entropy rate")[
  Under the stationary--ergodic hypothesis, with the loop holding the true
  predecessor context (so $c_t$ resolves $x_(t-1)$ and $q(dot mid(|) c_t) =
  q_(i dot)$ when $x_(t-1) = i$), the expected per-symbol spike-time converges,
  almost surely and in expectation, to $lambda$ times the *cross-entropy rate*:
  $ lim_(n arrow infinity) 1/n T_n = lim_(n arrow infinity) 1/n EE[T_n]
  = lambda dot overline(H)(p, q)
  = lambda sum_i pi_i sum_j P_(i j) (-log_2 q_(i j)). $ <eq-rate>
  This equals the *entropy-rate floor* $lambda dot overline(H)(p)$ if and only if
  $q_(i dot) = P_(i dot)$ for every context $i$ with $pi_i > 0$, and otherwise
  exceeds it by exactly $lambda$ times the average KL divergence,
  $ 1/n T_n arrow lambda(overline(H)(p) + overline(D)_("KL")(p || q)), quad
  overline(D)_("KL")(p || q) = sum_i pi_i D_("KL")(P_(i dot) || q_(i dot))
  gt.eq 0. $ <eq-floor-excess>
]

#proof[
  Take the expectation of the per-symbol latency $-lambda log_2 q(x_t mid(|)
  c_t)$ in @eq-total-time. Conditioning on the context $x_(t-1) = i$ (which
  occurs with stationary frequency $pi_i$) and on the realized symbol $x_t = j$
  (which, given $x_(t-1) = i$, occurs with probability $P_(i j)$), the per-symbol
  expectation is $lambda sum_i pi_i sum_j P_(i j)(-log_2 q_(i j)) = lambda dot
  overline(H)(p, q)$. Ergodicity replaces the ensemble expectation by the almost-
  sure time average $n^(-1) T_n$ (the Shannon--McMillan--Breiman / ergodic
  theorem for a stationary ergodic chain @shannon1948). The split into floor
  plus excess is the conditional cross-entropy decomposition $overline(H)(p, q) =
  overline(H)(p) + overline(D)_("KL")(p || q)$, applied row-by-row and averaged
  against $pi$; non-negativity of each $D_("KL")(P_(i dot) || q_(i dot))$ and its
  vanishing iff $q_(i dot) = P_(i dot)$ give the equality condition. #h(1fr)
  $square$
]

#intuition[
  Read @eq-floor-excess as the whole thesis in spike-times. The floor $lambda
  dot overline(H)(p)$ is what the *world* charges --- irreducible, set by the
  source's own conditional uncertainty, paid even by a perfect predictor. The
  excess $lambda dot overline(D)_("KL")(p || q)$ is what *being wrong* charges ---
  avoidable, vanishing exactly when the circuit's conditional model equals the
  source's. A learner can lower its spike-time bill only by shrinking the average
  KL, i.e. by making each context row $q_(i dot)$ resemble the true $P_(i dot)$.
  Spending less time is the same act as understanding the source better.
]

== Numerical confirmation

The companion validator (`validate.py`) runs the calibrated coder on a
$2 times 10^6$-symbol momentum-rover stream (the source defined in the
Markov-source section, stickiness $s = 0.7$) and measures the time-average
$n^(-1) T_n$ for three predictors:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, left),
    stroke: 0.5pt + luma(180),
    table.header[predictor $q$][measured $n^(-1) T_n$][theory ($lambda = 1$)][what it is],
    [perfect, $q = P$], [$0.9760$], [$0.9782$], [the entropy-rate floor],
    [memoryless, $q = pi$], [$1.7477$], [$1.7500$], [marginal cross-entropy],
    [wrong stickiness $s' = 0.4$], [$1.1114$], [$1.1133$], [floor $+$ avg KL],
  ),
  caption: [Time-average per-symbol spike-time (units of $lambda$) versus the
    cross-entropy rate @eq-rate, on a $2 times 10^6$-symbol stream. The measured
    column is a *sample mean*; the theory column is the expectation. The gaps
    above the floor $0.9782$ are exactly the average KL divergences.],
) <fig-stream-numerics>

Two honesty notes on the numbers (fix M). First, the conditional floor is
$overline(H)(p) = 0.9782$ bits/symbol *everywhere* in this paper; do not round it
to $0.979$, and do not call the learned circuit's later converged cost (which
settles slightly *above* the floor, at $0.9787$, a finite-learning residual) "the
floor." Second, the perfect-predictor sample mean $0.9760$ sits a hair *below*
the expectation $0.9782$: this is ordinary finite-sample fluctuation of a mean
around its expectation over $2 times 10^6$ draws, not a violation of the floor.
The floor binds the *expectation* @eq-rate; a single sample mean may land on
either side of it. The memoryless and wrong-stickiness means likewise track their
expectations to sampling error, and the excess of each above $0.9782$ reproduces
its average KL: a memoryless code wastes the full mutual information
$I(x_t; x_(t-1)) = 1.7500 - 0.9782 = 0.7718$ bits/symbol, while the wrong-momentum
predictor wastes only $0.1351$.

#honest[
  The averaging step imports the stationary--ergodic hypothesis above and, for
  the word "rate," the matched-structure condition that $q$ is itself a
  first-order conditional model. Both hold for the momentum rover by inspection,
  but they are genuine premises: a non-stationary or non-ergodic source, or a
  predictor whose context fails to resolve the source's memory order, would leave
  @eq-total-time intact (it is hypothesis-free) while making @eq-rate either ill-
  defined or merely a per-symbol average against the *wrong* floor. We state the
  hypotheses rather than bury them, and we keep the result *time*-valued: every
  number in @fig-stream-numerics is a spike-time in units of $lambda$, not a
  metabolic energy.
]

// ===== S7-recurrent =====
= Holding context and choosing winners: the recurrent ingredients <sec-recurrent>

The encoder of the previous section needs two things from a recurrent network: a
way to *hold* the running context $c_t$, and a way to *decide* which symbol was
sent. Both are standard circuit motifs. We introduce each from first principles
before assembling them, and we are careful to separate what each motif actually
computes from the convenient idealization we will lean on later.

== Attractors: holding a graded context

A recurrent network's state evolves under its own dynamics: a neuron's output
loops back to influence its own future input. A state the dynamics settle into
and remain at is a *fixed point* (an *attractor*); the set of starting states
that flow into it is its *basin of attraction*.

#background("attractors and graded persistent activity")[
  Most networks have a handful of *isolated* attractors (discrete memories). A
  carefully tuned recurrent network can instead possess a *continuous line — or
  ring — of fixed points*, a *line attractor* (resp. *ring attractor*)
  @seung1996 @benyishai1995. Along such a manifold the recurrent excitation
  exactly balances the leak, so the network can rest *anywhere* on the line and
  hold a graded analog value with no input: *graded persistent activity*, a
  biologically observed substrate for analog short-term memory @seung1996. A ring
  attractor closes the line into a loop and is the canonical model of a
  continuously tuned cortical variable such as head direction or orientation
  @benyishai1995. The position of the activity bump along the manifold *is* the
  stored value.
]

We use a line/ring attractor to hold the context $c_t$: the bump position encodes
a running summary of recent symbols, and the lateral weights $W$ read that
position out as logits (developed in the circuit section). It is tempting to say
the bump lets the prediction depend on *arbitrarily deep* history rather than a
fixed window. That overstates the case, and we correct it now.

#honest[
  *A line attractor stores a finite, noise-limited number of context states (fix
  G / BIO-06).* The manifold is a *single* graded analog value. Any physical
  realization has noise, so two bump positions are distinguishable only if they
  differ by more than the positional jitter. If the usable range of the bump
  spans a signal-to-noise ratio $"SNR"$, the number of reliably distinguishable
  states is $approx "SNR"$, i.e. about
  $ C approx log_2("SNR") quad "bits of context capacity." $ <eq:attractor-capacity>
  The attractor therefore holds *a finite, noise-limited number of
  distinguishable context states, capturing memory order up to that capacity* —
  not arbitrarily deep history. Sources whose statistics depend on more past
  symbols than @eq:attractor-capacity can encode cannot be predicted to their
  conditional floor by this substrate; the reachable floor is an
  attractor-capacity question, consistent with (not contradicting) the capacity
  limitations recorded in the Critical Evaluation.
]

== Winner-take-all: choosing one symbol

To decode we must commit to one symbol. The motif for committing is *competition*.

#background("winner-take-all and contralateral inhibition")[
  A *winner-take-all* (WTA) network is a set of units that each excite themselves
  and *inhibit each other* — *lateral* inhibition, or *contralateral* inhibition
  in the two-unit case where each unit suppresses its opposite. Started from
  graded inputs, the mutual inhibition amplifies the largest drive and suppresses
  the rest; the network settles to a state in which essentially one unit — *the
  winner* — remains active. WTA implements selection and decision.
]

A caution on what WTA guarantees. "At most one unit is ever active" is *false* as
a global, instantaneous invariant: during the integration transient several units
are simultaneously sub-threshold-active while the competition resolves. What the
motif certifies is a *settled* property — *eventually* exactly one settled winner
per decode window — with a quantified transient that the first-spike readout
tolerates. We state and verify this in the settled form ($sans("F") thin
sans("G")$) in the verification section (fix H), and never rely on instantaneous
mutual exclusion.

== Gain control: softmax versus divisive normalization

The decoder must also turn graded drives into something that behaves like a
probability distribution — a nonnegative vector summing to one. There are two
distinct objects here, and earlier drafts conflated them. We separate them
cleanly (fix A1), because the paper's partition-of-unity invariant $sum_j q_j =
1$ rests on getting this exactly right.

#definition[
  The *softmax* map sends logits $a_1, dots, a_N$ to
  $ q_j = (e^(a_j)) / (sum_(k=1)^N e^(a_k)), quad j = 1, dots, N. $ <eq:softmax>
  By construction $q_j > 0$ and $sum_(j=1)^N q_j = 1$ *exactly*: softmax is an
  *exact normalizer onto the probability simplex*. This is the map the circuit
  uses to produce its internal model, $q_j = "softmax"_j (W c_t)$ with logits
  $a_j = sum_i W_(i j) c_i$ (stated once, in the circuit section).
]

The biophysically canonical gain-control computation is *not* softmax but
*divisive normalization* @carandiniheeger2012: each unit's response is divided by
a term that grows with the *total* pool activity,
$ r_i = (a_i) / (sigma + sum_(j=1)^N a_j), $ <eq:divnorm>
with semi-saturation constant $sigma >= 0$ and nonnegative drives $a_i >= 0$.
This is the canonical cortical gain control — a single circuit explaining
contrast saturation, surround suppression and cross-orientation effects
@carandiniheeger2012. But it is crucial to see that @eq:divnorm does *not* by
itself put the responses on the simplex.

#proposition[
  Divisive normalization @eq:divnorm sums to
  $ sum_(i=1)^N r_i = (sum_i a_i) / (sigma + sum_i a_i) < 1 quad "whenever" sigma
  > 0, $ <eq:divnorm-sum>
  and $sum_i r_i -> 1$ only in the limit $sigma -> 0$. Hence divisive
  normalization is *not* a partition of unity; it is the *biophysical
  approximation* to an exact simplex-normalizer, valid as $sigma -> 0$.
]
#proof[
  Summing @eq:divnorm over $i$ gives @eq:divnorm-sum directly, since the
  denominator $sigma + sum_j a_j$ is common to every term. For $sigma > 0$ and
  total drive $sum_j a_j < infinity$ the ratio is strictly below $1$; as $sigma
  -> 0$ it tends to $1$.
]

#intuition[
  Read $sigma$ as a fixed "leak" added to the normalizing pool. Softmax
  normalizes by the *exact* pool $sum_k e^(a_k)$ with no additive slack, so it
  lands exactly on the simplex; divisive normalization normalizes by $sigma$ plus
  the (linearized) pool, so it always leaves a fraction $sigma slash (sigma +
  sum_j a_j)$ of the budget unspent. The two agree in the high-drive /
  small-$sigma$ regime. Softmax is additionally the exponential-family
  normalizer, which is exactly why its logits are *log-odds* and the lateral
  weights $W_(i j)$ read as log-odds (used in the calibration drive).
]

We therefore adopt the following frozen convention, which every downstream
section respects: *the exact simplex invariant $sum_j q_j = 1$ is carried by the
softmax @eq:softmax; divisive normalization @eq:divnorm is its $sigma -> 0$
biophysical realization, a gain control, not an exact normalizer.* Anywhere a
quantity must literally sum to one — the internal model $q(dot mid(|) c_t)$, the
partition-of-unity guarantee, the conservation of normalization under learning —
the obligation sits on the softmax. The learning rule's *sum-mode* conservation
($sum_j Delta W_(i j) = 0$, established in the plasticity section) then preserves
$sum_j q_j = 1$ at every step, because adding a constant to a softmax's logits
leaves its output unchanged.

#remark[
  These two motifs supply, respectively, the predictor's *memory* (line/ring
  attractor holding $c_t$) and the decoder's *choice* (WTA competition, gain
  control producing the simplex-valued readout). Both are imported from the
  literature — attractor short-term memory from @seung1996 @benyishai1995,
  divisive gain control from @carandiniheeger2012; the contribution here is how
  they compose into an exact latency code, not the motifs themselves. The circuit
  section assembles them.
]

// ===== S8-circuit =====
= The circuit: predictor loop, calibrated readout, temporal-WTA decoder

We now assemble the archetype. It has three stages in series, of which *only the
first contains a cycle* — and that, as @sec-neurons argued, is exactly where the
work that a feedforward pipeline cannot do gets done. Stage 1 is a recurrent
*predictor* that turns held context into a probability vector $q(dot mid(|)
c_t)$; stage 2 is the *calibrated readout* that races each symbol's probability
to a first-spike latency equal to its surprisal; stage 3 is a *temporal
winner-take-all (WTA) decoder* that names the symbol by which readout fires
first. The two notes this paper merges each supplied half of stage 1: the
calibration drive of the coder note assumed an internal model $q$ it never made
explicit, and the plasticity note supplied that model as a softmax over lateral
weights. We state the unified object once, here, and the gap closes (cross-cutting
reconciliation (ii) of the coherence brief).

#figure(
  block(width: 100%, inset: 10pt, fill: luma(245), radius: 4pt)[
    #set text(font: "DejaVu Sans Mono")
    #raw("  symbol x_t ──→┌──────────────────────────────────────┐
                │  (1) RECURRENT PREDICTOR LOOP        │  context c_t: a bump on a
   inhibitory   │      line / ring attractor           │  line/ring attractor — ONE
   feedback ←───┤      holds context c_t;              │  graded analog value, a
   (shunting /  │      logits a_j = Σ_i W_ij c_i;      │  finite (noise-limited)
    predictive) │      q_j = softmax_j(W c_t)          │  summary of recent symbols
                └───────────────────┬──────────────────┘
                                    │  drive  R·I_j = θ/(1 - q_j^α)
                                    ↓
                ┌──────────────────────────────────────┐
                │  (2) CALIBRATED READOUT / ENCODER    │  N LIF neurons race to θ;
                │      t*(q_j) = -λ log2 q_j           │  the realized symbol's
                │      (latency = surprisal, bits)     │  neuron fires at t = bits·λ
                └───────────────────┬──────────────────┘
                                    │  spike train  (time axis = bit tape)
                                    ↓
                ┌──────────────────────────────────────┐
                │  (3) TEMPORAL-WTA DECODER            │  first spike to cross θ
                │      lateral inhibition;             │  wins the window, sets
                │      F G (one settled winner/        │  decoded := emitted, then
                │      window); decode = first-        │  new_window resets the
                │      spike-takes-all                 │  layer = self-clocking
                └──────────────────────────────────────┘", lang: none)
  ],
  caption: [The three-stage spiking entropy coder. Stage (1) is the only cyclic
    stage: the only one that can carry the prediction back to cancel the expected
    drive (predictive subtraction, via inhibitory feedback) and the only one that
    holds context. Stages (2)–(3) are the CogSpike winner-take-all archetype,
    repurposed as a coder's encode/decode head. The predictor model
    $q = softmax(W c_t)$ is stated once here; everything downstream uses it.],
) <fig-circuit>

== Stage 1 — the predictor: a softmax over lateral weights

The predictor must turn the held context $c_t$ into a probability vector over the
alphabet $cal(X)$ (size $N$). We make this map explicit, since it is the object
the coder note left unspecified and the plasticity note supplied.

#background("the softmax map, and why it is the simplex normalizer")[
  Given a vector of real *logits* $a = (a_1, dots, a_N)$ — unconstrained drive
  levels, one per symbol — the *softmax* map produces a probability vector
  $ q_j = e^(a_j) / (sum_(k=1)^N e^(a_k)). $
  Each $q_j > 0$ and $sum_j q_j = 1$ *exactly and identically* in the logits:
  exponentiating makes every entry positive, and dividing by the common pool sum
  $sum_k e^(a_k)$ forces the entries to tile the unit interval. Softmax is the
  canonical way to read a set of unconstrained drives as a distribution on the
  probability simplex; adding a constant to *every* logit leaves $q$ unchanged
  (the *softmax gauge*).
]

#definition[
  *The predictor.* The recurrent population holds context $c_t = (c_1, dots,
  c_N)$ — the activity of $N$ context lines (@sec-recurrent) — and projects it
  through *lateral weights* $W_(i j)$ to form, for each readout $j$, the *logit*
  (net drive)
  $ a_j = sum_i W_(i j) c_i, $ <eq-logit>
  from which the *predicted distribution* is the softmax
  $ q_j (c_t) = softmax_j (W c_t) = e^(a_j) / (sum_k e^(a_k)). $ <eq-predictor>
  With a *one-hot* context $c = e_i$ ("the last symbol was $i$"), @eq-logit
  collapses to $a_j = W_(i j)$, so each context *row* $W_(i dot)$ is the
  circuit's belief about *what follows symbol $i$* — a log-odds vector whose
  softmax is the conditional law $q(dot mid(|) i)$.
]

#intuition[
  The lateral weight $W_(i j)$ is a *log-odds*: how much seeing $i$ argues for
  $j$ coming next. The attractor's job is to make $c_t$ a clean indicator of the
  relevant past; the weights' job is to convert that indicator into a calibrated
  bet. Reading @eq-predictor right to left: context selects a row of $W$, the row
  is a vector of bets, softmax turns the bets into a distribution. This is the
  one object the rest of the circuit consumes.
]

The partition-of-unity invariant $sum_j q_j = 1$ rests *entirely on the softmax*
of @eq-predictor — it holds identically, for any weights and any context. This is
reconciliation (i) of the coherence brief, and it matters: the biophysical
realization of @eq-predictor is *divisive normalization* @carandiniheeger2012, in
which each exponentiated drive is divided by a pool term $sigma + sum_k a_k$. That
gain-control sum is $sum_k a_k slash (sigma + sum_k a_k) < 1$, so divisive
normalization does *not* by itself enforce $sum_j q_j = 1$; it is the
$sigma arrow 0$ approximation to the exact simplex normalizer. We therefore put
the invariant on the softmax and treat divisive normalization as its
biophysical realization, valid as $sigma arrow 0$.

#background("holding the context: a finite, noise-limited memory")[
  The context $c_t$ is held as a bump of *graded persistent activity* on a line or
  ring attractor @seung1996 @benyishai1995 (@sec-recurrent): the network rests
  anywhere along a continuum of fixed points, and the resting position *is* the
  stored value. Crucially this stores *one graded analog value*, not an unbounded
  tape: membrane and synaptic noise make only $tilde.op log_2("SNR")$ distinct
  positions reliably distinguishable, so the attractor captures *a finite,
  noise-limited number of distinguishable context states*, i.e. memory order up
  to that capacity (fix G). For a first-order source the relevant context is the
  previous symbol, comfortably within capacity; deeper memory is bounded by it
  and is the subject of @sec-limitations.
]

*Predictive subtraction, with the correct sign.* The cycle's defining operation
is to cancel the *expected* drive so the soma integrates only the unpredicted
*residual* (@sec-neurons). Because the predicted drive is *excitatory*, cancelling
it requires *inhibitory* — shunting or hyperpolarizing — feedback, not an additive
depolarization "equal to the prediction" (fix F). The feedback path delivers
inhibition matched to $q_j$, so a perfectly predicted symbol meets balanced
excitation and inhibition and the soma sees near-zero net drive. The clean
arithmetic "soma integrates $"input" - "prediction"$" is the *shunting-balanced
idealization*, exact only when the inhibitory conductance tracks the excitatory
drive linearly over the operating range; we use it as an idealization and flag the
regime where it holds.

== Stage 2 — the calibrated readout: latency is surprisal

Each readout neuron $j$ is a leaky integrate-and-fire (LIF) unit (@sec-neurons)
driven by a current set from its predicted probability. The coder note's
calibration drive encodes *exactly* the softmax $q_j$ of @eq-predictor: the
predictor pre-charges readout $j$ with the drive
$ R I_j (c_t) = theta / (1 - q_j (c_t)^alpha), quad q_j (c_t) = softmax_j (W c_t), $ <eq-drive>
with the calibration exponent $alpha = lambda slash (tau ln 2)$ fixed once. By the
latency-calibration theorem (@sec-identity), the first-spike latency of readout
$j$ is then exactly
$ t^*(q_j) = -lambda log_2 q_j (c_t), $
the surprisal of symbol $j$ under the circuit's model, in seconds-per-bit units of
$lambda$. The realized symbol $x_t$ is the one whose readout fires; its latency is
$-lambda log_2 q(x_t mid(|) c_t)$, so over a stream the total first-spike time is
$lambda$ times the total surprisal — the cross-entropy result of @sec-identity.
The time axis is the bit tape, and $lambda$ (s/bit) is the *exact, proven*
calibration constant; we never silently exchange it for the model-dependent
energy constant $kappa$ (reconciliation (iii)).

#honest[
  The drive @eq-drive *blows up well inside the working range*, so $q approx 1$ is
  not physically reachable. With $tau = lambda = 1$ (hence $alpha = 1 slash ln 2
  approx 1.4427$):
  #figure(
    table(
      columns: (auto, auto, auto, auto, auto, auto),
      align: (right, right, right, right, right, right),
      stroke: 0.5pt + luma(180),
      table.header[$q$][$0.5$][$0.7$][$0.9$][$0.95$][$0.99$],
      [$R I slash theta = (1 - q^alpha)^(-1)$], [$1.6$], [$2.5$], [$7.1$], [$13$],
      [$70$],
    ),
    caption: [Calibration drive $R I slash theta$ versus model probability $q$
      ($alpha approx 1.4427$). The drive needed grows steeply: $approx 7.1 times$
      rheobase at $q = 0.9$ and $approx 70 times$ at $q = 0.99$.],
  ) <tab-drive>
  A real neuron has a *few-fold* rheobase ceiling, so the largest representable
  probability is pinned at $q_max approx 0.85$–$0.95$, with a small *floor cost*
  on near-certain symbols (they fire fast but not instantly). At the other end,
  $q arrow 0$ is the *noise-dominated* regime: the latency grows without bound and
  membrane noise makes the timing code least reliable there. This sharpens, rather
  than retracts, the timing-resolution honesty check of @sec-identity (fix E).
]

== Stage 3 — the temporal-WTA decoder: first-spike-takes-all

The decoder is the CogSpike winner-take-all archetype: $N$ units that excite
themselves and *laterally inhibit* one another (@sec-recurrent). Fed the readout
spike train, it must (a) decide *which* symbol was sent and (b) self-clock the
stream by delimiting one symbol from the next. We make the windowed semantics
concrete so the obligation is instantiable (fix H / FM-01).

#definition[
  *Windowed decode semantics.* A *window* is the integration epoch for one symbol,
  opened by the predicate $"new_window"$ (asserted when the layer is reset to rest)
  and closed by the first readout crossing threshold $theta$. Within a window we
  define the labels:
  - $"winner"_j$ — readout $j$ is the *settled* winner: it has crossed $theta$ and
    lateral inhibition has driven every other unit below $theta$ and held it there;
  - $"emitted"$ — the symbol whose readout actually crossed $theta$ first (the
    physical first-spike event);
  - $"decoded"$ — the symbol the decoder commits to, set by the
    *first-spike-takes-all* rule $"decoded" := "emitted"$ at the close of the
    window;
  - $"new_window"$ — the reset event that re-opens integration for the next symbol.
  *First-spike-takes-all* means the decode is committed at the *first* threshold
  crossing and is insensitive to the subsequent settling transient.
]

The naive safety claim "*at most one winner, always*" — $P_(>=1)[G(sum_j
"win"_j <= 1)]$ — is *false as a global invariant*: during the integration
transient several readouts are simultaneously subthreshold-active, and more than
one may briefly sit near $theta$ before lateral inhibition resolves the
competition (fix H). The property the archetype actually certifies is the
*settled* form.

#proposition[
  *(Settled uniqueness of the decoder; imported.)* In each window, after a
  bounded settling transient, the temporal-WTA layer
  reaches a state with *exactly one* settled winner, which persists until
  $"new_window"$:
  $ P_(>=1) [ space F space G space ( sum_j "winner"_j = 1 ) space ], $ <eq-fg>
  where $F G$ ("eventually, forever-after within the window") is the standard
  liveness-into-stability operator. The transient in which $sum_j "winner"_j$ may
  momentarily differ from $1$ is bounded by the WTA settling time $t_("settle")$,
  which first-spike-takes-all *tolerates*: the decode is fixed at the first
  crossing, before settling completes. The underlying dynamical fact — that
  symmetric contralateral inhibition has the single-winner configurations as its
  only stable attractors, each reached after a bounded transient — is the
  liveness-plus-stability property the CogSpike contralateral-inhibition
  archetype establishes by the mode decomposition recalled in @sec-correctness;
  we *import* it here rather than re-derive the $N$-symbol normal form, and flag
  the import in the Critical Evaluation (@sec-limitations).
]

#intuition[
  Read @eq-fg as "the decoder is *eventually decisive*, not *instantaneously
  decisive*." Several readouts charging toward threshold at once is not a bug; it
  is the race that *is* the computation. What must be true is that the race has a
  unique first finisher and that the network then *latches* that finisher —
  exactly the liveness ($F$, someone wins) plus stability ($G$, and stays the
  only winner) that the contralateral-inhibition archetype already proves. The
  global "$<= 1$" reading mistook a transient for an invariant.
]

This is the load-bearing correction to the slogan *"a bad model is slow, never
wrong."* That guarantee rests on the *decode rule*, not on any global
mutual-exclusion (fix H). The decoded symbol is whichever readout crosses
threshold first; the realized symbol's readout is driven by @eq-drive at latency
$-lambda log_2 q(x_t mid(|) c_t)$, which is *finite for every $q in (0, q_max]$*.
A poor model inflates that latency — the symbol is emitted *late* (high
cross-entropy, the slowness) — but it is still the realized symbol's readout that
fires, so $"decoded" = "emitted"$ holds regardless of how poor $q$ is. Correctness
is therefore *structural* (it survives any predictor), while the *speed* of the
code is *quantitative* (it is what learning improves). The two verification
classes of @sec-obligations inherit exactly this split.

Finally, the first spike *resets the layer* — asserts $"new_window"$ — which
restarts the integration clock for the next symbol. This is what makes the latency
code *self-clocking and prefix-free*: each spike delimits its own symbol, so the
decoder needs no external frame and no lookahead, achieving by *timing* what a
prefix code (@sec-info) achieves by the no-prefix rule. The first-spike latency
code is the time-domain image of an instantaneous code, and time-to-first-spike
readout is the canonical neural mechanism for it @thorpe1996 @thorpe2001.

#remark[
  *The information–circuit dictionary, in one table.* Every information-theoretic
  object of @sec-info–@sec-info has a concrete spiking realization in the three
  stages above; the partition-of-unity row now points at the softmax, not at
  divisive normalization (reconciliation (i)).
  #table(
    columns: (auto, auto),
    align: (left, left),
    stroke: 0.5pt + luma(200),
    table.header[*Information-theory object*][*Spiking realization*],
    [model probability $q_j$], [softmax of lateral logits, $q_j =
      softmax_j(W c_t)$ (@eq-predictor)],
    [drive encoding $q_j$], [$R I_j = theta slash (1 - q_j^alpha)$ on readout $j$
      (@eq-drive)],
    [surprisal $-log_2 q_j$], [first-spike latency $t^*_j slash lambda$ (*exact*,
      @sec-identity)],
    [entropy rate $overline(H)(p)$ (floor)], [min expected spike-time per symbol],
    [cross-entropy rate $overline(H)(p,q)$ (cost)], [actual spike-time emitted
      (@sec-identity)],
    [partition of $[0,1)$ (Kraft)], [$sum_j q_j = 1$ via softmax;
      basins tile attractor state],
    [normalization $sum_j q_j = 1$], [softmax simplex constraint (divisive
      normalization is the $sigma arrow 0$ realization)],
    [prefix-free / self-delimiting], [first spike resets layer (self-clocking)],
    [the predictor $P(x_t mid(|) "history")$], [persistent loop state $c_t$ on a
      line/ring attractor],
    [nearest-codeword decoding], [WTA settling to the settled winner (@eq-fg)],
  )
]

With the three stages assembled and the predictor $q = softmax(W c_t)$ made
explicit, the circuit is fully specified: @sec-correctness certifies stage 3
through the sum/difference mode decomposition, and @sec-learning supplies the
plasticity that drives $W$ — and hence $q$ — toward the source.

// ===== S9-modes =====
= Correctness via the sum/difference mode decomposition <sec-correctness>

The decode head of stage 3 (§8) is exactly the symmetric competitive network the
CogSpike program analyses with a *mode decomposition*, so the existing
closed-form machinery certifies the coder without new proof burden. We supply the
dynamical-systems background from first principles, then read off two guarantees
and, crucially, separate what each one rests on: *which symbol* is decoded
(a difference-mode question) and *that the model probabilities form a valid
partition of unity* (a sum-mode question). Throughout, the partition-of-unity
invariant $sum_j q_j = 1$ is placed on the *softmax* that defines the predictor
(coherence brief (i)–(ii)); divisive normalization enters only as the
biophysical gain-control realization, valid as its semi-saturation constant
$sigma arrow 0$.

#background("bifurcations, and the sum/difference modes")[
  A dynamical system is a rule for how a state evolves in time; here the state is
  the vector of competing readout activities. As a *control parameter* (for us,
  the strength of the lateral competition) is varied, the system's *fixed points*
  — states it can rest at — can change in number or stability. Such a qualitative
  change is a *bifurcation*. The one that matters for a two-way decision is the
  *pitchfork*: below a threshold a single symmetric state is stable (the two
  units tie, no decision is made); as the parameter crosses the threshold that
  symmetric state loses stability and *two* asymmetric stable states are born —
  one in which the first unit wins, one in which the second does. For a symmetric
  pair of competing units with activities $nu_1, nu_2$ it is natural to change
  coordinates to a *sum mode* $u = nu_1 + nu_2$ (the total activity) and a
  *difference mode* $d = nu_1 - nu_2$ (who is ahead). Near the symmetric fixed
  point these two coordinates *decouple* — the linearized dynamics act on $u$ and
  $d$ separately — so a two-unit competition splits into two independent scalar
  problems. The $N$-symbol generalization is handled mode-by-mode in the
  program's `closed_form_wta_multi` study.
]

#intuition[
  The change of variables is the whole trick. In the raw coordinates
  $(nu_1, nu_2)$ the competition is a coupled tangle. In $(u, d)$ it is two
  knobs: a *total-activity* knob $u$ that a gain control pins to a budget, and a
  *who-is-ahead* knob $d$ that a pitchfork pushes off centre. The two
  information-theoretic obligations of the coder fall, one each, onto these two
  knobs.
]

== Two invariants, two roles

We now attach each obligation to its mode. The difference-mode statement is
*structural* and unconditional. The sum-mode statement is the partition-of-unity
invariant, which we prove here directly from the softmax (it is immediate), and
which the companion learning rule then preserves at every step.

#proposition[
  *Difference mode = symbol selection.* For the symmetric two-readout block the
  difference coordinate $d = nu_1 - nu_2$ obeys a pitchfork in the competition
  strength: above the decision threshold the symmetric state $d = 0$ is unstable
  and the dynamics settle to one of the two signs of $d$. The settled *sign* of
  $d$ (the $arg max$ over readouts, for $N > 2$) names which readout crosses
  threshold first, i.e. the decoded symbol. The pitchfork threshold is thus the
  *decision boundary* between adjacent symbols, and the symmetric $d = 0$ locus
  is the boundary's knife-edge.
]

#proof[
  This is the difference-mode half of the WTA analysis the program already
  carries out in closed form. Near the symmetric fixed point the linearized
  difference-mode dynamics read $dot(d) = mu d - c d^3 + O(d^5)$ with $mu$ an
  increasing function of the competition strength and $c > 0$ set by the
  saturating nonlinearity; $mu < 0$ leaves $d = 0$ the only stable rest state
  (a tie), while $mu > 0$ destabilizes it and creates the pair
  $d = plus.minus sqrt(mu slash c)$ (a winner). Decoding reads the sign of the
  settled $d$. We do not reproduce the full $N$-symbol normal-form computation
  here; it is exactly the `closed_form_wta_multi` result, to which we defer for
  the quantified transient and the inverse-staircase ordering of the
  thresholds.
]

#proposition[
  *Sum mode = partition of unity.* The predictor's distribution is the softmax of
  the lateral drive (coherence brief (ii)), $q_j = softmax_j (W c_t) = e^(a_j)
  slash sum_k e^(a_k)$ with logits $a_j = sum_i W_(i j) c_i$. Then the
  partition-of-unity invariant
  $ sum_j q_j (c_t) = sum_j e^(a_j) / (sum_k e^(a_k)) = (sum_j e^(a_j)) / (sum_k
  e^(a_k)) = 1 $ <eq-partition>
  holds *exactly and identically* in $c_t$ and in $W$: the normalizing
  denominator is by construction the sum of the numerators. This is the
  information-theoretic Kraft constraint "the codeword intervals must tile
  $[0, 1)$" (§4) realized as a *normalization invariant on the sum mode* — the
  total exponentiated drive is divided out, so the responses share a unit budget.
]

#proof[
  Immediate from the definition of the softmax: the numerator summed over $j$
  equals the denominator. No dynamics, no fixed-point search, and no smallness
  assumption is required — the invariant is algebraic in the readout map itself.
]

#remark[
  *Where divisive normalization fits, and where the WTA settling dynamics are
  certified.* The biophysical gain control the decode head actually implements is
  Carandini–Heeger divisive normalization @carandiniheeger2012,
  $r_i = a_i slash (sigma + sum_j a_j)$, which sums to $sum_i a_i slash (sigma +
  sum_j a_j) < 1$ and therefore does *not by itself* enforce $sum_j q_j = 1$; it
  is the gain-control *approximation* to the simplex normalizer, exact only in
  the limit $sigma arrow 0$. The exact partition-of-unity invariant lives on the
  softmax of @eq-partition, not on divisive normalization. Two consequences. (a)
  *Preservation under learning is free:* the companion contralateral-inhibition /
  learning study shows that the delta-rule update conserves the sum mode per
  context, $sum_j Delta W_(i j) = eta c_i (sum_j y_j - sum_j q_j) = eta c_i
  (1 - 1) = 0$ @widrowhoff1960, so learning only moves predictive mass *between*
  symbols (the difference modes) and never disturbs @eq-partition; the validator
  confirms the per-context row-sum drift stays at machine precision. (b) *The
  full settling dynamics* — that the difference-mode pitchfork resolves to a
  unique winner within a bounded transient, and the quantitative WTA safety
  statement — are *deferred to the companion contralateral-inhibition study* and
  not re-derived here; this section supplies only the mode-by-mode bookkeeping
  that maps each information-theoretic obligation onto a mode. We flag this
  deferral explicitly per the proof-existence requirement rather than implying a
  self-contained dynamical proof.
]

#honest[
  The mutual-exclusion guarantee is a *settled-time* statement, not a global
  invariant. "At most one winner" is *false* as an always-property: during the
  integration transient several readouts are simultaneously sub-threshold-active.
  What the archetype certifies is the liveness-then-safety form — *eventually
  exactly one settled winner per decode window* ($F G$ in temporal-logic terms) —
  together with a quantified transient that the first-spike-takes-all rule
  tolerates. Accordingly, "a bad model is *slow*, never *wrong*" rests on the
  *decode rule* (the first readout to cross threshold names the symbol), not on
  any claim of global mutual exclusion. We carry the precise windowed temporal
  obligations in §11; here we only note that the difference-mode pitchfork is
  what *makes* a unique settled winner exist.
]

#intuition[
  The two information-theoretic constraints land on the two mode families the
  program already studies. *Which* symbol is decoded is a difference-mode
  question — a pitchfork past a threshold (Proposition on selection). *That the
  model probabilities sum to one* is a sum-mode question — but, properly placed,
  it is not a dynamical fixed-point burden at all: it is the algebraic identity
  @eq-partition of the softmax, which the learning rule then leaves untouched.
  So "decode correctly" and "the code is a valid probability partition" are not
  new proof obligations specific to the coder; they are, respectively, the
  uniqueness/liveness property of the WTA archetype (read through a decoding
  lens) and an immediate property of the predictor's normalization.
]

== Correctness is unconditional; optimality is earned

The mode split yields the organizing principle of the whole verification effort
(§11): *correctness is structural and holds for any predictor, optimality is
quantitative and is what learning improves.*

Losslessness needs only that the decode head elects exactly one settled winner
per window and that the winner names the emitted symbol — a *difference-mode*,
structural property of stage 3 that holds *no matter how poor the predictor $q$
is*. The partition-of-unity side (@eq-partition) is likewise structural: it is
true of every $W$, learned or not. A bad predictor therefore makes the code
*slow* — high cross-entropy rate, hence long total spike-time (Thm "Spike-time
equals cross-entropy", §5) — but it can never make the code *wrong*: every symbol
is still emitted and uniquely decoded, and the probabilities it commits to still
sum to one. Optimality — closing the gap $overline(D)_("KL")(p mid(||) q)$ down
to the entropy-rate floor — is the *earned*, predictor-dependent property, and it
is exactly the difference-mode mass that learning redistributes while the sum
mode stays pinned. The two verification obligations of §11 inherit precisely this
split: a structural/safety class that ignores $q$, and a quantitative/optimality
class that grades it.

// ===== S10-energy =====
= The model error: in seconds, and (with a model) in joules

Two sections ago we proved an *exact, time-valued* identity: the calibrated
readout fires at latency $t^*(q) = -lambda log_2 q$, so a stream's total
first-spike time is $lambda$ times its total surprisal, and its expected
per-symbol time is the cross-entropy rate $lambda dot overline(H)(p, q)$
(Theorems on latency calibration and spike-time$=$cross-entropy). Everything in
that statement is in *seconds*, and the constant $lambda$ (s/bit) is *exact and
proven* by the calibration drive — it is a property of the LIF differential
equation under @eq-calib, nothing more. This section first restates the model
error in those proven, time-valued terms, then asks the natural follow-up
question — *what does it cost in joules?* — and answers it *only* under an
explicit, declared energy model. The distinction is load-bearing, so we make it
the spine of the section.

== The proven part: model error is excess spike-time

The surprisal decomposition $overline(H)(p, q) = overline(H)(p) +
overline(D)_("KL")(p || q)$ (§3, lifted to rates in §6) is already a complete,
exact statement once read through the latency code.

#proposition[
  *Model error is excess latency.* A calibrated coder with predictor $q$ emits,
  per symbol in expectation, first-spike time
  $ lim_(n arrow.r infinity) 1 / n EE[T_n] = lambda dot overline(H)(p, q)
  = lambda overline(H)(p) + lambda dot overline(D)_("KL")(p || q), $ <eq-time-decomp>
  whose *irreducible* part $lambda overline(H)(p)$ is fixed by the source and
  whose *excess* $lambda dot overline(D)_("KL")(p || q) gt.eq 0$ is spent purely
  for the model being wrong, vanishing iff $q = p$.
]

#proof[
  Take expectations in the spike-time$=$cross-entropy theorem over the stationary
  source $p$, holding the loop's contexts at the true predecessors, and apply the
  rate form of the surprisal decomposition. Non-negativity of
  $overline(D)_("KL")$ and its vanishing exactly at $q = p$ are Gibbs'
  inequality @shannon1948.
]

This is the honest, substrate-free reading of "the model error": it is a number
of *seconds per symbol*, $lambda dot overline(D)_("KL")(p || q)$, and it is
proven by the same ODE that gave us @eq-calib. No biophysics beyond the LIF
membrane equation enters. Keep this firmly separate from what follows.

#honest[
  We resist the tempting one-liner "spike-time *is* energy." It is not an
  identity. The latency code proves a *time*; converting time to metabolic
  *joules* requires a model of where the joules go, and that model can fail. The
  rest of this section supplies one such model and is explicit about its regime
  of validity. The calibration constant $lambda$ (s/bit, proven) and the energy
  constant $kappa$ (J/bit, model-dependent, introduced below) are *different
  constants with different units*; we never silently swap one for the other.
]

== A minimal biophysical energy model

To talk about joules at all we must posit how a spiking circuit spends them. We
adopt the simplest model faithful to neural energetics, and we name its two
terms so the reader can see exactly which assumptions carry the conclusion.

#background("where a spiking neuron's energy goes")[
  A biological neuron's metabolic bill (ATP, ultimately) has two dominant
  components. (1) A *per-spike* cost: each action potential pumps ions back
  across the membrane to restore the resting gradient, a roughly *fixed* energy
  $E_("spk")$ per spike, independent of how long the neuron took to fire. This is
  the dominant signalling cost in cortex. (2) A *subthreshold* cost: while the
  membrane integrates drive toward threshold it leaks current continuously, and
  maintaining that drive against the leak dissipates power $approx P_("sub")$ for
  the duration of the integration — so its energy scales with the *latency*. A
  neuron that sits a long time just below threshold (a rare, weakly-driven
  symbol) accumulates leak the whole while.
]

#definition[
  *Minimal energy model.* In response to one symbol, charge a readout neuron
  $ E = underbrace(n_("spk") dot E_("spk"), "per-spike (count)")
        + underbrace(P_("sub") dot t^*, "subthreshold (latency)"), $ <eq-energy-model>
  where $n_("spk")$ is the number of spikes it emits, $E_("spk") > 0$ is the
  fixed per-spike ATP cost, $P_("sub") gt.eq 0$ is the integrated subthreshold
  drive power, and $t^*$ is the first-spike latency of @eq-latency. The
  *energy-per-bit* constant is $kappa$ (J/bit); its value depends on which of the
  two terms dominates, as we now show.
]

This model is deliberately spare, but it already exposes that "energy" and
"spike-time" answer to *different* terms: the per-spike term tracks a *count*,
the subthreshold term tracks a *latency*. Whether total energy is proportional
to surprisal therefore depends on which code the circuit runs.

== The corollary: a regime where energy is proportional to surprisal

#theorem("Energy-error corollary (model-dependent)")[
  Suppose the circuit runs a *spike-count* code in which the number of spikes a
  readout emits for its symbol is proportional to that symbol's surprisal,
  $n_("spk")(x) = beta dot (-log_2 q(x))$ for a fixed $beta > 0$, and suppose the
  per-spike term dominates the subthreshold term ($E_("spk") n_("spk") gt.double
  P_("sub") t^*$). Then with $kappa := beta E_("spk")$ (J/bit) the expected
  per-symbol energy is
  $ EE[E] = kappa dot overline(H)(p, q)
    = kappa overline(H)(p) + kappa dot overline(D)_("KL")(p || q), $ <eq-energy-decomp>
  so the *excess* energy paid for model error is exactly $kappa dot
  overline(D)_("KL")(p || q) gt.eq 0$, vanishing iff $q = p$.
]

#proof[
  Under the stated regime $E approx E_("spk") n_("spk") = beta E_("spk")(-log_2
  q(x)) = kappa dot I_q(x)$ per symbol. Take the stationary expectation and apply
  the surprisal-rate decomposition exactly as in @eq-time-decomp, with $kappa$ in
  place of $lambda$.
]

The corollary recovers the clean slogan — *excess energy equals KL divergence* —
but now as a *consequence of a declared model*, not as a fundamental identity. It
is the energy analogue of @eq-time-decomp, and it is true precisely when the code
is count-based and per-spike costs dominate. We state plainly where it breaks.

#honest[
  *Where the proportionality fails.* The corollary leans on two assumptions, and
  the *latency* code of §5 violates the first. In the latency code a symbol's
  surprisal is carried by *when* a single spike fires, not by *how many* fire:
  $n_("spk") = 1$ for every symbol, so the per-spike term contributes a *constant*
  $E_("spk")$ regardless of surprisal, and the surprisal-dependence lives entirely
  in the subthreshold term $P_("sub") t^* = P_("sub") lambda(-log_2 q)$. There the
  *sign of the inequality flips*: a *rare* symbol ($q$ small) has a *long* latency
  $t^*$, so it draws subthreshold power for *longer* and accumulates *more* leak —
  energy still grows with surprisal, but through the latency term, and only if
  $P_("sub")$ is the dominant cost. If instead the fixed per-spike cost dominates
  (the biologically typical case), the single-spike latency code spends *nearly
  the same energy on every symbol*, and total energy is *not* proportional to
  surprisal at all. A long-latency rare symbol may even draw *less instantaneous*
  power than a brisk common one while accumulating more *integrated* leak. So:
  the count code makes energy $prop$ surprisal via spike count; the latency code
  makes *time* $prop$ surprisal exactly (proven) but makes *energy* $prop$
  surprisal only under the subthreshold-dominated regime. The two codes optimize
  different ledgers, and the energy reading is a corollary of *which* ledger and
  *which* dominant cost — never an identity.
]

== The stupidity tax, scoped to the model

Within the energy corollary's regime we can quote the model error for the
momentum rover of §6 in joules. These numbers are *scoped to* @eq-energy-decomp
and presuppose the count-code / per-spike-dominated regime; they are not claims
about the latency code, and not fundamental thermodynamics.

The entropy-rate floor at stickiness $s = 0.7$ is $overline(H)(p) = 0.9782$
bits/symbol; the marginal (memoryless) entropy is $H(pi) = 1.7500$ bits/symbol.
A circuit that ignores the memory entirely — a memoryless predictor $q = pi$ —
pays cross-entropy rate equal to the marginal $1.7500$, an excess of
$ overline(D)_("KL")(p || q_("memoryless")) = H(pi) - overline(H)(p)
  = I(x_t; x_(t-1)) = 0.7718 "bits/symbol" $
over the floor. In the corollary's units that is a flat $0.7718 kappa$ joules per
symbol thrown away for not modelling the inertia — a *#calc.round(0.7718 / 0.9782 * 100)%* surcharge over the conditional floor, the *stupidity tax* for a
memoryless model. A circuit that *does* use memory but learns the *wrong*
stickiness $s' = 0.4$ pays a smaller but nonzero excess of
$overline(D)_("KL")(p || q_(s'=0.4)) = 0.1351$ bits/symbol, i.e. $0.1351 kappa$
joules per symbol (both values reproduced by `validate.py`).

#remark[
  *Two baselines, two percentages — keep them straight.* The *44%* figure quoted
  elsewhere in the note ("the memory is worth $0.7718$ bits") measures the
  mutual information $0.7718$ against the *marginal* $1.7500$ baseline: memory
  recovers $0.7718 slash 1.7500 approx 44%$ of the memoryless cost. The
  *stupidity-tax* percentages here measure the *same* $0.7718$ (and the
  $0.1351$) bits against the *conditional floor* $0.9782$: a memoryless model
  pays $0.7718 slash 0.9782 approx 79%$ *more* than a perfect one. Same divergences,
  different denominators; both are energy-corollary statements under
  @eq-energy-decomp, not properties of the proven time code.
]

#intuition[
  *Compression-as-intelligence, read thermodynamically — under a model.* In the
  energy corollary's regime the only adjustable part of the metabolic bill is
  $kappa dot overline(D)_("KL")(p || q)$; the floor $kappa overline(H)(p)$ is set
  by the world. So a metabolically bounded circuit that minimizes its spiking
  *count* is driven to shrink $overline(D)_("KL")$ — to make its internal model
  $q$ resemble the true conditional law $p$. Energy minimization *is* model
  learning, *in this regime*. We flag the scope deliberately: the *proven*
  version of this statement is the time-valued one (@eq-time-decomp), where
  "spend the least latency" is provably "predict best"; the joule-valued version
  is the corollary, true under the declared energy model.
]

#honest[
  *Numerics honesty.* The floor is $overline(H)(p) = 0.9782$ bits/symbol
  *everywhere*; a *learned* predictor converges to energy $0.9787$ — slightly
  *above* the floor, a finite-learning residual ($overline(D)_("KL") = 0.0004$),
  not below it — so we never call $0.979$ "the floor." The empirical
  perfect-predictor stream cost $0.9760$ reported in §5 is a *sample mean* over a
  finite stream and may dip below the expectation $0.9782$ by ordinary
  finite-sample fluctuation; that is not a violation of the bound. All
  stupidity-tax figures ($0.7718$ memoryless, $0.1351$ wrong-momentum) are
  divergences scoped to the energy corollary's model; the proven, unconditional
  statements of this paper are the time-valued ones.
]

// ===== S11-learning =====
= Learning to be unsurprised: the local rule, derived <sec-learning>

The circuit of the previous sections pays, in spikes, the surprisal
$ell_t = -log_2 q(x_t mid(|) c_t)$ of every symbol it emits, and its expected
per-symbol cost is the cross-entropy rate $overline(H)(p, q)$ of its internal
model $q$ against the source. That cost is minimized — driven to the
entropy-rate floor $overline(H)(p)$ — exactly when the model equals the world,
$q = P$. What was assumed, not built, is the *learner* that gets there. We build
it here, and the pleasant surprise is that it is not a separate apparatus: the
gradient of the circuit's own energy is, term for term, a quantity the substrate
already has at each synapse. This section derives that local rule from first
principles, states honestly which part of "the cost spike is the teaching
signal" is proved and which part is an imported physical hypothesis, and proves
the rule descends to the truth.

== Background a formal-methods reader needs

#background("gradient descent and its stochastic, decreasing-step form")[
  To minimize a differentiable scalar $V(W)$ of parameters $W$, *gradient
  descent* repeatedly steps downhill, $W arrow.l W - eta nabla V(W)$, with
  *learning rate* $eta > 0$; the gradient $nabla V$ is the vector of partial
  derivatives $partial V slash partial W_(i j)$ pointing in the direction of
  steepest increase, so its negative points downhill. If $V$ is *convex* (its
  graph curves upward everywhere, so any local minimum is global) descent reaches
  the global minimum. When $V$ is an average over data arriving one sample at a
  time, *stochastic gradient descent* (SGD) replaces the full gradient by a single
  sample's — a noisy but unbiased estimate. The *Robbins–Monro conditions*
  @robbinsmonro1951 pin down when SGD converges almost surely to the minimizer of
  a convex objective: a *decreasing* step schedule with $sum_t eta_t = infinity$
  (the steps do not sum out before arriving) and $sum_t eta_t^2 < infinity$ (the
  sampling noise is eventually damped). A *constant* step does not satisfy the
  second condition; it instead settles into a small *noise ball* of radius
  $O(eta)$ around the optimum, trading asymptotic accuracy for the ability to
  track a moving target.
]

#background("Hebbian plasticity, locality, and three-factor rules")[
  A *synaptic weight* $W_(i j)$ couples a presynaptic line $i$ to a postsynaptic
  unit $j$. A plasticity rule is *local* if its update to $W_(i j)$ uses only
  quantities physically present *at that synapse*: the presynaptic activity, the
  postsynaptic activity, and at most one globally broadcast scalar. *Hebbian*
  learning is the canonical local form — an update proportional to the *product*
  of pre- and postsynaptic activity, $Delta W_(i j) prop x_i^("pre")
  x_j^("post")$. A *three-factor* rule @fremauxgerstner2016 multiplies this
  Hebbian product by a third, global gate (a neuromodulator, or simply a
  learning-rate / reward signal): $Delta W_(i j) = (#[gate]) dot x_i^("pre") dot
  x_j^("post")$. Locality is what biology and on-chip neuromorphic hardware can
  implement without shuttling a global error vector to every synapse; it is the
  property a derived rule must have to be physically plausible. The rule we derive
  is exactly of this three-factor local form.
]

== The gradient identity (proved unconditionally)

Recall the predictor fixed earlier in the paper: the predicted distribution is
the *softmax* of the lateral drive, $q_j = e^(a_j) slash sum_k e^(a_k)$ with
logits $a_j = sum_i W_(i j) c_i$, so the softmax — not divisive normalization,
which is only its $sigma -> 0$ biophysical approximation — is what enforces the
partition of unity $sum_j q_j = 1$. We minimize the per-symbol energy, which by
the latency calibration is (up to a substrate constant) the surprisal
$ ell_t = -log_2 q(x_t mid(|) c_t) = -log_2 q_(x_t), quad q_j = e^(a_j) /
(sum_k e^(a_k)), quad a_j = sum_i W_(i j) c_i. $ <eq-ell>
Let $y_j = bb(1)[x_t = j]$ be the one-hot outcome indicator (whether symbol $j$
is the one that actually occurred). The following is pure algebra — the standard
softmax / cross-entropy gradient — and holds with no biophysical assumption
whatsoever.

#theorem("Energy gradient is a local Hebbian product")[
  The gradient of the per-symbol energy @eq-ell with respect to the lateral
  weight $W_(i j)$ is
  $ (partial ell_t) / (partial W_(i j)) = 1 / (ln 2) dot c_i dot (q_j - y_j). $ <eq-grad>
  Equivalently, the negative-gradient (descending) update, absorbing the constant
  $1 slash ln 2$ into the rate $eta$, is the *three-factor local rule* of
  Widrow and Hoff @widrowhoff1960,
  $ Delta W_(i j) = eta dot underbracket(c_i, "pre") dot underbracket((y_j -
  q_j), "signed residual" r_j), quad #[(global gate $eta$).] $ <eq-rule>
]

#proof[
  The softmax Jacobian is $partial q_m slash partial a_k = q_m (bb(1)[m = k] -
  q_k)$. Substituting into $ell_t = -log_2 q_(x_t) = -(1 slash ln 2) ln q_(x_t)$
  gives the standard cross-entropy–softmax logit gradient $partial ell_t slash
  partial a_k = (q_k - y_k) slash ln 2$ (the $1 slash ln 2$ converts nats to
  bits). Since $a_k = sum_i W_(i k) c_i$ is *linear* in the weights, $partial a_k
  slash partial W_(i j) = c_i bb(1)[k = j]$, and the chain rule collapses the sum
  to $partial ell_t slash partial W_(i j) = (q_j - y_j) c_i slash ln 2$, which is
  @eq-grad. Descending it with step $-eta'$ and writing $eta = eta' slash ln 2$
  yields @eq-rule.
]

The validator confirms @eq-grad against a finite-difference computation of
$partial ell_t slash partial W_(i j)$ to a maximum discrepancy of
$1.12 times 10^(-10)$ — the identity is exact, the residual numerical.

#honest[
  We claim no algorithmic novelty for @eq-rule: it is verbatim the delta /
  LMS rule @widrowhoff1960, and the sign pattern of the residual $r_j = y_j - q_j$
  — strengthen the predictor of what happened, weaken predictors of what did not —
  is the Rescorla–Wagner law of surprise-driven conditioning
  @rescorlawagner1972. The novelty (foregrounded in the next subsection and the
  Critical Evaluation) is the *interpretation*: that this residual is, under one
  explicit hypothesis, the same physical event the coder already emits.
]

== The cost spike is the teaching signal — stated as a conditional

Equation @eq-rule says the gradient *algebra* needs only the presynaptic context
$c_i$ and the postsynaptic *signed residual* $r_j = y_j - q_j$. It is tempting,
and it is the paper's central thesis, to read this as: *the surprise the circuit
pays is itself the error signal that lowers future surprise.* That reading is
genuinely available, but only conditionally, and we are careful to separate the
proved half from the imported half.

#theorem("Cost spike = gradient, conditionally")[
  *(Proved, unconditionally.)* The negative gradient of the per-symbol energy is
  $-partial ell_t slash partial W_(i j) = (1 slash ln 2) c_i (y_j - q_j)$
  (@eq-grad): a Hebbian product of presynaptic context with the signed residual.

  *(Realization hypothesis, imported.)* If the circuit physically emits the
  signed residual $r_j = y_j - q_j$ at readout $j$ — realized, since spikes are
  non-negative, by the two rectified ON/OFF error channels of the
  predictive-subtraction architecture (below) — then a local Hebbian correlation
  of those emitted error events with the active context line *is* the negative
  energy gradient, and the spikes the circuit pays as cost simultaneously carry
  the complete first-order gradient of that cost.
]

#honest[
  The realization hypothesis is doing real work, so we isolate it. *(a)* The
  gradient identity is algebra and needs nothing physical. *(b)* That the circuit
  *emits exactly* $y_j - q_j$ is a property imported from the
  predictive-subtraction architecture established earlier; it is a hypothesis
  about the substrate, not a theorem. *(c)* Crucially, this *error-population*
  observable is *distinct from the latency code* used to read out symbols. The
  latency code carries surprisal in spike *time* — one race-to-threshold readout
  per symbol, first-spike latency $t^* = -lambda log_2 q$ @thorpe1996
  @thorpe2001. The learning residual $r_j$ is a *separate* signal, carried by the
  two-channel error population, *not the same spikes*. Conflating them would be
  an equivocation; we do not. With these three caveats stated, the conditional
  identity — that the code event and the gradient event coincide — is the paper's
  genuine novelty, and we foreground it precisely *as a conditional*.
]

#intuition[
  Under the hypothesis, there is no error "backpropagated from a loss" living
  apart from the computation. When symbol $j$ occurs against a low prediction its
  error channel fires hard (large positive residual, large surprisal), and that
  same firing, gated by the active context line, *potentiates* $W_(i j)$ so the
  symbol is predicted better next time; when a symbol was expected but absent the
  OFF channel fires and $W_(i j)$ is *depressed*. The circuit pays surprise and,
  with the very events that constitute the payment, buys a smaller future
  surprise — but only because a separate error population renders the signed
  residual physical.
]

== The signed-residual wrinkle, and the predictive-subtraction sign

The residual $r_j = y_j - q_j in [-1, 1]$ is *signed*; spikes are not. This is
the implementation question the realization hypothesis turns on, and it has a
standard, honest answer.

#honest[
  Split the residual across *two rectified error channels*, as cortical
  predictive-coding microcircuits are independently argued to do
  @raoballard1999:
  $ r_j^+ = max(0, space y_j - q_j) quad ("error-ON: occurred more than predicted"), $
  $ r_j^- = max(0, space q_j - y_j) quad ("error-OFF: predicted but absent"), $
  with $r_j = r_j^+ - r_j^-$, so the rule @eq-rule becomes $Delta W_(i j) = eta
  c_i (r_j^+ - r_j^-)$ — a *potentiating* contact driven by the ON channel and a
  *depressing* contact driven by the OFF channel.

  *The mechanism's sign.* Predictive subtraction does *not* work by adding a
  depolarization "equal to the prediction." Cancelling the expected *excitatory*
  drive requires *inhibitory* — shunting or hyperpolarizing — feedback: the
  prediction $q_j$ is delivered as feedback inhibition (dendritic shunting) that
  cancels the expected component of the drive, so the soma integrates only the
  *unpredicted* residual. The exact additive form $"input" - "prediction"$ is an
  idealization, valid when the shunting conductance is in its linear regime and
  the prediction is delivered on the same timescale as the drive; outside that
  regime the cancellation is approximate. The cost of all this is a doubling of
  the error population (a known feature, not a bug, of two-compartment / E–I
  predictive-coding models) and the requirement that $q_j$ be available at the
  synapse as the tonic signal it already is. We flag the two channels rather than
  pretend a signed spike exists; everything below treats $r_j$ as the signed
  difference these channels jointly represent.
]

A structural dividend follows immediately, and ties the rule to the
partition-of-unity invariant carried by the softmax.

#proposition[
  *Learning lives in the difference modes; the sum mode is conserved.* For each
  context $i$,
  $ sum_j Delta W_(i j) = eta c_i (sum_j y_j - sum_j q_j) = eta c_i (1 - 1) = 0, $
  since $sum_j y_j = 1$ (one symbol occurs) and $sum_j q_j = 1$ (softmax). The
  total lateral weight out of each context — the "sum mode" — is therefore
  *exactly conserved at every step*. Learning only redistributes predictive mass
  between symbols (the difference modes); it never perturbs the softmax partition
  of unity. The validator measures the per-context row-sum drift over a full run
  at the level of machine epsilon, confirming the conservation numerically.
]

== Descent and convergence: an energy Lyapunov function

We now prove the rule does what it should — drive $q -> P$ while spending
monotonically less energy on average.

#background("Lyapunov functions, for the descent argument")[
  To certify that a dynamical system flows to a target state, exhibit a *Lyapunov
  function*: a scalar $V$ bounded below, zero only at the target, that *never
  increases* along the dynamics ($dot(V) <= 0$). The state can then only settle
  where $dot(V) = 0$. It is the continuous-time analogue of a loop variant in
  program verification — a quantity provably non-increasing until the goal.
]

The natural candidate is the *excess energy* itself — the gap between the
circuit's cross-entropy rate and the entropy-rate floor:
$ V(W) = overline(H)(p, q) - overline(H)(p) = sum_i pi_i space D_("KL")(P_(i dot)
|| q_(i dot)) >= 0, $ <eq-lyap>
which is zero iff $q_(i dot) = P_(i dot)$ for every visited context $i$. We
analyze the *averaged* (expected-over-data) learning dynamics. Following the
Lyapunov / descent convention of the coherence brief, we *define* the averaged
gradient flow as
$ dot(W) = -eta nabla V, $ <eq-flow>
so that descent is exact by construction (below), and we *separately* relate this
flow to the rule's expected step.

#theorem("Excess energy is a Lyapunov function; learning converges to the truth")[
  *Hypotheses.* (i) *Convexity scope:* the predictor is softmax over logits that
  are *linear* in $W$, for which $V$ is convex in $W$ (so there are no spurious
  local minima); (ii) *Ergodicity:* every context with $pi_i > 0$ is visited
  infinitely often (the momentum-rover chain is manifestly ergodic, so this holds).

  Under these hypotheses, along the averaged flow @eq-flow:
  + $dot(V) = nabla V dot dot(W) = -eta norm(nabla V)^2 <= 0$, with equality iff
    $q_(i dot) = P_(i dot)$ for every $i$ with $pi_i > 0$;
  + the unique stationary point is the global minimum $V = 0$, where $q = P$, so
    $V -> 0$ and $q -> P$: the circuit converges to the *true conditional law* and
    its expected energy descends monotonically to the entropy-rate floor.
  The minimizer is unique up to the softmax gauge (adding a constant to a context
  row $W_(i dot)$ leaves $q_(i dot)$ unchanged), which does not affect $q$.
]

#proof[
  *Convexity.* Each context contributes $-sum_j P_(i j) log_2 q_(i j)$, a
  cross-entropy of a softmax; the log-sum-exp term $log_2 sum_k e^(a_k)$ is convex
  in the logits $a$, the map $W -> a$ is linear, and the remaining terms are
  $W$-independent constants, so $V$ is a non-negative combination of convex
  functions of a linear image of $W$ — hence convex in $W$.

  *Gradient.* Differentiating @eq-lyap, only $-sum_j P_(i j) log_2 q_(i j)$
  depends on $W$, and $partial (log_2 q_(i k)) slash partial W_(i j) = (bb(1)[k =
  j] - q_(i j)) slash ln 2$, so
  $ (partial V) / (partial W_(i j)) = -pi_i (P_(i j) - q_(i j)) / (ln 2). $ <eq-gradV>

  *Descent.* By @eq-flow, $dot(V) = nabla V dot (-eta nabla V) = -eta
  norm(nabla V)^2 <= 0$, zero iff $nabla V = 0$, i.e. by @eq-gradV iff $q_(i j) =
  P_(i j)$ for every $i$ with $pi_i > 0$. By convexity this stationary point is
  the global minimum $V = 0$, reached as $t -> infinity$ under the ergodicity
  hypothesis.
]

#remark[
  *The averaged flow descends in the same direction the rule steps.* The
  expected single-step update of the local rule @eq-rule, using $EE[c_i] = pi_i$
  and $EE[y_j mid(|) c = e_i] = P_(i j)$, is
  $ EE[Delta W_(i j)] = eta space pi_i (P_(i j) - q_(i j)) = -eta dot ln 2 dot
  (partial V) / (partial W_(i j)), $ <eq-expstep>
  by @eq-gradV. So the rule's *expected* step is the negative $V$-gradient up to
  the positive scalar $ln 2$ (the bits$<->$nats factor): the SGD rule descends
  $V$ in expectation, while the *flow* @eq-flow we analyze for the exact identity
  $dot(V) = -eta norm(nabla V)^2$ is the cleanly-rescaled version. The validator
  measures the proportionality constant between $EE[Delta W]$ and $-nabla V$
  directly and recovers $k = 0.693 = ln 2$, confirming @eq-expstep empirically.
]

#intuition[
  The thermodynamic reading of the energy corollary — *excess energy = model
  error* — is here promoted to a *stability certificate*. The bits the circuit
  wastes are not merely a cost; they are the Lyapunov function whose dissipation
  *is* learning. "Spend less energy" and "predict the world better" are one
  monotone descent on one number.
]

== Asymptotics: decreasing step versus constant step

The averaged flow descends exactly; the *online* rule follows it only in
expectation, and its asymptotics depend entirely on the step schedule — a
distinction the numerics make concrete.

#honest[
  *Decreasing schedule $=>$ almost-sure convergence.* The validator uses the
  decreasing Robbins–Monro schedule $eta_t = eta_0 slash (1 + t slash t_0)$, which
  satisfies $sum_t eta_t = infinity$ and $sum_t eta_t^2 < infinity$
  @robbinsmonro1951; with the convex objective of the convergence theorem, SGD
  then converges *almost surely* to the minimizer $q = P$. *Constant schedule
  $=>$ an $O(eta)$ noise ball.* A constant step violates the square-summability
  condition and instead settles into a noise ball of radius $O(eta)$ around the
  optimum — the right choice for *tracking* a non-stationary source, where a
  frozen model is worse than a jittering one. The residual we report below is a
  *finite-time* quantity under the decreasing schedule, not an asymptotic floor;
  run longer and it shrinks.
]

*Numerical validation* (`learn_validate.py`). Running the local rule @eq-rule
online on the momentum-rover source ($s = 0.7$) from an ignorant uniform
predictor, under the decreasing schedule:

#figure(
  table(
    columns: (auto, auto, auto),
    align: (right, right, right),
    stroke: 0.5pt + luma(180),
    table.header[stage][energy $overline(H)(p,q)$ (bits/symbol)][excess $= overline(D)_("KL")(P || q)$ (bits/symbol)],
    [initial (uniform $q$)], [$2.0000$], [$1.0218$],
    [marginal-only model], [$1.7500$], [$0.7718$],
    [converged (learned)], [$0.9787$], [$0.0004$],
    [entropy-rate floor], [$0.9782$], [$0$],
  ),
  caption: [Online local learning on the momentum rover. Energy falls from
    $log_2 4 = 2.0$ bits (uniform), past the marginal-only baseline $H(pi) = 1.75$
    bits, to the *converged learned* value $0.9787$ bits — which sits *slightly
    above* the entropy-rate floor $overline(H)(P) = 0.9782$ bits by the
    finite-time residual $0.0004$, not at it. The gradient identity of
    Theorem (energy gradient) is confirmed to $1.12 times 10^(-10)$; the averaged
    flow is verified to descend strictly monotonically (monotone fraction $1.0$,
    maximum single-step increase $0.0$).],
) <fig-learn>

#honest[
  *Read the numbers exactly.* The entropy-rate floor is $0.9782$ bits/symbol
  *everywhere* in this paper; the converged learned energy $0.9787$ is *above* it
  by the $0.0004$ finite-learning residual, so $0.979$ must never be called "the
  floor." (The empirical perfect-predictor *stream* cost reported elsewhere,
  $0.9760$, is a finite-sample mean that may dip below the expectation $0.9782$ by
  ordinary fluctuation — not a violation of the floor.) The headline that the
  circuit learns to spend $44%$ less energy is scoped against the *marginal*
  baseline $1.75$ bits versus the conditional floor $0.978$ — i.e. it is the
  mutual information $I(x_t; x_(t-1)) = 0.7718$ bits the temporal structure was
  worth — and is a statement about *bits* (the proven latency-time observable),
  distinct from any joule-valued energy-corollary figure.
]

== Where this sits in the literature

Deriving @eq-rule from spiking energy *re-derives*, rather than invents, several
known objects, which is reassurance: the delta / LMS rule @widrowhoff1960
(the algebra of @eq-rule verbatim); Rescorla–Wagner surprise-driven conditioning
@rescorlawagner1972 (the sign pattern of $r_j$); predictive coding
@raoballard1999 (units carry prediction *errors*, synapses correlate them with
the predicting activity — our $c_i r_j$); and, as a *special, degenerate* case,
the free-energy principle @friston2010. We are careful with the last: a
point-estimate softmax predictor carries no posterior, so the variational bound
on surprise is trivially tight and the "free energy" is literally $ell_t$; this
is a delta-posterior special case of the FEP, not a deep equivalence, and we do
not oversell it. The contribution is the *grounding* — these become one
statement, *gradient descent on a spiking circuit's metabolic bookkeeping*, with
the error term identified (conditionally, per the realization hypothesis) with
the emitted code, and the descent certified by the energy itself as Lyapunov
function.

// ===== S12-identifications =====
= What the rule already is, and what is new

The local rule derived above —
$ Delta W_(i j) = eta dot underbracket(c_i, "pre") dot underbracket((y_j - q_j), "post residual") $ <eq-id-rule>
is not invented here. Deriving it from spiking energy *re-derives* four objects already in the literature. That is reassurance, not coincidence: a rule that fell out of the spiking-energy gradient and matched nothing known would be a reason for suspicion. We list the four honestly — three are genuine identities, the fourth (the free-energy principle) is a degenerate special case we decline to oversell — and then state precisely the three things this paper *does* contribute on top of them.

== Four identifications

#background("the four objects @eq-id-rule re-derives")[
  *(a) The delta / LMS rule* @widrowhoff1960. The oldest of the four: adjust a weight by *presynaptic activity times output error*, $Delta W prop x^("pre") (text("target") - text("output"))$. With presynaptic factor $c_i$ and error $y_j - q_j$, @eq-id-rule is this rule verbatim. The softmax–cross-entropy gradient *is* the delta rule; we claim no new algebra here (the novelty is the physical reading of the error term, below).

  *(b) Rescorla–Wagner conditioning* @rescorlawagner1972. Associative learning in animals is driven by the *surprise* of the outcome, $Delta V prop (lambda - sum V)$ — a predictor strengthens in proportion to how much the outcome *exceeded* what was already predicted, and predictors of absent outcomes weaken. That asymmetric strengthen-what-happened / weaken-what-didn't pattern is exactly the sign structure of the residual $r_j = y_j - q_j$: positive where the symbol occurred against a low prediction, negative where a prediction went unmet.

  *(c) Predictive coding* @raoballard1999. A hierarchy whose units carry *prediction errors* and whose synapses learn by *correlating those errors with the activity that predicted them*. The product $c_i dot r_j$ in @eq-id-rule is precisely an error–predictor correlation, and the two rectified error channels $r_j^+, r_j^-$ of the previous section are the error-coding units of a predictive-coding microcircuit. This is the closest structural match: predictive coding supplies the architecture (predictive subtraction, error units) that makes the physical reading of $r_j$ legitimate.

  *(d) The free-energy principle* @friston2010. Perception and learning both descend *variational free energy*, an upper bound on surprise. Our per-symbol cost $ell_t = -log_2 q(x_t | c_t)$ is the surprise, and learning descends it — so the rule *is* a free-energy descent. But see the honesty check below before reading this as a deep equivalence.
]

#honest[
  *The free-energy identification is a degenerate special case, not a deep equivalence.* Variational free energy is interesting precisely when the learner maintains a *posterior distribution* over hidden states and the bound $F = text("surprise") + D_("KL")(text("approx posterior") || text("true posterior"))$ is *slack* — the slack being the work the variational machinery does. Our predictor emits a *point estimate*: $q(dot | c_t) = text("softmax")(W c_t)$ is a single distribution over the *next observable symbol*, with no latent variable and hence no posterior to approximate. With a delta (point-mass) posterior the variational KL term is identically zero, so the bound is *trivially* tight and free energy collapses to bare surprise $ell_t$. Calling @eq-id-rule "free-energy descent" is therefore true but vacuous: it descends surprise, which any sensible learner does. We list (d) to place the work on the map, not to claim the free-energy principle as a result. The substantive identifications are (a)–(c).
]

== The thesis this instantiates: compression $=$ prediction $=$ intelligence

The four identifications sit inside a much older and broader thesis, which we state plainly so our contribution is not mistaken for it.

#background("compression is prediction is modeling — the Shannon–MDL line")[
  Shannon's source coding theorem @shannon1948 makes the equation *optimal compression $=$ knowing the source distribution* exact: the shortest expected code length for a source equals its entropy, and is achieved only by a code matched to the true law $p$ (assign length $-log_2 p(x)$ to symbol $x$). A *model* $q$ that differs from $p$ pays exactly the excess $overline(D)_("KL")(p || q)$ bits — surprise is the bill for a wrong model. *Minimum description length* @rissanen1978 turns this into an inductive principle: the best model of data is the one that *compresses it most*, counting the bits to describe the model itself; learning *is* compression. The *bits-back argument* @hintonvancamp1993 closes the loop for parameterized models, showing that the description length of a neural network's weights is itself a codeable, optimizable quantity — so "compress the data" and "learn good weights" are one objective. In this line, an agent that predicts its inputs well is, by definition, one that compresses them well; intelligence-as-prediction and intelligence-as-compression are the same thesis stated in two vocabularies.
]

None of this is ours. The thesis "compression $=$ prediction $=$ modeling" is Shannon–MDL–bits-back; the conditioning, predictive-coding, and delta-rule readings are (a)–(c). *What is new here is the spiking realization* — that this entire stack collapses onto the physics of a spiking circuit. The code length $-log_2 q$ is not an abstract bit count but a *spike latency* $t^*(q) = -lambda log_2 q$ (R1, proven exact via one time constant); the gradient that improves the model is not backpropagated from an external loss but is *the emitted error spike itself*, read again; and the descent of the loss is certified by the circuit's own metabolic energy acting as a Lyapunov function. The thesis was a statement about *quantities*; the contribution is a statement about *spikes*.

== Three-factor framing

@eq-id-rule is, in the taxonomy of biological plasticity, a *three-factor rule* @fremauxgerstner2016: a Hebbian pre$times$post product gated by a third, global signal,
$ Delta W_(i j) = underbracket(eta, "factor 3:" #h(2pt) "gate") dot underbracket(c_i, "factor 1:" #h(2pt) "pre") dot underbracket((y_j - q_j), "factor 2:" #h(2pt) "post"). $ <eq-id-three-factor>
The first factor is presynaptic context activity $c_i$; the second is the postsynaptic residual $y_j - q_j$ carried by the local error population; the third is the global rate / neuromodulatory gate $eta$ (which, recall, schedules as $eta_t = eta_0 slash (1 + t slash t_0)$ for almost-sure convergence). This places the rule in the class that contemporary plasticity theory regards as both biologically plausible and neuromorphically implementable: every factor is available *at the synapse* or as a *single broadcast scalar*, with no global error vector to route.

== The three genuine contributions

Stripping away what is inherited, exactly three things are new:

#block(inset: (left: 6pt))[
  *(C1) Exact latency$arrow.l.r$surprisal calibration through a single time constant.* Driving readout $j$ with $R I_j = theta slash (1 - q_j^alpha)$ makes its first-spike latency $t^*(q_j) = -lambda log_2 q_j$ *exactly* the surprisal (R1; @thorpe1996 @thorpe2001 supply time-to-first-spike coding as the substrate idea, but the exact $lambda$-calibration is ours and proven, time-valued, not energy-valued).

  *(C2) The conditional cost-spike $=$ gradient identity.* The very error spike the circuit emits as a symbol's *code* is, correlated with the active context, the complete first-order gradient that lowers that code's future cost. The gradient *algebra* is the delta rule (inherited); the *physical* claim — that the circuit emits exactly $y_j - q_j$ — is an explicit realization hypothesis imported from predictive subtraction @raoballard1999 and carried by the two-channel error population, and these are *not* the same spikes as the latency code of (C1). We foreground this as a *conditional* identity, which is precisely why it is the paper's novelty rather than a restatement of @widrowhoff1960.

  *(C3) A safety / optimality verification split.* The structural invariants (locality, sum-mode partition-of-unity preservation, weight boundedness) are model-checkable safety properties holding at *every* learning step against *any* data, while the optimality claim (descent to within $epsilon$ of the entropy-rate floor) is an *analytic* Lyapunov fact about the learning flow. Separating what a model checker certifies outright from what the Lyapunov analysis supplies is the verification contribution.
]

#intuition[
  The honest summary: the *machinery* is old — delta rule, surprise-driven conditioning, predictive coding, and the Shannon–MDL thesis that prediction is compression is modeling. What this paper adds is that the machinery *runs on spikes with no slack*: the bit is a latency, the gradient is the cost spike, and the loss-descent certificate is the energy bill. The free-energy principle is on the map but contributes nothing here beyond a vocabulary, because a point-estimate predictor has no posterior to be variational about.
]

// ===== S13-verification =====
= Verification obligations <sec-obligations>

The payoff of building the coder and its learning rule from first principles is that every claim we have made is a *checkable property* — and the properties sort cleanly into the two classes the mode decomposition already separated: *structural* facts that hold no matter how good or bad the predictor is, and *quantitative* facts that hold only to the degree the predictor is good. This is the formal-methods contribution: correctness is certified once and for all, while optimality is the quantity learning drives down, and the two never trade off.

We write the obligations in the property syntax of a probabilistic model checker such as PRISM, whose native object is exactly our setting: a discrete-time Markov chain (the source) coupled to a deterministic circuit (the coder). We use $P_(>=1)[phi.alt]$ for "the path formula $phi.alt$ holds almost surely", the temporal operators $G$ ("globally / at every step"), $F$ ("eventually"), $X$ ("at the next step"), and $U$ ("until"), and $R{"r"}_(=?)[dot]$ for "query the expected value of reward $"r"$". The templates referenced below live in `spiking_entropy_coder.pctl`.

#background("safety, liveness, and what a model checker actually does")[
  A *safety* property says "nothing bad ever happens" — formally $G(text("good"))$, an invariant that must hold at every reachable state. A *liveness* property says "something good eventually happens" — formally $F(text("good"))$, or the recurrent $G F(text("good"))$ ("infinitely often"). A *probabilistic model checker* takes a finite Markov model and a property and, by exhaustively analysing the reachable state space, returns either a proof that the property holds (with probability $1$, or with a computed probability) or a counterexample path. Two consequences matter here. First, it can only certify properties of a *finite* state space — so a quantity defined as a limit ($t arrow infinity$) is outside its reach and must be discharged analytically. Second, a *reward* (a number accumulated along paths) lets it answer quantitative questions — expected cost per step, expected cost to absorption — which is how we will read the coder's bit rate.
]

Before instantiating, we pin the *windowed semantics* the properties quantify over, so the temporal logic is concrete rather than aspirational. Time is partitioned into *symbol windows*: a window opens when the readout layer is reset and begins integrating, and closes when the decode rule fires. Within a window we use the labels of `spiking_entropy_coder.pctl`: $"win"_i$ is true once readout $i$ has crossed threshold this window; $"winner" = sum_i "win"_i$ counts how many readouts have fired; $"decoded"$ is the symbol id the decoder commits to; $"emitted"$ is the true source symbol $x_t$ driving that window; and $"new_window"$ marks the reset that opens the next window. With these in hand the obligations are instantiable against a concrete model — a DTMC over the momentum-rover source coupled to a four-neuron temporal-WTA readout, in the style of the program's `contra_inhib_fixed.prism`.

== Structural / safety: hold for any predictor

These are the *predictor-free* guarantees. They depend only on the *decode rule* of stage 3 — the winner-take-all settling and the layer reset — and are completely indifferent to the numerical quality of $q$. A bad model makes the code *slow* (high cross-entropy, many joules), but these properties guarantee it is *never wrong*.

A subtlety must be stated honestly, because the naive form of the first obligation is *false*. One is tempted to assert "at most one readout is ever active", $P_(>=1)[ G ("winner" <= 1)]$. But during the integration *transient* — before any readout has settled — several readouts are simultaneously *sub-threshold-active*, racing toward threshold; mutual exclusion is exactly what the competition is still in the process of establishing, not an invariant that holds throughout. The correct statement is a *settled* one, in the eventual-stabilisation form $F G$ that the contralateral-inhibition archetype actually certifies.

#definition[
  A symbol window has a *settled winner* if it reaches a state in which exactly one readout is above threshold and the lateral inhibition holds the others below it until the window closes. Write $"settled1" equiv ("winner" = 1 and "winner" text(" stable until ") "new_window")$.
]

#proposition[
  *Decode safety, in settled form.* For any predictor $q$, the coder satisfies, per window:
  - *Eventual unique winner (liveness + settled mutual exclusion).*
  $ P_(>=1) [ space G space F space "settled1" space ]. $ <eq-settled>
  - *Losslessness (the safety core).* Whenever a winner has settled, the decoded symbol equals the emitted one:
  $ P_(>=1) [ space G space ("settled1" arrow.r.double "decoded" = "emitted") space ]. $ <eq-lossless>
  - *Prefix-freeness (self-clocking).* A settled winner resets the layer before the next window's integration begins, so no two windows' spikes interleave:
  $ P_(>=1) [ space G space ("settled1" arrow.r.double X ("winner" = 0 space U space "new_window")) space ]. $ <eq-prefix>
]

#remark[
  The transient is *quantified*, not waved away: between window open and $"settled1"$ there is a bounded settling interval during which $"winner"$ may exceed $1$, and the first-spike-takes-all decode rule is precisely what tolerates it — the *first* readout to cross threshold names the symbol and triggers the reset, so the later sub-threshold activity of the losers is discarded by the reset rather than needing to have been suppressed in advance. The closed-form settling-time bound for this transient is supplied by the sum/difference-mode analysis of the WTA archetype (the program's `closed_form_wta_multi` study); we import it rather than re-derive it. This is the corrected reading of "a bad model is slow, never wrong": *slowness* is a long latency in stage 2 (a quantitative, predictor-dependent fact, below); *not-wrong* is @eq-lossless, which rests on the decode rule alone and never on a global mutual-exclusion invariant that does not hold.
]

These properties are pure reachability over a finite state space, so a model checker discharges them outright, once, against the archetype — and the certificate carries over to *every* predictor the learning rule will ever produce.

== Structural / safety of the weight dynamics

Learning adds a third structural axis the static coder did not have: properties of the *weight trajectory* itself, which must hold at *every* learning step, on *any* data. Two are immediate. *Locality* is a guard-level property of the update @eq-rule: $Delta W_(i j)$ reads only the presynaptic context $c_i$, the postsynaptic residual $r_j$, and the global gate $eta$ — no non-local term appears, checkable by inspection of the update expression. *Sum-mode invariance* we proved exactly: $sum_j Delta W_(i j) = 0$ at every step, so the partition-of-unity the softmax enforces is preserved throughout learning.

The third — *that the weights stay bounded and do not run away* — needs care, and the naive form is again unsound. One is tempted to assert $P_(>=1)[G(|W_(i j)| <= W_max)]$, a global-trajectory claim. But this is backed only by a fixed-point bound (the averaged optimum is $W_(i j) = log P_(i j) + "const"_i$, finite for an ergodic chain), and a fixed-point bound says nothing about the *transient* trajectory; worse, the absolute weight $W_(i j)$ is not even well-defined, because the softmax has a *gauge freedom* — adding any constant to a whole context row $W_(i dot)$ leaves $q_(i dot)$, and hence the loss and the dynamics, unchanged (Theorem on convergence, the gauge remark). The unconstrained weight can therefore drift along the gauge direction without bound while the predictor is perfectly stationary. The fix (fix I / FM-04) is to *gauge-fix*: state boundedness on the *centered* weight, the only gauge-invariant quantity, and give an *inductive* transient argument rather than a fixed-point assertion.

#definition[
  The *centered weight* is $tilde(W)_(i j) = W_(i j) - (1 slash n) sum_k W_(i k)$, the deviation of $W_(i j)$ from its context-row mean (with $n = |cal(X)|$). It is invariant under the softmax gauge (a row shift changes every $W_(i k)$ by the same constant, leaving $tilde(W)_(i j)$ fixed), and it is the *only* part of $W$ that affects $q$. Equivalently one may pin the gauge directly by a row-sum leak $sum_j W_(i j) = 0$, in which case $tilde(W) = W$.
]

#proposition[
  *Gauge-fixed weight boundedness, by induction on the transient.* Run the bounded rule @eq-rule with a step schedule $eta_t$ summable in square ($sum_t eta_t^2 < infinity$, e.g. the decreasing $eta_t = eta_0 slash (1 + t slash t_0)$ of @eq-rule's validator). Then there is a finite $W_max$ with
  $ P_(>=1) [ space G space (|tilde(W)_(i j)| <= W_max) space ] quad forall i, j. $ <eq-bound>
]

#proof[
  Work in the gauge-fixed coordinates $tilde(W)$, where the dynamics are the same softmax–cross-entropy descent (the gauge direction is quotiented out). Per step, the centered update is $Delta tilde(W)_(i j) = eta_t c_i (r_j - (1 slash n) sum_k r_k)$ with $c_i in {0, 1}$ (one-hot context) and each rectified residual component in $[-1, 1]$, so $|Delta tilde(W)_(i j)| <= 2 eta_t$ — each step moves a bounded amount. Induct on the window index. Inside the convex basin (Theorem: $V$ is convex in $W$, gauge-fixed) the averaged drift $EE[Delta tilde(W)_(i j)] = -eta_t dot ln 2 dot nabla_(tilde(W)) V$ points *toward* the finite minimiser $tilde(W)^* = (log P_(i dot))$ centered, so once $|tilde(W)_(i j) - tilde(W)_(i j)^*| <= rho$ the expected step is inward and the per-step excursion is at most $2 eta_t arrow 0$; the supermartingale convergence behind the Robbins–Monro theorem (@eq-rule's schedule, see below) then keeps the trajectory almost surely within a bounded neighbourhood $W_max = |tilde(W)^*|_infinity + rho + 2 eta_0$. The bound is on the transient, established inductively, not merely on the fixed point.
]

#honest[
  The boundedness certificate is genuinely an *inductive transient* argument and is sound only inside the convex basin with a square-summable step; an adversarial constant-$eta$ schedule on a non-ergodic chain (some context never visited, so its row is never corrected) can let an unvisited row drift in *its own* gauge-fixed directions. We state the hypotheses — ergodic chain (every context with $pi_i > 0$ visited), gauge-fixed coordinates, summable-square step — inside the obligation rather than in a footnote, because they are exactly the conditions under which a model checker's finite invariant @eq-bound is faithful to the real dynamics.
]

== Quantitative / optimality: predictor-dependent

These are the properties learning *improves* — the bit rate, the gap above the floor, and the descent toward it. They are the reason the circuit is "intelligent" in a checkable sense, and they are predictor-dependent by construction.

A *reward* makes the bit rate a model-checker query. Attach to each window the per-symbol latency reward $"time" = -lambda log_2 q(x_t mid(|) c_t)$ (the surprisal the calibrated readout spends, Theorem on spike-time = cross-entropy). Then the steady-state expected reward is the cross-entropy rate, and the optimality target is that it meet the entropy-rate floor within a tolerance $epsilon$:
$ R{"time"}_(=?) [ space S space ] <= lambda (overline(H)(p) + epsilon) = lambda (0.9782 + epsilon) "(rover, " s = 0.7 "),"$ <eq-cost>
where $overline(H)(p) = 0.9782$ bits/symbol is the closed-form floor of the momentum rover. The *stupidity tax* is the same query made differential: run @eq-cost for the learned predictor and for the memoryless baseline $q = pi$; the difference is $lambda overline(D)_("KL")(p || pi) = 0.7718$ bits/symbol — the bits the memory recovers, made checkable as the gap between two reward queries. And a learning step counts as *an improvement* exactly when it lowers $R{"time"}_(=?)[S]$ — the verifiable form of "the predictor is getting smarter".

The two limit properties — *that the cost descends* and *that it converges* — are the Lyapunov payoff, and here we must be scrupulous about *which tool certifies what*, because these are *not* finite-state model-checking facts.

#honest[
  *Convergence and descent are limit properties, not reachability queries (fix FM-05).* "The energy descends monotonically" and "the cost converges to within $epsilon$ of the floor" are *analytic* statements about the continuous learning *flow* ($t arrow infinity$), discharged by the Lyapunov theorem (excess energy is a Lyapunov function; $dot(V) = -eta norm(nabla V)^2 <= 0$ under the averaged flow $dot(W) = -eta nabla V$), *not* by exhausting a finite state space. The clean division of labour is: a model checker certifies (i) the *structural* invariants @eq-settled–@eq-bound, and (ii) the *converged* circuit's cost bound @eq-cost against a *frozen* weight $W$ — both finite-state queries; the *Lyapunov* theorem certifies that learning *reaches* that frozen $W$. Writing $F G (R{"time"} <= lambda(overline(H)(p) + epsilon))$ as a single model-checking obligation over the *joint* (source $times$ weights) chain would require the weight space to be finite, which it is not; the honest statement is the two-tool split, not one query that secretly assumes a discretised weight grid.

  Two further subtleties pin the numbers. *(a) Descent is a property of the averaged flow, not of every stochastic step.* With a *decreasing* Robbins–Monro schedule $eta_t = eta_0 slash (1 + t slash t_0)$ the online rule converges almost surely (the $0.0004$-bit residual at $2 times 10^6$ symbols is *finite-time*, not an asymptotic floor); with a *constant* step it settles instead into an $O(eta)$ noise ball. The validator uses the decreasing schedule, so its endpoint is convergence, not a noise ball. *(b) The converged learned energy is $0.9787$ bits/symbol — strictly above the floor $0.9782$* (the finite-learning residual). The floor is $0.9782$ everywhere; do not conflate the learned endpoint with it.
]

#honest[
  *Continuous-time exactness vs the discrete spike-count proxy (fix FM-06).* The reward $"time" = -lambda log_2 q$ is a *real-valued, continuous-time* quantity: it is exact only in a *timed* model — a continuous-time Markov chain or a probabilistic timed automaton, where window latency is a genuine clock value. In a pure *discrete*-time chain (which is what `contra_inhib_fixed.prism` is, and what the cheapest model-checking pass uses) one must *bin* time, and the reward degrades to a spike-*count* proxy for $-log_2 q$, reintroducing exactly the timing-resolution penalty of the latency-code honesty check: the recoverable precision is about $log_2(t^* slash delta t)$ bits, so the continuous-time "no integer penalty" advantage is *traded* for an analog binning penalty, small when the clock $delta t$ is fast relative to the firing rate. The count model is what the existing PRISM file supports; the *exact* latency statement @eq-cost wants the timed extension (CTMC / PTA). We state both targets so the gap between the proven continuous-time identity and the discretised proxy is visible, not papered over.
]

#intuition[
  The whole verification story is one table. *Structural* obligations (@eq-settled–@eq-bound) are finite-state, predictor-free, and discharged once by a model checker — they are the "never wrong" half. *Quantitative* obligations are the bit rate (@eq-cost, a reward query against a frozen model) and the descent/convergence to it (the Lyapunov half, discharged analytically) — they are the "as good as it has learned to be" half. The split is the formal-methods payoff: the safety certificate is *invariant under learning*, so retraining the predictor never reopens the correctness question; only the optimality query has to be re-run, and watching it fall is watching the circuit get smarter.
]

// ===== S14-critical-eval =====
= Critical evaluation: a referee report folded in <sec-limitations>

A paper that claims an *exact* correspondence between spike timing and information
should be the first to say where the correspondence is exact, where it is only
conditional, and where it is frankly open. We therefore subjected the two source
notes to an adversarial audit through five lenses — proof rigor, information and
coding theory, dimensional/algebraic consistency, learning-theory, and formal
verifiability, plus a neuroscience-plausibility pass. The audit returned $50$
confirmed findings: $13$ *major*, $36$ *minor*, $1$ *cosmetic* (zero blockers).
This section is the referee report folded back into the paper. We report (1) every
major finding and how the present text repairs it, with a forward reference to the
section that carries the fix; (2) issues we leave honestly *open*; (3) findings the
audit *raised and we reject*, with the reason; and (4) the reproduced-numerics
table, every entry of which an independent validator confirmed, including two new
machine checks. Throughout, "fixed in-text" means the defect no longer appears in
the merged paper; it does not mean the underlying difficulty has vanished.

#intuition[
  The single most useful output of the audit was a *coherence win*: the two notes
  complete each other. Note I left the predictor map $c_t arrow.r q(dot bar.v c_t)$
  and its normalization unspecified — its largest gap. Note II supplies it exactly:
  $q_j = "softmax"_j(W c_t)$, with logits equal to the lateral weights. Stating this
  once closes the gap, and pins the partition-of-unity invariant $sum_j q_j = 1$ on
  the *softmax* (exact) rather than on divisive normalization (only its $sigma arrow.r 0$
  approximation). See the frozen coherence brief, reconciliations (i)–(ii).
]

== Confirmed major findings and their in-text repairs

The thirteen major findings and their repairs are below — each applied in the
cross-referenced section; we state only what was wrong and what was done.

#repair("Lyapunov descent constant", [PR-01, IT-01, F1, BIO-10, FM-08, L1, L2])[The averaged-flow descent identity was printed as $dot(V) = -eta norm(nabla V)^2$, but the flow was *also* defined as $dot(W)_(i j) = pi_i (P_(i j) - q_(i j))$, which equals $-(ln 2) nabla V$, not $-eta nabla V$. Two incompatible definitions; wrong constant (the true constant is $ln 2 approx 0.693$, not the learning rate). Convexity scope and ergodicity hypothesis were buried in limitations.][Adopt one convention (fix B): define the gradient flow as $dot(W) = -eta nabla V$, giving $dot(V) = -eta norm(nabla V)^2 lt.eq 0$ *exactly*. Separately relate the rule's expected step $EE[Delta W_(i j)] = eta pi_i (P_(i j) - q_(i j)) = -eta (ln 2) nabla V_(i j)$ (the $ln 2$ is the bits$arrow.l.r$nats factor). Convexity (softmax-over-linear logits) and the ergodicity hypothesis ($pi_i > 0 arrow.r.double$ context visited) now sit *inside* the convergence theorem. See the convergence section.]

#repair("Cost-spike = teaching signal (made conditional)", [PR-05, L8, NOV-05])["The cost spike *is* the teaching signal" was stated as an unqualified identity. The proved part is only the gradient algebra; the load-bearing physical step — that the circuit *emits* exactly $y_j - q_j$ — was silently imported from the companion architecture, and then partly contradicted by the rectified-channel honesty block one section later.][Split the claim (fix C). (a) Prove only the algebra unconditionally: $-partial ell_t slash partial W_(i j) = c_i (y_j - q_j)$, standard softmax–cross-entropy, attributed to the delta / Widrow–Hoff rule @widrowhoff1960. (b) State the physical step as an explicit *hypothesis/realization*: the signed residual is carried by two rectified ON/OFF channels $r_j^+ = max(0, y_j - q_j)$, $r_j^- = max(0, q_j - y_j)$, in the predictive-coding sense @raoballard1999. (c) Say plainly these are *not* the same spikes as the latency code. This conditional identity is foregrounded as the paper's genuine novelty. See the learning-rule section.]

#repair("Spike-time vs. metabolic energy", [F3, BIO-13])[Spike-*time* ($lambda$, s/bit, proven) and metabolic *energy* ($kappa$, J/bit) were used interchangeably as "one spike cost," and old Theorem 6 asserted "excess energy $=$ KL" as an identity. The bits$arrow.r$joules bridge was never modeled; per-spike cost dominates metabolism, not per-unit latency.][Keep the exact *time* results (Theorems 1, 3): $t^* = -lambda log_2 q$ is exact and time-valued. Recast the energy claim as a *model-dependent corollary* (fix D) under an explicit minimal biophysical model (per-spike ATP cost $+$ integrated subthreshold drive power $times$ latency), stating the regime — e.g. a spike-*count* code — where total energy $prop$ surprisal, and flagging where it fails (a long-latency rare symbol may draw less instantaneous power yet accumulate more leak). Never "spike-time $=$ energy." See the energy-corollary section.]

#repair("Calibration-drive ceiling", [BIO-01])[The calibration drive $R I(q) = theta slash (1 - q^alpha)$ was presented as physiologically realizable across the whole working range; it in fact diverges well inside it.][Quantify the ceiling (fix E). With $tau = lambda = 1$ ($alpha = 1 slash ln 2 approx 1.4427$), $R I slash theta approx 7.1 times$ at $q = 0.9$ and $approx 70 times$ at $q = 0.99$. A few-fold rheobase ceiling pins $q_max approx 0.85$–$0.95$, with a floor cost on near-certain symbols; $q arrow.r 0$ is the noise-dominated regime where the timing code is least reliable. The $R I slash theta$-vs-$q$ table appears in the calibration section; this sharpens, not retracts, the honesty block.]

#repair("Predictive-subtraction sign", [BIO-04])[Predictive subtraction was described as "subthreshold depolarization equal to the expected input" — the wrong sign: cancelling expected *excitatory* drive requires *inhibitory* feedback.][Restate as feedback inhibition / dendritic shunting (fix F): the soma integrates only the unpredicted residual; exact additive $"input" - "prediction"$ is an idealization, with the regime in which it holds stated. See the predictive-subtraction section.]

#repair("Attractor memory capacity", [BIO-06])["Arbitrarily deep history" was attributed to a line/ring attractor — biophysically and information-theoretically incoherent, and in tension with the paper's own capacity limitation.][A line/ring attractor @seung1996 @benyishai1995 stores *one* graded analog value: finite, noise-limited ($tilde.op log_2 "SNR"$ bits). Replace with "a finite, noise-limited number of distinguishable context states, capturing memory order up to that capacity" (fix G). See the attractor-memory section.]

#repair("WTA safety in settled form", [FM-01])["A bad model is slow, never wrong" was discharged via a global mutual-exclusion invariant $PP_(gt.eq 1)[G("winner" lt.eq 1)]$, which is *false* during the integration transient (several readouts are transiently sub-threshold-active).][State the obligation in *settled* form (fix H): eventually exactly one settled winner per window, $F G$, with a quantified transient that first-spike-takes-all tolerates. The labels (#raw("winner, decoded, emitted, new_window", lang: none)) and windowed semantics are defined concretely so the PCTL is instantiable. "Slow, never wrong" rests on the *decode* rule, not on global mutual exclusion. See the verification section.]

#repair("Weight boundedness, gauge-fixed", [FM-04])[The weight-boundedness obligation $PP_(gt.eq 1)[G(abs(W_(i j)) lt.eq W_max)]$ is a global-trajectory claim backed only by a fixed-point bound, and is false along the softmax gauge direction.][Gauge-fix it (fix I): state boundedness on the centered weight $abs(W_(i j) - (1 slash n) sum_k W_(i k)) lt.eq W_max$, equivalently pin the row sum $sum_j W_(i j) = 0$ via the sum-mode conservation $sum_j Delta W_(i j) = 0$. Give the inductive transient argument (the trajectory stays in the compact gauge-fixed sublevel set $\{V lt.eq V(W(0))\}$), not just the fixed point. See the verification section.]

#repair("Channel-coding necessity overclaim", [NOV-07])["On a noisy channel recurrence is *provably necessary* (loopy BP has no finite feedforward equivalent)" — uncited and false as stated: any fixed-iteration loopy BP unrolls to a fixed-depth feedforward net.][Soften to a conjecture (fix J): exact loopy-BP fixed points are not reproduced by any *fixed-depth* feedforward unrolling, though truncated BP *can* be unrolled @yedidia2005 @gallager1962; future-work framing only. Never "provably necessary." See the outlook section.]

#repair("Missing citations", [NOV-01])[Neither source note cited anything.][Add the bibliography and cite every named result at first mention (fix K): entropy / source coding / entropy rate @shannon1948; Kraft–McMillan @kraft1949 @mcmillan1956; delta rule @widrowhoff1960; prediction-error learning @rescorlawagner1972; predictive coding @raoballard1999; FEP @friston2010; divisive normalization @carandiniheeger2012; TTFS @thorpe1996 @thorpe2001; arithmetic coding @rissanenlangdon1979; MDL @rissanen1978; bits-back @hintonvancamp1993; three-factor plasticity @fremauxgerstner2016; attractors @seung1996 @benyishai1995; stochastic approximation @robbinsmonro1951.]

The $36$ minor and $1$ cosmetic findings are applied in the same sections under
fixes A1–A3, L, M, N, O, P (softmax-as-normalizer; matched-structure cross-entropy
rate; derived timing-resolution penalty; degenerate-FEP hedge; consistent floor
numerics; Robbins–Monro vs noise ball; proof-existence flags; precise novelty
positioning). They are not individually tabulated here; each cites its contributing
finding ids at the point of use.

== Issues flagged open

Four difficulties survive the repairs as genuine open problems. We state them as
such rather than paper over them.

#openproblem[
  *The cost-spike$=$gradient identity is conditional on an emission premise.* The
  algebra $-partial ell_t slash partial W_(i j) = c_i(y_j - q_j)$ is unconditional
  (fix C/PR-05). The *physical* identity — that the circuit emits exactly the signed
  residual $y_j - q_j$, realized by the two rectified ON/OFF channels — is a
  hypothesis imported from the predictive-subtraction architecture @raoballard1999,
  not a theorem of this paper. The paper's central novelty therefore stands or falls
  with that premise; we have foregrounded it as conditional, but a circuit-level
  derivation (or refutation) is open.
]

#openproblem[
  *The energy reading is model-dependent.* The time identity $t^* = -lambda log_2 q$
  is exact (fix D/F3/BIO-13). Its translation into metabolic energy holds only under
  an explicit substrate model; the proportionality $"energy" prop "surprisal"$ is a
  property of, e.g., a spike-count code, and fails for a long-latency rare symbol that
  draws low instantaneous power yet accumulates more leak. All "$44%$ / stupidity-tax"
  energy statements are scoped to that corollary's model. A substrate-independent
  energy law is not claimed and remains open.
]

#openproblem[
  *Attractor capacity versus certified memory order.* The line/ring attractor stores
  a finite, noise-limited number of context states ($tilde.op log_2 "SNR"$ bits;
  fix G/BIO-06). Whether that capacity suffices for the memory order a given source
  demands — and how the certified convergence guarantees degrade as the context
  representation saturates — is unquantified. The capacity *bound* is honest; the
  matching *requirement* is open.
]

#openproblem[
  *Timed-model verification gap.* The PCTL obligations are stated and instantiated in
  the settled/gauge-fixed form (fixes H, I), but the quantitative-optimality
  obligation $PP_(gt.eq 1)[F G(R\{"time"\} lt.eq lambda(overline(H) + epsilon))]$ is
  *not* a finite-state model-checking property (FM-05): it certifies behavior up to an
  $O(eta)$ residual on a continuous weight space, and the transient WTA dynamics are
  timed, not discrete. Discharging it requires a timed/hybrid model and a tolerance
  band $epsilon$ above the entropy-rate floor; we provide the obligation, not yet the
  decision procedure.
]

== Findings considered and rejected

Adversarial verification proposed several stronger framings that we examined and
*declined* to adopt, with reasons. Recording them is part of honesty.

#honest[
  - *Strong ergodicity-as-a-major framing* (PR-02, F7, FM-07, L3, PR-08): proposed
    that Theorem 2 (soundness) and the convergence theorem fail without an ergodicity
    hypothesis stated as a major gap. *Rejected/downgraded.* The momentum-rover chain
    is manifestly ergodic for $s in [0,1)$, and Theorem 2 was independently reconciled
    to the valid absorbing-silence form. The finding survives only as the minor "state
    the hypothesis" note (A2/L2), now placed inside the theorem.

  - *FEP over-statement as deep equivalence* (IT-07, L9, BIO-12, NOV-03): the audit
    flagged "the variational bound is tight" as an over-claim. We *accept the hedge*
    but *reject* the framing that the identification is wrong: for a point-estimate
    softmax there is no posterior, so the bound is *trivially* tight (fix L). We present
    it as a special (delta-posterior) case of FEP @friston2010, neither deleting it nor
    overselling it.

  - *Novelty "it's just the delta rule"* (NOV-04): the gradient identity is correctly
    attributed to Widrow–Hoff @widrowhoff1960. *Rejected as a defect* — the algebra was
    never claimed as novel; the novelty is the *interpretation* (the cost spike as the
    teaching signal), which we foreground as conditional (fix C/P).

  - *Biophysical "vacuous / breaks-exactness" framings* (BIO-03, BIO-05, BIO-08,
    BIO-09, FM-03, IT-02, IT-06): proposed that several biophysical mappings are
    vacuous. *Rejected as over-stated* — the existing honesty blocks already hedge them;
    we keep those blocks, sharpened per fixes E (calibration ceiling) and F (inhibitory
    sign), rather than retract the mappings.
]

== Reproduced numerics

Every quantitative claim was re-derived by an independent validator. All entries
below match the paper to the stated precision. The final two rows are *new* machine
checks added during the audit (#raw("critique_checks.py", lang: none)).

#table(
  columns: (auto, auto, 1fr),
  align: (left, right, left),
  stroke: 0.5pt + luma(180),
  table.header[*Quantity*][*Value*][*Note*],
  [Source entropy $H(pi)$], [$1.7500$], [marginal, bits/symbol],
  [Entropy rate $overline(H)(P)$ at $s = 0.7$], [$0.9782$], [conditional *floor*, bits/symbol],
  [Mutual information $I(x_t; x_(t-1))$], [$0.7718$], [$= H(pi) - overline(H)(P)$],
  [LIF latency calibration error], [$1.56 times 10^(-5)$], [vs $t^* = -lambda log_2 q$],
  [Stream cost: perfect predictor], [$0.9760$], [sample mean; theory $0.9782$ (fix M)],
  [Stream cost: memoryless], [$1.7477$], [theory $1.7500$],
  [Stream cost: wrong-momentum], [$1.1114$], [theory $1.1133$],
  [Gradient-identity residual], [$1.12 times 10^(-10)$], [$-partial ell slash partial W$ vs $c_i(y_j - q_j)$],
  [Final learned energy], [$0.9787$], [*above* floor $0.9782$ — finite-learning residual],
  [Final average KL], [$0.0004$], [finite-time, not asymptotic floor (fix N)],
  [Stupidity tax: memoryless], [$0.7718$], [energy-corollary model only],
  [Stupidity tax: wrong-momentum], [$0.1351$], [energy-corollary model only],
  [Stationarity drift $norm(pi P - pi)$], [$2.78 times 10^(-17)$], [machine precision],
  [Lyapunov monotone fraction], [$1.0$], [max increase $= 0.0$],
  [*New:* sum-mode row-sum drift], [$approx$ machine-$epsilon$], [$sum_j Delta W_(i j) = 0$ preserved; fix I],
  [*New:* Lyapunov constant $k$], [$0.693 = ln 2$], [unscaled flow gives $dot(V) = -(ln 2) norm(nabla V)^2$; main text adopts $dot(W) = -eta nabla V$ so $dot(V) = -eta norm(nabla V)^2$ (fix B)],
)

#remark[
  Three reading rules the numerics enforce, never to be misquoted downstream.
  (i) The entropy-rate *floor* is $0.9782$ *everywhere*; the converged learned energy
  $0.9787$ sits slightly *above* it (the finite-learning residual), so $0.979$ is *not*
  the floor (fix M/IT-10). (ii) The perfect-predictor sample mean $0.9760$ may dip below
  the expectation $0.9782$ by ordinary finite-sample fluctuation (standard error
  $tilde.op 10^(-3)$ over $2 times 10^6$ symbols) — this is not a violation of the floor,
  which bounds the *expected* per-symbol cost (fix M/PR-09/IT-03). (iii) The $44%$
  energy headline compares the marginal baseline $1.75$ against the conditional floor
  $0.978$; the energy "stupidity tax" figures are scoped to the energy corollary's
  model (fix D, fix M).
]

The two new checks close the loop on the two largest algebraic repairs: the measured
descent constant is $ln 2$ to within numerical error (vindicating fix B over the
printed $eta$), and the sum-mode conservation preserves every row sum to machine
precision (vindicating the gauge-fixed boundedness invariant of fix I at every
learning step, not merely at the fixed point).

// ===== S15-outlook =====
= Outlook: the research programme this opens

The learned coder of the preceding sections is the first rung, not the ladder.
Each direction below names a concrete next step, several reachable with the
existing DEQ and verification toolkits, and each is stated as an *open problem*
rather than a result. They share one organising principle: the circuit's
metabolic energy is its loss, so every extension is a question about *which loss
the same local, energy-descending substrate can be made to minimise*, and
*which of its guarantees survive*.

== Deeper memory: eligibility traces and temporal-difference credit

The local rule we derived correlates a residual at time $t$ with the context
$c_t$ *active at the same step*. That is exactly right when the useful context is
the immediately preceding symbol — a first-order source. It is *not* enough when
the predictive structure spans several past symbols, held as a graded bump that
drifts along the line/ring attractor (recall fix G: the attractor stores a
*finite, noise-limited* number of distinguishable context states, not arbitrary
history). Then the residual that finally resolves at time $t$ must be credited to
synapses that were active *earlier*.

#background("eligibility traces and temporal-difference learning")[
  The standard device for crediting a present error to past activity is an
  *eligibility trace*: a fading per-synapse memory
  $ e_(i j) arrow.l gamma e_(i j) + c_i, $
  decaying with factor $gamma in [0, 1)$, so that a synapse stays "eligible" for
  reward for a while after it was active. Pairing the trace with the residual,
  $ Delta W_(i j) = eta dot e_(i j) dot r_j, $
  assigns a late error to recently-active synapses in proportion to how recently
  they fired. This is the mechanism of *temporal-difference* learning,
  TD($lambda$), in which a single scalar prediction is corrected by bootstrapping
  from its own later values; biologically the trace is a slow synaptic
  eligibility signal multiplied by a delayed neuromodulatory gate — precisely the
  three-factor form @eq-rule already has, with the instantaneous pre-factor
  $c_i$ replaced by its low-pass history $e_(i j)$ @fremauxgerstner2016.
]

#openproblem[
  *Certified memory order under eligibility traces.* The attractor's hold time
  and the trace decay $gamma$ jointly bound the reachable memory order, but the
  two interact: a trace longer than the attractor's noise-limited persistence
  credits errors to a context the network can no longer distinguish, while a
  trace shorter than the predictive horizon throws away recoverable mutual
  information. *Open:* derive the reachable conditional-entropy floor as a joint
  function of attractor capacity ($tilde.op log_2 "SNR"$ bits) and $gamma$, and
  certify it — an attractor-capacity question squarely inside the
  `closed_form_wta_multi` and population toolkits. Does the Lyapunov descent of
  the convergence theorem survive the bootstrap, given that TD targets are
  *non-stationary* (they move as $W$ moves)?
]

== Non-stationary sources: tracking, and a regret bound

The convergence theorem assumed a *fixed* source and used a *decreasing* step
schedule $eta_t = eta_0 slash (1 + t slash t_0)$, which — by the Robbins–Monro
conditions $sum_t eta_t = infinity$, $sum_t eta_t^2 < infinity$ — drives the
weights almost surely to the unique minimiser @robbinsmonro1951. That same
decreasing step is fatal for a *drifting* world: once $eta_t arrow 0$ the model
freezes, stale. Tracking a non-stationary source instead wants a *constant* step
$eta$, which (as fix N records) trades almost-sure convergence for a steady-state
*noise ball* of radius $O(eta)$ around the moving optimum — a controllable lag,
not an asymptotic floor.

#openproblem[
  *A regret bound for the spiking learner.* The clean target is an
  online-learning *regret* statement: bound the total excess energy
  $sum_(t=1)^T (ell_t - ell_t^*)$ over $T$ steps against a comparator. Against
  the single best fixed predictor one expects $O(sqrt(T))$; against a *path* of
  predictors of total variation $V_T$, an $O(V_T)$ or $O(sqrt(V_T T))$ *dynamic*
  regret. *Open:* prove such a bound for the constant-step local rule @eq-rule on
  the softmax predictor (the loss is convex in the logits, the favourable case),
  and express the optimal $eta$ as a function of the source's drift rate — a
  verifiable online guarantee attached to a physical spiking substrate. The
  energy reading gives the regret an operational meaning: it is literal wasted
  joules accumulated while chasing a moving world.
]

== Learned lossy compression and rate–distortion

Everything so far was *lossless* source coding: the decoder reconstructs the
symbol exactly. Replacing the hard winner-take-all decode with a *graded* WTA — a
soft argmax obtained by relaxing the sum-mode gain toward its sub-saturating
regime — lets the decoder commit to a *coarsened* symbol, merging
hard-to-distinguish outcomes and spending fewer spikes. The graded-bump width is
then a rate–distortion knob: wider bumps mean coarser symbols, lower rate, higher
reconstruction distortion.

What learning adds is that the knob becomes *trainable*. Adding a distortion
penalty $d(x_t, hat(x)_t)$ to the per-symbol energy makes the *same* descent
trade reconstruction error against spikes, so the circuit learns the *quantiser*
(which symbols to merge) jointly with the predictor — a spiking analogue of a
learned lossy codec.

#openproblem[
  *A verifiable distortion bound for a learned spiking quantiser.* Augment the
  energy to $ell_t + beta dot d(x_t, hat(x)_t)$ and ask whether the local rule
  still descends a Lyapunov function (now the *Lagrangian* of the rate–distortion
  problem at multiplier $beta$), and whether the mode decomposition still yields
  the difference-mode / sum-mode split that made correctness independent of
  optimality. *Open:* derive, from the graded-WTA closed form, a *certifiable*
  upper bound on distortion at a given spike rate — a quantitative obligation in
  the same family as the near-entropy cost property, but for the lossy regime.
]

== A realistic anchor: neuromorphic event streams

The momentum rover earns its place by keeping every quantity closed-form, but it
is a rehearsal. The real target is data that *is already spikes*: a
dynamic-vision-sensor (event-camera) stream, whose pixels emit events only on
local brightness change. Such streams are heavily spatiotemporally redundant —
objects move predictably — so a recurrent predictor that pre-charges the expected
next events by *inhibitory* feedback (per fix F, shunting the expected drive so
the soma integrates only the residual) and emits only the unpredicted events is a
literal, hardware-native compressor. No re-encoding step intervenes: the input is
spikes, the code is spikes, the residual *is* the compressed stream.

#openproblem[
  *Close the loop on real event data.* The rover's first-order conditional law
  becomes, for an event camera, a high-dimensional spatiotemporal predictor whose
  context is the attractor's running scene estimate. *Open:* does the
  energy-as-loss identity (fix D: time-valued and exact; energy-valued only under
  the spike-count corollary's biophysical model) hold up when the alphabet is the
  pixel-event field rather than four moves, and what is the realised compression
  ratio against a conventional event-stream codec? This is where "spikes as bits"
  stops being a calibrated identity on a toy chain and becomes a measurable codec
  on a neuromorphic sensor.
]

== On-chip neuromorphic plasticity

Because the rule @eq-rule is *local* and *three-factor* — each weight update reads
only its presynaptic context activity, its postsynaptic residual spike, and a
single broadcast gate — it is implementable on neuromorphic hardware *without* a
global backward pass. There is no error vector to shuttle across the chip; the
teaching signal is the error spike the synapse already sees.

#openproblem[
  *A self-objective neuromorphic chip with a verification target.* The energy
  interpretation hands such a chip a built-in, physically meaningful objective —
  *minimise your own spike energy* — and the structural invariants discharged
  earlier (locality, sum-mode invariance, gauge-fixed weight boundedness; recall
  the centering $abs(W_(i j) - (1 slash n) sum_k W_(i k)) <= W_max$ of fix I)
  give it a *verification* target that a model checker can certify against the
  realised hardware update. *Open:* does a physical implementation, with its
  finite weight precision, quantised gate, and device noise, still preserve the
  sum-mode partition-of-unity invariant to within a certified tolerance, and does
  its energy still descend monotonically in the averaged sense?
]

== Meta-level: learning the calibration

Two constants were treated as fixed: the exponent $alpha = lambda slash (tau ln
2)$ that calibrates latency to surprisal, and the step $eta$. Both are in
principle tunable to the source — $alpha$ to its timescale, $eta$ to its
volatility. A slow outer loop that adapts them closes the system into a
self-calibrating coder.

#openproblem[
  *A Lyapunov certificate for the outer loop.* The inner loop's convergence rests
  on a clean energy Lyapunov function. *Open:* does the coupled inner–outer system
  (fast weights descending energy, slow $(alpha, eta)$ descending some
  meta-objective such as time-to-floor or steady-state regret) admit its own joint
  Lyapunov certificate, or can the two timescales interfere? Note the realism
  ceiling of fix E: $alpha$ cannot be pushed arbitrarily, since the calibration
  drive $R I(q) = theta slash (1 - q^alpha)$ already demands $tilde.op 7 times$
  rheobase at $q = 0.9$ and pins $q_max approx 0.85$–$0.95$ — the outer loop must
  respect that ceiling.
]

== Channel coding: where cycles might become unavoidable

Everything in this paper was *source* coding — removing redundancy. Its mirror
image is *channel* coding, which *adds* structured redundancy so a message
survives a *noisy* channel, and it is the one place where the recurrence of an
SNN looks not merely helpful but structurally distinctive.

#background("error-correcting codes and iterative decoding")[
  A *low-density parity-check* (LDPC) code protects a message with a sparse set of
  parity constraints and is decoded by *belief propagation*: an iterative
  message-passing procedure on a bipartite graph @gallager1962. When that graph
  has cycles — "loopy" BP — the procedure is genuinely recurrent, and its fixed
  points are exactly the stationary points of a variational objective, the *Bethe
  free energy* @yedidia2005. An attractor network settling to its nearest stored
  pattern is a physical instance of the same iterative, cyclic relaxation: the
  same recurrent substrate that compresses on its forward pass could
  error-correct on its settling pass.
]

It is tempting to claim that recurrence is here *provably necessary* — that loopy
belief propagation has no feedforward equivalent. We do not make that claim, and
fix J requires us to say why plainly.

#openproblem[
  *Is recurrence necessary, or only advantageous, for iterative decoding?* What is
  true and citable is narrower: the *exact fixed points* of loopy belief
  propagation — the stationary points of the Bethe free energy @yedidia2005 — are
  *not reproduced by any fixed-depth feedforward unrolling*, because a fixed
  number of message-passing rounds computes a fixed-depth approximation, not the
  converged fixed point. But this is *not* a proof of necessity: a *truncated* BP
  schedule can be unrolled into a feedforward network (this is exactly how many
  learned decoders are built), and whether some bounded-depth feedforward circuit
  can match loopy-BP *decoding performance* (as opposed to reproducing its exact
  fixed points) is open. *We therefore state it as a conjecture:* recurrence is
  *advantageous* — plausibly essential for matching converged loopy-BP error rates
  at finite depth — and frame the matter as future work, not as a theorem. The
  honest target is to characterise the gap between a depth-$k$ unrolling and the
  loopy-BP fixed point as a function of $k$ and the graph's girth, on the same
  attractor substrate this paper already verifies.
]

#intuition[
  The unifying wish behind all eight problems is one substrate, both halves of
  Shannon's programme: a single recurrent spiking circuit that *compresses* on its
  forward pass (the source coder of this paper) and *protects* on its settling
  pass (an iterative channel decoder), with a single local plasticity rule tuning
  both. Whether that wish is a theorem or merely an aesthetic is precisely what
  the open problems above are meant to decide — and we have been careful to label
  which clauses are proven, which are model-dependent corollaries, and which are
  conjecture.
]

// ===== S16-appendix =====
= Appendix: reproduced numerics and property templates

This appendix is the paper's reproducibility kernel. It collects, in one place, (a) the full table of numbers quoted throughout the main text, each with the exact command that regenerates it; (b) the two adversarial checks added during review that close the two load-bearing gaps in the original notes (sum-mode conservation and the Lyapunov-rate constant); and (c) the probabilistic-temporal-logic (PCTL) property templates that turn the informal "safety vs optimality" split into machine-checkable obligations, stated in the corrected *settled* form. Everything here is pure-`numpy` and self-contained: no result below depends on a learned artifact, a random seed beyond the ones listed, or any package outside the project virtual environment `deq/.venv`.

A reader who only wants to trust the numbers can run the three scripts and compare against @tbl-repro; a reader who wants to verify the formal obligations can instantiate @lst-pctl against a concrete model.

== A. Reproduced numerics

Three scripts regenerate every quantitative claim. All are invoked with the project interpreter (`deq/.venv/bin/python`) from the repository root, so that the relative paths resolve:

#block(width: 100%, fill: luma(245), inset: 6pt)[
#raw("# Source statistics + LIF latency calibration + stream identity
deq/.venv/bin/python research/compression/validate.py

# Gradient identity + online learning + averaged-flow Lyapunov descent
deq/.venv/bin/python research/compression/learn_validate.py

# The two review checks: sum-mode conservation + Lyapunov-rate constant
deq/.venv/bin/python research/compression/critique_checks.py", lang: none)]

The momentum-rover source is fixed throughout: stationary law $pi = (1/2, 1/4, 1/8, 1/8)$ over the alphabet $cal(X) = {U, D, L, R}$, transition matrix $P = s I + (1-s) bb(1) pi^top$ with stickiness $s = 0.7$ (the "stay-or-resample" chain). The latency law is $t^*(q) = -lambda log_2 q$ with $tau = lambda = theta = 1$, so the calibration exponent is $alpha = lambda slash (tau ln 2) = 1 slash ln 2 approx 1.4427$ (the latency-calibration theorem in the circuit section). Streams are $n = 2 times 10^6$ symbols, seed $7$, sampled from $P$ started at $pi$.

@tbl-repro lists each reported quantity, the value it must reproduce, and the script that emits it. Numbers are quoted *exactly* as the validators print them; do not round-trip them through any other figure.

#figure(
  table(
    columns: (1fr, auto, auto),
    align: (left, left, left),
    stroke: 0.5pt + luma(180),
    table.header[*Quantity (symbol)*][*Reproduced value*][*Script*],
    [Marginal symbol entropy $H(pi)$], [$1.7500$ bits/sym], [`validate.py`],
    [Conditional entropy rate $overline(H)(P)$, $s=0.7$], [$0.9782$ bits/sym], [`validate.py`],
    [Mutual information $I(x_t; x_(t-1)) = H(pi) - overline(H)(P)$], [$0.7718$ bits/sym], [`validate.py`],
    [Stationarity drift $max abs(pi P - pi)$], [$2.78 times 10^(-17)$], [`validate.py`],
    [LIF first-spike latency error $max abs(t^* + log_2 q)$], [$1.56 times 10^(-5)$ s], [`validate.py`],
    [Stream cost — perfect predictor $q = P$ (empirical / theory)], [$0.9760 slash 0.9782$], [`validate.py`],
    [Stream cost — memoryless $q = pi$ (empirical / theory)], [$1.7477 slash 1.7500$], [`validate.py`],
    [Stream cost — wrong momentum $s' = 0.4$ (empirical / theory)], [$1.1114 slash 1.1133$], [`validate.py`],
    [Stupidity tax — memoryless $overline(D)_("KL")(p || pi)$], [$0.7718$ bits/sym], [`validate.py`],
    [Stupidity tax — wrong momentum], [$0.1351$ bits/sym], [`validate.py`],
    [Gradient identity residual $max abs((q - e_j) slash ln 2 - "num")$], [$1.12 times 10^(-10)$], [`learn_validate.py`],
    [Final learned energy $overline(H)(p,q)$ (vs floor $0.9782$)], [$0.9787$ bits/sym], [`learn_validate.py`],
    [Final excess $overline(D)_("KL")(p || q)$], [$0.0004$ bits/sym], [`learn_validate.py`],
    [Averaged-flow Lyapunov monotone fraction], [$1.0$], [`learn_validate.py`],
    [Averaged-flow max energy *increase*], [$0.0$], [`learn_validate.py`],
    [Sum-mode row-sum drift $max abs(sum_j W_(i j) - "init")$], [$approx$ machine-$epsilon$], [`critique_checks.py`],
    [Lyapunov-rate constant $k$ in $dot(V) = -k norm(nabla V)^2$], [$0.693 = ln 2$], [`critique_checks.py`],
  ),
  caption: [Reproduced numerics. Every figure quoted in the main text appears here with its regenerating command. Empirical stream costs are sample means over $n = 2 times 10^6$ symbols (seed $7$); theory columns are the closed-form cross-entropy rates $overline(H)(p,q) = sum_i pi_i sum_j P_(i j)(-log_2 q_(i j))$.],
) <tbl-repro>

Three readings of this table are load-bearing, and each fixes a way the source notes mis-stated the numbers.

#honest[
*Read the floor as $0.9782$, never $0.979$.* The conditional entropy-rate floor is $overline(H)(P) = 0.9782$ bits/symbol (fix M). The *learned* model's converged energy is $0.9787$ — strictly *above* the floor by the finite-learning residual $overline(D)_("KL")(p||q) = 0.0004$. So $0.9787$ is not "the floor reached," it is "the floor plus a small residual"; never round the two together and never call $0.979$ "the floor." Conversely the empirical *perfect-predictor* stream cost is $0.9760$, which sits *below* the expectation $0.9782$. That is not a violation of the source-coding bound: $0.9760$ is a *sample mean* over a finite stream and may dip below its expectation $overline(H)(P) = 0.9782$ by ordinary finite-sample fluctuation. The theory column ($0.9782$) is the quantity the inequality constrains; the empirical column is an estimator of it.
]

#remark[
*Scope of the "44%" and "stupidity-tax" figures.* The recurrent cycle recovers $H(pi) - overline(H)(P) = 1.7500 - 0.9782 = 0.7718$ bits/symbol, i.e. $44%$ of the marginal cost — this compares the *marginal* baseline $1.75$ against the *conditional* floor $0.978$ (fix M); it is a bits/symbol statement about the latency/time code (the latency-calibration theorem), exact and substrate-free. The "stupidity tax" of a mismatched model — $0.7718$ bits/symbol for the memoryless predictor, $0.1351$ for the wrong-momentum guess — is the excess cross-entropy rate $overline(D)_("KL")(p||q)$, again a bits/symbol time-code quantity. Any restatement of these as *energy* (joules) is scoped to the energy *corollary*'s biophysical model (the $kappa$ J/bit substrate constant), which is model-dependent and distinct from the exact time calibration $lambda$ (coherence brief (iii)); the energy reading holds in the spike-count regime and may fail for long-latency rare symbols.
]

== B. The two review checks

The original notes left two quantitative claims unverified. `critique_checks.py` adds exactly two checks, both of which *PASS*; they are the empirical backbone of fixes B (Lyapunov constant) and the sum-mode half of the partition-of-unity invariant (coherence brief (i)).

=== B.1 Sum-mode conservation (CHECK 1)

The online rule updates one context row per step by $Delta W_(i dot) = eta_t (e_j - q)$ with $q = "softmax"(W_(i dot))$. Because $sum_j (e_j)_j = 1$ and $sum_j q_j = 1$ exactly (softmax is the simplex normalizer — coherence brief (i)), every update has zero row-sum:
$ sum_j Delta W_(i j) = eta_t (sum_j (e_j)_j - sum_j q_j) = eta_t (1 - 1) = 0 . $ <eq-summode>
Hence each row-sum of $W$ is an *exact invariant* of learning: the update lives entirely in the difference modes, never the sum mode. This is the discrete analogue of Note II's conservation law $sum_j Delta W_(i j) = 0$, and it is what preserves the partition-of-unity $sum_j q_j = 1$ at every step — *on the softmax*, not on divisive normalization (which sums to $sum a slash (sigma + sum a) < 1$ and only approaches unity as $sigma -> 0$).

The check starts $W$ from a *nonzero* Gaussian init (so "constant row-sums" is a non-trivial test, not the vacuous "$0$ stays $0$"), runs $2 times 10^5$ online steps with the decreasing schedule $eta_t = eta_0 slash (1 + t slash t_0)$ ($eta_0 = 0.2$, $t_0 = 5 times 10^4$), and tracks both the per-step update row-sum and the cumulative drift of each row-sum from its initial value. Both stay below $10^(-9)$ — the drift is at machine precision, confirming @eq-summode holds along a full trajectory and not merely at a fixed point (the inductive transient argument of fix I).

=== B.2 Lyapunov-rate constant (CHECK 2)

The original "$dot(V) = -eta norm(nabla V)^2$" dropped a $ln 2$ factor. With the excess energy $V(W) = sum_i pi_i overline(D)_("KL")(P_(i dot) || q_(i dot))$ measured in *bits*, its analytic gradient is $nabla V_(i j) = -pi_i (P_(i j) - q_(i j)) slash ln 2$, while the averaged flow advances $dot(W)_(i j) = pi_i (P_(i j) - q_(i j)) = -ln 2 dot nabla V_(i j)$. Therefore
$ dot(V) = chevron.l nabla V, dot(W) chevron.r = -ln 2 dot norm(nabla V)^2 , $ <eq-lyaprate>
so the constant is $ln 2$ (the bits$arrow.l.r$nats conversion), *not* $eta$ (fix B). CHECK 2 integrates the averaged flow with $d t = 10^(-3)$ for $4 times 10^4$ steps, estimates $k = (-dot(V)) slash norm(nabla V)^2$ by finite difference, and reports the median over the latter three-quarters (discarding the finite-difference transient). The measured constant is $k = 0.693 = ln 2$, matching @eq-lyaprate and clearly distinct from the paper's claimed $eta$. This is the empirical witness that the corrected descent identity adopted in the main text — define the flow as $dot(W) = -eta nabla V$ so that $dot(V) = -eta norm(nabla V)^2 <= 0$ holds *by construction*, and relate it to the rule's expected step $bb(E)[Delta W_(i j)] = eta pi_i (P_(i j) - q_(i j)) = -eta dot ln 2 dot nabla V_(i j)$ — has the right constant.

#remark[
*Robbins–Monro, not a noise ball.* The online learner uses the *decreasing* schedule $eta_t = eta_0 slash (1 + t slash t_0)$, which satisfies the stochastic-approximation conditions for almost-sure convergence @robbinsmonro1951; the final $overline(D)_("KL")(p||q) = 0.0004$ is a *finite-time* residual, not an asymptotic floor (fix N). A *constant*-$eta$ rule would instead settle into an $O(eta)$ noise ball around the optimum. The two regimes are different and the table's $0.0004$ belongs to the first.
]

== C. PCTL property templates

The note's informal split — *correctness* (any predictor) vs *optimality* (predictor-dependent) — is made machine-checkable as two property classes in `spiking_entropy_coder.pctl`. They are instantiated against a concrete model: a DTMC over the momentum-rover source coupled to a $4$-neuron temporal winner-take-all (WTA) readout, in the style of the contralateral-inhibition PRISM model. The atomic labels are defined concretely so the formulas are instantiable:

- `win_i` — readout neuron $i$ ($i in 1..4$) crossed threshold first in the current symbol window;
- `winner` — the count $sum_i$ `win_i` (number of settled winners this window);
- `decoded` — symbol id elected by the WTA decode rule (the index of the first crosser);
- `emitted` — the true source symbol $x_t$ for this window;
- `new_window` — the window-boundary marker (layer reset before the next integration);
- reward `"time"` — per-window latency $-lambda log_2 q(x_t mid(|) c_t)$, the surprisal of the realized symbol.

The crucial correction (fix H, WTA-safety) is that "at most one winner" is *false* as a global invariant: during the integration transient several readouts are simultaneously sub-threshold-active, so `winner` momentarily exceeds $1$. The certifiable obligation is the *settled* form — eventually, and thereafter stably, exactly one settled winner per window (an `F G` shape) — together with the *decode* guarantee that whoever settles first is the correct symbol. "A bad model is slow, never wrong" rests on this decode rule, not on any global mutual-exclusion.

#figure(
  block(width: 100%, fill: luma(245), inset: 6pt)[
#raw("// (A) STRUCTURAL / SAFETY  -- hold for ANY predictor q (correctness)

// A1: Settled uniqueness. Eventually a window reaches a state from which
//     exactly one settled winner persists (NOT a global G(winner<=1)).
P>=1 [ F G (winner = 1) ]

// A2: Liveness / decodability. Every window eventually elects a winner.
P>=1 [ G F (winner >= 1) ]

// A3: Losslessness on the DECODE rule. Once a winner has settled, the
//     decoded symbol equals the emitted one: decode(encode(x)) = x.
P>=1 [ G ( (winner = 1) => (decoded = emitted) ) ]

// A4: Self-clocking (prefix-freeness). A settled winner is followed by a
//     reset before the next window's integration; windows do not interleave.
P>=1 [ G ( (winner = 1) => X (winner = 0 U new_window) ) ]


// (B) QUANTITATIVE / OPTIMALITY -- predictor-dependent

// B1: Steady-state expected per-window latency. Optimality target is the
//     entropy-rate floor:  R{\"time\"}=?[S]  ==  lambda*(Hrate + eps),
//     Hrate = 0.9782 bits/symbol (validate.py).
R{\"time\"}=? [ S ]

// B2: Transient cost to absorb a length-N stream. Add an absorbing 'done'
//     state; compare against lambda*Hrate*N for finite verification.
R{\"time\"}=? [ F done ]

// B3: Stupidity tax, checkable. Evaluate B1 for the learned q and for the
//     memoryless q=pi; the gap is lambda*D_KL(p||pi) = 0.7718 bits/symbol
//     (validate.py). A learning step is an improvement iff it lowers B1.

// B4: Sanity sandwich (Shannon source coding, physical form):
//     lambda*Hrate  <=  R{\"time\"}=?[S]  <  lambda*Hrate + (timing penalty).
//     In continuous latency the classic '+1 bit' slack collapses; the
//     residual slack is the analog timing-resolution penalty.", lang: none)
  ],
  caption: [Corrected PCTL templates (`spiking_entropy_coder.pctl`). Class A (safety) is predictor-independent and certified once; class B (optimality) is the predictor-dependent quantity learning drives down. A1 is stated in the *settled* `F G` form, not the false global `G(winner<=1)` (fix H); losslessness A3 is guarded on the decode rule.],
) <lst-pctl>

#intuition[
The payoff of the A/B split is that *correctness never trades against optimality*. The structural class A is discharged once, for the worst predictor imaginable — a uniform $q = (1/4,...,1/4)$ still decodes losslessly, it is merely *slow* (every window's first crosser is still the correct symbol; the race only takes longer). The quantitative class B is the dial training turns: B1 falls monotonically from $lambda log_2 4 = 2lambda$ (uniform init) toward the floor $lambda dot 0.9782$, and B3 makes the gap to a baseline an explicit checkable reward difference. This is the formal-methods analogue of the bits/symbol story: a bad model pays the stupidity tax in *latency* (class B), never in *correctness* (class A).
]

#honest[
*Two honest caveats on the templates.* (i) For the reward `"time"` to equal the cross-entropy *rate* $lambda dot overline(H)(p,q)$ in the steady-state operator `S`, the predictor $q$ must itself be a first-order conditional model matched to the source's structure (fix A2); for a mismatched-structure $q$, `R{"time"}=?[S]` is a per-symbol average but not literally the cross-entropy rate. (ii) A pure DTMC has no continuous time, so `"time"` must be realized as a spike-*count* proxy, which reintroduces an integer-bit / timing-resolution penalty (the B4 slack); the exact continuous "+1-bit collapse" is a property of the analog latency readout, not of the DTMC abstraction. Both caveats are inherited from the time-vs-energy and resolution honesty blocks in the main text and do not affect the *structural* class A, which is timing-free.
]

#bibliography("refs_compression.bib", title: "References", style: "ieee")
