// Research Note: A Verifiable Spiking Entropy Coder
// Self-contained: compression == prediction == intelligence, built from
// scratch and ported to recurrent SNN circuits.
// CogSpike / research/compression, June 2026

#set document(
  title: "Spikes as Bits: A Verifiable Spiking Entropy Coder",
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
#set math.equation(numbering: "(1)")

// Theorem-like environments (consistent with research/limits note)
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
// Context a reader may need, supplied BEFORE it is used.
#let background(title, body) = block(width: 100%, inset: 8pt,
  fill: rgb("#f1faf3"), stroke: (left: 2pt + rgb("#3a9d5d")),
  [📘 *Background — #title.* #body])
#let honest(body) = block(width: 100%, inset: 8pt, fill: rgb("#fff7f0"),
  stroke: (left: 2pt + rgb("#d98a4a")), [⚖️ *Honesty check:* #body])

#align(center)[
  #text(size: 16pt, weight: "bold")[Spikes as Bits]
  #v(0.2em)
  #text(size: 13pt)[A Verifiable Spiking Entropy Coder, and the
  Compression–Prediction Equivalence in Recurrent Neural Circuits]
  #v(0.4em)
  #text(size: 10pt, style: "italic")[
    `research/compression/`, June 2026 · CogSpike research note
  ]
]

#v(0.5em)

#block(width: 100%, inset: 10pt, fill: luma(247), radius: 4pt)[
  *Abstract.* We construct a recurrent spiking neural circuit whose energy bill
  is, exactly, the information-theoretic cost of its own predictions, and we
  state the obligations that certify it. The construction rests on one
  observation: a spiking neuron driven precisely to its firing threshold by an
  *expected* input stays silent, while a neuron whose input violates what its
  recurrent context predicted fires — so a circuit that subtracts its own
  prediction emits, per input symbol, a spike cost equal to the *surprisal*
  $-log_2 q(x_t mid(|) "context")$ of that symbol under the circuit's internal
  model $q$. Summed over a stream, the circuit's total spike-time equals the
  *cross-entropy* of its predictions against the source. Shannon's entropy is
  the floor; the excess is the Kullback–Leibler divergence between model and
  world, paid in literal joules. The circuit is "intelligent" to exactly the
  degree it closes that gap — and every clause of that statement is a property
  we can write down and check. This note is self-contained: it builds the one
  differential equation, the information measures, the coding theory, the
  Markov source, and the recurrent dynamics it needs, then assembles the coder
  and writes its verification obligations in the archetype-and-model-checking
  language the CogSpike program already uses for contralateral inhibition.
]

= Orientation: the claim, and what we must build

A long-standing idea links three words that sound unrelated:

#block(inset: (left: 10pt, top: 4pt, bottom: 4pt))[
  *compression* $space arrow.l.r space$ *prediction* $space arrow.l.r space$
  *intelligence.*
]

The middle equivalence is the precise one. To compress data you must predict
it: a coder that knows the probability of the next symbol can spend few bits on
likely symbols and many on rare ones, reaching the theoretical minimum. The
converse holds too: anything that compresses well must, inside it, *be*
predicting well — it must hold an accurate model of the data's structure. The
flanking word "intelligence" is the informal claim that *building such a model
is what understanding is.* Our goal is to make all three words mean something
concrete and checkable in a *spiking neural network* (SNN), and in particular
to pin down what the SNN's defining feature — that it is a *circuit with
cycles*, not a feedforward pipeline — actually buys.

The plan, and the order in which we supply background:

#block(inset: (left: 6pt))[
  *§2* introduces spiking neurons from a single differential equation, and the
  one trick — *predictive subtraction* — that needs a cycle. *§3* introduces the
  information measures (surprisal, entropy, cross-entropy, divergence) from
  scratch. *§4* introduces just enough coding theory (prefix codes, the Kraft
  inequality, the integer-bit penalty) to state what "optimal" means. *§5*
  proves the central identity exactly: with the right drive, a neuron's
  first-spike *time* equals a symbol's surprisal, so a stream's total spike-time
  equals the cross-entropy — and, because time is continuous, with no rounding
  penalty. *§6* introduces sources *with memory* (Markov chains, entropy rate)
  and picks one that forces the cycle to do work. *§7* introduces the recurrent
  ingredients (attractors, winner-take-all, gain control). *§8* assembles the
  circuit; *§9* certifies its correctness through a mode decomposition; *§10*
  reads the cost in joules; *§11* writes the model-checking obligations; *§12*
  states the intelligence claim precisely; *§13–14* are honest limitations and
  outlook.
]

We assume a reader fluent in *formal methods* — probabilistic model checking,
discrete- and continuous-time Markov models, temporal logic, the
safety/liveness distinction — and we build everything else (neural dynamics,
information theory, coding theory, dynamical systems) as we go.

= Spiking neurons, from one differential equation

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
  $ tau dot(V)(t) = -V(t) + R I(t), $
  where $tau > 0$ is the membrane time constant (in seconds), $R$ the input
  resistance, and $dot(V) = d V slash d t$ the time derivative. On reaching a
  threshold $theta$ the neuron emits a spike and $V$ resets to $0$. Between
  spikes this is a linear first-order ordinary differential equation (ODE).
]

#background("reading a first-order ODE")[
  An ODE relates a quantity to its own rate of change. Here $dot(V) = (-V + R
  I) slash tau$ says: the rate at which $V$ changes equals (target $- $ current
  value), scaled by $1 slash tau$. Whenever the right-hand side is zero the
  quantity stops moving — that is an *equilibrium* (here $V = R I$). The
  constant $tau$ sets the timescale: large $tau$ means sluggish, small $tau$
  means snappy. "Solving" the ODE means finding $V(t)$ as an explicit function
  of time given a starting value $V(0)$.
]

#intuition[
  Read $tau dot(V) = -V + R I$ as a leaky bucket. $R I$ is the inflow rate; the
  $-V$ term is a leak proportional to how full the bucket already is. Left alone
  with constant inflow, the level settles where inflow balances leak, at $V = R
  I$. A spike is the bucket brimming over a rim at height $theta$; resetting to
  $0$ empties it. Everything we need is "how long until the next overflow."
]

For constant drive $I$ from rest $V(0) = 0$, the ODE solves to
$ V(t) = R I (1 - e^(-t slash tau)), $
a curve rising monotonically toward its equilibrium $R I$. If $R I > theta$ the
curve crosses threshold; setting $V(t^*) = theta$ and solving for $t^*$ gives
the *first-spike latency*
$ t^*(I) = tau ln ( (R I) / (R I - theta) ). $ <eq-latency>

This single formula is the heart of everything. Note two ways to read a neuron's
output, both of which we will use:

- *Count reading.* A neuron driven harder overflows more often; spike *count*
  over a window grows with the drive. A surprising input means a large drive
  means a burst of spikes.
- *Latency reading.* A neuron driven just barely above threshold fills slowly
  and fires *late*; a strongly driven neuron fires *early*. The first-spike
  *time* carries the message, and the spike *delimits its own symbol* — the
  instant it happens, the symbol is over.

#intuition[
  *Why a cycle is the active ingredient.* Suppose a feedback loop has already
  charged a neuron with subthreshold depolarization equal to the input it
  *expects*. When the true input arrives, the neuron effectively sees only the
  *residual*, $"input" - "prediction"$. A perfectly predicted input adds nothing
  new — near-zero residual, near-zero firing. This *predictive subtraction* is
  the one operation a feedforward stage cannot perform across time: it requires
  a path carrying the prediction back to the input. With it, the emitted spike
  train stops being a copy of the symbol stream and becomes a *surprise* stream.
  That is the whole mechanism; the rest of the note quantifies it.
]

= Information, measured in bits

We now make "surprise" numerical. Let a source emit symbols from a finite
alphabet $cal(X)$, the true symbol probabilities being $p(x)$; let $q(x)$ be the
probabilities our circuit's *internal model* assigns.

#definition[
  The *information content* (or *surprisal*) of a symbol $x$ under model $q$ is
  $ I_q (x) = -log_2 q(x) = log_2 (1 slash q(x)) "bits". $
  The *entropy* of the source is the expected surprisal under the *true* law,
  $ H(p) = -sum_x p(x) log_2 p(x), $
  the *cross-entropy* is the expected surprisal a model $q$ pays on data drawn
  from $p$,
  $ H(p, q) = -sum_x p(x) log_2 q(x), $
  and their difference is the *Kullback–Leibler (KL) divergence*
  $ D_("KL")(p || q) = H(p, q) - H(p) = sum_x p(x) log_2 (p(x)) / (q(x)) >= 0, $
  which is zero iff $q = p$.
]

#background("why the logarithm, and why these are the right measures")[
  Two demands fix the form. First, surprise should *decrease* with probability:
  a certain event ($q arrow 1$) carries no information ($I arrow 0$); an
  impossible one ($q arrow 0$) carries infinite information. Second, the
  surprise of *independent* events must *add*: learning two unrelated facts is
  twice the news. But independent probabilities *multiply*, and the logarithm is
  the unique continuous function turning products into sums, $log(a b) = log a +
  log b$. So $I_q = -log q$ is forced; base 2 merely names the unit "bit"
  (one bit = the information in one fair coin flip). *Entropy* is then the
  average surprise of the source — the irreducible uncertainty per symbol.
  *Cross-entropy* is the average surprise your *model* registers, which is at
  least the entropy and equals it only when the model is exactly right. That
  excess gap is the *KL divergence*: a non-negative number measuring how wrong
  $q$ is, in bits.
]

#intuition[
  The decomposition $H(p, q) = H(p) + D_("KL")(p || q)$ is the entire thesis in
  one line. $H(p)$ is what the *world* costs — irreducible. $D_("KL")$ is what
  *being wrong* costs — your model's avoidable waste. A learner can only lower
  its cross-entropy by shrinking $D_("KL")$, i.e. by making $q$ resemble the
  true $p$ — which is to say, by *coming to understand the source.* Below,
  $H(p, q)$ will literally be the circuit's spike-time bill.
]

= Codes, and the price of whole bits

To say the circuit is "near-optimal" we must say optimal *against what*. The
benchmark is coding theory, which we now sketch — it is also what makes our
continuous-time advantage (§5) legible.

#background("codes, prefix-freeness, and the Kraft inequality")[
  A *binary code* assigns each symbol $x$ a finite string of bits $C(x)$, of
  length $ell(x)$; a message is sent by concatenating codewords. A code is
  *prefix-free* (equivalently *instantaneous*) if no codeword is an initial
  segment of another. Then a receiver reading left to right knows a codeword has
  ended the moment it completes one — no delimiters, no lookahead. Geometrically,
  reserving a codeword of length $ell$ claims a sub-interval of $[0, 1)$ of width
  $2^(-ell)$ (all infinite strings beginning with that codeword), and prefix-
  freeness says the claimed intervals must not overlap. This is the *Kraft
  inequality*: a prefix-free binary code with lengths ${ell_i}$ exists iff
  $ sum_i 2^(-ell_i) <= 1. $
]

#background("the integer penalty, and arithmetic coding")[
  Minimizing expected length $L = sum_i p_i ell_i$ subject to Kraft gives the
  ideal lengths $ell_i = -log_2 p_i$, for which $L = H(p)$ exactly. But codeword
  lengths must be *whole numbers*. Rounding up to $ell_i = ceil(-log_2 p_i)$
  yields Shannon's bound
  $ H(p) <= L^* < H(p) + 1, $
  the "$+1$" being the cost of spending a whole bit where a fraction would do.
  *Arithmetic coding* removes this waste by refusing to encode symbols one at a
  time: it represents an entire *sequence* as a single sub-interval of $[0, 1)$
  whose width equals the sequence's probability, then transmits a point inside
  it. Pinning a point to enough precision costs $approx -log_2 P("sequence")$
  bits *total*, so the per-symbol rounding penalty melts away as the sequence
  grows. Arithmetic coding thus reaches the entropy floor — but only in the
  long-sequence limit, by amortization.
]

Keep two facts in hand: (i) the entropy $H(p)$ is the floor on average code
length, and (ii) discrete prefix codes pay up to one extra bit per symbol for
the indivisibility of bits, escapable only by amortizing over long blocks. In
§5 the spiking substrate will sidestep (ii) *per symbol*, because *time* is not
quantized into whole bits.

= The exact identity: spike-time is surprisal

Here is the move that turns "spikes are bits" from slogan into theorem. We
*calibrate* a readout neuron's drive so that the latency law @eq-latency outputs
precisely a symbol's surprisal.

#theorem("Latency calibration")[
  Fix a time-per-bit constant $lambda > 0$ and set the exponent $alpha = lambda
  slash (tau ln 2)$. Drive a readout neuron whose model probability for its
  symbol is $q in (0, 1)$ with the current
  $ R I(q) = theta / (1 - q^alpha). $ <eq-calib>
  Then its first-spike latency is exactly
  $ t^*(q) = -lambda log_2 q = lambda dot I_q. $
]

#proof[
  Substitute @eq-calib into the latency law @eq-latency. The argument of the
  logarithm becomes
  $ (R I) / (R I - theta) = (theta slash (1 - q^alpha)) / (theta slash (1 -
  q^alpha) - theta) = 1 / (1 - (1 - q^alpha)) = q^(-alpha). $
  Hence $t^* = tau ln(q^(-alpha)) = -alpha tau ln q = -(lambda slash ln 2) ln q
  = -lambda log_2 q$, using $alpha tau = lambda slash ln 2$ and $ln q = (log_2
  q)(ln 2)$.
]

The limiting behaviour is exactly what a code should do. A *certain* symbol
($q arrow 1$) demands $R I arrow infinity$ and fires *instantly* — zero bits,
zero time. An *impossible* symbol ($q arrow 0$) demands $R I arrow theta^+$
(barely above the minimum current that fires at all) and fires *never* —
infinite surprisal. Everything between interpolates smoothly and exactly. A
numerical integration of the LIF ODE under @eq-calib (companion script
`validate.py`) reproduces $t^*(q) = -log_2 q$ to the integrator's step size
($~10^(-5)$, pure discretization error) across $q in {0.0375, dots, 0.98}$.

Now stream it. Suppose the recurrent loop presents, at each step $t$, a
conditional model $q(dot mid(|) c_t)$ where $c_t$ is the loop's *context state*,
and the realized symbol is $x_t$. The readout for that symbol fires at latency
$-lambda log_2 q(x_t mid(|) c_t)$. Summing over the stream:

#theorem("Spike-time equals cross-entropy")[
  The total first-spike time to emit a length-$n$ stream is
  $ T_n = lambda sum_(t = 1)^n (-log_2 q(x_t mid(|) c_t)) = lambda dot
  ("total surprisal"). $
  Taking expectations over a stationary source $p$, with the loop holding the
  true contexts, the expected per-symbol time is the *cross-entropy rate*
  $ lim_(n arrow infinity) 1 / n EE[T_n] = lambda dot overline(H)(p, q), $
  equal to the entropy-rate floor $lambda dot overline(H)(p)$ iff $q = p$, and
  exceeding it by $lambda dot overline(D)_("KL")(p || q)$ otherwise.
]

The companion script confirms this on a $2 times 10^6$-symbol stream (source of
§6): a perfect predictor spends $0.976$ time-units per symbol (theory $0.978$),
a memoryless predictor spends $1.748$ (theory $1.750$), a wrong-parameter
predictor spends $1.111$ (theory $1.113$). The empirical means track the
cross-entropy rates to sampling error, and the gaps above the floor are exactly
the KL divergences.

== A price the binary tree pays and the time axis does not

Recall the integer penalty of §4: discrete prefix codes obey $H(p) <= L < H(p)
+ 1$, the slack coming from rounding fractional ideal lengths up to whole bits,
removable only by amortizing over long blocks. Spike *time* carries no such
constraint.

#theorem("Continuous time has no integer penalty")[
  The latency $t^*(q) = -lambda log_2 q$ is a *real* number; it is not required
  to be an integer multiple of any quantum. Therefore the calibrated spiking
  coder achieves the per-symbol cost $lambda(-log_2 q(x_t))$ *exactly*, for
  *every single symbol* — not merely in a block limit. With $q = p$ its expected
  cost is exactly $lambda H(p)$, the floor, with *zero* slack.
]

#intuition[
  Discrete prefix codes live on a *binary tree*: every codeword is a whole
  number of left/right turns, so you can only ever spend an integer number of
  bits, and a symbol whose ideal cost is $3.737$ bits gets charged $4$. The
  spiking coder lives on the *real time axis*: it spends $3.737$ bits' worth of
  *duration*, no rounding. The Kraft inequality's integer ceilings relax, in
  continuous time, into ordinary real inequalities — met with equality.
]

#honest[
  This advantage is bounded *below* by physics, not combinatorics, and we state
  the bound rather than hide it. (a) The "certain symbol fires instantly" limit
  needs unbounded drive; a real *refractory period* (a brief post-spike dead
  time) and current saturation impose a *minimum* latency $t_(min)$, hence a
  *maximum* representable probability $q_(max) < 1$ and a small floor cost on
  near-certain symbols. (b) Finite timing resolution $delta t$ and membrane
  noise turn the exact real number into a measured one: the recoverable
  precision is about $log_2(t^* slash delta t)$ bits, so the integer penalty is
  *replaced* by an analog timing-resolution penalty, not abolished. The honest
  statement: the spiking coder trades the combinatorial $+1$-bit rounding
  penalty for an analog timing penalty that is small when the clock is fast
  relative to the firing rate — exactly the regime neuromorphic hardware
  targets.
]

= Sources with memory, and one that forces the cycle to work

So far the loop's prediction $q(dot mid(|) c_t)$ could depend on context, but we
have not said why context *helps*. It helps only if the source has *temporal
structure* — if the next symbol is statistically tied to the past. A
*memoryless* source (successive symbols independent) gives a cycle nothing to
do: a fixed feedforward code is already optimal. So the case study must begin by
choosing a source whose past genuinely predicts its future, and by *quantifying*
the help up front.

#background("Markov chains, stationary law, and entropy rate")[
  A *(first-order) Markov chain* over states $cal(X)$ is specified by transition
  probabilities $P_(i j) = Pr(x_(t+1) = j mid(|) x_t = i)$: the next symbol
  depends on the current one and no earlier history. A distribution $pi$ over
  states is *stationary* if it is unchanged by one step, $pi P = pi$ — the
  long-run fraction of time spent in each state. Two entropies now differ. The
  *marginal* entropy $H(pi)$ treats symbols as if independent with frequencies
  $pi$; it is what a memoryless code pays. The *entropy rate* conditions on the
  predecessor,
  $ overline(H)(P) = sum_i pi_i H(P_(i dot)), quad H(P_(i dot)) = -sum_j P_(i j)
  log_2 P_(i j), $
  the average residual uncertainty *given* the last symbol; it is the true floor
  for a predictor that uses memory. Their difference is the *mutual information*
  $I(x_t; x_(t-1)) = H(pi) - overline(H)(P) >= 0$ between consecutive symbols —
  precisely the bits a memory can recover. (Markov chains are also the native
  object of probabilistic model checkers, which we exploit in §11.)
]

We pick a one-parameter source designed so the arithmetic is clean and the
*only* thing changing is the memory.

#definition[
  *Momentum rover.* A rover moves on ${U, D, L, R}$ with long-run frequencies
  $pi = (1 slash 2, 1 slash 4, 1 slash 8, 1 slash 8)$, but with *inertia*: with
  "stickiness" $s in [0, 1)$ it repeats its last move; otherwise it draws a
  fresh move from $pi$. The transition matrix is
  $ P_(i j) = s dot bb(1)[i = j] + (1 - s) pi_j. $
]

#proposition[
  For every $s in [0, 1)$ the momentum rover has stationary distribution exactly
  $pi$.
]
#proof[
  $(pi P)_j = sum_i pi_i (s bb(1)[i = j] + (1 - s) pi_j) = s pi_j + (1 - s) pi_j
  sum_i pi_i = s pi_j + (1 - s) pi_j = pi_j$, since $sum_i pi_i = 1$.
]

So the *marginal* statistics — and a memoryless code's cost — are pinned at
$H(pi) = -(1 / 2 log_2 1 / 2 + 1 / 4 log_2 1 / 4 + 2 dot 1 / 8 log_2 1 / 8) =
1.75$ bits/symbol for *all* $s$. Only the *conditional* structure moves with
$s$. At $s = 0.7$ (worked in `validate.py`):

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    stroke: 0.5pt + luma(180),
    table.header[context][$P(dot mid(|) "ctx")$ on its own move][$H(dot mid(|)
      "ctx")$],
    [after U], [$0.85$], [$0.835$ bits],
    [after D], [$0.775$], [$1.051$ bits],
    [after L], [$0.7375$], [$1.192$ bits],
    [after R], [$0.7375$], [$1.192$ bits],
  ),
  caption: [Per-context conditional entropies of the momentum rover at $s =
    0.7$. Strongly self-predictive after the common move U; least so after the
    rare moves L, R.],
)

The stationary average is the entropy rate
$ overline(H)(P) = 0.5(0.835) + 0.25(1.051) + 0.125(1.192) + 0.125(1.192) =
bold(0.978) "bits/symbol", $
against the marginal $bold(1.750)$. *The memory is worth $0.772$ bits per
symbol* — a $44%$ reduction that a memoryless spike code throws away and a
recurrent predictor recovers. This number is the case study's reason to exist,
and it is fixed before any circuit is built: it is the mutual information
$I(x_t; x_(t-1)) = H(pi) - overline(H)(P) = 0.772$ bits between a symbol and its
predecessor.

#remark[
  Sweeping $s$ traces the whole spectrum: $s arrow 0$ recovers a memoryless
  source (memory worthless, $0.772 arrow 0$); $s arrow 1$ approaches
  deterministic repetition (memory worth almost the full $1.75$ bits). The
  recurrent advantage is a *dial*, and we have its closed form.
]

= Holding context and choosing winners: the recurrent ingredients

The encoder of §5 needs two things from a recurrent network: a way to *hold* the
context $c_t$, and a way to *decide* which symbol was sent. Both are standard
circuit motifs, which we introduce before use.

#background("attractors and graded persistent activity")[
  A recurrent network's state moves under its own dynamics. A state the dynamics
  settle into and stay at is a *fixed point* or *attractor*; the set of starting
  states that flow into it is its *basin of attraction*. Most networks have
  isolated attractors, but a carefully tuned one can have a *continuous line (or
  ring) of fixed points* — a *line attractor* (*ring attractor*). The network
  can then rest *anywhere* along that line, holding a graded analog value
  indefinitely: *graded persistent activity*, a biologically observed substrate
  for short-term memory. This is how our loop stores the running context $c_t$
  without a fixed-size window — the position along the attractor *is* the memory.
]

#background("winner-take-all and divisive normalization")[
  A *winner-take-all* (WTA) network is a set of units that excite themselves and
  *inhibit each other* (lateral or contralateral inhibition). Started from
  graded inputs, its competition amplifies the largest and suppresses the rest,
  settling to a state where essentially one unit — *the winner* — is active. It
  implements selection / decision. *Divisive normalization* (Carandini–Heeger)
  is a companion canonical computation: each unit's response is divided by a term
  growing with the *total* pool activity, $r_i = a_i slash (sigma + sum_j a_j)$,
  so the responses share a bounded budget. It is a gain control — the neural form
  of forcing a set of numbers to behave like a probability distribution that
  sums to one.
]

These two motifs supply, respectively, the predictor's memory and the decoder's
choice. We use both in the next section.

= The circuit: predictor loop, calibrated readout, temporal-WTA decoder

We assemble the archetype. It has three stages; only the first contains a cycle,
and that is the point.

#figure(
  block(width: 100%, inset: 10pt, fill: luma(249), radius: 4pt)[
    #set text(size: 9.5pt, font: "DejaVu Sans Mono")
    #align(left)[
```
  symbol x_t ──►┌──────────────────────────────┐
                │  (1) RECURRENT PREDICTOR LOOP │   context state c_t
                │      line/ring attractor      │   (graded persistent
   feedback ◄───┤   holds running history;      │    activity = a running
   (predictive  │   lateral weights output      │    summary of the past)
    subtraction)│   q(·|c_t) as a drive pattern │
                └──────────────┬───────────────┘
                               │ drives R·I(q_i) = θ/(1−q_iᵅ)
                               ▼
                ┌──────────────────────────────┐
                │  (2) CALIBRATED READOUT       │   four LIF neurons race;
                │      ENCODER                  │   the true symbol's neuron
                │   t*(q_i) = −λ log₂ q_i       │   fires at latency = bits
                └──────────────┬───────────────┘
                               │ spike train (time axis = bit tape)
                               ▼
                ┌──────────────────────────────┐
                │  (3) TEMPORAL-WTA DECODER     │   first spike wins, names
                │   contralateral inhibition;   │   the symbol, resets the
                │   "exactly one winner" +      │   layer = self-clocking,
                │   basin volume ∝ q_i          │   prefix-free
                └──────────────────────────────┘
```
    ]
  ],
  caption: [The spiking entropy coder. Stage (1) is the only cyclic stage and is
    the only one that can perform predictive subtraction and hold unbounded-order
    context. Stages (2)–(3) are the CogSpike winner-take-all archetype,
    repurposed as a coder's encode/decode head.],
)

*Stage 1 — the predictor (the cycle).* A recurrent population holds a persistent
summary of recent moves as a bump on a line/ring attractor (§7). This *is* the
context variable $c_t$. Lateral projections turn $c_t$ into a pattern of
subthreshold depolarization on four move-neurons that encodes $q(dot mid(|)
c_t)$: after a U, the U-neuron is pre-charged near threshold, the others held far
from it. The feedback path performs the predictive subtraction of §2 and lets
the context depend on *arbitrarily deep* history rather than a fixed window.
*This is the stage a feedforward network structurally cannot replace.*

*Stage 2 — the encoder (calibrated readout).* The depolarization pattern sets
each readout's drive to $R I(q_i) = theta slash (1 - q_i^alpha)$ (@eq-calib), so
each neuron's first-spike latency equals its symbol's surprisal (Thm 1). The
realized symbol's neuron fires at $-lambda log_2 q(x_t mid(|) c_t)$ — the time
axis becomes the bit tape, and total transmission time is total information
(Thm 3).

*Stage 3 — the decoder (temporal WTA).* The first neuron to cross threshold
wins, *names the symbol by its identity*, and resets the layer — which restarts
the integration clock for the next symbol. This is the contralateral-inhibition
WTA archetype the program already verifies ("always eventually exactly one
winner"), here doing double duty: its *mutual exclusion* is unambiguous
decoding, and the reset makes the latency code *self-clocking and prefix-free*
(each spike delimits its own symbol, achieving by timing what a prefix code
achieves by the no-prefix rule of §4).

#remark[
  *The dictionary, in one table.*
  #table(
    columns: (auto, auto),
    align: (left, left),
    stroke: 0.5pt + luma(200),
    table.header[*Information-theory object*][*Spiking realization*],
    [symbol probability $q_i$], [drive $R I_i = theta slash (1 - q_i^alpha)$ on
      readout $i$],
    [surprisal $-log_2 q_i$], [first-spike latency $t^*_i slash lambda$
      (Thm 1, *exact*)],
    [entropy rate $overline(H)(p)$ (floor)], [min expected spike-time per
      symbol],
    [cross-entropy $overline(H)(p, q)$ (cost)], [actual spike-time emitted
      (Thm 3)],
    [$D_("KL")(p || q)$ (model error)], [excess spike-time = wasted joules
      (Thm 6)],
    [partition of $[0, 1)$ (Kraft)], [partition of attractor state into basins],
    [bigger interval $arrow$ likelier symbol], [bigger basin $arrow$ expected
      state],
    [prefix-free / self-delimiting], [first spike resets layer (self-clocking)],
    [the predictor $P(x_t mid(|) "history")$], [persistent loop state $c_t$
      (line attractor)],
    [normalization $sum_i q_i = 1$], [divisive gain control on the *sum mode*
      (§9)],
    [nearest-codeword decoding], [WTA settling to the nearest stored state],
  )
]

= Correctness via the sum/difference mode decomposition

The decode head is exactly the symmetric competitive network the program
analyzes with a *mode decomposition*, so the existing closed-form machinery
certifies the coder. We supply the dynamical-systems background, then read off
the two guarantees.

#background("bifurcations, and the sum/difference modes")[
  As a control parameter (here, the strength of the competition) crosses a
  threshold, a dynamical system can change the *number or stability* of its fixed
  points — a *bifurcation*. The relevant one is the *pitchfork*: below threshold
  a single symmetric state is stable (the units tie, no decision); above it the
  symmetric state goes unstable and two asymmetric stable states appear (one unit
  or the other wins). For a symmetric pair of competing units with activities
  $nu_1, nu_2$, it is natural to change coordinates to a *sum mode* $u = nu_1 +
  nu_2$ (total activity) and a *difference mode* $d = nu_1 - nu_2$ (who is
  ahead). Near the fixed point these decouple, so a two-unit competition splits
  into two scalar problems; the $N$-symbol case is handled mode-by-mode in the
  program's `closed_form_wta_multi` study.
]

#proposition[
  *Two invariants, two roles.*
  - The *difference mode* is *symbol selection*: its sign (argmax for $N > 2$) is
    which readout fires first, i.e. the decoded symbol. The pitchfork threshold
    is the decision boundary between symbols.
  - The *sum mode*, clamped by divisive normalization (§7), enforces $sum_i q_i =
    1$. The Kraft "the intervals must tile $[0, 1)$" rule of §4 becomes a
    *normalization invariant on the sum mode* — the partition-of-unity guarantee
    — and the sum-mode fixed-point analysis is exactly its proof.
]

#intuition[
  The two information-theoretic constraints land on the two mode families the
  program already studies. *Which* symbol is decoded is a difference-mode
  question (a pitchfork past a threshold). *That the probabilities sum to one*
  is a sum-mode question (a clamped total activity). So "decode correctly" and
  "the code is a valid probability partition" are not new proof burdens — they
  are the uniqueness/liveness and the fixed-point invariants the WTA archetype
  already carries, read through an information-theoretic lens.
]

The payoff is a clean separation: *correctness is unconditional, optimality is
earned.* Losslessness needs only that the WTA always elects exactly one winner —
a *structural* property of stage 3 that holds *no matter how poor the predictor
$q$ is.* A bad predictor makes the code *slow* (high cross-entropy, many joules)
but never *wrong*. The two verification obligations of §11 inherit this split.

= The model error, in joules

Make the energy reading literal. Let one spike cost $kappa$ (joules-per-bit, or,
in the latency code, seconds-per-bit $lambda$). Then per symbol:

#theorem("Excess energy equals KL divergence")[
  A circuit whose predictor is $q$ spends, per symbol, expected energy
  $ EE["cost"] = kappa dot overline(H)(p, q) = kappa(overline(H)(p) +
  overline(D)_("KL")(p || q)), $
  whose *irreducible* part $kappa overline(H)(p)$ is set by the source and whose
  *excess* $kappa dot overline(D)_("KL")(p || q) >= 0$ is paid purely for being
  wrong, vanishing iff $q = p$.
]

For the momentum rover this is concrete: a memoryless circuit pays $kappa(0.978
+ 0.772)$ — an extra $0.772 kappa$ joules per symbol, a $79%$ surcharge over the
floor, simply for failing to capture the memory. A circuit that learns the wrong
stickiness $s' = 0.4$ pays a smaller but nonzero $0.135 kappa$ surcharge
(`validate.py`).

#intuition[
  *This is the compression-as-intelligence claim as thermodynamics.* The only
  way for the circuit to lower its own metabolic bill on a *structured* stream is
  to lower $D_("KL")(p || q)$ — to make its internal model $q$ match the world's
  $p$. Energy minimization *is* model learning. A metabolically bounded recurrent
  circuit that minimizes its spiking is, provably, driven toward the true
  conditional distribution of its input: its energy budget is its loss function,
  and descending joules descends surprise.
]

= Verification obligations

The contribution to the formal-methods program is that every claim above is a
*checkable property*, and the properties fall into the two classes §9 separated.
The source is a discrete-time Markov chain — the native model of probabilistic
model checkers such as PRISM — and the coder is deterministic given the chain.
Companion templates are in `spiking_entropy_coder.pctl`; we write them in
PRISM's property syntax, where $P_(=1)[phi.alt]$ asserts a path formula
$phi.alt$ holds almost surely, $G$/$F$ are "globally/eventually", and
$R{"r"}_(=?)[dot]$ queries an expected reward.

*Structural / safety — hold for any predictor $q$.*
- *Mutual exclusion (unambiguous decode).* At most one winner per symbol:
  $ P_(=1) [ space G space (sum_i "win"_i <= 1) space ]. $
- *Liveness (decode terminates; the code is complete).* Every symbol elects a
  winner: $ P_(=1) [ space G space F space (sum_i "win"_i = 1) space ]. $
- *Losslessness.* The decoded symbol always equals the emitted one:
  $ P_(=1) [ space G space ("decoded" = "emitted") space ]. $
- *Prefix-freeness (structural).* Each winner resets the layer before the next
  integration window opens — a guard-level invariant of stage 3, needing no
  state-space search.

*Quantitative / optimality — predictor-dependent, the thing learning improves.*
- *Near-entropy cost.* With a per-symbol latency reward `"time"`, the expected
  steady-state cost meets the floor within tolerance:
  $ R{"time"}_(=?) [ space S space ] <= lambda (overline(H)(p) + epsilon). $
- *Monotone improvement under learning.* Any update lowering
  $overline(D)_("KL")(p || q)$ lowers the cost reward — the verifiable form of
  "the predictor is getting smarter."

#honest[
  The continuous-time latency reward is a quantity for a continuous-time or
  timed model (CTMC / probabilistic timed automaton); in a *discrete*-time chain
  one bins time and the reward becomes a spike-*count* proxy for $-log_2 q$,
  reintroducing exactly the timing-resolution penalty of the §5 honesty check.
  The count model is what the existing `contra_inhib_fixed.prism` already
  supports; the exact latency statement wants the timed extension. We state both
  so the gap is visible rather than papered over.
]

= The intelligence claim, made precise

Collecting the pieces, the slogan reduces to one precise sentence about this
circuit:

#block(width: 100%, inset: 10pt, fill: rgb("#f0f7ff"), radius: 4pt,
  stroke: (left: 3pt + rgb("#4a90d9")))[
  *Minimizing a verifiable spike-energy property* $R{"time"}_(=?)[S]$ *over a
  structured source provably drives the recurrent predictor* $q$ *toward the
  true conditional law* $p$ — because the only adjustable part of the cost is
  $overline(D)_("KL")(p || q) >= 0$ (Thm 6), uniquely minimized at $q = p$. The
  cycle is necessary, and its worth is the closed-form mutual information
  $I(x_t; x_(t-1))$ (§6). Correctness (§9) is independent of all this.
]

The circuit is therefore "intelligent" in an operational, checkable sense: it is
the member of its parametric family that spends the least energy on its input,
and spending the least energy is provably the same as predicting it best.

= Honest limitations

#honest[
  - *Learning is assumed, not derived.* We show that the *optimum* of the energy
    property is $q = p$ and that energy descent points there; we do not give the
    plasticity rule that performs the descent. A local predictive-coding rule
    (error-spike-gated adjustment of the feedback weights) is the natural
    candidate and the obvious next step.
  - *Attractor capacity bounds the memory order.* A finite line/ring attractor
    holds finite context; sources with very long memory exceed it, capping how
    far below $H(pi)$ the circuit can reach. The reachable floor is an attractor-
    capacity question — squarely within the `closed_form_wta_multi` and
    population toolkits.
  - *Wires versus time.* The race-to-threshold readout names the symbol by
    *which* wire fires, so its compression benefit is realized as *time/energy*
    per symbol, not as fewer channels. A single-line variant — one spike whose
    latency localizes a cumulative-probability interval, with the WTA doing the
    interval lookup (basin volume $prop q_i$) — is the route to true *bit-rate*
    optimality on one channel; we sketch it in §14 rather than claim it here.
  - *Noise.* Membrane noise blurs latencies; the §5 timing-resolution penalty is
    its quantitative footprint, and the door to the channel-coding thread below.
]

= Outlook: three threads worth pulling

*Lossy compression and rate–distortion.* Replace the hard WTA with a *graded*
WTA (a soft argmax via the sum-mode gain): the decoder then commits to a
*coarsened* symbol, trading reconstruction error for fewer spikes. The graded-
WTA bump width is the rate–distortion knob, and the mode decomposition gives its
closed form — a lossy spiking coder with a tunable, verifiable distortion bound.

*A realistic anchor: neuromorphic event streams.* The momentum rover keeps the
mathematics closed-form, but the real target is data that *is already spikes*: a
dynamic-vision-sensor (event-camera) stream, whose pixel events fire only on
brightness change. Such streams are spatiotemporally redundant (objects move
predictably), so a recurrent predictor that pre-charges the expected next events
and emits only residual spikes is a literal, hardware-native compressor — "spikes
as bits" with no re-encoding. The rover is the one-dimensional rehearsal for the
predictive subtraction an event-camera codec needs.

*Channel coding, where cycles become unavoidable.*
#background("error-correcting codes and iterative decoding")[
  This note did *source* coding — removing redundancy. *Channel* coding adds
  structured redundancy so a message survives a *noisy* channel. *Low-density
  parity-check (LDPC)* codes are decoded by *belief propagation*: an iterative
  message-passing procedure on a graph that, when the graph has cycles ("loopy"),
  is genuinely recurrent and approximately minimizes a quantity called the
  *Bethe free energy*. An attractor network settling to its nearest stored
  pattern is a physical instance of exactly this iterative, cyclic decoding.
]
On a noisy channel, recurrence stops being merely helpful (better prediction)
and becomes *provably necessary* (loopy belief propagation has no finite
feedforward equivalent). The same circuit that compresses on its forward pass
error-corrects on its settling pass — one recurrent substrate, both halves of
Shannon's program.

#v(0.5em)
#line(length: 100%, stroke: 0.5pt + luma(180))
#block(inset: (top: 4pt))[
  #text(size: 9pt, style: "italic")[
    Reproduce the numbers: `deq/.venv/bin/python research/compression/validate.py`.
    Property templates: `research/compression/spiking_entropy_coder.pctl`.
    Orientation: `research/compression/README.md`.
  ]
]
