// Research Note II: Learning to Be Unsurprised
// A local plasticity rule that descends the spiking entropy coder's energy.
// Follow-up to spiking_entropy_coder.typ. Self-contained.
// CogSpike / research/compression, June 2026

#set document(
  title: "Learning to Be Unsurprised",
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

// Theorem-like environments (shared with note I)
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

#align(center)[
  #text(size: 16pt, weight: "bold")[Learning to Be Unsurprised]
  #v(0.2em)
  #text(size: 13pt)[A Local Plasticity Rule that Descends the Spiking Entropy
  Coder's Energy]
  #v(0.4em)
  #text(size: 10pt, style: "italic")[
    `research/compression/`, June 2026 · CogSpike research note II
    #linebreak()
    (follow-up to _Spikes as Bits: A Verifiable Spiking Entropy Coder_)
  ]
]

#v(0.5em)

#block(width: 100%, inset: 10pt, fill: luma(247), radius: 4pt)[
  *Abstract.* A companion note built a recurrent spiking circuit whose energy
  bill equals the information cost of its own predictions, but left one thing
  assumed rather than derived: the learning that makes the predictions good. We
  supply it. We show that gradient descent on the circuit's per-symbol energy —
  the surprisal $-log_2 q(x_t mid(|) "context")$ it pays in spikes — *is* a
  local, three-factor Hebbian rule of the form $Delta W_(i j) = eta dot c_i dot
  (y_j - q_j)$: presynaptic context $c_i$, times a postsynaptic residual $y_j -
  q_j$, times a global gate $eta$. The residual is exactly the *error spike* the
  encoder already emits by predictive subtraction — so the very spikes the
  circuit pays as its *code* are, in the same event, the *teaching signal* that
  makes it pay less next time. We prove the rule descends the energy: the excess
  energy (the model–world Kullback–Leibler divergence) is a Lyapunov function for
  the averaged dynamics, the loss is convex in the weights, and the unique
  fixed point is the true conditional law $q = p$. We find a structural bonus —
  learning is confined to the difference modes, leaving the sum-mode
  normalization invariant exactly preserved throughout — and we discharge the new
  verification obligations (locality, weight-boundedness, energy descent,
  convergence). A numerical run takes the circuit from $2.0$ down to $0.979$
  bits/symbol, the entropy-rate floor of the source. We close with the research
  programme this opens: eligibility traces for deeper memory, tracking of
  non-stationary sources, learned lossy quantization, and on-chip neuromorphic
  plasticity.
]

= Orientation: the one thing the first note assumed

The companion note _Spikes as Bits_ (in this directory) established a recurrent
spiking circuit — a *spiking entropy coder* — with a sharp property: the time it
spends spiking to emit a symbol equals that symbol's information content under
the circuit's internal model, so its total energy over a stream equals the
*cross-entropy* of its predictions against the source. The irreducible part of
that energy is the source's entropy; the avoidable part is the divergence
between the circuit's model $q$ and the world's law $p$, paid in joules. The
note's closing admission was explicit: it showed that the *optimum* of the
energy is $q = p$, but it did *not* give the rule that gets there. It assumed a
learner; it did not build one.

This note builds it, and the result is cleaner than expected. The learning rule
is not a separate apparatus bolted onto the coder. It is *the same spikes,
read again*: the residual (error) spike that the encoder emits as a symbol's
*cost* is, correlated with the context that predicted it, exactly the gradient
that lowers the cost. Energy, code length, loss, and the descent direction are
one object. The circuit that pays the most surprise is, in that very act,
teaching itself to be less surprised.

The plan, with background supplied before use: *§2* recaps — self-containedly —
the three results from the first note we build on, and fixes the predictor
model. *§3* gives the background a formal-methods reader needs on gradient
descent and on what makes a neural learning rule *local*. *§4* derives the rule
and proves the cost spike is the teaching signal. *§5* handles the one honest
wrinkle (residuals are signed; spikes are not). *§6* proves descent and
convergence via a Lyapunov argument and validates it numerically. *§7* identifies
the rule with four things already in the literature, so it is not a novelty out
of nowhere. *§8* writes the new verification obligations. *§9* states the unified
picture. *§10–11* are the research programme and honest limitations.

We again assume a reader fluent in *formal methods* (probabilistic model
checking, Markov models, temporal logic, safety/liveness, Lyapunov stability as
used in verification) and build the rest.

= Recap: the coder we are teaching

We restate the pieces we need so this note stands alone; full derivations are in
the companion.

#background("the source and the predictor model")[
  The source is a *first-order Markov chain* — the next symbol depends on the
  current one through transition probabilities $P_(i j) = Pr(x_(t+1) = j mid(|)
  x_t = i)$ — with long-run state frequencies $pi$ (the *stationary*
  distribution, $pi P = pi$). The running example is the *momentum rover* on
  ${U, D, L, R}$: $P_(i j) = s dot bb(1)[i = j] + (1 - s) pi_j$ with $pi = (1/2,
  1/4, 1/8, 1/8)$ and stickiness $s = 0.7$; it keeps the marginal entropy $H(pi)
  = 1.75$ bits but has *entropy rate* $overline(H)(P) = sum_i pi_i H(P_(i dot)) =
  0.978$ bits, the floor for a predictor that uses memory.
  The circuit's *predictor* is a recurrent population whose persistent state
  $c_t$ (a one-hot, or graded, summary of recent symbols — here the previous
  symbol) drives four readout neurons through lateral weights $W$. Writing the
  *logit* (net drive) of readout $j$ as $a_j = sum_i W_(i j) c_i$, the predicted
  probability is the *softmax*
  $ q_j = e^(a_j) / (sum_k e^(a_k)), $
  realized in the circuit by *divisive normalization* (gain control on the total
  activity — the "sum mode") dividing each neuron's exponentiated drive by the
  pool sum. With a one-hot context $c = e_i$ (last symbol was $i$), the logit is
  simply $a_j = W_(i j)$, so each context row $W_(i dot)$ is the circuit's
  belief about "what follows symbol $i$."
]

The three results we build on, restated:

#block(inset: (left: 6pt))[
  *(R1) Calibration.* Driving a readout for a symbol of model probability $q$
  with current $R I(q) = theta slash (1 - q^alpha)$ makes its first-spike latency
  exactly $t^*(q) = -lambda log_2 q$ — the symbol's surprisal.

  *(R2) Spike-time = cross-entropy.* Hence the expected per-symbol cost is the
  cross-entropy rate $overline(H)(p, q)$, equal to the entropy-rate floor
  $overline(H)(p)$ iff $q = p$.

  *(R3) Excess energy = KL.* The expected cost decomposes as
  $ EE["cost per symbol"] = kappa (overline(H)(p) + overline(D)_("KL")(p || q)),
  $ <eq-excess>
  for a substrate constant $kappa$; the avoidable surcharge is $kappa$ times the
  model–world divergence $overline(D)_("KL")(p || q) = sum_i pi_i sum_j P_(i j)
  log_2 (P_(i j) slash q_(i j)) >= 0$, zero iff $q = p$.
]

Our whole task is to drive @eq-excess's second term to zero, *locally and
online*, and to certify that we do.

= Background: gradient descent, and what "local" means for a synapse

#background("gradient descent and its stochastic form")[
  To minimize a differentiable function $V(W)$ of parameters $W$, *gradient
  descent* repeatedly steps downhill: $W arrow.l W - eta nabla V(W)$, with
  *learning rate* $eta > 0$. If $V$ is *convex* (curves upward everywhere) this
  reaches the global minimum. When $V$ is an average over data that arrives one
  sample at a time, *stochastic gradient descent* (SGD) replaces the full
  gradient with the gradient of a single sample's loss — a noisy but unbiased
  estimate. SGD converges almost surely to the minimizer of a convex objective
  provided the steps satisfy the *Robbins–Monro conditions* $sum_t eta_t =
  infinity$ (steps don't sum-out before arriving) and $sum_t eta_t^2 < infinity$
  (noise is eventually damped); a *constant* step instead converges to a small
  "noise ball" of radius $O(eta)$ around the optimum.
]

#background("Hebbian plasticity, locality, and three-factor rules")[
  In a neural circuit, a *synaptic weight* $W_(i j)$ couples a presynaptic
  neuron $i$ to a postsynaptic neuron $j$. A learning rule is *local* if its
  update to $W_(i j)$ uses only quantities physically available *at that
  synapse*: the presynaptic activity, the postsynaptic activity, and at most a
  broadcast global signal. *Hebbian* learning is the canonical local form —
  "fire together, wire together" — an update proportional to the *product* of
  pre- and postsynaptic activity, $Delta W_(i j) prop x_i^("pre") x_j^("post")$.
  A *three-factor* rule multiplies this Hebbian product by a third, global gate
  (e.g. a neuromodulator such as dopamine, or simply a learning-rate signal):
  $Delta W_(i j) = (#[gate]) dot x_i^("pre") dot x_j^("post")$. Locality matters
  for two reasons: it is what biology can plausibly implement, and it is what
  neuromorphic hardware can implement *on-chip* without shuttling global error
  vectors around. The rule we derive is exactly of this three-factor local form.
]

= The rule, derived: the cost spike is the teaching signal

We minimize the per-symbol energy. By (R2) the energy of emitting symbol $x_t$
in context $c_t$ is (up to the constant $kappa$) the surprisal
$ ell_t = -log_2 q(x_t mid(|) c_t) = -log_2 q_(x_t), quad q_j = e^(a_j) /
(sum_k e^(a_k)), quad a_j = sum_i W_(i j) c_i. $
We differentiate $ell_t$ with respect to the weights. Let $y_j = bb(1)[x_t = j]$
be the one-hot outcome (a spike on readout $j$ iff symbol $j$ occurred).

#theorem("Energy gradient is a local Hebbian product")[
  The gradient of the per-symbol energy with respect to the lateral weight
  $W_(i j)$ is
  $ (partial ell_t) / (partial W_(i j)) = 1 / (ln 2) dot c_i dot (q_j - y_j), $
  so the (negative-gradient) SGD update, absorbing $1 slash ln 2$ into the rate
  $eta$, is the *three-factor local rule*
  $ Delta W_(i j) = eta dot underbracket(c_i, "pre") dot underbracket((y_j -
  q_j), "post residual"), quad #[(global gate $eta$).] $ <eq-rule>
]

#proof[
  The softmax derivative is $partial a_k$-wise $partial q_m slash partial a_k = q_m
  (bb(1)[m = k] - q_k)$, giving the standard cross-entropy–softmax gradient
  $partial ell_t slash partial a_k = (q_k - y_k) slash ln 2$ (the $1 slash ln 2$ from
  bits versus nats). Since $a_k = sum_i W_(i k) c_i$ is linear in the weights,
  $partial a_k slash partial W_(i j) = c_i bb(1)[k = j]$, and the chain rule gives
  $partial ell_t slash partial W_(i j) = (q_j - y_j) c_i slash ln 2$. Descending this
  gradient (step $-eta'$, $eta = eta' slash ln 2$) yields @eq-rule.
]

Now read @eq-rule physically. The post factor $r_j := y_j - q_j$ is the
*residual*: actual outcome minus prediction. But that is *precisely* the
quantity the encoder already computes and emits. In the companion note, the
predictor pre-charges each readout with subthreshold depolarization equal to its
prediction $q_j$; when the true input $y_j$ arrives, predictive subtraction
leaves the readout firing only the residual $y_j - q_j$. That residual *is* the
emitted error spike — the thing whose latency or count encodes the symbol's cost.

#theorem("The cost spike is the teaching signal")[
  The negative gradient of the circuit's per-symbol energy with respect to the
  lateral weight $W_(i j)$ equals the presynaptic context activity $c_i$ times
  the residual error spike $r_j = y_j - q_j$ that the encoder emits for readout
  $j$. Hence the spikes the circuit pays as its *code* carry, in the same
  events, the complete first-order gradient of its own energy; a Hebbian
  correlation of those spikes with the context descends it.
]

#intuition[
  There is no separate "error backpropagated from a loss." The surprise the
  circuit suffers *is* the error signal. When symbol $j$ happens against a low
  prediction, its readout fires hard (large positive residual, large cost) — and
  that same hard firing, multiplied by the active context line, *potentiates*
  $W_(i j)$ so the symbol is predicted better next time. When a symbol was
  expected but did not occur, the prediction $q_j$ goes unmatched (negative
  residual) and $W_(i j)$ is *depressed*. The circuit pays surprise and, with
  the identical spikes, buys a smaller future surprise. Cost and learning are
  one event.
]

A first structural dividend, foreshadowing §6 and tying back to the companion's
mode decomposition:

#proposition[
  *Learning lives in the difference modes; the sum mode is invariant.* For each
  context $i$, $sum_j Delta W_(i j) = eta c_i (sum_j y_j - sum_j q_j) = eta c_i
  (1 - 1) = 0$. The total lateral weight out of each context — the "sum mode"
  that divisive normalization clamps to enforce $sum_j q_j = 1$ — is therefore
  *exactly conserved at every step*. Learning only moves predictive mass between
  symbols (the difference modes); it never disturbs the partition-of-unity
  invariant.
]

= The one wrinkle: signed residuals on a non-negative substrate

The residual $r_j = y_j - q_j in [-1, 1]$ is *signed*, but spikes are
non-negative. This is a real implementation question, with a standard and
honest answer.

#honest[
  Split the residual across *two rectified error channels*, as cortical
  predictive-coding microcircuits are independently argued to do:
  $ r_j^+ = max(0, y_j - q_j) quad ("error-ON: occurred more than predicted"),
  $
  $ r_j^- = max(0, q_j - y_j) quad ("error-OFF: predicted but absent"), $
  so $r_j = r_j^+ - r_j^-$, and the rule @eq-rule becomes $Delta W_(i j) = eta
  c_i (r_j^+ - r_j^-)$ — a *potentiating* contact driven by the ON channel and a
  *depressing* contact driven by the OFF channel. The cost is a doubling of the
  error population (a known feature, not a bug, of two-compartment / E–I
  predictive-coding models), and the prediction $q_j$ must be available at the
  synapse as the tonic depolarization it already is. We flag this rather than
  pretend a signed spike exists. Everything below treats $r_j$ as the signed
  difference these two channels jointly represent.
]

= Descent and convergence: an energy Lyapunov function

We now prove the rule does what it should: drive $q arrow p$ while monotonically
spending less energy.

#background("Lyapunov functions, for the descent argument")[
  To show a dynamical system flows to a desired state, exhibit a *Lyapunov
  function*: a scalar $V$ that is bounded below, is zero only at the target, and
  *never increases* along the dynamics ($dot(V) <= 0$). Then the state can only
  settle where $dot(V) = 0$. It is the continuous-time analogue of a loop variant
  in program verification — a quantity that provably decreases until the goal.
]

The natural candidate is the *excess energy* itself, from (R3):
$ V(W) = overline(H)(p, q) - overline(H)(p) = sum_i pi_i D_("KL")(P_(i dot) ||
q_(i dot)) >= 0, $ <eq-lyap>
which is zero iff $q_(i dot) = P_(i dot)$ for every visited context. Consider
the *averaged* (expected-over-data) update — the deterministic gradient flow SGD
follows in the small-step limit:
$ dot(W)_(i j) = EE[Delta W_(i j)] slash eta = pi_i (P_(i j) - q_(i j)), $
the expectation of @eq-rule using $EE[c_i] = pi_i$ and $EE[y_j mid(|) c = e_i] =
P_(i j)$.

#theorem("Excess energy is a Lyapunov function; learning converges to the truth")[
  Along the averaged flow $dot(W) = -eta nabla V$:
  + $V$ is *convex* in $W$ (cross-entropy of a softmax is convex in its logits,
    which are linear in $W$), so it has no spurious local minima;
  + $dot(V) = -eta norm(nabla V)^2 <= 0$, with equality iff $q_(i dot) = P_(i
    dot)$ for all $i$ with $pi_i > 0$;
  + hence $V arrow 0$ and $q arrow P$: the circuit converges to the *true
    conditional law*, and its energy descends monotonically to the entropy-rate
    floor.
  The minimizer is unique up to the softmax gauge (adding a constant to a context
  row $W_(i dot)$ leaves $q_(i dot)$ unchanged), which does not affect $q$.
]

#proof[
  Convexity: $-log_2 (sum_k e^(a_k))$-type log-sum-exp terms are convex in $a$,
  the map $W arrow a$ is linear, and $V$ is a non-negative combination of such
  terms plus $W$-independent constants; a non-negative combination of convex
  functions composed with a linear map is convex. Computing the gradient of
  @eq-lyap: only the $-sum_j P_(i j) log_2 q_(i j)$ part depends on $W$, and
  $partial (log_2 q_(i k)) slash partial W_(i j) = (bb(1)[k = j] - q_(i j)) slash ln 2$,
  so $partial V slash partial W_(i j) = -pi_i (P_(i j) - q_(i j)) slash ln 2$ — i.e.
  $nabla V prop -EE[Delta W]$, confirming the rule is gradient descent on $V$.
  Then $dot(V) = nabla V dot dot(W) = -eta norm(nabla V)^2 <= 0$, zero iff
  $nabla V = 0$, i.e. $q_(i j) = P_(i j)$ wherever $pi_i > 0$. By convexity this
  stationary point is the global minimum $V = 0$, reached as $t arrow infinity$.
]

#intuition[
  The thermodynamic statement of the companion note — *excess energy = model
  error* — is here promoted to a *stability certificate*. The joules the circuit
  wastes are not just a cost; they are the Lyapunov function whose dissipation
  *is* learning. "Spend less energy" and "predict the world better" are the same
  monotone descent, and (R3) is the receipt that they are the same number.
]

*Numerical validation* (`learn_validate.py`). Running the local rule @eq-rule
online on the momentum rover, from an ignorant uniform predictor:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (right, right, right, right),
    stroke: 0.5pt + luma(180),
    table.header[step $t$][energy $E(W)$ (bits)][excess $= overline(D)_("KL")$][$max_(i j) |q_(i j) - P_(i j)|$],
    [$0$], [$2.000$], [$1.022$], [$0.600$],
    [$10^3$], [$1.015$], [$0.037$], [$0.173$],
    [$10^4$], [$1.010$], [$0.032$], [$0.116$],
    [$10^5$], [$0.996$], [$0.018$], [$0.058$],
    [$10^6$], [$0.979$], [$0.0005$], [$0.021$],
    [$2 times 10^6$], [$0.979$], [$0.0004$], [$0.009$],
  ),
  caption: [Online local learning on the momentum rover ($s = 0.7$). Energy falls
    from $log_2 4 = 2.0$ bits (uniform) through $1.75$ (marginal-only) to the
    entropy-rate floor $overline(H)(P) = 0.978$ bits; the excess energy (average
    KL) and the model error both vanish. The gradient identity of Thm 1 is
    confirmed to $10^(-10)$; the averaged-flow energy of Thm 5 is verified to
    descend *strictly monotonically* (zero increases over $3000$ steps).],
)

The circuit teaches itself, with nothing but local correlations of its own cost
spikes, to spend $44%$ less energy — exactly the mutual information the cycle was
worth.

= What the rule already is: four identifications

The rule @eq-rule is not invented here; deriving it from spiking energy
*re-derives* four well-known objects, which is reassurance, not coincidence.

#background("the four, in one place")[
  *(a) The delta rule* (Widrow–Hoff): adjust a weight by presynaptic activity
  times the output error — @eq-rule verbatim. *(b) Rescorla–Wagner* conditioning:
  learning is driven by the *surprise* of the outcome, $Delta prop (lambda -
  sum V)$ — "strengthen the predictor of what happened, weaken predictors of what
  didn't," which is the sign pattern of $r_j$. *(c) Predictive coding*
  (Rao–Ballard): a hierarchy whose units carry prediction *errors* and whose
  synapses learn by correlating those errors with the predicting activity — our
  $c_i r_j$ exactly. *(d) The free-energy principle* (Friston): perception and
  learning both descend *variational free energy*, an upper bound on surprise;
  here the bound is tight and the free energy is literally $ell_t$, the spike
  cost.
]

The contribution is not a new rule but a new *grounding*: these four become a
single statement — *gradient descent on a spiking circuit's metabolic energy* —
with the error term identified as the emitted code, and with the descent
certified by the energy itself as Lyapunov function.

= New verification obligations

Note I split its obligations into structural/safety (predictor-independent) and
quantitative/optimality (predictor-dependent). Learning adds a third axis —
properties of the *weight dynamics* — and we again separate what model checking
certifies outright from what the Lyapunov analysis supplies. Templates extend
`spiking_entropy_coder.pctl`.

*Structural / safety (hold at every learning step, any data).*
- *Locality.* $Delta W_(i j)$ reads only $c_i$, $r_j$, and the global $eta$ — a
  guard-level property of the update, checkable by inspection (no non-local
  term appears).
- *Sum-mode invariance.* The partition of unity is preserved throughout
  learning: $ P_(=1) [ space G space (sum_j W_(i j) = "const"_i) space ]
  quad forall i $ (Prop. 1).
- *Weight boundedness / no runaway.* The averaged fixed point is $W_(i j) =
  log P_(i j) + "const"_i$; for an ergodic chain $P_(i j)$ is bounded away from
  $0$, so weights stay bounded — a stability/safety invariant: $ P_(=1) [ space
  G space (|W_(i j)| <= W_max) space ]. $

*Quantitative / limit (the Lyapunov payoff).*
- *Energy descent.* Under the averaged dynamics the expected per-symbol cost is
  non-increasing: $R{"energy"}$ at epoch $t+1 <= R{"energy"}$ at epoch $t$
  (Thm 5(ii)).
- *Convergence to the floor.* The cost reaches within $epsilon$ of the entropy-
  rate floor and stays there: $ P_(=1) [ space F space G space
  (R{"energy"} <= kappa(overline(H)(p) + epsilon)) space ]. $

#honest[
  Convergence and monotone descent are *analytic* (Lyapunov) facts about the
  continuous learning flow, not finite-state reachability queries; a model
  checker certifies the *structural* invariants above and the *converged*
  circuit's cost bound (against a frozen $W$), while the Lyapunov theorem
  certifies that learning *reaches* that frozen $W$. As in note I, the exact
  cost is a timed-model quantity; a discrete-time chain uses a spike-count proxy
  and inherits the timing-resolution penalty discussed there. We state the
  division of labour rather than overclaim that one tool does all of it.
]

= The unified picture

#block(width: 100%, inset: 10pt, fill: rgb("#f0f7ff"), radius: 4pt,
  stroke: (left: 3pt + rgb("#4a90d9")))[
  In this circuit, *four quantities are one*: the symbol's *code length*
  $-log_2 q$, the predictor's *loss* $ell_t$, the metabolic *energy* the spike
  costs, and — through @eq-excess and @eq-lyap — the *Lyapunov function* whose
  descent is learning. The emitted *error spike* is simultaneously the
  transmitted code and the gradient. So "compress better," "predict better,"
  "spend less energy," and "learn" are not four goals to be balanced; they are
  one monotone descent on one number, performed by a local rule the substrate
  can physically run.
]

= The research programme this opens

The rule is the first rung. Each limitation below names a concrete next step,
several reachable with the existing DEQ and verification toolkits.

*Deeper memory: eligibility traces and temporal-difference credit.*
#background("eligibility traces and TD learning")[
  The rule as derived uses a *first-order* context (the previous symbol). When
  the useful context spans several past symbols — held as a graded bump on the
  attractor — the residual at time $t$ must be credited to weights that were
  active *earlier*. The standard device is an *eligibility trace*: a fading
  memory $e_(i j) arrow.l gamma e_(i j) + c_i$ at each synapse, decaying with
  factor $gamma$, so the update $Delta W_(i j) = eta dot e_(i j) dot r_j$ assigns
  a late error to recently-active synapses. This is *temporal-difference*
  learning (TD($lambda$)); biologically the trace is a slow synaptic eligibility
  signal. The attractor's hold time and the trace's $gamma$ jointly set the
  reachable memory order — an attractor-capacity question for the
  `closed_form_wta_multi` / population toolkits.
]

*Non-stationary sources: tracking, and a regret bound.* A world whose statistics
drift needs a *constant* step $eta$ (decreasing steps freeze a stale model);
then $q$ tracks $p$ within an $O(eta)$ energy noise-ball, trading tracking lag
against steady-state excess. The clean target is a *regret* statement: total
excess energy over $T$ steps is $O(sqrt(T))$ against the best fixed predictor,
or $O(V_T)$ against a path of predictors of total variation $V_T$ — verifiable
online-learning guarantees attached to a spiking substrate.

*Learned lossy compression.* Note I sketched a *graded*-WTA decoder as a
rate–distortion knob; here the knob becomes *learnable*. Adding a distortion
penalty to the energy makes the same descent trade reconstruction error for
spikes, learning the *quantizer* (which symbols to merge) jointly with the
predictor — a spiking analogue of learned lossy codecs, with the distortion
bound following from the mode decomposition.

*On-chip neuromorphic plasticity.* Because @eq-rule is local and three-factor,
it is implementable in neuromorphic hardware *without* a global backward pass:
each synapse needs only its pre-trace, the local error spike, and a broadcast
gate. The energy interpretation gives such a chip a built-in, physically
meaningful objective — minimize your own spike energy — and the §8 invariants
give it a verification target.

*Meta-level: learning the calibration.* The constant $alpha$ that calibrates
latency to surprisal (R1), and the step $eta$, are themselves tunable; a
slow outer loop that adapts them to the source's timescale and volatility closes
the system into a self-calibrating coder. Whether that outer loop admits its own
Lyapunov certificate is open.

= Honest limitations

#honest[
  - *Convexity is in the logits, not in every parameterization.* We proved
    convergence for the softmax-over-lateral-weights predictor, where the loss is
    convex in $W$. A deep recurrent predictor (nonlinear in its parameters) loses
    global convexity; descent still holds locally but global convergence does
    not follow, and is the usual open question for nonconvex learning.
  - *Averaged versus stochastic.* Monotone descent is a property of the
    *averaged* flow (Thm 5); the online rule follows it only in expectation and
    settles in an $O(eta)$ noise ball (the numerics show $0.0004$ bits residual).
    The strict-monotonicity claim is for the gradient flow, not every SGD step —
    stated plainly to match the numerics.
  - *The signed-error doubling is a real cost.* §5's two rectified channels double
    the error population; whether a single mixed channel suffices (e.g. via a
    baseline-subtracted rate) is a substrate question we leave open.
  - *Credit assignment beyond first order is sketched, not proved.* The
    eligibility-trace extension above is standard but its interaction with the
    attractor's finite hold time — and hence the *certified* reachable memory
    order — is not yet worked out.
]

#v(0.5em)
#line(length: 100%, stroke: 0.5pt + luma(180))
#block(inset: (top: 4pt))[
  #text(size: 9pt, style: "italic")[
    Reproduce the numbers:
    `deq/.venv/bin/python research/compression/learn_validate.py`.
    Companion note (the coder this one teaches):
    `research/compression/spiking_entropy_coder.typ`.
    Property templates: `research/compression/spiking_entropy_coder.pctl`.
  ]
]
