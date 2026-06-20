#import "../report_style.typ": *
#show: setup

#report_header("e08", "The emission premise: ON/OFF fidelity vs the gradient identity",
  "Tier 2 · the paper's CENTRAL OPEN PROBLEM · empirically narrowing the conditional cost-spike = gradient identity")

#claim[
  The paper's headline novelty — the *cost-spike $=$ gradient* identity — is stated
  *conditionally*. The gradient *algebra* is unconditional,
  $ -partial ell_t slash partial W_(i j) = (1 slash ln 2) thin c_i (y_j - q_j), $
  but the *physical* identity requires the circuit to *emit exactly* the signed
  residual $r_j = y_j - q_j$. Since spikes are non-negative, the paper realizes
  $r_j$ by *two rectified ON/OFF error channels*
  $ r^+ = max(0, y - q), quad r^- = max(0, q - y), quad r = r^+ - r^-, $
  and *flags this as open*: "a circuit-level derivation (or refutation) is open."
  This experiment does *not* prove it. It *empirically characterises and narrows*
  it: how faithfully does a real spiking ON/OFF population emit $y - q$, and *how
  much emission fidelity does the gradient identity actually require* for learning
  to still reach $q = P$?
]

#honest[
  *What this experiment is, and is not.* It is *not* a circuit-level derivation of
  the emission premise — that remains open. It is a *characterisation*: a measured
  fidelity map of a real spiking ON/OFF population (part a), a measured
  corruption $arrow.r$ failure boundary for the learning rule (part b), and the
  *placement of the real emission point on that boundary* (the load-bearing
  result). Throughout we honour the paper's *distinct-observables* caveat — the
  error population is *not the same spikes* as the latency code (part c) — and the
  e07 *shunting-balance* regime, in which the clean "input $-$ prediction"
  arithmetic holds only on a matched timescale.
]

= What we built

Three parts, in one `e08_emission_premise/run.py`:

#method[
  *(a)* A *real two-channel ON/OFF spiking population* in Nengo: two rectified
  ensembles ($200$ LIF neurons each, intercepts $tilde.op cal(U)(0, 1)$, encoders
  $= +1$, radius $1.2$), so each represents *only the positive part* of its drive.
  $r^+$ is driven by $(y - q)$, $r^-$ by $(q - y)$; the *emitted* residual is
  $r_"emitted" = "decode"(r^+) - "decode"(r^-)$. We sweep $y in {0, 1}$ against
  $q in [0.05, 0.95]$ ($19$ points) and read the settled-tail rate decode.

  *(b)* A *fast numpy delta-rule learner* (lifting `learn_validate.py`'s online
  update $W_i arrow.l W_i + eta(e_j - "softmax"(W_i))$, constant
  $eta = 0.05$ — the e09/e10 spiking regime) whose emitted residual we *corrupt*
  three ways: ON/OFF *gain mismatch* $g = g_"off" slash g_"on"$ (sweep $1 arrow.r
  0$), rectification *dead-zone* $theta_r$ (sweep $0 arrow.r 0.5$), and *loop
  delay* $d$ (residual built from a per-context prediction stale by $d$ visits).
  For each level we run $40{,}000$ symbols to convergence and record the final
  excess KL.

  *(c)* A *separability* probe: the latency readout (calibrated drive $R I(q) =
  theta slash (1 - q^alpha)$, first-spike $t^* = -lambda log_2 q$) re-measured
  while a *separate* ON/OFF error population is driven to saturation, sharing *no
  neurons and no current path*.
]

= (a) The fidelity map: a real ON/OFF population emits $y - q$ faithfully, error at the kink

#finding[
  The spiking two-channel population emits the signed residual with a *root-mean-
  square error of $0.0051$* over the whole $(y, q)$ grid — a *faithful* emission.
  The two ON/OFF decode gains are *matched* to three decimals ($g^+ = 1.010$,
  $g^- = 1.011$; imbalance $0.001$), so the rectified-channel difference carries
  $y - q$ with essentially no systematic sign bias. The largest single-cell error
  is $0.0156$, at $(y = 1, q = 0.95)$ — i.e. $|y - q| = 0.05$, *right at the
  rectification kink*.
]

#finding[
  *The error is concentrated at the kink, exactly as the rectification picture
  predicts.* Stratifying by distance to the sign change $|y - q|$: the RMS error
  *near the kink* ($|y - q| <= 0.15$) is $0.0120$, while *far from it*
  ($|y - q| >= 0.5$) it is $0.0017$ — a *$7 times$* concentration. Where the signed
  residual changes sign, both rectified channels are near-silent and the
  finite-population decode of a small positive value is least accurate; away from
  the kink one channel carries a clean, well-decoded value. This is the physical
  fingerprint of half-wave rectification: the nondifferentiable point of
  $max(0, dot)$ is where a spiking realisation is least faithful.
]

#figure(image("results/e08_fidelity_map.pdf", width: 100%),
  caption: [*(a) Emission fidelity.* Left: the emitted residual (blue dots,
    spiking ON/OFF decode) tracks the ideal $y - q$ (red) for both $y = 0$ and
    $y = 1$ across $q$; RMS error $0.0051$. Right: the discrepancy
    $|r_"emitted" - (y - q)|$ vs distance from the rectification kink $|y - q|$ —
    the error spikes as $|y - q| arrow.r 0$ (RMS $0.0120$ near vs $0.0017$ far),
    the rectifier's nondifferentiable point.])

#intuition[
  The rectifier kink is where the two channels hand off. A residual of $+0.02$ asks
  the ON channel to emit a tiny positive value while the OFF channel sits silent;
  with positive-intercept tuning, very few neurons are even active there, so the
  rate decode of that small value is noisiest. The honest reading: the ON/OFF
  realisation is *most* faithful exactly where the residual is *large* (a confident
  error, far from the kink) — which is precisely where the learning rule wants the
  biggest, cleanest update. The fidelity is worst where the update should be small
  anyway.
]

= (b) The corruption $arrow.r$ failure boundary, and where real emission lands on it

#finding[
  *Clean baseline.* With the *uncorrupted* emitted residual the constant-$eta$
  delta rule converges to excess KL $= 0.0043$ bits/symbol ($max|q - P| = 0.049$)
  — the e09/e10 constant-learning-rate noise ball. We call a corrupted run *broken*
  when its excess KL exceeds $0.1043$ bits (clean $+ 0.10$).
]

#finding[
  *Two genuine boundaries are mapped, with critical values.*
  - *(i) ON/OFF gain mismatch.* Convergence survives down to $g_"crit" = 0.60$:
    even with the OFF channel at $60%$ of the ON channel's gain the rule still
    reaches the floor (excess $0.058$). Below that the asymmetric residual
    systematically biases the update, $q$ saturates, and the excess climbs steeply
    ($0.197$ at $g = 0.50$, $1.75$ at $g = 0.20$).
  - *(ii) Rectification dead-zone.* Convergence survives up to $theta_(r,"crit") =
    0.150$: a dead-zone that zeroes residuals smaller than $0.15$ still learns
    (excess $0.054$). Beyond it the *small* residuals that sharpen the conditional
    peaks never fire, so the peaks never sharpen — excess $0.153$ at $theta_r =
    0.20$, and $max|q - P|$ degrades from $0.067$ to $0.40$ as $theta_r arrow.r
    0.5$.
  ]

#finding[
  *(iii) Loop delay is benign at the working learning rate — an informative
  negative — and breaks only in a delay $times$ step-size interaction.* At the
  e09/e10 constant $eta = 0.05$, a per-context prediction stale by up to $d = 12$
  visits leaves the excess essentially untouched ($0.0043 arrow.r 0.0050$): between
  visits to a row the prediction barely moves, so a stale residual is almost a fresh
  one. Convergence breaks only when we *also* raise the step size to $eta = 0.6$,
  where a stale residual makes the row over- and under-shoot and oscillate —
  convergence then fails at $d_"crit"("fast") = 10$. This is exactly the paper's
  caveat (and e07's shunting-balance regime): the cancellation is clean only when
  *the prediction is delivered on the drive's timescale*; the failure is a
  delay$times$rate interaction, not delay alone.
]

#figure(image("results/e08_corruption_boundary.pdf", width: 100%),
  caption: [*(b) Corruption $arrow.r$ failure boundary.* Final excess KL vs each
    corruption; red dashed $=$ failure threshold ($0.104$), green dotted $=$ clean
    ball ($0.0043$). *(i)* gain mismatch breaks below $g_"crit" = 0.60$; the purple
    line marks the *real spiking* gain ratio $g = 1.00$ — far inside the converges
    region. *(ii)* dead-zone breaks above $theta_(r,"crit") = 0.150$; real spiking
    has $theta_r approx 0$. *(iii)* loop delay (log-$y$): benign at the working
    $eta = 0.05$ (blue), breaking only at the aggressive $eta = 0.6$ (orange,
    $d_"crit" = 10$).])

#finding[
  *The load-bearing result: real spiking emission falls firmly inside the
  converges region.* The measured ON/OFF gain ratio is $g_"real" = 1.000 >=
  g_"crit" = 0.60$. More directly, when we drive the *very same* delta-rule learner
  with the *actual measured spiking emission curve* $r_"emitted"(y, q)$ from part
  (a) — substituting the imperfect physical residual for the ideal $y - q$ at every
  update — it *still converges*: excess KL $= 0.0381$ bits/symbol, $max|q - P| =
  0.053$, comfortably below the $0.104$ failure threshold. The real emission's RMS
  error ($0.0051$) is *an order of magnitude smaller* than the smallest corruption
  that breaks learning (the dead-zone needs $theta_r > 0.15$; the gain needs $g <
  0.6$, a $40%$ imbalance).
]

#intuition[
  This is the answer to "how much emission fidelity does the gradient identity
  require." *Not much.* The identity is *robust* to realistic emission imperfection:
  it does *not* require the circuit to emit *exactly* $y - q$. A $40%$ ON/OFF gain
  imbalance, or a dead-zone swallowing residuals up to $0.15$, is tolerated; the
  real spiking population is well within both. The premise the paper flags as open
  is, *empirically, the easy direction* — physical emission is faithful enough that
  the algebra's fixed point ($q = P$) survives. What remains open is the *circuit-
  level derivation* that the ON/OFF channels emit $y - q$ in the first place; this
  experiment narrows *what fidelity that derivation must achieve* (a modest bar),
  it does not supply the derivation.
]

#honest[
  *We did not close the premise, and we are precise about what "robust" means.* The
  corruptions we inject are *parametric* (a gain ratio, a dead-zone width, a delay)
  — they model how an *imperfect* ON/OFF population would deviate, not whether such
  a population *exists* as a derived circuit. The real-emission point we place on
  the map is the *steady-state rate decode* of a hand-wired two-ensemble network,
  not a learned or derived predictive-coding microcircuit. The honest claim is
  bounded: *given* an ON/OFF realisation of the quality a standard Nengo population
  achieves, the gradient identity's learning consequence ($q arrow.r P$) is
  preserved. The existence/derivation of that realisation from the
  predictive-subtraction loop is the open problem, untouched.
]

= (c) The two channels are separable: zero cross-talk

#finding[
  *The latency code and the ON/OFF error channel are distinct observables, as the
  paper insists ("not the same spikes").* Driving a *separate* ON/OFF error
  population to *saturation* leaves the latency readout's first-spike time
  *bit-identical*: the maximum cross-talk over $q in {0.1, dots, 0.9}$ is
  $0.0000$ ms (mean $0.0000$ ms), to the $10^(-4)$ s timing grid. The
  latency$arrow.l.r$surprisal relationship is intact regardless of error-channel
  state: correlation $r = 1.0000$ with the error channel OFF and $r = 1.0000$ with
  it saturated.
]

#figure(image("results/e08_separability.pdf", width: 70%),
  caption: [*(c) Channel separability.* First-spike latency vs surprisal: the ideal
    $t^* = -lambda log_2 q$ (red), the latency code with the error channel OFF
    (blue circles), and with the error channel *saturated* (green crosses) — the
    points coincide exactly. Maximum latency cross-talk $0.0000$ ms.])

#intuition[
  The separability is *structural*, and that is the point. The latency code lives in
  the *timing* of one race-to-threshold readout; the error signal lives in the
  *rate* of a distinct ON/OFF population. They share no neurons and no current path,
  so saturating one cannot perturb the other — there is nothing to conflate. This is
  why the paper can foreground the cost-spike$=$gradient identity *as a conditional*
  without equivocation: the "cost" event (a latency spike) and the "gradient" event
  (an ON/OFF error spike) are *different physical events* that the identity *claims
  coincide in information content*, not in substrate. e08 confirms the substrates are
  genuinely independent.
]

= Acceptance

#accept_table((
  (true, [(a) spiking ON/OFF emits $y - q$ with RMS error $0.0051 < 0.12$]),
  (true, [(a) emission error largest near the kink (RMS $0.0120$ near $>$ $0.0017$ far)]),
  (true, [(a) ON/OFF decode gains matched (imbalance $0.001 < 0.15$)]),
  (true, [(b) clean baseline learns to the floor (excess KL $0.0043 < 0.06$)]),
  (true, [(b) gain-mismatch boundary mapped (critical $g_"crit" = 0.60 in (0, 1)$)]),
  (true, [(b) dead-zone boundary mapped (critical $theta_(r,"crit") = 0.150 in (0, 0.5)$)]),
  (true, [(b) *real* spiking emission converges (driven by $r_"emitted"$: excess KL $0.0381 <$ fail $0.104$)]),
  (true, [(b) real spiking gain ratio above critical ($g_"real" = 1.00 >= g_"crit" = 0.60$)]),
  (true, [(c) latency code unchanged by a saturated error channel (cross-talk $0.0000$ ms $< 0.2$ ms)]),
  (true, [(c) latency$arrow.l.r$surprisal correlation survives error activity ($r = 1.0000 > 0.99$)]),
))

#finding[
  *10/10 passed.* The experiment *narrows* the paper's central open premise without
  closing it. *(a)* A real spiking ON/OFF population emits the signed residual
  $y - q$ with RMS error $0.0051$, the error concentrated $7 times$ at the
  rectification kink — faithful where it matters (large residuals). *(b)* Learning
  tolerates substantial emission corruption — a gain imbalance down to $g = 0.60$
  and a dead-zone up to $theta_r = 0.15$ — and the *real* emission, whether placed
  by its gain ratio ($g = 1.00$) or by driving the learner with the *measured
  emission curve* (excess KL $0.0381$), lands firmly inside the converges region:
  *the gradient identity is robust to realistic emission imperfection and does not
  require exact $y - q$.* Loop delay is benign at the working learning rate, breaking
  only in a delay$times$step-size interaction ($d_"crit" = 10$ at $eta = 0.6$),
  consistent with e07's timescale-matching caveat. *(c)* The error population and the
  latency code are *distinct observables* with *zero* measured cross-talk, exactly as
  the paper's distinct-observables caveat requires. *What remains open* — untouched by
  this experiment — is the *circuit-level derivation* that the ON/OFF channels emit
  $y - q$ from the predictive-subtraction loop; e08 establishes only that the
  *fidelity bar* such a derivation must clear is modest, and a standard spiking
  population clears it.
]
