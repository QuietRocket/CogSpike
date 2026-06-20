#import "../report_style.typ": *
#show: setup

#report_header("e07", "Predictive subtraction: the one operation that needs a cycle",
  "Tier 2 · circuit stage 1 + the sign · paper claim #5 + FIX F (inhibitory feedback)")

#claim[
  The recurrent loop cancels the *expected* drive via *inhibitory* feedback — *not*
  an additive depolarization "equal to the prediction" — so the soma integrates only
  the *unpredicted residual* $I_"eff" = I_"in" - I_"pred"$ (paper eq. predsub, FIX F).
  A perfectly predicted symbol meets balanced excitation $+$ inhibition: the soma sees
  $approx 0$ net drive and is *near-silent*; a surprising symbol leaves a large
  residual and *fires*. The emitted spike train stops being a copy of the symbol
  stream and becomes a *surprise* stream. The paper also flags the *shunting-balance
  idealization*: the exact additive $I_"in" - I_"pred"$ holds only when the inhibitory
  conductance tracks the excitatory drive linearly over the operating range.
]

= Why this is invisible in numpy

#intuition[
  The *sign* of the feedback is exactly the thing pure algebra cannot show. In numpy,
  `input - prediction` and `-(prediction - input)` are the same number; the minus sign
  is a bookkeeping choice. In a real spiking soma it is *physics*: the membrane is
  clamped at $V gt.eq 0$ (Nengo) and fires only when $V$ crosses $theta$. Subtracting
  the prediction drains $V$ toward a *sub-threshold* equilibrium — the neuron quiets;
  *adding* it doubles the drive and the neuron fires *earlier*. e07 is the experiment
  where the FIX-F sign becomes a measured fact about who spikes.
]

= What we built

#method[
  One rest-pinned LIF soma (`make_lif`, $V(0)=0$, $tau_(r c)=20$ ms), with *two*
  currents injected directly into `ens.neurons` (gain $1$, bias $0$): an excitatory
  sensory input $J_"in"$ and a feedback prediction current entering with a chosen
  *sign*, so the soma integrates $J_"eff" = J_"in" + "sign" dot J_"pred"$. $"sign" =
  -1$ is *inhibitory* feedback (subtraction, the correct FIX-F sign); $"sign" = +1$ is
  *additive* feedback (the wrong sign). The prediction is *matched to the model
  probability* $q$: $J_"pred" = q dot J_"in"$, so a perfect prediction ($q arrow 1$)
  cancels the whole excitatory drive (balanced E/I) and an unpredicted symbol ($q
  arrow 0$) cancels nothing. A separate builder integrates the *conductance-based
  shunt* $-g_"inh"(V - E_"inh")$ directly at the fine timestep for the idealization
  study. All code is `e07_predictive_subtraction/run.py`; $delta t = 10^(-4)$.
]

= Test 1 — the sign control (the FIX-F demonstration)

A symbol arrives with a fixed excitatory drive $J_"in" = 8 times$ rheobase; the loop
predicts it with probability $q$ and feeds back $J_"pred" = q J_"in"$ either
*inhibitorily* ($J_"eff" = J_"in" - J_"pred"$) or *additively* ($J_"eff" = J_"in" +
J_"pred"$).

#figure(image("results/e07_sign_control.pdf", width: 99%),
  caption: [*Left:* first-spike latency vs prediction quality $q$. Inhibitory feedback
    (blue) pushes the well-predicted symbols to *silence* (capped marker) as E/I
    balances; additive feedback (red) fires *earlier* than the un-cancelled baseline
    (grey dotted) for every $q$ — runaway, never cancelling. *Right:* the net soma
    current $J_"eff"$. Subtraction drives $J_"eff" arrow 0$ into the sub-threshold band
    (green); addition drives it up to $16 times$ rheobase.])

#finding[
  The two signs split cleanly. *Inhibitory:* a well-predicted symbol ($q gt.eq 0.95$)
  meets $J_"eff" = 0.4 times$ rheobase or less — *below threshold, so it never fires
  (silent)*. *Additive:* the same symbol meets $J_"eff" = 15.6 times$ rheobase and
  fires at $1.4$ ms — *earlier* than the un-cancelled baseline ($q=0$) latency of
  $2.70$ ms, and never silent at any $q$. Between the extremes the inhibitory latency
  rises monotonically with the prediction ($2.70 arrow 3.70 arrow 5.80 arrow 10.80
  arrow 35.90$ ms as $q: 0 arrow 0.85$, then silence), tracing E/I cancellation in real
  spike times; the additive latency *falls* ($2.70 arrow 1.30$ ms), the drive doubling.
  This is the FIX-F sign as a measured dichotomy: *subtract to quiet, add to run away.*
]

#figure(image("results/e07_voltage_traces.pdf", width: 80%),
  caption: [Three membrane traces for the *predicted-then-surprising* contrast. A
    predicted symbol ($q=0.95$) under *inhibition* (blue) charges to a sub-threshold
    plateau and never reaches $theta$ — silenced. A surprising symbol ($q=0.1$) under
    the same inhibition (green) keeps a large residual and crosses $theta$ — fires. The
    same predicted symbol under *additive* feedback (red dashed) is driven hard and
    crosses $theta$ first of all — the wrong sign turns a cancellation into an early
    spike.])

= Test 2 — the surprise stream: output tracks surprisal, not the symbol

Over a $120$-symbol momentum-rover stream ($s = 0.7$, seed $7$) with the *true*
predictor $q(x_t mid(|) c_t) = P_(c_t, x_t)$, each symbol arrives with $J_"in" = 6.5
times$ rheobase and is inhibited by $J_"pred" = q J_"in"$. We chose $J_"in"$ so the
rover's *most confident* move ($q_max = 0.85$) lands the residual $J_"eff" = J_"in"(1
- q_max) = 0.98 times$ rheobase — just *below* threshold — so the best-predicted
symbols are genuinely silenced while surprising moves keep a supra-threshold residual.

#figure(image("results/e07_surprise_scatter.pdf", width: 72%),
  caption: [Residual output (spike count per window) vs the symbol's surprisal $-log_2
    q(x_t mid(|) c_t)$, coloured by the *emitted symbol*. The output rises with
    *surprisal* (red fit, $r = +0.94$), not with the symbol identity (the four symbol
    colours interleave along the trend). Low-surprisal windows sit at *zero* output —
    the predicted symbols the loop has cancelled away.])

#finding[
  The residual spike count correlates with *surprisal* at $r = +0.942$, but with the
  raw symbol identity at only $r = +0.237$ (control). $37 slash 120 = 31%$ of windows
  are *silenced* — and these are exactly the well-predicted ones: their mean surprisal
  is $0.234$ bits, versus $1.453$ bits for the windows that fire. The emitted train is
  a *surprise* stream: the loop has subtracted away everything it expected, leaving
  spikes only where the world departed from the model. This is the qualitative claim of
  the paper's neuron section ("the spike train becomes a surprise stream") made a
  measured correlation.
]

#intuition[
  The correlation is high but not unity, and that is honest: the linear inhibition
  $J_"pred" = q J_"in"$ makes the residual $J_"in"(1-q)$, whose first-spike latency is
  *monotone* in $q$ but not the *exact* $-lambda log_2 q$ of the calibrated readout
  (e02). e07 tests the *sign and the cancellation* of the loop; the *exact* surprisal-
  as-time identity is e02's job. Here surprisal and residual output rise together — the
  mechanism is right — without claiming the calibrated identity a second time.
]

= Test 3 — the shunting-balance regime: where the linear idealization holds

The analyzed model treats inhibition as a clean *subtractive current*, residual $J_"in"
- J_"pred"$. Real synaptic inhibition is partly *shunting* (divisive): a conductance
$-g_"inh"(V - E_"inh")$ that, with $E_"inh" = 0$ at rest, turns the membrane into $tau
dot(V) = -(1 + g_"inh") V + J_"in"$ — rescaling *both* the steady state to
$J_"in" slash (1 + g_"inh")$ *and* the leak time constant to $tau slash (1 +
g_"inh")$. We pick the shunt that reproduces the *same sub-threshold steady state* as a
subtractive prediction, $g_"inh" = q slash (1 - q)$, so the two forms are *identical at
the equilibrium* (matched to $9 times 10^(-16)$) and every difference that remains is
purely *dynamical*.

#figure(image("results/e07_shunt_regime.pdf", width: 78%),
  caption: [Subtractive (blue) vs steady-state-matched shunting (red) first-spike
    latency, against prediction quality $q$. The steady states are *equal by
    construction*, yet the shunt fires *earlier* — and ever more so as $q arrow 1$ —
    because it also divides the time constant by $(1 + g_"inh")$. The relative departure
    (grey dotted, right axis) stays under $10%$ only in the small-conductance band
    (green) and climbs to $80%$ near balance.])

#finding[
  With the steady states matched exactly, the subtractive idealization is recovered
  *only* while the shunt conductance is small relative to the leak. The relative latency
  departure $|t_"shunt" - t_"sub"| slash t_"sub"$ is $lt.eq 0.098$ for $g_"inh" lt.eq
  0.10$ ($q lt.eq 0.09$), first exceeds $10%$ at $q = 0.15$ ($g_"inh" = 0.18$), and
  reaches $0.807$ near balance ($q gt.eq 0.80$, $g_"inh" gt.eq 4.1$): there the shunt
  fires at $7.4$ ms where the subtractive model takes $38.4$ ms. By $q = 0.88$ ($g_"inh"
  = 7.2$) the residual drops below rheobase and *both* forms fall silent. The "clean
  arithmetic $"input" - "prediction"$" is therefore an idealization with a *narrow*
  domain of validity — small conductance — exactly as the paper's honesty check states.
]

#honest[
  *The linear regime is genuinely narrow, and we report that rather than widen it.* The
  match $g_"inh" = q slash (1 - q)$ holds the *steady-state* residual identical, so the
  shunt and the subtractive current cancel the *same amount of drive*; the discrepancy
  is entirely the divisive *time-constant* rescaling, which speeds the membrane by a
  factor $(1 + g_"inh")$ regardless of how small the residual is. That is why even a
  modest prediction ($q = 0.15$, $g_"inh" = 0.18$) already shifts the spike time by
  $> 10%$. The paper carries the additive form (eq. predsub) forward as the *analyzed*
  model and flags the conductance-based implementation as the gap; e07 measures that gap
  and confirms it widens monotonically toward balance. The residual still *shrinks* with
  a good prediction under either form — the cancellation is real — but the spike *time*
  is only the clean subtractive value in the small-conductance limit.
]

= Acceptance

#accept_table((
  (true, [inhibitory feedback *silences* well-predicted symbols ($q gt.eq 0.95 arrow$ no spike)]),
  (true, [additive (wrong) sign *never* silences — no cancellation at any $q$]),
  (true, [additive sign fires *earlier* than baseline ($1.4$ vs $2.70$ ms — runaway)]),
  (true, [residual output tracks surprisal (positive corr, $r = +0.942 > 0.5$)]),
  (true, [residual tracks surprisal, *not* the raw symbol ($r_"sym" = +0.237 < r$)]),
  (true, [well-predicted windows silenced ($0.234$ bits), surprising windows fire ($1.453$ bits)]),
  (true, [subtractive idealization holds at small shunt conductance (departure $lt.eq 0.098$ at $g lt.eq 0.10$)]),
  (true, [E/I balance / linear idealization degrades near balance (departure $arrow 0.807$ at $g gt.eq 4$)]),
))

#finding[
  *8/8 passed.* The FIX-F sign is a measured dichotomy: inhibitory feedback silences
  the predicted ($J_"eff" arrow 0$, balanced E/I) and lets the surprising fire; the
  additive sign doubles the drive and runs away, never cancelling — a distinction
  *impossible* to see in the pure-algebra `input - prediction`. Over a rover stream the
  residual spike train becomes a *surprise* stream (output tracks surprisal at $r =
  0.94$, the symbol at only $0.24$). And the clean subtractive arithmetic is honestly
  bounded: matched to a divisive shunt at the steady state, it is the exact spike time
  only while the conductance is small relative to the leak, departing by up to $80%$ as
  the prediction approaches balance. This is the one stage where the *cycle* does work
  no feedforward unrolling reproduces, and the substrate confirms both its power and its
  idealization.
]
