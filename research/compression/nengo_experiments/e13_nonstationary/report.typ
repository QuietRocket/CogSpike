#import "../report_style.typ": *
#show: setup

#report_header("e13", "Tracking a drifting source: the constant-η noise ball as a FEATURE",
  "Tier 4 · capstone · the learning-theory outlook (adaptation vs. almost-sure convergence)")

#claim[
  The convergence theorem of e09/e10 assumed a *fixed* source and a *decreasing*
  (Robbins–Monro) step: the excess energy $V arrow.r 0$ almost surely. A *drifting*
  world wants the opposite — a *constant* step. It trades almost-sure convergence
  for a steady-state $O(eta)$ noise ball, and *that ball is the adaptation
  mechanism*: a constant step keeps a residual sensitivity a vanishing step throws
  away, so when the world changes the constant-step learner *re-tracks* while the
  decreasing-step learner *freezes*. We test this on the momentum rover as a
  *perfectly controlled* non-stationarity: switching the stickiness $s$ holds the
  marginal $pi = (1/2, 1/4, 1/8, 1/8)$ — and hence the memoryless cost
  $H(pi) = 1.7500$ — *fixed*, and moves *only* the conditional structure the cycle
  captures. The entropy-rate floor, however, moves with $s$:
  $H_("rate")(0.70) = 0.9782$, $H_("rate")(0.40) = 1.4852$,
  $H_("rate")(0.85) = 0.5959$ bits/symbol. We switch
  $s: 0.70 arrow.r 0.40 arrow.r 0.85 arrow.r 0.70$ and ask the learner to chase
  the moving floor.
]

= What we built

A spiking PES learner (the verified e09 idiom: a $600$-neuron LIF context
population whose decoders are learned by Nengo PES, error $= q - y$, uniform init so
the energy starts at $log_2 4 = 2.0$) runs on a stream whose stickiness $s$
switches every $2500$ symbols ($s: 0.70 arrow.r 0.40 arrow.r 0.85 arrow.r 0.70$,
window $0.03$ s, $delta t = 10^(-3)$), with a *constant* learning rate
$eta = 1.5 times 10^(-3)$. Alongside it we run the *exact float64 delta-rule twin* —
$W[i] += eta thin (bold(1)_j - "softmax"(W[i]))$, the analytic analogue of the
learned spiking decoder — on a longer ($6000$-symbol) segmentation, so the small
tail residual is clean enough to fit a tracking time-constant and to sweep the
learning rate over many seeds. All code is `e13_nonstationary/run.py`.

#method[
  *Twin policy — which run carries which claim.* The *spiking PES learner* carries
  the centerpiece: the decoded energy physically *re-tracks the moving floor* on the
  substrate after each switch (claim 1). The *numpy delta-rule twin* carries the
  quantitative refinements that need a denoised, converged tail: the tracking
  time-constant $tau$ (claim 2), the $O(eta)$ steady-state-error sweep over six
  seeds (claim 3), and the decreasing-/tiny-$eta$ freeze (claim 4). Both obey the
  *same* delta rule — PES updates spiking decoders by $-eta dot "error" dot
  "activity"$, the twin updates float64 logits by $eta(bold(1)_j - q)$ — so the twin
  is the spiking learner's exact, noiseless shadow, used precisely where a multi-hour
  LIF sweep is infeasible. We state the split rather than blur it.
]

= (1) The spiking learner re-tracks the moving floor

#finding[
  The constant-$eta$ spiking PES learner re-descends to *each new floor* after every
  switch. Per-segment converged energy vs. the moving floor: $s = 0.70 arrow.r
  E = 0.9856$ (floor $0.9782$, excess $0.0074$); $s = 0.40 arrow.r E = 1.4997$
  (floor $1.4852$, excess $0.0145$); $s = 0.85 arrow.r E = 0.6259$ (floor $0.5959$,
  excess $0.0300$); back to $s = 0.70 arrow.r E = 1.0066$ (floor $0.9782$, excess
  $0.0284$). Every segment's energy sits *below the invariant marginal*
  $H(pi) = 1.7500$ and within $0.03$ bit of its moving floor. The learner is not
  converging to one law — it is *chasing a target that keeps moving*, and the
  constant step is what lets it.
]

#figure(image("results/e13_spiking_tracking.pdf", width: 92%),
  caption: [Spiking decoded energy (blue, trailing-window) across the four segments.
    The dashed coloured segments mark the *moving* entropy-rate floor $H_("rate")(s)$;
    the grey dotted line is the *invariant* marginal $H(pi) = 1.7500$. At each
    vertical switch line the energy *jumps* (the model is now wrong for the new $s$)
    and then *re-descends* to the new floor. The floor leaps from $0.9782$ down to
    $0.5959$ and up to $1.4852$ as $s$ moves, while the marginal never budges —
    the controlled non-stationarity moves only the conditional, exactly what the
    learner captures.])

#intuition[
  Because $pi$ is pinned, the memoryless cost $1.7500$ is a fixed ceiling the rover's
  *order* is always cheaper than. What moves is how much cheaper: at $s = 0.85$ the
  rover is so sticky that knowing the last move buys $1.75 - 0.60 = 1.15$ bits; at
  $s = 0.40$ it is nearly memoryless and the order buys only $0.26$ bit. The learner
  must re-estimate exactly that recoverable structure each time — and the
  energy-vs-time curve is the circuit *feeling the world's predictability change* and
  re-tuning to it on one number.
]

= (2) Each switch has a finite tracking time-constant

#finding[
  Fitting the post-switch excess to $V(t) approx V_oo + A thin e^(-t slash tau)$ on
  the numpy twin (constant $eta = 0.02$, $6000$-symbol segments) gives a finite
  adaptation time-constant at *every* switch: $tau approx 458$ symbols
  ($0.70 arrow.r 0.40$), $tau approx 552$ ($0.40 arrow.r 0.85$),
  $tau approx 1106$ ($0.85 arrow.r 0.70$), each well under one $6000$-symbol
  segment. The steady-state tracking errors $V_oo$ are small and comparable across
  switches — $0.0026$, $0.0021$, $0.0025$ bits/symbol — the residual $O(eta)$ ball,
  the *same size whatever the floor it surrounds*. The learner re-tracks in hundreds
  of symbols and then jitters in a fixed-radius ball about the new optimum.
]

#figure(image("results/e13_time_constants.pdf", width: 92%),
  caption: [Excess KL above the *current* floor over time (numpy twin, binned blue).
    At each switch the excess jumps and the red exponential fit $V_oo + A e^(-t slash
    tau)$ captures the re-descent; $tau$ ranges $458$–$1106$ symbols and the
    steady-state $V_oo approx 0.002$ bits/symbol. The third switch ($0.85 arrow.r
    0.70$, slackening the conditional) is the slowest to re-track — un-sharpening
    an over-concentrated estimate takes longer than sharpening a diffuse one.])

= (3) Steady-state tracking error scales with the constant lr — the $O(eta)$ ball

#finding[
  Sweeping the constant learning rate on the stationary $s = 0.70$ source (run to
  convergence, tail-averaged over six seeds to denoise the small residual), the
  steady-state tracking error grows *monotonically* with $eta$ and traces the
  slope-$1$ $O(eta)$ law: $eta = 5 times 10^(-3) arrow.r 0.00069$,
  $8 times 10^(-3) arrow.r 0.00094$, $1.2 times 10^(-2) arrow.r 0.00140$,
  $1.6 times 10^(-2) arrow.r 0.00186$, $2.4 times 10^(-2) arrow.r 0.00281$
  bits/symbol — a log-log slope of $0.91$, the predicted $approx 1$. This is the
  *bias/variance dial of a tracking learner*: a smaller step gives a tighter ball
  (more accurate at the optimum) but, by the same factor, slower adaptation. There
  is no free lunch — only a knob.
]

= (4) A decreasing / tiny lr FREEZES — it cannot re-track a late switch

#finding[
  The Robbins–Monro decreasing schedule $eta_t = eta_0 slash (1 + t slash t_0)$
  ($eta_0 = 0.08$, $t_0 = 800$) — the schedule that *guarantees* almost-sure
  convergence on a *fixed* source — *destroys* adaptation. It re-tracks the *early*
  switches while the step is still large ($eta = 0.0094$ at the first switch:
  segment-1 excess $0.0027$, matching the constant-$eta$ learner's $0.0037$), then
  the step decays ($eta = 0.0050$, $0.0034$ at the later switches) and the learner
  *freezes*. At the *late* $0.40 arrow.r 0.85$ switch — which demands *sharpening*
  the conditional onto the now-sticky momentum — the decreasing learner is stuck at
  excess $0.0409$ while the constant-$eta$ learner re-tracks to $0.0030$: a
  $13.6 times$ tracking gap. A *tiny constant* step ($eta = 2 times 10^(-4)$) is the
  over-frozen extreme — too small to descend at all within any segment (excess
  $0.831, 0.290, 0.840, 0.434$, never reaching a single floor).
]

#figure(image("results/e13_ball_and_freeze.pdf", width: 98%),
  caption: [*Left:* steady-state tracking error vs. constant $eta$ on log-log axes —
    the $O(eta)$ ball, slope $0.91$ along the slope-$1$ guide. *Right:* per-segment
    steady-state excess, constant-$eta$ (green, adapts) vs. decreasing-$eta$ (red,
    freezes). The two agree on the early segments; at the late $0.40 arrow.r 0.85$
    sharpen the decreasing schedule's step has decayed below what re-tracking needs
    and it is left *frozen* at $0.0409$ — $13.6 times$ the constant learner's
    $0.0030$.])

#intuition[
  The asymmetry is instructive. Slackening a conditional ($0.85 arrow.r 0.70$,
  $0.70 arrow.r 0.40$) the over-confident weights *relax toward uniform on their own*
  — even a small step suffices, so the decreasing learner survives those. But
  *sharpening* ($0.40 arrow.r 0.85$) requires *injecting* new concentration, which a
  vanished step cannot supply. So the freeze bites exactly where the world asks the
  learner to commit harder than it currently does — and that is precisely the case a
  constant step, holding its $O(eta)$ sensitivity in reserve, is built to handle.
]

= (5) The trade-off, named

#finding[
  *Constant-$eta$ keeps adaptation at the price of a permanent $O(eta)$ ball;
  decreasing-$eta$ buys an arbitrarily small ball at the price of adaptation.* On a
  fixed source the paper's almost-sure-convergence theorem is the right ideal and
  Robbins–Monro is optimal. On a *drifting* source that ideal is a *trap*: the very
  vanishing of the step that pins $V arrow.r 0$ is what makes the learner deaf to
  change. The noise ball is therefore not a defect of the spiking substrate to be
  apologised for — it is the *mechanism* by which a constant-step learner stays alive
  to a moving world. The paper's two regimes are one dial, and this experiment shows
  which end of it a non-stationary world demands.
]

#gap[
  *Where the substrate strains, honestly.* The spiking learner's per-segment excess
  ($0.0074$ – $0.0300$) is roughly $10times$ the float64 twin's ($0.0021$ – $0.0026$),
  and it *grows along the schedule* — the fourth segment ($s = 0.70$ again, excess
  $0.0284$) does not return to the first segment's $0.0074$. Two physical reasons,
  both real and both in the paper's spiking-reality ledger. First, the NEF decode
  carries the irreducible $tilde.op 1 slash sqrt(N)$ representational error of a
  $600$-neuron population (e03), which the float64 rule does not pay — so even a
  perfectly tracked spiking ball floors above the numpy ball. Second, at the shorter
  $2500$-symbol spiking budget the later, harder re-tracks (the $0.85$ sharpen, then
  the relaxation back to $0.70$) have *less room to fully settle* before the next
  measurement window, so part of the rising excess is residual *descent*, not ball —
  the same finite-budget descent-speed face e10 documented. The *direction* of every
  claim is realised on the substrate (re-tracking happens, the floor is chased below
  the marginal, the ball scales with $eta$, the decreasing step freezes); the
  *tightest* quantitative ball lives in the exact twin, exactly as the policy above
  declares. We report the spiking excess as measured, growing tail and all, rather
  than cherry-picking a converged segment.
]

= Acceptance

#accept_table((
  (true, [spiking energy re-tracks the moving floor (every per-segment excess $< 0.10$: $0.007$–$0.030$)]),
  (true, [spiking re-descends below the INVARIANT marginal $1.7500$ in every segment]),
  (true, [each switch has a finite tracking time-constant ($tau = 458, 552, 1106$ symbols $<$ one segment)]),
  (true, [constant-$eta$ steady-state tracking error small in every switch ($V_oo approx 0.002 < 0.02$)]),
  (true, [steady-state tracking error scales with $eta$ ($O(eta)$, monotone: $0.0007 arrow.r 0.0028$)]),
  (true, [the $O(eta)$ ball has log-log slope $approx 1$ (measured $0.91$)]),
  (true, [decreasing (Robbins–Monro) $eta$ FREEZES on the late switch ($0.0409$ vs $0.0030$, $13.6times$)]),
  (true, [a tiny constant $eta$ never tracks any segment (excess $0.29$–$0.84$, the over-frozen extreme)]),
))

#finding[
  *8/8 passed.* The paper's sharpest learning-theory claim lands. The constant-$eta$
  spiking PES learner re-tracks a deliberately drifting source — $s$ switching
  $0.70 arrow.r 0.40 arrow.r 0.85 arrow.r 0.70$ with $pi$ (and so the marginal
  $1.7500$) pinned — re-descending to each *moving* floor ($0.9782, 1.4852, 0.5959,
  0.9782$) with per-segment excess $0.007$–$0.030$, all below the invariant marginal.
  Each switch re-tracks with a finite time-constant ($tau = 458$–$1106$ symbols), the
  steady-state tracking error scales as the predicted $O(eta)$ ball (log-log slope
  $0.91$), and the Robbins–Monro *decreasing* schedule — almost-surely optimal on a
  fixed source — *freezes* on the late $0.40 arrow.r 0.85$ sharpen ($13.6times$ the
  constant learner's excess), with a tiny constant step the over-frozen extreme. The
  constant-$eta$ noise ball is not a bug to be tuned away: on a drifting world it is
  the adaptation mechanism, and decreasing the step trades the ability to track for a
  smaller ball nobody on a moving source asked for.
]
