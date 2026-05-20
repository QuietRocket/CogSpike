// Experiment A: dynamic-tau calibration via Bode sweep
#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 14pt, weight: "bold")[
    Experiment A — Dynamic τ calibration
  ]
  #v(0.2em)
  #text(size: 10pt, style: "italic")[
    `deq/closed_form_neg_loop/followups/`, May 2026
  ]
]

= Question

The parent study (Phase 2) found that single-pole $H(omega)$ with the
locked static calibration ($tau_m = 2.35$ ticks, fit on the
steady-state $f$–$I$ curve) over-estimates the negative-loop ringing
period by a constant factor of $approx 4$ at the FCS default cell
($T_("pred") = 15.92$ vs FCS period $4$). The hypothesis raised in the
parent note's §8 was that *re-fitting* $tau$ against a dynamic-response
experiment would close the gap. Experiment A tests this.

= Method

Drive an isolated FCS neuron (no recurrent connections) with
Bernoulli-modulated input

$ x(t) tilde "Bern"(p_("thin") dot (1 + epsilon sin(omega t))) $

at $p_("thin") = 0.7$ (calibration regime), $epsilon = 0.2$, weight
$w_("drive") = 11$. Sweep $omega$ over 12 frequencies in
$[0.05, 1.6]$ rad/tick. For each $omega$: simulate $T = 4000$ ticks,
smooth the spike train into a rate trace, fit cosine + sine
coefficients at $omega$ on the post-warmup window to extract output
amplitude $|R(omega)|$ and phase. Empirical transfer

$ |H(omega)| = |R(omega)| / (epsilon dot p_("thin")). $

Fit single-pole model $|H(omega)|^2 = g^2 / (1 + omega^2 tau^2)$ via
`scipy.optimize.curve_fit`. Recalibrate Phase 2 by rescaling
$T_("pred")$ proportionally to $tau$.

= Result

#figure(image("/deq/closed_form_neg_loop/followups/results/expA/bode_fit.pdf", width: 100%),
  caption: [Bode response of the isolated FCS neuron, AC modulation
  at $p_("thin") = 0.7, epsilon = 0.2$. *Left*: $|H(omega)|$ in log–log;
  blue dots are FCS measurements (averaged over 4 trials), purple
  curve is the single-pole fit, grey dashed is the static $tau_m$
  prediction. *Right*: phase. The fit gives $tau_("fit") = 5.02$ ticks
  with $R^2 = 0.97$ — *larger* than the static $tau_m = 2.35$, not
  smaller as the parent-note hypothesis predicted.]) <fig-bode>

#table(
  columns: (auto, auto),
  table.header([*Quantity*], [*Value*]),
  [Static $tau_m$ (from `closed_form_wta` calibration)], [2.35 ticks],
  [Dynamic $tau_("fit")$ (this experiment)], [5.02 ticks],
  [Ratio $tau_("fit") slash tau_m$], [2.13],
  [Fit $R^2$], [0.97],
  [DC gain $g_("fit")$], [1.02],
  [-3 dB frequency $1 slash tau_("fit")$], [0.20 rad/tick],
)

= The recalibrated $T_("pred")$ widens the gap

Rescaling Phase 2 by $tau_("fit") slash tau_m$:

#table(
  columns: (auto, auto, auto, auto),
  table.header([*Cell set*], [*Static $T_("pred")$*], [*Recalibrated $T_("pred")$*], [*FCS measured*]),
  [Default $(-11, 11)$], [15.92], [33.98], [4],
  [Mean over FCS-period-4 cells (498)], [13.6], [29.0], [4],
)

The recalibration moves $T_("pred")$ *further* from the FCS period,
not closer. The hypothesis that the factor-of-4 gap was a static-vs-dynamic
calibration mismatch is *falsified*.

#figure(image("/deq/closed_form_neg_loop/followups/results/expA/recal_T_pred.pdf", width: 100%),
  caption: [*Left*: recalibrated $T_("pred")$ heatmap (clipped at 8
  for visibility) — much of the grid now sits above 8 ticks. *Right*:
  scatter of $T_("pred")$ vs FCS-measured period. Purple dots
  (recalibrated, strict-P5 cells) lie *higher* than the orange-x
  markers (static, Phase 2 result), confirming the gap widens.]) <fig-recal>

= Why the hypothesis was wrong: a structural diagnosis

The Bode experiment reveals an important property of the FCS neuron's
linear response. At the FCS negative-loop oscillation frequency
$omega^star = 2 pi \/ 4 approx 1.57$ rad/tick, the empirical
$|H(omega^star)| approx 0.075$ — the neuron's *firing rate* is
essentially unresponsive to AC drive at that frequency. The Bode
rolloff begins around $omega approx 0.2$ rad/tick, an order of magnitude
below the closed-loop oscillation frequency.

So the closed-loop period-4 oscillation in the negative-loop motif
lives *deep in the rolled-off regime* of the single-neuron firing-rate
transfer function. *No* single-pole $H(omega)$ model — regardless of
how $tau$ is chosen — can predict period 4 oscillation because the
linearized firing rate has no resonance there. The factor-of-4 gap
is a *structural* property of single-pole linearization, not a
*parametric* one.

What sustains the FCS oscillation at $omega^star = 1.57$ rad/tick is
*spike-and-reset nonlinearity*: $A$ spikes when its windowed
integrator crosses threshold, resets, integrates again, spikes again
— a process that produces a deterministic limit cycle whose period
is set by the integrator's 5-tap geometry and the integer weight
balance, *not* by the firing-rate response time constant. Phase 3's
quasi-renewal mesoscopic captures this because its age-distribution
stepper explicitly encodes the reset; single-pole $H(omega)$ does
not, because there is no "reset" in the kernel $1 \/ (1 + i omega tau)$.

#block(
  width: 100%, inset: 9pt,
  fill: rgb("#fff6e8"),
  stroke: (left: 2pt + rgb("#c08020")),
)[
  *Reframed Phase 2 verdict.* The single-pole $H(omega)$ Jacobian
  eigenvalue $"Im"(lambda)$ is *not* a closed-loop oscillation
  frequency. It is a *fictitious linear-response ringing rate*
  derived under the assumption that the firing rate responds to AC
  drive perturbations linearly, which Experiment A shows breaks down
  by an order of magnitude at the relevant frequency. Phase 2's
  factor-of-4 is therefore the ratio of the linear-resonance pole to
  the actual nonlinear limit-cycle frequency, and that ratio is set
  by the FCS integrator geometry rather than by any single
  parameter.
]

= Verdict

*Experiment A FALSIFIES the calibration hypothesis.* Dynamic-response
fitting gives $tau_("fit") = 5.0$ (not the $tau \/ 4 approx 0.6$ that
would close the Phase 2 gap), and the recalibrated $T_("pred")$ is
*worse*, not better. The Phase 2 factor-of-4 is a *structural* sign
that single-pole linearization is inadequate for the negative-loop
closed-loop oscillation, regardless of the time-constant value.

This sharpens the parent note's three-lens reading: only the
quasi-renewal mesoscopic (Phase 3) can capture FCS's period-4
oscillation, because only it carries the spike-and-reset
nonlinearity. Static Siegert (Phase 1) and linear $H(omega)$
(Phase 2) are *fundamentally* the wrong tool for this kind of
question; refitting parameters won't rescue them.

= Caveats

- The Bode experiment uses Bernoulli-thinned input at $p_("thin") =
  0.7$ to match the calibration regime. The FCS negative-loop oracle
  uses constant input ($p_("thin") = 1$ effectively); the dynamic
  response there might differ. However, the *order-of-magnitude*
  conclusion — that $|H(omega^star)|$ is much less than 1 at the
  closed-loop oscillation frequency — is robust to such regime
  changes (a higher $p_("thin")$ would make the neuron more
  saturated, only narrowing the linear regime further).
- The single-pole fit captures the low-frequency rolloff well
  ($R^2 = 0.97$). Phase data at high frequencies are noisy due to
  the small AC amplitude there; we did not attempt a higher-order
  transfer function fit.

#v(0.6em)

#text(size: 9pt, fill: luma(80))[
  Output: `results/expA/{bode.npz, bode_fit.pdf, recal_T_pred.pdf}`,
  `results/expA.log`.
]
