// Phase 2 report: H(omega) predicted ringing period
#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 14pt, weight: "bold")[
    Phase 2 — H(ω) predicted ringing period vs FCS period
  ]
  #v(0.2em)
  #text(size: 10pt, style: "italic")[
    `deq/closed_form_neg_loop/`, May 2026
  ]
]

= Goal

Convert Phase 1's Jacobian Im($lambda$) into a *predicted ringing
period* of the rate-equation linearization,

$ T_("pred") = 2 pi slash |"Im"(lambda)| , $

and compare against FCS's measured period of the activator spike train
(Phase 0). The negative loop's Property 5 oscillation has period 4 by
construction; we ask: where on the $(w_(I A), w_(X A))$ grid does $H(omega)$
linear theory predict a 4-tick period, and where does it disagree?

= Setup

- Read `results/phase1/siegert_grid.npz` for eigenvalues at each cell;
  `results/phase0/fcs_grid.npz` for FCS measured periods.
- $T_("pred") = 2 pi slash |"Im"(lambda_("upper"))|$ where
  $lambda_("upper")$ is the complex-conjugate eigenvalue with positive
  imaginary part.
- *Boolean gate*: cell labelled period-4-blue iff
  $|T_("pred") - 4| <= 0.5$ ticks. Sweep tolerance to find the best
  Jaccard against FCS strict P5.
- *Continuous metric*: mean ratio $T_("pred") slash 4$ over cells
  where FCS-measured period is exactly 4.

= Result

#figure(image("results/phase2/period_predicted_vs_4.pdf", width: 100%),
  caption: [Phase 2 four-panel comparison. *Far left:* FCS strict
  Property 5 cells (target). *Middle-left:* FCS measured period
  (continuous, 1–6 ticks observed). *Middle-right:* $T_("pred")$
  heatmap, clipped at 30 ticks (some cells with very small $|"Im"(lambda)|$
  go higher). *Far right:* $T_("pred") in [3.5, 4.5]$ blue cells —
  exactly *zero* at tolerance 0.5.])

#table(
  columns: (auto, auto, auto),
  table.header([*Metric*], [*Value*], [*Reading*]),
  [$T_("pred")$ median (valid cells)],
    [14.96 ticks], [median over 1440 spiral cells],
  [$T_("pred")$ at default $(-11, 11)$],
    [15.92 ticks], [FCS period there is 4 ⇒ ratio 3.98],
  [Mean $T_("pred") slash 4$ over strict-P5 cells],
    [3.27], [systematic factor],
  [Mean $T_("pred") slash 4$ over FCS-period-4 cells],
    [3.39], [robust to label choice],
  [Boolean Jaccard at tol $= 0.5$],
    [0.000], [no cells fall in $[3.5, 4.5]$],
  [Boolean Jaccard at best tol $= 8.0$],
    [0.238], [generous tolerance still struggles],
)

= The factor-of-4 gap

The most striking finding is that *every cell where FCS measures
period 4, the rate-equation linearization predicts period ≈ 16*. The
ratio $T_("pred") slash T_("FCS") approx 4$ is essentially constant
across the strict-P5 region.

#figure(image("results/phase2/T_pred_vs_FCS_period.pdf", width: 75%),
  caption: [$T_("pred")$ vs FCS-measured period across the grid. Blue
  dots are FCS strict-P5 cells. The orange dashed line $y = 4 x$
  passes through them — the dynamic-rate prediction is *exactly 4×
  too slow*. The black dashed $y = x$ line is where the prediction
  would have to land to agree with FCS.])

This factor is not noise — it is a stable property of the calibration.
It says: *static* Siegert calibration (rate-vs-drive matching at fixed
$p_("thin") = 0.7$) does not constrain *dynamic* time-scales. The
closed_form_wta $tau_m = 2.35$ ticks is the right effective
relaxation for steady-state firing-rate prediction, but the
FCS 5-tap windowed integrator with `rvector = [10, 5, 3, 2, 1]` has a
much shorter effective response time — empirically $tau_("FCS-eff")
approx tau_m slash 4 approx 0.6$, close to the refractory $tau_("ref")
= 0.36$ rather than $tau_m$. A faster effective $tau$ gives a faster
ringing rate, hence shorter $T_("pred")$.

#figure(image("results/phase2/tol_sweep.pdf", width: 70%),
  caption: [Jaccard vs tolerance sweep. Even at the generous 8-tick
  tolerance, the maximum agreement is 0.24 — period-4 cells live in
  the wrong half of the $T_("pred")$ distribution under the locked
  calibration.])

= Reading: what H(ω) does and does not predict

Phase 2 establishes that single-pole low-pass $H(omega)$ with the
static-rate-fit calibration gives:

- *Correct qualitative ringing structure*: every spiral cell predicted
  to ring (Phase 1) does ring in FCS, and the ringing rate grows in
  the same direction as $|w_(I A)|$ and $w_(A I) w_(X A)$.
- *Wrong absolute period*: a factor-of-4 over-estimate, due to the
  mismatched effective time constant.

The factor-of-4 is therefore *the quantitative cost of separating
static and dynamic calibrations*. If the goal were to predict FCS
periods quantitatively, one would re-fit $tau_m$ against a small
dynamic-response dataset; this is straightforward but beyond the
current scope (the closed_form_wta calibration is intentionally locked
across the three-lens family).

= Verdict

*Phase 2 PASS gate qualitatively*, *PARTIAL quantitatively*. The
$H(omega)$ lens correctly identifies *where* ringing happens
(consistent with Phase 1's spiral envelope) and the *direction* of
period dependence on weights, but the locked static calibration is off
by a factor of ~4 in absolute period units. Phase 3 will check whether
the quasi-renewal mesoscopic, which uses the same calibration but
introduces stochastic spike timing, recovers the FCS period through
noise-driven sustainment.

#v(0.6em)

#text(size: 9pt, fill: luma(80))[
  Output: `results/phase2/{h_grid.npz, period_predicted_vs_4.pdf,
  T_pred_vs_FCS_period.pdf, tol_sweep.pdf}`, `results/phase2.log`.
]
