// Experiment C: 3-neuron negative loop with one delayer
#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 14pt, weight: "bold")[
    Experiment C — 3-neuron negative loop (A → D → I → A)
  ]
  #v(0.2em)
  #text(size: 10pt, style: "italic")[
    `deq/closed_form_neg_loop/followups/`, May 2026
  ]
]

= Question

The negative-loop family includes "delayer-extended" variants
(DeMaria et al. 2020 Fig. 3): insert a delayer cell $D$ between the
activator $A$ and inhibitor $I$, giving the chain $A -> D -> I -> A$
(with $I$ inhibiting $A$). Does the three-lens framework generalize?
Specifically, does each lens predict the right period for the longer
loop?

= Method

*Topology* (defined inline in `expC_three_neuron.py`):

#align(center)[
  $W = mat(0, 0, w_(I A); w_(A D), 0, 0; 0, w_(D I), 0),
   quad B = vec(w_(X A), 0, 0)$
]

Default weights: $w_(X A) = w_(A D) = w_(D I) = 11$, $w_(I A) = -11$
(same canonical values as the 2-neuron motif). Same locked
calibration as the parent thread.

Run all three lenses at the default cell:
- *FCS oracle* (`lif_fcs.simulate`) for $T = 64$ ticks; measure
  period via 3-cycle tail-match detector.
- $w_(I A)$ *sweep* on $[-30, -1]$ with $w_(A D) = w_(D I) = 11$ fixed,
  to identify the period-stable region.
- *Siegert FP*: solve the 3-D self-consistency
  $bold(nu) = bold(Phi)(bold(mu)(bold(nu)), bold(sigma)(bold(nu)))$ for
  $bold(nu) = (nu_A, nu_D, nu_I)$.
- *Jacobian eigenvalues*: $A = (1 slash tau_m) (-I + "diag"(g) dot alpha W)$,
  $3 times 3$. Extract dominant complex pair.
- *$H(omega)$ T_pred*: $2 pi slash |"Im"(lambda_("dom"))|$ with
  static $tau_m$ (and recalibrated with $tau_("dyn")$ from
  Experiment A).
- *QR mesoscopic* at $N = 500$, $T = 400$.

= Result

#figure(image("/deq/closed_form_neg_loop/followups/results/expC/three_neuron.pdf", width: 100%),
  caption: [Four-panel summary. *Top-left*: FCS spike trains for $A, D, I$
  at default weights. $A$ pattern: `011100`, period 6 — *two ticks
  longer* than the 2-neuron's period 4. $D$ and $I$ trace the same
  pattern delayed by 1 and 2 ticks respectively (one for each
  excitatory link). *Top-right*: period vs $w_(I A)$ sweep — the
  3-neuron motif's period 6 region runs $w_(I A) in [-11, -6]$ with
  period 7 at $[-21, -12]$ and 8 at $[-30, -22]$. Heavier inhibition
  pushes the period longer; very light inhibition collapses the loop
  to period 2 (co-firing). *Middle-left*: 3 Jacobian eigenvalues —
  one real-negative $-0.807$ (fast decay mode), one complex-conjugate
  pair $-0.235 plus.minus 0.331 i$ (slow ringing mode). *Middle-right*:
  QR mesoscopic trace at $N = 500$ — clear period-$approx 6$
  oscillation. *Bottom*: bar chart comparing predictions across
  2-neuron and 3-neuron motifs.]) <fig-3n>

#table(
  columns: (auto, auto, auto, auto),
  table.header(
    [*Prediction*],
    [*2-neuron value*],
    [*3-neuron value*],
    [*Tracks FCS?*],
  ),
  [FCS-measured period], [4], [*6*], [—],
  [Siegert FP $nu^star$],
    [$(0.35, 0.18)$], [$(0.43, 0.26, 0.08)$], [—],
  [Jacobian eigenvalues],
    [$-0.43 plus.minus 0.39 i$],
    [$-0.81; -0.23 plus.minus 0.33 i$], [—],
  [$H(omega)$ static $T_("pred")$],
    [15.92], [19.01], [*No (×3.2)*],
  [$H(omega)$ recal $T_("pred")$ (with $tau_("dyn")$ from Exp A)],
    [33.98], [40.58], [*No (worse)*],
  [QR period (mesoscopic)],
    [4.05 ($N = 2000$)], [*6.38* ($N = 500$)], [*Yes (±1.5)*],
)

= Findings

*F1: FCS gives period $4 + 2 = 6$ for the +1-delayer extension.* Each
extra cell in the loop adds 2 ticks to the period (one for the
excitatory link's spike-emission delay, one for the integration step
between threshold-crossings). The 2-neuron motif has
$"period" = 4 = 2 + 2$ (one A-to-I delay, one I-to-A delay); the
3-neuron has $"period" = 6 = 2 + 2 + 2$. This is a clean linear
scaling that anticipates the general n-delayer FCS Fig. 3 series
($"period" = 2 (n + 1)$ for $n$ delayer cells with default integer
weights).

*F2: Single-pole $H(omega)$ over-estimates the period again, by a
slightly different factor.* The 2-neuron factor was 16/4 = 4.0; the
3-neuron factor is 19/6 ≈ 3.2. The factor is *not* constant across
motifs, indicating it depends on the specific Jacobian eigenvalue
structure rather than being a universal calibration scaling. As in
Experiment A, recalibrating with $tau_("dyn") = 5.02$ widens the gap
rather than closing it (40.58 vs FCS 6).

*F3: Quasi-renewal recovers the FCS period in the 3-neuron motif
too.* QR at $N = 500$ gives FFT-dominant period 6.38, matching FCS's
6 within ±1.5 ticks. The same mechanism that worked in Phase 3 of
the main study (age-distribution + spike-reset) carries to longer
loops. *The three-lens framework generalizes.*

*F4: The $w_(I A)$ sweep reveals a period-progression structure.* As
$|w_(I A)|$ grows, the period steps up: 6, 7, 8 in discrete plateaux.
This is the discrete-tick analog of inhibition strength setting the
recovery time after a co-fire event. Below $|w_(I A)| approx 5$, the
loop loses oscillation (period 2 = co-fire lock).

= Verdict

*Experiment C PASS.* The 3-neuron negative loop (A → D → I → A)
gives FCS period 6, the QR mesoscopic predicts 6.38, the
$H(omega)$ linear lens again over-estimates with a similar
structural factor (~3.2). The three-lens machinery generalizes
cleanly: each lens's diagnostic role from the 2-neuron study
carries over.

This corroborates the parent note's bottom line: nonlinear
refractoriness (quasi-renewal) is the right tool for FCS period
prediction across the negative-loop family; single-pole linear
$H(omega)$ is structurally inadequate at any length.

= Open follow-ups (for n-delayer scaling)

We did not run the full $n$-delayer scan (n = 2, 3, ...). Based on
F1's linear-scaling observation, we conjecture FCS period
$= 2 (n + 1)$ for the n-delayer extension with default integer
weights (n = 0 gives period 2 for a single neuron with self-loop;
n = 1 = 2-neuron motif gives 4; n = 2 = this experiment gives 6).
Verification across n in {2, 3, 4, 5} would complete the scaling
story; it is omitted here for scope.

#v(0.6em)

#text(size: 9pt, fill: luma(80))[
  Output: `results/expC/{three_neuron.npz, three_neuron.pdf}`,
  `results/expC.log`.
]
