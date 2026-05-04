// Phase 1 report: Siegert (static) reading of FCS Property 7
#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 14pt, weight: "bold")[
    Phase 1 — Siegert (static) reading of Property 7
  ]
  #v(0.2em)
  #text(size: 10pt, style: "italic")[
    `deq/closed_form_wta/`, May 2026
  ]
]

= Goal

Read FCS Fig. 10's WTA boundary through the *static* lens of the
Siegert formula
$ nu = Phi(mu, sigma) = 1 / (tau_("ref") + tau_m sqrt(pi) integral_((V_r - mu)/sigma)^((V_("th") - mu)/sigma) "erfcx"(-u) d u). $
Apply the self-consistent fixed-point enumeration
$nu_i^* = Phi(mu_i(nu^*), sigma_i(nu^*))$ on the same $40 times 40$
integer grid Phase 0 scanned with the FCS oracle.

= Setup

- *Calibration*: $alpha = 0.250$, $beta = 4.29 dot 10^(-3)$,
  $tau_m = 2.35$, $tau_("ref") = 0.36$ -- locked from the prior closed-
  form thread (`deq/closed_form/results/phase1_grid.npz`); not
  recalibrated.
- *Self-consistency*: 1-D scalar reduction
  $r_1 - Phi(mu_1(nu_2(r_1)), sigma_1) = 0$ via brentq with $161$
  initial samples. Returns all roots (single FP, or three: the
  bistable case).
- *Label*: WTA-capable iff $exists$ FP with
  $|nu_1^* - nu_2^*| >= 0.30$, which captures both bistable and
  strongly-asymmetric monostable cells.

= Result

#figure(image("/deq/closed_form_wta/results/phase1/siegert_vs_fcs.pdf", width: 100%),
  caption: [Phase 1 vs Phase 0. Left: FCS oracle ($1014 / 1600$ blue).
  Middle: Siegert FP enumeration ($1482 / 1600$ blue). Right: cell-
  by-cell disagreement (black: Siegert says WTA, FCS says synchronous;
  orange: the reverse). The black region is exactly the diagonal-red
  staircase of Phase 0.])

#table(
  columns: 2,
  table.header([*Metric*], [*Value*]),
  [Siegert WTA-capable cells], [$1482 / 1600 = 92.6 %$],
  [FCS Phase 0 blue cells], [$1014 / 1600 = 63.4 %$],
  [Jaccard agreement], [$0.680$],
  [Recall (Siegert-blue $|$ FCS-blue)], [*0.996*],
  [Siegert-blue, FCS-red (staircase)], [$472 / 586 = 80.5 %$ of FCS-red],
  [Siegert-red, FCS-blue (boundary miss)], [$4 / 1014 = 0.4 %$ of FCS-blue],
)

The asymmetry of the disagreement is the central finding. Siegert
recovers $99.6 %$ of the FCS-blue cells (recall $approx 1$) but adds
$472$ false positives -- precisely the diagonal-red staircase
documented in Phase 0. The smooth-rate fixed-point structure cannot
distinguish "$nu_1^* != nu_2^*$ exists" (which holds across the
staircase too, as a bistable solution to the rate equations) from
"the integer-tick dynamics commit to that solution within $4$ ticks"
(which fails in the staircase, where small asymmetries get absorbed
into a synchronous oscillation period).

== Reading

Siegert is the *upper bound* on FCS-WTA: any cell that admits a
WTA-asymmetric stationary distribution under the rate equations is a
candidate for FCS-WTA stabilization. The integer-tick dynamics carve
out the diagonal staircase from this upper bound. Subsequent phases
fill the gap: Phase 2 uses $H(omega)$ to predict decision *latency*
(closing in on the $4$-tick gate as a continuous contour); Phase 3
uses quasi-renewal finite-$N$ noise to dissolve the staircase
(stochasticity breaks the synchronous lock).

= Verdict

*Phase 1 PASS* (recall $0.996 >= 0.95$ gate). Siegert is a high-recall
*superset* predictor of FCS-blue cells. The $472$-cell Siegert-blue /
FCS-red gap is precisely the integer-tick diagonal-staircase that
smooth-rate theory cannot resolve -- the meaningful complement that
Phase 2 ($H(omega)$ latency) and Phase 3 (finite-$N$ noise) fill.

#v(0.6em)

#text(size: 9pt, fill: luma(80))[
  Output: `results/phase1/siegert_grid.npz`,
  `results/phase1/siegert_vs_fcs.pdf`,
  `results/phase1/siegert_spread.pdf`,
  `results/phase1.log`.
]
