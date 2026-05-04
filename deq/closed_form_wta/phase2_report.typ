// Phase 2 report: H(omega) eigenvalue gate vs FCS Property 7
#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 14pt, weight: "bold")[
    Phase 2 — $H(omega)$ eigenvalue gate vs FCS Property 7
  ]
  #v(0.2em)
  #text(size: 10pt, style: "italic")[
    `deq/closed_form_wta/`, May 2026
  ]
]

= Goal

Test whether the rate-equation Jacobian eigenvalues at the Siegert
fixed points predict FCS's "stabilization within $4$ ticks" Boolean.
The hypothesis (A2):
$ "FCS-blue at" (w_(12), w_(21)) <==> |"Re"(lambda_("dom"))| > 1\/T_("FCS") = 1\/4 $
where $lambda_("dom")$ is the dominant eigenvalue of
$A = (1\/tau_m)(-I + "diag"(g) J)$, $J$ the contralateral connectivity in
Siegert units ($alpha dot W_("FCS")$), $g_i = partial Phi \/ partial mu$
at the operating point. If A2 holds, $H(omega)$ recovers FCS's
Boolean gate as a continuous $|"Re"(lambda)|$ contour.

= Setup

- *Calibration* (locked from `closed_form/phase1`):
  $alpha = 0.250$, $beta = 4.29 dot 10^(-3)$, $tau_m = 2.35$.
- *FP selection* per cell:
  - bistable ($3$ FPs): evaluate at the symmetric saddle (decision = saddle escape).
  - mono-stable asymmetric ($1$ FP, $|nu_1^* - nu_2^*| >= 0.30$): evaluate at the unique asymmetric FP.
  - mono-stable symmetric: no WTA possible, label red regardless.
- *Gate*: cell blue iff regime supports WTA *and*
  $|"Re"(lambda_("dom"))| > 0.25$.

= Result

#figure(image("/deq/closed_form_wta/results/phase2/h_gate_vs_fcs.pdf", width: 100%),
  caption: [Phase 2: 4-panel comparison. Left to right: FCS oracle,
  Phase 1 Siegert FP, Phase 2 $H(omega)$ gate, continuous
  $"Re"(lambda_("dom"))$ map. The H-gate version
  ($1191 / 1600$ blue) is qualitatively similar to Siegert ($1482 / 1600$)
  but does not recover the diagonal-staircase structure.])

#table(
  columns: 2,
  table.header([*Metric*], [*Value*]),
  [H-gate blue cells], [$1191 / 1600 = 74.4 %$],
  [Jaccard vs FCS at default gate $|lambda| > 0.25$], [$0.677$],
  [Best Jaccard over threshold sweep $[0, 2]$], [$0.687$ at $|lambda| > 0.20$],
  [Phase 1 Siegert Jaccard], [$0.680$],
  [*Improvement of $H(omega)$ over Siegert*], [*$-0.003$*],
  [Mean $"Re"(lambda)$ at FCS-blue stable FPs], [$-0.357$],
  [Mean $"Re"(lambda)$ at FCS-red stable FPs], [$-0.293$],
)

The negligible improvement is an *informative* result: it falsifies the
hypothesis that FCS-red cells in the diagonal staircase are simply
slow-decay cells. The eigenvalue distributions of FCS-blue and FCS-red
cells overlap heavily (mean separation $0.06$, with both standard
deviations well above this), so no $|lambda|$ contour separates them
cleanly. The staircase cells have well-separated rate-equation FPs
($|nu_1^* - nu_2^*| in [0.30, 0.50]$ typically) and respectable decay
rates ($-"Re"(lambda) in [0.20, 0.40]$); they are simply *not* near
bifurcation. The gap between Siegert/$H(omega)$ envelope and FCS reality
is *not* a smooth-rate timescale issue.

== Reading

This negative result is one of the *headline complementarity findings*
of the thread: the $H(omega)$ rate-equation linearization is the
correct dynamic complement to the Siegert static gain, and on the
negative-loop motif (companion `closed_form_note`) it predicts
oscillation frequencies to within $10degree$ phase agreement on FCS
LI&F. But on contralateral inhibition, where the failure mode is
*synchronous integer-tick locking* rather than rate-equation slow
dynamics, $H(omega)$ has no purchase. Phase 3 (quasi-renewal at finite
$N$) is the correct tool: stochastic finite-size noise is exactly what
the synchronous lock needs to break, so the staircase should shrink
under finite-$N$ noise.

= Verdict

*Phase 2 PARTIAL.* The gate metric ($H(omega)$ Jaccard $approx$ Siegert
Jaccard, improvement $-0.003$) shows that the integer-tick diagonal-
staircase artefact is *orthogonal* to rate-equation timescales, not
explained by linear-response slow decay. The result motivates Phase 3:
the staircase is a discrete-tick determinism artefact, and finite-$N$
stochasticity (quasi-renewal) is the right framework to dissolve it.

#v(0.6em)

#text(size: 9pt, fill: luma(80))[
  Output: `results/phase2/h_grid.npz`,
  `results/phase2/h_gate_vs_fcs.pdf`,
  `results/phase2.log`.
]
