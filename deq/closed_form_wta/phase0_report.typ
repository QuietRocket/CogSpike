// Phase 0 report: FCS-coordinate reproduction of Property 7
#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 14pt, weight: "bold")[
    Phase 0 — FCS-coordinate reproduction of Property 7
  ]
  #v(0.2em)
  #text(size: 10pt, style: "italic")[
    `deq/closed_form_wta/`, May 2026
  ]
]

= Goal

Reproduce De Maria et al. 2020 Fig. 10 — the verification of
*winner-takes-all* (Property 7) on the 2-neuron contralateral inhibition
motif — using the FCS-accurate oracle
`deq/archetypes/lif_fcs.py:simulate` over the same scaled-integer
grid in $(w_(12), w_(21))$ that FCS scanned by Lustre + Kind2.

= Setup

- *Oracle*: `lif_fcs.simulate` with `tau = 105`, `rvector = [10, 5, 3, 2, 1]`.
- *Topology*: `topologies.contralateral(w_12, w_21)` — two mutually
  inhibiting neurons each driven by external input `self_drive = 11`
  (delayer threshold).
- *Grid*: $40 times 40$ over scaled-integer pairs
  $(w_(12), w_(21)) in {-40, -39, ..., -1}^2$, matching FCS's visual
  resolution.
- *Stimulus*: constant input = sequence of $1$s on both external lines.
- *Initial condition*: zero `mem`, zero `localS_prev` --- *no symmetry
  breaker*. FCS's Lustre encoding has no implicit breaker either; at
  perfectly symmetric weights $w_(12) = w_(21)$, both neurons follow
  identical trajectories and lock into synchronous oscillation, which
  counts as no-WTA (red).
- *Gate* (FCS Fig. 10): cell labelled *blue* if, in the post-warmup
  window $t in [4, 49]$, one neuron fires at rate $>= 0.99$ and the
  other at rate $<= 0.01$; *red* otherwise.

= Result

#figure(image("/deq/closed_form_wta/results/phase0/fcs_grid.pdf", width: 75%),
  caption: [Phase 0 reproduction of FCS Fig. 10. $1014 / 1600 = 63.4 %$
  cells WTA-stable (blue); the rest red. The structure is *three diagonal
  red blocks in a sea of blue*, matching FCS's reported pattern.])

#table(
  columns: 3,
  table.header([*Block*], [*Range $(w_(12), w_(21))$*], [*Synchronous mode*]),
  [I (top-left)], [$|w| in [32, 40]$ both], [period-2 lock],
  [II (middle)], [$|w| in [13, 31]$ both], [period-3 lock],
  [III (bottom-right)], [$|w| in [1, 12]$ both], [co-firing / period-4 lock],
)

Each red block is a region where integer-tick FCS dynamics absorb small
asymmetries into the same synchronous firing period; the boundaries
between blocks are the discrete tick-count thresholds at which a
different period takes over. This staircase is the signature of the
*spike-timing lock* phenomenon documented in the population thread
as the failure mode of smooth Wilson--Cowan reduction. It is invisible
to any rate equation because it is a discrete-time integer artefact;
Phase 1 will quantify exactly how invisible.

= Spot-checks

- $(w_(12), w_(21)) = (-30, -30)$ [symmetric]: both N1 and N2 fire
  pattern `010010010010...` (period 3, rate $0.348$). No WTA. *Red.*
- $(w_(12), w_(21)) = (-5, -30)$ [asymmetric, weak $w_(12)$]: N1 fires
  only at $t = 1$ then silent ($nu_1 = 0$); N2 fires every tick from
  $t = 1$ ($nu_2 = 1$). Clean WTA, N2 wins. *Blue.*
- $(w_(12), w_(21)) = (-2, -2)$ [very weak both]: both fire pattern
  `010101010101...` (period 2, co-firing). No WTA. *Red.*

= Verdict

*Phase 0 PASS.* The diagonal-staircase three-block red-zone structure
of FCS Fig. 10 reproduces faithfully under the FCS-LI&F oracle with no
symmetry breaker. Phase 1+ will compare rate-equation predictors
(Siegert / $H(omega)$ / quasi-renewal) against this ground truth, with
a quantitative gap expected: smooth-rate theory cannot see the
diagonal-red staircase because it is an integer-tick artefact.

#v(0.6em)

#text(size: 9pt, fill: luma(80))[
  Output: `results/phase0/fcs_grid.{npz,pdf}`,
  `results/phase0/rate_diff.pdf`,
  `results/phase0.log`.
]
