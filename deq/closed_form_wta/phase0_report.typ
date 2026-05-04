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
- *Grid*: $12 times 12$ over scaled-integer pairs
  $(w_(12), w_(21)) in {-40, -36, ..., -2}^2$, finer than FCS's
  ${0, -10, -20, -30, -40, -infinity}$ axis labels.
- *Stimulus*: constant input = sequence of $1$s on both external lines.
- *Gate* (FCS Fig. 10): cell labelled *blue* if, in the post-warmup
  window $t in [4, 49]$, one neuron fires at rate $>= 0.99$ and the
  other at rate $<= 0.01$; *red* otherwise.

== Symmetry-breaker (caveat)

Our oracle is synchronous-parallel: at perfectly symmetric weights
$w_(12) = w_(21)$, both neurons follow identical trajectories and no
winner emerges, so all symmetric cells would label red. FCS's Lustre
encoding implicitly breaks ties via the language's variable evaluation
order. We mimic this by setting `initial_mem[i, 4] = -6` on the
disfavoured neuron, putting its $V(0) = 104$ at exactly $tau - 1$ so it
fails to fire at $t = 0$ while the favoured neuron fires (`V(0) = 110`).
Two variants are reported:

- *LUSTRE*: a single fixed bias (N1-favoured). FCS-faithful in the sense
  that Lustre also commits to one tie-breaking order. Asymmetric red
  zone (one arm only).
- *WTA_CAPABLE*: cell blue if either bias (N1-favoured or N2-favoured)
  yields WTA. Symmetric over $(w_(12), w_(21))$. The rate-equation-
  natural reading: "is bistable WTA *possible*?" Used as ground truth
  for Phase 1+ comparisons.

= Result

#figure(image("/deq/closed_form_wta/results/phase0/fcs_grid.pdf", width: 100%),
  caption: [Phase 0 reproduction. Left: LUSTRE variant
  ($136 / 144 = 94.4 %$ blue). Right: WTA_CAPABLE variant
  ($140 / 144 = 97.2 %$ blue). Both show red only in the weakest-
  inhibition corner; LUSTRE additionally has the asymmetric arm where
  $|w_(12)|$ is small but $|w_(21)|$ varies, because the fixed N1-bias
  cannot survive when N2 escapes weak $w_(12)$ inhibition.])

The four cells red in *both* variants are the corner
$(w_(12), w_(21)) in {-2, -5}^2$ — both inhibitions too weak for either
neuron to suppress the other regardless of initial bias. This matches
FCS Fig. 10's qualitative structure (red zone near the origin).

= Spot-check

At $w_(12) = w_(21) = -30$ with N1-favoured bias the FCS oracle
produces:

- N1 spike train: `0111111111111111...` (fires every tick from $t = 1$)
- N2 spike train: `0000000000000000...` (silent)
- Post-warmup rates: $nu_1 = 1.000$, $nu_2 = 0.000$. WTA-stable: yes.

Without the symmetry breaker, the same cell yields
$nu_1 = nu_2 = 0.348$ (synchronous oscillation, period $3$) — a
deterministic tie that FCS's Lustre also faces, and that motivates the
breaker.

= Verdict

*Phase 0 PASS.* Visual reproduction of FCS Fig. 10 confirmed; red zone
in expected weak-inhibition corner under both variants. Phase 1+ will
compare against the *WTA_CAPABLE* variant as the rate-equation-
consistent ground truth.

#v(0.6em)

#text(size: 9pt, fill: luma(80))[
  Output: `results/phase0/fcs_grid.{npz,pdf}`, `results/phase0/rate_diff.pdf`,
  `results/phase0.log`.
]
