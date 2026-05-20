// Phase 0 report: FCS-coordinate reproduction of Property 5 (negative loop)
#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 14pt, weight: "bold")[
    Phase 0 — FCS-coordinate reproduction of Property 5
  ]
  #v(0.2em)
  #text(size: 10pt, style: "italic")[
    `deq/closed_form_neg_loop/`, May 2026
  ]
]

= Goal

Reproduce De Maria et al. 2020 §6.2.5, *Property 5* (oscillation in a
negative loop): given two delayer-style neurons in a negative-loop
motif and a constant input sequence of $1$s, the activator $A$ fires
the periodic pattern $1100$ and the inhibitor $I$ echoes one tick
later. Run the FCS-accurate oracle
`deq/archetypes/lif_fcs.py:simulate` verbatim over a 2-D
scaled-integer grid in $(w_(I A), w_(X A))$, with $w_(A I) = 11$ fixed.

= Setup

- *Oracle*: `lif_fcs.simulate` with `tau = 105`,
  `rvector = [10, 5, 3, 2, 1]`.
- *Topology*: `topologies.negative_loop(w_XA, w_AI, w_IA)`. Two neurons
  $A$ (activator, index 0) and $I$ (inhibitor, index 1). $A$ receives
  external $X$ (weight $w_(X A)$) and inhibitory feedback from $I$
  (weight $w_(I A) < 0$). $I$ is excited by $A$ (weight $w_(A I) > 0$).
- *Grid*: $(w_(I A), w_(X A)) in {-40, ..., -1} times {1, ..., 40}$,
  $40 times 40 = 1600$ cells. $w_(A I)$ fixed at $11$ (FCS default).
- *Stimulus*: constant external input $X(t) = 1 forall t$.
- *Window*: $T_"max" = 64$ ticks, post-warmup window
  $t in [16, 63]$ (48 ticks $= 12$ candidate periods).

Two per-cell labels:

- *strict_p5*: $A$'s post-warmup spike train matches some cyclic
  rotation of `1100`, repeated.
- *broad_osc*: $A$'s post-warmup train has a regular period $p in [2,
  12]$ with mixed firing (at least one spike and one silence per
  cycle).

= Result

#figure(image("results/phase0/prop5_strict.pdf", width: 80%),
  caption: [Strict Property 5: cells whose activator spike train is a
  cyclic rotation of `1100`. Gold ring marks the FCS default cell
  $(w_(I A), w_(X A)) = (-11, 11)$ where Property 5 is canonically
  stated. *445 / 1600 = 27.8%* of cells satisfy the strict pattern.])

#figure(image("results/phase0/osc_broad.pdf", width: 80%),
  caption: [Broad oscillation: cells with any regular period
  $in [2, 12]$ and mixed firing. *946 / 1600 = 59.1%*. The complement
  consists of saturation cells (period 1, $A$ fires every tick — when
  external drive overwhelms inhibition) and silent cells (drive too
  weak to push $A$ above threshold).])

#figure(image("results/phase0/period_map.pdf", width: 80%),
  caption: [FCS-measured period of $A$ (grey = no regular period in
  $[2, 12]$, mostly saturation at $w_(X A) gt.eq |w_(I A)|$ where $A$
  fires every tick). The period-4 band (yellow-green) hugs the
  $w_(X A) tilde |w_(I A)|$ diagonal — the regime where the
  external drive is approximately balanced by the inhibitory feedback.
  Period 3, 5, 6 bands flank it.])

#table(
  columns: (auto, auto, auto),
  table.header([*Period*], [*Cells*], [*Reading*]),
  [1], [654], [$A$ fires every tick (drive overwhelms inhibition)],
  [2], [8], [co-fire / period-2 lock],
  [3], [149], [period-3 lock],
  [4], [498], [Property-5 territory (1100 family)],
  [5], [207], [longer cycles],
  [6], [84], [longer cycles],
)

= Sanity gate

FCS default cell $(w_(I A), w_(X A)) = (-11, 11)$ reproduces Property 5
verbatim:

$ A: 011001100110011001100110 dots quad I: 001100110011001100110011 dots $

so strict_p5 $= 1$, period $= 4$. *Phase 0 PASS.*

= Verdict

The negative-loop FCS oracle exhibits a coherent period-4 region
($w_(X A) approx |w_(I A)|$, the "balanced" diagonal) where Property 5
holds, flanked by other periodic regions (3, 5, 6) and by saturation
regions (period 1) where the inhibition is too weak to interrupt $A$.
Phase 1 will check whether Siegert rate theory predicts these regions
from fixed-point + Jacobian structure; Phase 2 will check whether the
$H(omega)$ ringing-period prediction $T_"pred" = 2 pi slash |"Im"(lambda)|$
lands at $4$ on the period-4 band; Phase 3 will check whether
quasi-renewal noise at finite $N$ sustains the ringing into a
stochastic limit cycle.

#v(0.6em)

#text(size: 9pt, fill: luma(80))[
  Output: `results/phase0/{fcs_grid.npz, prop5_strict.pdf,
  osc_broad.pdf, period_map.pdf}`, `results/phase0.log`.
]
