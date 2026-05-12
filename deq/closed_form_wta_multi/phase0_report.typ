// Phase 0 (multi-N): FCS oracle on uniform all-to-all inhibition.
#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 14pt, weight: "bold")[
    Phase 0 (multi-N) --- FCS oracle on uniform inhibition
  ]
  #v(0.2em)
  #text(size: 10pt, style: "italic")[
    `deq/closed_form_wta_multi/`, May 2026
  ]
]

= Goal

Reproduce the FCS Property 7 question (winner-takes-all stability within
4 ticks) at $N > 2$ neurons by sweeping the FCS-LI&F oracle over uniform
all-to-all lateral inhibition. The 2-neuron prequel
(`deq/closed_form_wta/phase0_fcs_baseline.py`) swept the $(w_(12), w_(21))$
plane on $40 times 40$; this thread squashes the weight space to one
scalar $w$ (shared across all edges) and adds $N in {2, 3, 4, 6, 10}$ as
the second axis. Two `drive_bump` values $in {0, 1}$ control the
$S_N$-symmetry breaker that the FCS Lustre semantics otherwise lacks.

= Setup

- *Oracle*: `lif_fcs.simulate` with $tau = 105$, `rvector = [10, 5, 3, 2, 1]`.
- *Topology*: `topologies.all_to_all_inhibition(N, w, drive_bump)`. Every
  neuron self-drives with `self_drive` $= 11$ (delayer threshold) and
  inhibits every other neuron with scaled-integer weight $w$. `drive_bump`
  is added to neuron $0$'s self-drive only.
- *Sweep*: $w in {-40, -39, ..., -1}$; $N in {2, 3, 4, 6, 10}$; `drive_bump`
  $in {0, 1}$. 400 cells total.
- *Stimulus*: constant $1$ on every external line.
- *Initial condition*: zero `mem`, zero `localS_prev` --- FCS-faithful.
- *Strict gate*: post-warmup ($t in [4, 49]$) `rate_max` $>= 0.99$ AND
  `second_max` $<= 0.01$.
- *Margin gate*: post-warmup `rate_max - second_max` $>= 0.30$.

The strict gate is the direct $N$-generalization of the 2-neuron FCS Fig. 10
gate (`rate_max` $>= 0.99$ AND `rate_min` $<= 0.01$). The margin gate is
softer and aligns with the gates of Phase 1 (Siegert FP spread) and
Phase 3 (QR rate spread).

= Result

#figure(
  image("results/phase0/fcs_grid_multi.pdf", width: 95%),
  caption: [Phase 0 FCS oracle WTA labels across $(w, N, "drive_bump")$.
  Left column: strict gate. Right column: margin gate. Each panel: two
  rows of dots for `drive_bump` $in {0, 1}$; columns are $w$.]
)

== Headline counts (blue cells per $N$, `drive_bump` $= 1$)

#table(
  columns: 4,
  inset: 6pt,
  table.header([*$N$*], [*Strict*], [*Margin*], [*Notes*]),
  [2], [9 / 40], [10 / 40], [Matches 2-neuron diagonal scan (red staircase + asymmetric-breaker WTA arms).],
  [3], [4 / 40], [9 / 40], [Strict gate misses several near-WTA cells where the winner fires at rate $approx 0.98$ but `second_max` $> 0.01$.],
  [4], [2 / 40], [6 / 40], [WTA harder: integer-tick lock collapses into period-$N$ round-robin at most $w$.],
  [6], [0 / 40], [8 / 40], [Strict gate fails entirely; margin gate recovers a pocket of WTA cells at intermediate $|w|$.],
  [10], [0 / 40], [3 / 40], [Strict fails; margin retains only a thin band near $|w| approx$ mid-range.],
)

At `drive_bump` $= 0$ every cell is red under both gates --- the
FCS Lustre semantics with symmetric weights and zero initial mem keeps
all $N$ neurons in identical synchronous trajectories. This generalizes
the 2-neuron diagonal-staircase phenomenon: at every $N$, perfect $S_N$
symmetry forces synchronous lock (the staircase becomes a *plane*).

== Spot-checks at `drive_bump = 1`, $w = -30$

#table(
  columns: 3,
  inset: 6pt,
  table.header([*$N$*], [*Post-warmup rates*], [*Pattern*]),
  [2],  [`[0.35, 0.35]`], [Period-3 lock; no WTA.],
  [3],  [`[0.98, 0.00, 0.00]`], [Near-clean WTA --- winner fires almost every tick. Strict gate misses by $0.01$.],
  [4],  [`[0.20, 0.20, 0.20, 0.20]`], [Period-5 round-robin; no WTA.],
  [6],  [`[0.17, ..., 0.17]`], [Period-6 round-robin; no WTA.],
  [10], [`[0.17, ..., 0.17]`], [Same period-6 lock as $N = 6$.],
)

The spot-checks show two distinct behaviours: at small $N$ (here $N = 3$)
the drive-bumped neuron can dominate via near-saturation; at large $N$
($N >= 4$) the integer-tick dynamics absorb the $+1$ bump into the same
round-robin period and no neuron escapes the symmetric orbit.

== Continuous WTA strength

#figure(
  image("results/phase0/fcs_winner_fraction.pdf", width: 92%),
  caption: [Continuous WTA strength (rate_max $-$ second_max) per
  $(w, N, "drive_bump")$. At `drive_bump` $= 0$ the heatmap is uniformly
  near-zero (no asymmetry). At `drive_bump` $= 1$ there is a non-monotone
  $N$-dependence: $N = 6$ shows a pocket of strong asymmetry at
  intermediate $|w|$, while $N = 4$ and $N = 10$ are flat.]
)

= Verdict

*Phase 0 PASS (margin gate).* The FCS oracle produces WTA-stable cells
at every $N in {2, 3, 4, 6, 10}$ under the margin gate, with 36/200
cells blue at `drive_bump` $= 1$ overall.

*Two empirical findings* (to be revisited in subsequent phases):

#enum(
  [The 2-neuron *staircase* generalizes to $N$-way *synchronous lock*
   under FCS-Lustre symmetric weights (`drive_bump` $= 0$). Smooth-rate
   theory will be invisible to this at every $N$.],
  [The strict gate (`rate_max` $>= 0.99$, `second_max` $<= 0.01$) is
   *increasingly stringent* with $N$ because the small $+1$ integer
   bump on neuron $0$ is dwarfed by $(N - 1)$-fold symmetric inhibition.
   Phase 1's Siegert orbit decomposition will need to handle the
   $S_(N-1)$-broken-symmetry case to predict these near-WTA cells.],
)

#v(0.6em)

#text(size: 9pt, fill: luma(80))[
  Output: `results/phase0/fcs_grid_multi.{npz,pdf}`,
  `results/phase0/fcs_winner_fraction.pdf`, `results/phase0.log`.
]
