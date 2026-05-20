// Multi-neuron WTA closed-form study (sibling to deq/closed_form_wta/).
// Phases 0-3 mirror the 2-neuron pipeline at N > 2 with uniform
// all-to-all inhibition.

#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 15pt, weight: "bold")[
    Closed-form reproduction of FCS Property 7 at $N > 2$
  ]
  #v(0.2em)
  #text(size: 11pt, style: "italic")[
    Three lenses re-applied to uniform all-to-all inhibition, calibrated
    against the FCS-LI&F oracle on the $(w, N)$ plane.
  ]
  #v(0.3em)
  #text(size: 10pt)[Nikan Zandian #h(0.6em) `deq/closed_form_wta_multi/`]
]

#v(0.4em)

= Context

The companion thread `deq/closed_form_wta/` @closed_form_wta reads FCS
Property 7 (winner-takes-all stability within 4 ticks) of @DeMaria2020 on
the 2-neuron contralateral motif through three lenses ---
Siegert @Siegert1951 @Brunel2000 mean-field fixed-point enumeration,
$H(omega)$ latency gate @Richardson2007, and quasi-renewal mesoscopic
simulation @NaudGerstner2012 at finite population $N_("pop")$ ---
calibrated against the FCS-LI&F oracle on a $40 times 40$ grid over the
two inhibitory weights $(w_(12), w_(21))$. The three 2-neuron findings:

#enum(
  [Siegert mean-field tracks the FCS Fig. 10 WTA boundary at Jaccard
   $approx 0.80$. The $approx 20%$ gap is the *diagonal staircase* of
   integer-tick synchronous lock that smooth-rate theory cannot
   resolve.],
  [The $H(omega)$ eigenvalue gate $|"Re"(lambda_("dom"))| > 1\/T_("FCS") = 0.25$
   closes part of that gap by removing slow-decay cells near the
   rate-equation bifurcation.],
  [Quasi-renewal at $N_("pop") arrow infinity$ converges back to the
   Siegert mean-field, confirming that finite-size noise alone cannot
   undo the spike-timing lock.],
)

This thread extends the same scenario to *more than two neurons*, sweeping
uniform all-to-all lateral inhibition across $N in {2, 3, 4, 6, 10}$ and
$w in {-40, ..., -1}$. The driving question: do the three lenses' relative
predictive ordering carry over from $N = 2$ to $N > 2$?

#text(size: 10pt, style: "italic", fill: rgb(40, 80, 40))[
  Spoiler. At $N > 2$ the relationship breaks in two distinct, opposite
  ways. Both are exhibits of "FCS integer-tick discreteness sees what
  smooth theory cannot", but they have opposite signs.
]

= Setup

== Topology

Uniform all-to-all inhibition: $W$ has $W_(i,j) = w$ for $i != j$ and
$W_(i,i) = 0$. Every neuron has its own external drive of magnitude
$"self_drive" = 11$ (the FCS delayer threshold), with neuron 0 receiving
an extra integer self-drive bump $"drive_bump" in {0, 1}$ to optionally
break the FCS Lustre-faithful $S_N$ permutation symmetry. The constructor
`deq/archetypes/topologies.py:all_to_all_inhibition` reduces to
`contralateral(w, w)` at $N = 2$, $"drive_bump" = 0$ (bit-exact unit-checked).

== Gates

The Phase 0 oracle uses two gates side by side:

#enum(
  [*Strict* (FCS Fig. 10 generalization): post-warmup $"rate_max" >= 0.99$
   AND $"second_max" <= 0.01$. Coincides with the 2-neuron gate at $N = 2$.],
  [*Margin* (aligns with Phases 1, 3): $"rate_max" - "second_max" >= 0.30$.],
)

Phases 1, 2, 3 use the *margin gate* for cross-comparison.

= Phase 0 --- FCS oracle at $N > 2$

#figure(
  image("../results/phase0/fcs_grid_multi.pdf", width: 96%),
  caption: [Phase 0 FCS oracle labels across $(w, N, "drive_bump")$. Left
  column: strict gate; right: margin gate. Per panel: two rows of dots
  for `drive_bump` $in {0, 1}$.]
)

== Headline counts (`drive_bump = 1`)

#table(
  columns: 4,
  inset: 5pt,
  table.header([*$N$*], [*Strict*], [*Margin*], [*Comment*]),
  [2],  [9 / 40], [10 / 40], [Bumped diagonal cuts the 2-neuron staircase.],
  [3],  [4 / 40], [9 / 40],  [Near-WTA cells (winner $approx 0.98$, second $> 0.01$).],
  [4],  [2 / 40], [6 / 40],  [Integer-tick lock collapses into period-$N$ round-robin.],
  [6],  [0 / 40], [8 / 40],  [Strict gate fails; margin recovers a pocket.],
  [10], [0 / 40], [3 / 40],  [Thin band at $w in {-15, -16, -7}$ only.],
)

At `drive_bump = 0` every cell is red at every $N$. The FCS Lustre-faithful
$S_N$ symmetry forces all $N$ neurons into identical synchronous
trajectories. *The 2-neuron diagonal staircase becomes an $N$-way
synchronous lock at every $N$.*

= Phase 1 --- Siegert FP enumeration (multi-restart fsolve)

The 2-neuron 1-D scalar reduction does not extend past $N = 2$. We
initially attempted a *symmetric-orbit ansatz* under $S_N$ (orbits of
shape $(nu_W^k, nu_L^(N-k))$, exact 2-D system per $k$), but the
multi-restart fsolve diagnostic flagged that the broken-symmetry
$S_(N-1)$ case at `drive_bump = 1` admits FPs with *three or more
distinct rates*. For instance at $N = 3$, $w = -30$, `drive_bump = 1`,
fsolve from random ICs returns the FP $[0.23, 0.12, 0.03]$, which the
"bumped + $(N-1)$-tied losers" ansatz cannot represent.

Per the plan's risk-flag, Phase 1 falls back to *multi-restart fsolve in
the full $N$-dimensional rate vector*. The IC set is 30 per cell: corner
ICs (symmetric, $k$-winners-$(N-k)$-losers for $k = 1, ..., 4$, extreme
and mild variants) plus random fills. WTA-capable iff some FP has
rate_max $-$ rate_2nd $>= 0.30$.

#figure(
  image("../results/phase1/siegert_orbits_vs_fcs.pdf", width: 96%),
  caption: [Phase 1 Siegert FP enumeration (right) vs FCS margin gate
  (left), per $N$, per `drive_bump`. At every $N$, $J_("db=0") = 0$:
  Siegert sees bistable FPs that the symmetric FCS dynamics cannot
  select.]
)

== Two distinct invisibility phenomena

#enum(
  [*`drive_bump = 0`: staircase invisibility generalizes to $N$-way lock.*
   Siegert sees the bistable asymmetric FP; FCS sees synchronous lock.
   FCS-blue $= 0$ everywhere; Siegert-blue large. Direct generalization
   of the 2-neuron staircase.],
  [*`drive_bump = 1`, high $N$: inverse staircase.* At $N = 10$, the
   $w$ values where FCS sees clean WTA ($w in {-15, -16, -7}$) admit
   *no asymmetric rate-equation FP at all*. 270 curated initial
   conditions all converge to the unique symmetric FP
   $(0.13, 0.046, ..., 0.046)$ with spread $0.09 < 0.30$. The FCS
   integer-tick dynamics produce WTA *below* the smooth-rate
   bifurcation threshold.],
)

This second phenomenon is the *opposite* of the 2-neuron staircase. At
$N = 2$, FCS sees synchronous lock that Siegert overpredicts away
(FCS-red, Siegert-blue). At $N = 10$, FCS sees integer-tick WTA that
Siegert under-predicts (FCS-blue, Siegert-red). Recall (Siegert $inter$
FCS / FCS) at `drive_bump = 1`:

#table(
  columns: 3,
  inset: 5pt,
  table.header([*$N$*], [*FCS-blue (margin)*], [*Recall*]),
  [2],  [10], [0.90],
  [3],  [9],  [0.89],
  [4],  [6],  [0.50],
  [6],  [8],  [0.50],
  [10], [3],  [0.00],
)

Recall degrades monotonically with $N$, collapsing to zero at $N = 10$.

= Phase 2 --- $H(omega)$ latency gate

The $N times N$ rate-equation Jacobian at the $k = 1$ winner FP has a
*clean block structure* under the $S_(N - 1)$ loser-permutation subgroup.
The numerics at $"drive_bump" = 1$, $w = -30$ show an $(N - 2)$-fold
degenerate "loser-permutation" eigenvalue locked at $-0.424$ in Siegert
units at every $N >= 3$; the two non-degenerate eigenvalues are the
winner-loser-mean 2-block.

#figure(
  image("../results/phase2/eigvalue_spectrum_vs_N.pdf", width: 78%),
  caption: [Dominant Jacobian eigenvalue Re($lambda_("dom")$) across
  $(w, N)$. For $|w| >= 19$ the asymmetric FP is stable at every $N$;
  the gate $|"Re"(lambda_("dom"))| > 0.25$ passes there.]
)

The gate, however, *cannot rescue* Phase 1's two failure modes. It
evaluates eigenvalues at the *existing* rate-equation FP --- where
that FP does not exist (inverse staircase) or is not selected
(staircase), no downstream gate helps. Overall
$J(H, "FCS-margin", "db=1") = 0.149$ vs $J("Siegert") = 0.164$:
the gate slightly tightens Siegert by removing some slow-decay cells
but does not improve agreement with FCS.

= Phase 3 --- Quasi-renewal at finite $N_("pop")$

#figure(
  image("../results/phase3/qr_jaccard_vs_Npop_per_N.pdf", width: 96%),
  caption: [Left: Jaccard(QR, FCS-margin) vs $N_("pop")$, one curve per
  $N_("neurons")$. Right: Jaccard(QR, Siegert). Mean-field convergence
  is clean at every $N_("neurons")$.]
)

== Mean-field convergence headline

QR at $N_("pop") = 2000$ vs Siegert mean-field:

#table(
  columns: 3,
  inset: 5pt,
  table.header([*$N_("neurons")$*], [*J(QR, FCS)*], [*J(QR, Siegert)*]),
  [2],  [0.29], [*1.00*],
  [3],  [0.30], [*0.90*],
  [4],  [0.11], [*0.86*],
  [6],  [0.15], [*0.88*],
  [10], [0.00], [*0.82*],
)

Mean Jaccard $= 0.89$, comfortably above the $0.70$ gate. The *rate
equation is still the right large-$N_("pop")$ object* at every
$N_("neurons") in [2, 10]$.

Finite-$N_("pop")$ broadening *amplifies with $N_("neurons")$*: at
$N_("neurons") = 2$ we need $N_("pop") approx 50$ for QR to reproduce
all Siegert-blue cells; at $N_("neurons") = 10$ we need
$N_("pop") >= 500$. Each additional competitor amplifies the
noise-induced symmetry restoration of the asymmetric basin.

Crucially, *neither QR nor Siegert recovers the FCS-only inverse-staircase
cells* at $N_("neurons") = 10$ (any $N_("pop")$). The inverse staircase
is genuinely beyond rate-equation theory and its stochastic mesoscopic
approximation.

= Conclusions

The 2-neuron $J approx 0.80$ Siegert-vs-FCS agreement does *not* extend
to $N > 2$. The two phenomena that emerge in its place are:

#enum(
  [*$N$-way synchronous lock at `drive_bump = 0`*: the diagonal
   staircase of the 2-neuron thread is one slice of a more general $S_N$
   integer-tick synchrony at every $N$. Smooth-rate theory is invisible
   to it at any $N$.],
  [*Inverse staircase at large $N$ with `drive_bump = 1`*: a *new*
   phenomenon not visible at $N = 2$. The FCS integer-tick dynamics
   produce clean WTA at specific integer $w$ values *below the smooth-rate
   bifurcation threshold*. The 1-bumped-neuron, $(N - 1)$-symmetric-loser
   configuration is a discrete-arithmetic attractor that has no
   rate-equation counterpart.],
)

The $H(omega)$ latency gate cannot bridge either gap because both are
properties of the FCS dynamics itself, not of the linearization at any
rate-equation FP. The quasi-renewal stochastic mesoscopic equation
converges to Siegert mean-field at every $N$ in $[2, 10]$, confirming
that the rate equation *is* the right large-$N_("pop")$ object, but
inheriting Siegert's blindness to the integer-tick phenomena.

*Methodological observation.* The 2-neuron's 1-D scalar reduction is a
small-$N$ luxury. The natural multi-$N$ tool is the symmetric-orbit
ansatz under $S_N$, exact for uniform $W$ at zero symmetry breaker. With
$"drive_bump" > 0$ the broken-symmetry structure becomes richer than the
ansatz can handle and *multi-restart fsolve in the full $N$-dimensional
rate vector* is the necessary fallback. The plan's risk-flag in this
direction proved load-bearing.

= Pointer to companion thread

For the 2-neuron Property 7 reading and the original three-lens analysis,
see `deq/closed_form_wta/note/closed_form_wta_note.pdf`
(`refs.bib` carried forward verbatim).

#bibliography("refs.bib", title: "References", style: "ieee")
