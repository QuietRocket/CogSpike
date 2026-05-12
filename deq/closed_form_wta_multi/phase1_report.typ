// Phase 1 (multi-N): Siegert FP enumeration vs FCS oracle.
#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 14pt, weight: "bold")[
    Phase 1 (multi-N) --- Siegert FP enumeration via multi-restart fsolve
  ]
  #v(0.2em)
  #text(size: 10pt, style: "italic")[
    `deq/closed_form_wta_multi/`, May 2026
  ]
]

= Goal

Enumerate Siegert mean-field fixed points of the $N$-neuron uniform
all-to-all inhibition motif and compare WTA-capability against the
Phase 0 FCS oracle on the $(w, N, "drive_bump")$ grid.

= Method

The 2-neuron thread used a *1-D scalar reduction* of the FP equation
that does not extend past $N = 2$. We initially attempted a
*symmetric-orbit ansatz* under $S_N$ (orbits of shape $(nu_W^k, nu_L^(N-k))$)
which is closed-form 2-D, but the implementation's
multi-restart fsolve diagnostic at `drive_bump` $= 1$, $w = -30$
flagged that the broken-symmetry $S_(N-1)$ case admits FPs with
*three or more distinct rates* per orbit (e.g., at $N = 3$:
$[0.23, 0.12, 0.03]$). Per the plan's risk-flag, we fall back to
*multi-restart fsolve in the full $N$-dimensional rate vector*.

The fsolve enumeration uses curated initial conditions:

- All-equal at $nu in {0.05, 0.3, 0.5, 0.95}$.
- $k$-winners / $(N - k)$-losers at extreme ($nu_W = 0.98$, $nu_L = 0.005$)
  and mild ($nu_W = 0.6$, $nu_L = 0.05$) for $k = 1, ..., 4$.
- Random ICs filling out 30 restarts per cell.

WTA-capable iff some FP has rate-spread (rate_max $-$ rate_2nd) $>= 0.30$
(matches Phase 0 margin gate and Phase 3 QR gate).

= Result

#figure(
  image("results/phase1/siegert_orbits_vs_fcs.pdf", width: 96%),
  caption: [Phase 1 Siegert FP enumeration (right column) vs Phase 0 FCS
  oracle margin-gate labels (left column), per $N$, per `drive_bump`.
  At every $N$, $J_{"db=0"} = 0$: Siegert sees bistable FPs that the
  symmetric FCS Lustre dynamics cannot select.]
)

#figure(
  image("results/phase1/siegert_max_spread.pdf", width: 80%),
  caption: [Phase 1 max FP spread (continuous). Above $|w| approx 19$ the
  asymmetric FP appears at every $N$, with spread $approx 0.54$ saturating
  at deep $|w|$. Below the bifurcation Siegert sees only a tiny bumped
  advantage ($approx 0.09$).]
)

== Headline numbers (Jaccard vs FCS margin gate, per $(N, "drive_bump")$)

#table(
  columns: 5,
  inset: 5pt,
  table.header([*$N$*], [*db*], [*Sieg blue*], [*FCS blue*], [*Jaccard*]),
  [2],  [0], [26 / 40], [0 / 40],  [0.00],
  [2],  [1], [30 / 40], [10 / 40], [0.29],
  [3],  [0], [24 / 40], [0 / 40],  [0.00],
  [3],  [1], [29 / 40], [9 / 40],  [0.27],
  [4],  [0], [22 / 40], [0 / 40],  [0.00],
  [4],  [1], [28 / 40], [6 / 40],  [0.10],
  [6],  [0], [19 / 40], [0 / 40],  [0.00],
  [6],  [1], [25 / 40], [8 / 40],  [0.14],
  [10], [0], [15 / 40], [0 / 40],  [0.00],
  [10], [1], [22 / 40], [3 / 40],  [0.00],
)

= Two distinct invisibility phenomena

The Phase 1 vs Phase 0 disagreement has *two opposite signs* at $N > 2$:

== `drive_bump = 0`: staircase invisibility generalizes to $N$-way lock

Siegert sees the bistable asymmetric FP (when it exists), but FCS
Lustre semantics with symmetric weights forces all $N$ neurons into
identical synchronous trajectories --- the 2-neuron staircase becomes
an $N$-way synchronous lock at every $N$. FCS-blue = 0 at every cell;
Siegert-blue is large. This is the *direct generalization* of the
$N = 2$ phenomenon, just stronger.

== `drive_bump = 1`, high $N$: inverse staircase --- FCS-blue, Siegert-red

At $N = 10$, `drive_bump` $= 1$, $w in {-15, -16, -7}$ FCS produces
*clean WTA* (e.g., at $w = -16$: post-warmup rate = $(0.96, 0, 0, ...)$,
spread $= 0.957$) while *no rate-equation FP with spread $>= 0.30$ exists*
at these $w$ values --- 270 curated initial conditions all converge to
the unique symmetric FP at $(0.13, 0.046, ..., 0.046)$ (spread $approx 0.09$).

The smooth-rate theory says: at $|w| < 19$, only the symmetric FP exists;
the bumped neuron has at most $approx 0.09$ rate advantage. The FCS
integer-tick dynamics say: at the specific integer values
$w in {-15, -16, -7}$, the $+1$ drive bump deterministically tips the
integer-arithmetic membrane into the bumped neuron firing first; one
firing event sends $-16$ inhibition to all others, which is enough
(in integer-tick semantics) to keep them sub-threshold *permanently*,
even though the smooth-rate average rate of the bumped neuron is too low
to sustain that suppression in mean-field.

This is the *inverse* of the 2-neuron staircase: at $N = 2$ the integer
dynamics see synchronous lock that smooth theory misses (FCS-red,
Siegert-blue); at $N = 10$ the integer dynamics see WTA that smooth
theory misses (FCS-blue, Siegert-red).

== Recall by $N$ at `drive_bump = 1`

Recall = $|"Siegert-blue" inter "FCS-blue"| \/ |"FCS-blue"|$:

#table(
  columns: 4,
  inset: 5pt,
  table.header([*$N$*], [*FCS-blue (margin)*], [*Sieg $inter$ FCS*], [*Recall*]),
  [2],  [10], [$approx 9$], [0.90],
  [3],  [9],  [$approx 8$], [0.89],
  [4],  [6],  [$approx 3$], [0.50],
  [6],  [8],  [$approx 4$], [0.50],
  [10], [3],  [0],          [0.00],
)

Recall *degrades monotonically with $N$*. At $N = 2$ Siegert is a
high-recall predictor (just as in the 2-neuron thread, which achieved
$J approx 0.80$ on the full $(w_(12), w_(21))$ grid). At $N = 10$,
recall is zero: the rare cells where FCS sees WTA are *not* the cells
where Siegert sees an asymmetric FP basin.

= Verdict

*Phase 1 PARTIAL.* The plan's pass-gate (recall $>= 0.85$ per $N$ at
`drive_bump` $= 1$) holds at $N = 2, 3$ but degrades to $0.50$ at
$N = 4, 6$ and to $0$ at $N = 10$. The expected "Siegert is a
high-recall superset" relationship breaks at $N >= 4$.

The two reported findings *are* the result of this phase, not pass/fail
issues with the implementation:

#enum(
  [The 2-neuron staircase generalizes to $N$-way synchronous lock at
   every $N$ when `drive_bump = 0` --- and this is *still*
   invisible to smooth-rate theory.],
  [A new phenomenon emerges at large $N$ with `drive_bump = 1`: the
   FCS integer-tick dynamics produce WTA at specific integer $w$ values
   *below the smooth-rate bifurcation threshold* (the "inverse
   staircase"). Phase 2's $H(omega)$ latency gate will not help here
   because there is no rate-equation FP to evaluate the eigenvalue at.],
)

#v(0.6em)

#text(size: 9pt, fill: luma(80))[
  Output: `results/phase1/siegert_orbits.npz`,
  `siegert_orbits_vs_fcs.pdf`, `siegert_max_spread.pdf`,
  `orbit_count_vs_N.pdf`, `results/phase1.log`.
]
