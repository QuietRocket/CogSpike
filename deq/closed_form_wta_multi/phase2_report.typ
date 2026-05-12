// Phase 2 (multi-N): H(omega) latency gate.
#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 14pt, weight: "bold")[
    Phase 2 (multi-N) --- $H(omega)$ latency-gate reading of FCS Property 7
  ]
  #v(0.2em)
  #text(size: 10pt, style: "italic")[
    `deq/closed_form_wta_multi/`, May 2026
  ]
]

= Goal

Predict decision latency at each $(w, N, "drive_bump")$ cell from the
rate-equation Jacobian eigenvalues at the Siegert FP, and gate

$ |"Re"(lambda_("dom"))| > 1 / T_("FCS") = 0.25 $

as the FCS-prescribed condition for "decision faster than 4 ticks".
The $N$-neuron generalization needs the full $N times N$ Jacobian
spectrum at the winner-orbit FP from Phase 1.

= Method

Per cell:

#enum(
  [Lift the largest-spread FP $bold(nu)^* in [0, 1]^N$ from Phase 1's
   multi-restart fsolve enumeration.],
  [Compute per-neuron transfer gain $g_i = d Phi / d mu$ at the
   FP-induced inputs $(mu_i^*, sigma_i^*)$ for $i = 1, ..., N$.],
  [Build the $N times N$ rate-equation Jacobian in Siegert units:
   $ J = alpha dot.c W_("uniform"), quad W_("uniform")[i, j] = w (i != j), 0 (i = j) $
   and apply the existing
   `transfer.jacobian_eigenvalues(J, "gains", tau_m)` which computes
   eigenvalues of $A = (1 / tau_m) (-I + "diag"(g) J)$.],
  [Blue iff `sieg_labels[cell]` $= 1$ AND $|"Re"(lambda_("dom"))| > 0.25$.],
)

= Closed-form spectrum at uniform $W$

At the $k = 1$ winner orbit, the $N times N$ Jacobian has a *clean
block structure* under the $S_(N-1)$ subgroup that permutes the
$(N - 1)$ losers. The numerics at $"drive_bump" = 1$, $w = -30$ confirm:

#table(
  columns: 3,
  inset: 5pt,
  table.header([*$N$*], [*Re(eigvals)*], [*Loser-symm eigenvalue*]),
  [2],  [$[-0.457, -0.394]$], [---],
  [3],  [$[-0.470, -0.424, -0.382]$], [$-0.424$ (1-fold)],
  [4],  [$[-0.481, -0.424, -0.424, -0.372]$], [$-0.424$ (2-fold)],
  [6],  [$[-0.498, -0.424^times 4, -0.357]$], [$-0.424$ (4-fold)],
  [10], [$[-0.526, -0.424^times 8, -0.334]$], [$-0.424$ (8-fold)],
)

The middle $N - 2$ eigenvalues lock at exactly $-0.424$ --- the
$(N - 2)$-fold-degenerate eigenvalue of the loser-permutation
subspace, equal to $alpha dot.c w dot.c g_L dot.c (-1) / tau_m + (-1) / tau_m$
where $-1$ is the eigenvalue of $(bold(1) bold(1)^T - I)$ restricted to
the orthogonal-to-$bold(1)_(N-1)$ subspace of the loser block. The two
non-degenerate eigenvalues (most-negative and least-negative real
parts) correspond to the winner-loser-mean 2-block.

All eigenvalues are *negative real* → asymmetric FP is stable.

= Result

#figure(
  image("results/phase2/h_gate_vs_fcs_multi.pdf", width: 96%),
  caption: [Phase 2 $H(omega)$ gate (right column) vs Phase 1 Siegert
  (middle) vs FCS oracle margin gate (left), per $N$, per `drive_bump`.
  Each panel: two rows of dots for `drive_bump` $in {0, 1}$.]
)

#figure(
  image("results/phase2/eigvalue_spectrum_vs_N.pdf", width: 80%),
  caption: [Dominant Jacobian eigenvalue Re($lambda_("dom")$) across
  $(w, N)$. For $|w| >= 19$ at every $N$, $|"Re"(lambda_("dom"))| > 0.25$
  (gate passes). Below the bifurcation Re($lambda$) approaches
  $-alpha / tau_m$ from above, all stable.]
)

== Jaccard table at `drive_bump = 1`, vs FCS margin gate

#table(
  columns: 6,
  inset: 5pt,
  table.header([*$N$*], [*H-blue*], [*Sieg-blue*], [*FCS-blue (m)*], [*J(H)*], [*J(Sieg)*]),
  [2],  [24], [30], [10], [0.214], [0.290],
  [3],  [22], [29], [9],  [0.192], [0.267],
  [4],  [21], [28], [6],  [0.125], [0.097],
  [6],  [19], [25], [8],  [0.174], [0.138],
  [10], [17], [22], [3],  [0.000], [0.000],
)

The $H(omega)$ gate *tightens* the Siegert envelope (fewer blue cells
at every $N$). At $N >= 4$ this tightening produces a small Jaccard
improvement; at $N <= 3$ it removes some FCS-aligned Siegert cells and
the Jaccard regresses. *Overall db=1 J = 0.149* (H) vs *0.164* (Siegert)
--- the latency gate does not meaningfully help.

The threshold sweep finds *best $J = 0.174$ at threshold $approx 0.15$*
(below the FCS-prescribed $0.25$), confirming there is no clean
eigenvalue threshold that recovers the FCS-blue cells.

= Why the $H(omega)$ gate fails to close the gap

The Phase 1 analysis identified two divergence modes between Siegert
and FCS at $N > 2$:

#enum(
  [*`drive_bump = 0`:* FCS Lustre symmetry forces $N$-way synchronous
   lock; Siegert sees the bistable FP but FCS can never select it.
   The $H(omega)$ gate evaluates eigenvalues at the same bistable FP
   --- same FP, same blindness to FCS's synchronous solution.],
  [*`drive_bump = 1`, high $N$ (inverse staircase):* FCS produces
   clean WTA at specific integer $w$ values where *no asymmetric FP exists*.
   The gate cannot evaluate eigenvalues of a non-existent FP, so these
   FCS-blue cells remain Siegert-red, $H$-red.],
)

Neither failure mode is recoverable by inserting a frequency-domain
gate downstream of the static FP enumeration. Both are properties of
the *FCS dynamics*, not of the rate-equation linearization at any FP.

= Verdict

*Phase 2 PARTIAL.* The pass-gate $J(H) >= 0.70$ from the 2-neuron
thread does not hold at $N > 2$. The closed-form $N times N$ spectrum
at the $k = 1$ winner orbit is elegant and analytically tractable
(the $(N - 2)$-fold loser-permutation degeneracy locks at $-0.424$
in Siegert units), but the gate's *predictive value* is bounded by
the Phase 1 FP enumeration, which itself fails to capture the
FCS-dynamics-only WTA cells. The latency-gate machinery is correct
and reusable; the limitation is upstream.

#v(0.6em)

#text(size: 9pt, fill: luma(80))[
  Output: `results/phase2/h_grid_multi.npz`,
  `h_gate_vs_fcs_multi.pdf`, `eigvalue_spectrum_vs_N.pdf`,
  `results/phase2.log`.
]
