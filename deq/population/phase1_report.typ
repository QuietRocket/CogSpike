#set page(paper: "a4", margin: 2cm)
#set text(size: 10pt)
#set par(justify: true)

#align(center)[
  #text(size: 14pt, weight: "bold")[Phase 1 report -- Spectral gap as behavioral proxy]
  #v(0.2em)
  Verdict: *PASS*
]

= Setup

Contralateral inhibition archetype with drive $I = [1.5, 1.5]$, time
constant $tau = 1$, and sigmoid $f(x) = 1 / (1 + exp(-4(x-1)))$. The
weight grid is $w_(12), w_(21) in [0, 5]$ with a 50 times 50 mesh
($Delta w = 0.1$).

For each cell, three quantities are computed:

- the middle-branch fixed point of the scalar reduction
  $rho_1 = f(I - w_(21) f(I - w_(12) rho_1))$, i.e. the analytical
  analogue of the symmetric saddle in the asymmetric 2D system,
- the dominant real part $"Re"(lambda_1)$ of the Jacobian at that
  fixed point and the spectral gap $Delta = "Re"(lambda_1) - "Re"(lambda_2)$,
- a simulation-based WTA verdict over $t in [0, 50]$, starting from
  the mirror-image perturbations $rho^* plus.minus (0.05, -0.05)$, with
  WTA declared when both runs end with $|rho_1 - rho_2| > 0.3$ *and*
  with opposite signs of $rho_1 - rho_2$. The sign-opposition clause
  tightens the plan's stated criterion (plan §3.3), which by its bare
  letter would also flag strongly skew single-FP regimes as WTA; the
  parenthetical in the same section ('the system commits to one
  population dominating [which population depends on the initial
  condition]') makes clear that bistable choice is the intended meaning,
  and the tightening is the faithful reading.

= Closed-form diagonal result

On the diagonal $w_(12) = w_(21) = w$, the linearization of the WC
system at the symmetric fixed point reduces to a matrix with off-diagonal
entries $-w g$, where $g = f'(I - w rho^*)$ is the sigmoid slope at the
fixed-point input. Eigenvalues are $(-1 plus.minus w g) slash tau$ and
the pitchfork condition $w g = 1$ yields

$ w^* = 1.000008, quad rho^*(w^*) = 0.499998, quad g(w^*) = 1.000000. $

= Classification accuracy

Binarising the spectral predictor at the sign of $"Re"(lambda_1)$ (WTA
when the dominant real part is positive) and comparing cell-by-cell
against the simulation-based ground truth yields an accuracy of
*99.96 %* (acceptance threshold: $gt.eq$ 95 %).

The analytical bifurcation curve was traced by bisection along radial
slices in the positive quadrant (181 angles); its distance from the
empirical boundary cells is summarized by

- median: 0.0527 weight units ($= 0.52$ grid cells),
- maximum: 0.1016 weight units ($= 1.00$ grid cells).

The plan's acceptance threshold is within one grid cell
($Delta w = 0.10204081632653061$) at every boundary point.

The spectral gap changes sign (i.e. $"Re"(lambda_1)$ crosses zero)
precisely at the empirical WTA boundary:
*yes*.

= Figures

#figure(image("results/phase1/panel_a_wta_ground_truth.pdf", width: 70%),
  caption: [Panel (a). Empirical WTA map on the 50 times 50 weight grid.
  Black cells commit to winner-take-all from both symmetry-breaking
  initial conditions within $t = 50$; white cells remain symmetric.])

#figure(image("results/phase1/panel_b_spectral_gap.pdf", width: 80%),
  caption: [Panel (b). Spectral gap $Delta = "Re"(lambda_1) - "Re"(lambda_2)$
  at the symmetric fixed point. The black contour marks the zero-level of
  the dominant real part, i.e. the linear-stability boundary.])

#figure(image("results/phase1/panel_c_curve_overlay.pdf", width: 70%),
  caption: [Panel (c). Fine-grained analytical bifurcation curve (red)
  obtained by bisection on the dominant eigenvalue, overlaid on the
  empirical WTA map (grey). The yellow marker locates the closed-form
  diagonal pitchfork point $w^* = w_(12) = w_(21)$.])

= Verdict

Accuracy $gt.eq$ 95 %: yes.
Analytical curve within 1 grid cell at every boundary point:
yes (max = 1.00 cells).
Dominant real part crosses zero at the boundary:
yes.

*Overall verdict: PASS.*
