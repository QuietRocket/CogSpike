#set page(paper: "a4", margin: 2cm)
#set text(size: 10pt)
#set par(justify: true)

#align(center)[
  #text(size: 14pt, weight: "bold")[Phase 5 report -- Generalization to
  other archetypes]
  #v(0.2em)
  Verdict: *PASS*
]

= Subtask 5A -- Simple series chain

An $n$-population feed-forward chain with equal inter-stage weight $w$
and drive only on stage 0 has a recursive steady state
$rho_0 = f(I)$, $rho_k = f(w dot rho_(k-1))$. With no recurrence the
Jacobian is triangular and all eigenvalues equal $-1 slash tau$, so
convergence to the steady state is exponential and free of oscillation;
this is the natural population-level analogue of a feed-forward gain
cascade.

The table below reports the maximum relative error between the
analytical recursion and the numerical steady state over a weight sweep
$w in [0.2, 4.0]$ and chains of lengths $n in {2, 3, 5, 10}$.

#table(columns: 2,
  [chain length], [max rel. error over $w$-sweep],
  [$n = 2$], [$0.000e+00$],
  [$n = 3$], [$1.431e-13$],
  [$n = 5$], [$2.593e-12$],
  [$n = 10$], [$9.450e-11$],
)

Acceptance (5 % relative error): *PASS*
(overall max = $9.450e-11$).

#figure(image("results/phase5/series.pdf", width: 75%),
  caption: [5A. Final-stage activity $rho_{n - 1}^*$ versus chain weight
  $w$ for chains of lengths 2, 3, 5, 10. Solid lines are the analytical
  recursive-sigmoid prediction; open dots are the numerical steady
  states. The curves overlap within line width.])

= Subtask 5B -- Parallel composition

A driver population feeding $n$ independently-weighted downstream
populations produces a block-triangular Jacobian: the driver's $1 times 1$
self-block is $-1 slash tau$, the downstream $n times n$ block is
diagonal (no cross-coupling), and the only non-zero off-diagonal block
is the driver $arrow$ downstream coupling. Every eigenvalue therefore
equals $-1 slash tau$ exactly, independent of the gain vector
$(w_(i))_(i=1)^(n)$. The fixed point is closed-form:
$rho_0 = f(I)$, $rho_k = f(w_k rho_0)$.

#table(columns: 5,
  [$n$], [FP $L^(oo)$ error], [downstream off-diag mag.],
  [downstream $arrow$ driver coupling], [eigenvalue error],
  [$n = 2$], [$0.00e+00$], [$0.00e+00$], [$0.00e+00$], [$0.00e+00$],
  [$n = 4$], [$0.00e+00$], [$0.00e+00$], [$0.00e+00$], [$0.00e+00$],
  [$n = 8$], [$0.00e+00$], [$0.00e+00$], [$0.00e+00$], [$0.00e+00$],
)

Acceptance (structure to machine precision, FP within 5 %):
*PASS*.

#figure(image("results/phase5/parallel_jacobian.pdf", width: 60%),
  caption: [5B. Jacobian of a parallel composition with $n = 8$ downstream
  populations. The block-triangular structure is visible: the driver's
  row (row 0) depends only on itself; each downstream row (rows 1--8) has
  a non-zero entry only on its own diagonal and on column 0.])

= Subtask 5C -- Positive loop saddle-node bifurcation

Two mutually exciting populations with $w_(12) = w_(21) = w$ and zero
drive admit a symmetric scalar reduction $rho = f(w rho)$. The
saddle-node fold of the FP curve occurs when the slope condition
$w k rho(1 - rho) = 1$ is met simultaneously with the FP equation.
Eliminating $rho$ gives the analytical saddle-node weight(s) at
$I = 0$, tabulated below against a numerical FP-count scan.

#table(columns: 3,
  [analytical $w_("SN")$], [numerical transition], [rel. error],
  [$1.6818$], [$1.6756$], [$0.371%$],
  [$5.2778$], [$5.2733$], [$0.086%$],
)

Acceptance (5 % relative error): *PASS*.

#figure(image("results/phase5/positive_loop.pdf", width: 75%),
  caption: [5C. Fixed-point branches of the symmetric positive loop as
  functions of the loop weight $w$, at zero drive. A single low-active
  branch persists for small $w$; above the analytical saddle-node weight
  (red dashed line) a high-active branch and a middle saddle appear.])

= Verdict

- 5A series chain: PASS
- 5B parallel composition: PASS
- 5C positive loop saddle-node: PASS

Overall: *PASS*.
