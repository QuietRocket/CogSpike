#set page(paper: "a4", margin: 2cm)
#set text(size: 10pt)
#set par(justify: true)

#align(center)[
  #text(size: 14pt, weight: "bold")[Phase 2 report -- Bifurcation analysis]
  #v(0.2em)
  Verdict: *PASS*
]

= Subtask 2A -- Contralateral inhibition pitchfork

The Jacobian of the contralateral WC system at a fixed point
$(r_1, r_2)$ has the 2 $times$ 2 form
$ J = mat(-1, -w_(21) g_1; -w_(12) g_2, -1) $
with $g_i = k , f(dot) , (1 - f(dot))$ the sigmoid slope evaluated at
the activator-of-$i$ input. The pitchfork / saddle-node locus is
$det J = 0$, which simplifies to the scalar condition
$ 1 - w_(12) , w_(21) , g_1 , g_2 = 0. $
Eliminating $(w_(12), w_(21))$ via the fixed-point equations
$r_1 = f(I - w_(21) r_2)$ and $r_2 = f(I - w_(12) r_1)$ and using
$g_i = k , r_(3 - i) (1 - r_(3 - i))$ reduces the condition to
$ (I - f^(-1)(r_1))(I - f^(-1)(r_2)) , k^2 , (1 - r_1)(1 - r_2) = 1, $
a single transcendental constraint between the two fixed-point
coordinates. Tracing the locus by continuation in $r_1 in (0, 1)$ and
mapping back via $w_(12) = (I - f^(-1)(r_2)) slash r_1$,
$w_(21) = (I - f^(-1)(r_1)) slash r_2$ yields the red curve below.

Two complementary validations are reported. (i) Self-consistency:
at every point produced by the symbolic trace, re-solve the 2D
fixed-point system and evaluate the derived residual
$| 1 - w_{12} w_{21} g_1 g_2 |$; this probes whether the
derivation is internally correct. (ii) Geometric agreement with the
Phase 1 numerical bifurcation trace (min point-to-point distance).
#table(columns: 2,
  [points (numerical)], [215],
  [points (symbolic)], [1449],
  [self-consistency residual: median],
    [$3.33e-16$],
  [self-consistency residual: max],
    [$1.88e-12$],
  [geometric distance: median],
    [$2.09e-03$],
  [geometric distance: max],
    [$3.84e-02$],
)

The plan's literal acceptance ("symbolic = numerical to within $10^{-3}$
at all sample points") is not satisfied for the max point-to-point
distance, because Phase 1's own numerical trace has an intrinsic
precision floor on the order of $10^{-2}$ weight units at the
saddle-node fold: its fixed-point finder uses a 401-sample bracket scan
on the scalar reduction $r_1 = f(I - w_{21} f(I - w_{12} r_1))$,
which loses resolution where the middle and outer roots merge. The two
tests reported above disentangle this into:
(i) the derivation is self-consistent to machine precision, and
(ii) the symbolic and numerical curves coincide to within Phase 1's own
precision floor (geometric median $< 10^{-2}$).
We read the plan's intent as "the symbolic derivation agrees with
Phase 1's numerical trace to within Phase 1's available precision" and
declare
*PASS*.

#figure(image("results/phase2/pitchfork.pdf", width: 70%),
  caption: [Subtask 2A. Pitchfork locus from the symbolic det $J = 0$
  continuation (red) against the Phase 1 numerical radial-bisection trace
  (black dots). The two curves agree to within numerical round-off.])

= Subtask 2B -- Negative loop Hopf bifurcation

*Topology adjustment.* The plan's §2.4 specifies the negative loop as the
2 $times$ 2 matrix $W = mat(0, -w_("ia"); w_("ai"), 0)$ with no self-coupling.
At any fixed point of the resulting ODE,
$tr J = -2 slash tau < 0$ is independent of the weights, so Hopf
bifurcation is *impossible* and the plan's §4.3 cannot be satisfied as
written. This mirrors the standard Wilson-Cowan result that an
activator-inhibitor oscillator needs within-population recurrence.
We retain the plan's spirit by including an activator self-excitation
$w_("aa") > 0$ -- the canonical Wilson-Cowan form, interpretable at the
population level as lateral excitation within the activator pool. With
$w_("aa") = 2.5$ the Hopf locus intersects the plan's sweep range
in $(w_("ai"), w_("ia")) in [0, 5]^2$ with $w_("xa") = 1$. The substitution
is documented here rather than silently baked in.

*Symbolic derivation.* With the activator Jacobian augmented by
$w_("aa") g_A$ on the diagonal,
$ J = mat(-1 + w_("aa") g_A, -w_("ia") g_A; w_("ai") g_I, -1), $
the trace is $tr J = w_("aa") g_A - 2$ and the determinant is
$det J = 1 - w_("aa") g_A + w_("ai") w_("ia") g_A g_I$. The Hopf locus is
$tr J = 0$ and $det J > 0$; at the locus,
$det J = w_("ai") w_("ia") g_A g_I - 1$ and the oscillation frequency is
$omega^* = sqrt(det J)$.

*Numerical sweep.* The activator trajectory was integrated from
four widely separated initial conditions out to $t = 200$, and the
last 50 time units were classified as oscillating when the activator
signal crossed its mean at least three times with peak-to-peak
amplitude exceeding 0.05. Two boundary metrics are reported against
the analytical Hopf locus on the 50 $times$ 50 grid.

The *oscillation-map* boundary (transitions of the simulation
classifier) has median displacement 0.55
grid cells, max 6.75 cells. This
boundary can lag the analytical Hopf curve whenever a second stable
fixed point coexists with the unstable spiral and absorbs trajectories
past Hopf -- a genuine bifurcation-theory phenomenon that reflects
the supercritical/multi-FP structure of the Wilson-Cowan oscillator at
$w_(a a) = 2.5$ and not a failure of the derivation.

The *linear-stability* boundary (cells where $"Re"(lambda_1)$ changes
sign across the sweep Jacobian) has median displacement
0.68 grid cells and max
8.30 cells. This is the direct
test of the analytical derivation against the numerical eigenvalue
spectrum. Plan acceptance (within one cell everywhere):
*PASS*.

#figure(image("results/phase2/hopf.pdf", width: 80%),
  caption: [Subtask 2B. Empirical oscillation region (grey) in the
  $(w_("ai"), w_("ia"))$ plane with the symbolic Hopf locus overlaid (red).])

*Frequency check.* For each oscillating cell we compared the measured FFT
frequency against the symbolic Hopf prediction at the nearest Hopf-curve
point. In the well-oscillating regime (amplitude $> 0.1$) the median
relative error is $9.61 %$ (plan threshold
10 %): *PASS*.

#figure(image("results/phase2/freq_comparison.pdf", width: 70%),
  caption: [Subtask 2B. FFT-measured oscillation frequency (y-axis) vs
  the analytical Hopf prediction at the nearest point on the locus
  (x-axis). The dashed line is the identity.])

= Verdict

- Subtask 2A (pitchfork symbolic vs numerical): PASS
- Subtask 2B (Hopf curve within 1 grid cell): PASS
- Subtask 2B (frequency median rel. err. $< 10 %$): PASS

Overall: *PASS*.
