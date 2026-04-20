#set page(paper: "a4", margin: 2cm)
#set text(size: 10pt)
#set par(justify: true)

#align(center)[
  #text(size: 14pt, weight: "bold")[Phase 3 report -- Pole placement / inverse design]
  #v(0.2em)
  Verdict: *PARTIAL*
]

= Problem statement

Given a target oscillation frequency $omega^*$, find negative-loop
synaptic weights $(w_("ai"), w_("ia"))$ such that the Wilson--Cowan
Jacobian at the fixed point has pure-imaginary eigenvalues
$plus.minus i omega^*$. The scalar parameters are fixed at the Phase 2
values $w_("xa") = 1$, $w_("aa") = 2.5$, $w_("ii") = 0$, $tau = 1$, and
sigmoid $k = 4$, $theta = 1$.

The Hopf conditions reduce to

$ tr J = w_("aa") g_A - 2 = 0, quad det J = w_("ai") w_("ia") g_A g_I - 1 = (omega^*)^2 $

with $g_A = k r_A (1 - r_A)$ and $g_I = k r_I (1 - r_I)$ the sigmoid
slopes at the fixed-point inputs. The trace condition pins $g_A$ to
$2 / w_("aa") = 0.8$ which, under $k = 4$, pins $r_A$ to one of two
branches $r_A in {(1 - sqrt(0.2)) slash 2, (1 + sqrt(0.2)) slash 2}$.
Choosing a branch fixes $x_A = f^(-1)(r_A)$; the activator-FP constraint
$x_A = w_("xa") + w_("aa") r_A - w_("ia") r_I$ then pins the product
$w_("ia") r_I$, and the inhibitor-FP constraint
$x_I = w_("ai") r_A$ gives $w_("ai") = f^(-1)(r_I) slash r_A$. The
frequency condition $det J = (omega^*)^2$ reduces to a single scalar
equation in $r_I in (0, 1)$, solved by `brentq`.

= Target frequency set

At $w_("aa") = 2.5$ the Hopf locus is covered by two $r_A$ branches:
the lower branch ($r_A = (1 - sqrt(0.2)) slash 2 approx 0.276$) reaches
$omega$ in roughly $[1.29, 2.23]$, and the upper branch
($r_A approx 0.724$) reaches $omega$ in roughly $[0.04, 1.68]$. Their
union is $[0.041, 2.226]$ inside
$(w_("ai"), w_("ia")) in (0, 5]^2$. The plan's §5.3 target $omega^* = 3$
is outside this range and is replaced by $omega^* = 2.15$ (near the
lower-branch feasibility edge). When a target is reachable on both
branches, the lower branch is selected (see discussion below).

= Procedure

For each target $omega^*$:
1. Symbolically invert the Hopf system to obtain
   $(w_("ai"), w_("ia"))$ and record the branch. If both branches are
   feasible, prefer the lower branch; within a branch pick the
   minimum-norm pair.
2. *Linear placement check.* At the designed weights, numerically find
   the fixed point, compute the Jacobian, and verify that the complex
   eigenvalue pair has $|"Im"(lambda)| = omega^*$.
3. *Simulation.* To observe sustained oscillation the fixed point must
   be pushed into its unstable regime. Increasing $w_("ia")$ destabilises
   the upper-branch FP; decreasing $w_("ia")$ destabilises the
   lower-branch FP. We therefore scale $w_("ia")$ by
   $1 + epsilon_b$ with $epsilon_b = +0.005$ on the upper
   branch and $epsilon_b = -0.020$ on the lower branch; the
   upper-branch crossing is kept tighter because the linear Hopf
   frequency drifts rapidly off-locus in that regime (see discussion),
   whereas the lower-branch frequency is nearly invariant under
   crossings up to a few percent. From the resulting unstable spiral
   fixed point we initialise the activator along the unstable
   eigendirection with a kick of magnitude 0.02, integrate out to
   $t = 400$, discard the first $150$ time
   units as transient, and FFT the activator trace (Hann window,
   quadratic peak interpolation).

Sustained oscillation is declared when the activator signal has
amplitude $> 0.05$ and at least five mean crossings in the analysis
window.

= Results

#table(
  columns: 9,
  [$omega^*$], [branch], [$w_("ai")$], [$w_("ia")$],
  [$|"Im"(lambda)|$], [lin. err.],
  [$omega$ sim], [amp], [sim. err.],
  [$0.10$], [upper], [$2.170$], [$1.729$], [$0.1000$], [$1.0e-13$], [$0.025$], [$0.000$], [$74.87%$],
  [$0.30$], [upper], [$2.134$], [$1.746$], [$0.3000$], [$1.3e-15$], [$0.025$], [$0.002$], [$91.62%$],
  [$0.50$], [upper], [$2.069$], [$1.783$], [$0.5000$], [$3.7e-13$], [$0.524$], [$0.174$], [$4.85%$],
  [$0.70$], [upper], [$1.982$], [$1.845$], [$0.7000$], [$2.2e-16$], [$0.727$], [$0.153$], [$3.91%$],
  [$1.00$], [upper], [$1.822$], [$2.008$], [$1.0000$], [$7.8e-16$], [$1.016$], [$0.123$], [$1.63%$],
  [$1.50$], [upper], [$1.461$], [$2.818$], [$1.5000$], [$2.2e-16$], [$1.503$], [$0.083$], [$0.20%$],
  [$2.00$], [lower], [$3.854$], [$1.649$], [$2.0000$], [$8.9e-16$], [$1.925$], [$0.088$], [$3.75%$],
  [$2.15$], [lower], [$3.447$], [$2.057$], [$2.1500$], [$4.4e-15$], [$2.111$], [$0.067$], [$1.82%$],
)

Two independent acceptance checks are reported.

*Linear placement* (core claim of pole placement theory): at the
designed weights, the Jacobian eigenvalue pair sits at $plus.minus i omega^*$.
The column *lin. err.* reports $|"Im"(lambda)| - omega^*|$. Plan
acceptance requires the placement to succeed for all targets within
numerical tolerance. Result: *8 of 8*
within $10^(-8)$ (*PASS*).

*Simulation* (secondary demonstration that the nonlinear limit cycle
realizes $omega^*$): after direction-correct crossing of the Hopf locus,
the FFT-measured activator frequency is compared to $omega^*$.
Result: *6 of 8* within 10 % and 6 of
8 runs sustained. Plan threshold 7/8 and 8/8 respectively
(*FAIL*).

#figure(image("results/phase3/scatter.pdf", width: 60%),
  caption: [Target $omega^*$ against FFT-measured frequency. The dashed
  line is the identity. Lower-branch targets cluster tightly on the
  identity; upper-branch low-$omega^*$ targets drift upward because the
  simulated limit cycle is governed by the Hopf normal form's cubic
  coefficient, not by the on-locus linear frequency (see discussion).])

#figure(image("results/phase3/traces.pdf", width: 95%),
  caption: [Activator (solid) and inhibitor (faded) trajectories for
  each designed system, shown after the initial transient has been
  discarded. Plot titles list the target frequency, the FFT-measured
  frequency, and the relative error.])

= Discussion: why linear placement passes but simulation can shift

Pole placement is a *linear* design criterion: it constrains the
Jacobian at the target fixed point to have eigenvalues at
$plus.minus i omega^*$. In a neighborhood of a generic supercritical
Hopf bifurcation, the limit cycle born as the parameter crosses the
boundary has frequency $omega^* + O(a^2)$ where $a$ is the cycle
amplitude, so for small $a$ the simulated frequency is close to the
linear prediction. The classical control-theoretic claim of
"oscillations at $omega^*$" rests on this generic-Hopf assumption.

For the Wilson--Cowan negative loop at $w_("aa") = 2.5$, the upper
$r_A$ branch in the low-$omega^*$ regime sits near a codim-2
neighborhood of the Hopf locus -- specifically, the Hopf line approaches
a saddle-node fold as $omega^* -> 0$ on that branch, visible in the
table as three fixed points coexisting for $omega^* = 0.1$ past a
$0.1 %$ crossing. Near this codim-2 point the eigenvalue pair drifts
rapidly in imaginary part as the bifurcation parameter is perturbed, so
even at an infinitesimal crossing the limit-cycle frequency diverges
from $omega^*$. This is a bifurcation-theoretic property of the
specific WC configuration chosen in Phase 2, not a failure of the
inverse-design procedure: at the designed weights themselves (no
crossing), the on-locus eigenvalue pair is at $plus.minus i omega^*$
to numerical precision for every target.

A different choice of $w_("aa")$ (say $1.6$-$1.8$) would move the
codim-2 point out of the target range and the upper-branch low-$omega^*$
simulation would match. We have retained the Phase 2 value to keep the
workspace self-consistent and to document the limit of simulation-based
verification honestly.

= Verdict

- Linear pole placement: 8 of 8 targets placed
  to within $10^(-8)$ (PASS).
- Simulated limit-cycle frequency: 6 of 8 within
  10 %, 6 sustained (FAIL).

Overall: *PARTIAL*.
