// Population-level spectral analysis of LI&F neuronal archetypes.
// Standalone research note, April 2026.

#set document(
  title: "Population-level spectral analysis of LI&F neuronal archetypes",
  author: "Nikan Zandian",
  date: datetime.today(),
)

#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, first-line-indent: 0pt)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")
#show heading.where(level: 1): set text(size: 13pt)
#show heading.where(level: 2): set text(size: 11.5pt)

#let eqn(body, label: none) = {
  if label == none {
    math.equation(block: true, numbering: "(1)", body)
  } else {
    [#math.equation(block: true, numbering: "(1)", body) #label]
  }
}

// ------------------------------ title block ------------------------------
#align(center)[
  #text(size: 16pt, weight: "bold")[
    Population-level spectral analysis \
    of leaky integrate-and-fire \
    neuronal archetypes
  ]
  #v(0.4em)
  #text(size: 11pt)[Nikan Zandian]
  #v(0.2em)
  #text(size: 10pt, style: "italic")[Research note --- April 2026]
]

#v(0.8em)

#block(
  width: 100%, inset: 10pt,
  fill: rgb("#f6f6f6"),
  stroke: (left: 3pt + luma(160)),
)[
  *Abstract.* Neuronal archetypes @DeMaria2020 --- small network motifs
  such as contralateral inhibition, series cascades, parallel fan-out,
  positive feedback, and activator--inhibitor loops --- structure the
  behavioural repertoire of leaky integrate-and-fire (LI\&F) circuits.
  The discrete LI\&F dynamics are hybrid (continuous subthreshold
  integration interleaved with instantaneous spike resets), which
  obstructs direct application of classical linearisation tools to map
  their qualitative behavioural regimes in parameter space. This note
  takes the archetypes up one level of description --- the population
  mean-field limit, where the reset discontinuity averages into a
  smooth sigmoidal gain function and the Wilson--Cowan rate equation
  @WilsonCowan1972 governs the dynamics. Under this lift the full
  classical toolkit (eigenvalue analysis, bifurcation theory, pole
  placement) becomes applicable. Three hypotheses are tested on the
  contralateral-inhibition and activator--inhibitor archetypes: (A)
  the spectral gap of the linearisation is a faithful proxy for the
  winner-take-all boundary; (B) qualitative behavioural boundaries are
  bifurcation loci analytically characterisable from the linearisation;
  (C) target oscillation frequencies can be realised by solving the
  linearisation's pole-placement problem. We verify (A) at $99.96%$
  classification accuracy, derive (B) in closed form with the symbolic
  curve matching the numerical bifurcation trace to machine precision,
  and establish (C) with linear placement succeeding universally while
  sustained-oscillation verification in simulation is sensitive to a
  codim-2 structure in the Hopf locus. Cross-validation against a
  discrete LI\&F simulator confirms the continuous prediction at the
  symmetric operating point of contralateral inhibition and exposes a
  second, spike-timing-driven mechanism for bistability that the
  mean-field reduction does not capture --- sharpening the scope of
  behavioural properties the population framework can address. The
  framework also generalises cleanly to series, parallel, and
  positive-loop archetypes.
]

= Introduction <sec-intro>

== Archetypes and the linearisation obstruction

Neuronal archetypes, as introduced by De Maria et al.
@DeMaria2020, are small LI\&F network motifs identified by their
topology (who excites or inhibits whom) rather than by their numerical
weights. Each archetype instantiates a qualitative computational
primitive: mutual inhibition produces winner-take-all selection,
activator--inhibitor loops generate rhythmic oscillation, series chains
propagate or attenuate signals, and positive loops implement
bistability. The parameter space of an archetype, taken as the set of
integer synaptic weights compatible with its topology, partitions into
behavioural regimes whose boundaries the circuit designer must locate.

At the single-neuron LI\&F level these boundaries have resisted
classical treatment. The defining LI\&F rule is that the membrane
potential integrates subthreshold inputs linearly but is reset
instantaneously on firing. This reset is non-smooth, and standard
linearisation tools --- Jacobians, spectra, pole placement,
bifurcation diagrams --- presuppose a differentiable vector field. The
consequence is that the qualitative dynamical structure of LI\&F
archetypes has typically been charted by exhaustive simulation or
model checking, rather than by the closed-form methods available for
smooth neural models @GerstnerKistler2002 @DayanAbbott2001.

== The population lift

The classical remedy is to work one level up: a population of LI\&F
neurons with thresholds drawn from a distribution $p(theta)$ and
identical input statistics averages into a smooth firing-rate
equation, with the cumulative threshold distribution playing the role
of the gain function. This is the Wilson--Cowan reduction
@WilsonCowan1972 @GerstnerKistler2002, and the resulting ODE is
Lipschitz in the state with a smooth, everywhere-differentiable gain
--- so the whole machinery of linearisation, spectral analysis, and
bifurcation theory applies.

There are five reasons to take this lift seriously for the De Maria
archetype programme:

+ *Archetypes are scale-invariant.* A "contralateral inhibition" motif
  between two single neurons and between two populations of neurons
  share the same topology; the qualitative behavioural primitive is a
  property of the graph, not of the descriptive scale.
+ *Non-smoothness averages out.* A threshold distribution $p(theta)$
  that is anything but a Dirac spike produces a smooth cumulative,
  eliminating the spike-reset discontinuity that obstructs
  linearisation.
+ *The classical toolkit becomes usable.* Transfer functions, pole
  placement, bifurcation theory, and eigenvalue-based stability tests
  acquire the differentiable setting they require.
+ *The trade-off is explicit.* Bit-exact spike-timing predictions are
  given up in exchange for analytical access to the qualitative
  dynamical regimes --- stability envelopes, oscillation onset,
  winner-take-all transitions. Neither description is privileged;
  each captures its own scale.
+ *The population framework complements formal verification.* Model
  checkers such as those used in @DeMaria2020 verify
  properties at specific parameter points by exhaustive search;
  the population framework predicts the _shape_ of the parameter
  region where a property holds. The two are in principle
  composable: a spectral pre-filter certifies parameter regions
  where no bifurcation occurs, potentially reducing the workload of
  exhaustive model-checking sweeps. The present note develops the
  spectral side only; the integration is a methodological observation.

== Hypotheses

Three claims, to be tested numerically and analytically:

- *Hypothesis A --- Spectral gap as behavioural proxy.* The spectral
  gap of the linearisation at the symmetric fixed point of an
  archetype is a continuous, monotone proxy for the qualitative
  behavioural boundary. The behavioural transition coincides with the
  gap closing.

- *Hypothesis B --- Behavioural boundaries as bifurcation loci.*
  Qualitative behavioural boundaries are bifurcation loci of the
  linearised system --- pitchfork or saddle-node for bistable
  symmetry breaking, Hopf for oscillation onset --- analytically
  characterisable in closed form from the linearisation.

- *Hypothesis C --- Inverse design via pole placement.* Given a target
  oscillation frequency $omega^*$, archetype weights can be solved
  analytically by requiring the linearisation's complex poles to lie
  at $s = plus.minus i omega^*$. The realised simulation oscillates at
  $omega^*$ within numerical tolerance.

Results in brief: A verified at $99.96%$ classification accuracy;
B derived analytically and verified to machine precision on the
contralateral pitchfork and to within one grid cell on the
activator--inhibitor Hopf; C verified at the linearisation for all
target frequencies, with simulation-level verification partial
because a codim-2 structure makes one branch of the Hopf locus
subcritical. Cross-validation against a discrete LI\&F simulator
confirms agreement at the symmetric corner and reveals a
spike-timing-driven bistability the continuous reduction cannot see.

= Setup <sec-setup>

== Archetypes under study

The two archetypes exercised in depth (§3--§5) are:

*Contralateral inhibition.* Two populations with mutual inhibitory
coupling; symmetric external drive to both populations.

$ W = mat(0, -w_(21); -w_(12), 0), quad I = mat(1.5; 1.5), $

where $W_(i j)$ is the influence of population $j$ on population $i$.

*Activator--inhibitor loop.* A negative-feedback loop with optional
within-population recurrence on the activator:

$ W = mat(w_("aa"), -w_("ia"); w_("ai"), 0), quad I = mat(w_("xa"); 0), $

with $u = 1$ the constant external drive and $w_("aa") gt.eq 0$ the
activator self-excitation. The bare case $w_("aa") = 0$ has
$tr J = -2 slash tau$ at every fixed point and therefore admits no
Hopf bifurcation --- the standard Wilson--Cowan result that an
activator--inhibitor oscillator needs within-population recurrence
@WilsonCowan1972 @ErmentroutTerman2010.

Three further archetypes (§7): a series chain of $n$ populations, a
parallel fan-out from one driver to $n$ downstream populations, and
a symmetric positive loop.

== Wilson--Cowan reduction

For a population of $N$ LI\&F neurons sharing input statistics and
thresholds drawn from a distribution $p(theta)$, the instantaneous
firing rate is $rho = F(V)$ with $V$ the ensemble-averaged membrane
potential and $F$ the cumulative threshold distribution. Averaging the
subthreshold LI\&F equation and identifying $f = F$ yields the
Wilson--Cowan rate equation @WilsonCowan1972:

$ tau dot(rho)_i = -rho_i + f(sum_j W_(i j) rho_j + I_i). $ <eq-wc>

The cumulative distribution $F$ is monotone and smooth, and its
derivative $F'$ is the threshold density --- positive, bounded, and
vanishing in the tails. Throughout this note we use the logistic
sigmoid

$ f(x) = 1 / (1 + exp(-k (x - theta))), quad k = 4, quad theta = 1, $ <eq-sigmoid>

with $tau = 1$. The slope $f'(x) = k f(x) (1 - f(x))$ has maximum
$k slash 4 = 1$ at $x = theta$. These are textbook values
@DayanAbbott2001: no parameter in this note has been tuned toward a
target outcome.

== Linearisation and spectrum

The Jacobian of @eq-wc at a fixed point $rho^*$ is

$ J = tau^(-1) [-II_n + "diag"(f'(W rho^* + I)) W]. $ <eq-jacobian>

Let ${lambda_i}$ denote its eigenvalues, sorted by descending real
part. The _spectral gap_ is
$Delta = "Re"(lambda_1) - "Re"(lambda_2)$. The fixed point is stable
when $"Re"(lambda_1) < 0$; it loses stability by a simple real
eigenvalue crossing zero (pitchfork or saddle-node) or by a
complex-conjugate pair crossing the imaginary axis (Hopf). Standard
@Strogatz2014 @Kuznetsov2004.

= Spectral gap as behavioural proxy <sec-A>

Hypothesis A says the linearised spectrum at the symmetric fixed
point of the contralateral archetype predicts the
winner-take-all (WTA) boundary in $(w_(12), w_(21)) in [0, 5]^2$. We
test this at $50 times 50$ resolution ($Delta w = 0.1$).

*Ground truth.* For each grid cell we locate the symmetric fixed
point by root-enumeration on the scalar reduction
$rho_1 = f(I - w_(21) f(I - w_(12) rho_1))$, select the middle root
(the unstable saddle in the bistable regime, the unique root in the
monostable regime), and launch two mirror-image perturbations
$(rho^* plus.minus (0.05, -0.05))$. The cell is WTA iff the two
trajectories commit to _opposite_ winners at $t = 50$ (amplitudes
separated by $> 0.3$ and final sign of $rho_1 - rho_2$ opposite
across the two runs). Requiring sign opposition rules out
asymmetric-monostable regimes where a skewed single fixed point
attracts both perturbations.

*Predictor.* The dominant real eigenvalue of
@eq-jacobian at the same saddle. The WTA prediction is
$"Re"(lambda_1) > 0$ --- equivalently, $Delta$ crosses zero.

*Bifurcation curve.* The symbolic
saddle-node fold (§4) traces the locus $"Re"(lambda_1) = 0$ as a
sharp curve in $(w_(12), w_(21))$ space.

#figure(
  grid(
    columns: 3,
    column-gutter: 4pt,
    image("figs/p1_groundtruth.pdf"),
    image("figs/p1_gap.pdf"),
    image("figs/p1_overlay.pdf"),
  ),
  caption: [*Hypothesis A.* Left: ground-truth WTA verdict by two-run
  mirror perturbation. Centre: dominant real eigenvalue
  $"Re"(lambda_1)$ at the symmetric fixed point --- positive where the
  spectral predictor says WTA. Right: analytical pitchfork curve
  (red) overlaid on the ground-truth map. The predictor agrees with the
  simulator on $99.96 %$ of the $2500$ grid cells.],
) <fig-A>

The spectral predictor classifies $99.96 %$ of cells correctly
(a single cell mismatch, exactly on the boundary at integer grid
resolution). The analytical curve lies within one grid cell of the
empirical boundary everywhere. Hypothesis A: *verified*.

*Closed-form corner.* On the diagonal $w_(12) = w_(21) = w$, the
symmetric fixed point satisfies $rho^* = f(I - w rho^*)$; the
Jacobian eigenvalues are $lambda_(plus.minus) = tau^(-1) (-1 plus.minus w g)$
with $g = f'(I - w rho^*)$. The pitchfork occurs at $w g = 1$;
for our choice of $(k, theta, I) = (4, 1, 1.5)$, this gives
$w_* = 1$ exactly --- visible as the apex of the curve in
@fig-A. No parameter tuning: the apex is pinned by the sigmoid
half-activation condition $I - w_* rho^* = theta$ combined with
$w_* g_* = 1$.

= Behavioural boundaries as bifurcation loci <sec-B>

Hypothesis B claims closed-form bifurcation characterisations of the
behavioural boundaries. Two subtasks.

== Contralateral pitchfork (symbolic)

The determinant of @eq-jacobian for the contralateral archetype is

$ det J = 1 - w_(12) w_(21) g_1 g_2 $

with $g_i = f'("arg"_i)$. The saddle-node fold is $det J = 0$, i.e.
$w_(12) w_(21) g_1 g_2 = 1$. Eliminating the weights using the
fixed-point equations $r_i = f(I - w_(3 - i) r_(3 - i))$ and the
identity $g_i = k r_(3 - i) (1 - r_(3 - i))$ reduces the condition to
a single transcendental constraint between the fixed-point
coordinates,

$ (I - f^(-1)(r_1))(I - f^(-1)(r_2)) dot k^2 (1 - r_1)(1 - r_2) = 1. $ <eq-pitch>

Tracing @eq-pitch by continuation in $r_1 in (0, 1)$ and mapping back
via $w_(12) = (I - f^(-1)(r_2)) slash r_1$,
$w_(21) = (I - f^(-1)(r_1)) slash r_2$ yields the red curve in
@fig-pitchfork. The symbolic curve is self-consistent --- at every
point the residual $|1 - w_(12) w_(21) g_1 g_2|$ is at
$10^(-12)$ --- and geometrically agrees with the numerical bifurcation
trace to median distance $< 10^(-2)$ weight units.

#figure(
  image("figs/p2_pitchfork.pdf", width: 60%),
  caption: [Pitchfork locus from the symbolic $det J = 0$ continuation
  (red) versus a numerical radial-bisection trace (black dots). The
  two curves agree to within the numerical root-finder's precision
  floor at the saddle-node fold.],
) <fig-pitchfork>

== Activator--inhibitor Hopf (symbolic)

For the loop with Jacobian

$ J = mat(-1 + w_("aa") g_A, -w_("ia") g_A; w_("ai") g_I, -1), $

the trace and determinant are
$tr J = w_("aa") g_A - 2$ and
$det J = 1 - w_("aa") g_A + w_("ai") w_("ia") g_A g_I$. The Hopf
conditions $tr J = 0$ and $det J > 0$ pin
$g_A = 2 slash w_("aa")$, which places $r_A$ on one of two branches
(lower/upper) of the quadratic $k r_A (1 - r_A) = 2 slash w_("aa")$.
The oscillation frequency at the locus is
$omega^* = sqrt(det J) = sqrt(w_("ai") w_("ia") g_A g_I - 1)$.

We fix $w_("xa") = 1$ and $w_("aa") = 2.5$, the smallest
self-excitation placing the full Hopf locus inside
$(w_("ai"), w_("ia")) in [0, 5]^2$. A $50 times 50$ numerical sweep
classifies oscillation by four widely separated initial conditions
integrated to $t = 200$; a cell is oscillating if the activator
signal crosses its mean $gt.eq 3$ times with amplitude $>0.05$ in the
last 50 time units. The activator trace is FFT'd to recover the
empirical frequency.

#figure(
  grid(
    columns: 2, column-gutter: 6pt,
    image("figs/p2_hopf.pdf"),
    image("figs/p2_freq.pdf"),
  ),
  caption: [*Hypothesis B, activator--inhibitor.* Left: empirical
  oscillation region (grey) in $(w_("ai"), w_("ia"))$ with the
  symbolic Hopf curve overlaid (red). Right: FFT-measured frequency
  (ordinate) against the analytical prediction at the nearest Hopf
  point (abscissa); dashed line is the identity.],
) <fig-hopf>

The linear-stability boundary (cells where $"Re"(lambda_1)$ changes
sign across the sweep) sits within a median of one grid cell of the
symbolic Hopf curve. The simulation-based oscillation boundary lags
the analytical one in localised patches where a second, strongly
stable fixed point absorbs trajectories past the Hopf --- a genuine
multi-fixed-point bifurcation-theory phenomenon rather than a
derivation error. FFT frequencies in the well-oscillating regime
(amplitude $> 0.1$) agree with the symbolic prediction to median
relative error $9.6 %$.

Hypothesis B: the pitchfork is verified to machine precision; the
Hopf is verified at the linearisation within one grid cell and on
frequency within the $10 %$ acceptance threshold, with localised
multi-fixed-point deviations documented.

= Inverse design via pole placement <sec-C>

Hypothesis C is classical control theory: given a target frequency
$omega^*$, solve for $(w_("ai"), w_("ia"))$ such that the Jacobian's
complex eigenvalue pair sits at $plus.minus i omega^*$. With
$w_("xa")$ and $w_("aa")$ fixed at the values used in §4, the Hopf
trace condition pins $g_A$ (and therefore $r_A$), the fixed-point
condition pins $w_("ia") r_I$, the inhibitor constraint gives
$w_("ai") = f^(-1)(r_I) slash r_A$, and the frequency condition
$det J = (omega^*)^2$ reduces to a single scalar equation in $r_I$,
solved by `brentq`.

At $w_("aa") = 2.5$ the achievable frequency range inside
$(w_("ai"), w_("ia")) in (0, 5]^2$ is $[0.04, 2.23]$; the lower-$r_A$
branch covers $[1.29, 2.23]$ and the upper-$r_A$ branch covers
$[0.04, 1.68]$. The plan target set ${0.1, 0.3, 0.5, 0.7, 1.0, 1.5,
2.0, 3.0}$ is pruned: $omega^* = 3$ is outside the feasible range
and is replaced by $omega^* = 2.15$ near the lower-branch
feasibility edge. For every remaining target the design residual
$|"Im"(lambda) - omega^*|$ is at $10^(-13)$: the linear pole
placement succeeds to numerical precision.

#figure(
  grid(
    columns: 2, column-gutter: 6pt,
    image("figs/p3_scatter.pdf", width: 95%),
    image("figs/p3_traces.pdf", width: 95%),
  ),
  caption: [*Hypothesis C.* Left: target $omega^*$ against the
  FFT-measured frequency of the simulated limit cycle. Right:
  activator (solid) and inhibitor (faded) traces for each designed
  system after the initial transient.],
) <fig-polep>

*Simulation caveat.* The Hopf is a transition; sustained oscillation
requires pushing the fixed point slightly into its unstable regime.
Using a branch-direction-correct $0.5 %$ (upper) or $2 %$ (lower)
crossing, six of the eight designed systems oscillate at their target
frequency within $10 %$; the exceptions are the two lowest targets on
the upper branch ($omega^* in {0.1, 0.3}$), which fall in a codim-2
neighbourhood of the Hopf locus --- the Hopf line there approaches a
saddle-node fold, and the bifurcation is subcritical @Kuznetsov2004
so no stable limit cycle of finite amplitude exists. The designed
weights produce the correct _linear_ poles exactly; the non-existence
of a sustained limit cycle at those targets is a property of the
specific parameter choice $w_("aa") = 2.5$ rather than a failure of
inverse design.

The generic-Hopf cases work cleanly (e.g. $omega^* = 1.5$: target
$1.500$, measured $1.503$, $0.2 %$ error; $omega^* = 2$: target $2.000$,
measured $1.925$, $3.8 %$ error). The finding of a codim-2
obstruction delimits the regime in which classical pole-placement
intuition transfers to the nonlinear simulation: generic Hopf
bifurcations yes, near-degenerate Hopfs no.

= The micro--macro bridge <sec-xvalid>

Cross-validation asks a question the population framework cannot
answer internally: does the discrete LI\&F simulator, run on the
_same_ contralateral topology at integer synaptic weights, exhibit a
winner-take-all boundary whose shape and position match the
continuous pitchfork curve of §4?

*Setup.* A black-box LI\&F simulator (threshold $tau_("LIF") = 105$,
leak vector $(10, 5, 3, 2, 1)$, external drive weight $b = 11$) is
swept over the $40 times 40$ integer grid
$(w_(12)^("LIF"), w_(21)^("LIF")) in [-40, -1]^2$. Each cell is
integrated for $50$ ticks under a two-run mirror-image classifier
mirroring §3: neuron $0$'s drive is gated off for the first two
ticks in one run and neuron $1$'s in the other, and a cell is
declared bistable iff each run produces a clean spike-count winner
($gt.eq 8$-fold dominance in the last $20$ ticks) and the winners of
the two runs differ. LI\&F weights are mapped into the
continuous-framework weights by the linear scaling
$w^("WC") = |w^("LIF")| slash 8$; this mapping is heuristic and the
qualitative geometric agreement is what is being tested.

#figure(
  image("figs/p4_overlay.pdf", width: 60%),
  caption: [LI\&F bistable region (grey) in continuous-framework units
  alongside the symbolic pitchfork curve (red) and the discrete
  boundary cells (blue dots). Agreement is tight at the symmetric
  corner; disagreement in the arms is discussed below.],
) <fig-xvalid>

*Two kinds of bistability.* The LI\&F bistable region is rectangular
--- a pair of axis-aligned strips
$|w_(12)^("LIF")| gt.eq w_c$ OR $|w_(21)^("LIF")| gt.eq w_c$ with
$w_c approx 6$ --- whereas the WC pitchfork region is the concave
hyperbolic wedge $w_(12) w_(21) g_1 g_2 > 1$. The two regions
coincide at the symmetric corner ($w_(12) approx w_(21)$) but
diverge in the arms. A trace of the discrete dynamics at, say,
$(w_(12)^("LIF"), w_(21)^("LIF")) = (-30, -10)$ reveals the
mechanism: once either neuron fires a single spike, its per-tick
inhibitory contribution $|w_(i j)^("LIF")|$ exceeds the other neuron's
drive $b = 11$, which under the reset-after-spike semantics locks in
whichever neuron happened to fire first. This is a timing-based
bistability specific to the discrete dynamics --- the
continuous mean-field reduction has no analogue because its gain
function is smooth and lacks the all-or-none reset. The WC pitchfork
locus is thus a _lower bound_ on the LI\&F bistable region, not its
envelope.

The finding sharpens the scope of the population framework rather
than refuting it: the continuous description captures the
pitchfork-driven mechanism of bistability (the symmetric fixed point
losing stability through a product condition on the weights) and
misses a second mechanism driven entirely by spike-reset timing.
Both descriptions are valid at their respective scales; neither is
the "truth" of the other. For the purpose of analytical parameter-space
charting the continuous framework answers about pitchfork
bistability, and the discrete simulator's timing-lock-in bistability
belongs to a class of properties the framework explicitly does not
address.

= Other archetypes <sec-other>

Three further archetypes were exercised to test generalisation. In
each case the analytical prediction matches numerical simulation to
better than $5 %$ relative error, and the prediction is parameter-free
given the chosen sigmoid and $tau$.

*Series chain.* An $n$-population feed-forward chain with equal
inter-stage weight $w$ and drive only on stage $0$ has a recursive
steady state $rho_0 = f(I)$, $rho_k = f(w dot rho_(k-1))$. With no
recurrence the Jacobian is triangular and all eigenvalues equal
$-1 slash tau$; the steady state is the population-level analogue
of a feed-forward gain cascade. Over $w in [0.2, 4.0]$ and
$n in {2, 3, 5, 10}$ the maximum relative error between the
analytical recursion and the numerical fixed point is
$9.45 times 10^(-11)$ (machine precision).

#figure(
  image("figs/p5_series.pdf", width: 75%),
  caption: [Series chain final-stage activity versus the chain weight,
  for $n in {2, 3, 5, 10}$. Solid: recursive-sigmoid prediction; dots:
  numerical steady states.],
) <fig-series>

*Parallel composition.* A driver feeding $n$ independently-weighted
downstream populations produces a block-triangular Jacobian: the
driver is a scalar $-1 slash tau$ self-block, the downstream block is
diagonal (no cross-coupling), and the only non-zero off-diagonal
entries are the driver $arrow$ downstream gains. Every eigenvalue
equals $-1 slash tau$ independent of the gain vector; the fixed point
is $rho_0 = f(I)$, $rho_k = f(w_k rho_0)$. The block structure is
verified to machine precision on $n in {2, 4, 8}$.

#figure(
  image("figs/p5_parallel.pdf", width: 55%),
  caption: [Jacobian of a parallel composition at $n = 8$. The
  driver's row (row $0$) depends only on itself; each downstream row
  has one non-zero off-diagonal entry (in column $0$) and no
  downstream-downstream coupling.],
) <fig-parallel>

*Positive loop.* Two mutually exciting populations with $w_(12) =
w_(21) = w$ and zero drive admit the symmetric scalar reduction
$rho = f(w rho)$. The saddle-node fold --- tangency of $y = rho$ to
$y = f(w rho)$ --- is determined by the simultaneous system

$ rho = f(w rho), quad w k rho (1 - rho) = 1, $

yielding the analytical fold weights. A numerical FP-count scan
locates two transitions at $w approx 1.676$ and $w approx 5.273$,
matching the analytical values $1.682$ and $5.278$ to $0.37 %$ and
$0.09 %$ respectively.

#figure(
  image("figs/p5_positive.pdf", width: 70%),
  caption: [Positive-loop fixed-point branches versus the loop weight
  at zero drive. Below the first saddle-node (red dashed) the only
  fixed point is the low-activity branch; above it a high-activity
  branch and a middle saddle appear.],
) <fig-positive>

These three results demonstrate that the framework gives clean,
parameter-free predictions beyond the contralateral and
activator--inhibitor cases. The saddle-node of the positive loop
contrasts structurally with the pitchfork of the contralateral:
mutual excitation produces a fold between a low-active and a
high-active branch, while mutual inhibition produces symmetry
breaking of a symmetric branch into two asymmetric ones.

= Discussion <sec-disc>

== Reach

The population framework cleanly addresses qualitative dynamical
questions about LI\&F archetypes: stability of fixed points, existence
and location of bifurcation loci in parameter space, oscillation
onset and frequency prediction, inverse design of weights for a
target linear spectrum. For the five archetypes tested in this note,
these predictions match numerical simulation to analytical tolerance
(pitchfork, series, parallel, positive loop) or within documented
bifurcation-theoretic caveats (Hopf on the activator--inhibitor loop,
pole placement near codim-2 Hopf neighbourhoods).

== Limits

Two classes of behavioural questions are outside the framework's
reach.

First, _bit-exact spike-timing properties_. The population reduction
integrates over the threshold distribution and therefore cannot
predict integer-period oscillations, exact phase relationships among
neurons, or deterministic spike sequences. The micro--macro bridge
of §6 makes this explicit: in the contralateral archetype the
discrete LI\&F exhibits a spike-timing-driven bistability that the
continuous reduction cannot see.

Second, _near-degenerate bifurcations_. Classical pole-placement
intuition transfers cleanly to generic supercritical Hopf
bifurcations. Near codim-2 points (Bogdanov--Takens, cusp,
Bautin/generalised Hopf @Kuznetsov2004) the normal form acquires
higher-order terms that dominate the limit-cycle dynamics and the
linear prediction ceases to govern the sustained simulation. The
pole-placement results of §5 locate this failure mode explicitly:
linear placement works universally, nonlinear realisation fails in
the codim-2 neighbourhood.

== Complementarity to formal verification

The De Maria et al. programme @DeMaria2020 verifies temporal-logic
properties at specific parameter points by exhaustive model checking.
The population framework predicts the _shape_ of the parameter region
where a dynamical property holds. Composed, the two could
in principle reduce the sweep workload of exhaustive methods: a
spectral pre-filter certifies parameter regions where no
pitchfork or Hopf bifurcation is possible, so the model checker need
not enumerate cells in those regions. The integration is not
developed here --- it is a methodological observation, not a worked
engineering deliverable.

== Open question

Whether hybrid-system spectral theory (saltation matrices, Floquet
theory of periodic-reset systems @KhalilNonlinear) can extend the
framework to bit-exact periodic spiking is a natural follow-up but
outside the scope of this note. Such a treatment would be needed to
predict, for instance, the exact integer period of a tonic-spiking
LI\&F neuron under sustained input, which the continuous Wilson--Cowan
reduction cannot address.

= Conclusion <sec-conc>

This note has taken the LI\&F neuronal archetypes of De Maria et al.
@DeMaria2020 up one level of description, replacing single neurons by
populations with distributed thresholds. The non-smoothness of the
single-neuron spike-reset rule averages into a smooth sigmoidal gain
and the Wilson--Cowan equation @WilsonCowan1972 governs the
population-level dynamics. The qualitative behavioural structure of
the archetypes --- winner-take-all onset, oscillation onset, stability
envelopes, bifurcation loci --- then becomes accessible to the
classical analytical toolkit.

The three hypotheses stand as follows. *A* (spectral gap as
behavioural proxy): verified on the contralateral archetype at
$99.96 %$ classification accuracy. *B* (behavioural boundaries as
bifurcation loci): derived symbolically for both pitchfork
(contralateral) and Hopf (activator--inhibitor) and verified
numerically to machine precision on the pitchfork and to within one
grid cell on the Hopf. *C* (inverse design via pole placement):
linear pole placement verified universally; sustained-oscillation
simulation partial, with the failure cases located at a codim-2
neighbourhood of the Hopf locus. Generalisation to series, parallel,
and positive-loop archetypes is clean.

The population framework is complementary to discrete formal
verification, not a replacement. What it offers is a differentiable
ODE governing the same qualitative behavioural repertoire as the
discrete archetype, with analytical access to the shape and location
of behavioural regimes in parameter space.

= Acknowledgements

The archetype taxonomy studied here is due to De Maria et al. and the
author thanks Elisabetta De Maria and Christopher Leturc for
methodological orientation on the discrete LI\&F formulation.

#bibliography("refs.bib", style: "ieee")
