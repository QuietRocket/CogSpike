// Population-level spectral analysis of LI&F neuronal archetypes.
// v2 (April 2026): rewritten for an FCS-native audience.

#set document(
  title: "Population-level spectral analysis of LI&F neuronal archetypes (v2)",
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

// ------------------------------ title block ------------------------------
#align(center)[
  #text(size: 16pt, weight: "bold")[
    Population-level spectral analysis \
    of leaky integrate-and-fire \
    neuronal archetypes
  ]
  #v(0.4em)
  #text(size: 10pt, style: "italic")[v2 --- written for an FCS-native audience]
  #v(0.2em)
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
  *Abstract.* The neuronal archetypes of De Maria et al. @DeMaria2020
  are small LI\&F network motifs whose qualitative behavioural primitives
  --- contralateral inhibition produces winner-take-all selection,
  activator--inhibitor loops oscillate, positive loops are bistable ---
  are properties of the topology, not the integer synaptic weights. The
  FCS approach charts these behaviours by Lustre encoding plus model
  checking with Kind2, on integer-grid sweeps. The discrete LI\&F dynamics
  are hybrid (continuous subthreshold integration interleaved with
  instantaneous spike resets), which obstructs direct application of
  classical bifurcation tools to map the behavioural regimes
  analytically. This note takes the archetypes up one level of
  description: a population of LI\&F neurons with distributed thresholds
  averages into a smooth Wilson--Cowan rate equation
  @WilsonCowan1972 governed by a sigmoidal gain function, and the
  classical toolkit (Jacobians, eigenvalues, bifurcation theory, pole
  placement) becomes immediately usable. We rebuild the contralateral
  archetype in this framework and show that the winner-take-all
  mechanism falls out of two lines of algebra: mutual inhibition is a
  feedback loop with gain $w_(12) w_(21) g_1 g_2$, and the
  symmetry-breaking pitchfork is the unity contour of that loop gain ---
  the same Barkhausen criterion used in classical electronics. Three
  hypotheses on the population framework are then tested numerically:
  (A) the spectral gap predicts the winner-take-all boundary at $99.96 %$
  classification accuracy; (B) the pitchfork and Hopf loci are
  derivable in closed form; (C) target oscillation frequencies are
  achievable by linear pole placement. The framework generalises to
  series, parallel, and positive-loop archetypes. Cross-validation
  against the discrete LI\&F simulator agrees at the symmetric corner
  and reveals a second, spike-timing-driven bistability that the
  population framework cannot see --- sharpening the scope of behaviours
  the framework can address. The population framework is offered as a
  *companion* to the FCS programme, not a replacement: it predicts the
  shape of behavioural parameter regions, where Lustre/Kind2 verifies
  properties at specific points.
]

= Introduction <sec-intro>

== Why the lift

The FCS programme @DeMaria2020 verifies temporal-logic properties of
LI\&F archetypes by encoding them as Lustre nodes and discharging the
properties to model checkers (Kind2) and theorem provers (Coq). The
verification is point-wise: each cell of an integer weight grid is
either certified (the property holds) or refuted (with a Lustre
counter-example trace). The approach is rigorous, automatic, and
delivers the kind of formal guarantee that makes archetypes a
practical engineering primitive.

Two questions, however, lie just outside its natural reach. First,
*the shape of the parameter region*. FCS Fig. 10 shows that
contralateral inhibition's winner-take-all stabilises within four
ticks once the synaptic weights are large enough; what is the
analytical form of that boundary, and how does it scale beyond the
sweep range $|w^("LIF")| <= 40$ on which Fig. 10 is plotted? Second,
*inverse design*. Given a target oscillation frequency, what
synaptic weights realise it? The FCS apparatus can verify a candidate
weight assignment but does not solve for one.

The obstruction to addressing both questions with classical methods
is the LI\&F reset rule itself. The membrane potential integrates
linearly until it crosses threshold, then resets instantaneously to
zero. This non-smoothness rules out Jacobians, spectra, pole
placement, and bifurcation diagrams in any direct sense, because
those tools all presuppose a differentiable vector field. The
qualitative dynamical structure of LI\&F archetypes has therefore
been charted by exhaustive simulation or model checking, rather than
by the closed-form methods available for smooth neural models
@GerstnerKistler2002 @DayanAbbott2001.

The classical remedy is to work one level up. A population of LI\&F
neurons with thresholds drawn from a distribution $p(theta)$ averages
into a smooth firing-rate equation, with the cumulative threshold
distribution playing the role of the gain function (#ref(<sec-wc>) gives
the derivation). The non-smoothness is not papered over --- it is
*averaged away* by the population. The resulting Wilson--Cowan ODE
@WilsonCowan1972 is differentiable everywhere and admits the full
classical toolkit. This note develops the spectral analysis of De
Maria archetypes in this lifted framework and confronts its
predictions with the discrete LI\&F simulator they came from.

== Hypotheses and headline results

Three claims, tested on the contralateral and activator--inhibitor
archetypes:

- *Hypothesis A --- Spectral gap as behavioural proxy.* The dominant
  eigenvalue of the linearised Wilson--Cowan system at the symmetric
  fixed point predicts the winner-take-all boundary.
- *Hypothesis B --- Behavioural boundaries as bifurcation loci.* The
  pitchfork (contralateral) and Hopf (activator--inhibitor) loci are
  derivable in closed form from the linearisation.
- *Hypothesis C --- Inverse design via pole placement.* Given a target
  oscillation frequency $omega^*$, archetype weights can be solved by
  requiring the linearisation's complex poles at $plus.minus i omega^*$.

In brief: A holds at $99.96 %$ classification accuracy on a $50 times
50$ grid; B is verified to machine precision on the pitchfork, within
one grid cell on the Hopf; C succeeds universally at the linearisation,
with sustained-oscillation simulation partial because of a codim-2
neighbourhood of the Hopf locus. The framework also generalises to
series, parallel, and positive-loop archetypes
(#ref(<sec-other>)).

== Complementarity with FCS

The framework is complementary to FCS, not a substitute. FCS verifies
*at a point*: given specific integer weights, does the property hold?
The population framework predicts *the shape*: across continuous weight
space, where does the bifurcation lie? Composed in principle, a
spectral pre-filter could certify regions where no bifurcation occurs,
reducing the workload of exhaustive model-checking sweeps; we treat
this as a methodological observation, not a delivered engineering
deliverable. The cross-validation in #ref(<sec-xvalid>) makes one
direction of the relationship explicit: at the symmetric corner of
the contralateral sweep the discrete and continuous descriptions agree;
in the asymmetric arms the discrete simulator exhibits a
spike-timing-driven bistability that the continuous reduction cannot
see.

== Scaling motivation

A specific question motivated this note. FCS Fig. 10 plots the
winner-take-all stabilisation property on the contralateral grid
$(w_(12)^("LIF"), w_(21)^("LIF")) in [-40, -1]^2$. The boundary
between teal (stabilises in 4 ticks) and red (does not) sits around
$|w^("LIF")| approx 30$. Does that boundary continue smoothly as
weights grow further, or do new features emerge? An exhaustive sweep
out to (say) $|w| = 200$ in Lustre is feasible but tedious; an
analytical answer in continuous parameter space is preferable. The
population framework gives that answer (#ref(<sec-other>),
#ref(<sec-xvalid>)): the rectangular-strip boundary persists at
$|w^("LIF")| approx 7$ unchanged out to $|w| = 200$, with no new
features. The continuous theory, in turn, predicts a *different*
boundary --- the pitchfork hyperbola --- that lies at the symmetric
corner where it agrees with the LI\&F result and bends through the
arms where the LI\&F simulator's spike-timing locks dominate the
geometry. The two boundaries together delimit the scope where each
description applies.

= From discrete LI\&F to a population rate equation <sec-wc>

This section bridges the FCS LI\&F formalism (FCS Definition 1) to
the Wilson--Cowan rate equation that governs the population
framework. The bridge is in two parts: a derivation sketch
(#ref(<sec-wc-derivation>)) and a variable translation table
(#ref(<sec-bridge>)).

== Derivation sketch <sec-wc-derivation>

Take $N$ LI\&F neurons (in the FCS sense, Definition 1) all sharing
the same input statistics. Crucially, the neurons differ in *one*
respect: each has its own firing threshold $theta_i$, drawn from a
common distribution $p(theta)$. The shared FCS threshold
$tau_("LIF") = 105$ becomes a population mean $chevron.l theta chevron.r$, and the
spread of $p$ is what gives the population its smooth response.

*Step 1: counting.* At time $t$, with shared input $x(t)$, the
fraction of neurons firing is the fraction whose threshold has been
crossed:
$ rho(t) = (1 / N) abs({i : theta_i <= x(t)}) -> integral_(-infinity)^(x(t)) p(theta) dif theta = F(x(t)), $ <eq-counting>
where $F$ is the cumulative threshold distribution.

*Step 2: smoothness emerges.* $F$ is smooth (assuming $p$ is). This
is the conceptual move on which everything else rests: the
spike-reset non-smoothness of *individual* neurons is averaged away
by the population. The population responds smoothly to its input even
though each constituent neuron does not. We are not modelling-around
the LI\&F reset; we are *observing the same dynamics at a coarser
scale*, and the reset becomes invisible.

*Step 3: leak.* The continuous-time analogue of the FCS leak vector
$(10, 5, 3, 2, 1)$ over $sigma = 5$ ticks is an exponential leak
$dot(V) = -V slash tau$ with effective time constant $tau approx 5$
ticks. Combining with the input drive and applying the same population
average gives the relaxation equation $tau dot(rho) = -rho + F(x)$.

*Step 4: weights.* For multiple coupled populations indexed by $i$,
with synaptic weights $W_(i j)$ giving the influence of population
$j$ on population $i$, the input to population $i$ is
$sum_j W_(i j) rho_j + I_i$. Writing $f equiv F$ for the population
gain:
$ tau dot(rho)_i = -rho_i + f(sum_j W_(i j) rho_j + I_i). $ <eq-wc>
This is the Wilson--Cowan equation @WilsonCowan1972.

*Step 5: choice of gain.* Throughout this note we use the logistic
sigmoid
$ f(x) = 1 / (1 + exp(-k (x - theta_*))), quad k = 4, quad theta_* = 1, $ <eq-sigmoid>
with $tau = 1$. The slope is
$f'(x) = k f(x) (1 - f(x))$, peaking at $k slash 4 = 1$ at the inflection
$x = theta_*$. These are textbook values @DayanAbbott2001; no parameter
in this note has been tuned. The constant external drive used
throughout is $I = 1.5$.

*The takeaway.* The smooth sigmoid is not a modelling choice imposed on
LI\&F dynamics. It is what LI\&F dynamics *look like at population
scale*, derived rigorously from a distributed-threshold ensemble.

== Variable bridge <sec-bridge>

A reader trained on FCS Definition 1's discrete 4-tuple
$(tau, r, p, y)$ encounters new symbols below. The following table
translates each.

#figure(
  table(
    columns: (auto, 1.5fr, 1.4fr, 2fr),
    align: (left, left, left, left),
    table.header(
      [*Symbol*], [*In the population framework*], [*FCS analogue*], [*Intuition*],
    ),
    [$rho_i$], [Firing rate of population $i$], [Boolean spike $y_i$ in FCS Def. 1], [Fraction of neurons in population $i$ firing per unit time. The FCS Boolean spike becomes a continuous probability under population averaging.],
    [$tau$], [Membrane time constant in @eq-wc], [Encoded jointly by $r$ and the $sigma$-tick window], [How long a perturbation persists before decaying. The FCS leak vector $(10,5,3,2,1)$ over $sigma = 5$ is the discrete analogue of an exponential leak with $tau approx 5$ ticks.],
    [$f(x)$], [Sigmoidal gain function in @eq-sigmoid], [Cumulative of the threshold distribution, *averaged over the population*], [Each FCS neuron has the same $tau_("LIF") = 105$. With distributed thresholds, the fraction firing at input $x$ is the fraction with $theta <= x$ --- a smooth CDF.],
    [$k$], [Sigmoid slope], [Inverse spread of $p(theta)$], [Sharpness. Large $k$ recovers the FCS Dirac spike (all neurons share one threshold); small $k$ describes broader populations.],
    [$theta_*$], [Sigmoid midpoint], [The population mean of $tau_("LIF")$], [Where 50% of the population fires.],
    [$I$], [Constant external drive], [$b$ times the input train, where $b = 11$ in the FCS setup], [The continuous note abstracts the binary input into a constant drive $I = 1.5$.],
    [$W_(i j)$], [Influence of population $j$ on population $i$], [Synaptic weight $w_(j i)$ from FCS Def. 1], [Population-averaged per-synapse impact. The cross-validation in #ref(<sec-xvalid>) uses the heuristic scaling $w^("WC") = abs(w^("LIF")) slash 8$.],
    [$g equiv f'(x)$], [Sigmoid slope at operating point $x$], [No direct FCS analogue], [Linear-response gain. Far from threshold the population is saturated and $g$ is small; near threshold it is responsive and $g$ is large.],
    [$J$], [Jacobian of the rate equation], [No direct FCS analogue], [The matrix that says how small perturbations evolve around a fixed point. Stability and bifurcations are read off its eigenvalues.],
    [$lambda_i$], [Eigenvalues of $J$], [No direct FCS analogue], [Modes of perturbation. $"Re"(lambda) > 0$ means the mode grows; $"Re"(lambda) = 0$ marks the bifurcation.],
  ),
  caption: [*Variable bridge.* Each population-framework symbol is grounded in
  its FCS analogue (where one exists) and an intuition for the FCS-native
  reader.],
) <tbl-bridge>

== Linearisation and spectrum

Linearising @eq-wc around a fixed point $rho^*$ gives the Jacobian
$ J = tau^(-1) [-II_n + "diag"(f'(W rho^* + I)) W]. $ <eq-jacobian>
Sort the eigenvalues by descending real part: ${lambda_i}$. The fixed
point is stable when $"Re"(lambda_1) < 0$; it loses stability when
either a real eigenvalue crosses zero (pitchfork or saddle-node) or
a complex-conjugate pair crosses the imaginary axis (Hopf). The
*spectral gap* $Delta = "Re"(lambda_1) - "Re"(lambda_2)$ measures how
decisively the dominant mode dominates. Standard
@Strogatz2014 @Kuznetsov2004; the next section makes the abstract
construction concrete on the contralateral archetype.

= The contralateral archetype: topology in the equations <sec-contra>

The brief running through this section is a question Elisabetta De
Maria asked after reading the v1 of this note: "this is a two-neuron
WTA, shouldn't the behaviour be visible directly in the ODE's
equations?" The answer is yes, and this section makes the topology of
mutual inhibition --- the FCS Fig. 1f graph --- legible step by step
in the equations of @eq-wc.

== From topology to ODE

FCS Fig. 1f shows two neurons each inhibiting the other, with
external excitatory drive on both. At population scale the same
topology applies, with $rho_1$ and $rho_2$ the firing rates of the
two populations, mutual inhibition with weights $w_(12), w_(21) > 0$
(both negated when entering the input to $f$), and symmetric drive
$I$. Writing @eq-wc out longhand --- not as a matrix --- gives:
$ tau dot(rho)_1 &= -rho_1 + f(I - w_(12) rho_2), \
  tau dot(rho)_2 &= -rho_2 + f(I - w_(21) rho_1). $ <eq-contra>
The structure to notice: each population's input is its drive $I$
*minus* its inhibition from the other population. There is no
self-coupling (no $rho_1$ inside the first $f$, no $rho_2$ inside the
second). The mutual-inhibition graph of FCS Fig. 1f is exactly
visible in the absence of self-terms and in the minus signs. Every
algebraic operation that follows in this section is a manipulation of
@eq-contra; the topology is preserved.

== Symmetric fixed point

Set $w_(12) = w_(21) = w$ (symmetric case first; the asymmetric case
follows in #ref(<sec-bif>)). The symmetry $rho_1 = rho_2 = rho^*$ is
preserved by the dynamics, so a symmetric fixed point satisfies
$ rho^* = f(I - w rho^*). $ <eq-sym-fp>
This is a scalar implicit equation. For our parameters $(k, theta_*, I)
= (4, 1, 1.5)$ it has either one root (the symmetric activity is the
unique attractor) or three roots (the symmetric activity is the
unstable saddle, two new asymmetric branches are stable). The
transition between the two regimes is the pitchfork.

#figure(
  image("figs/rho_star_curve.pdf", width: 75%),
  caption: [Fixed-point branches as a function of the diagonal weight $w =
  w_(12) = w_(21)$. Below $w_* = 1$ the symmetric branch (grey) is the
  unique attractor; above, two asymmetric branches (teal) emerge ---
  one with $rho_1 > rho_2$ (population 1 wins), one with the reverse.
  The dashed red line at $w_* = 1$ is the closed-form pitchfork apex
  (#ref(<sec-pitchfork-corner>)).],
) <fig-rho>

== Linearisation and the pitchfork in two lines

Set $rho_i = rho^* + delta_i$ in @eq-contra and Taylor-expand to first
order. With $g equiv f'(I - w rho^*)$ the slope at the operating
point:
$ tau dot(delta)_1 &= -delta_1 - w g space delta_2, \
  tau dot(delta)_2 &= -delta_2 - w g space delta_1. $ <eq-lin>
The diagonal $-1$'s come from leak; the off-diagonal $-w g$'s come
from the cross-inhibition multiplied by the population's
responsiveness $g$. Mutual inhibition in the topology produces mutual
negative off-diagonal coupling in the linearisation. *The sign
structure of the original graph is the sign structure of $J$.*

The natural decomposition of @eq-lin uses sum and difference modes:
$s = delta_1 + delta_2$ and $d = delta_1 - delta_2$. The swap symmetry
of @eq-contra makes them decouple cleanly:
$ tau dot(s) = -(1 + w g) space s, quad
  tau dot(d) = -(1 - w g) space d. $ <eq-modes>
The sum mode $s$ ("both populations together") always decays. The
difference mode $d$ ("one above, one below") decays when $w g < 1$
and *grows* when $w g > 1$. The pitchfork is at $w g = 1$:

- $w g < 1$: deviations decay. The symmetric state is stable. No
  winner.
- $w g > 1$: deviations amplify exponentially at rate $(w g - 1)
  slash tau$. The symmetric state is the unstable saddle. *Which*
  population wins is determined by the sign of the perturbation
  $d_0$.

Two lines of algebra and the WTA mechanism is in plain view: mutual
inhibition with sufficient gain destabilises the symmetric mode and
amplifies any asymmetry.

== Pencil on its tip

A natural follow-up: with strictly identical parameters and zero
noise, doesn't the symmetric fixed point persist forever? Yes ---
*unstably*, like a pencil balanced on its tip. The symmetric activity
$rho^*$ is a fixed point of @eq-contra at any $w$, and a noise-free
trajectory starting exactly on it stays on it. But the symmetric mode
is unstable past $w_* = 1$, so any deviation whatsoever --- numerical
rounding, biological noise, asymmetry in initial conditions --- is
amplified at exponential rate $(w g - 1) slash tau$ until the system
commits to one of the two asymmetric attractors. The choice of winner
is determined by the *sign* of the initial perturbation, not by the
parameters. Biologically this is the right behaviour: a contralateral
inhibition circuit *should* produce decisive winners, and it does so
by being a feedback amplifier for any breach of symmetry.

== Loop-gain interpretation: the Barkhausen criterion

The pitchfork condition $w g = 1$ has a physical reading that should
be familiar to anyone who has done classical feedback analysis. Trace
a small asymmetry around the inhibitory loop:

#block(inset: (left: 14pt, top: 4pt, bottom: 4pt))[
  Suppose population 1 fires slightly more than population 2: $delta_1
  > delta_2$. Population 1's excess firing inhibits population 2 more
  strongly, reducing population 2's firing by approximately $g_2
  w_(12) delta_1$. Population 2's *reduced* firing inhibits population
  1 *less*, increasing population 1's firing by approximately $g_1
  w_(21) (g_2 w_(12) delta_1)$. The loop has closed with net gain
]
$ "loop gain" = w_(12) w_(21) g_1 g_2 quad text("per round trip.") $

If the loop gain exceeds 1, asymmetries amplify on each round trip
and the symmetric state is unstable. If below 1, asymmetries decay.
*This is exactly the Barkhausen criterion* used in classical
electronics to decide whether a feedback amplifier oscillates: the
loop gain must exceed unity. In the symmetric case $w_(12) = w_(21) =
w$ and $g_1 = g_2 = g$, so the condition reduces to $w^2 g^2 = 1$,
i.e. $w g = 1$ as in @eq-modes.

The conceptual move: the *product* of the weights matters because
perturbations *multiply* around feedback loops. This is why the WTA
boundary is a hyperbola (a level set of a product), not a square or a
straight line. The hyperbola is the loop-gain unity contour.

== The asymmetric case

For $w_(12) eq.not w_(21)$ the sum/difference decomposition no longer
diagonalises $J$ exactly. But the loop gain still appears in the
determinant. Computing $det J$ from @eq-jacobian for the contralateral
$W$ gives
$ det J = (1 - w_(12) w_(21) g_1 g_2) slash tau^2. $ <eq-det>
The *saddle-node fold* of the asymmetric system --- where two fixed
points merge and disappear --- is exactly $det J = 0$, i.e. the
loop-gain unity contour
$ w_(12) w_(21) g_1 g_2 = 1, $ <eq-fold>
with $g_i$ now evaluated at the asymmetric saddle. #ref(<sec-bif>)
traces this contour symbolically and shows it agrees with the numerical
saddle-node trace to machine precision. The pitchfork apex
(#ref(<sec-pitchfork-corner>)) is the special case of @eq-fold on the
symmetry diagonal.

== Closed-form pitchfork corner <sec-pitchfork-corner>

On the diagonal $w_(12) = w_(21) = w$ the symmetric fixed point
satisfies @eq-sym-fp and the pitchfork condition $w g = 1$ closes by
the sigmoid identity. The midpoint of the sigmoid is reached when its
argument equals $theta_* = 1$, i.e. $I - w rho^* = 1$ giving
$rho^* = (I - 1) slash w$. At the midpoint $f(theta_*) = 0.5$, so
combining with $rho^* = f(I - w rho^*) = f(theta_*) = 0.5$ gives
$rho^* = 1 slash 2$, hence $w_* = 2 (I - 1) = 1$ for $I = 1.5$.
Plugging back: $g_* = k f^* (1 - f^*) = 4 dot 0.5 dot 0.5 = 1$, so
$w_* g_* = 1$ as required. *The apex of the WTA boundary is at $w_* =
1$ exactly, pinned by the sigmoid half-activation condition with no
free parameters.*

= Verifying the pitchfork prediction against simulation <sec-A>

This section tests Hypothesis A on the contralateral archetype. The
existing v1 of this note showed a $99.96 %$ match between the
spectral predictor and a $50 times 50$ ground-truth WTA sweep with
the asymptotic time horizon $t = 50 tau$. After v1 circulated, the
advisors raised the apparent disagreement with FCS Fig. 10's diagonal
verdict; this section disambiguates the property and shows the same
sweep with the time-bounded classifier matched to FCS \S 6.3.4.

== Two distinct properties

The "winner takes all" headline names a *family* of properties, not a
single one. Three are distinguishable in our setup:

- *FCS Property 7* (\S 6.3.4): the contralateral archetype eventually
  commits to one population, with no time bound. This is the
  qualitative claim.
- *FCS Fig. 10/11 verification*: the property strengthened to a
  4-tick time bound. *"Blue (resp. red) points represent pairs of
  weight values for which the stabilization is (resp. is not)
  reached within the first four time units"* (FCS \S 6.3.4).
- *The asymptotic WTA classifier* used in the v1 of this note: a
  Wilson--Cowan trajectory committed to a winner with $abs(rho_1 -
  rho_2) > 0.3$ at $t = 50 tau$.

The three are nested but inequivalent. The first is weakest (any
time horizon will do); the second is strongest (must commit fast);
the third is intermediate (commit eventually but in a continuous
relaxation that has no integer-tick analogue). The v1 of this note
showed the third; FCS Fig. 10 shows the second. This is the source
of the apparent diagonal disagreement.

== Asymptotic and time-bounded panels

We re-ran the same WTA classifier on the same $50 times 50$ grid with
two time horizons: $t = 50 tau$ (the original) and $t = 4 tau$ (the
direct continuous-side analogue of FCS Fig. 10's 4-tick bound).

#figure(
  image("figs/p1_t4_panel.pdf", width: 100%),
  caption: [WTA verdict on the WC contralateral archetype, with FCS
  Fig. 10 colour convention (teal $=$ WTA holds, red $=$ does not).
  *Left: asymptotic.* Trajectories integrated to $t = 50 tau$. *Right:
  time-bounded.* Same classifier with $t = 4 tau$. The two panels share
  the same bistable region; the time-bounded panel is slightly thinner
  at the boundary because the asymmetric mode's exponential growth rate
  $(w g - 1) slash tau$ vanishes at the pitchfork and needs more time
  to clear the commitment threshold there. The black curve is the
  symbolic pitchfork locus from #ref(<sec-bif>).],
) <fig-A>

The bistable region's *geometry* is preserved across the two time
horizons --- the same hyperbolic wedge bounded by the pitchfork curve.
Only the boundary cells, where the linear growth rate is small,
differ. Quantitatively: 1213 of 2500 cells are WTA at $t = 50 tau$;
1184 at $t = 4 tau$ --- a 2.4% reduction. *Within $4 tau$ the
asymmetric mode commits to amplitude $> 0.3$ across most of the
bistable region.* The spectral predictor's $99.96 %$ accuracy on the
asymptotic ground truth therefore holds robustly down to short time
horizons; the v1 result is not an artefact of the long integration.

== Why the two panels disagree on the diagonal

Reading the panels along the diagonal $w_(12) = w_(21) = w$ resolves
the apparent disagreement with FCS Fig. 10. Both panels are *teal*
along the diagonal, in agreement: the population framework says the
contralateral archetype goes WTA at $w > 1$ (the pitchfork apex of
#ref(<sec-pitchfork-corner>)) under either time horizon. FCS Fig. 10's
diagonal is *red* over the same range because its property is
strictly stronger: not "WTA eventually" or "WTA within $4 tau$ in the
continuous sense", but "stabilises within four discrete LI\&F ticks".
The discrete LI\&F system needs many more ticks to accumulate the
spike-count statistics that the FCS classifier reads --- ticks are
not population time units --- so the boundary at which discrete
stabilisation reaches 4 ticks sits much further out, around
$abs(w^("LIF")) approx 30$. *The intrinsic time scales of the two
descriptions are different*; the brief expectation that simply
shrinking the WC time horizon to $t = 4 tau$ would reproduce FCS Fig. 10
is wrong, and the right reading is that the two are measuring
different things.

A useful corollary: the population framework's pitchfork apex at $w_*
= 1$ corresponds to $abs(w^("LIF")) approx 8$ under the heuristic
scaling of #ref(<sec-xvalid>), while FCS Fig. 10's fast-WTA boundary
is at $abs(w^("LIF")) approx 30$. The factor-of-roughly-four gap
between *"the bistable mode exists"* and *"the bistable mode commits
within four discrete ticks"* is itself informative: the pitchfork is
necessary for WTA, but fast WTA additionally requires the asymmetric
mode growth rate to be large enough to clear a fixed amplitude
threshold within the time budget.

== Hypothesis A: verdict

The dominant real eigenvalue of @eq-jacobian classifies WTA cells at
$99.96 %$ accuracy on the $t = 50 tau$ ground truth (single boundary
mismatch); the time-bounded panel agrees with the spectral predictor
on $97.6 %$ of cells, with all disagreements located at the boundary
where the linear-growth rate is small. *Verified.*

= Bifurcation loci as loop conditions <sec-bif>

This section presents the closed-form derivations for the two
bifurcation loci tested in this note --- the contralateral pitchfork
and the activator--inhibitor Hopf --- and frames both as conditions on
the loop gain of the underlying topology.

== Contralateral pitchfork (loop-gain unity)

@eq-fold gave the pitchfork locus as $w_(12) w_(21) g_1 g_2 = 1$.
Eliminating the weights through the fixed-point equations $rho_i = f(I -
w_(3-i) rho_(3-i))$ and the identity
$g_i = k rho_(3-i) (1 - rho_(3-i))$ (writing the slope in terms of
the firing rate of the *other* population, since each $g_i$ is
evaluated at the input to neuron $i$) reduces the locus to a single
transcendental constraint between the fixed-point coordinates,
$ (I - f^(-1)(rho_1))(I - f^(-1)(rho_2)) dot k^2 (1 - rho_1)(1 - rho_2) = 1. $ <eq-pitchfork-symbolic>
Tracing this constraint by continuation in $rho_1 in (0, 1)$ and
recovering the corresponding weights via $w_(12) = (I - f^(-1)(rho_2))
slash rho_1$ and $w_(21) = (I - f^(-1)(rho_1)) slash rho_2$ gives the
red curve in @fig-pitchfork. Numerically the residual
$abs(1 - w_(12) w_(21) g_1 g_2)$ at every continuation point is at
$10^(-12)$, and a numerical bisection trace agrees to machine precision.

#figure(
  image("figs/p2_pitchfork.pdf", width: 60%),
  caption: [Pitchfork locus from the symbolic loop-gain unity condition
  $w_(12) w_(21) g_1 g_2 = 1$ (red, traced by continuation) versus a
  numerical radial-bisection trace (black dots). The two agree to
  within the root-finder's precision floor at the saddle-node fold.],
) <fig-pitchfork>

The pitchfork is *the* level set of the loop-gain product. There is no
other algebraic surface to look at; the bifurcation is the topology's
loop gain crossing unity.

== Activator--inhibitor Hopf (negative loop)

The activator--inhibitor archetype is the negative-feedback loop of FCS
Fig. 1d. Self-excitation $w_("aa") >= 0$ on the activator gives it
some inertia (a bare $w_("aa") = 0$ gives $tr J = -2 slash tau$ and
admits no Hopf @WilsonCowan1972 @ErmentroutTerman2010). With
self-excitation included, the Jacobian becomes
$ J = mat(-1 + w_("aa") g_A, -w_("ia") g_A; w_("ai") g_I, -1), $
so $tr J = w_("aa") g_A - 2$ and $det J = 1 - w_("aa") g_A + w_("ai")
w_("ia") g_A g_I$. The Hopf locus needs $tr J = 0$ and $det J > 0$:
- *Trace condition.* $g_A = 2 slash w_("aa")$ pins the activator's
  operating-point slope.
- *Determinant condition.* $det J = w_("ai") w_("ia") g_A g_I - 1$,
  so the Hopf locus is *the loop-gain unity condition for the negative
  loop*: the product around the cycle (activator drives inhibitor with
  gain $w_("ai") g_I$, inhibitor inhibits activator with gain $w_("ia")
  g_A$, i.e. loop gain $w_("ai") w_("ia") g_A g_I$) equals 1.

Both bifurcations are loop-gain unity conditions, the pitchfork on
the positive-sign-product loop of mutual inhibition and the Hopf on
the negative-sign-product loop of activator--inhibitor. The
*sign-product* of the loop selects the bifurcation family; see
#ref(<sec-thomas>).

The oscillation frequency at the Hopf locus is
$ omega^* = sqrt(det J) = sqrt(w_("ai") w_("ia") g_A g_I - 1). $ <eq-hopf-freq>
Fixing $w_("xa") = 1$ and $w_("aa") = 2.5$ (the smallest
self-excitation placing the full Hopf locus inside $(w_("ai"),
w_("ia")) in [0, 5]^2$), a numerical $50 times 50$ sweep classifies
each cell as oscillating if the activator trace crosses its mean at
least three times in the last $50$ time units with amplitude $> 0.05$.

#figure(
  grid(
    columns: 2, column-gutter: 6pt,
    image("figs/p2_hopf.pdf"),
    image("figs/p2_freq.pdf"),
  ),
  caption: [*Activator--inhibitor Hopf.* Left: empirical oscillation
  region (grey) with the symbolic Hopf curve overlaid (red). Right:
  FFT-measured frequency vs the analytical prediction at the nearest
  Hopf point.],
) <fig-hopf>

Linear stability and the symbolic Hopf locus agree within a median of
one grid cell. The empirical oscillation boundary lags the linear one
in localised patches where a strongly stable second fixed point absorbs
trajectories past the Hopf --- a multi-fixed-point bifurcation
phenomenon, not a derivation error. FFT frequencies in the
well-oscillating regime (amplitude $> 0.1$) agree with @eq-hopf-freq
at median relative error $9.6 %$.

== Sign products around loops <sec-thomas>

Writing the bifurcation conditions as loop-gain unity exposes a
unifying classification. Define the *sign product* of a feedback loop
as the product of the signs of the synaptic weights traversed around
the cycle.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, left),
    table.header(
      [*Archetype*], [*Off-diag signs of $W$*], [*Sign product*], [*Bifurcation family*],
    ),
    [Contralateral inhibition (FCS Fig. 1f)], [$(-, -)$], [$+$], [Pitchfork (saddle-node) at $det J = 0$],
    [Activator--inhibitor (FCS Fig. 1d)], [$(+, -)$], [$-$], [Hopf at $tr J = 0$, $det J > 0$],
    [Positive loop (#ref(<sec-other>))], [$(+, +)$], [$+$], [Saddle-node fold at $det J = 0$],
  ),
  caption: [*Sign-product classification of bifurcation families.* The
  sign product of a feedback loop selects the family; the magnitude of
  the loop gain locates the bifurcation in parameter space.],
) <tbl-thomas>

This is the spiking-network analogue of *Thomas's rules* for gene
regulatory networks @Kuznetsov2004: positive loops (sign product $+$)
support multistability through saddle-nodes and pitchforks; negative
loops (sign product $-$) support oscillation through Hopf
bifurcations. The classification carries from gene networks to
spiking-network archetypes because the underlying mathematical fact
--- the sign of a loop's product determines the family of
bifurcations admissible at unity loop gain --- is topology-driven and
does not depend on the specific functional form of $f$, only on its
monotonicity.

The advisors will recognise Thomas's framework from the FCS \S 2
related-work survey, where Thomas-style discrete models are listed as
one of the qualitative-modelling formalisms for biological systems.
The population framework recovers the Thomas-rule classification at
LI\&F population scale, with concrete loop-gain expressions in place
of qualitative sign annotations.

= Inverse design via pole placement <sec-C>

Hypothesis C is classical control theory applied to the
activator--inhibitor archetype: given a target frequency $omega^*$,
solve for $(w_("ai"), w_("ia"))$ such that the Jacobian's
complex-conjugate pair lies at $plus.minus i omega^*$. The Hopf trace
condition pins $g_A$ (and so $rho_A$); the fixed-point condition pins
$w_("ia") rho_I$; the inhibitor argument fixes $w_("ai") = f^(-1)(rho_I)
slash rho_A$; and the frequency condition $det J = (omega^*)^2$
reduces to a single scalar equation in $rho_I$, solved by `brentq`.

At $w_("aa") = 2.5$ the achievable frequency range inside
$(w_("ai"), w_("ia")) in (0, 5]^2$ is $[0.04, 2.23]$; the lower-$rho_A$
branch covers $[1.29, 2.23]$ and the upper-$rho_A$ branch covers
$[0.04, 1.68]$. Out of the plan target set $\{0.1, 0.3, 0.5, 0.7,
1.0, 1.5, 2.0, 3.0\}$, $omega^* = 3$ is outside the feasible range and
is replaced by $omega^* = 2.15$ near the lower-branch feasibility edge.
For every remaining target the design residual
$abs("Im"(lambda) - omega^*)$ is at $10^(-13)$: *the linear pole
placement succeeds to numerical precision.*

#figure(
  grid(
    columns: 2, column-gutter: 6pt,
    image("figs/p3_scatter.pdf", width: 95%),
    image("figs/p3_traces.pdf", width: 95%),
  ),
  caption: [*Hypothesis C.* Left: target $omega^*$ against the
  FFT-measured frequency of the simulated limit cycle. Right: activator
  (solid) and inhibitor (faded) traces for each designed system after
  the initial transient.],
) <fig-polep>

*Simulation caveat.* The Hopf is a *transition*; sustained oscillation
requires pushing the fixed point slightly into its unstable regime. With
a branch-direction-correct $0.5%$ (upper) or $2%$ (lower) crossing,
six of the eight designed systems oscillate at their target frequency
within $10%$. The exceptions are the two lowest targets on the upper
branch ($omega^* in {0.1, 0.3}$), which fall in a *codim-2*
neighbourhood of the Hopf locus (a Bogdanov--Takens or generalised-Hopf
neighbourhood @Kuznetsov2004) where the Hopf line approaches a
saddle-node fold and the bifurcation is subcritical, so no stable
limit cycle of finite amplitude exists. The designed weights produce
the correct *linear* poles exactly; the absence of a sustained limit
cycle there is a property of the specific parameter choice $w_("aa") =
2.5$, not a failure of the inverse-design framework.

The generic-Hopf cases work cleanly (e.g. $omega^* = 1.5$: target
$1.500$, measured $1.503$, $0.2%$ error; $omega^* = 2$: target $2.000$,
measured $1.925$, $3.8%$ error). The codim-2 obstruction delimits the
regime in which classical pole-placement transfers cleanly to the
nonlinear simulation: generic supercritical Hopfs yes, near-degenerate
Hopfs no.

= Cross-validation against discrete LI\&F <sec-xvalid>

Two questions about the population framework cannot be answered by
the framework alone. (i) Does the discrete LI\&F simulator on the
*same* contralateral topology agree with the continuous prediction
where the descriptions overlap? (ii) Does the rectangular boundary
that the existing FCS-style sweep (and our v1 Phase 4 result) shows
out to $abs(w^("LIF")) <= 40$ continue smoothly at higher weights, or
do new structures emerge --- the scaling motivation of
#ref(<sec-intro>)? This section answers both.

== Setup

A black-box LI\&F simulator (FCS Lustre semantics: threshold
$tau_("LIF") = 105$, leak vector $(10, 5, 3, 2, 1)$, external drive
weight $b = 11$) is swept over the integer grid
$(w_(12)^("LIF"), w_(21)^("LIF")) in [-200, -1]^2$ at unit step. Each
cell is integrated for $50$ ticks under a two-run mirror-image
classifier mirroring #ref(<sec-A>): neuron $0$'s drive is gated off for
the first two ticks in one run and neuron $1$'s in the other; a cell
is bistable iff each run produces a clean spike-count winner ($>= 8$:1
ratio in the last $20$ ticks) and the winners differ. LI\&F weights
map into continuous units by the heuristic linear scaling $w^("WC") =
abs(w^("LIF")) slash 8$.

== Two kinds of bistability at the symmetric corner

The LI\&F bistable region (within $abs(w) <= 40$) is a pair of
axis-aligned strips $abs(w_(12)^("LIF")) >= w_c$ OR
$abs(w_(21)^("LIF")) >= w_c$ with $w_c approx 7$. The WC pitchfork
region is the concave hyperbolic wedge $w_(12) w_(21) g_1 g_2 > 1$.
The two regions agree at the symmetric corner $w_(12) approx w_(21)$
but diverge in the arms. A trace of the discrete dynamics at, say,
$(w_(12)^("LIF"), w_(21)^("LIF")) = (-30, -10)$ reveals the mechanism
for the disagreement: once either neuron fires a single spike, its
per-tick inhibitory contribution $abs(w_(i j)^("LIF"))$ exceeds the
other neuron's drive $b = 11$, and under the reset-after-spike
semantics whichever neuron fires first locks in. *This is a
spike-timing bistability specific to the discrete dynamics*; the
continuous mean-field reduction has no analogue because its gain is
smooth and lacks the all-or-none reset. The WC pitchfork locus is
therefore a *lower bound* on the LI\&F bistable region, not its
envelope.

The continuous description captures pitchfork-driven bistability ---
the symmetric mode losing stability through the loop-gain unity
condition --- and misses the spike-timing-driven mechanism. Both
descriptions are valid at their own scale; neither is the "truth" of
the other. For analytical parameter-space charting at the symmetric
corner the population framework gives the answer; for the
spike-timing locks in the asymmetric arms only the discrete simulator
will do.

== Scaling: does the rectangular boundary persist?

To answer the scaling question raised in #ref(<sec-intro>), we
re-ran the LI\&F sweep on a $200 times 200$ integer grid covering
$abs(w^("LIF")) in [1, 200]$. The result:

#figure(
  image("figs/p4_lif_extreme.pdf", width: 65%),
  caption: [LI\&F bistable region (teal) on $[1, 200]^2$ in
  inhibitor-magnitude units, with the original Phase 4 sweep range
  $abs(w^("LIF")) <= 40$ outlined dashed. The rectangular-strip
  boundary at $abs(w^("LIF")) approx 7$ persists out to $abs(w) = 200$
  with no new features.],
) <fig-extreme>

Numerically: the boundary in $abs(w_(12)^("LIF"))$ has median, 10th
and 90th percentile all at exactly $7$ across the $200$ columns of
the sweep --- the boundary is *flat* and constant. No new structure
emerges between $abs(w) = 40$ and $abs(w) = 200$. The
spike-timing-lock mechanism that drives the rectangular geometry
saturates well below $abs(w) = 40$ and stays saturated thereafter.

This is a clean answer to the scaling question: *the FCS-style
boundary geometry of the contralateral archetype does not develop
new features at higher weights*. The population framework's
prediction --- a smooth pitchfork hyperbola at the symmetric corner
--- is independently the answer in continuous parameter space, where
the discrete simulator's rectangular envelope does not apply.

The full FCS Fig. 10 comparison in continuous units, set against the
WC pitchfork locus, is in @fig-overlay below.

#figure(
  image("figs/p4_overlay.pdf", width: 60%),
  caption: [LI\&F bistable region (grey, mapped into WC units by
  $w^("WC") = abs(w^("LIF")) slash 8$) alongside the symbolic
  pitchfork curve (red). Agreement at the symmetric corner; rectangular
  envelope from spike-timing locks in the arms.],
) <fig-overlay>

A direct comparison against FCS Fig. 10 itself would require running
the FCS Lustre code on the same grid. The Lustre source is held in a
deprecated I3S Redmine instance and we do not have access; the
comparison here is to our own black-box LI\&F simulator implementing
the FCS \S 6.2 semantics verbatim.

= Other archetypes <sec-other>

Three further archetypes were exercised to test that the framework
generalises beyond the contralateral and activator--inhibitor cases.
In each case the analytical prediction matches numerical simulation
to better than $5%$ relative error, parameter-free given the chosen
sigmoid and $tau$.

*Series chain.* An $n$-population feed-forward chain with equal
inter-stage weight $w$ has a recursive steady state $rho_0 = f(I)$,
$rho_k = f(w rho_(k-1))$. With no recurrence the Jacobian is
triangular and all eigenvalues equal $-1 slash tau$; the steady state
is the population-level analogue of a feed-forward gain cascade. Over
$w in [0.2, 4.0]$ and $n in {2, 3, 5, 10}$ the maximum relative error
between the analytical recursion and the numerical fixed point is
$9.45 times 10^(-11)$ (machine precision).

#figure(
  image("figs/p5_series.pdf", width: 75%),
  caption: [Series chain final-stage activity vs the chain weight, for
  $n in {2, 3, 5, 10}$. Solid: recursive-sigmoid prediction; dots:
  numerical steady states.],
) <fig-series>

*Parallel composition.* A driver feeding $n$ independently-weighted
downstream populations produces a block-triangular Jacobian: the
driver is a scalar $-1 slash tau$ self-block, the downstream block is
diagonal, and the only non-zero off-diagonal entries are the driver-to-downstream gains. Every eigenvalue equals $-1 slash tau$. Verified
to machine precision on $n in {2, 4, 8}$.

#figure(
  image("figs/p5_parallel.pdf", width: 55%),
  caption: [Jacobian of a parallel composition at $n = 8$.],
) <fig-parallel>

*Positive loop.* Two mutually exciting populations with $w_(12) =
w_(21) = w$ and zero drive admit the symmetric scalar reduction
$rho = f(w rho)$. The saddle-node fold (tangency of $y = rho$ to
$y = f(w rho)$) is the simultaneous system $rho = f(w rho)$, $w k rho
(1 - rho) = 1$; numerical FP-count locates two transitions at $w
approx 1.676$ and $w approx 5.273$, matching the analytical fold
weights $1.682$ and $5.278$ to $0.37 %$ and $0.09 %$.

#figure(
  image("figs/p5_positive.pdf", width: 70%),
  caption: [Positive-loop fixed-point branches vs the loop weight at
  zero drive. Below the first saddle-node the only fixed point is the
  low-activity branch; above it a high-activity branch and a middle
  saddle appear.],
) <fig-positive>

The *contrast between the contralateral and positive-loop archetypes*
is structurally illuminating. Both have positive sign-product around
the feedback cycle (#ref(<tbl-thomas>)) and so admit the same
bifurcation family --- saddle-node or its symmetric cousin the
pitchfork. But the *qualitative* outcomes differ: mutual inhibition
breaks the symmetric branch into two asymmetric attractors (winner
selection); mutual excitation folds a low-activity branch into a
high-activity branch (active/inactive switching). The same loop-gain
mathematics produces both behaviours, with the sign of $W$'s
off-diagonal entries setting which.

= Discussion <sec-disc>

== Reach

The population framework cleanly addresses qualitative dynamical
questions about LI\&F archetypes: stability of fixed points, bifurcation
locations in parameter space, oscillation onset and frequency, inverse
design of weights for a target linear spectrum. For the five archetypes
exercised here, predictions match simulation to analytical tolerance
(pitchfork, series, parallel, positive loop) or within documented
bifurcation-theory caveats (Hopf on the activator--inhibitor loop;
pole placement near codim-2 Hopf neighbourhoods).

== Limits

Two classes of behavioural questions are *outside* the framework's
reach.

First, *bit-exact spike-timing properties*. The population reduction
integrates over the threshold distribution and so cannot predict
integer-period oscillations, exact phase relationships among neurons,
or deterministic spike sequences. The micro--macro bridge of
#ref(<sec-xvalid>) makes this concrete: in the contralateral
archetype the discrete LI\&F exhibits a spike-timing bistability that
the continuous reduction cannot see. A class of FCS properties --- the
delayer-effect classification of #ref(<sec-A>), the period-extension
property of FCS \S 6.3.1, the integer-period oscillation of FCS
\S 6.3.2 --- belongs entirely on the discrete side; the population
framework can predict the *parameter regime* where these properties
become available but not their integer-tick details.

Second, *near-degenerate bifurcations*. Classical pole placement
transfers cleanly to generic supercritical Hopf bifurcations. Near
codim-2 points (Bogdanov--Takens, cusp, Bautin/generalised Hopf
@Kuznetsov2004) the normal form acquires higher-order terms that
dominate the limit-cycle dynamics and the linear prediction ceases to
govern the sustained simulation. The pole placement results of
#ref(<sec-C>) locate this failure mode explicitly.

== Complementarity to FCS

The De Maria et al. programme @DeMaria2020 verifies temporal-logic
properties at specific parameter points by exhaustive model checking
or theorem proving. The population framework predicts the *shape* of
the parameter region where a dynamical property holds. Composed in
principle, the two could reduce the workload of exhaustive
model-checking sweeps: a spectral pre-filter certifies parameter
regions where no pitchfork or Hopf is possible, so the model checker
need not enumerate cells in those regions. We treat this as a
methodological observation, not an engineering deliverable; the
present note develops the spectral side only.

The reverse direction is also worth highlighting. The discrete LI\&F
behaviour the population framework cannot see ---
spike-timing-driven bistability, integer-tick oscillation periods,
filter cascades --- is exactly what the FCS Lustre/Kind2/Coq
machinery is designed to verify. The two scales of description
divide the behavioural questions cleanly: continuous parameter-space
boundaries on the population side, integer-precise temporal-logic
properties on the discrete side. Each tool is best at what the other
cannot do.

== Open questions

Whether hybrid-system spectral theory (saltation matrices, Floquet
theory of periodic-reset systems @KhalilNonlinear) can extend the
population-framework reach to bit-exact periodic spiking is a natural
follow-up but outside the scope of this note. So is a worked
implementation of the spectral pre-filter compositional protocol, which
would require choosing a target FCS property family and the Lustre
encoding to compose against.

= Conclusion <sec-conc>

This note has taken the LI\&F neuronal archetypes of De Maria et al.
@DeMaria2020 up one level of description --- single neurons replaced
by populations with distributed thresholds. The non-smoothness of the
single-neuron spike-reset rule averages into a smooth sigmoidal gain
and the Wilson--Cowan equation @WilsonCowan1972 governs the
population-level dynamics. The qualitative behavioural structure of
the archetypes --- winner-take-all selection, oscillation onset,
bistability, stability envelopes, bifurcation loci --- becomes
accessible to the classical analytical toolkit.

The contralateral archetype is the headline example. Mutual
inhibition is a feedback loop with gain $w_(12) w_(21) g_1 g_2$, and
the winner-take-all transition is the pitchfork at unity loop gain
--- two lines of algebra after the population reduction. The Hopf in
the activator--inhibitor loop is the same condition on the
negative-sign-product loop. Sign products around feedback loops
classify the bifurcation family (Thomas's rules at LI\&F population
scale); loop-gain magnitudes locate the bifurcation in parameter
space. The three hypotheses tested numerically all hold (#ref(<sec-A>),
#ref(<sec-bif>), #ref(<sec-C>)), and the framework generalises to
series, parallel, and positive-loop archetypes (#ref(<sec-other>)).

The population framework is a *companion* to the FCS programme, not a
replacement. FCS verifies properties at specific parameter points;
the population framework predicts the shape of the parameter region.
The cross-validation in #ref(<sec-xvalid>) makes the boundary
explicit: at the symmetric corner the two descriptions agree; in the
asymmetric arms the discrete simulator's spike-timing locks dominate
in a way the continuous reduction cannot see. Each scale answers what
the other cannot.

= Acknowledgements

The archetype taxonomy studied here is due to De Maria et al. and the
author thanks Elisabetta De Maria and Christopher Leturc for
methodological orientation on the discrete LI\&F formulation, and for
the specific feedback on the v1 of this note that motivated the v2
restructuring.

#bibliography("refs.bib", style: "ieee")
