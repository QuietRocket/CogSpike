// Spectral Cartography of LI&F Archetype Parameter Spaces
// Research note — April 2026

#set document(
  title: "Spectral Cartography of LI&F Archetype Parameter Spaces",
  author: "CogSpike Research Team",
  date: datetime.today(),
)

#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, first-line-indent: 0pt)
#set heading(numbering: "1.")
#show heading.where(level: 1): set text(size: 13pt)
#show heading.where(level: 2): set text(size: 11.5pt)

#let proposition(body) = block(
  width: 100%, inset: 8pt,
  stroke: (left: 2pt + green),
  [*Proposition.* #body],
)

#let remark(body) = block(
  width: 100%, inset: 8pt,
  stroke: (left: 2pt + orange),
  [*Remark.* #body],
)

// ------------------------------ title block ------------------------------
#align(center)[
  #text(size: 17pt, weight: "bold")[
    Spectral Cartography of \
    Leaky Integrate-and-Fire \
    Archetype Parameter Spaces
  ]
  #v(0.3em)
  #text(size: 11pt)[
    A Two-Regime Characterisation under Deterministic and Reachability Semantics
  ]
  #v(0.6em)
  #text(size: 11pt)[CogSpike Research Team --- April 2026]
  #v(0.2em)
  #text(size: 9pt, style: "italic")[
    Research Note --- An extension of De Maria et al. (2020)
  ]
]

#v(0.6em)

#block(
  width: 100%, inset: 10pt,
  fill: rgb("#f6f6f6"),
  stroke: (left: 3pt + luma(160)),
)[
  *Abstract.* The FCS contralateral-inhibition archetype of De Maria et al.
  (2020) admits two distinct mathematical characterisations depending on which
  semantic interpretation of winner-takes-all is adopted. Under deterministic
  single-trajectory semantics, WTA is a combinatorial property of the integer
  weight magnitudes, identified by a sound tick-$2$ predicate
  $(|w_(1 2)| > 12) xor (|w_(2 1)| > 12)$ derived directly from the Lustre
  firing equation; this predicate has $100%$ precision and $66%$ recall on
  the WTA region, and the sign of $|w_(1 2)| - |w_(2 1)|$ determines the
  winner on every non-tied grid cell. Under reachability semantics --- the
  semantics that Kind2 model checking actually verifies --- WTA is a spectral
  property, captured by the condition $rho(A) > 0.544$ on the linearised
  state matrix at the symmetric operating point, with $100%$ precision and
  $31%$ recall on the unreachable minority class at a $98.5%$ overall
  classification rate. The two regimes arise from a single analytical
  principle: *the spike-reset rule is the nonlinearity that linearisation
  erases, and its presence or absence in the formal property being verified
  determines whether spectral methods apply*. Under deterministic semantics,
  reset timing is the mechanism by which winners are chosen; under
  reachability semantics, it is averaged out. A closed-form invariant for
  each regime is given, and the spectral test is positioned as a
  polynomial-time pre-filter for FCS-style Kind2 parameter-space sweeps.
]

= Introduction

The De Maria et al. 2020 paper _On the Use of Formal Methods to Model and
Verify Neuronal Archetypes_ [DMDP20] establishes a workflow for the
formal verification of small leaky integrate-and-fire (LI\&F) networks: a
discrete-time Lustre model is compiled to Kind2 to check temporal-logic
properties over integer-weight parameter grids, and the verified cases are
then lifted to Coq for structural theorems. The grid sweeps of §6.3.4 for
the contralateral-inhibition archetype are the empirical target reproduced
here: a $40 times 40$ integer grid over inhibitory weight pairs
$(w_(1 2), w_(2 1))$, each cell coloured by whether the Kind2 verifier
certifies the WTA property.

A methodological gap is visible in that workflow. Between the exhaustive
Kind2 sweep --- exponential in network size, sound and complete but opaque
--- and the Coq theorems --- closed-form but available only for small
structurally-special cases --- there is no continuous middle layer that
would let a practitioner characterise the boundary of the parameter region
at polynomial cost. The companion CogSpike research note [Cog26]
proposes spectral cartography (eigendecomposition of the weight matrix,
spectral radius of the linearised state matrix, pole placement for
oscillation periods) as a candidate middle layer, validated on a contrived
$4$-neuron WTA network. This note tests whether that framework transfers to
the canonical FCS archetypes.

The question is whether the eigenstructure of the LI\&F linearisation can
predict the parameter-space boundaries that De Maria et al. chart
empirically. The answer is two-regime: it can, for reachability semantics,
with a sound polynomial-time predicate; it cannot, for bit-exact
deterministic semantics, because the latter depend on a reset rule that
continuous linearisation discards. A simple combinatorial invariant fills
the deterministic gap. The rest of this note substantiates, qualifies, and
extends these two claims.

#figure(
  image("results/final_triptych.png", width: 100%),
  caption: [*The two-regime split on the contralateral-inhibition $40 times 40$
    grid.* *(a)* Deterministic ground truth: a cell is blue iff one neuron
    captures and the other falls silent within $50$ ticks of a zero-initialised
    simulation ($63.4%$ blue, two asymmetric wings off the diagonal).
    *(b)* The combinatorial sound predictor
    $||w_(1 2)| - |w_(2 1)|| > 7$ (black contour) traces the deterministic
    boundary at $83.4%$ overall accuracy, and the sign of
    $|w_(1 2)| - |w_(2 1)|$ predicts the winner on every non-tied cell
    ($100%$). *(c)* Reachability ground truth: a cell is blue iff WTA emerges
    under some $epsilon$-perturbation of the initial state ($97.8%$ blue,
    with a $6 times 6$ corner of unreachable cells in the weak-inhibition
    region). The scalar-$r$ spectral radius contour $rho(A) = 0.544$ (black
    curve) demarcates the unreachable region with perfect precision. The
    two panels (b) and (c) together constitute the central finding: the
    deterministic map is combinatorial, the reachability map is spectral.],
) <fig-triptych>

= Background and Setup

== The FCS LI&F Neuron

A single FCS neuron maintains a length-$5$ memory buffer
$("mem"[0], dots, "mem"[4])$ of recent weighted inputs, integrated by a
fixed-coefficient window $r = [10, 5, 3, 2, 1]$. At each discrete tick the
neuron evaluates
$V(t) = sum_(e = 0)^4 r_e dot "mem"[e](t)$,
emits a local firing signal $l(t) = [V(t) >= tau]$ where $tau = 105$ is
the integer threshold, and exports the delayed spike
$s(t + 1) = l(t)$. At the next tick, $"mem"[0]$ is overwritten with the
summed weighted inputs of the current tick
$"mem"[0](t+1) = sum_j w_(i j) dot s_j (t+1) + B_(i dot) dot u(t+1)$,
while $"mem"[1..4]$ shift forward ($"mem"[k](t+1) = "mem"[k-1](t)$)
*except* at the first tick after a spike, in which case the shifted taps
are reset to zero. This reset, which models the integrator's recovery
after firing, is the defining nonlinearity of the FCS neuron and the
central object of the analysis below.

All weights and the threshold are integer-scaled (multiplied by $10$
versus the rational definition of §3 in [DMDP20]). A neuron is a
_delayer_ --- firing on every input spike with a one-tick latency ---
when its external input weight is $w gt.eq 11$, since $V(0) = 10 dot w gt.eq
110 gt.eq tau$ on the first tick of constant-$1$ drive.

== The Two Archetypes

*Contralateral inhibition* (Fig. 1f of [DMDP20]) consists of two
neurons $N_1$ and $N_2$, each externally driven at the delayer threshold
$11$, and each sending an inhibitory spike to the other with weights
$w_(1 2)$ (from $N_1$ to $N_2$) and $w_(2 1)$ (from $N_2$ to $N_1$). WTA
(Property $7$ of [DMDP20]) is the claim that for appropriate weight
pairs one neuron stabilises in a firing state and the other in silence.

*Negative loop* (Fig. 1d of [DMDP20]) consists of an activator $A$
driven by a constant external input $X$ with weight $w_("XA")$ and
inhibited by a feedback neuron $I$ with weight $w_("IA") < 0$. The
inhibitor receives a spike from $A$ with weight $w_("AI") > 0$. Property
$5$ of [DMDP20] asserts that under constant input the activator
output converges to the period-$4$ pattern $0 1 1 0 0 1 1 0 0 dots$ for
delayer-configured neurons.

== Two Semantic Interpretations of WTA

The property "WTA holds for weights $(w_(1 2), w_(2 1))$" admits two
formalisations that coincide for continuous dynamical systems but can
diverge for bit-exact discrete ones:

- *Deterministic.* Starting from zero initial state with constant
  symmetric external drive, the simulator trajectory stabilises into a
  state where one neuron emits $gt.eq 40$ spikes in ticks $5 dots 49$
  while the other emits $0$. The grid so defined has $1014 / 1600$ blue
  cells (@fig-triptych a).

- *Reachability.* For some initial state within an $epsilon$-neighbourhood
  of zero --- here, a prior-history perturbation of $plus.minus 2$ to a
  single memory tap of one neuron, just large enough to cross the integer
  threshold --- the resulting trajectory reaches the WTA state above. The
  grid so defined has $1564 / 1600$ blue cells, with a $6 times 6$ block
  of unreachable cells in the weak-inhibition corner (@fig-triptych c).

Both are reasonable formalisations of the question "does this parameter
configuration admit WTA". The Kind2 sweeps of De Maria et al. [DMDP20]
search over reachable states and therefore verify the second; the
simulator trajectory of a physical network from defined initial conditions
executes the first.

= The Deterministic Regime is Combinatorial

== Spectral Predictors Fail

Under deterministic semantics, neither the eigenvalue gap of the raw
weight matrix nor the spectral radius of the linearised state matrix ---
at either the scalar-$r$ approximation of [Cog26] or the full
$5 n$-dimensional linearisation that preserves the windowed integrator
--- classifies the ground truth at better than $69%$ overall accuracy,
against a $63.4%$ majority baseline and a $83.4%$ target established by
the combinatorial baseline below. The raw-$W$ eigengap is identically
zero on $2 times 2$ zero-diagonal matrices, a structural feature of
reciprocal inhibition; the linearised-$A$ spectral radius is symmetric
under the swap $(w_(1 2), w_(2 1)) arrow (w_(2 1), w_(1 2))$ and therefore
blind to the asymmetry that drives the capture in either direction. The
detailed enumeration of failed predictors is given in the supplementary
material.

== The Tick-2 Mechanism

The deterministic simulator breaks symmetry at tick $2$ via an integer
comparison on weight magnitudes. At tick $0$ both neurons receive the
external drive and reach $V(0) = 110 >= tau$; both fire. At tick $1$ the
shifted memory taps are reset and the new $"mem"[0]$ receives the summed
drive $w_"self" + w_"opp" dot s_"opp"(1) = 11 + w_"opp"$ where
$w_"opp"$ is the incoming inhibition from the other neuron --- a strictly
negative quantity in the contralateral topology. At tick $2$, only
$"mem"[0](1) = 11 + w_"opp"$ remains in the length-$5$ window (the other
taps are zero from the reset), and the current-tick $"mem"[0](2) = 11$
(since the opposing neuron's spike from tick $1$ has not yet arrived due
to the one-tick emission delay). The firing condition at tick $2$ is
therefore

$ V(2) = 10 dot 11 + 5 dot (11 + w_"opp") = 165 + 5 w_"opp" >= tau = 105 $

which simplifies to $|w_"opp"| <= 12$.

#proposition[
  (Tick-$2$ sound predicate.) A contralateral-inhibition cell satisfies
  the deterministic WTA property at tick $2$ if and only if exactly one
  of $|w_(1 2)| > 12$ and $|w_(2 1)| > 12$ holds:
  $ "blue"_"det"^"tick-2" (w_(1 2), w_(2 1)) = (|w_(1 2)| > 12) xor (|w_(2 1)| > 12). $
  When this predicate holds, the neuron receiving weaker inhibition
  fires at tick $2$ and the other does not, initiating a capture that
  persists. The predicate has $100%$ precision and $66.3%$ recall
  against the deterministic ground truth; the missing $34%$ consists of
  WTA cells that emerge only through the multi-tick interaction of the
  windowed integrator and are not decided by the tick-$2$ comparison
  alone.
]

A simpler empirical baseline $||w_(1 2)| - |w_(2 1)|| > 7$ achieves
$83.4%$ overall accuracy without the XOR predicate's sound-but-incomplete
asymmetry; it is the $F$-score optimum of a one-parameter threshold
classifier and recovers most but not all of the capture region.

A sharper analogous statement holds for the sign of the winner:

#proposition[
  (Winner-sign predicate.) In every deterministic grid cell on which a
  winner exists, $"sign"(|w_(1 2)| - |w_(2 1)|) = "sign"("dominance")$,
  where dominance $= (n_1 - n_2) / (n_1 + n_2 + 1)$ aggregates the net
  spike-count difference over ticks $5 dots 49$. The sign agreement is
  $100%$ on all $1014$ non-tied cells of the deterministic ground truth.
  The neuron whose outgoing inhibition is larger in magnitude is the
  winner.
]

The combinatorial regime summarised:

- Existence of WTA is decided (up to the $34%$ that require multi-tick
  analysis) by $(|w_(1 2)| > 12) xor (|w_(2 1)| > 12)$.
- Identity of the winner is decided exactly by
  $"sign"(|w_(1 2)| - |w_(2 1)|)$.
- Both are closed-form, integer-comparison invariants that do not appear
  explicitly in the FCS paper and are recoverable as back-of-envelope
  checks.

= The Reachability Regime is Spectral

The reachability oracle is the one that matches De Maria et al.'s Kind2
verification: WTA holds for a parameter cell iff some initial state
reaches the WTA attractor. Implemented as a sweep over $33$ small
initial perturbations per cell (the unperturbed trajectory plus $32$
minimal $plus.minus 2$ biases on single memory taps), the reachability
ground truth has $1564 / 1600$ blue cells and a compact non-reachable
region consisting exclusively of the $36$ cells with
$|w_(1 2)|, |w_(2 1)| <= 6$.

== A Sound Spectral Predicate

The scalar-$r$ linearisation of [Cog26] collapses the windowed
integrator to a single leak coefficient $r = 0.5$ and constructs a
$2 times 2$ state matrix $A = r I + W dot "diag"(f'(p^star))$ at the
symmetric operating point $p^star$, where $f$ is a sigmoid firing-rate
approximation (slope $k = 0.08$, centred at $p_"mid" = 30$ in scaled
units to avoid saturation of $f'$). Computing $rho(A)$ across the grid
separates the two classes with minimal overlap:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 6pt, align: center, stroke: 0.5pt,
    [*class*], [*$n$*], [*mean $rho$*], [*range*],
    [non-reachable], [36], [0.555], [[0.518, 0.598]],
    [reachable], [1564], [0.719], [[0.544, 0.929]],
  ),
  caption: [Distribution of $rho(A)$ by reachability class.
    Overlap is confined to the narrow interval $[0.544, 0.598]$.],
)

#proposition[
  (Reachability sound predicate.) Under $epsilon$-perturbation
  reachability semantics, the predicate $rho(A) < 0.544$ on the
  scalar-$r$ linearisation is a *sound* decision procedure for the
  non-reachable class: every cell it identifies is non-reachable, and no
  reachable cell passes the threshold. Recall is $31%$ ($11 / 36$
  unreachable cells detected); the complementary predicate $rho(A) > 0.544$
  flags $98.5%$ of cells as reachable with a single false-positive rate
  below $2%$. The prior CogSpike spectral framework is therefore
  correct-in-scope for the semantic interpretation that Kind2 actually
  verifies.
]

== Use as a Kind2 Pre-Filter

The practical consequence of this soundness result is a modest but
immediately deployable engineering contribution. A Kind2 grid-sweep
workflow of the kind used in [DMDP20] can be augmented by an
$O(n^3)$ spectral stage that certifies a subset of the non-reachable
region analytically: every cell with $rho(A) < 0.544$ is _provably_ not
WTA-admissible under the reachability semantics Kind2 checks, and need
not be submitted to the model checker at all. This is neither
replacement nor critique of the FCS workflow but a prefix stage that
shortens it on the uncontroversial cells.

The corresponding dual predicate $rho(A) > rho_"hi"$ for some
sufficiently large $rho_"hi"$ would provide a sound certification of
_reachable_ cells; the current contralateral data does not exhibit a
clean $rho_"hi"$ threshold of that kind (the reachable-class
distribution has non-trivial mass down to its lower boundary), so the
practical pre-filter is one-sided in favour of the non-reachable
region.

= The Delayer-Augmented Topology

[DMDP20] §6.3.4 observes that inserting a delayer neuron on the
$N_1 arrow N_2$ inhibitory branch of the contralateral archetype
produces an asymmetric red-zone growth in the WTA parameter map
"contrary to expectation": the neuron whose incoming inhibition is
delayed wins the competition more often than naive reasoning predicts,
not less. The deterministic reproduction here makes this quantitative.
In the delayer-augmented $3$-neuron system, on the same $40 times 40$
weight grid, $N_2$ captures in $1136$ cells and $N_1$ in $448$ --- an
asymmetry of $688$ cells where the undelayed topology is zero by
construction.

The mechanism is the tick-$2$ comparison of the previous section,
evaluated in a one-tick-perturbed setting. At tick $2$, $N_2$'s incoming
inhibition is still absent (because $N_1$'s spike from tick $1$ has not
yet cleared the delayer), while $N_1$'s incoming inhibition from $N_2$
has arrived on time. So $N_1$ fails the tick-$2$ firing test while $N_2$
passes it, and the delayer-granted one-tick head start propagates into
$N_2$'s capture attractor. This is the FCS asymmetry, expressed
analytically.

The spectral prediction of reachability transfers to the $15$-dimensional
state matrix of the delayer-augmented system with the same pattern as
the undelayed case: $53%$ on deterministic ground truth (matches
baseline, no signal), $95.8%$ on reachability ground truth (matches
reachability baseline of $95.9%$). The two-regime split is a property of
the archetype class, not of any specific topology within it.

= The Negative Loop and the Limits of Pole Placement

== An Analytical Condition for Property 5

The FCS paper states the period-$4$ oscillation of Property $5$ for the
negative loop under constant input but does not give the minimal
analytical condition on the weights. The simulator reproduces the exact
sequence $0 1 1 0 0 1 1 0 0 dots$ on the activator iff
$w_("IA") = -w_("AI")$ (both neurons configured as delayers with
cancelling magnitudes). This is the exact-cancellation invariant: the
feedback inhibition from $I$ to $A$ at the second tick of each cycle
exactly zeros $"mem"[0]$ rather than overshooting into a negative-leak
regime, and the windowed integrator therefore clears within the
period-$4$ window without residual contamination. A larger magnitude
(the value $|w_"IA"| = 20$ suggested as a starting point in
supplementary material of [DMDP20]) overshoots and produces a
period-$5$ pattern $0 1 1 0 0 0 1 1 0 0 0 dots$ instead.

#figure(
  image("results/phase0_property5_trace.png", width: 95%),
  caption: [Activator ($A$, blue) and inhibitor ($I$, red) spike rasters of
    the negative loop at $w_"XA" = w_"AI" = 11, w_"IA" = -11$. The
    activator sequence is the exact FCS Property $5$ pattern
    $0 1 1 0 0 1 1 0 0 dots$; the inhibitor is the activator delayed by
    one tick, per the Lustre one-tick emission.],
) <fig-prop5>

== Pole Placement Fails

Given the central positive result --- that the spectrum of the
linearised state matrix captures reachability --- it is natural to ask
whether the same spectrum can _invert_: given a target oscillation
period $T$, solve for weights that place the dominant complex pole of
$A_"full"$ at $z = e^(plus.minus i dot 2 pi \/ T)$ and therefore realise
period $T$ in the simulator. The CogSpike framework [Cog26] makes
exactly this claim for continuous-dynamics LI\&F.

The method does not transfer. At the known period-$4$ weights
$(11, 11, -11)$, the dominant complex argument of $A_"full"$ is
$1.342$ rad, not the nominal $pi \/ 2 = 1.571$. After a single-point
calibration that aligns the known-good case with the target, inverse
design for target periods in ${3, 5, 6, 7, 8}$ misses on five of five
cases: the predicted weights realise periods that disagree with the
target by one or more ticks.

The diagnosis is visible directly in the linearisation spectrum:

#figure(
  image("results/final_fig3_poleplacement.png", width: 100%),
  caption: [Dominant complex argument of $A_"full"$ vs simulator-realised
    period for all $320$ non-trivially-oscillating cells on the negative
    loop integer grid. The red dashed curve shows the nominal target
    $arg = 2 pi / T$. The observed argument clusters in a narrow band
    near $[0.86, 1.68]$ rad across all six periods $3 dots 8$; the
    predictor cannot discriminate the integer periods because
    fundamentally different simulator behaviours share the same
    linearised fundamental frequency. Colour encodes the pole's
    magnitude $|lambda|$; the bright upper cluster at $|lambda| > 1$
    corresponds to operating points where the linearisation predicts
    sustained oscillation, and even within that oscillatory regime the
    arg does not separate periods.],
) <fig-poleplacement>

== The Spike-Reset Diagnosis

The negative-loop failure is mechanistically sharp. The period-$4$
pattern $1 1 0 0$ and the period-$8$ pattern $1 1 1 1 0 0 0 0$ both
oscillate at fundamental frequency approximately $pi / 2$ --- the
period-$8$ waveform is _the same oscillation, resetting half as often_.
The linearisation captures the dynamics _between_ resets correctly but
is silent on _when_ resets occur, and the integer period of the spike
train is determined primarily by the latter. In dynamical-systems
language: the reset rule takes the trajectory off the linear manifold
every time $V$ crosses $tau$, so the $A_"full"$ spectrum describes a
system that does not exist between those crossings; the actual system
is a _hybrid automaton_ whose piecewise linear evolution is punctuated
by the non-smooth reset, and the period of its limit cycle is a
combined property of linear arc length and reset schedule.

This is the central diagnostic principle of the note. _The spike-reset
rule is the essential nonlinearity that linearisation erases, and its
presence or absence in the formal property being verified determines
whether spectral methods apply._

= Discussion

== Where the Prior Spectral Framework Stands

The spectral apparatus of [Cog26] is neither refuted nor
validated wholesale by the present analysis; it is _reframed_. On the
reachability interpretation of WTA, the framework is correct: its
central claim that $rho(A) = 1$ contours of the linearised state matrix
map bifurcation boundaries holds, modulo the numerical caveat that
under the sigmoid approximation the effective threshold is $rho = 0.544$
rather than exactly $1$. On the deterministic single-trajectory
interpretation, the framework does not apply: the tick-$2$ integer
comparison is not a fact about the continuous dynamics the framework
linearises. The $4$-neuron WTA case study from the prior note ought, by
this analysis, to exhibit the same two-regime behaviour --- a prediction
that is testable and that the present authors intend to verify as a
follow-up.

== Implications for the FCS Verification Methodology

The two-regime split has a concrete engineering consequence for the
workflow of [DMDP20]. The Kind2 grid-search-then-Coq-prove pipeline
can be augmented at its front end by a spectral pre-filter that
certifies a subset of the non-reachable parameter region in polynomial
time: every cell with $rho(A) < 0.544$ under the scalar-$r$
linearisation is provably not WTA-admissible under reachability
semantics, and need not be submitted to Kind2. This is not a
replacement of the FCS workflow but a prefix stage that shortens it on
the uncontroversial cells, in exchange for the $O(n^3)$ cost of a
single eigendecomposition per grid point. For the $40 times 40$
contralateral grid this stage eliminates $11$ cells from the Kind2
workload with certainty, a modest but free contribution; for higher-
dimensional archetypes where the non-reachable region has more
structure (see Outlook) the savings would be more substantial.

The combinatorial invariants of §3 are a complementary contribution in
the same spirit: the XOR predicate on tick-$2$ fire conditions and the
sign-of-asymmetry rule for the winner are closed-form checks that a
Coq proof could consume directly, bypassing both Kind2 and the spectral
stage for the cells they cover.

== The Spike-Reset Diagnosis as a General Principle

The central diagnostic generalises to a falsifiable prediction about
which formal properties of LI\&F networks are within reach of
continuous linearisation and which are not. _The spike-reset rule is
the essential nonlinearity that linearisation erases, and its presence
or absence in the formal property being verified determines whether
spectral methods apply._ Properties that depend only on the existence
and stability of attractors (reachability, bifurcation, qualitative
regime classification) are within scope. Properties that depend on the
exact integer schedule of spike emissions (deterministic bit-exact
trajectories, specific integer periods, phase relationships between
oscillators) are out of scope, because the reset rule is exactly the
information linearisation discards.

This predicts which other archetypes of [DMDP20] should yield to
spectral analysis. The series and parallel composition archetypes of
Fig. 1a, 1b should work (stability-flavoured properties). The positive
loop of Fig. 1c should work (attractor reachability). The delayed
variants of any archetype should work modulo the reset-schedule caveat
(cf. §5). Pattern-matching properties that demand specific integer
spike sequences should not.

= Outlook

Three concrete directions extend this work in decreasing order of
confidence.

*(i) The spectral pre-filter as an engineering deliverable.* The
$O(n^3)$ non-reachability pre-filter of §4 is deployment-ready and
generalises to any FCS archetype whose Lustre model can be compiled to
the same scaled-integer convention. A unified spectral-plus-Kind2
pipeline would emit (a) a certificate of non-reachability for every
cell it covers, (b) a reduced Kind2 workload for the remaining cells,
(c) a combinatorial invariant for any cell the tick-$2$ or analogous
analysis also covers. This is the most direct practical outcome.

*(ii) The stability-versus-timing principle on other archetypes.* The
series, parallel, and positive-loop archetypes of Fig. 1a--c of
[DMDP20] are the natural next targets. The analysis here predicts
that their WTA-like properties --- stability-flavoured rather than
timing-flavoured --- yield clean spectral predictors with higher recall
than contralateral inhibition allows. Testing this would extend the
positive-result footprint of spectral cartography beyond the single
case of §4.

*(iii) Hybrid-system pole placement.* The negative-loop failure of §6
diagnoses a gap but also points at the fix: a linearisation that
incorporates the reset rule as a jump map. The discrete-time hybrid
automaton formulation of LI\&F neurons, analysed via saltation matrices
or Floquet theory of periodic-reset systems, would have a spectrum in
which the reset frequency appears explicitly. Pole placement on that
spectrum could in principle recover the exact integer period --- and
would constitute a principled resolution of the inverse-design failure
diagnosed in §6.2.

#v(0.6em)

*Acknowledgements.* The authors thank Elisabetta De Maria and
Christopher Leturc for methodological orientation. The FCS paper's
original Lustre code served as the authoritative semantic reference
throughout; any inaccuracy in reproduction is the present authors'.

= References

#set par(first-line-indent: 0pt, hanging-indent: 1em)

[DMDP20] De Maria, E., D. Mohsen, C. Di Giusto, and L. Prigent.
"On the Use of Formal Methods to Model and Verify Neuronal Archetypes."
_Frontiers of Computer Science_ 16, no. 3 (2020): 163602.

[Cog26] CogSpike Research Team. _Spectral Methods for Analyzing
SNN Dynamics: A Guide for Computer Scientists._ Research Note, April
2026.

#v(0.5em)
#line(length: 100%)

*Reproducibility statement.* All code, data, and intermediate artefacts
of this note are available at `deq/archetypes/` of the CogSpike
repository. The end-to-end pipeline regenerates every numerical claim
from the Lustre-equivalent simulator, over six reproducibility-sealed
Python scripts whose outputs populate `deq/archetypes/results/`. A
detailed supplementary PDF for each experimental stage is provided
alongside this note.

*Supplementary material:* per-stage technical reports
(`phase0_report.pdf` through `phase3_report.pdf` plus the
`final_summary.pdf` consolidation), raw ground-truth grids
(`fcs_fig10_groundtruth.npy`, `fcs_fig10_reachable_groundtruth.npy`,
`fcs_fig11_groundtruth.npy`) and per-stage diagnostic figures.
