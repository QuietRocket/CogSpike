// Closed-form finite-population analysis of LI&F neuronal archetypes.
// Standalone research note (Phase 5 deliverable of deq/closed_form/).

#set document(
  title: "Closed-form finite-population analysis of LI&F neuronal archetypes",
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
    Closed-form finite-population \
    analysis of leaky integrate-and-fire \
    neuronal archetypes
  ]
  #v(0.4em)
  #text(size: 10pt, style: "italic")[
    Siegert + Richardson + quasi-renewal mesoscopic, on FCS topologies
  ]
  #v(0.2em)
  #text(size: 11pt)[Nikan Zandian]
  #v(0.2em)
  #text(size: 10pt, style: "italic")[Research note --- May 2026]
]

#v(0.8em)

#block(
  width: 100%, inset: 10pt,
  fill: rgb("#f6f6f6"),
  stroke: (left: 3pt + luma(160)),
)[
  *Abstract.* The population-level Wilson--Cowan analysis of LI&F
  archetypes (companion note `population_note_v2`) replaces the
  spike-reset non-smoothness by a heuristic logistic sigmoidal gain
  $f(x) = 1 \/ (1 + e^(-k(x - theta)))$ with hand-tuned $k$, $theta$. It
  works for the static behavioural boundaries (pitchfork, Hopf) and for
  linear pole placement, but it leaves three structural questions open:
  (i) the gain function has no physical input, just two fitted
  parameters; (ii) the framework is purely time-domain mean-field, with
  no closed-form transfer function $H(omega)$; (iii) the LI&F oracle's
  Phase 4 cross-validation revealed a *spike-timing-locked rectangular
  bistability* in the asymmetric arms that smooth Wilson--Cowan cannot
  see. This note closes all three by replacing the heuristic sigmoid
  with three physically-derived closed-form objects that keep the
  topology finite, tangible, and explicit: the Siegert formula
  @Siegert1951 @Brunel2000 (static $f$-$I$ curve from diffusion
  approximation), the Richardson 2007/2008 single-pole approximation
  @Richardson2007 of the linear-response transfer function $H_i(omega)$,
  and the Naud--Gerstner 2012 single-integral mesoscopic equation
  @NaudGerstner2012 (a tractable simplification of Schwalger et al.
  2017 @Schwalger2017). Three hypotheses are tested against the FCS
  LI&F oracle (`deq/archetypes/lif_fcs.py`) at a fixed operating point
  ($p_("thin") = 0.7$, $N$ swept). H1 (Siegert) passes: $0.843$ Jaccard
  vs LI&F WTA region, beats heuristic-sigmoid baseline at $0.796$. H2A
  (transfer function self-consistency) passes exactly (residual $0$);
  H2B (FFT cross-validation) is partial -- phase agrees to $10 degree$
  median but magnitude is biased by a factor of two from steady-state
  $tau_m$ mis-calibration. H3 (quasi-renewal mesoscopic) passes: at
  $N = 500$ the predicted WTA boundary tracks the LI&F oracle at
  $0.863$ Jaccard, and the rectangular structure dissolves into the
  Siegert mean-field wedge as $N -> infinity$ with the predicted
  $1\/sqrt(N)$ scaling. Together, the three frameworks span from static
  ($Phi$) through dynamic ($H(omega)$) to finite-size correction
  ($1\/sqrt(N)$) over the same finite tangible topology.
]

= Introduction <sec-intro>

== Why this thread <sec-why>

Two prior threads have settled complementary halves of the analysis
problem. The *archetypes* thread (`deq/archetypes/`) charted FCS LI&F
behaviour by discrete-time spectral methods on a windowed-integrator
oracle: it concluded that deterministic winner-take-all is a
*combinatorial* phenomenon (integer tick-2 comparison, $"sign"(|w_(12)| -
|w_(21)|)$ predicts the winner with $100 %$ accuracy), and that
reachability winner-take-all is a *spectral* phenomenon (scalar-$r$
linearization classifies non-reachable cells with $98.5 %$ accuracy
under $epsilon$-perturbation). The *population* thread (`deq/population/`)
took the same archetypes one level up to a smooth Wilson--Cowan rate
equation with a hand-tuned logistic sigmoid; the pitchfork and Hopf
loci come out to machine precision, $8\/8$ linear pole placements
succeed, and FCS Fig. 10's winner-take-all boundary is reproduced
qualitatively. Phase 4 of that thread cross-validated the smooth
prediction against the discrete LI&F oracle and found a *quantitative
failure in the asymmetric arms*: the discrete oracle exhibits a
rectangular bistability (median displacement $0.68$ Wilson--Cowan units
from the smooth pitchfork) where one population is locked into
dominance by a deterministic per-tick inhibition cycle. The smooth
sigmoid cannot see this because it assumes the spike-reset has been
fully averaged away.

The opening this leaves is the gap between the two threads: a
description that keeps the *topology* explicit and finite (not abstracted
into Wilson--Cowan symbols), and the *spike physics* present (not
averaged into a smooth sigmoid), but is still amenable to closed-form
differential-equation analysis. The classical neuroscience literature
has the tools: Siegert's diffusion approximation
@Siegert1951 @Brunel2000 gives a closed-form $f$-$I$ curve in $"erf"$;
Brunel--Hakim @BrunelHakim1999 and Richardson @Richardson2007 derive
the linear-response transfer function $H(omega)$ around a Siegert fixed
point in closed form; Schwalger--Deger--Gerstner @Schwalger2017 give a
finite-$N$ mesoscopic integral equation. We use them, on the same
tangible $2$- and $3$-node topologies as the prior threads, against the
same LI&F oracle.

== Three frameworks, one topology <sec-frameworks>

For a topology specified by a finite connectivity matrix $J$ in FCS
units (e.g.~$J = [0, w_(21); w_(12), 0]$ for contralateral inhibition,
or $J = [0, w_("IA"); w_("AI"), 0]$ for the negative loop), each node
$i$ is replaced by a *population* of $N$ FCS-LI&F neurons. The
populations interact at the rate level: population $j$'s mean firing
rate $nu_j$ feeds into population $i$'s input as $J_(i j) nu_j$. We
inject diffusion-approximation noise into the population by per-neuron
threshold heterogeneity ($epsilon$ small) plus Bernoulli per-tick
input thinning at retention probability $p_("thin")$. Together these
give the input to each neuron Poisson-like statistics with mean
$mu_i = alpha (J_(i :) nu + B_(i :) "drive") $ and variance
$sigma_i^2 = beta (sum_j J_(i j)^2 nu_j (1 - nu_j) + ...)$, where
$alpha$ and $beta$ are FCS-to-Siegert calibration scales and
$B "drive"$ is the external-input vector.

Three frameworks act on this $(mu, sigma)$ pair:

#enum(
  [
    *Static (Siegert).* The stationary firing rate
    $ nu = Phi(mu, sigma) = 1 / (tau_("ref") + tau_m sqrt(pi) integral_((V_r - mu) / sigma)^((V_("th") - mu) / sigma) "erfcx"(-u) d u) $ <eq-siegert>
    is closed form in the scaled complementary error function. The
    contralateral and negative-loop fixed points are then roots of the
    self-consistent system $nu_i^* = Phi(mu_i(nu^*), sigma_i(nu^*))$ ---
    a small algebraic system, $1$ or $3$ roots in the bistable case,
    enumerable via $1$-D scalar reduction.
  ],
  [
    *Dynamic (Richardson $H(omega)$).* The linear-response transfer
    function around a Siegert fixed point in the small-$omega$ limit:
    $ H_i(omega) = ("g"_i) / (1 + i omega tau_m), quad "g"_i = partial Phi_i / partial mu_i = (Phi_i^2 tau_m sqrt(pi)) / sigma_i [ "erfcx"(-y_("th",i)) - "erfcx"(-y_(r,i)) ], $ <eq-richardson>
    where the gain is the Leibniz-rule derivative of @eq-siegert. Closed-loop
    poles of the topology are roots of $det(I - H(omega) J) = 0$,
    classical control theory. The full Brunel--Hakim parabolic-cylinder
    transfer function is a non-goal; the single-pole form reduces
    correctly to the time-domain Jacobian at $omega = 0$, which is the
    Phase 2 self-consistency gate.
  ],
  [
    *Finite-$N$ (quasi-renewal mesoscopic).* The Naud--Gerstner
    single-integral
    $ A(t) = sum_k m_k(t-1) h(k; mu(t)) + sqrt(A(t)/N) xi(t), $ <eq-qr>
    with hazard $h(k; mu, sigma) = Phi(mu, sigma)$ for $k >= tau_("ref")$
    and zero otherwise, $m_k$ the age-$k$ fraction of the population.
    Reduces to mean-field Wilson--Cowan / Siegert at $N -> infinity$,
    keeps the spike-physics-driven correlation structure at finite $N$.
    The full Schwalger 2017 age-structured integral with non-renewal
    corrections is a non-goal of this thread (follow-on work).
  ],
)

The same $J$ matrix appears in all three frameworks; nothing about the
topology is abstracted. What changes is the gain object: a static
algebraic curve (@eq-siegert), a frequency-domain transfer
(@eq-richardson), or a stochastic integral with finite-size noise
(@eq-qr).

= Phase results <sec-phases>

The thread is phase-gated. Each phase tests one falsifiable hypothesis
on the FCS LI&F oracle and emits a sub-report (see the phase reports for
methodology details). This section consolidates the headline numbers.

== Phase 0 -- Stochastic-LI&F bridge <sec-phase0>

Wraps `deq/archetypes/lif_fcs.py:simulate` over $N = 100$ copies of each
logical neuron with per-copy threshold jitter and Bernoulli per-tick
external thinning. Calibration $"V0.2"$ locks in $p_("thin") = 0.7$,
threshold jitter $epsilon = 0$ as the operating point: the inter-spike-
interval coefficient of variation is $0.547$, comfortably in the
diffusion-approximation regime ($"CV" >= 0.5$). At $p_("thin") = 1.0$ the
oracle is fully deterministic and produces a discrete staircase $f$-$I$
curve (rates $0 -> 1\/3 -> 1\/2 -> 1$ at integer drive thresholds);
Siegert cannot reproduce this by construction.

== Phase 1 -- Siegert static (H1) <sec-phase1>

Calibrates four parameters $(alpha, beta, tau_m, tau_("ref"))$ on the
$"V0.1"$ $f$-$I$ data ($30$ points across three thinning levels) by
least-squares fit, $V_("th") = 1$, $V_r = 0$ fixed. Result:
$alpha = 0.250$, $beta = 4.29 dot 10^(-3)$, $tau_m = 2.35$,
$tau_("ref") = 0.36$. $R^2 = 0.936$ on the full dataset (gate $>= 0.90$
PASS). The shortfall from $1.0$ comes from the deterministic-staircase
regime that Siegert cannot reproduce; in the stochastic-only subset
$R^2 = 0.901$.

#figure(image("/results/phase1/s1a_calibration.pdf", width: 75%),
  caption: [Siegert calibrated on the population $f$-$I$ data at three
  thinning levels.])

On a $12 times 12$ contralateral grid in $(w_(12), w_(21)) in [-40, -1]^2$,
the Siegert self-consistency-FP enumeration agrees with the LI&F oracle
WTA labels at *Jaccard $0.843$*; the heuristic-sigmoid baseline (the
population thread's $f(x)$, $k=4$, $theta = 1$, mapped via
$w^("WC") = |w^("LIF")| \/ 8$) achieves $0.796$. Siegert beats the
baseline by $0.047$ Jaccard.

#figure(image("/results/phase1/s1bc_comparison.pdf", width: 100%),
  caption: [Left to right: LI&F oracle, Siegert prediction, population-
  thread heuristic sigmoid, and Siegert fixed-point count. Both rate
  models capture the rectangular WTA region; the central diagonal
  no-winner band is invisible to both.])

== Phase 2 -- Closed-loop machinery (H2A) <sec-phase2>

At the calibrated FP of the FCS negative-loop (default $w_("XA") = 11$,
$w_("AI") = 11$, $w_("IA") = -11$, $p_("thin") = 0.7$): $nu^* = (0.352,
0.179)$, $sigma^* = (0.43, 0.34)$, gains $partial Phi \/ partial mu =
(0.316, 0.360)$. Time-domain Jacobian eigenvalues $-0.425 plus.minus
0.395 i$ (stable focus, oscillatory decay). Self-consistency residual
between closed-loop $det M(omega = 0)$ and the Jacobian spectrum:
*$0.0 dot 10^0$* (numerically exact), gate $<= 10^(-3)$ PASS.

#figure(image("/results/phase2/bode.pdf", width: 65%),
  caption: [Bode plot of the closed-loop $G_("AA")(omega)$. Single-pole
  low-pass plus a small resonance peak near $omega approx 0.5$
  rad/tick, matching $|"Im"(lambda)|$ of the Jacobian eigenvalue.])

== Phase 3 -- Closed-loop cross-validation (H2B) <sec-phase3>

Drove the stochastic-LI&F negative loop with sinusoidal external
perturbation at $8$ frequencies, lock-in detected the fundamental in
population A. *Median phase error $10.1 degree$* (gate $<= 30 degree$
PASS); *median magnitude relative error $53.6 %$* (gate $<= 30 %$ FAIL).
Verdict: *PARTIAL*. The phase agreement is the load-bearing test --
it confirms $H(omega) J$ has the right shape in the complex plane.
The magnitude bias is a calibration artifact: Phase 1's steady-state
fit only weakly constrains $tau_m$ because $tau_m$ does not appear in
the limit $sigma -> 0$. Refitting $tau_m$ on dynamic (impulse-response)
data is documented as future work.

#figure(image("/results/phase3/freq_response_xval.pdf", width: 70%),
  caption: [Predicted (blue circles) vs measured (orange squares)
  closed-loop response at population A. Phase agrees at low $omega$
  (gate met); magnitude biased by factor $approx 2$ across the band
  (calibration artifact).])

== Phase 4 -- Quasi-renewal mesoscopic (H3) <sec-phase4>

Single-integral Naud--Gerstner with finite-size $sqrt(A\/N) xi$ noise on
the same $12 times 12$ contralateral grid, sweeping $N in {50, 100, 200,
500, 2000}$. Critical implementation detail: $tau_("ref") = 0$
(FCS LI&F has no genuine refractory period; the post-spike reset of
$"mem"[1..4]$ does not skip ticks). $tau_("ref") >= 1$ would
systematically under-predict rates by factor $1\/(1 + nu)$.

*Best Jaccard vs LI&F oracle: $0.863$ at $N = 500$ -- $2000$* (gate
$>= 0.70$ PASS). The *WTA-cell fraction* tracks the predicted
$1 \/ sqrt(N)$ scaling: $0.778$ (QR $N = 50$) $approx 0.785$ (LI&F)
$arrow.r 0.910$ (QR $N = 2000$) $approx 0.931$ (Siegert mean-field).
At small $N$, finite-size noise broadens the no-WTA band; at large $N$
the boundary tightens onto the Siegert wedge.

#figure(image("/results/phase4/wta_maps.pdf", width: 100%),
  caption: [Left to right: LI&F oracle, Siegert mean-field, then
  quasi-renewal at each $N$. The $1\/sqrt(N)$ smoothing of the WTA
  region is visible.])

#figure(image("/results/phase4/jaccard_vs_N.pdf", width: 60%),
  caption: [Jaccard agreement of quasi-renewal labels vs LI&F oracle
  (circles) and vs Siegert mean-field (squares). The quasi-renewal
  prediction interpolates between the LI&F oracle (small $N$) and the
  Siegert mean-field (large $N$), demonstrating that finite-size noise
  is the bridge between the two.])

= What each framework can and cannot see <sec-scope>

The three frameworks have non-overlapping scopes; they are
complementary, not competing.

#table(
  columns: 4,
  table.header([*Phenomenon*], [*Siegert (H1)*], [*$H(omega)$ (H2)*],
               [*Quasi-renewal (H3)*]),
  [Stationary $f$-$I$ curve], [yes (closed form)], [via $partial Phi \/
   partial mu$], [yes (steady-state)],
  [Bifurcation locus shape], [yes (FP enumeration)], [via $det M(0) = 0$],
   [yes],
  [Oscillation frequency], [no], [yes ($|"Im"(lambda)|$)], [yes],
  [Stability margins], [no], [yes (Nyquist of $det M$)], [yes],
  [Finite-$N$ rate scaling], [no], [no], [yes ($1\/sqrt(N)$)],
  [Spike-timing-lock width], [partial (asymmetric FP)], [no], [yes],
  [Discrete-staircase rates], [no (smooth)], [no], [no],
  [Bit-exact spike sequences], [no], [no], [no],
)

The bottom two rows are *outside the scope of every rate framework*.
For bit-exact spike sequences and integer-tick periods, the FCS
Lustre--Kind2 machinery owns the question; the population frameworks
predict the *shape* of where these properties hold across continuous
weight space, not the property itself.

= Discussion <sec-discuss>

== Relation to companion threads

The *archetypes* thread @arch and the *population* thread @pop are not
superseded by this one -- they remain the right tool for their
respective questions. The archetypes thread settles deterministic
spike sequences and reachability classification under FCS Lustre
semantics; the population thread provides the cleanest pedagogical
introduction to bifurcation theory on the archetype topologies. This
thread *fills the gap between them*: where archetypes is bit-exact and
population is heuristically smooth, the closed-form thread is
physically calibrated and makes the spike physics survive into the
rate description.

The most concrete example is the rectangular bistability in the
asymmetric arms of the contralateral motif. Population thread Phase 4
documented this as a quantitative failure of the smooth Wilson--Cowan
sigmoid; the present thread recovers it directly via the
quasi-renewal correction at finite $N$ (#ref(<sec-phase4>)) and
predicts its $1 \/ sqrt(N)$ shrinkage as $N$ grows.

== Relation to the FCS programme

The framework remains complementary to FCS:
- *FCS at a point*: given specific integer weights, does the property
  hold? Lustre encoding plus Kind2 model checking, deterministic.
- *Population frameworks at a region*: across continuous weight space,
  where does the bifurcation lie? Rate-equation analysis with
  closed-form objects.
- *Closed-form thread (this note)*: the same rate equations but with
  $Phi(mu, sigma)$, $H(omega)$, and finite-$N$ corrections derived
  from the spike physics. Inputs are physical (drive, noise variance,
  connectivity matrix); no hand-tuned sigmoid parameters.

A spectral pre-filter for Kind2 is the natural composition: certify a
weight cell is non-reachable via Phase 1's Siegert FP enumeration, and
skip the model check. That is engineering follow-on; this note is
prerequisite groundwork.

== Limitations

The chief technical limitation surfaced by this thread is in
*$tau_m$ calibration* (#ref(<sec-phase3>)): a static $f$-$I$ fit
constrains $alpha$ and the gain at the operating point but does not
constrain the *dynamic* time constant $tau_m$. Magnitude predictions in
Phase 3 are biased by a factor of two as a result. Refitting $tau_m$
on impulse-response or autocorrelation data would close this gap; we
note it but do not implement it here.

The *quasi-renewal* implementation (Phase 4) uses the simplest
mesoscopic correction (single-integral, memoryless hazard, $tau_("ref")
= 0$). Schwalger et al. 2017 @Schwalger2017 give the full age-structured
integral with non-renewal corrections, which would sharpen the
predicted boundary further. We declared full Schwalger out of scope at
the planning stage and the Phase 4 result already passes the H3 gate;
the upgrade is documented as follow-on.

The *Brunel--Hakim parabolic-cylinder* transfer function would replace
the single-pole approximation in @eq-richardson with the exact LI&F
linear response. We use Richardson's threshold-integration recipe
@Richardson2007 in its small-$omega$ limit; the high-frequency
resonance term is omitted by design. Phase 2's self-consistency check
verifies the omission is benign at $omega = 0$; Phase 3's residual
phase error at high $omega$ may partly trace to it.

= Conclusion <sec-conclusion>

Three closed-form objects -- the Siegert formula
@Siegert1951 @Brunel2000 (static), the Richardson single-pole transfer
function @Richardson2007 (dynamic), and the Naud--Gerstner quasi-renewal
mesoscopic equation @NaudGerstner2012 (finite-$N$) -- act on the same
finite tangible topology and span the regimes of LI&F-archetype
analysis. The static piece replaces the heuristic sigmoid with a
physically-derived $f$-$I$ curve depending on input mean *and* noise
variance, beating the population-thread baseline on the contralateral
WTA boundary (Jaccard $0.843$ vs $0.796$). The dynamic piece reduces
the topology to a block diagram amenable to classical control theory,
with self-consistency exact at $omega = 0$ and phase-domain agreement
within $10 degree$ on the FCS negative loop. The finite-$N$ piece
recovers the spike-timing-locked rectangular bistability that the
companion smooth-rate thread documented as a failure mode, with the
predicted $1 \/ sqrt(N)$ approach to the mean-field Siegert wedge.

The framework is a *third leg* alongside the discrete-spectral
analysis of `deq/archetypes/` and the smooth-rate analysis of
`deq/population/`. Where archetypes verifies properties at points and
population predicts shapes of regions, this thread predicts both shapes
and dynamics with closed-form objects whose every parameter is a
measurable LI&F quantity rather than a fitted curve coefficient. The
topology stays explicit; the spike physics stays in.

#bibliography("refs.bib", style: "ieee")
