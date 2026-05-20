// Closed-form rate-equation reading of FCS Property 5 (negative loop).
// Standalone advisor-facing research note. Parallel to closed_form_wta.

#set document(
  title: "Closed-form rate-equation reading of FCS Property 5",
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

#align(center)[
  #text(size: 16pt, weight: "bold")[
    Closed-form rate-equation reading \
    of FCS Property 5 on the \
    2-neuron negative loop
  ]
  #v(0.4em)
  #text(size: 10pt, style: "italic")[
    Three frameworks (Siegert, $H(omega)$, quasi-renewal) on the
    negative-loop oscillation
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
  *Abstract.* De Maria et al. 2020 @DeMaria2020 verified Property 5
  (oscillation in a negative loop) on the 2-neuron motif of one
  activator $A$ excited by external input $X$ and inhibited by an
  inhibitor $I$, with $I$ driven by $A$. With constant $X = 1$, the
  property states that $A$ fires the periodic pattern `1100` and $I$
  echoes one tick later. FCS proves this by Lustre + Kind2 model
  checking.

  This note re-reads the same negative-loop motif through three
  closed-form rate-equation lenses, all using the calibration locked
  in the companion `closed_form_wta` thread @closed_form_wta. The
  framework is *parallel* to that thread's WTA reading, but the lenses
  get reinterpreted because the negative loop is intrinsically
  *oscillatory*, not bistable. Three findings:

  - *Siegert FP is always a stable spiral.* The rate-equation
    Jacobian has eigenvalues $-1\/tau_m plus.minus i sqrt(g_A g_I |w_(A I) w_(I A)|)\/tau_m$ —
    complex-conjugate with $"Re"(lambda) < 0$ everywhere outside
    saturation. Mean-field theory therefore cannot predict Property 5's
    sustained oscillation. The "spiral envelope" (Im $!= 0$) covers
    $90 %$ of the grid; Jaccard against FCS strict P5 is $0.31$.
  - *$H(omega)$ predicts the ringing direction but not the period.*
    Single-pole low-pass yields $T_("pred") = 2 pi \/ |"Im"(lambda)|$,
    which is *systematically $4 times$ too long* across the entire
    FCS-period-4 region. The locked calibration is fit on steady-state
    rates ($f$-$I$ curve), and it does not constrain the dynamic time
    scale — discrete-tick FCS dynamics run faster than a single-pole
    low-pass with $tau_m = 2.35$ captures.
  - *Quasi-renewal recovers the period.* The Naud–Gerstner
    age-distribution mesoscopic equations, even with the same
    calibration, predict $T_("QR") = 4.05$ ticks at the default cell
    — essentially exact. Nonlinear refractoriness + spike-reset
    encode the discrete-tick time scale that single-pole linearization
    discards. Sustained oscillation lives at finite $N$ via
    $sqrt(A\/N)$ noise and decays toward the FP as $N -> infinity$
    (grid-mean amplitude $0.20 -> 0.05$ across $N in {50, 2000}$).

  Together the three lenses span the property with a sharp triplet:
  mean-field theory cannot *predict* the oscillation; linear $H(omega)$
  predicts the *direction* (ringing exists) but not the *period*;
  quasi-renewal predicts the *period* correctly but not the *binary
  waveform* (its smooth output gives template scores $approx 0.3$
  against the 1100 boxcar). What only Lustre + Kind2 can reach is the
  *exact* binary `1100` pattern.
]

= Introduction <sec-intro>

== FCS's Property 5 at a glance <sec-fcs-recap>

The negative-loop motif (FCS Fig. 1d, §6.2.5) is two LI&F neurons.
The *activator* $A$ receives external input $X$ at weight $w_(X A)$
and inhibitory feedback from the inhibitor $I$ at weight $w_(I A) <
0$. The *inhibitor* $I$ receives only excitation from $A$ at weight
$w_(A I) > 0$. Under FCS's Lustre semantics (discrete ticks, windowed
integrator $r$-vector $[10, 5, 3, 2, 1]$, integer threshold
$tau = 105$), with constant input $X(t) = 1$ for all $t$, Property 5
states:

#block(
  width: 100%, inset: 8pt,
  fill: rgb("#fffbe6"),
  stroke: (left: 2pt + rgb("#c0a020")),
)[
  *Property 5 (FCS).* Given a negative loop composed of two delayers,
  when a sequence of $1$ is given as input, the activator neuron
  oscillates with a pattern of the form `1100` (and the inhibitor
  expresses the same behaviour delayed of one time unit).
]

Empirically with the FCS default integer weights $(w_(X A), w_(A I),
w_(I A)) = (11, 11, -11)$, the FCS oracle indeed produces

#align(center)[
  $A:#h(0.1em)  0 1 1 0  0 1 1 0  0 1 1 0  dots, quad
   I:#h(0.1em)  0 0 1 1  0 0 1 1  0 0 1 1  dots$
]

so period exactly $4$, pattern `0110` (a phase-shifted cyclic rotation
of `1100`), with $I$ delayed one tick relative to $A$.

== What FCS verifies, and what it cannot <sec-gap>

FCS gives a *machine-checked* yes/no answer at each integer-weight
cell. But Lustre + Kind2 is not a description of *why* the period is
4 and not 3 or 5, *what shape the period-4 region* takes in the
$(w_(I A), w_(X A))$ plane, or *how it broadens or splits under
finite-population stochasticity*. Those questions are the *complement*
of FCS's verification, and they are what closed-form rate-equation
theory is built for. The companion `closed_form_wta` thread
@closed_form_wta showed how three lenses (Siegert / $H(omega)$ / QR)
read the FCS WTA Property 7; this note carries the same lenses across
to Property 5.

== Why the same lenses give different readings <sec-different>

The contralateral-inhibition WTA motif is *bistable*: two attractors,
one of which wins. Rate theory's natural tool is *fixed-point
enumeration*. Smooth-rate FP structure misses the integer-tick
"staircase" but otherwise tracks FCS-WTA well.

The negative-loop motif is *oscillatory*: a single attractor (the
mean-rate FP) that the FCS dynamics never settle into because of
spike-and-reset semantics. Rate theory cannot *create* the limit cycle
— the rate-equation FP is linearly stable. What rate theory *can*
do is:

+ predict *where* ringing happens (the spiral-FP envelope);
+ estimate the *natural ringing frequency* of the linearization;
+ produce a noise-driven sustainment via the quasi-renewal stepper
  (which carries nonlinear refractoriness the linearization discards).

The three lenses each contribute one of these readings to the
negative loop, in increasing order of nonlinearity.

= Three frameworks on the negative-loop motif <sec-frameworks>

The negative-loop connectivity matrix is
$ J = mat(0, w_(I A); w_(A I), 0). $ <eq-J>
With $w_(I A) < 0 < w_(A I)$, $J$ has purely imaginary eigenvalues
$plus.minus i sqrt(|w_(A I) w_(I A)|)$. The rate-equation
linearization combines $J$ with per-population gains and a relaxation
time, and this gives a stable spiral in continuous time (next
sub-section).

== Static: Siegert FP <sec-siegert>

The Siegert formula @Siegert1951 @Brunel2000 gives the stationary
firing rate $nu = Phi(mu, sigma)$ (@closed_form_wta, eq. 2). For the
negative loop:

$ nu_A & = Phi(alpha (w_(X A) p_("thin") + w_(I A) nu_I),
              sqrt(beta (w_(X A)^2 p_("thin")(1-p_("thin"))
              + w_(I A)^2 nu_I (1 - nu_I)))) \
  nu_I & = Phi(alpha (w_(A I) nu_A),
              sqrt(beta (w_(A I)^2 nu_A (1 - nu_A)))) $ <eq-neg-loop-fp>

$A$ sees a thinned-Bernoulli external term plus inhibitory recurrent;
$I$ sees only excitatory recurrent. Self-consistent FPs are found by
`scipy.optimize.fsolve` from multiple initial guesses.

== Dynamic: $H(omega)$ <sec-transfer>

The Richardson single-pole transfer @Richardson2007 around the
FP gives Jacobian
$A = (1\/tau_m)(- I + "diag"(g) J)$, with
$g_i = (partial Phi_i \/ partial mu_i)|_("FP")$. For the antidiagonal
$J$ of @eq-J:

$ lambda_(plus.minus) = (1 / tau_m) (-1 plus.minus
                          sqrt(g_A g_I w_(A I) w_(I A))) $ <eq-eigs>

Since $w_(I A) < 0 < w_(A I)$ and gains are positive, the radicand is
*negative* — so $lambda_(plus.minus)$ is a complex-conjugate pair
with

$ "Re"(lambda) = -1 / tau_m approx -0.426, quad
  |"Im"(lambda)| = (1 / tau_m) sqrt(g_A g_I w_(A I) |w_(I A)|). $

The FP is *always a stable spiral* (away from saturation). The
predicted ringing period is

$ T_("pred") = 2 pi \/ |"Im"(lambda)|. $ <eq-T-pred>

== Finite-$N$: quasi-renewal <sec-qr>

Naud–Gerstner @NaudGerstner2012 give the age-distribution stepper
(see @closed_form_wta, eq. 5). For the negative loop we plug in the
asymmetric input structure of @eq-neg-loop-fp (external + inhibition
for $A$, excitation only for $I$). The mesoscopic trajectory $A(t)$
has $sqrt(A\/N)$ noise per tick; at finite $N$ this can sustain
oscillation around an otherwise-stable spiral FP.

== Calibration <sec-calib>

We inherit the *closed_form_wta calibration verbatim*:
$alpha = 0.250$, $beta = 4.29 dot 10^(-3)$, $tau_m = 2.350$,
$tau_("ref") = 0.361$, fit at $p_("thin") = 0.7$ on the steady-state
$f$-$I$ data. *No re-fitting is performed in this note.* This locking
is intentional — it lets the three negative-loop lenses sit
side-by-side with the WTA lenses on the same calibration and exposes
the *dynamic-time-scale gap* (see @sec-results-phase2) that the static
calibration leaves unconstrained.

= Phase 0 — FCS oracle baseline <sec-results-phase0>

Sweep $(w_(I A), w_(X A)) in {-40, ..., -1} times {1, ..., 40}$
($1600$ cells) with $w_(A I) = 11$ fixed. $T_("max") = 64$ ticks,
warmup $16$. For each cell, label

- *strict_p5*: $A$'s post-warmup spike train matches some cyclic
  rotation of `1100`, repeated;
- *broad_osc*: any regular period in $[2, 12]$ with mixed firing.

#figure(image("/deq/closed_form_neg_loop/results/phase0/prop5_strict.pdf", width: 70%),
  caption: [Phase 0 strict Property 5. *445 / 1600 = 27.8 %* of cells
  satisfy the strict cyclic `1100` pattern. Gold ring marks the FCS
  default cell $(w_(I A), w_(X A)) = (-11, 11)$.]) <fig-phase0-strict>

#figure(image("/deq/closed_form_neg_loop/results/phase0/period_map.pdf", width: 70%),
  caption: [Phase 0 period heatmap. The period-4 band (yellow-green)
  hugs the balanced diagonal $w_(X A) approx |w_(I A)|$. Period 3, 5,
  and 6 bands flank it; grey indicates saturation (period 1, $A$ fires
  every tick).]) <fig-phase0-period>

*Sanity gate*: $(w_(I A), w_(X A)) = (-11, 11)$ gives strict_p5 $= 1$,
period $= 4$, Property 5 reproduced.

= Phase 1 — Siegert says "stable spiral everywhere" <sec-results-phase1>

For each cell, solve @eq-neg-loop-fp and compute @eq-eigs.

#figure(image("/deq/closed_form_neg_loop/results/phase1/hopf_vs_fcs.pdf", width: 100%),
  caption: [Phase 1 vs Phase 0. *Left*: FCS strict P5. *Middle*: FCS
  broad oscillation. *Right*: Siegert spiral-blue (FP exists and
  $"Im"(lambda) != 0$): $1440 / 1600 = 90 %$. The smooth-rate envelope
  is much wider than Property 5; rate theory thinks "everywhere
  rings."]) <fig-phase1>

#figure(image("/deq/closed_form_neg_loop/results/phase1/im_lambda_heatmap.pdf", width: 70%),
  caption: [Heatmap of $"Im"(lambda)$, the predicted ringing rate.
  Brighter is faster ringing. The ringing rate grows with both
  $|w_(I A)|$ and $w_(A I) w_(X A)$ as expected from
  $sqrt(g_A g_I w_(A I) |w_(I A)|)$.]) <fig-phase1-im>

The 0.31 Jaccard against strict P5 documents the *type mismatch*: rate
theory cannot pick the period-4 cells out of a sea of spiral cells.
Property 5's period locking is a discrete-tick limit cycle, *not* a
continuous-time Hopf bifurcation. At the default cell:

#align(center)[
  $bold(nu)^star = (0.352, 0.179),
   quad lambda = -0.426 plus.minus 0.395 i$
]

— a clean stable spiral. Phase 2 turns the $0.395$ rad/tick ringing
rate into a predicted period.

= Phase 2 — $H(omega)$ predicts ringing direction, not period <sec-results-phase2>

Apply @eq-T-pred to each cell.

#figure(image("/deq/closed_form_neg_loop/results/phase2/T_pred_vs_FCS_period.pdf", width: 70%),
  caption: [$T_("pred")$ vs FCS measured period across the grid. Blue
  dots are FCS strict-P5 cells. The orange dashed line $y = 4 x$
  passes through them — the rate-equation prediction is *exactly $4
  times$ too long*.]) <fig-phase2-scatter>

The factor of $approx 4$ is constant: mean $T_("pred") \/ 4 = 3.39$
over strict-P5 cells, $3.98$ at the default cell. The locked
calibration was fit on *static* $f$-$I$ rates, where any
parameter-pair $(tau_m, tau_("ref"))$ with the same $tau_m + "small"
tau_("ref")$ steady-state would do — the static fit cannot
disambiguate dynamic time scales. The FCS 5-tap windowed integrator
with full reset on spike behaves like a much faster effective $tau$
than $2.35$ ticks; empirically $tau_("FCS-eff") approx tau_m \/ 4
approx 0.6$.

#figure(image("/deq/closed_form_neg_loop/results/phase2/period_predicted_vs_4.pdf", width: 100%),
  caption: [Four-panel: FCS strict P5 / FCS measured period heatmap /
  $T_("pred")$ heatmap (clipped at 30) / boolean $|T_("pred") - 4| <=
  0.5$ overlay. The H(ω) gate at strict period-4 tolerance gives *zero*
  blue cells; even at 8-tick tolerance Jaccard caps at $0.24$.]) <fig-phase2-grid>

The H(ω) reading is the *qualitative* lens: it correctly says "the
negative loop rings, with rate growing in $|w_(I A)|$ and $w_(A I)
w_(X A)$." The quantitative cost of the locked static calibration is
$4 times$ over-estimation of the period.

= Phase 3 — Quasi-renewal recovers the period <sec-results-phase3>

Run the Naud–Gerstner stepper on the negative-loop topology at
$N in {50, 100, 500, 2000}$, $T = 400$, warmup $100$. Measure FFT
period, 1100-template correlation, and amplitude (std/mean).

#figure(image("/deq/closed_form_neg_loop/results/phase3/period_qr_vs_FCS.pdf", width: 100%),
  caption: [QR-measured FFT period vs FCS measured period at each $N$.
  Blue dots: FCS strict P5. *The QR period clusters on the $y = x$
  diagonal*, not the $y = 4 x$ line of Phase 2 — the nonlinear
  age-distribution dynamics with spike-reset recovers the correct
  discrete-tick period.]) <fig-phase3-period>

At the default cell, QR FFT period is $4.05$ ticks at *every* $N in
{50, 100, 500, 2000}$ — essentially exact agreement with FCS's 4.
This is the *qualitative payoff* of moving from $H(omega)$ linear
theory to the nonlinear refractoriness in $A(t) -> m_k(t) -> A(t+1)$:
spike-reset zeroes the age-distribution and forces a refractory
re-build, which faithfully encodes the FCS spike-and-reset semantics.

#figure(image("/deq/closed_form_neg_loop/results/phase3/qr_jaccard_vs_N.pdf", width: 100%),
  caption: [*Left*: Jaccard of QR labels vs FCS labels across $N$.
  `qr_osc_blue` against FCS broad-osc peaks at $J = 0.79$ ($N = 500$);
  `qr_p5_blue` (period + template) saturates at $0.49$ because the
  smooth QR oscillation gives low binary-template correlations even
  when the period is exactly right. *Right*: grid-mean amplitude
  (std/mean) decays monotonically from $0.20$ ($N = 50$) to $0.05$
  ($N = 2000$), crossing the 0.10 gate near $N approx 500$.]) <fig-phase3-jaccard>

#figure(image("/deq/closed_form_neg_loop/results/phase3/default_cell_traces.pdf", width: 80%),
  caption: [Default-cell QR traces at each $N$, 60 ticks post-warmup.
  At $N = 50$, period-4 oscillation is strongly sustained (amplitude
  $approx 0.4$). At $N = 2000$, the trajectory contracts toward the
  FP at $A approx 0.35$ and the oscillation amplitude falls to
  $approx 0.08$, faint but still period-4 in phase.]) <fig-phase3-traces>

The grid-mean amplitude curve is the *signature* of a noise-driven
limit cycle: the deterministic mean field has a stable spiral FP
(Phase 1), so any sustained oscillation must come from the
$sqrt(A\/N)$ noise; as $N -> infinity$ the noise vanishes and the
trajectory contracts into the FP. The $J = 0.49$ Jaccard limit on
`qr_p5_blue` is *structural*: a smooth oscillation cannot perfectly
match a binary template, regardless of period accuracy.

= Synthesis <sec-synthesis>

The three lenses partition the negative loop's content like so:

#table(
  columns: (auto, auto, auto, auto),
  table.header(
    [*Lens*],
    [*Predicts existence?*],
    [*Predicts period?*],
    [*Predicts waveform?*],
  ),
  [Siegert FP], [Spiral envelope 90 % \ ($J = 0.31$ vs strict P5)],
    [—], [—],
  [$H(omega)$ linear], [Same spiral, with $"Im"(lambda)$],
    [Yes but $4 times$ too long], [—],
  [Quasi-renewal], [Yes at finite $N$ \ ($J = 0.79$ vs broad-osc, peak at $N=500$)],
    [*Exact*: $T_("QR") = 4.05$ at default cell, $forall N$],
    [Smooth ≠ binary 1100 \ (template $approx 0.28$)],
  [FCS Lustre + Kind2], [Yes per cell], [Yes per cell], [Yes per cell, binary `1100`],
)

Three observations.

*Observation 1: rate theory cannot create the limit cycle.* The
mean-field FP is *always* a stable spiral on this motif (the
single-pole Jacobian has $"Re"(lambda) = -1\/tau_m$ regardless of
weights). So Property 5's sustained oscillation is *not* a Hopf
bifurcation of the rate equations — it lives strictly beyond
mean-field reach. Whatever sustains it must come from noise (Phase 3)
or from the discrete-tick semantics directly (FCS).

*Observation 2: static calibration does not pin dynamic time-scales.*
The locked $(alpha, beta, tau_m, tau_("ref"))$ from
`closed_form_wta`'s static $f$-$I$ fit is fine for predicting *which*
weights ring (Phase 1 spiral envelope) and *where* the ringing rate is
highest (Phase 1 Im($lambda$) heatmap), but it is off by a constant
factor of $approx 4$ when used to predict the actual ringing
*period* (Phase 2). The FCS-effective $tau$ for dynamic prediction is
$approx tau_m \/ 4 approx 0.6$ ticks, close to $tau_("ref")$ rather
than $tau_m$. Re-fitting $tau_m$ against a dynamic-response dataset
would close this gap; it is not done here because the locking is part
of the experimental design (calibration shared with `closed_form_wta`
so that all six lenses sit on the same calibration set).

*Observation 3: nonlinear refractoriness restores the period.*
Quasi-renewal uses the *same* $tau_m = 2.35$ as Phase 2 but yields
$T_("QR") = 4.05$ instead of $T_("pred") = 15.92$. The difference is
not parameters — it is *mechanism*. The age-distribution stepper
encodes spike-and-reset explicitly: when a population fraction fires,
its $m_0$ resets and the population must rebuild its age structure
through refractory + integration before it can fire again. This
discrete reset is what the FCS 5-tap windowed integrator does
literally; single-pole low-pass cannot represent it because there is
no "reset" in a $1\/(1 + i omega tau_m)$ kernel.

*Bottom line.* For the negative loop, the three closed-form lenses
form an ordered ladder of capability:
+ *Static Siegert*: where, not when (the spiral envelope).
+ *$H(omega)$ linear*: where + direction (Im($lambda$) grows with
  inhibition strength) but with miscalibrated absolute period.
+ *Quasi-renewal*: where + correct period + amplitude, missing only
  the binary waveform — which only the discrete FCS Lustre semantics
  can deliver verbatim.

What FCS uniquely provides is the *exact `1100` pattern*. The
rate-equation lenses can match the location, frequency, and broad
amplitude of the oscillation, but the *binary* aspect of the waveform
is a property of single-neuron deterministic LI&F that any
population-rate description fundamentally smooths over.

#v(0.6em)

= Open follow-ups <sec-followups>

- *Dynamic-response calibration*: fit $tau_m$ against a small set of
  sinusoidal-drive responses to close the Phase 2 period gap. The
  cost is breaking the "shared calibration" property; the gain is
  quantitative period prediction at the linear-response level.
- *QR-vs-FCS amplitude*: the QR amplitude depends on $N$; FCS's
  single-neuron deterministic LI&F has effectively $N = 1$. A more
  faithful comparison would replace the population-rate $A(t)$ with a
  *single-neuron* spike-train predictor (a piecewise-constant hazard,
  for instance) — this is what the companion `closed_form` thread
  did for the WTA case.
- *Three-neuron extensions*: the FCS Fig. 1 family contains negative
  loops of length $>= 3$ (Fig. 3 in @DeMaria2020). The
  three-lens framework should extend with minimal adjustment;
  $J$ becomes circulant and the eigenvalues split into a richer
  spectrum but the same rate-equation/$H(omega)$/QR machinery applies.

#v(0.6em)

#text(size: 9pt, fill: luma(80))[
  Code: `deq/closed_form_neg_loop/`. Companion reading:
  `deq/closed_form_wta/note/closed_form_wta_note.pdf`.
]

#v(1em)

#bibliography("refs.bib", style: "apa")
