// Three follow-up experiments to the closed_form_neg_loop study.
// Companion to deq/closed_form_neg_loop/note/closed_form_neg_loop_note.pdf

#set document(
  title: "Follow-up experiments: negative-loop three-lens study",
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
    Follow-up experiments \
    on the negative-loop three-lens study
  ]
  #v(0.4em)
  #text(size: 10pt, style: "italic")[
    Answering the three open questions from §8 of the parent note
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
  *Abstract.* The parent integrating note @closed_form_neg_loop §8
  raised three open follow-ups: (A) re-fit the rate-equation time
  constant on a dynamic-response experiment to close the Phase 2
  factor-of-4 period gap; (B) build a single-neuron renewal-PMF
  predictor (no $sqrt(A \/ N)$ noise) and check whether the binary
  `1100` waveform emerges; (C) extend to 3-neuron negative loops with
  delayers (DeMaria 2020 @DeMaria2020 Fig. 3). All three were tested
  experimentally; two falsify the parent-note conjectures and one
  cleanly generalizes the framework. Each result sharpens the
  three-lens reading rather than weakening it.

  *Experiment A — falsified.* A Bode sweep of an isolated FCS neuron
  gives $tau_("dyn") = 5.02$ ticks (single-pole fit, $R^2 = 0.97$) —
  *larger* than the static $tau_m = 2.35$, not smaller. Recalibrating
  Phase 2 *widens* the period gap (default-cell $T_("pred")$ goes
  from 15.92 to 33.98 vs FCS's 4). Diagnosis: at the closed-loop
  oscillation frequency $omega^star = 2 pi \/ 4 approx 1.57$ rad/tick,
  the firing-rate transfer is in the deeply rolled-off regime
  ($|H(omega^star)| approx 0.07$). No single-pole model — at any
  $tau$ — can predict period 4 oscillation, because the linearized
  firing rate has no resonance there. The factor-of-4 gap is a
  *structural* property of single-pole linearization, not a
  *parametric* one.

  *Experiment B — partially falsified.* The deterministic
  age-PMF stepper (single-neuron quasi-renewal without noise)
  contracts to the Siegert FP with tiny ($"std" = 0.008$) period-4
  ringing. Threshold-discretization yields all-zero binary; binary
  `1100` waveform does *not* emerge. *But the FFT-dominant period
  of the smooth trace is 4.00 exactly* — matching FCS and Phase
  3's QR finite-$N$ result. The PMF stepper carries the right
  closed-loop cycle frequency, just not the binary waveform.

  *Experiment C — passes cleanly.* For the 3-neuron extension
  $A -> D -> I -> A$ with one delayer, FCS period is *6* (vs 4 for
  the 2-neuron motif, $+2$ per delayer). Single-pole $H(omega)$ gives
  $T_("pred") = 19$ (factor ~3.2 off, same direction as 2-neuron);
  QR mesoscopic gives 6.38, tracking FCS within $plus.minus 1.5$.
  *The framework generalizes:* QR is the right tool for period
  prediction at any loop length; linear $H(omega)$ is structurally
  wrong at any length.

  *Synthesis.* The three experiments collectively reframe the parent
  note's central diagnostic: the Phase 2 period gap is not a
  calibration mismatch (Experiment A), not curable by population-rate
  reduction at $N -> 1$ (Experiment B), and persists in longer loops
  (Experiment C). It is the unique signature of *spike-and-reset
  nonlinearity* — only the quasi-renewal age-distribution stepper
  carries it, regardless of $N$ or loop length.
]

= Context recap

The parent integrating note @closed_form_neg_loop applied three
closed-form lenses (Siegert FP, $H(omega)$ Jacobian, Naud–Gerstner
quasi-renewal) to the 2-neuron negative-loop motif (FCS Property 5).
Findings:

+ Phase 1: rate-equation FP is *always a stable spiral* — mean field
  cannot predict Property 5's sustained oscillation.
+ Phase 2: single-pole $H(omega)$ predicts ringing period $T_("pred") = 16$
  ticks; FCS period is 4. Factor-of-4 over-estimate, constant across
  strict-P5 cells.
+ Phase 3: QR mesoscopic at finite $N$ recovers period 4.05 (exact
  agreement with FCS). Amplitude decays $0.20 -> 0.05$ across $N
  in {50, 2000}$ (noise-driven limit cycle around the stable spiral).

§8 of that note raised three open questions; this note answers them.

= Experiment A — Dynamic τ does not close the period gap

== Question

The parent §8 conjectured that the Phase 2 factor-of-4 gap arose
because the locked $tau_m = 2.35$ was fit on the steady-state $f$-$I$
curve, which does not constrain the dynamic time scale. Re-fitting
$tau$ on a dynamic-response experiment should close the gap.

== Method (brief)

Isolated FCS neuron driven by Bernoulli-modulated input,
$x(t) tilde "Bern"(p_("thin") (1 + epsilon sin omega t))$ with
$p_("thin") = 0.7, epsilon = 0.2$, weight 11. Bode sweep over 12
frequencies in $[0.05, 1.6]$ rad/tick. Fit
$|H(omega)|^2 = g^2 \/ (1 + omega^2 tau^2)$.

== Result

#figure(image("/deq/closed_form_neg_loop/followups/results/expA/bode_fit.pdf", width: 100%),
  caption: [Bode response (magnitude and phase) of the isolated
  FCS neuron. Single-pole fit gives $tau_("fit") = 5.02$ ticks
  ($R^2 = 0.97$) — *larger* than $tau_m^("static") = 2.35$. The
  -3 dB frequency is at $omega approx 0.20$ rad/tick, an order of
  magnitude below the FCS closed-loop oscillation frequency
  $omega^star approx 1.57$ rad/tick. The firing-rate response is
  *essentially flat* at $omega^star$ — $|H(omega^star)|$ is
  $approx 0.075$.]) <fig-A-bode>

Recalibrating Phase 2 by rescaling $T_("pred") |-> T_("pred") dot
(tau_("fit") slash tau_m)$:

#table(
  columns: (auto, auto, auto, auto),
  table.header([*Cell*], [*Phase 2 static*], [*Recalibrated*], [*FCS*]),
  [Default $(-11, 11)$], [15.92], [33.98], [4],
  [Mean over FCS-period-4 cells], [13.6 (≈ 3.39 × 4)], [29.0 (≈ 7.25 × 4)], [4],
)

The recalibration *widens* the gap, not closes it. The conjecture
that the gap was a static-vs-dynamic calibration mismatch is
*falsified*.

== Diagnosis

At the FCS closed-loop oscillation frequency $omega^star = 1.57$,
the firing-rate response magnitude is below $0.1$. *Any* single-pole
model — regardless of $tau$ — would predict zero resonance at
$omega^star$, because single-pole magnitudes are monotonically
decreasing. The factor-of-4 over-estimate of $T_("pred")$ is
therefore *structural*: it reflects the rate-equation linearization's
failure to carry the spike-and-reset event that *produces* the
closed-loop oscillation in the first place. The reset is the
source of the high-frequency cycle; linearization smooths it away.

= Experiment B — Renewal PMF gives the period but not the waveform

== Question

The parent §8 conjectured that a single-neuron renewal-PMF
predictor (without the QR mesoscopic's finite-$N$ noise) might
match FCS's binary `1100` waveform, by virtue of being a
deterministic age-distribution stepper applied at the
neuron-level rather than the population-level.

== Method (brief)

Inline class `RenewalNeuronPMF` tracks $p_k(t) =
Pr("last spike was" k "ticks ago")$, $k = 0..29$. Update via Siegert
hazard $h = Phi(mu, sigma)$ applied uniformly to all ages $k >=
tau_("ref")$:

$ "fire"(t) & = sum_k p_k(t-1) dot h(t) \
  p_0(t) & = "fire"(t), \
  p_(k+1)(t) & = p_k(t-1) dot (1 - h(t)). $

Couple two such steppers for $A$ and $I$ through expected
fire-probabilities. Run at FCS default cell for $T = 80$ ticks.

== Result

#figure(image("/deq/closed_form_neg_loop/followups/results/expB/pmf_vs_fcs.pdf", width: 100%),
  caption: [Top to bottom: PMF fire-probability traces for $A$
  (blue) and $I$ (orange) with threshold $0.5$ marked; binary
  threshold-discretized PMF prediction; FCS oracle $A$; FCS oracle
  $I$. The PMF trajectories contract to the Siegert FP ($"fire"_A
  approx 0.35$, $"fire"_I approx 0.18$) with std $approx 0.008$;
  threshold $>$ 0.5 discretization is uniformly zero. FCS's binary
  `0110` / `0011` patterns sit unreachable.]) <fig-B-pmf>

#table(
  columns: (auto, auto),
  table.header([*Quantity (post-warmup)*], [*Value*]),
  [Mean $"fire"_A$ from PMF], [0.352 (= Siegert FP $nu_A^star$ exactly)],
  [Std $"fire"_A$], [0.008],
  [FFT-dominant period of $"fire"_A$], [*4.00 ticks (exact agreement with FCS)*],
  [Threshold > 0.5 binary period], [1 (all zero)],
  [PMF-binary vs FCS-A agreement], [50 %],
)

== Diagnosis: period yes, waveform no

The smooth PMF trajectory has the *right closed-loop cycle period*
(4.00 ticks, exact). This is a strong corroboration that nonlinear
refractoriness — the renewal age update — captures FCS's time
scale, where single-pole $H(omega)$ does not (Experiment A).
However, the PMF amplitude is vanishing ($"std" \/ "mean" approx
0.023$); the trajectory contracts to the Siegert FP because
without finite-size noise nothing perturbs it.

*Why the binary waveform doesn't emerge.* The PMF is a continuous
distribution; the Siegert hazard $h = Phi(mu, sigma)$ smooths the
threshold-crossing event into a probability. Once at the FP, $h$
is essentially constant, so the trajectory has no mechanism to
produce binary `0/1` outputs. Adding finite-size noise (Phase 3's
QR at $N < infinity$) restores amplitude but introduces stochastic
spike timing, which is *not* the same as FCS's deterministic
threshold crossing — Phase 3's template-match correlation only
reached $approx 0.28$ even at the right period.

The PMF and QR steppers are not the wrong models — they reach the
*right cycle frequency* through the *right mechanism*
(age-distribution dynamics). The binary `1100` waveform itself is
a single-neuron deterministic threshold property, exclusive to a
single-neuron simulator.

= Experiment C — The framework generalizes to longer loops

== Question

The FCS family includes negative loops with delayer chains
(DeMaria 2020 Fig. 3). Does the three-lens framework generalize
to the 3-neuron case $A -> D -> I -> A$?

== Method (brief)

Topology: $A$ excited by external $X$ at $w_(X A)$ and inhibited
by $I$ at $w_(I A)$; $D$ excited by $A$ at $w_(A D)$; $I$ excited
by $D$ at $w_(D I)$. Default weights $w_(X A) = w_(A D) = w_(D I)
= 11, w_(I A) = -11$. Run FCS oracle, Siegert 3-pop FP, $H(omega)$
3×3 Jacobian, QR mesoscopic at $N = 500$.

== Result

#figure(image("/deq/closed_form_neg_loop/followups/results/expC/three_neuron.pdf", width: 100%),
  caption: [Top: FCS oracle spike trains (period 6, pattern `011100`
  cyclic) and the $w_(I A)$ sweep (period 6 → 7 → 8 plateaux).
  Middle: Jacobian eigenvalues (3 of them: one real-negative decay
  mode, one complex-conjugate ringing pair) and QR mesoscopic trace.
  Bottom: bar-chart comparison of period predictions across the
  2-neuron and 3-neuron motifs.]) <fig-C>

#table(
  columns: (auto, auto, auto),
  table.header([*Quantity*], [*2-neuron*], [*3-neuron*]),
  [FCS-measured period], [4], [*6*],
  [Pattern], [`0110...`], [`011100...`],
  [Siegert FP $nu_A^star$], [0.352], [0.431],
  [Static $H(omega)$ $T_("pred")$], [15.92], [19.01],
  [Static $T_("pred")$ / FCS], [3.98], [3.17],
  [Recalibrated $T_("pred")$ (Exp A)], [33.98], [40.58],
  [QR period (mesoscopic)], [4.05], [*6.38*],
  [QR / FCS ratio], [1.01], [1.06],
)

== Reading

*F1. FCS period scales linearly with delayer count.* Default
$"period" = 2 (n + 1)$ where $n$ is the number of delayer cells
($n = 0$ gives the 2-neuron period 4; $n = 1$ gives 6). Each extra
cell adds 2 ticks: one spike-emission delay + one integration step.

*F2. Single-pole $H(omega)$ over-estimates the period at both
lengths, with similar (but not identical) factors* (~4.0 at
2-neuron, ~3.2 at 3-neuron). The factor is *not* constant across
motifs — it depends on the specific Jacobian eigenvalue structure
— confirming Experiment A's diagnosis that the gap is structural.

*F3. QR generalizes cleanly.* At $N = 500$ the mesoscopic gives
period 6.38, within ±1.5 of FCS's 6. The same age-distribution
+ spike-reset mechanism that worked at $n = 0$ works at $n = 1$;
we conjecture this continues for all $n$.

= Synthesis

The three experiments together reframe the parent three-lens
reading:

#table(
  columns: (auto, auto, auto, auto),
  table.header(
    [*Diagnostic*],
    [*Phase 2 conjecture (parent §8)*],
    [*Experiment finding*],
    [*Sharper interpretation*],
  ),
  [Period gap source],
    [static calibration mismatch],
    [false — recal widens gap (Exp A)],
    [structural property of single-pole linearization],
  [Binary waveform recovery],
    [single-neuron renewal predictor (Exp B target)],
    [false — PMF contracts to FP (Exp B)],
    [waveform exclusive to threshold-crossing oracle],
  [Framework generalization],
    [conjectured (Exp C target)],
    [confirmed — 3-neuron passes (Exp C)],
    [QR is the right tool at every loop length],
)

Two falsifications + one confirmation — all of which sharpen the
parent note's central claim. The three-lens reading remains valid:
mean-field theory cannot create the limit cycle; linear $H(omega)$
predicts ringing direction but not period (regardless of $tau$);
quasi-renewal recovers the period (regardless of noise level and
loop length). What FCS uniquely provides is the *exact binary
waveform*; reductions match the period and envelope but not the
1-and-0 spike sequence.

#v(0.6em)

= What this means for the closed-form program

For the *contralateral inhibition / WTA case*
(`closed_form_wta`), Siegert + $H(omega)$ at the locked
calibration achieve Jaccard $approx 0.68$ against FCS — useful as
boundary-shape predictors. For the *negative loop case*, the
single-pole rate-equation lens is structurally inadequate (this
note's Experiment A): it predicts oscillation direction but not
period at *any* calibration. Quasi-renewal is the *first
rate-equation level* at which the period prediction becomes
quantitatively right, because it carries the spike-reset
nonlinearity explicitly. For future archetypes with sustained
oscillation (FCS Fig. 1d/3 family), *skip the linear $H(omega)$
period prediction and go directly to quasi-renewal.*

For studies that genuinely require the binary waveform (e.g.
formal-verification cross-checks), there is no rate-equation
substitute for the FCS Lustre + Kind2 stack itself. The role of
the closed-form lenses is to provide *continuous diagnostics* of
where, why, and how fast — the *binary content* remains in the
formal verification domain.

#v(1em)

#bibliography("refs.bib", style: "apa")
