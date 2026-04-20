// Phase 3 Report — Pole placement on the negative loop
// CogSpike / LI&F Archetypes — April 2026

#set document(
  title: "Phase 3: Inverse Pole Placement on the Negative Loop",
  author: "CogSpike Research Team",
  date: datetime.today(),
)

#set page(paper: "a4", margin: (x: 2.5cm, y: 2.5cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#let finding(body) = block(
  width: 100%, inset: 8pt,
  fill: rgb("#f0fff0"),
  stroke: (left: 2pt + rgb("#2e8b57")),
  [*Finding.* #body],
)

#let negresult(body) = block(
  width: 100%, inset: 8pt,
  fill: rgb("#fff0f0"),
  stroke: (left: 2pt + rgb("#b22222")),
  [*Negative result.* #body],
)

#let remark(body) = block(
  width: 100%, inset: 8pt,
  stroke: (left: 2pt + orange),
  [*Remark.* #body],
)

#let intuition(body) = block(
  width: 100%, inset: 8pt,
  fill: rgb("#f0f7ff"),
  stroke: (left: 2pt + rgb("#4a90d9")),
  [*Intuition:* #body],
)

#align(center)[
  #text(size: 18pt, weight: "bold")[
    Phase 3 --- Inverse Pole Placement \
    on the Negative Loop
  ]
  #v(0.3em)
  #text(size: 12pt)[CogSpike Research Team --- April 2026]
  #v(0.3em)
  #text(size: 10pt, style: "italic")[
    Sanity check passes after calibration; inverse design fails 5/6.
    Direct simulator enumeration is the right tool for this job.
  ]
]

#v(1em)

*Abstract.* We tested Hypothesis 3 (pole placement on $A_"full"$ yields
weights that realise a target oscillation period in the discrete FCS
simulator). The sanity check at the Phase 0 known-good period-$4$ point
reveals a systematic bias: the linearisation's dominant complex arg
at $(w_"AI"=11, w_"IA"=-11)$ is $1.342$ rad rather than the nominal
$pi/2 = 1.571$. After a single-point calibration factor
$alpha = 1.342 / (pi/2) approx 0.854$, the sanity check recovers the
correct weights for period $4$. Inverse design for other periods ($3, 5, 6, 7, 8$)
fails ($0/5$ match rate even with calibration), and a diagnostic sweep
shows why: the dominant complex arg of $A_"full"$ sits in a narrow
band of $[0.86, 1.33]$ rad across *all* simulator periods $3$ through $8$,
with only a weak negative correlation ($r = -0.51$) between simulator
period and predicted arg. The spike-reset nonlinearity erased by the
linearisation is what actually determines the integer period. Direct
enumeration over integer $(w_"AI", w_"IA")$ pairs finds all target
periods in well under a second --- the right tool for this job is brute
force, not pole placement.

= Setup

We build $A_"full"$ (10-dim, from Phase 1b) at the non-spiking fixed point
of the negative-loop linearisation. Target poles are placed at
$z = e^(plus.minus i dot 2 pi \/ T)$ on the unit circle for each target
period $T$. We numerically search over integer $(w_"AI", w_"IA") in
[1, 30] times [-30, -1]$ to minimise $|arg lambda_"dom"(A_"full") -
alpha dot 2 pi \/ T|$, with a soft bonus for $|lambda| gt.eq 1$ (oscillatory
regime). The simulator is then run with the predicted weights and the
realised period is measured by exact self-matching on the tail of the
activator spike train.

= Sanity Check

#finding[
  At the Phase 0 known-good weights $(w_"XA" = w_"AI" = 11, w_"IA" = -11)$:
  - $A_"full"$ dominant complex arg = $1.342$ rad
    (target $pi/2 = 1.571$, error $0.229$ rad)
  - $|lambda| = 2.21$ → linearisation correctly predicts sustained oscillation
  - Simulator measured period = $4$ ✓ (activator:
    $011001100110011001100110dots$)
  - Naive pole-placement target ($pi/2$) *fails* to recover the known-good
    weights; calibrated target ($1.342$) *does* recover them exactly.
]

This is the "did the period-$4$ weights already satisfy the pole-placement
equation" check the verifier asked for. Strictly: *no, they did not* ---
the raw pole at $pi/2$ is an imagined target that the linearisation never
reaches. What the linearisation actually produces is a pole at
$1.342$ rad, and the corresponding simulator period is indeed $4$. The
right workflow is therefore a single-point calibration, not direct pole
placement.

= Pole-Placement Sweep

Calibration factor $alpha = 0.854$ applied throughout. Target periods
$T in {3, 4, 5, 6, 7, 8}$.

#figure(
  image("results/phase3_spike_trains.png", width: 100%),
  caption: [Activator spike trains at predicted weights for each target
    period. Only $T = 4$ hits (blue trace). All other targets produce
    simulator periods that disagree with the target: $T = 5$ aimed at
    gives a period-$8$ train, $T = 6$ gives $8$, $T = 7$ gives $4$,
    $T = 8$ gives $4$. $T = 3$ produces no clean period.],
)

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  inset: 5pt,
  align: center,
  stroke: 0.5pt,
  [*T*], [*$w_"AI"$*], [*$w_"IA"$*], [*arg*], [*$T"_pred"$*],
  [*$T"_sim"$*], [*match*],
  [3], [4], [-26], [1.653], [3.80], [---], [✗],
  [4], [11], [-11], [1.342], [4.68], [*4*], [*✓*],
  [5], [5], [-18], [1.074], [5.85], [8], [✗],
  [6], [5], [-13], [0.905], [6.94], [8], [✗],
  [7], [12], [-6], [0.784], [8.02], [4], [✗],
  [8], [12], [-6], [0.784], [8.02], [4], [✗],
)

Match rate $1/6$ (only the calibration point itself).

#figure(
  image("results/phase3_pole_diagram.png", width: 75%),
  caption: [Target poles (filled circles) on the unit circle vs realised
    $A_"full"$ dominant poles ($times$) for each target period. The
    realised poles cluster in a narrow angular band around $1$ rad
    regardless of target, with magnitudes varying but always outside
    the unit circle for $|lambda| > 1$ cases.],
)

= Diagnostic: What does arg_pred mean?

To check whether the linearisation has *any* ability to discriminate
integer periods, we inverted the search: for each simulator period $T$
that exists in the grid, took one exemplar cell and computed
$arg lambda_"dom"(A_"full")$ there.

#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 5pt, align: center, stroke: 0.5pt,
  [*$T_"sim"$*], [*$(w_"AI", w_"IA")$*], [*arg_pred*], [*$2 pi/T$*], [*$alpha dot 2 pi/T$*],
  [3], [(+9, -12)], [1.329], [2.094], [1.789],
  [4], [(+7, -12)], [1.270], [1.571], [1.342],
  [5], [(+6, -12)], [1.182], [1.257], [1.074],
  [6], [(+6, -30)], [1.312], [1.047], [0.895],
  [7], [(+5, -12)], [0.860], [0.898], [0.767],
  [8], [(+5, -30)], [1.217], [0.785], [0.671],
)

Correlation$(T_"sim", "arg_pred") = -0.51$.

#negresult[
  The linearisation's predicted arg sits in a narrow band
  $[0.86, 1.33]$ rad across all six simulator periods. Period-$3$ and
  period-$6$ cells both give arg $approx 1.32$; period-$4$ and period-$8$
  cells both give arg $approx 1.22$. The linearisation cannot
  discriminate these periods, which is why the inverse search picks
  weights whose linearisation has the right arg but whose simulator
  realises a different period.
]

= Why Pole Placement Fails Here

#intuition[
  The $A_"full"$ linearisation captures the continuous-time linear dynamics
  around the non-spiking fixed point. It correctly reports "sustained
  oscillation exists at roughly frequency $pi/2$" --- that is, the
  spectrum does tell us an oscillation will happen. But the actual
  integer period of the simulator's spike train is determined by the
  spike-reset rule: every time $V$ crosses threshold, $"mem"[1..4]$ is
  zeroed. This discrete reset takes the trajectory off the linear
  manifold and back onto a new initial condition, from which the linear
  dynamics picks up for another roughly-$pi/2$ cycle. The integer period
  is the combination of linear oscillation rate and reset frequency,
  and the latter is not encoded in the linearisation's spectrum.
]

Concretely: a period-$4$ spike pattern ($1100$) and a period-$8$ pattern
($11110000$) both oscillate with fundamental frequency close to $pi/2$
--- the period-$8$ wave is just "the same oscillation, resets less
often". The linearisation sees them as the same because it erases the
reset. Distinguishing them requires modelling the reset explicitly,
which cannot be done with a single linear matrix.

= What Works Instead

Direct enumeration over the integer weight grid
$(w_"AI", w_"IA") in [1, 30] times [-30, -1]$ takes $approx 1$ second on
a laptop and finds at least one weight pair for every simulator period
in ${3, 4, 5, 6, 7, 8}$. The reachable-period distribution is:

#table(
  columns: (auto, auto, auto),
  inset: 5pt, align: center, stroke: 0.5pt,
  [*period $T$*], [*cells*], [*example $(w_"AI", w_"IA")$*],
  [3], [104], [(9, -12)],
  [4], [200], [(7, -12)],
  [5], [288], [(6, -12)],
  [6], [158], [(6, -30)],
  [7], [12], [(5, -12)],
  [8], [18], [(5, -30)],
  [none], [120], [---],
)

For FCS integer LI\&F specifically, brute-force simulator enumeration
subsumes pole placement both in speed and accuracy --- spectral
cartography loses to the oracle it was meant to replace.

= Closing the Hypothesis

#negresult[
  Hypothesis 3 (inverse pole placement for target oscillation period) is
  *not validated* on the FCS negative loop. The sanity check passes only
  after a single-point calibration to the known-good weights, and the
  calibration does not generalise: other target periods are missed. The
  linearisation correctly identifies *that* oscillation occurs but not
  *at what integer period*.
]

= Outlook

The three hypotheses status is now:

- *H1* (eigenvalue gap predicts WTA): not validated under deterministic
  semantics; inapplicable under either scalar-$r$ or $A_"full"$.
- *H2* ($rho(A) = 1$ contour predicts bifurcation): not validated under
  deterministic semantics; *validated* under Kind2-reachability semantics
  (Phase 1c) at $98.5%$ classification with clean red/blue separation.
- *H3* (pole placement inverse design): not validated.

The final summary figure and writeup will present this as a two-part
result: spectral cartography correctly describes reachability questions
(where Kind2 itself operates) but not bit-exact deterministic dynamics
or inverse period design. The combinatorial baseline $|w_(1 2)| - |w_(2 1)|$
captures the deterministic contralateral structure that no linearisation
reaches.

#v(0.8cm)
#line(length: 100%)
#text(size: 9pt, style: "italic")[
  Generated from `deq/archetypes/phase3_pole_placement.py`. Artifacts in
  `deq/archetypes/results/phase3_*`.
]
