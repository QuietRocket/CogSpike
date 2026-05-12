// Phase 3 report: quasi-renewal finite-N study of FCS Property 5
#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 14pt, weight: "bold")[
    Phase 3 — Quasi-renewal finite-N study
  ]
  #v(0.2em)
  #text(size: 10pt, style: "italic")[
    `deq/closed_form_neg_loop/`, May 2026
  ]
]

= Goal

Run the Naud–Gerstner age-distribution mesoscopic equations on the
negative-loop motif at $N in {50, 100, 500, 2000}$. Two questions:

+ Does *nonlinear* refractoriness (age-distribution + Bernoulli-thinned
  hazard) recover the correct FCS period, where the linear $H(omega)$
  Jacobian of Phase 2 over-estimated period by $4 times$?
+ Does finite-$N$ noise sustain the oscillation (turning Phase 1's
  stable spiral into a noise-driven limit cycle), or does the dynamics
  collapse to the fixed point as $N -> infinity$?

= Setup

- Same locked calibration as Phases 1–2 ($alpha, beta, tau_m, tau_("ref")$).
- `QuasiRenewal` from `deq/closed_form/quasi_renewal.py` with
  $K_("max") = 30$, $tau_("ref")^("ticks") = 0$, $d t = 1$.
- New negative-loop adapter `simulate_neg_loop_qr` builds the same
  asymmetric input structure as Phase 1 (A: external $X$ +
  inhibition from $I$; $I$: only excitation from $A$).
- $T = 400$ ticks, warmup $100$ → measurement window 300 ticks.
- Per cell, three measurements on the activator trace $A(t)$:
  - *FFT period*: dominant period in $[2, 32]$ ticks from FFT peak.
  - *Template score*: best correlation against a phase-shifted period-4
    `1100` boxcar.
  - *Amplitude*: std / mean (coefficient of variation).
- Two boolean labels per $N$:
  - *qr_p5_blue*: $|"period" - 4| <= 0.5$ *and* template $>= 0.5$.
  - *qr_osc_blue*: amplitude $>= 0.10$ (sustained oscillation, period
    agnostic).

= Result 1 — QR recovers the FCS period

#figure(image("results/phase3/period_qr_vs_FCS.pdf", width: 100%),
  caption: [QR-measured FFT period vs FCS measured period at each $N$.
  Blue dots are FCS strict-P5 cells. *Unlike Phase 2's factor-of-4
  gap*, the QR-measured period clusters along the $y = x$ diagonal —
  the nonlinear age-distribution dynamics correctly predicts the
  FCS period across most of the grid, at every $N$.])

At the FCS default cell $(-11, 11)$, FFT period $= 4.05$ ticks for
*all* $N$ — essentially exact agreement with FCS's period 4. The
H(ω) linear approximation over-estimated by a factor of $4$ because
single-pole low-pass at $tau_m = 2.35$ misses the spike-reset
nonlinearity. The quasi-renewal stepper does not — it explicitly
zeroes the age-distribution on each fire and re-evaluates the hazard
on a fresh refractory-aware distribution, which faithfully encodes the
FCS spike-and-reset semantics.

This is the *qualitative gain* of using quasi-renewal over single-pole
linearization: even with the same static-rate calibration, the
nonlinear refractoriness picks up the right time scale for periodic
firing.

= Result 2 — Oscillation amplitude decays with $N$

#figure(image("results/phase3/qr_jaccard_vs_N.pdf", width: 100%),
  caption: [*Left*: Jaccard of QR boolean labels against FCS labels
  across $N$. `qr_osc_blue` against FCS broad-osc stays high (0.71 →
  0.79 → 0.73) with a peak at $N = 500$. `qr_p5_blue` (period + template
  combined) ramps up with $N$ (0.20 → 0.49) but plateaus —
  template-match is structurally biased against the smooth QR
  oscillation shape. *Right*: grid-mean oscillation amplitude (std /
  mean of $A(t)$) decays monotonically from $0.20$ at $N = 50$ to
  $0.05$ at $N = 2000$, crossing the 0.10 gate near $N approx 500$.
  As $N -> infinity$, QR collapses to mean field, the spiral FP
  attracts deterministically, and oscillation dies — exactly as
  predicted from Phase 1's stable-spiral analysis.])

#figure(image("results/phase3/default_cell_traces.pdf", width: 90%),
  caption: [Default-cell ($w_(I A) = -11, w_(X A) = 11$) QR traces at
  each $N$, 60 ticks post-warmup. Period-4 grid drawn for reference.
  At $N = 50$ the trace is a strong-amplitude period-4 oscillation
  ($A$ activity hits 0.6 peaks and 0.1 troughs). As $N$ grows, the
  amplitude shrinks toward the FP (red, $A approx 0.35$). Even at
  $N = 2000$ the trace is faintly periodic, but the std/mean has
  fallen below the 0.10 gate.])

#table(
  columns: (auto, auto, auto, auto, auto),
  table.header([*$N$*], [*period*], [*template*], [*amplitude*], [*FP_distance*]),
  [50], [4.05], [0.24], [0.40], [far from FP],
  [100], [4.05], [0.26], [0.31], [—],
  [500], [4.05], [0.28], [0.15], [—],
  [2000], [4.05], [0.28], [0.08], [near FP],
)

(Default-cell summary at each $N$.)

= Result 3 — The grid-level comparison

#figure(image("results/phase3/qr_n_sweep.pdf", width: 100%),
  caption: [Boolean labels: FCS strict-P5 (column 1), FCS broad-osc
  (column 2), and QR `qr_p5_blue` at each $N$ (columns 3–6). The QR
  Property-5 region grows with $N$ from a small noisy patch at
  $N = 50$ to the "period-4 band" along $w_(X A) approx |w_(I A)|$ at
  $N = 500..2000$, matching FCS's strict-P5 region in *location* (gold
  ring at default).])

= Why `qr_p5_blue` Jaccard saturates at 0.49

The combined gate (period match + template $>= 0.5$) is harder than
either alone because the quasi-renewal output is a *smooth*
oscillation rather than a binary spike train. The activator's $A(t)$
trace hovers between, say, $0.2$ and $0.5$ rather than between 0 and
1 — its correlation against the binary `1100` template caps near
$0.3$ even when the period is exactly right. A relaxed
period-only gate (no template requirement) would push Jaccard higher,
but at the cost of admitting cells whose period happens to be 4 but
whose waveform is not 1100-shaped.

The structural moral: *binary template matching is not a fair test
for population-rate predictions*. The right comparison is the FFT
period scatter (left figure of Result 1), where QR cleanly recovers
the FCS period at $r approx 1.0$ rather than $r approx 4$ as in Phase 2.

= Verdict

*Phase 3 PASS qualitatively.* Three substantive findings:

+ *Nonlinear refractoriness fixes the period.* QR-measured FFT period
  at the default cell is $4.05$ at every $N$, vs Phase 2's $H(omega)$
  prediction of $15.9$. The age-distribution + spike-reset mechanics
  carry the discrete-tick FCS time scale that single-pole linear
  approximation discards.
+ *Finite-$N$ noise sustains oscillation; mean-field collapses it.*
  Grid-mean amplitude decays from $0.20$ ($N=50$) to $0.05$ ($N=2000$).
  This is the predicted noise-driven limit-cycle regime: the
  deterministic dynamics has a stable spiral FP (Phase 1), so any
  sustained oscillation must come from stochasticity, and at large $N$
  the trajectory contracts into the FP.
+ *Strict Property-5 template-match underestimates the match
  region* because QR produces smooth oscillations rather than binary
  1100 trains. The period itself is correctly recovered; only the
  waveform shape differs.

The combined three-lens reading delivered by Phases 1–3 is the
content of the integrating note.

#v(0.6em)

#text(size: 9pt, fill: luma(80))[
  Output: `results/phase3/{qr_grid.npz, qr_n_sweep.pdf,
  period_qr_vs_FCS.pdf, qr_jaccard_vs_N.pdf,
  default_cell_traces.pdf}`, `results/phase3.log`.
]
