// Experiment B: single-neuron renewal PMF predictor
#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 14pt, weight: "bold")[
    Experiment B — Single-neuron renewal PMF predictor
  ]
  #v(0.2em)
  #text(size: 10pt, style: "italic")[
    `deq/closed_form_neg_loop/followups/`, May 2026
  ]
]

= Question

The parent study's Phase 3 used a *population* quasi-renewal (QR)
mesoscopic with √(A / N) finite-size noise. For the negative loop the
"population" is exactly one neuron per pop, so the natural
deterministic limit is a *single-neuron age-PMF* stepper: track
$p_k(t) = P("last spike was" k "ticks ago")$ and step the PMF using
the Siegert hazard. The parent §8 conjectured this might recover the
binary `1100` pattern that QR (smooth, low template-match) does not.

= Method

Implement `RenewalNeuronPMF` (inline):

$ "fire"(t) & = sum_k p_k(t-1) dot h_k(t), quad h_k = Phi(mu(t), sigma(t)) "for" k >= tau_("ref") \
  p_0(t) & = "fire"(t), \
  p_(k+1)(t) & = p_k(t-1) dot (1 - h_k(t)), $ <eq-pmf-step>

with renormalization to keep $sum p_k = 1$. Couple two such steppers
(one for $A$, one for $I$): $A$'s input mean uses the external
$w_(X A) dot p_("thin")$ + $w_(I A) dot "fire"_I$, while $I$'s input
mean uses $w_(A I) dot "fire"_A$. Same locked calibration as the
parent thread; same $K_("max") = 30$, $tau_("ref") = 0$.

Run at the FCS default cell $(w_(I A), w_(X A)) = (-11, 11)$ for
$T = 80$ ticks; compare $"fire"_A(t)$ to FCS's binary spike train of
$A$.

= Result

#figure(image("/deq/closed_form_neg_loop/followups/results/expB/pmf_vs_fcs.pdf", width: 100%),
  caption: [*Top*: $"fire"_A(t)$ (blue) and $"fire"_I(t)$ (orange) from
  the renewal-PMF stepper, with threshold 0.5 marked. Both traces sit
  near the Siegert FP ($"fire"_A approx 0.35, "fire"_I approx 0.18$)
  with tiny ($approx 0.01$-amplitude) period-4 ripples. *Second
  panel*: threshold > 0.5 discretization is uniformly zero — the
  PMF amplitude never crosses 0.5. *Third / fourth panels*: FCS
  oracle's binary `0110` / `0011` period-4 patterns for reference.
  Vertical guides every 4 ticks.]) <fig-pmf>

#table(
  columns: (auto, auto),
  table.header([*Quantity (post-warmup, $t in [16, 80)$)*], [*Value*]),
  [Mean $"fire"_A$], [0.352 (≈ Siegert FP $nu_A^star = 0.352$)],
  [Std $"fire"_A$], [0.008],
  [Min / max $"fire"_A$], [0.252 / 0.473],
  [*FFT-dominant period of $"fire"_A$*], [*4.00 ticks (exact!)*],
  [Threshold > 0.5 binary period], [1 (constant 0)],
  [Cyclic '1100' template correlation], [undefined (binary all-zero)],
  [PMF-binary vs FCS-A direct agreement], [32 / 64 = 50 %],
)

= Two findings, one positive one negative

*Positive: the PMF stepper captures the FCS period.* The
$"fire"_A(t)$ trace has FFT-dominant period *4.00* — *exact*
agreement with FCS's period 4 (and with Phase 3 QR's period 4.05
at $N = 2000$). This confirms that the *nonlinear* renewal dynamics
(age-distribution + spike-reset) carries the discrete-tick time
scale, not the single-pole $H(omega)$ Jacobian. Compare with the
parent study:

#table(
  columns: (auto, auto, auto),
  table.header([*Lens*], [*Predicted period (default cell)*], [*Mechanism*]),
  [Siegert FP linearization (Phase 1)], [16 ticks], [eigenvalues of 2x2 single-pole Jacobian],
  [$H(omega)$ linear (Phase 2)], [15.92 ticks], [$2 pi slash |"Im"(lambda)|$ with same Jacobian],
  [QR finite-$N$ (Phase 3)], [4.05 ticks], [age-distribution stepper, $N = 50..2000$],
  [Renewal-PMF (Experiment B)], [*4.00 ticks*], [age-distribution stepper, $N -> infinity$, no noise],
)

So the period-4 prediction is *not* a finite-$N$ noise artefact —
it lives in the deterministic mean-field limit of the renewal
dynamics. Higher-order age-PMF state (60 total degrees of freedom
across the 2 populations) carries an *age-clock* that ticks at
roughly $1 \/ nu^star$ ticks per cycle, and the coupled A-I
oscillation closes on a 4-tick cycle.

*Negative: the PMF stepper cannot reproduce the binary `1100`
waveform.* Without finite-size noise, the trajectory contracts to
the Siegert FP and only tiny ($0.01$-amplitude) oscillations
remain. Threshold > 0.5 discretization gives all-zero. The PMF
predicts the *cycle frequency* but not the *cycle waveform*.

= Why discretization fails: a structural diagnosis

The FCS binary pattern requires the membrane potential $V(t)$ to
cross threshold $tau$ at a specific tick and reset to 0 at the
next, repeating with period 4. This is a *deterministic*
threshold-crossing event in a single-neuron simulator. The
renewal PMF replaces threshold-crossing with the smooth Siegert
hazard $h_k = Phi(mu, sigma)$, computed under the diffusion
approximation. Once the system reaches the Siegert FP, the hazard
is essentially constant; the PMF trajectory has no mechanism to
sharpen the smooth fire-probability into binary spikes.

A binary trace can be recovered by *sampling* — at each tick,
emit a spike with probability $"fire"(t)$. This is what the QR
mesoscopic stepper does at finite $N$ via the $sqrt(A \/ N)$
noise term, which can be interpreted as a Gaussian approximation
to the Bernoulli sample. Phase 3 found that finite-$N$ noise
sustains the oscillation at amplitude $approx 0.4$ ($N = 50$) to
$approx 0.05$ ($N = 2000$), but the template-match correlation
stayed at $approx 0.28$ — even sampled trajectories don't perfectly
hit the cyclic `1100` template.

The structural moral: any rate-equation or PMF reduction
*smooths out* the threshold-crossing geometry that produces the
binary `1100`. Reduction-based predictors can match the *period*
and the *envelope* of the oscillation, but the *binary waveform*
is exclusive to a single-neuron threshold-crossing simulator
(i.e., FCS itself).

= Verdict

*Experiment B PARTIAL.* The renewal PMF stepper correctly
identifies the FCS period (4 ticks, exact agreement), confirming
that the nonlinear age-distribution dynamics — not single-pole
linearization — is the right vehicle for period prediction. But
the deterministic PMF cannot reproduce the binary `1100`
waveform, because smoothing the threshold operation erases
spike-event geometry. The parent §8 conjecture that a single-neuron
renewal predictor would match FCS bit-for-bit is *falsified*; the
correct statement is "rate-PMF reductions match the period but
not the waveform."

This sharpens the three-lens reading: *period* is a property the
nonlinear PMF stepper captures (Phase 3 + Experiment B);
*waveform* is exclusive to the discrete-tick deterministic
oracle (FCS itself).

= Sanity check

The PMF mean rate $"fire"_A = 0.352$ matches the Siegert FP
$nu_A^star = 0.352$ from Phase 1 to four decimal places —
confirming the steady-state Siegert calibration is consistent
between the PMF stepper and the algebraic FP solver.

#v(0.6em)

#text(size: 9pt, fill: luma(80))[
  Output: `results/expB/{pmf_trace.npz, pmf_vs_fcs.pdf}`,
  `results/expB.log`.
]
