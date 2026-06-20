#import "../report_style.typ": *
#show: setup

#align(center)[
  #text(size: 9pt, fill: luma(110))[CogSpike · research/compression/nengo_experiments · synthesis]
  #v(0.2em)
  #text(size: 17pt, weight: "bold")[Spikes as Bits, Learned — in Real Spikes]
  #v(0.2em)
  #text(size: 11pt, style: "italic")[
    A 16-experiment Nengo embodiment of the spiking entropy coder: where the paper's
    idealized identities survive in a real spiking substrate, and where they strain
  ]
  #v(0.3em)
  #text(size: 9.5pt)[Final experimental report · 142/142 acceptance checks · all numbers re-runnable via `run_all.sh`]
]
#v(0.6em)

#block(width: 100%, inset: 10pt, fill: luma(247), radius: 4pt)[
  *Abstract.* The paper `unified_spiking_compression.typ` ("Spikes as Bits, Learned")
  proves that a calibrated leaky integrate-and-fire readout fires its first spike at
  latency $t^*(q) = -lambda log_2 q$ — surprisal as a spike *time* — so that a stream's
  total spike-time is the cross-entropy of its predictions, the entropy rate is the
  floor, and a local three-factor delta rule descends the excess to zero. Its existing
  computational backing is *pure `numpy`*: it proves the mathematics is self-consistent
  but runs no spiking simulator. This report *embodies the entire construction in a real
  spiking neural network* (Nengo, reference LIF simulator) across a graded ladder of 16
  experiments, from a single calibrated neuron to a learned end-to-end coder and four
  open-problem capstones. The headline: *the paper's exact identities survive in real
  spikes to the timing grid and machine precision, and every residual is exactly one of
  the honest limits the paper itself flagged — now measured.* The calibration identity
  holds to $approx 1$ timestep; the gradient algebra to $5.8 times 10^(-10)$; the
  Lyapunov constant is $ln 2 = 0.693131$; sum-mode conservation holds to $9 times
  10^(-17)$; decode losslessness is exact and predictor-free. The quantitative optimality
  claims hold with a *small, decomposed* spiking overhead (bits/symbol lands $0.0008$–$0.05$
  above the floor, attributable to $delta t$ rounding, the NEF $1 slash sqrt(N)$ softmax
  error, and the constant-rate PES noise ball). One claim came out *cleaner* than
  hypothesized — the latency code is refractory-immune. And the paper's one genuinely
  *open* premise (that a circuit physically emits the residual $y - q$) is empirically
  *narrowed*: a real two-channel population emits $y - q$ faithfully enough that learning
  still converges, though the circuit-level derivation remains open. The unifying thesis
  — *compressing, predicting, spending less, and learning are one monotone descent on one
  number* — is realized in spikes: in the learned coder, bits/symbol fall from $1.985$ to
  $1.026$ while decode accuracy rises to the Bayes-optimal ceiling, anti-correlated at
  $r = -0.98$.
]

= The thesis of these experiments

The numpy validators (`validate.py`, `learn_validate.py`, `critique_checks.py`) prove the
paper's *mathematics*. They do so by assuming the substrate behaves ideally: one clean LIF
ODE at Euler $delta t = 10^(-5)$ from exactly $V(0) = 0$, a float64 softmax, a literal
$4 times 4$ weight matrix, infinite populations, an infinitely fine clock. Nothing spikes,
nothing is noisy, no population is finite, the clock is perfect, and a "weight" is a number.

This suite replaces *each* idealization with a real spiking mechanism and *measures the
gap*:

#table(
  columns: (1fr, 1fr),
  align: (left, left),
  stroke: 0.5pt + luma(200),
  table.header[*Idealization in `numpy`*][*Reality the Nengo experiment injects*],
  [one noiseless LIF, $V(0)=0$, no refractory], [`nengo.LIF` with $tau_("ref")$, membrane noise, randomized $V(0)$],
  [Euler $delta t = 10^(-5)$], [a $delta t$-quantized spike clock ($10^(-3)$–$10^(-4)$)],
  [softmax in float64 (exact)], [softmax decoded from a finite, heterogeneous population],
  [$W$ is a literal matrix], [PES learns *decoders*, not literal $W_(i j)$],
  [infinite population], [finite $N$: representational + decode + sampling noise],
  [instantaneous WTA], [lateral inhibition with a real settling transient],
  [$c_t = e_i$ exactly], [context is an attractor bump that drifts and takes time to update],
)

Every honesty check the paper flags as prose — the refractory floor, the $q_max$ ceiling,
the finite-$delta t$ penalty, the $q arrow 0$ noise, the predictive-subtraction sign, the
attractor capacity, the conditional emission premise — becomes a measurable spiking
phenomenon here. *A strain that matches a predicted failure mode is a confirmation, not a
defeat.*

= The ladder, at a glance

All experiments share one shared package (`spikecoder/`) whose source and information
measures are byte-compatible with the validators, so every spiking result is asserted
against the *same* closed-form constants ($overline(H)(p) = 0.9782$, $H(pi) = 1.7500$,
$I = 0.7718$ bits/symbol). Each experiment folder carries its own `run.py`, Typst
`report.pdf`, and cached results.

#table(
  columns: (auto, 1fr, auto),
  align: (left, left, center),
  stroke: 0.5pt + luma(190),
  inset: 5pt,
  table.header[*Exp*][*Claim tested → headline spiking result*][*✓*],
  [*e01*], [LIF charging law → first spike obeys $tau_(r c) ln(J slash (J{-}1))$ to $<0.91 delta t$; $tau_(r c)$ recovered to $20.018$ ms. *Refractory-immune* from rest.], [6/6],
  [*e02*], [Calibration $t^*(q) = -lambda log_2 q$ → exact to $<= 1 delta t$ over $q in [0.0375, 0.98]$; drive table matches paper; $q_max = 0.93$; noise CV $0.008 arrow 0.110$ as $q arrow 0$.], [5/5],
  [*e03*], [Softmax is the exact normalizer → NEF RMSE $0.056 arrow 0.0095$ ($1 slash sqrt(N)$); partition of unity becomes representational; divisive norm sums to $<1$.], [8/8],
  [*e04*], [Calibrated readout race → $arg max q$ fires first; stream mean bits/symbol $0.9790 slash 1.7521 slash 1.1142$ (vs $0.9782 slash 1.7500 slash 1.1133$).], [10/10],
  [*e05*], [First-spike-takes-all decode → losslessness $=0$ error for perfect *and* uniform $q$ ("slow, never wrong"); errors concentrate at small latency margin; settled $F G$ winner.], [8/8],
  [*e06*], [Attractor context → ring holds the previous symbol to $<5.7°$ over $300$ ms; capacity $C = 7.93$ bits $>> log_2 4$; write/settle $< $ ISI.], [5/5],
  [*e07*], [Predictive subtraction (the sign) → inhibitory feedback silences predicted symbols, additive sign runs away ($15.6 times$); residual tracks surprisal $r = +0.942$.], [8/8],
  [*e08*], [The *open* premise: cost-spike $=$ gradient → ON/OFF emits $y{-}q$ to RMS $0.0051$; learning robust to realistic emission; $g_("crit")=0.60$. Narrowed, not closed.], [10/10],
  [*e09*], [Delta rule via PES → learned $q arrow P$ ($max|q{-}P| = 0.152$); energy $2.0 arrow 1.030$; gradient identity to $5.8 times 10^(-10)$; ON/OFF variant identical.], [8/8],
  [*e10*], [Excess energy is Lyapunov → constant of descent $= ln 2 = 0.693131$; sum-mode drift $9 times 10^(-17)$; noise ball $prop eta$; partition $7.8 times 10^(-4)$ (spiking) vs $10^(-14)$ (numpy).], [9/9],
  [*e11*], [Full coder (frozen) → mean bits/symbol $0.9790$ on the floor $0.9782$; losslessness $=0$; three baselines reproduced; overhead decomposed ($delta t$ $+1.4$ mbit).], [13/13],
  [*e12*], [Learned coder (*the money plot*) → bits/symbol $1.985 arrow 1.026$, accuracy $0.379 arrow 0.803$ (Bayes $0.8031$), $r = -0.98$; continuous pipeline lossless.], [12/12],
  [*e13*], [Non-stationary tracking → constant-$eta$ re-tracks each moving floor; steady error $prop eta$ (slope $0.91$); decreasing-$eta$ freezes ($13.6 times$ worse).], [8/8],
  [*e14*], [Deeper memory → 2nd-order source $I_2 = 0.782$; first-order learner stuck; eligibility trace recovers $approx 100%$ of $I_2$ (gated on $gamma > 0$).], [11/11],
  [*e15*], [Lossy graded-WTA → spiking rate–distortion curve, rate $0.979 arrow 0.320$ vs distortion $0 arrow 0.197$; trainable Lagrangian; safety → bounded distortion.], [12/12],
  [*e16*], [Neuromorphic event stream → predictive circuit emits only unpredicted events; $8.2 times$ compression; ratio tracks predictability ($r = +0.841$).], [9/9],
)

#align(center)[#text(weight: "bold")[Total: 142 / 142 acceptance checks.]]

= What holds *exactly* in spikes

A surprising amount of the paper is not merely *approximated* in the substrate — it is
reproduced to the timing grid or to machine precision.

#finding[
  *The latency identity is exact to the clock.* $t^*(q) = -lambda log_2 q$ holds to
  $<= 1 delta t$ across the whole representable band (e02). The only residual is that a
  spike lands on the $delta t$ grid — exactly the *timing-resolution penalty* the paper
  names as the continuous-time replacement for the discrete code's integer-bit penalty.
  At $delta t = 10^(-4)$ it is $approx 0.04$ ms.

  *The gradient algebra is exact.* The delta-rule post-factor $(y_j - q_j)$ is the
  negative natural-log-loss gradient to $5.8 times 10^(-10)$ (finite-difference) and
  $0$ algebraically (e09) — Nengo's PES *is* the Widrow–Hoff rule.

  *The Lyapunov constant is exact.* The averaged-flow descent rate is $k = 0.693131 = ln 2$
  (e10), vindicating the paper's `critique_checks` correction over the original $eta$.

  *Sum-mode conservation is exact.* $sum_j Delta W_(i j) = 0$ holds to $9 times 10^(-17)$
  from a nonzero init (e10) — the partition of unity is preserved by the *learning rule
  itself*, not enforced by hand.

  *Losslessness is exact and predictor-free.* First-spike-takes-all decodes $0$ errors
  for *any* predictor — perfect, uniform, or wrong (e05, e11, e12). Correctness is
  structural; only speed is quantitative. "A bad model is slow, never wrong" is literally
  true in spikes.
]

= What holds with a small, *decomposed* overhead

The quantitative optimality claims — the bit rate, the descent to the floor — hold up,
but the spiking substrate adds a small overhead that the experiments *decompose into named
sources* rather than hide.

#table(
  columns: (auto, auto, 1fr),
  align: (left, right, left),
  stroke: 0.5pt + luma(195),
  table.header[*Quantity*][*spiking value*][*overhead source (measured)*],
  [perfect-predictor bits/symbol (e04/e11)], [$0.9790$], [$+0.7$ mbit: $delta t$ rounding $+1.4$, sampling $-0.7$],
  [learned-predictor energy (e09/e12)], [$1.030$], [$+52$ mbit: constant-$eta$ PES *noise ball* (e10)],
  [partition defect $|sum q - 1|$ (e03/e10)], [$8 times 10^(-4)$], [NEF $1 slash sqrt(N)$ decode of a finite population],
  [non-stationary per-segment excess (e13)], [$0.007$–$0.030$], [$approx 10 times$ the float64 twin: NEF + finite-budget descent],
)

#intuition[
  The pattern is uniform: *the direction and structure of every claim is realized on the
  substrate; the tightest quantitative value lives in the float64 twin.* The spiking
  learner converges to a *noise ball* (radius $prop eta$, e10/e13) rather than to the
  exact floor, because a finite population decoded at constant rate cannot reach the
  $0.0005$-bit residual of a literal-matrix decreasing-step SGD. This is not a failure of
  the paper — it is the paper's own constant-$eta$ prediction, made physical, with the
  noise now coming from neurons and sampling rather than minibatches.
]

#figure(
  image("../e02_calibration/results/e02_calibration.pdf", width: 72%),
  caption: [*The foundational identity (e02).* In a real Nengo LIF neuron, the first-spike
    latency equals the surprisal $-lambda log_2 q$ to within one timestep, across the whole
    representable band — the spike's *time* is the symbol's information content.],
)

= One claim came out cleaner than the paper hypothesized

#finding[
  *The latency code is refractory-immune.* The project plan (and a natural reading of the
  paper) expected the LIF refractory period $tau_("ref")$ to impose a constant latency
  *offset* that would realize the predicted minimum-latency floor. It does not (e01): the
  refractory period is *post-spike dead time* and delays only the *second* spike. Because
  the coder resets each readout to rest per symbol window, every readout fires its first
  spike from $V = 0$, so $tau_("ref")$ never enters the latency. The paper's idealized law
  (which has no refractory term) is therefore matched *exactly* by Nengo's first spike,
  with $tau equiv tau_(r c)$. The real minimum-latency floor comes from *drive saturation*
  ($q_max = 0.93$ at a $10 times$ rheobase ceiling) and the $delta t$ clock — not from
  $tau_("ref")$ (e02).
]

= The one genuinely open premise, narrowed

The paper is scrupulous that its central novelty — "the cost spike *is* the teaching
signal" — is *conditional*: the gradient algebra is unconditional, but the *physical*
identity requires the circuit to emit exactly the signed residual $y_j - q_j$, via two
rectified ON/OFF channels. The paper flags this as open.

#finding[
  e08 *empirically narrows* it. A real two-channel spiking ON/OFF population emits $y - q$
  with RMS error $0.0051$ (concentrated $7 times$ at the rectification kink). Feeding the
  *physically emitted, imperfect* residual into the learner, learning still reaches $q = P$
  (excess KL $0.038 < $ the failure threshold $0.104$). The corruption→failure boundary is
  mapped: learning breaks only at ON/OFF gain mismatch below $g_("crit") = 0.60$ or a
  rectification dead-zone above $theta_("r,crit") = 0.150$ — and a standard spiking
  population sits comfortably inside the convergent region. The latency channel and the
  error channel show *zero* cross-talk (distinct observables, as the paper insists).
]

#honest[
  e08 *narrows but does not close* the premise. Its corruptions are parametric models of an
  imperfect population, and its "real emission" point is the steady-state rate decode of a
  *hand-wired* two-ensemble network, not a learned predictive-coding microcircuit derived
  from the predictive-subtraction loop. The contribution is to show that *the fidelity bar
  such a derivation must clear is modest, and a standard spiking population clears it*. The
  circuit-level derivation itself remains the paper's open problem.
]

= The unifying thesis, in one figure

#intuition[
  e12 is the payoff. The learned coder starts from an ignorant uniform predictor and runs
  the local rule online while *its own learned predictions drive the latency clock*. As it
  learns, its spike-time bill falls — measured from *actual first-spike latencies*, not a
  separately computed loss — from $1.985$ bits/symbol (uniform init) past the marginal
  $1.7500$ to $1.026$, while decode accuracy rises from $0.379$ to $0.803$, which is
  *exactly* the Bayes-optimal top-1 ceiling $sum_c pi_c max_j P_(c j) = 0.8031$. Bits and
  accuracy are anti-correlated at $r = -0.98$. *Compressing better, predicting better, and
  learning are one monotone descent on one number — performed by a local rule the spiking
  substrate physically runs.*
]

#figure(
  image("../e12_learned_coder/results/e12_money_plot.pdf", width: 82%),
  caption: [*The money plot (e12).* As the local rule learns online, the learned coder's
    mean per-symbol first-spike time (measured from real spikes) falls from $1.985$ bits
    (uniform init) past the marginal $1.7500$ toward the floor $0.9782$, while decode
    accuracy rises to the Bayes-optimal ceiling $0.8031$. Compression, prediction, and
    learning, one descent on one number, in a spiking circuit.],
)

= The program extends: the four capstones

The capstones confirm the construction is a first rung, not a ceiling. *Non-stationary
tracking* (e13): with a constant step the spiking learner re-tracks each moving entropy-rate
floor as the source drifts (the marginal held fixed), with steady-state error $prop eta$ —
the $O(eta)$ noise ball is an *adaptation mechanism*, and a decreasing step *freezes* and
fails. *Deeper memory* (e14): a second-order source carries $I_2 = 0.782$ bits a first-order
learner cannot touch; a lag-tagged eligibility trace $e <- gamma e + c$ recovers $approx 100%$
of it (and is worthless at $gamma = 0$, so the trace *is* the mechanism), with the FIX-G
attractor capacity (16 vs 4 states) the next bottleneck. *Lossy compression* (e15): relaxing
the hard WTA to a graded bump traces a clean spiking rate–distortion curve (rate $0.979 arrow
0.320$, distortion $0 arrow 0.197$), with a trainable per-context Lagrangian — and the safety
contract honestly downgrades from exact losslessness to bounded distortion. *Event streams*
(e16): on a synthetic DVS-like source whose input *is already spikes*, a predictive inhibitory
circuit emits only the unpredicted events — an $8.2 times$ compression whose ratio tracks the
stream's predictability ($r = +0.841$), the residual concentrating exactly at motion onset and
reversal.

= Reproducibility

Everything is re-runnable. `bash run_all.sh` self-tests the shared package, runs all 16
experiments (each printing PASS/FAIL against its numeric acceptance criteria), and compiles
every report PDF including this synthesis. The environment is a single `uv`-managed project
(Python 3.12, Nengo 4.1.0; pure-`numpy` reference simulator, no GPU). The shared package was
*never modified* by any experiment (verified per commit); each tier was committed behind a
gate. Seeds are fixed (`SEED = 7`), so every number above is deterministic on re-run.

#finding[
  *Bottom line.* "Spikes as Bits, Learned" is not only internally consistent mathematics —
  it *runs on a real spiking substrate*. Its exact identities survive to the clock and to
  machine precision; its quantitative claims survive with a small overhead that decomposes
  cleanly into the very limits the paper flagged; one claim is cleaner than hoped
  (refractory-immunity); and its single open premise is empirically narrowed to a modest,
  achievable fidelity bar. The bit is a latency, the gradient is a residual spike, the loss
  is the energy bill — and a circuit that minimizes its own spiking learns to predict the
  world.
]
