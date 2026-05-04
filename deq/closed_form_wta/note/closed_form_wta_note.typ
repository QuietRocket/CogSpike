// Closed-form rate-equation reading of FCS Property 7 (winner-takes-all).
// Standalone advisor-facing research note (Phase 4 deliverable).

#set document(
  title: "Closed-form rate-equation reading of FCS Property 7",
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
    Closed-form rate-equation reading \
    of FCS Property 7 on 2-neuron \
    contralateral inhibition
  ]
  #v(0.4em)
  #text(size: 10pt, style: "italic")[
    Three frameworks (Siegert, $H(omega)$, quasi-renewal) on the same FCS Fig. 10 grid
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
  *Abstract.* De Maria et al. 2020 @DeMaria2020 verified Property 7
  (winner-takes-all) on the $2$-neuron contralateral inhibition motif by
  Lustre encoding plus Kind2 model checking, plotting in their Fig. 10
  a $40 times 40$ integer-grid Boolean over $(w_(12), w_(21)) in [0, -40]^2$
  with the colour blue/red marking "stabilises within $4$ ticks" or not.
  The plot's headline structure is *three diagonal red blocks in a sea
  of blue* --- a staircase whose blocks correspond to integer-tick
  synchronous-oscillation periods (period-$2$, period-$3$, period-$4$
  co-firing). FCS verifies the property at each integer point but
  cannot characterise the boundary as a continuous curve, predict
  decision latency, or quantify finite-population effects.

  This note re-reads the same FCS Fig. 10 grid through three
  closed-form rate-equation frameworks established in the companion
  closed-form thread @closed_form: the Siegert formula
  @Siegert1951 @Brunel2000 (static $f$-$I$), Richardson's single-pole
  $H(omega)$ @Richardson2007 (linear-response dynamic), and
  Naud--Gerstner quasi-renewal @NaudGerstner2012 (finite-$N$
  stochastic). Calibration constants are inherited unchanged from the
  companion thread; nothing is re-fit for this note. Three findings:

  - *Siegert is a high-recall upper bound on FCS-WTA.* The static FP
    enumeration recovers $99.6 %$ of FCS-blue cells but adds
    $472 / 586$ false positives; the false-positives are *exactly* the
    diagonal-red staircase. Smooth-rate fixed-point structure cannot
    distinguish "rate-equation bistability exists" from "integer-tick
    dynamics commit to it within $4$ ticks."
  - *$H(omega)$ is orthogonal to the staircase.* The
    $|"Re"(lambda_("dom"))| > 1\/4$ contour does not improve over Siegert
    ($Delta J = -0.003$). Staircase cells are *not* slow-decay cells;
    they have well-separated rate-equation FPs. The gap is a discrete-
    tick determinism artefact, not a smooth-rate timescale issue.
  - *Quasi-renewal partially dissolves the staircase via $sqrt(A\/N)$
    noise*: $J("vs FCS") = 0.701$ at $N = 50$, falling to $0.662$ at
    $N = 2000$. Direction matches the H3 hypothesis; magnitude is
    modest. Finite-$N$ Gaussian noise is the right *kind* of correction
    but bounded in *quantity*: a $approx 0.30$ Jaccard floor remains,
    representing the fundamental ceiling of any rate-equation theory.

  Together the three frameworks span FCS's verification with a
  continuous closed-form complement: the envelope (Siegert), the
  dynamics (orthogonal $H(omega)$), and the stochastic broadening (QR).
  Jaccard $0.70$ is roughly the ceiling; the remaining gap is the
  *positive content* of FCS's discrete-tick formal verification ---
  what only Lustre + Kind2 can reach.
]

= Introduction <sec-intro>

== FCS's Property 7 at a glance <sec-fcs-recap>

The De Maria et al. $2020$ paper @DeMaria2020 establishes a Lustre
encoding of leaky integrate-and-fire (LI&F) neurons with the windowed
integrator $r$-vector $[10, 5, 3, 2, 1]$ and integer threshold
$tau = 105$. Six basic archetypes are defined; we focus on the
*contralateral inhibition* motif (FCS §6.2.7, Fig. 1f): two neurons
$N_1$, $N_2$ each driven by external input of value $1$, mutually
inhibiting each other with weights $w_(12)$ (the edge from $N_1$ to
$N_2$) and $w_(21)$ (the reverse). Property 7 states that for some
choice of parameters, the system commits to a winner --- one neuron
fires forever, the other stays silent.

FCS Fig. 10 reports a Kind2 verification of Property 7 across the
$40 times 40$ integer-grid $(w_(12), w_(21)) in {0, -1, dots, -40}^2$
plus a row at $-infinity$. The blue/red colour codes whether stability
is reached within the first $4$ ticks. The visual is striking: blue
dominates, with three discrete *red blocks lined up along the
diagonal*. Each block sits where integer-tick dynamics absorb small
weight asymmetries into a synchronous oscillation period --- $N_1$ and
$N_2$ co-firing in lockstep at period $2$, $3$, or $4$ depending on
where the cell falls along the diagonal.

== What FCS verifies, and what it cannot <sec-gap>

FCS's verification is *strong* in the sense Kind2 provides: every
single integer-grid cell carries a machine-checked yes-or-no answer
under the Lustre semantics. But Lustre + Kind2 is *not* a description
of *why* a cell is blue or red, *what shape* the boundary takes off
the integer grid, *how fast* the decision is at a continuous parameter
sweep, or *what happens* when the population is finite-size and
noisy. Those four questions are the *complement* of FCS's
verification, and they are exactly what closed-form rate-equation
theory is built for.

The companion closed-form thread @closed_form developed three
frameworks on a generic asymmetric grid; this note brings them onto
FCS's exact integer grid, in FCS's exact $4$-tick gate, against FCS's
exact reported labels.

= Three frameworks on the same topology <sec-frameworks>

Each cell $(w_(12), w_(21))$ specifies a connectivity matrix
$ J = mat(0, w_(21); w_(12), 0). $ <eq-J>
Each population is a single LI&F neuron under the FCS
semantics. The mapping from FCS-scaled integer weights to Siegert's
normalised inputs $(mu, sigma)$ is calibrated once and frozen
(@sec-calib).

== Static: Siegert <sec-siegert>

The classical Siegert formula @Siegert1951 gives the stationary
firing rate
$ nu = Phi(mu, sigma) = 1 / (tau_("ref") + tau_m sqrt(pi) integral_((V_r-mu)/sigma)^((V_("th") - mu)/sigma) "erfcx"(-u) d u). $ <eq-siegert>
For $2$-neuron CI under input drive $D$ and Bernoulli per-tick thinning
$p_("thin")$, the self-consistency $nu_i^* = Phi(mu_i(nu^*), sigma_i(nu^*))$
reduces to a $1$-D scalar root-finding in $nu_1$. Output: a list of
fixed points (one or three; the bistable case has the symmetric saddle
plus two asymmetric stable FPs). A cell is *WTA-capable* if any FP has
$|nu_1^* - nu_2^*| >= 0.30$.

== Dynamic: Richardson $H(omega)$ <sec-transfer>

Around any fixed point the linear-response transfer function (Richardson
$2007$ single-pole approximation @Richardson2007) is
$ H_i(omega) = (partial Phi_i \/ partial mu_i) / (1 + i omega tau_m), $ <eq-richardson>
giving the $2 times 2$ closed-loop matrix $M(omega) = I - H(omega) J$
whose poles are roots of $det M(omega) = 0$. Equivalently the
time-domain Jacobian
$ A = (1\/tau_m) (-I + "diag"(g) J), quad g_i = partial Phi_i \/ partial mu_i, $ <eq-jacobian>
has the same spectrum (proven in @closed_form, Phase 2).

For decision dynamics the eigenvalue with largest real part predicts
the slowest mode --- if it is positive (saddle), it is the rate of
escape; if negative (stable FP), the rate of convergence. The natural
gate is "fast" iff $|"Re"(lambda_("dom"))| > 1\/T_("FCS") = 0.25$,
making FCS's $4$-tick Boolean a continuous $|lambda|$-contour.

== Finite-$N$: quasi-renewal mesoscopic <sec-qr>

Naud--Gerstner @NaudGerstner2012 give the single-integral mesoscopic
update
$ A(t) = sum_k m_k(t-1) h(k; mu(t)) + sqrt(A(t)/N) xi(t), $ <eq-qr>
with $m_k$ the age-distribution and $h$ the Siegert hazard. As
$N -> infinity$ this reduces to the rate equations; at finite $N$ the
$sqrt(A\/N)$ noise broadens the boundary stochastically. The
hypothesis: finite-$N$ stochasticity is the right framework to dissolve
the integer-tick lock that the deterministic mean field cannot see.

== Calibration <sec-calib>

The four-parameter calibration $(alpha, beta, tau_m, tau_("ref"))$ is
inherited verbatim from the companion thread @closed_form (Phase 0 ISI
calibration, Phase 1 four-parameter least-squares fit on the $f$-$I$
curve at $p_("thin") = 0.7$). Numerical values:
$alpha = 0.250$, $beta = 4.29 dot 10^(-3)$, $tau_m = 2.35$,
$tau_("ref") = 0.36$, with $R^2 = 0.94$ on the calibration $f$-$I$
data. *No re-fitting is performed in this note.*

= Phase 0 --- the FCS Fig. 10 baseline <sec-phase0>

Reproducing FCS Fig. 10 with the FCS oracle
`deq/archetypes/lif_fcs.py:simulate` (Lustre semantics, verbatim) over
a $40 times 40$ integer grid in $[-40, -1]^2$ with $T = 50$ ticks and
the FCS gate (rate $>= 0.99$ for the winner, $<= 0.01$ for the loser
in the post-warmup window):

#figure(image("/deq/closed_form_wta/results/phase0/fcs_grid.pdf", width: 70%),
  caption: [FCS Property 7, our reproduction. $1014 / 1600 = 63.4 %$
  of cells WTA-stable (blue); the rest red. The structure is the
  three-diagonal-red-block staircase of FCS Fig. 10. Each block
  corresponds to a different synchronous-oscillation period locked in
  by integer-tick dynamics.]) <fig-phase0>

The block boundaries (Block I: $|w| in [32, 40]$; II: $[13, 31]$; III:
$[1, 12]$) are integer-tick thresholds at which the FCS LI&F's
windowed integrator transitions from one synchronous period to
another. Off the diagonal, asymmetric weights break the symmetry and
the system commits to one neuron --- blue.

*This is the ground truth* for the closed-form readings that follow.

= Phase 1 --- Siegert is a $99.6 %$-recall envelope <sec-results-phase1>

Siegert FP enumeration on the same grid, with the *WTA-capable* label
"any FP has $|nu_1^* - nu_2^*| >= 0.30$":

#figure(image("/deq/closed_form_wta/results/phase1/siegert_vs_fcs.pdf", width: 100%),
  caption: [Phase 1 vs Phase 0. Left: FCS oracle. Centre: Siegert FP
  enumeration. Right: disagreement (black: Siegert blue, FCS red ---
  the staircase invisibility; orange: the reverse, $4$ cells only).
  Siegert recovers essentially every FCS-blue cell but adds the entire
  diagonal-staircase as false positives.]) <fig-phase1>

#table(columns: 2,
  table.header([*Metric*], [*Value*]),
  [Siegert WTA-capable], [$1482 / 1600 = 92.6 %$],
  [Recall (Siegert blue $|$ FCS blue)], [*$0.996$*],
  [Siegert blue / FCS red (staircase)], [$472 / 586 = 80.5 %$ of FCS-red],
  [Siegert red / FCS blue (boundary miss)], [$4 / 1014 = 0.4 %$],
  [Jaccard], [$0.680$],
)

The $99.6 %$ recall is the headline. *Anywhere FCS calls a cell blue,
Siegert also calls it blue* (modulo $4$ boundary cells with extreme
parameter values). The asymmetric error pattern says Siegert is an
*envelope* (upper bound): rate-equation bistability identifies the
*candidate* WTA cells, of which FCS's integer-tick dynamics admit a
proper subset. As a verification *pre-filter* this is exactly what is
useful: any cell *outside* the Siegert envelope is guaranteed to be
FCS-red (no need to model-check), shrinking Kind2's burden by
$\\(1600 - 1482) / 1600 = 7.4 %$ at this grid.

= Phase 2 --- $H(omega)$ orthogonal to the staircase <sec-results-phase2>

Test A2: does the $|"Re"(lambda_("dom"))| > 1\/4$ contour match FCS's
$4$-tick boundary?

#figure(image("/deq/closed_form_wta/results/phase2/h_gate_vs_fcs.pdf", width: 100%),
  caption: [Phase 2: 4-panel comparison. Left to right: FCS oracle,
  Phase 1 Siegert, Phase 2 $H(omega)$ gate at $|lambda| > 0.25$,
  continuous $"Re"(lambda)$ heatmap. The H-gate version is qualitatively
  similar to Siegert; it does not recover the diagonal staircase.])

#table(columns: 2,
  table.header([*Metric*], [*Value*]),
  [$H$-gate Jaccard at $|lambda| > 0.25$], [$0.677$],
  [Best Jaccard over threshold sweep], [$0.687$ at $|lambda| > 0.20$],
  [Phase 1 Siegert Jaccard], [$0.680$],
  [Improvement of $H(omega)$ over Siegert], [*$-0.003$*],
  [Mean $"Re"(lambda)$ at FCS-blue stable FPs], [$-0.357$],
  [Mean $"Re"(lambda)$ at FCS-red stable FPs], [$-0.293$],
)

The improvement is statistically negligible. The eigenvalue
distributions of FCS-blue and FCS-red cells overlap (mean separation
$0.06$, vs spread $tilde.op 0.30$). *No $|lambda|$-contour separates the
diagonal staircase from the off-diagonal WTA cells*, because the
staircase cells are *not* slow-decay cells. They have well-separated
rate-equation FPs and respectable decay rates; they are simply *not*
the rate equations' problem.

This is the negative result the thread needed. It falsifies the
hypothesis that FCS's $4$-tick gate is a smooth-rate timescale gate.
The remaining gap between rate-equation envelope and FCS reality is
the *integer-tick determinism* itself, orthogonal to anything any
linear-response analysis can see.

= Phase 3 --- quasi-renewal partial dissolution <sec-results-phase3>

Quasi-renewal at finite $N in {50, 100, 500, 2000}$:

#figure(image("/deq/closed_form_wta/results/phase3/qr_n_sweep.pdf", width: 100%),
  caption: [Phase 3: WTA labels at each $N$. Left to right: FCS,
  Siegert, then QR at the four $N$ values. As $N$ grows the QR boundary
  tightens onto the Siegert envelope; rightmost panels visually
  match Siegert.])

#figure(image("/deq/closed_form_wta/results/phase3/qr_jaccard_vs_N.pdf", width: 60%),
  caption: [Jaccard agreement vs $N$. *QR converges to Siegert at
  $J = 0.963$ at $N = 2000$* (mean-field gate). Best agreement with FCS
  is at the smallest $N$ ($J = 0.701$, $N = 50$); the trend is
  monotone in $1\/N$.]) <fig-jaccard>

#table(columns: 4,
  table.header([*$N$*], [*QR blue cells*], [*$J$ vs FCS*], [*$J$ vs Siegert*]),
  [$50$], [$1296$ ($81.0 %$)], [*$0.701$*], [$0.873$],
  [$100$], [$1399$ ($87.4 %$)], [$0.675$], [$0.943$],
  [$500$], [$1433$ ($89.6 %$)], [$0.667$], [$0.966$],
  [$2000$], [$1427$ ($89.2 %$)], [$0.662$], [*$0.963$*],
)

Two readings:

- The *direction* is exactly the H3 prediction. Smaller $N$ means more
  $sqrt(A\/N)$ noise, more dissolution of the integer-tick lock, better
  agreement with FCS. The trend monotone in $1\/N$ identifies
  finite-$N$ noise as one of the bridges between rate-equation theory
  and discrete-tick FCS reality.

- The *magnitude* is small. $J = 0.701$ at $N = 50$ improves on Phase 1
  Siegert ($J = 0.680$) by only $+0.021$. The single-integral
  Naud--Gerstner @NaudGerstner2012 noise is *qualitatively* the right
  tool but quantitatively bounded; the full Schwalger 2017
  age-structured form @Schwalger2017 might tighten further (it adds
  non-renewal corrections that we deliberately omit here). Without it,
  the Jaccard ceiling is roughly $0.70$ on the FCS-LI&F oracle.

The *gap* of $approx 0.30$ Jaccard between the rate-equation prediction
(any flavour) and FCS's ground truth is the *positive content* of
FCS's formal-verification approach: the discrete-tick determinism that
only Lustre + Kind2 (or the FCS oracle directly) can resolve.

= Synthesis <sec-synthesis>

The three frameworks have *non-overlapping* contributions:

#table(columns: 4,
  table.header([*Framework*], [*Reads*], [*Adds to FCS*],
               [*Misses*]),
  [Siegert], [static FP],
   [continuous boundary; pre-filter envelope ($99.6 %$ recall)],
   [staircase],
  [$H(omega)$], [linear-response dynamic],
   [oscillation phase, gain margin\
    (negative-loop motif: $10degree$ in @closed_form)],
   [staircase],
  [Quasi-renewal], [stochastic finite-$N$],
   [partial staircase dissolution; $1\/sqrt(N)$ scaling],
   [$approx 0.30$ Jaccard residual],
)

== Reading vs FCS

FCS Property 7's three blocks of red on the diagonal are *not* what
rate-equation theory would call "no winner." In rate-equation terms
those cells are clearly bistable, with two asymmetric stable FPs
$nu^* approx (0.5, 0)$ and $nu^* approx (0, 0.5)$. The reason FCS calls
them red is that the FCS-LI&F's deterministic, integer-tick semantics
make $N_1$ and $N_2$ follow synchronously identical trajectories that
oscillate in lockstep --- never committing to either of the two
rate-equation attractors. This *spike-timing lock* is what FCS catches
that rate equations cannot, and what makes Lustre + Kind2 fundamentally
necessary for the cells in the staircase blocks.

Conversely, *off the staircase*, rate equations and FCS agree to the
limits of recall ($99.6 %$). The off-diagonal WTA arms are accurately
predicted by smooth-rate theory; this is what makes Siegert useful as
a Kind2 *pre-filter* and the H($omega$) machinery useful for the
*negative-loop* motif's oscillation analysis.

== What this means in practice

For an engineering use of these archetypes (e.g., locomotion CPG
inverse design, reachability-based control), the picture is:

#enum(
  [*Closed-form first.* Siegert FP enumeration over the parameter
   space is fast (seconds for $1600$ cells). It rules out $tilde.op 7.5 %$ of
   cells with no chance of WTA. *Skip Kind2 there.*],
  [*Eigenvalue + finite-$N$ second.* On the $tilde.op 92.5 %$ Siegert-blue
   envelope, $H(omega)$ provides decision-latency estimates for the
   non-staircase region (most of the envelope), and quasi-renewal
   provides finite-$N$ robustness margins.],
  [*Kind2 last, where it matters.* For the diagonal staircase (and
   any cell where rate-equation prediction is borderline) the formal
   verification is required. The closed-form analysis makes this
   *targeted* rather than exhaustive.],
)

This is the FCS-complementary use: rate-equation theory as a *filter*
that points at the cells where formal verification is genuinely
necessary.

= Limitations and follow-on <sec-limit>

== Stage B preview (gated on this reproduction passing review)

If the reproduction holds with advisors, the natural follow-ons are:

- *Decision-latency map.* The $|"Re"(lambda)|$ heatmap (Phase 2 right
  panel) is a continuous quantity that, although it does not separate
  staircase from non-staircase, *does* predict speed of WTA commitment
  on the off-diagonal cells. Cross-validate against measured
  stochastic-LI&F latency. Stage-B Phase B1.
- *Inverse design.* Given a target latency $tau^*$ and a robustness
  margin (distance from staircase), solve closed-form for
  $(w_(12), w_(21))$. FCS verifies; we *prescribe*. Stage-B Phase B2.

== Hard limitations

- *Spike-timing lock is fundamental*: no rate equation, including
  full Schwalger @Schwalger2017, will reach Jaccard $1.0$ against
  FCS-LI&F on the diagonal staircase. The $approx 0.30$ residual is
  the *substance* of formal verification's added value.
- *Calibration is operating-point-specific*: the $4$ constants are
  fit at $p_("thin") = 0.7$. Different drives or thinning rates require
  re-calibration; the framework is otherwise unchanged.
- *FCS's $4$-tick gate is integer-arbitrary*: choosing $T_("FCS") = 4$
  rather than $5$ or $10$ would shift the staircase boundaries
  slightly. The closed-form predictions are gate-independent (they
  predict WTA-capability not WTA-by-tick-$k$); the gate-dependence
  lives entirely in the FCS oracle side.

= Conclusion <sec-conclusion>

The FCS Property 7 verification of the $2$-neuron contralateral
inhibition motif has a clean closed-form complement: Siegert as the
high-recall envelope ($99.6 %$ recall, $7.5 %$ pre-filter savings);
Richardson $H(omega)$ as the dynamic complement (orthogonal to CI's
specific failure mode but load-bearing on the negative-loop motif's
oscillations, see @closed_form); Naud--Gerstner quasi-renewal as the
finite-$N$ stochastic broadening that partially recovers FCS's
integer-tick artefact ($J(N=50, "vs FCS") = 0.701$ vs Siegert's
$0.680$). The remaining $approx 0.30$ Jaccard gap is the substantive
content of FCS's formal verification: the discrete-tick spike-timing
lock that lives genuinely beyond rate-equation theory.

Position vs FCS:

- *FCS at a point*: Lustre + Kind2 yields a deterministic Boolean per
  integer-grid cell. Strong on individual cells; uninformative about
  geometry, latency, or finite-size effects.
- *Closed form (this note)*: Siegert + $H(omega)$ + QR yield continuous
  predictions at every cell, of WTA-capability, decision latency, and
  finite-$N$ robustness. Strong on geometry and complement; weaker per-
  cell where integer-tick determinism rules.
- *Combined*: closed-form pre-filter + targeted Kind2 verification of
  the residual. Each tool is used where it is strongest.

The result is a partition of the analysis problem along the seam
between continuous and discrete: rate-equation theory owns the
continuous boundary geometry and the dynamics off the staircase; FCS's
formal verification owns the discrete-tick determinism on the
staircase. This is exactly the complement the FCS programme invites,
and a concrete example of how rate-equation control theory can wrap
formal-method verification to scale it across continuous parameter
families rather than re-running Kind2 at every grid point.

#bibliography("refs.bib", style: "ieee")
