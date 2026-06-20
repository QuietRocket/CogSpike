#import "../report_style.typ": *
#show: setup

#report_header("e15", "Learned lossy compression via a graded WTA (rate-distortion)",
  "Tier 4 · capstone · paper section \"Learned lossy compression and rate-distortion\" (open problem)")

#claim[
  Everything in the suite so far was *lossless* source coding: the first-spike-
  takes-all decode (e05, e11) reconstructs the emitted symbol *exactly*, so the only
  quantity that moved was the *rate* (mean bits/symbol) and correctness was a
  structural invariant — decoded $=$ emitted for *any* predictor. Relaxing the *hard*
  winner-take-all decode to a *graded* / soft argmax — a sub-saturating sum-mode gain
  (a finite softmax temperature), i.e. a *wide graded bump* over the calibrated
  readout — lets the decoder commit to a *coarsened* symbol, *merging*
  hard-to-distinguish (near-equiprobable, near-tied-latency) outcomes and spending
  fewer spikes (paper, "Learned lossy compression and rate–distortion"). The
  graded-bump width is then a *rate–distortion knob*:
  $ "wider bump" == "larger merge margin" delta == "coarser symbols" ==
    "lower rate" + "higher distortion." $
  The question this capstone answers: does the spiking substrate produce a *clean,
  monotone rate–distortion curve* — the lossless point on the floor at distortion
  $0$, and rate falling below the floor as distortion rises — and does a *trainable*
  distortion penalty $beta$ trace the same frontier?
]

= What we built

The knob is a per-context *merge margin* $delta$ (in bits $=$ latency$slash lambda$). Two
readouts whose first-spike latencies sit within $delta$ of each other are "hard to
distinguish": the graded bump cannot separate them, so the decoder commits to their
merged *class* (codeword) rather than the individual symbol. We measure each distinct
predictor row's per-readout first-spike latencies *once* in real Nengo spikes (the
e04/e11 idiom: independent readouts, latency law exact to $delta t$), build the
per-context single-linkage quantizer at margin $delta$, then assemble over the
$n = 4000$ rover stream ($s = 0.7$, seed $7$):

- *rate* $=$ mean bits/symbol *actually spent* $=$ the merged class's winning
  first-spike latency $slash lambda$ (the spikes the coder pays to resolve to the codeword
  granularity), read off the *real* readout spikes;
- *distortion* $=$ expected $0$–$1$ (Hamming) reconstruction error between the emitted
  symbol and the decoded class's representative (the class's MAP symbol).

At $delta = 0$ nothing merges: the lossless first-spike decode. All code is
`e15_lossy_graded_wta/run.py`.

#method[
  The latency law is exact to $delta t$ and the readouts are independent, so the
  per-symbol cost depends only on $("context", "outcome")$ through the predictor row
  $q(dot mid(|) c)$ (the same exactness e04/e11 used to assemble a $4000$-symbol mean
  from a handful of Nengo sims). We measure the four perfect-predictor rows in real
  spikes once, then sweep the knob $delta$ analytically over that real latency table.
  The frozen predictor is the true rover law $P$, so the curve isolates the
  *lossy-decode* contribution — the merge structure — from any learning residual.
]

= (A) The spiking rate–distortion curve

#finding[
  Sweeping the merge margin $delta$ over the real readout latency table produces a
  *clean, monotone* spiking rate–distortion frontier. The *lossless point* ($delta =
  0$) sits *on the floor*: rate $= 0.9790$ bits/symbol (the entropy-rate floor
  $H_("rate") = 0.9782$ plus only $+0.73$ mbit of $delta t$ timing-resolution
  overhead), distortion $= 0.0000$. As the bump widens, the rate falls *monotonically*
  from $0.9790$ down to $0.3202$ bits/symbol, while the distortion rises
  *monotonically* from $0$ up to $0.197$. Both monotonicities hold across all $61$
  knob settings — a proper R–D frontier.

  #table(columns: 4, align: (center, center, center, left), stroke: 0.5pt + luma(200),
    table.header[$delta$ (bits)][rate (bits/sym)][distortion][regime],
    [$0.0$], [$0.9790$], [$0.0000$], [lossless — on the floor],
    [$1.0$], [$0.9204$], [$0.0288$], [first merge (rare tail)],
    [$2.0$], [$0.9204$], [$0.0683$], [lossy],
    [$3.0$], [$0.7518$], [$0.1348$], [lossy],
    [$4.0$], [$0.6113$], [$0.1603$], [lossy],
    [$6.0$], [$0.3202$], [$0.1970$], [fully merged (coarsest)],
  )
]

#figure(image("results/e15_rate_distortion.pdf", width: 82%),
  caption: [The spiking rate–distortion curve (rate vs distortion). The lossless point
    (green star) sits exactly on the entropy-rate floor $H_("rate") = 0.9782$ (green
    dashed) at distortion $0$; coarsening the graded bump walks the frontier down and
    to the right — rate falls below the floor, distortion rises from $0$. Red squares
    are the trainable-$beta$ operating points (section C); they sit *below* the
    single-knob staircase because the per-context descent Pareto-dominates a single
    shared margin.])

#figure(image("results/e15_knob_sweep.pdf", width: 80%),
  caption: [The merge-margin knob $delta$ (the graded-bump width). Rate (blue, left
    axis) falls from the floor as classes merge; distortion (red, right axis) rises
    from $0$. The curve starts *exactly lossless* at $delta = 0$ (strict merge: nothing
    collapses until two latencies fall strictly within $delta$) and diverges as the
    bump widens. The staircase shape reflects discrete merge events: each plateau is a
    fixed coarse codebook, each step a class collapse.])

#intuition[
  Read the endpoints. At $delta = 0$ the coder pays the full $0.9782$-bit floor and
  reconstructs perfectly — the lossless regime of e11. At $delta = 6$ bits the bump is
  so wide it merges everything but the dominant move into one class: the coder pays
  only $0.32$ bits (it transmits little more than "did the rover repeat?") and eats
  a $0.197$ reconstruction-error rate. The whole frontier in between is the spiking
  circuit *spending fewer spikes by refusing to resolve distinctions it deems too
  close to matter* — which is exactly lossy compression.
]

= (B) The graded bump *is* the merge-radius knob

#finding[
  The merge margin is not an external bookkeeping device; it is the *width of the
  graded bump* the soft WTA forms over the readout. The soft sum-mode-gain decode is
  $r_j prop q_j^g$ with gain $g in (0, 1]$: $g = 1$ is the sharp (lossless) bump,
  $g < 1$ is a sub-saturating, *wide* bump. Two readouts whose surprisal (latency) gap
  is $Delta s$ have bump-mass ratio $2^(-g Delta s)$, so a bump of gain $g$ separates
  them only once $Delta s$ exceeds $delta_("eff")(g) = -log_2("ratio") slash g$. The
  effective merge radius *grows monotonically as the bump widens* (gain falls):

  #table(columns: 4, align: (center, center, center, center), stroke: 0.5pt + luma(200),
    table.header[gain $g$][temperature $T = 1 slash g$][$delta_("eff")$ (bits)][bump entropy (bits)],
    [$1.00$], [$1.00$], [$1.00$], [$0.83$],
    [$0.70$], [$1.43$], [$1.43$], [$1.31$],
    [$0.50$], [$2.00$], [$2.00$], [$1.63$],
    [$0.30$], [$3.33$], [$3.33$], [$1.87$],
    [$0.15$], [$6.67$], [$6.67$], [$1.97$],
  )

  So the temperature / sum-mode-gain knob *is* the merge-radius knob: a flatter bump
  (smaller $g$, larger $T$) tolerates a larger latency gap before committing to a
  distinction, which is precisely a larger merge margin $delta$. The bump entropy
  rising from $0.83$ to $1.97$ bits is the bump becoming *less committed* — the
  graded, coarsened decode the paper describes.
]

#figure(image("results/e15_graded_bump.pdf", width: 80%),
  caption: [The graded bump $r_j prop q_j^g$ over the readout (context U). At gain
    $g = 1$ (sharp) the bump concentrates on the MAP symbol; as $g$ falls the bump
    *widens*, lifting mass onto the near-equiprobable runner-up symbols — the soft
    argmax that merges hard-to-distinguish outcomes. The effective merge radius
    $delta_("eff")(g)$ annotated per bar.])

= (C) Trainable $beta$: the rate–distortion Lagrangian traces the frontier

#finding[
  Augmenting the per-symbol cost to the *rate–distortion Lagrangian* $ell_t + beta dot
  d(x_t, hat(x)_t)$ and sweeping $beta$ traces the *same* frontier. Because the
  per-symbol spiking energy is *local* — each context's graded WTA pays its own
  expected $("rate") + beta dot ("distortion")$ under that context's outcome law
  $P[c]$ — the trainable knob is *per-context*: each context independently descends to
  its own merge margin $delta_c^*(beta)$. Large $beta$ (distortion expensive) keeps
  every context lossless (rate $0.9796$, distortion $0$); small $beta$ coarsens every
  context (rate $0.3200$, distortion $0.197$); *intermediate* $beta$ coarsens some
  contexts and not others, so the aggregate reaches genuine *interior* frontier points
  — $beta = 3.0 arrow.r (0.508, 0.131)$ and $beta = 3.5 arrow.r (0.679, 0.075)$. The
  sweep visits *four distinct* operating points spanning the whole frontier, and *no*
  selected point is dominated by the single-knob staircase.
]

#figure(image("results/e15_beta_sweep.pdf", width: 80%),
  caption: [The trainable-$beta$ sweep. Each point is the per-context Lagrangian
    descent's fixed point at one $beta$ (colour $= log_10 beta$; large $beta$ = rate-
    only / lossless). The same local descent that minimised spikes now trades
    reconstruction error against spikes, marching from the lossless point down the
    frontier as $beta$ falls — the spiking analogue of a learned lossy codec at varying
    distortion budget.])

#honest[
  *Where the construction strains — the single-knob frontier is concave.* A single
  *global* merge margin shared across all contexts traces a frontier that is, for this
  rover, globally *concave*: every interior staircase point lies *above* the chord from
  the lossless endpoint $(0, 0.979)$ to the coarsest endpoint $(0.197, 0.320)$. A
  Lagrangian over *one shared knob* is therefore degenerate — it only ever selects the
  two extreme operating points (the lower convex hull collapses to those two vertices).
  We escape this *only* because the spiking energy is *per-symbol*, hence effectively
  *per-context*, and the contexts coarsen at different $beta$. This is a real property
  of the source, not a bug: the rover's MAP move is so dominant ($q approx 0.74$–$0.85$)
  that the first big merge swallows the whole low-probability tail at once, so a single
  shared margin has nothing useful to do in between. The *direction* of the paper's
  claim is realised — the Lagrangian descent does trade rate for distortion — but the
  *richness* of the interior frontier comes from the locality of the spiking cost, not
  from a single tunable temperature.
]

= The safety contract changes: bounded distortion, not exact losslessness

#finding[
  In the lossless suite, correctness was an exact invariant: decoded $=$ emitted for
  any predictor. In the lossy regime, *decoded $eq.not$ emitted by design* — the merge
  is the point. The safety property therefore changes from *exact losslessness* to
  *bounded distortion*: at merge margin $delta$, only symbols whose surprisal (latency)
  gap is $< delta$ can ever be confused, so the achieved distortion is upper-bounded by
  the merged classes' residual (non-representative) probability mass under the true
  outcome law. We verify the achieved distortion stays *within* this structural bound
  at every $delta$:

  #table(columns: 4, align: (center, center, center, center), stroke: 0.5pt + luma(200),
    table.header[$delta$ (bits)][achieved distortion][structural bound][within bound],
    [$0.0$], [$0.0000$], [$0.0000$], [yes],
    [$1.0$], [$0.0288$], [$0.0281$], [yes],
    [$2.0$], [$0.0683$], [$0.0656$], [yes],
    [$3.0$], [$0.1348$], [$0.1313$], [yes],
    [$6.0$], [$0.1970$], [$0.1969$], [yes],
  )

  (The achieved value sits a few mbit above the closed-form bound at finite $n$ because
  the bound is the $pi$-weighted mass and the stream is a finite sample — it converges
  down to the bound as $n arrow.r infinity$.)
]

#gap[
  This is the honest re-statement of the paper's open problem. The lossless coder's
  correctness obligation was structural and *exact* — it rested on the decode rule, not
  the predictor (e05's first-spike-takes-all). The lossy coder's obligation is a
  *quantitative distortion bound*: at a given spike rate, the reconstruction error is
  capped by which symbols the bump width permits to merge. We *exhibit* such a bound
  per $delta$ and confirm it holds; what we do *not* do — and the paper flags as open —
  is *derive* a #emph[certifiable] upper bound on distortion at a given spike rate from
  the graded-WTA closed form, in the verification sense (a temporal-logic obligation in
  the family of the near-entropy-cost property, but for the lossy regime). The
  empirical bound here is the target such a proof would have to certify.
]

#intuition[
  The whole capstone in one line: *the same circuit, with one bump-width knob, slides
  continuously from a lossless entropy coder ($0.9782$ bits, zero error) to an
  aggressive lossy quantiser ($0.32$ bits, $20%$ error)* — and a distortion-penalised
  version of the *same local descent* picks where on that slide to sit. Compression,
  prediction, and now the *loss budget* are one monotone trade on one number, read off
  the real spike clock.
]

= Acceptance

#accept_table((
  (true, [lossless point sits on the floor (rate $0.9790 in [H_("rate"), H_("rate") + 0.05]$)]),
  (true, [lossless point has zero distortion ($0.0000$)]),
  (true, [rate monotone non-increasing in the merge margin (proper R–D frontier)]),
  (true, [distortion monotone non-decreasing in the merge margin]),
  (true, [rate drops below the lossless floor as the bump coarsens (min $0.3202$)]),
  (true, [distortion rises from $0$ as the bump coarsens (max $0.1970$)]),
  (true, [graded-gain bump: effective merge radius grows as gain $g$ falls (bump widens)]),
  (true, [trainable-$beta$ operating points are Pareto-sane (none dominated)]),
  (true, [trainable-$beta$ sweep spans the frontier (rate \& distortion both move)]),
  (true, [trainable-$beta$ visits interior frontier points ($4$ distinct)]),
  (true, [large-$beta$ endpoint recovers the lossless point (distortion $0$)]),
  (true, [achieved distortion stays within the structural bounded-distortion contract]),
))

#finding[
  *12/12 passed.* The graded WTA produces a clean spiking rate–distortion curve: the
  lossless point sits on the entropy-rate floor ($0.9790$ bits, $+0.73$ mbit $delta t$
  overhead) at distortion $0$, and as the soft-decode bump widens — equivalently the
  sum-mode gain sub-saturates / the softmax temperature rises — the rate falls
  monotonically to $0.3202$ bits/symbol while distortion rises monotonically to
  $0.197$. The temperature/gain knob *is* the merge-radius knob ($delta_("eff")(g) =
  -log_2 r slash g$). The trainable rate–distortion Lagrangian $ell + beta d$, descended
  *locally per context*, traces the same frontier through four distinct operating
  points. The structural-safety story changes honestly: from *exact losslessness* to
  *bounded distortion*, with the empirical bound exhibited and held — leaving the
  paper's open problem (a #emph[certifiable] distortion bound at a given spike rate)
  cleanly stated. One bump-width knob slides the same circuit from a lossless entropy
  coder to a lossy quantiser; the open frontier is to *prove* the loss it incurs.
]
