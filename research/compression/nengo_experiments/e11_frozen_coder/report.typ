#import "../report_style.typ": *
#show: setup

#report_header("e11", "The full end-to-end spiking entropy coder (frozen perfect predictor)",
  "Tier 4 · end-to-end capstone · paper claim #3 + the three-stage circuit, composed and run whole")

#claim[
  Compose the three verified stages into *one* coder and confirm the headline
  information-theoretic identity holds through the *whole* spiking pipeline. The
  running context $c_t$ (the previous symbol) feeds a *frozen perfect* predictor
  $q(dot mid(|) c_t) = P[c_t]$ — the true rover row (note $"softmax"(log P) = P$, so a
  frozen perfect predictor just looks up the rover row); a *calibrated readout bank*
  (stage 2) races each readout to a first-spike latency equal to its surprisal; a
  *temporal winner-take-all decoder* (stage 3) commits to the first readout to fire.
  Run end to end on the rover stream ($s = 0.7$, seed $7$, per-window reset), the
  perfect predictor's mean per-symbol first-spike time / $lambda$ should land on the
  *entropy-rate floor* $H_"rate" = 0.9782$ bits/symbol, the decode should be *lossless*
  (decoded $=$ emitted), and mismatched predictors should pay the *KL stupidity tax*:
  memoryless $q = pi -> 1.7500$, wrong-momentum $s' = 0.4 -> 1.1133$.
]

= What we built

The three stages composed as one circuit:

#method[
  *Stage 1 (context, frozen).* The context $c_t$ is the previous symbol $x_(t-1)$; the
  frozen perfect predictor reads off the true rover row $q = P[c_t]$. No learning, no
  softmax decode error — this is the *idealized* coder against which e12 (the learned
  coder) and e06 (the attractor-held context) are measured. *Stage 2 (calibrated readout
  race).* `build_readout_bank` — $N = 4$ rest-pinned LIFs, readout $j$ fed
  $R I(q_j) = theta slash (1 - q_j^alpha)$ directly into `.neurons` — so its first spike
  from rest lands at $t^*(q_j) = -lambda log_2 q_j$. *Stage 3 (temporal WTA).*
  `metrics.decode_first_spike` over `metrics.per_window_first_spikes`: the first readout
  to cross threshold names the symbol.

  We run the coder *two ways*, both honest. (i) *The genuine continuous pipeline*:
  ONE Nengo simulation over $400$ windows ($160$ ms each, $48$ ms per-window blank
  reset), the *emitted* symbol's line driven at $q(x_t mid(|) c_t)$ (the encoder model
  of e05), decoded by first-spike-takes-all — the real coder, stage $1 -> 2 -> 3$. (ii)
  *The lookup-assembled long-stream mean*: because the $N$ readouts are independent and
  the latency law is exact to $approx delta t$ (e01/e02/e04), the per-symbol bits depend
  only on $(c_t, x_t)$ through $q(dot mid(|) c_t)$. We measure each *distinct* predictor
  row's per-readout latency once in real spikes (4 perfect $+$ 1 memoryless $+$ 4 wrong
  rows), then assemble the $n = 4000$-symbol mean by $(c_t, x_t)$ lookup — a handful of
  sims, not millions. We *check the two agree* on the same $400$ windows. All code is in
  `e11_frozen_coder/run.py`.
]

= (1) Losslessness: decoded $=$ emitted, end to end, noise-free

#finding[
  The genuine continuous pipeline (stage $1 -> 2 -> 3$) decodes *every* symbol
  correctly over the $400$-window run: *decode error $= 0.0000$* for the perfect
  predictor (no-spike windows $= 0$), and *$0.0000$ for a uniform $q = 1 slash 4$* as
  well. The uniform predictor is *slower* — mean *$1.936$* bits/symbol (each symbol
  $q = 1 slash 4 -> 2$ bits $= 40$ ms) versus the perfect predictor's *$0.985$* — but
  *never wrong*. This is the paper's structural-safety claim made literal: *correctness
  rests on the decode rule, not on the predictor* (a bad model is slow, never wrong).
]

#intuition[
  Losslessness is *structural*. The first readout to cross threshold names the symbol;
  since the *emitted* symbol's line is the one being driven, it is the only line that
  fires (encoder model), so the decode is exact regardless of *how confident* the model
  was. The predictor's quality controls only the *latency* (how many bits/joules the
  symbol costs), never the *identity*. The uniform model pays $2$ bits for every symbol
  and gets all of them right.
]

= (2) Rate $=$ floor: the headline number

#finding[
  *The headline.* Over the $n = 4000$-symbol rover stream, the frozen perfect
  predictor's mean per-symbol first-spike time / $lambda$ is
  $ "mean bits/symbol" = bold(0.9790) quad "vs the entropy-rate floor" quad H_"rate" = 0.9782, $
  *$+0.73$ mbits above the floor* — inside the acceptance band $[H_"rate", H_"rate" +
  0.05]$. The genuine continuous pipeline lands at *$0.9853$* bits/symbol on its
  $400$-window prefix (the lookup over the same prefix gives $0.9949$; the two agree to
  $9.7$ mbits — the continuous sim and the per-row lookup are the *same coder*). The
  spiking entropy coder spends, per symbol, *exactly the entropy rate of the source* —
  the full circuit realizes claim #3 end to end.
]

#figure(image("results/e11_timeline.pdf", width: 88%),
  caption: [The end-to-end pipeline running (perfect predictor), first $24$ symbols.
    Each marker is one window's winner first-spike latency; the green dashed line is the
    floor $lambda H_p = 19.56$ ms. Every emitted symbol (green label) is decoded
    correctly — the per-window latency *is* that symbol's surprisal, and the stream mean
    sits on the floor.])

= (3) Three baselines: the stupidity tax of a worse model

#finding[
  The three-predictor table reproduces `validate.py` part B2 *in real spikes*, end to
  end, with the correct ordering and the correct KL gaps:

  #table(columns: 5, align: (left, center, center, center, center), stroke: 0.5pt + luma(200),
    table.header[predictor $q$][spike mean][numpy surprisal][theory $H(p, q)$][$delta t$ overhead],
    [perfect $q = P$], [*0.9790*], [0.9775], [0.9782], [$+1.4$ mbit],
    [memoryless $q = pi$], [1.7521], [1.7484], [1.7500], [$+3.7$ mbit],
    [wrong momentum $s' = 0.4$], [1.1142], [1.1128], [1.1133], [$+1.5$ mbit],
  )

  The *efficiency ordering perfect $<$ wrong $<$ memoryless* holds in spikes. The
  *stupidity taxes* match the closed form: the memoryless model pays
  $1.7521 - 0.9790 = bold(0.773)$ extra bits/symbol (theory: the mutual information
  $I = 0.7718$ the cycle recovers), and the wrong-momentum model pays
  $1.1142 - 0.9790 = bold(0.135)$ extra (theory $D_"KL" = 0.1351$). A worse internal
  model spends measurably more spikes-as-bits per symbol — and the excess *is* the KL
  divergence between the model and the world.
]

#figure(image("results/e11_bits_per_symbol.pdf", width: 80%),
  caption: [The flagship reproduction. Measured spiking-coder mean (blue) vs
    cross-entropy-rate theory (red) for the three predictors; the green dashed line is
    the entropy-rate floor $0.9782$, the grey dotted line the marginal $1.7500$. The
    perfect predictor sits on the floor; the memoryless model is pushed all the way up
    to the marginal (it has thrown away the momentum); the wrong-momentum model sits in
    between. Better model $=$ fewer bits, exactly $H(p, q)$.])

#intuition[
  This one figure *is* the paper's thesis. The floor $0.9782$ is what the source costs
  if you know its conditional law; the marginal $1.7500$ is what it costs if you know
  only the frequencies. The gap between them, $0.77$ bits, is the *momentum* — and the
  memoryless model, by ignoring the cycle, pays all of it back as latency. Compressing
  better, predicting better, and spending fewer spikes are *one* number, read off the
  spike clock.
]

= (4) Where the overhead comes from: dt rounding vs context error

The perfect predictor sits $+0.73$ mbits above the floor. We *decompose* this spiking
overhead into its independent physical sources.

#finding[
  *Source (a): the $delta t$ rounding tax.* A spike can land only on the $delta t$
  grid, and rounding a continuous latency up to the next grid point is *biased upward*.
  Holding the predictor *and* the context perfect, only the clock changes: at the *fine*
  clock $delta t = 10^(-4)$ the mean is $0.9790$ ($+1.4$ mbit over the same-stream
  surprisal); at the *coarse* clock $delta t = 10^(-3)$ it is $0.9951$ ($+17.6$ mbit).
  *Coarsening the clock $10 times$ adds $16.2$ mbits* — and both vanish as $delta t -> 0$.
  This is the spiking analogue of the integer-bit penalty: continuous time carries none,
  a *clocked* substrate pays an $O(delta t)$ resolution tax.
]

#finding[
  *Source (b): context error.* A *perfect-lookup* context (the previous symbol read
  exactly) adds *zero* bits. But the e06 ring attractor that holds $c_t$ *drifts*, so a
  fraction $epsilon$ of windows read the *wrong* previous symbol, selecting the wrong
  predictor row. We sweep the misread rate over a $1500$-symbol arm: the added cost is
  *linear at $+20.4$ mbits per $1%$ context-misread* ($0.9985 -> 1.0496 -> 1.1184$ at
  $epsilon = 0, 2, 5%$). The crucial point: e06 measured that the ring holds all $4$
  states *well inside the half-cell margin over an inter-symbol interval* — its
  effective misread rate is $approx 0%$ — so the attractor-held context adds only
  $approx 0.05$ bits even at a generous $2%$ misread, and essentially nothing at e06's
  measured fidelity.
]

#figure(image("results/e11_overhead.pdf", width: 96%),
  caption: [Overhead decomposition. *(a)* The $delta t$-rounding tax (perfect context):
    $+1.4$ mbit at the fine clock, $+17.6$ mbit at the coarse clock — the integer-bit
    penalty's spiking analogue, vanishing as $delta t -> 0$. *(b)* Context error: each
    $1%$ of held-context misreads adds $+20.4$ mbits; e06's drifting ring sits at the
    left edge (misread $approx 0%$), so context contributes $approx 0$ at measured
    fidelity.])

#honest[
  *The overhead attribution, stated plainly* (perfect context, fine clock, above the
  floor $0.9782$): the *total* spiking overhead on this $4000$-symbol stream is
  $bold(+0.73)$ mbit, which splits as *sampling* (finite-stream luck of this seed's
  prefix) $-0.69$ mbit $+$ *dt rounding* $+1.41$ mbit $+$ *context error* $+0.00$ mbit
  (perfect lookup). The sampling term is signed and shrinks with $n$; the dt term is
  signed *positive* and shrinks with $delta t$; the context term is what e06 measured to
  be $approx 0$ over an ISI but which an imperfect attractor would inflate at $20.4$
  mbits/$%$. *The headline is honest:* the coder reaches the floor to within a fraction
  of a millibit, and every millibit of the residual is named and attributed to a
  physical cause (clock resolution, finite stream, attractor drift) — none of it to a
  failure of the identity itself.
]

= Acceptance

#accept_table((
  (true, [end-to-end perfect-predictor decode error $= 0$ (losslessness)]),
  (true, [end-to-end uniform-predictor decode error $= 0$ (slow, never wrong)]),
  (true, [uniform predictor slower than perfect ($1.936$ vs $0.985$ bits/symbol)]),
  (true, [perfect lookup mean in $[H_"rate", H_"rate"+0.05]$ ($0.9790$, floor $0.9782$)]),
  (true, [genuine continuous pipeline mean in band ($0.9853$)]),
  (true, [continuous pipeline matches the lookup ($|"diff"| = 9.7$ mbit)]),
  (true, [memoryless mean approaches $H_"marginal" = 1.7500$ ($1.7521$)]),
  (true, [wrong-momentum mean approaches $1.1133$ ($1.1142$)]),
  (true, [efficiency ordering perfect $<$ wrong $<$ memoryless]),
  (true, [memoryless stupidity tax $=$ mutual info $0.7718$ ($0.773$)]),
  (true, [wrong-momentum stupidity tax $= 0.1351$ ($0.135$)]),
  (true, [dt-rounding overhead positive, grows with coarser clock ($1.4 -> 17.6$ mbit)]),
  (true, [context-error overhead positive and attributable ($20.4$ mbit/$%$ misread)]),
))

#finding[
  *13/13 passed.* The flagship reproduction holds end to end: the *full three-stage
  spiking entropy coder* — frozen perfect predictor $-> $ calibrated readout race $->$
  temporal first-spike-takes-all decode — spends, per symbol, *exactly the entropy rate
  of the source*, $bold(0.9790)$ bits on the floor $0.9782$, while decoding *losslessly*
  (zero errors, any predictor). A worse model pays its KL stupidity tax in spikes —
  memoryless $1.7521$ (the full $+0.77$-bit mutual-information penalty), wrong-momentum
  $1.1142$ ($+0.135$ bits) — with the efficiency ordering intact. The $+0.73$-mbit
  residual above the floor is *decomposed and named*: $delta t$ rounding ($+1.4$ mbit
  fine, $+17.6$ coarse, vanishing as $delta t -> 0$), finite-stream sampling, and
  attractor-context drift ($20.4$ mbit/$%$, but $approx 0$ at e06's measured ring
  fidelity). The identity *spike-time $=$ surprisal*, summed to *stream-cost $=$
  cross-entropy rate*, survives the whole spiking pipeline. e12 replaces the frozen
  perfect predictor with the *learned* one (e09) to close the loop.
]
