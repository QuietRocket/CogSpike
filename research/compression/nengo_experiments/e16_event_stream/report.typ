#import "../report_style.typ": *
#show: setup

#report_header("e16", "A neuromorphic event stream: spikes-as-bits leaves the toy chain",
  "Tier 4 capstone · the realistic anchor · paper outlook (the event-camera target)")

#claim[
  The momentum rover (e01–e12) is a *rehearsal*. The paper's real target is data that
  *is already spikes*: a dynamic-vision-sensor / event-camera stream whose pixels emit
  events only on local brightness *change*. Such streams are heavily *spatiotemporally
  redundant* — objects move predictably — so a recurrent predictor that *pre-charges*
  the expected next events by *inhibitory feedback* (the e07 sign: shunt the expected
  drive so the soma integrates only the residual) and emits only the *unpredicted*
  events is a literal, hardware-native compressor: *input is spikes, the code is spikes,
  the residual IS the compressed stream*. We build a tiny synthetic event camera, run
  the predictive circuit, and measure the *compression ratio* and *where the residual
  concentrates*.
]

= The synthetic event camera

#intuition[
  A real DVS pixel is silent until its brightness *changes*, then it fires an *ON* spike
  on a rising edge and an *OFF* spike on a falling edge. A rigid object translating at
  constant velocity therefore lights up only its *two moving edges*, and those edges
  *move predictably*: the ON edge at pixel $p$ this frame is the ON edge at pixel $p+v$
  next frame. That is exactly the redundancy a predictive coder eats. The novelty —
  the thing a compressor *cannot* predict away — lives only where the motion *breaks*:
  the object *appears*, *reverses*, or *changes speed*.
]

#method[
  *Source.* A 1-D array of $P = 24$ pixels; a bright bar of width $4$ translates at a
  programmed velocity, *wrapping* mod $P$. A pixel emits an *ON* event when its
  brightness rises $0 arrow 1$ (bar arrives) and *OFF* when it falls $1 arrow 0$ (bar
  leaves). We program a velocity *schedule* with steady phases punctuated by three
  *unpredictable moments*: the bar *appears* (onset, frame 3), the motion *reverses*
  $+v arrow -v$ (frame 19), and a *velocity jump* $-v arrow +2v$ (frame 31). $41$ frames.

  *Predictive circuit.* Two LIF soma banks — one *ON* channel, one *OFF* channel, each
  of $P$ neurons (`make_lif`, $V(0)=0$, current injected at `ens.neurons` with gain $1$,
  bias $0$: the e07 idiom). The raw event at pixel $p$ drives soma $p$ *excitatorily*
  ($J = 6 times$ rheobase). A recurrent *shift predictor* anticipates the next frame by
  rolling the previous frame's event map by the motion velocity $hat(v)$ and feeds an
  *inhibitory* current $-J$ to each predicted pixel/polarity. The soma integrates the
  e07 residual $J_"eff" = J dot (e_"raw" - e_"pred")$ (raw minus predicted event): a correctly-anticipated event
  meets balanced E/I ($J_"eff" = 0$, *silent*); an unpredicted one keeps $J_"eff" = +J$
  and *fires*; an over-predicted one ($J_"eff" = -J$) is clamped at $V gt.eq 0$, silent.
  A per-frame blank resets every soma to rest (the e05/e11 windowed-reset idiom), all
  $P$ pixels of a channel run as one ensemble in a single continuous Nengo simulation,
  $delta t = 10^(-4)$. The *headline* predictor uses an *adaptive* motion model
  $hat(v)[k] = v[k-1]$ — a causal one-frame-memory velocity estimate, the best an online
  shift estimator can do — which tracks steady motion at *any* velocity and is wrong for
  *exactly one frame* at each transition. Code: `e16_event_stream/run.py`.
]

= The stream is sparse and redundant

#finding[
  The raw stream carries $98$ events over $41$ frames ($2.39$/frame; $51$ ON, $47$ OFF),
  a *sparsity* of $0.050$ — only $5%$ of pixel·frame·polarity cells fire. Under the
  adaptive motion model, $0.895$ of each frame's events are *anticipated* by the
  previous frame rolled by $hat(v)$: the stream is genuinely *spatiotemporally
  redundant*, which is the fuel the compressor burns. (A raw pass-through control with
  *no* predictor transcribes all $98$ events into exactly $98$ spikes — the soma bank is
  a faithful channel, so every reduction below is the predictor's doing, not a
  thresholding loss.)
]

= The headline: spikes in, fewer spikes out, residual at the novelty

#figure(image("results/e16_raster.pdf", width: 99%),
  caption: [*Top:* the raw event raster (pixel × frame), ON red / OFF blue. The bar's two
    edges trace clean diagonals — steady $+v$ motion (frames 3–18), the *reversal* to
    $-v$ (frame 19), and the $+2v$ *jump* (frame 31), all heavily redundant. *Bottom:*
    the residual after predictive subtraction. The entire redundant interior is
    *cancelled away*; the $12$ surviving spikes sit *only* on the three transition frames
    (and their one-frame re-lock). $98 arrow 12$ events: an $8.2 times$ compression with
    the residual landing exactly on the novelty.])

#finding[
  The predictive circuit compresses $98$ raw events to $12$ residual spikes — a
  *compression ratio of $8.17 times$*. The residual is *perfectly concentrated*: all
  $12$ residual spikes ($100%$) fall in the novel frames (onset, reversal, jump, and
  their one-frame re-lock), while the $80$ raw events of *steady predictable motion* are
  compressed to *zero* residual ($80 times$, complete cancellation). The emitted spike
  train is no longer a copy of the moving bar — it is a *surprise stream* that fires only
  when the world departs from the motion model, which is the e07 claim made literal on a
  spatial event stream.
]

#figure(image("results/e16_per_frame.pdf", width: 99%),
  caption: [Per-frame event counts. Raw events (blue) run at $2$/frame through steady
    motion and $4$/frame after the jump; the adaptive-model residual (red) is *zero
    everywhere except the three shaded novelty windows*. The grey line is the *static
    $+v$* predictor's residual: a motion model that does *not* adapt leaks $2$ residual
    events through *every* post-reversal and post-jump frame — the honest contrast that
    the compression is the *motion model's* doing.])

= Compression tracks predictability

#figure(image("results/e16_compression_vs_predictability.pdf", width: 74%),
  caption: [Compression ratio vs stream predictability, sweeping a velocity *jitter* $rho$
    (each frame the bar makes an extra random $plus.minus 1$ hop with probability $rho$,
    corrupting the redundancy a fixed $+v$ roll can capture). As predictability falls from
    $0.94$ ($rho = 0$) to $0.24$ ($rho = 1$), compression collapses from $16.4 times$
    toward $1.31 times$ — no redundancy, nothing to compress. Correlation $r = +0.84$.])

#finding[
  Compression is not a free lunch — it is *exactly* the stream's predictability cashed
  out. On a purely steady $+v$ bar ($rho = 0$, predictability $0.94$) the fixed-velocity
  circuit reaches $16.4 times$ compression ($82 arrow 5$ events). As velocity jitter
  corrupts the motion ($rho: 0 arrow 1$), predictability decays to $0.24$ and the
  compression ratio falls monotonically in trend to $1.31 times$ ($84 arrow 64$): a
  stream with no redundancy is *incompressible*, and the circuit correctly emits nearly
  every event. The compression ratio and the predictability correlate at $r = +0.84$ —
  the compressor delivers precisely the redundancy that is there to take.
]

= The honesty check: this is a rehearsal, and the motion model is the lever

#honest[
  *This is synthetic and low-dimensional, and we say so.* One 1-D array, $P = 24$ pixels,
  a single rigid bar, a hand-built velocity schedule. It demonstrates the *compression
  mechanism* — inhibitory pre-charge of predicted events, residual = surprise — *not* a
  learned vision system on real DVS data. Two honest caveats are made explicit and
  measured:

  *(i) The compression is the motion model's doing, not the substrate's.* The headline
  uses an *adaptive* one-frame-lag velocity estimate $hat(v)[k] = v[k-1]$. A *static*
  $+v$ predictor that does not adapt still compresses ($1.92 times$) but *leaks* residual
  through every steady $-v$ and $+2v$ frame ($0.29$ of its residual is on the novelty,
  vs $1.00$ for the adaptive model): a wrong motion model smears the residual across the
  phases it cannot fit. So the result is a statement about the *predictor* the inhibitory
  loop is given — the substrate faithfully emits whatever residual the model leaves.
  Even the adaptive model pays a *one-frame lag* at each transition (its $hat(v)$ is a
  frame stale), which is precisely why the residual is on the transition frame *and its
  successor* — the realistic cost of any causal online estimator.

  *(ii) The predictor is provided, not learned here.* $hat(v)$ is read from the true
  motion. Learning the shift online (a PES-style velocity estimate from the residual's
  bulk displacement, in the spirit of e09) is the natural next step and is *not* claimed.
  What e16 establishes is the *architecture*: spikes in, spikes out, the residual is the
  compressed stream, and its size is governed by predictability — the bridge from the
  toy rover chain to data that is already spikes.
]

#gap[
  The inhibitory subtraction inherits e07's *shunting-balance* idealization: here the
  prediction either fully matches an event (exact cancellation, $J_"eff" = 0$) or does
  not (full residual, $J_"eff" = plus.minus J$), because the event maps are binary, so
  the clean subtractive arithmetic is exact *per pixel*. A graded-brightness or
  conductance-based DVS model would reintroduce the divisive-shunt time-constant gap e07
  characterized — a real-camera extension, flagged not closed.
]

= Acceptance

#accept_table((
  (true, [raw pass-through (no predictor) faithfully transcribes every raw event ($98 arrow 98$)]),
  (true, [synthetic event stream is *sparse* (sparsity $0.050 < 0.2$)]),
  (true, [stream is *spatiotemporally redundant* (adaptive-model predictability $0.895 > 0.5$)]),
  (true, [predictive circuit *compresses* (ratio $8.17 times > 1$)]),
  (true, [residual *concentrates at novelty* ($100%$ of residual in the onset/reversal/jump frames)]),
  (true, [steady predictable motion is compressed away ($80$ raw $arrow 0$ residual, $80 times$)]),
  (true, [compression *tracks predictability* (corr $+0.84 > 0.5$ over the jitter sweep)]),
  (true, [compression collapses toward $1 times$ at max jitter ($1.31 times < 16.4 times$)]),
  (true, [adaptive motion model beats the static $+v$ model ($8.17 times > 1.92 times$)]),
))

#finding[
  *9/9 passed.* A synthetic event camera emits a sparse ($5%$), redundant ($0.895$
  predictable) spike stream as a bar translates, reverses, and jumps. A recurrent
  inhibitory predictor — the e07 sign on a spatial event field — pre-charges the expected
  edge motion and emits only the residual: $98 arrow 12$ events ($8.17 times$), with
  *every* residual spike on the three unpredictable transitions and steady motion
  cancelled to zero. Across a velocity-jitter sweep the compression ratio falls from
  $16.4 times$ to $1.31 times$ as predictability decays ($r = +0.84$): the compressor
  cashes out exactly the redundancy present, no more. This is the *spikes-as-bits leaves
  the toy chain* anchor — input is spikes, the code is spikes, the residual is the
  compressed stream — delivered honestly as a low-dimensional rehearsal whose compression
  is governed by the motion model the inhibitory loop is handed.
]
