#import "../report_style.typ": *
#show: setup

#report_header("e04", "The calibrated readout race: lowest surprisal fires first",
  "Tier 1 · circuit stage 2 · paper claim #2 (the readout bank) + the stream identity")

#claim[
  A bank of $N = 4$ calibrated LIF readouts races to threshold: readout $j$ is
  driven by the calibration current $R I(q_j)$, so its first spike from rest lands
  at latency $t^*(q_j) = -lambda log_2 q_j$ (e01/e02). Two predictions follow.
  *(a)* The readout with the *largest* model probability $q_j$ fires *first*, and its
  latency equals $-lambda log_2 max_j q_j$ — the network reads its own MAP symbol off
  as a spike *time*. *(b)* Over a rover stream, feeding each window the predictor row
  for the previous symbol and reading the *realized* symbol's readout latency, the
  $pi$-weighted mean spike-time / $lambda$ equals the cross-entropy rate $H(p, q)$:
  $0.9782$ bits for the perfect predictor $q = P$ (the entropy-rate floor),
  $1.7500$ for the memoryless $q = pi$, $1.1133$ for a wrong-momentum guess.
]

= What we built

For each distinct predictor row we build a fresh `build_readout_bank` (the shared,
unit-tested builder: $N$ independent rest-pinned LIFs, readout $j$ fed
$R I(q_j) = theta slash (1 - q_j^alpha)$ directly into `.neurons`), drive it from
rest at $delta t = 10^(-4)$, and record every readout's first-spike time. Because
the readouts are independent and the latency law is exact to $approx delta t$
(e01/e02), the stream means are assembled by lookup over the distinct predictor rows
(4 for $q = P$, 1 for $q = pi$, 4 for $q = P_"wrong"$) — a handful of Nengo sims, not
millions of single-symbol runs. All code is in `e04_readout_bank/run.py`.

= (a) The race: highest $q$ fires first, at its surprisal

#finding[
  For every context the readout for $arg max_j P_(i j)$ wins the race, and the
  *full* latency ordering across all four readouts matches the surprisal ordering
  $-log_2 P_(i j)$ exactly. The winner's latency equals $-lambda log_2 max_j q_j$ to
  $<= 0.45 delta t$:

  #table(columns: 6, align: (center,)*6, stroke: 0.5pt + luma(200),
    table.header[context][$max_j q_j$][winner][winner lat (ms)][$-lambda log_2 max q$ (ms)][$|"err"| slash delta t$],
    [U], [0.850], [U], [4.700], [4.689], [0.11],
    [D], [0.775], [D], [7.400], [7.355], [0.45],
    [L], [0.7375], [L], [8.800], [8.786], [0.14],
    [R], [0.7375], [R], [8.800], [8.786], [0.14],
  )

  The sticky rover ($s = 0.7$) puts the most mass on *repeating the last move*, so
  the winning readout is always the "repeat" symbol — exactly what a temporal WTA
  on this bank decodes (e05 closes that loop).
]

#figure(image("results/e04_race_raster.pdf", width: 80%),
  caption: [The readout race for context U. Each blue bar is a readout's measured
    first spike (rows sorted by $q$, highest on top); the red $times$ is the analytic
    $-lambda log_2 q_j$. The $q = 0.85$ readout fires at $4.7$ ms; the three
    low-probability readouts fire $50$–$75$ ms later. Spike *time* is surprisal, and
    the first to cross threshold is the MAP symbol.])

#figure(image("results/e04_winner_latency.pdf", width: 78%),
  caption: [Winner first-spike latency per context, measured (blue) vs the analytic
    surprisal of the MAP symbol $-lambda log_2 max_j q_j$ (red). The two agree to a
    fraction of $delta t$ everywhere.])

#intuition[
  Each readout is the e02 calibration identity in miniature; the *bank* turns the
  four parallel identities into a competition whose winner-and-time jointly encode
  $(arg max_j q_j, -log_2 max_j q_j)$ — the MAP symbol *and* its information content,
  in a single first spike.
]

= (b) The stream identity: spike-time mean = cross-entropy rate

#finding[
  Over an $n = 4000$-symbol rover stream (seed $7$), the $pi$-weighted mean
  spike-latency / $lambda$ reproduces the three-row table of `validate.py` part B2:

  #table(columns: 5, align: (left, center, center, center, center), stroke: 0.5pt + luma(200),
    table.header[predictor $q$][spike mean][numpy surprisal][theory $H(p, q)$][$delta t$ overhead],
    [perfect $q = P$], [0.9790], [0.9775], [0.9782], [$+0.0014$],
    [memoryless $q = pi$], [1.7521], [1.7484], [1.7500], [$+0.0037$],
    [wrong momentum $s' = 0.4$], [1.1142], [1.1128], [1.1133], [$+0.0015$],
  )

  The perfect predictor's spike-time mean ($0.9790$) sits on the entropy-rate floor
  $H_"rate" = 0.9782$; the memoryless model pays $0.9790 arrow 1.7521 approx
  H_"marginal" = 1.7500$ (its excess $approx 0.77$ bits is the mutual information the
  cycle recovers — the "stupidity tax" of ignoring momentum); the wrong-momentum
  model pays $1.1142 approx 1.1133$ (an excess KL of $0.135$ bits). The efficiency
  ordering *perfect $<$ wrong $<$ memoryless* holds in real spikes.
]

#figure(image("results/e04_bits_per_symbol.pdf", width: 78%),
  caption: [Mean bits/symbol, measured spike-time mean (blue) vs cross-entropy-rate
    theory (red), for the three predictors. The spike code spends exactly the
    cross-entropy rate of whatever model drives it — the better the model, the fewer
    spikes-as-bits per symbol.])

#honest[
  *The $delta t$ timing-resolution overhead is real, and it is the spiking analogue
  of the integer-bit penalty.* Every measured spike mean sits *above* both the numpy
  surprisal of the *same* sampled stream ($+1.4$ to $+3.7$ mbits) and the closed-form
  cross-entropy rate. The gap is not sampling noise — it is *signed*: a spike can only
  land on the $delta t = 10^(-4)$ grid, and the rounding of a continuous latency to
  the next grid point is biased *upward* (a spike scheduled at $4.689$ ms is recorded
  at $4.700$ ms). Fast/confident symbols (small surprisal, $approx$ few grid steps)
  carry the largest relative rounding, so the memoryless model — which fires *more*
  fast spikes on the high-$pi$ symbol U — accrues the largest overhead ($+3.7$ mbits).
  This is the paper's prediction made literal: continuous time carries no integer-bit
  penalty, but a *clocked* substrate pays an $O(delta t)$ resolution tax that shrinks
  with the timestep (e01 Fig. 3). It is a finding, not a failure: the acceptance band
  for the perfect predictor is $[H_"rate", H_"rate" + 0.05]$, and the measured value
  lands $0.0008$ bits above the floor.
]

= (c) The discrete-clock penalty: ties only when surprisal gaps are small

A second consequence of the clock: at finite $delta t$ two readouts can land their
first spikes in the *same* $delta t$ bin — an unbreakable tie the continuous-time
code never has. A tie occurs exactly when the top-two surprisal gap
$lambda |log_2 q_1 - log_2 q_2| < delta t$.

#finding[
  *On the natural rover the penalty never bites.* The sticky chain puts
  $0.74$–$0.85$ on the repeat move and $<= 0.15$ on the rest, so the winner's race
  against the runner-up is decided by a *huge* margin — the smallest top-two surprisal
  gap across contexts is $46.0$ ms, $46times$ the *coarse* $delta t = 1$ ms bin. The
  measured tie rate is $0.00$ at both $delta t = 10^(-4)$ and $delta t = 10^(-3)$. A
  far-from-uniform source has no integer-penalty problem at all.
]

#gap[
  *The mechanism is real, and we provoke it.* We engineer a deliberately near-tied
  distribution $q = [0.4500, 0.4453, 0.0523, 0.0523]$ whose top-two surprisal gap is
  $0.300$ ms — engineered to straddle the two clocks ($delta t_"fine" = 0.1$ ms $<
  0.300$ ms $< delta t = 1.0$ ms). The race then *resolves at the fine clock* (top-2
  realized gap $0.300$ ms, distinct bins, no tie) but *fuses at the coarse clock*
  (both first spikes share one $1$ ms bin → tie). This is the integer-bit penalty's
  spiking analogue made concrete: the clock loses information about the winner exactly
  when two model probabilities are nearly equal, and a finer clock recovers it.
]

#figure(image("results/e04_tie_rate.pdf", width: 74%),
  caption: [Readout-tie (collision) rate vs the timestep. Green: the natural rover —
    zero ties at any clock, because its top-2 surprisal gaps ($>= 46$ ms) dwarf
    $delta t$. Blue: an engineered near-tie ($0.30$ ms gap) — resolved at $delta t =
    10^(-4)$, fused at $delta t = 10^(-3)$. The discrete clock penalizes only the
    near-uniform races.])

= Acceptance

#accept_table((
  (true, [$arg max_j q_j$ readout fires first for every context (U, D, L, R)]),
  (true, [winner latency $= -lambda log_2 max_j q_j$ within $approx delta t$ (max $0.45 delta t$)]),
  (true, [full per-context latency order $=$ surprisal order for all contexts]),
  (true, [perfect-predictor mean in $[H_"rate", H_"rate"+0.05]$ ($0.9790$, floor $0.9782$)]),
  (true, [memoryless mean approaches $H_"marginal" = 1.7500$ ($1.7521$)]),
  (true, [wrong-momentum mean approaches $1.1133$ ($1.1142$)]),
  (true, [efficiency ordering perfect $<$ wrong $<$ memoryless holds in spikes]),
  (true, [$delta t$ timing-resolution overhead is real, signed, and positive ($+1.4$–$3.7$ mbits)]),
  (true, [natural rover never ties (top-2 gaps $>= 46$ ms $>> delta t$)]),
  (true, [engineered near-tie ($0.30$ ms gap) resolves at $delta t_"fine"$, fuses at $delta t$]),
))

#finding[
  *10/10 passed.* The calibrated readout bank realizes the paper's claim #2 in real
  spikes: the lowest-surprisal readout wins the race, its first-spike time is the
  surprisal of the MAP symbol, and the $pi$-weighted stream mean equals the
  cross-entropy rate of whatever predictor drives it — $0.9790$ bits on the
  entropy-rate floor for a perfect model, $1.75$ for a memoryless one. The two honest
  costs are the *signed* $O(delta t)$ timing-resolution overhead (the integer-bit
  penalty's analogue, $approx +0.0014$ bits for the perfect predictor) and a
  discrete-clock tie that bites only when two model probabilities are nearly equal —
  which the natural rover never does. The bank is now ready for the temporal
  winner-take-all decoder (e05).
]
