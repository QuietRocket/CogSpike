#import "../report_style.typ": *
#show: setup

#report_header("e02", "Spike time is surprisal: the calibration identity",
  "Tier 0 · single-neuron primitive · paper Theorem 1 + the §4 honesty checks")

#claim[
  Drive a readout whose model probability is $q$ with the calibration current
  $R I(q) = theta slash (1 - q^alpha)$, $alpha = lambda slash (tau ln 2)$. Then its
  first-spike latency is *exactly* $t^*(q) = -lambda log_2 q$ — the surprisal of $q$,
  in seconds (paper Theorem 1). The paper then flags three physical limits: finite
  drive caps the representable probability at $q_max approx 0.85$–$0.95$; the
  rare-symbol ($q arrow 0$) end is noise-dominated; finite timing resolution
  replaces the discrete code's integer-bit penalty.
]

= The exact identity, in real spikes

With $lambda = tau_(r c) = 20$ ms and $alpha = 1 slash ln 2 approx 1.4427$, we drive
a single rest-pinned `nengo.LIF` (e01) with $R I(q)$ for a grid of $q$ and measure
the first-spike latency at $delta t = 10^(-4)$.

#figure(image("results/e02_calibration.pdf", width: 78%),
  caption: [First-spike latency vs surprisal. Every measured point (blue) lies on
    the line $t^* = -lambda log_2 q$ (red): the spike's *time* is the symbol's
    information content, in bits $times lambda$.])

#finding[
  The identity holds to a maximum of *$1.00 delta t$* across $q in [0.0375, 0.98]$
  — i.e. exact up to the timing grid, the spiking analogue of the numpy validator's
  $1.56 times 10^(-5)$ Euler residual. Representative rows:

  #table(columns: 5, align: (center,)*5, stroke: 0.5pt + luma(200),
    table.header[$q$][$R I slash theta$][$t^*$ meas (ms)][$-lambda log_2 q$ (ms)][$|"err"| slash delta t$],
    [0.95], [14.02], [1.50], [1.48], [0.20],
    [0.90], [7.09], [3.10], [3.04], [0.60],
    [0.50], [1.58], [20.10], [20.00], [1.00],
    [0.125], [1.05], [60.10], [60.00], [1.00],
    [0.0375], [1.01], [94.80], [94.74], [0.61],
  )

  The drive column reproduces the paper's table on the nose ($7.09times$ rheobase at
  $q=0.9$, $14times$ at $q=0.95$, $35times$ at $q=0.98$). Because the first spike
  from rest carries no refractory term (e01), the *paper-faithful* calibration
  already yields the exact identity — no refractory-corrected drive is required (a
  simplification over the project plan's two-arm hypothesis).
]

= Honesty check (a): finite drive caps $q_max$

#figure(image("results/e02_drive_ceiling.pdf", width: 74%),
  caption: [The calibration drive $R I slash theta = (1 - q^alpha)^(-1)$ diverges
    well inside the working range. A few-fold rheobase ceiling pins a maximum
    representable probability $q_max$.])

#gap[
  A biophysical neuron supplies only a few-fold rheobase current. Capping the drive
  at a ceiling pins $q_max$ and imposes a *floor latency* $t_min$ on the most
  confident symbols (they fire fast, but not arbitrarily fast):

  #table(columns: 3, align: (center,)*3, stroke: 0.5pt + luma(200),
    table.header[drive ceiling][$q_max$][floor latency $t_min$],
    [$5times$ rheobase], [$0.857$], [$4.46$ ms],
    [$10times$ rheobase], [$0.930$], [$2.11$ ms],
    [$15times$ rheobase], [$0.953$], [$1.38$ ms],
  )

  Directly measured: with the drive capped at $10times$, symbols with $q > q_max =
  0.930$ all *saturate* to the same floored latency ($2.2$ ms) instead of their
  desired $t^*$ ($q=0.98$ "wants" $0.58$ ms but is floored at $2.2$). This is the
  paper's $q_max approx 0.85$–$0.95$ prediction, now a measured saturation — and the
  real origin of the minimum-latency floor (drive saturation), not $tau_("ref")$.
]

= Honesty check (b): the rare-symbol tail is noise-dominated

#figure(image("results/e02_noise_tail.pdf", width: 74%),
  caption: [First-spike latency mean $plus.minus$ std under membrane noise
    ($sigma = 0.15$, 40 trials/q). As $q arrow 0$ the drive approaches rheobase, the
    neuron lingers near threshold, and the latency becomes a broad random variable.])

#finding[
  The coefficient of variation of the latency grows monotonically as the symbol
  gets rarer — from $"CV" = 0.008$ at $q = 0.7$ to $0.110$ at $q = 0.04$ (a $14times$
  increase). Near rheobase the membrane sits a long time just below threshold where
  noise dominates the drift, so the rarest symbols are encoded by long,
  noise-jittered latencies. The exactly-representable band is therefore an *interior*
  window of $q$: bounded above by drive saturation (a), below by timing noise (b).
]

= Acceptance

#accept_table((
  (true, [calibration $t^*(q) = -lambda log_2 q$ exact to $approx delta t$ (max $1.00 delta t$)]),
  (true, [drive table matches the paper ($q = 0.9 arrow.r 7.09times$ rheobase)]),
  (true, [$q_max in [0.85, 0.95]$ for a $10times$ ceiling ($q_max = 0.930$)]),
  (true, [symbols above $q_max$ saturate to a floored latency]),
  (true, [latency noise (CV) grows as $q arrow 0$ ($0.008 arrow 0.110$)]),
))

#finding[
  *5/5 passed.* The central exact identity of the paper — *spike time is surprisal*
  — holds in a real Nengo LIF to the timing grid, and the three honesty checks the
  paper flags as prose become measured curves: a drive ceiling that pins $q_max =
  0.93$, a noise-dominated rare-symbol tail, and an $O(delta t)$ resolution floor.
  Tier 0 establishes that the substrate honours the calibration; the higher tiers
  build the predictor, the decoder, and the learning on top of it.
]
