#import "../report_style.typ": *
#show: setup

#report_header("e01", "The LIF first-spike latency law, in real spikes",
  "Tier 0 · single-neuron primitive · paper claim #1 (the charging law)")

#claim[
  A leaky integrate-and-fire (LIF) neuron driven by a constant current charges as
  $V(t) = R I (1 - e^(-t slash tau))$ and fires its first spike at latency
  $t^*(I) = tau ln(R I slash (R I - theta))$ (paper §2, eq. for $t^*$). Before any
  calibration: does Nengo's *actual* spiking LIF obey this law, and what does the
  real substrate add that the pure-`numpy` validator — one clean ODE at Euler
  $delta t = 10^(-5)$, started exactly at $V(0)=0$ — never sees?
]

= What we built

A single `nengo.LIF` neuron ($tau_(r c) = 20$ ms, $tau_("ref") = 2$ ms, threshold
normalized to $1$), driven by a constant current $J = R I$ injected *directly into
the neuron* (`Connection` into `ens.neurons`, gain $1$, bias $0$, so the input
current equals $J$ in threshold units). We sweep $J in [1.05, 12]$, probe the spike
train and the membrane voltage, and record the first two spike times. All code is
`e01_lif_latency/run.py`; it imports the shared `spikecoder` package.

= Two spiking-reality facts the numpy validator cannot see

== (1) Nengo randomizes the initial membrane voltage

#gap[
  Nengo initializes each LIF's membrane voltage *uniformly in $[0, 1)$* to
  desynchronize a population. At $J = 2.0$ the analytic first-spike latency from
  rest is $13.86$ ms, but across 12 random seeds the measured first spike scatters
  over $[2.00, 13.80]$ ms — a neuron that happens to start near threshold fires
  almost immediately. For a *first-spike latency code*, where the spike *time* is
  the message, this randomization is fatal. We pin the initial voltage to rest
  ($V(0)=0$) via `initial_state={"voltage": Choice([0])}`; the pinned neuron lands
  at $13.90$ ms, matching the analytic $13.86$ ms to within one $delta t$.
]

#figure(image("results/e01_init_voltage.pdf", width: 78%),
  caption: [Pinning the initial voltage. Grey: first-spike latency under Nengo's
    default randomized $V(0)$ (12 seeds), scattered across the whole sub-threshold
    band. Blue square: pinned $V(0)=0$, on the analytic value (red).])

#intuition[
  In the coder, every symbol window *resets* the readout layer to rest before
  integrating the next symbol. That reset is exactly this pin: each readout fires
  its first spike from $V=0$, so the latency is a clean function of the drive and
  nothing else.
]

== (2) The first spike from rest carries no refractory delay

#finding[
  The refractory period $tau_("ref")$ is *post-spike dead time*: it delays the
  *inter-spike interval*, not the first spike from rest. The measured first-spike
  latency matches $tau_(r c) ln(J slash (J-1))$ (with *no* $tau_("ref")$ term) to a
  maximum of $0.91 delta t$ across the sweep, whereas the "$tau_("ref") + tau_(r c)
  ln(dot)$" law misses by $approx 19.6 delta t$. The inter-spike interval, in
  contrast, matches $tau_("ref") + tau_(r c) ln(J slash (J-1))$ to $0.042$ ms. A
  least-squares fit recovers $tau_(r c) = 20.018$ ms (true $20.0$ ms).
]

#figure(image("results/e01_latency_law.pdf", width: 80%),
  caption: [First-spike latency vs drive. Nengo's measured first spikes (blue) lie
    on $tau_(r c) ln(J slash (J-1))$ (red, *no* refractory term), not on the
    inter-spike-interval law (grey dashed, $+tau_("ref")$).])

This is a *cleaner* result than the project plan hypothesized. The plan expected a
constant $tau_("ref")$ offset to appear and to *realize* the paper's predicted
minimum-latency floor. It does not: because the coder resets to rest per symbol,
*the latency code is refractory-immune*. The paper's idealized law (which has no
refractory term) is therefore matched *exactly* by Nengo's first-spike-from-rest
latency, with $tau equiv tau_(r c)$. The minimum-latency floor that does exist comes
from drive saturation and timing resolution, not from $tau_("ref")$ (developed in
e02).

== (3) The timing-resolution floor

#figure(image("results/e01_dt_quantization.pdf", width: 70%),
  caption: [The latency-error floor tracks the simulation timestep $delta t$: the
    spike can only land on the time grid, so the residual is $O(delta t)$, not the
    $O(10^(-5))$ Euler error of the numpy validator. This is the paper's
    finite-timing-resolution honesty check made literal.])

#gap[
  The numpy validator's only error was its Euler step ($1.56 times 10^(-5)$). Here
  the dominant error is the *clock*: a spike is recorded on the nearest $delta t$
  grid point, so the latency error is $O(delta t)$ ($0.037$ ms at $delta t =
  10^(-4)$, $0.54$ ms at $delta t = 10^(-3)$). This is precisely the
  timing-resolution penalty the paper flags as the replacement for the discrete
  code's integer-bit penalty — and it is the reason latency-critical experiments
  use $delta t = 10^(-4)$.
]

= Acceptance

#accept_table((
  (true, [first spike matches $tau_(r c) ln(J slash (J-1))$ within $approx delta t$ (max $0.91 delta t$)]),
  (true, [first spike does *not* carry $tau_("ref")$ (the offset law misses by $approx 19.6 delta t$)]),
  (true, [recovered $tau_(r c) = 20.018$ ms within $2%$ of the true $20.0$ ms]),
  (true, [inter-spike interval matches $tau_("ref") + tau_(r c) ln(dot)$ (refractory in the 2nd spike)]),
  (true, [randomized $V(0)$ spreads the first spike over $> 5 delta t$ (motivates pinning)]),
  (true, [latency-error floor decreases with $delta t$ (timing-resolution penalty)]),
))

#finding[
  *6/6 passed.* Nengo's real spiking LIF obeys the paper's first-spike latency law
  exactly (to the timing grid), once the initial voltage is pinned to rest. The
  refractory period does not corrupt the latency code, and the residual error is the
  honest timing-resolution penalty $O(delta t)$. This single neuron is the
  foundation every later experiment stands on.
]
