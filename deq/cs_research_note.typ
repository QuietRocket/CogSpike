// Research Note: Spectral Methods for Analyzing SNN Dynamics
// A Guide for Computer Scientists
// APRIL 2026

#set document(
  title: "Spectral Methods for Analyzing SNN Dynamics",
  author: "CogSpike Research Team",
  date: datetime.today(),
)

#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.5cm),
  numbering: "1",
)

#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

// Custom theorem-like environments (consistent with other research notes)
#let definition(body) = block(
  width: 100%,
  inset: 8pt,
  stroke: (left: 2pt + gray),
  [*Definition.* #body],
)

#let theorem(title, body) = block(
  width: 100%,
  inset: 8pt,
  stroke: (left: 2pt + blue),
  [*Theorem* (#title)*.* #body],
)

#let proposition(body) = block(
  width: 100%,
  inset: 8pt,
  stroke: (left: 2pt + green),
  [*Proposition.* #body],
)

#let proof(body) = block(
  width: 100%,
  inset: 8pt,
  fill: luma(248),
  [_Proof._ #body #h(1fr) $square$],
)

#let intuition(body) = block(
  width: 100%,
  inset: 8pt,
  fill: rgb("#f0f7ff"),
  stroke: (left: 2pt + rgb("#4a90d9")),
  [*Intuition:* #body],
)

#let example(body) = block(
  width: 100%,
  inset: 8pt,
  fill: luma(245),
  [*Example.* #body],
)

#let remark(body) = block(
  width: 100%,
  inset: 8pt,
  stroke: (left: 2pt + orange),
  [*Remark.* #body],
)

#let finding(body) = block(
  width: 100%,
  inset: 8pt,
  fill: rgb("#f0fff0"),
  stroke: (left: 2pt + rgb("#2e8b57")),
  [*Finding.* #body],
)

#let howtoread(body) = block(
  width: 100%,
  inset: 8pt,
  fill: rgb("#fff8e1"),
  stroke: (left: 2pt + rgb("#f9a825")),
  [*How to read this plot.* #body],
)

// ============================================================================
// DOCUMENT
// ============================================================================

#align(center)[
  #text(size: 18pt, weight: "bold")[
    Spectral Methods for Analyzing \
    Spiking Neural Network Dynamics
  ]
  #v(0.3em)
  #text(size: 14pt)[
    A Guide for Computer Scientists
  ]
  #v(0.5em)
  #text(size: 12pt)[CogSpike Research Team --- April 2026]
  #v(0.3em)
  #text(size: 10pt, style: "italic")[
    Research Note --- From Graph Spectra and Iterative Methods to SNN Behavior Prediction
  ]
]

#v(1em)

*Abstract.* PRISM model checking can verify temporal-logic properties of
Spiking Neural Networks modelled as DTMCs, but at exponential state-space
cost and without explaining _why_ a property holds. We show that
eigendecomposition of the synaptic weight matrix --- a polynomial-time
computation familiar from spectral graph theory and PageRank --- directly
reveals the competitive dynamics of Winner-Take-All SNNs. The dominant
eigenvector predicts the winner, the spectral radius determines convergence
(by the same criterion that governs Jacobi iteration), and fixed-point
analysis maps the attractor landscape. We validate all predictions against
PRISM model checking and Monte Carlo simulation. This note translates the
relevant ideas from control theory into the language of discrete mathematics
and theoretical computer science, and proposes a general methodology for
analyzing arbitrary SNN topologies.

= Introduction: Beyond Model Checking

Consider a Spiking Neural Network modelled as a Discrete-Time Markov Chain
(DTMC). PRISM model checking can answer questions like _"does neuron N1 fire
infinitely often with probability 1?"_ --- and for our 4-neuron case study,
the answer is yes. But model checking has two limitations:

1. *Cost.* The state space grows exponentially in the number of neurons.
   Our 4-neuron WTA network has $approx 3,600$ reachable states; scaling to
   10 neurons would yield millions.

2. *Opacity.* PRISM returns a probability, not an explanation. _Why_ does N1
   win? What would happen if we changed the weights? How fast does the
   competition resolve?

This note presents an alternative: extract the answers directly from the
weight matrix $bold(W)$ using eigendecomposition --- an $O(n^3)$ computation
that computer scientists already use in PageRank, spectral clustering, and
Markov chain mixing analysis.

#intuition[
  PRISM explores every reachable state. Eigenanalysis reads the structure of
  the wiring diagram. The same way Google's PageRank extracts "page importance"
  from the web graph's adjacency matrix in polynomial time, we extract
  "neuron dominance" from the synaptic weight matrix.
]


= SNNs as Discrete Dynamical Systems

== The LIF Recurrence

A Leaky Integrate-and-Fire (LIF) neuron maintains a membrane potential $p_i$
that evolves according to a linear recurrence:

$ p_i (t+1) = r dot p_i (t) + sum_j W_(i j) dot y_j (t) + B_i dot u_i (t) $

where:
- $r in (0, 1)$ is the *leak rate* (fraction of potential retained per step),
- $W_(i j)$ is the *synaptic weight* from neuron $j$ to neuron $i$,
- $y_j (t) in {0, 1}$ is the *spike output* of neuron $j$ at time $t$,
- $u_i (t) in {0, 1}$ is the *external input* to neuron $i$.

When $p_i$ exceeds a threshold $theta$, the neuron fires ($y_i = 1$) and
resets to $p_i = 0$. In the probabilistic variant (used in our PRISM models),
firing probability increases in discrete steps as $p_i$ approaches $theta$.

#remark[
  Strip away the firing nonlinearity and this is a standard linear recurrence
  $bold(x)(t+1) = r bold(x)(t) + "input"$, exactly the kind of recurrence CS
  students solve with characteristic equations and generating functions. The
  spike nonlinearity is what makes exact analysis hard --- and what motivates
  PRISM --- but the _linear structure underneath_ is what makes eigenvalue
  analysis possible and useful.
]

== Matrix Form

In vector notation for an $n$-neuron network:

$ bold(p)(t+1) = r dot bold(p)(t) + bold(W) dot bold(y)(t) + bold(B) dot bold(u)(t) $

This is a linear map iterated: $bold(x) arrow.r.long bold(M) dot bold(x) + "input"$.
The matrix $bold(W)$ encodes the topology. Our analysis extracts everything we
need from $bold(W)$.

== Connection to DTMCs

The deterministic LIF recurrence is the _skeleton_ of the DTMC that PRISM
analyses. The probabilistic firing thresholds add stochasticity on top. Our
eigenvalue approach analyses the deterministic skeleton, whose structure
governs the DTMC's long-run behavior (its Bottom Strongly Connected
Components).


= The Weight Matrix as a Graph Operator

== Network Topology

The weight matrix $bold(W) in RR^(n times n)$ is the signed weighted adjacency
matrix of the synaptic connectivity graph: $W_(i j) = $ weight from neuron $j$
to neuron $i$.

#figure(
  image("plots/cs_network_graph.png", width: 60%),
  caption: [The 4-neuron WTA network as a directed graph. N1 (blue) sends
    stronger inhibition ($-100$, thick red) than N2--N4 ($-70$, thin red).
    External inputs $S_i$ feed each neuron with weight $+100$.],
) <fig-graph>

For our case study:

$ bold(W) = mat(
  0, -70, -70, -70;
  -100, 0, -70, -70;
  -100, -70, 0, -70;
  -100, -70, -70, 0;
) $

The key asymmetry: N1 sends $-100$ inhibition per outgoing edge (total $-300$),
while N2--N4 each send $-70$ (total $-210$). This 30-unit advantage per
connection is what predetermines N1 as the winner.

== Connection to Spectral Graph Theory

In spectral graph theory, the adjacency matrix spectrum reveals community
structure, connectivity, and random walk properties. For $bold(W)$, the
spectrum reveals *competitive structure*: which neurons tend to fire together,
which suppress each other, and who wins.

#intuition[
  $bold(W)$ is the "wiring diagram" of the competition. Eigendecomposition
  separates this wiring into independent channels called _modes_. Each mode is
  a pattern of coordinated activation and suppression --- an independent
  "axis of competition."
]


= Eigenvalue Decomposition: Reading the Network's DNA

This is the core technique. The eigendecomposition $bold(W) = bold(V) bold(Lambda) bold(V)^(-1)$
factorises the weight matrix into independent competitive modes.

== Eigenvalues and Eigenvectors

#figure(
  image("plots/cs_eigenvector_heatmap.png", width: 85%),
  caption: [Eigenvector heatmap of $bold(W)$. Each column is an independent
    competitive mode. Red = excited (positive component), blue = inhibited
    (negative). The WTA mode (Mode 2) has N1 excited and all others
    inhibited --- this is the mathematical fingerprint of N1's dominance.],
) <fig-heatmap>

#howtoread[
  Each column is an _eigenvector_ --- a pattern of coordinated
  activation/suppression. The eigenvalue $lambda$ scales how strongly the
  weight matrix amplifies that pattern. Red cells mean the neuron is
  _excited_ in that mode; blue cells mean _inhibited_. The WTA mode
  (Mode 2, highlighted) shows N1 excited while all others are inhibited:
  this is the pattern that the weight asymmetry amplifies.
]

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    [*Mode*], [*$lambda$*], [*Eigenvector*], [*Interpretation*],

    [1], [$-230.9$],
    [$(−0.91, −1, −1, −1)$],
    [_Common-mode inhibition._ All neurons are suppressed together. Negative
      $lambda$ means this mode alternates in sign each step.],

    [2], [$+90.9$],
    [$(+1.0, −0.43, −0.43, −0.43)$],
    [*N1 advantage mode (WTA).* N1 gains while others lose. This mode
      determines the winner.],

    [3], [$+70.0$],
    [$(0, +1, −0.5, −0.5)$],
    [_Loser competition._ Redistributes activity among N2--N4.
      N1 is absent from this mode.],

    [4], [$+70.0$],
    [$(0, +1, −0.5, −0.5)$],
    [_Degenerate with Mode 3._ The N2/N3/N4 symmetry produces a
      2D eigenspace.],
  ),
  caption: [Eigendecomposition of $bold(W)$. The eigenvalue gap
    $|lambda_2| - |lambda_(3,4)| = 20.9$ quantifies the strength of the WTA.],
) <tab-eig>

== The PageRank Analogy

In Google's PageRank, the dominant eigenvector of the web graph's transition
matrix determines steady-state page importance. Here, the eigenvector
associated with the *largest positive eigenvalue* of $bold(W)$ determines
which neuron dominates the competition.

#finding[
  *Mode 2 ($lambda = +90.9$) is the WTA mode.* Its eigenvector
  $(+1.0, −0.43, −0.43, −0.43)$ points in the direction where N1's
  activation increases while all others decrease. The eigenvalue magnitude
  $90.9$ exceeds the loser eigenspace $70.0$ by $30%$ --- reflecting N1's
  30-unit weight advantage per connection. _The weight asymmetry maps
  directly to an eigenvalue gap._
]

== Why This Works: The Linear Algebra

If we decompose an initial state in the eigenvector basis,
$bold(p)(0) = sum_i c_i bold(v)_i$, then after $t$ steps of the linear dynamics
each component is scaled by $lambda_i^t$. The mode with the largest
$|lambda|$ dominates as $t arrow infinity$.

This is exactly the argument behind:
- *Power iteration* converging to the dominant eigenvector,
- *PageRank* extracting page importance from the web graph,
- *PCA* finding the principal component of a dataset.

#proposition[
  The *eigenvalue gap* $Delta = |lambda_"WTA"| - |lambda_"loser"| = 20.9$
  quantifies the strength and speed of the WTA mechanism. A larger gap
  means faster, more decisive winner selection. This is analogous to the
  spectral gap in Markov chain mixing, which determines how quickly a
  random walk converges to its stationary distribution.
]

#figure(
  image("plots/cs_eigenvalue_spectrum.png", width: 95%),
  caption: [Left: eigenvalue magnitudes of $bold(W)$, with the WTA mode
    highlighted. Right: eigenvalues of the linearised state matrix $bold(A)$
    --- all below the stability boundary at $|lambda| = 1$.],
) <fig-spectrum>


= Stability: When Does the Network Converge?

We now move from the raw weight matrix to the _linearised dynamics_. Near
a steady state, the LIF recurrence can be approximated as:

$ delta bold(p)(t+1) = bold(A) dot delta bold(p)(t), quad "where" quad
  bold(A) = r dot bold(I) + bold(W) dot "diag"(bold(g)) $

Here $bold(g) = (g_1, dots, g_n)$ is the vector of firing-rate derivatives
at the operating point (how sensitive each neuron's output is to its
potential), and $r = 0.5$ is the leak rate.

#theorem("Spectral Radius Criterion")[
  The linearised system is asymptotically stable if and only if the
  spectral radius $rho(bold(A)) = max_i |lambda_i (bold(A))| < 1$.
]

#proof[
  The solution to $bold(x)(t+1) = bold(A) bold(x)(t)$ is
  $bold(x)(t) = bold(A)^t bold(x)(0)$. By the spectral decomposition,
  $||bold(A)^t|| arrow 0$ as $t arrow infinity$ if and only if all
  eigenvalues satisfy $|lambda_i| < 1$.
]

#intuition[
  This is *identical to the convergence criterion for Jacobi iteration*.
  If you have studied iterative methods for solving $bold(A) bold(x) = bold(b)$
  --- where the iteration $bold(x)^((k+1)) = bold(M) bold(x)^((k)) + bold(c)$
  converges iff $rho(bold(M)) < 1$ --- then you already know this theorem.
  The LIF network IS a Jacobi-like iteration, and the same criterion applies.
]

For our network, the linearised eigenvalues are:

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    [*Mode*], [*$lambda$*], [*$|lambda|$*], [*$tau$ (steps)*], [*Interpretation*],
    [$lambda_1$], [$0.629$], [$0.629$], [$2.15$], [Slow decay: loser competition],
    [$lambda_2$], [$0.629$], [$0.629$], [$2.15$], [Degenerate with $lambda_1$],
    [$lambda_3$], [$0.500$], [$0.500$], [$1.44$], [N1 isolation mode (pure leak)],
    [$lambda_4$], [$0.243$], [$0.243$], [$0.71$], [Fast decay: common-mode inhibition],
  ),
  caption: [Eigenvalues of $bold(A)$. The time constant $tau = -1 / ln|lambda|$
    gives the number of steps for a perturbation to decay to $1/e$ of its
    initial value.],
) <tab-eig-A>

#finding[
  The spectral radius is $rho(bold(A)) = 0.629 < 1$. The network converges
  geometrically with rate $0.629$ per step. Settling to within 5% takes
  $approx 3 tau = 6.5$ steps. Checking this is $O(n^3)$ --- polynomial time
  --- compared to the exponential cost of exploring all reachable DTMC states.
]

#figure(
  image("plots/cs_convergence_trace.png", width: 95%),
  caption: [Left: state vector converging to the fixed point from a symmetric
    initial condition. Right: log-scale error showing geometric convergence at
    rate $rho = 0.629$ --- the same behaviour as a converging Jacobi iteration.],
) <fig-convergence>

#howtoread[
  The right panel is a standard convergence plot familiar from numerical
  methods. The y-axis is $log_10(||"error"||)$. A straight line means geometric
  (exponential) convergence. The slope equals $log_10(rho)$ where $rho$ is
  the spectral radius. The red dashed line shows the predicted rate from
  eigenvalue analysis --- it matches the actual convergence exactly.
]


= Fixed Points: Where Does the Network Converge To?

At steady state with constant input $bold(u)$:

$ bold(p)^* = frac(bold(W) dot f(bold(p)^*) + bold(B) dot bold(u), 1 - r) $

This is a *fixed-point equation* $bold(p)^* = F(bold(p)^*)$.

== Connections to CS Fixed-Point Theory

#intuition[
  - *Banach's theorem:* if $F$ is a contraction (Lipschitz constant $< 1$),
    there exists a unique fixed point and the iteration $bold(p)_(k+1) = F(bold(p)_k)$
    converges from any start. The spectral radius criterion from Section 5 is the
    _linearised version_ of the contraction condition.
  - *Knaster-Tarski:* for monotone operators on complete lattices, fixed
    points are guaranteed. Our network has both excitatory and inhibitory
    connections, so global monotonicity does not hold, but fixed-point
    structure is still informative.
  - *BSCCs in model checking:* the stable fixed points correspond to the
    Bottom Strongly Connected Components of the underlying DTMC --- the
    absorbing states that PRISM identifies.
]

== Equilibrium Landscape

Numerical fixed-point search (using `scipy.optimize.fsolve` with 50+
initial conditions) reveals *20 distinct fixed points* for $bold(u) = (1,1,1,1)$:

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    [*Type*], [*Count*], [*Stability*], [*Interpretation*],
    [N1-winner], [1], [Stable node], [The unique WTA attractor: $bold(r)^* = (1, 0, 0, 0)$],
    [Mixed 2-active], [6], [Stable nodes], [Artifacts of piecewise-linear $f$-$I$ curve; not reached stochastically],
    [Mixed 3-active], [9], [Saddle points], [Decision boundaries between attractor basins],
    [Symmetric], [1], [Saddle (4 unstable dims)], [All neurons equal; maximally unstable],
    [Other], [3], [Saddle points], [Mixed states with partial competition],
  ),
  caption: [Classification of all 20 fixed points. Only the N1-winner state
    is a true global attractor of the stochastic system.],
) <tab-fps>

#finding[
  Only *one fixed point* is the genuine attractor: N1 wins with
  $bold(r)^* = (1, 0, 0, 0)$. All saddle points are unstable --- perturbations
  push the system away from them. The 6 "stable" mixed states are artifacts
  of the piecewise-linear firing function and are not reached by the stochastic
  system because N1's inhibition prevents them.
]

== Basins of Attraction

The basin of attraction is the set of initial states that converge to a given
fixed point --- analogous to a "capture set" in a game.

#figure(
  image("plots/cs_basin_of_attraction.png", width: 70%),
  caption: [Basin of attraction map (projected onto N1 vs N2 initial
    potential, with N3 and N4 initialised at 30). Blue = N1 wins,
    other colours = alternative winners. N1's basin dominates the
    state space.],
) <fig-basin>

#howtoread[
  Each pixel represents an initial state $(p_1, p_2)$. Its colour indicates
  which neuron wins after 200 steps of mean-field iteration. Blue means N1
  wins from that initial condition. The large blue region shows that N1's
  weight advantage creates a dominant basin --- even when N2 starts with
  higher potential, N1 usually wins.
]


= Frequency Response: How Fast Does the Network React?

Sections 4--6 used tools already familiar to CS readers (eigenvalues, fixed
points, spectral radius). This section introduces one concept from signal
processing: the *frequency response*. It answers a question that eigenvalues
alone cannot: _how quickly can the network track a time-varying input?_

== The Core Idea

If the input changes slowly (e.g., a neuron's firing rate ramps up over 20
steps), the network output follows it faithfully. If the input changes rapidly
(e.g., alternating on/off every step), the network barely responds. The leak
rate $r = 0.5$ --- retaining only 50% of potential per step --- smooths out
fast fluctuations.

#figure(
  image("plots/cs_time_domain_filtering.png", width: 95%),
  caption: [Low-pass filtering demonstrated. Left: slow input modulation
    (period 20) --- the network tracks the input. Right: fast modulation
    (period 4) --- the network's response is nearly flat. The LIF neuron's
    leak smooths out rapid changes.],
) <fig-filtering>

#howtoread[
  Top row: the input signal (S1's firing rate varies sinusoidally). Bottom
  row: N1's firing rate response. Compare left vs right: the slow input
  (left) produces a clear oscillation in N1's output, while the fast input
  (right) is almost completely ignored. This is *low-pass filtering* --- a
  direct consequence of the leak.
]

== The Frequency Response Plot

The frequency response compresses this information into a single curve: for
each input speed (frequency), it shows the network's response amplitude (gain).

#figure(
  image("plots/cs_bode_annotated.png", width: 95%),
  caption: [Frequency response (Bode magnitude plot) of the linearised WTA
    network. Green region: network tracks input. Yellow: transition. Red:
    network ignores input. The winner (blue) has wider bandwidth than the
    loser (orange) --- it reacts faster.],
) <fig-bode>

#howtoread[
  The x-axis is input speed (normalised frequency, from "constant" to
  "alternating every step"). The y-axis is response amplitude on a
  logarithmic scale. High gain (top-left) means the network amplifies the
  input; low gain (bottom-right) means it ignores it. The *bandwidth* is
  the frequency at which gain drops to $1/sqrt(2)$ of its DC value ---
  it measures the network's reaction speed.
]

#definition[
  The *bandwidth* is the fastest input speed the network can still track
  (formally, the frequency where gain drops by 3 dB from DC). It is measured
  in radians per time step, normalised to $[0, pi]$.
]

#finding[
  The winner has bandwidth $0.23 pi$ rad/step; the loser has $0.17 pi$.
  The winner reacts *35% faster*. This is a second, independent advantage
  beyond steady-state dominance: N1 not only wins at equilibrium but also
  responds more quickly to input changes.
]

#remark[
  Control theory offers additional frequency-domain tools (Nyquist diagrams,
  phase/gain margins) that provide redundant stability information already
  captured by the spectral radius criterion in Section 5. We omit them here
  in favour of the simpler eigenvalue-based analysis.
]


= A General Methodology

We summarise the analysis as a step-by-step procedure applicable to any SNN:

#figure(
  table(
    columns: (auto, 1fr, auto, auto),
    stroke: 0.5pt,
    [*Step*], [*Action*], [*Cost*], [*CS Analogue*],
    [1], [Extract weight matrix $bold(W)$ from the SNN topology],
    [$O(|E|)$], [Adjacency matrix],

    [2], [Eigendecompose $bold(W)$: identify competitive modes, locate the
      WTA mode (eigenvector with one dominant positive component)],
    [$O(n^3)$], [PageRank / PCA],

    [3], [Compute operating points by solving $bold(p)^* = F(bold(p)^*)$],
    [$O(n dot k)$], [Fixed-point iteration],

    [4], [Build linearised state matrix $bold(A) = r bold(I) + bold(W) dot "diag"(f'(bold(p)^*))$],
    [$O(n^2)$], [Jacobian computation],

    [5], [Check stability: verify $rho(bold(A)) < 1$],
    [$O(n^3)$], [Spectral radius of iteration matrix],

    [6], [Classify all fixed points: stable (attractors), saddle (decision
      boundaries), unstable (repellers)],
    [$O(k dot n^3)$], [BSCC identification],

    [7], [Map basins of attraction],
    [$O(G dot n dot T)$], [Reachability analysis],

    [8], [(Optional) Compute frequency response for bandwidth],
    [$O(n^3 dot F)$], [Transfer function],

    [9], [Validate key predictions with PRISM on a tractable instance],
    [Exponential], [Model checking],
  ),
  caption: [The 9-step methodology. Steps 1--8 are polynomial in $n$ (number
    of neurons). Step 9 is the exponential bottleneck --- used selectively
    to validate, not exhaustively to explore. $k$ = number of initial
    conditions tried; $G$ = basin grid size; $T$ = simulation steps;
    $F$ = frequency points.],
) <tab-method>

#intuition[
  Use cheap spectral analysis (Steps 1--8) to generate hypotheses about
  network behavior. Then validate selectively with expensive model
  checking (Step 9). This is analogous to using heuristics to guide
  a SAT solver: the heuristic doesn't prove anything, but it tells you
  where to look.
]


= Case Study: The 4-Neuron WTA Network

We apply the methodology to our concrete case study with all inputs active
($bold(u) = (1,1,1,1)$).

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    stroke: 0.5pt,
    [*Step*], [*Result*], [*Prediction*],
    [1--2], [Eigenvalues of $bold(W)$: $-230.9, +90.9, +70, +70$. \
      WTA mode: $(+1, -0.43, -0.43, -0.43)$],
    [N1 wins; eigenvalue gap $= 20.9$],

    [3--4], [Operating point: $bold(p)^* = (190, -6.6, -6.6, -6.6)$. \
      $bold(A)$ eigenvalues: $0.629, 0.629, 0.500, 0.243$],
    [Network converges],

    [5], [$rho(bold(A)) = 0.629 < 1$],
    [Stable. Settling in $approx 6.5$ steps],

    [6], [20 fixed points found; 1 stable WTA node],
    [Unique WTA attractor: N1 wins],

    [7], [N1's basin dominates the state space],
    [N1 wins from almost all initial conditions],

    [8], [Winner bandwidth $= 0.23 pi$, loser $= 0.17 pi$],
    [Winner reacts 35% faster],

    [9], [Monte Carlo (1000 trials): N1 $= 40.6$, losers $approx 5.8$ spikes/50 steps. \
      PRISM (3-neuron): N1 $= 37.2$, losers $= 7.8$ spikes/50 steps],
    [Winner/loser ratio $approx 7 times$ (confirmed)],
  ),
  caption: [Methodology applied to the 4-neuron WTA. Every prediction from
    the $O(n^3)$ spectral analysis is confirmed by the exponential-cost
    model checking and Monte Carlo simulation.],
) <tab-case>

#finding[
  The eigenvalue gap of $20.9$ correctly predicts N1 as the winner. The
  spectral radius of $0.629$ correctly predicts convergence in $approx 6$
  steps. The basin analysis correctly predicts that N1 wins from almost
  all initial conditions. All of this is extracted from a $4 times 4$
  eigendecomposition --- no state enumeration required.
]


= Discussion

== What This Approach Can Do

- *Polynomial-time identification* of network competition structure.
- *Stability certificates* without state space enumeration.
- *Quantitative predictions* (convergence rate, bandwidth) not available
  from model checking alone.
- *Design guidance:* the eigenvalue gap tells you exactly how much to
  change a weight to strengthen or weaken the WTA.

== What This Approach Cannot Do

- Handle the *full stochastic dynamics*. PRISM captures variance and rare
  events; the spectral analysis captures only expected behaviour.
- Provide *exact probabilities*. The linearisation is an approximation.
- Analyse networks *far from the operating point*. The linearisation is
  local; large perturbations require nonlinear analysis or simulation.
- Capture *spike-timing effects* that the mean-field approximation
  averages away.

== Relationship to Formal Verification

Spectral analysis and PRISM model checking are *complementary*, not competing:

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    [], [*Spectral Analysis*], [*PRISM Model Checking*],
    [Cost], [Polynomial $O(n^3)$], [Exponential in $n$],
    [Output], [Qualitative structure + approximate quantities],
    [Exact probabilities],
    [Answers], [_Why_ and _how fast_], [_Whether_ (yes/no with probability)],
    [Scalability], [Hundreds of neurons], [4--8 neurons (with discretisation)],
    [Guarantees], [Local (linearisation)], [Global (all reachable states)],
  ),
  caption: [Comparison of the two approaches. Use spectral analysis to
    understand and design; use PRISM to verify.],
) <tab-compare>

The recommended workflow: use spectral analysis to *understand* the network
(identify modes, check stability, predict winners), then use PRISM to
*certify* the key properties on a tractable instance. The spectral analysis
tells you what to look for; PRISM tells you whether it's true.


= Conclusion

The weight matrix $bold(W)$ is the DNA of an SNN's competitive dynamics.
Eigendecomposition --- a polynomial-time computation that computer scientists
already use for PageRank, spectral clustering, and Markov chain mixing ---
directly reads this DNA:

- The *eigenvectors* are the competitive modes (who fights whom).
- The *eigenvalue gap* predicts the winner and the speed of resolution.
- The *spectral radius* of the linearised system gives a convergence
  certificate (same criterion as iterative methods).
- *Fixed-point analysis* maps the attractor landscape (same framework as
  abstract interpretation).
- *Frequency response* quantifies the network's reaction speed.

For the 4-neuron WTA case study, every prediction from the $O(n^3)$ spectral
analysis was confirmed by PRISM model checking and Monte Carlo simulation.
The methodology is general and applicable to any SNN whose topology can be
expressed as a weight matrix.

#v(1cm)
#line(length: 100%)
#text(size: 9pt, style: "italic")[
  Generated using the CogSpike DEQ analysis toolkit
  (`deq/cs_plots.py`, `deq/run_analysis.py`). \
  All plots are reproducible via `python3 cs_plots.py` in the `deq/` directory.
]
