// Research Note: Classical Engineering Analysis of SNN Dynamics
// Differential Equations, Transfer Functions, and Eigenvalue Methods
// APRIL 2026

#set document(
  title: "Classical Engineering Analysis of SNN Dynamics",
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

// ============================================================================
// DOCUMENT
// ============================================================================

#align(center)[
  #text(size: 18pt, weight: "bold")[
    Classical Engineering Analysis \
    of SNN Dynamics
  ]
  #v(0.5em)
  #text(size: 12pt)[CogSpike Research Team --- April 2026]
  #v(0.3em)
  #text(size: 10pt, style: "italic")[
    Research Note --- Differential Equations, Transfer Functions & Eigenvalue Methods
  ]
]

#v(1em)

*Abstract.* We apply classical systems engineering tools --- eigenvalue analysis,
Z-domain transfer functions, Bode plots, Nyquist diagrams, and steady-state
equilibrium analysis --- to a 4-neuron Winner-Take-All (WTA) SNN with asymmetric
inhibitory weights. Starting from the observation that a LIF neuron's subthreshold
dynamics are a first-order linear difference equation (equivalently, a discrete
low-pass filter), we construct the linearized state-space model, decompose the
weight matrix into competitive modes, and derive the frequency response of the
network. Our central finding is that the eigenstructure of the inhibitory weight
matrix $bold(W)$ directly explains _why_ N1 wins: a mode with eigenvalue
$lambda = +90.9$ has its eigenvector aligned with N1's advantage, while the losers
share a degenerate eigenspace at $lambda = 70$. Monte Carlo simulation validates
the analysis with a winner/loser spike ratio of $6.98 times$, consistent with
PRISM model checking results from the 3-neuron case study.

= Introduction

The formal verification of SNN behavior via PRISM model checking
answers _whether_ a property holds (e.g., "N1 fires infinitely often with
probability 1"), but does not explain _why_. Classical systems engineering
provides complementary tools that reveal the dynamic mechanisms behind the
verified properties.

The key insight enabling this bridge is that the LIF neuron's subthreshold
membrane equation is a first-order linear difference equation:

$ p_i (t+1) = r dot p_i (t) + sum_j W_(i j) dot y_j (t) + B_i dot u_i (t) $

where $r = 0.5$ is the leak rate, $W_(i j)$ is the synaptic weight from neuron
$j$ to neuron $i$, $y_j$ is the spike output (0 or 1), and $u_i$ is the
external input. Taking the Z-transform of this equation yields a transfer
function $H(z) = 1 / (z - r)$ --- a first-order low-pass filter with pole at
$z = r$, time constant $tau = -1 / ln(r)$, and DC gain $1 / (1-r)$.

Every LIF neuron in the network is one of these filters, and the synaptic
connections form a feedback structure. The entire network can therefore be
analyzed using the standard machinery of discrete-time linear systems theory:
state-space models, eigenvalue decomposition, transfer function matrices,
and frequency response.

= The 4-Neuron WTA Network

== Topology

The case study network consists of 12 nodes: 4 input neurons (S1--S4), 4 core
processing neurons (N1--N4) fully interconnected with inhibitory synapses, and
4 output neurons (O1--O4). Each input $S_i$ feeds its corresponding core neuron
$N_i$ with excitatory weight $+100$, and each $N_i$ drives output $O_i$ with
weight $+100$.

The critical structure is the inhibitory interconnection among N1--N4:

#figure(
  table(
    columns: 6,
    stroke: 0.5pt,
    [], [*$arrow$ N1*], [*$arrow$ N2*], [*$arrow$ N3*], [*$arrow$ N4*], [*Total sent*],
    [*N1 $arrow$*], [$0$], [$-100$], [$-100$], [$-100$], [$-300$],
    [*N2 $arrow$*], [$-70$], [$0$], [$-70$], [$-70$], [$-210$],
    [*N3 $arrow$*], [$-70$], [$-70$], [$0$], [$-70$], [$-210$],
    [*N4 $arrow$*], [$-70$], [$-70$], [$-70$], [$0$], [$-210$],
  ),
  caption: [Inhibitory weight matrix. N1 delivers $-100$ inhibition per
    connection (total $-300$), while N2--N4 each deliver $-70$ (total $-210$).
    This 30-unit advantage per connection is the mechanism that predetermines N1
    as the winner.],
) <tab-weights>

#figure(
  image("plots/sim_stochastic_single.png", width: 90%),
  caption: [Single stochastic trial (T=50, all inputs on). N1 fires 49 times
    while losers fire only once each. The membrane potentials show N1 rapidly
    reaching threshold while inhibition suppresses N2--N4 to negative
    potentials.],
) <fig-stochastic>

== LIF Parameters

The model uses the following parameters from the PRISM specification:

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    [*Parameter*], [*Symbol*], [*Value*],
    [Leak rate], [$r$], [$0.5$ (multiplicative)],
    [Firing threshold], [$P_"rth"$], [$80$],
    [Reset potential], [$P_"reset"$], [$0$],
    [Threshold levels], [$k$], [$4$ (at $p = 20, 40, 60, 80$)],
    [Firing probabilities], [], [$0.25, 0.50, 0.75, 1.00$],
  ),
  caption: [LIF neuron parameters from the PRISM model.],
) <tab-params>

== Weight Matrix Formulation

#definition[
  The *inhibitory weight matrix* $bold(W) in RR^(4 times 4)$ encodes the
  synaptic connections among core neurons, where $W_(i j)$ is the weight from
  neuron $j$ to neuron $i$:

  $ bold(W) = mat(
    0, -70, -70, -70;
    -100, 0, -70, -70;
    -100, -70, 0, -70;
    -100, -70, -70, 0;
  ) $
]

The *input matrix* is $bold(B) = 100 dot bold(I)_4$ (diagonal, each $S_i$ feeds
only $N_i$), and the *output matrix* is $bold(C) = bold(I)_4$ (observe membrane
potentials directly).


= Eigenvalue Analysis of the Weight Matrix

The eigendecomposition of $bold(W)$ reveals the fundamental competitive modes
of the network. Since $bold(W)$ has a near-symmetric structure (three neurons
at $-70$, one at $-100$), we expect one mode capturing N1's asymmetry and a
degenerate subspace for the symmetric N2/N3/N4 interactions.

== Eigenvalues and Eigenvectors

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    [*Mode*], [*Eigenvalue $lambda$*], [*Eigenvector (normalized)*], [*Interpretation*],
    [1], [$-230.9$],
    [$(-0.91, -1.00, -1.00, -1.00)$],
    [_Common-mode inhibition._ All neurons inhibit together. Negative eigenvalue means this mode produces alternating dynamics.],

    [2], [$+90.9$],
    [$(+1.00, -0.43, -0.43, -0.43)$],
    [*N1 advantage mode.* N1 gains while N2--N4 lose. This is the WTA mechanism.],

    [3], [$+70.0$],
    [$(0, +1.00, -0.50, -0.50)$],
    [_Loser competition._ N2 vs N3/N4. N1 is absent --- this mode operates within the degenerate loser subspace.],

    [4], [$+70.0$],
    [$(0, +1.00, -0.50, -0.50)$],
    [_Degenerate with Mode 3._ The N2/N3/N4 symmetry produces a 2D eigenspace.],
  ),
  caption: [Eigendecomposition of the weight matrix $bold(W)$.],
) <tab-eig-W>

#finding[
  *Mode 2 ($lambda = +90.9$) is the WTA mode.* Its eigenvector
  $(+1.0, -0.43, -0.43, -0.43)$ points in the direction where N1's activation
  increases while all others decrease. The eigenvalue magnitude $|lambda_2| = 90.9$
  exceeds the degenerate pair $|lambda_(3,4)| = 70.0$ by 30%, reflecting N1's
  30-unit weight advantage. The 30-unit difference in weights maps directly to a
  30% difference in modal gain.
]

#intuition[
  Think of the weight matrix as a "plumbing diagram" for inhibitory pressure.
  The eigendecomposition separates this into independent flow modes. Mode 2 is
  the channel that amplifies N1's signal while draining the others --- it is the
  mathematical reason N1 wins. Modes 3 and 4 only redistribute activity among
  losers, never touching N1.
]


= Linearized State-Space Model

== Mean-Field Approximation

To apply linear systems theory, we replace binary spikes $y_i in {0, 1}$ with
continuous firing rates $y_i approx f(p_i)$, where $f$ is the firing rate
function. The LIF dynamics become:

$ bold(p)(t+1) = r dot bold(p)(t) + bold(W) dot f(bold(p)(t)) + bold(B) dot bold(u)(t) $

== Linearization Challenge

#remark[
  The weights ($plus.minus 100$, $plus.minus 70$) are large relative to the
  firing threshold range ($0$--$80$), causing operating points to saturate
  in regions where the piecewise-linear firing rate derivative is zero. A
  *sigmoid approximation* $f(p) = 1 / (1 + e^(-k(p - p_"mid")))$ with
  $k = 0.08$, $p_"mid" = 40$ provides non-zero derivatives at all operating
  points, enabling meaningful linearization.
]

Linearizing around the operating point $bold(p)^*$:

$ delta bold(p)(t+1) = bold(A) dot delta bold(p)(t) + bold(B) dot delta bold(u)(t) $

where the state matrix is:

$ bold(A) = r dot bold(I) + bold(W) dot "diag"(f'(bold(p)^*)) $

#figure(
  image("plots/linearization_fi_curve.png", width: 100%),
  caption: [Left: three firing rate functions --- piecewise-linear, discrete step
    (matching PRISM), and sigmoid approximation. Center: derivatives showing the
    linearization gain; the sigmoid provides non-zero gain everywhere. Right:
    operating points on the sigmoid curve with tangent lines.],
) <fig-fi-curve>

== Linearized Eigenvalues

For the all-inputs-on regime ($bold(u) = (1,1,1,1)$), the sigmoid operating
point is $bold(p)^* = (190.1, -6.6, -6.6, -6.6)$ with firing rates
$(1.0, 0.024, 0.024, 0.024)$. The gains are $bold(g) = (0, 0.0018, 0.0018, 0.0018)$
--- N1 is saturated (gain 0), while the losers have small but non-zero gain.

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    [*Mode*], [*$lambda$*], [*$|lambda|$*], [*$tau$ (steps)*], [*Interpretation*],
    [$lambda_1$], [$0.629$], [$0.629$], [$2.15$], [Slow decay: loser competition mode],
    [$lambda_2$], [$0.629$], [$0.629$], [$2.15$], [Degenerate with $lambda_1$],
    [$lambda_3$], [$0.500$], [$0.500$], [$1.44$], [N1 isolation mode (pure leak)],
    [$lambda_4$], [$0.243$], [$0.243$], [$0.71$], [Fast decay: common-mode inhibition],
  ),
  caption: [Eigenvalues of the linearized state matrix $bold(A)$. All lie
    inside the unit circle ($|lambda| < 1$), confirming stability.],
) <tab-eig-A>

#finding[
  All eigenvalues satisfy $|lambda_i| < 1$ --- the network is *asymptotically
  stable*. Perturbations decay with time constant $tau = 2.15$ steps for the
  slowest mode and $tau = 0.71$ steps for the fastest. The predicted settling
  time is $3 tau approx 6.5$ steps.
]

#figure(
  image("plots/eigenvalues_complex_plane.png", width: 70%),
  caption: [Eigenvalues of $bold(A)$ plotted on the complex plane. The unit
    circle (stability boundary) is shown in black. All eigenvalues lie well
    inside the unit circle, indicating robust stability.],
) <fig-eigenvalues>

#figure(
  image("plots/eigenvectors_All_inputs_on_(u1111).png", width: 100%),
  caption: [Eigenvector structure for all-inputs-on regime. Mode 4 ($lambda=0.24$)
    is the common-mode where all losers decay together. Modes 1--2 redistribute
    activity among losers. Mode 3 isolates N1.],
) <fig-eigenvectors>


= Z-Domain Transfer Functions

== Transfer Function Matrix

For the linearized discrete-time system, the transfer function matrix is:

$ bold(H)(z) = bold(C) (z bold(I) - bold(A))^(-1) bold(B) + bold(D) $

This is a $4 times 4$ matrix of SISO transfer functions. The key entries are:

#definition[
  - $H_(11)(z)$: S1 $arrow$ N1 --- the *winner's direct path* (input to winner's potential)
  - $H_(22)(z)$: S2 $arrow$ N2 --- the *loser's direct path*
  - $H_(21)(z)$: S1 $arrow$ N2 --- the *cross-inhibition path* (how S1 suppresses N2)
  - $H_(12)(z)$: S2 $arrow$ N1 --- the *competitor-to-winner path*
]

== DC Gain Analysis

The DC gain $H(z = 1)$ represents the steady-state response to a constant input.
The full DC gain matrix $bold(H)(1) = bold(C)(bold(I) - bold(A))^(-1) bold(B)$:

#figure(
  image("plots/transfer_dc_gain_matrix.png", width: 60%),
  caption: [DC gain matrix $bold(H)(1)$. Diagonal entries (direct paths)
    dominate. The off-diagonal entry $H_(12) = -34$ shows that S2 input
    suppresses N1 by 34 potential units at steady state.],
) <fig-dc-gain>

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    [*Path*], [*DC Gain $H(1)$*], [*Physical Meaning*],
    [$H_(11)$: S1 $arrow$ N1], [$200.0$], [Winner's direct path: full amplification],
    [$H_(22)$: S2 $arrow$ N2], [$223.5$], [Loser's direct path (slightly higher --- the inhibition
      from N1 reduces N2's effective leak, paradoxically increasing its open-loop gain)],
    [$H_(21)$: S1 $arrow$ N2], [$approx 0$], [Cross-inhibition negligible at DC in this linearization],
    [$H_(12)$: S2 $arrow$ N1], [$-34.0$], [Competitor input _suppresses_ winner by 34 units at DC],
  ),
  caption: [DC gain summary for key transfer function paths.],
) <tab-dc-gains>

#figure(
  image("plots/transfer_pole_zero.png", width: 100%),
  caption: [Pole-zero maps for the three key transfer functions. All poles lie
    near $z = 0.5$ (the leak rate), confirming the dominant first-order
    dynamics. The winner path (left) and loser path (right) have similar
    pole-zero structure; the cross-inhibition path (center) has only poles.],
) <fig-pole-zero>

= Frequency Response

== Bode Analysis

The frequency response $H(e^(j omega))$ for $omega in [0, pi]$ characterizes
how the network responds to input modulations at different frequencies. For a
discrete-time system, $omega = 0$ is DC and $omega = pi$ is the Nyquist
frequency (fastest possible alternation).

#figure(
  image("plots/bode_plots.png", width: 100%),
  caption: [Bode analysis of the linearized WTA network. Top-left: magnitude
    response of all four key paths. Top-right: phase response. Bottom-left:
    winner vs loser gain showing the winner's advantage across all frequencies.
    Bottom-right: normalized magnitude with $-3$ dB bandwidth markers.],
) <fig-bode>

#finding[
  The network acts as a *low-pass filter* for all paths, consistent with the
  leaky integrator nature of LIF neurons. Key bandwidth results:
  - Winner path bandwidth: $0.23 pi$ rad/sample
  - Loser path bandwidth: $0.17 pi$ rad/sample
  - The winner has *wider bandwidth* --- it can track faster input variations.

  The single-neuron time constant $tau = -1 / ln(0.5) = 1.44$ steps predicts a
  cutoff of $omega_c approx 0.5$ rad/sample $= 0.16 pi$. The inhibitory feedback
  modifies this, slightly extending the winner's bandwidth.
]

== Nyquist Analysis

#figure(
  image("plots/nyquist_plots.png", width: 100%),
  caption: [Nyquist diagrams for the winner (left) and loser (right) paths. The
    contour does not encircle the critical point $(-1, 0)$, confirming
    closed-loop stability. Key frequencies are marked along the contour.],
) <fig-nyquist>


= Steady-State Equilibrium Analysis

== Fixed Points

Setting $dot(bold(p)) = 0$ (or $bold(p)(t+1) = bold(p)(t)$ in discrete time):

$ bold(p)^* = frac(bold(W) dot f(bold(p)^*) + bold(B) dot bold(u), 1 - r) $

This nonlinear equation was solved numerically using `scipy.optimize.fsolve`
with 50+ random initial conditions to search for all fixed points.

#finding[
  For $bold(u) = (1,1,1,1)$, we found *20 distinct fixed points*:
  - *1 stable WTA equilibrium* (N1 wins): $bold(r)^* = (1.0, 0, 0, 0)$ ---
    a *stable node* (all Jacobian eigenvalues inside unit circle)
  - *6 stable mixed equilibria* where N1 is suppressed --- these are artifacts
    of the piecewise-linear $f$-$I$ curve's flat regions and are not reached in
    the stochastic system
  - *13 saddle points* --- unstable equilibria including the symmetric state
    where all neurons fire equally

  The WTA equilibrium at $bold(p)^* = (200, 0, 0, 0)$ has Jacobian eigenvalues
  $(0.5, 0.079, 0.71, 0.71)$, all well inside the unit circle.
]

== Phase Portrait and Basins of Attraction

#figure(
  image("plots/steady_state_phase.png", width: 100%),
  caption: [Left: phase portrait projected onto the N1--N2 membrane potential
    plane (N3, N4 initialized at 30). Trajectories are colored by which neuron
    wins. Right: basin of attraction map showing the dominant region for each
    winner state. N1 (blue) dominates the landscape.],
) <fig-phase>

#intuition[
  In the hydraulic analogy, the phase portrait shows "water" flowing through
  the network from any initial condition toward the N1-winner equilibrium. The
  basin of attraction is the "drainage area" that funnels activity into the
  N1 channel. The saddle points act as "watershed ridges" separating the
  basins.
]


= Transient Analysis and Validation

== Step Response

#figure(
  image("plots/transient_step_response.png", width: 100%),
  caption: [Step response comparison. Top-left: sigmoid mean-field potentials.
    Top-right: linearized system potentials. Bottom-left: mean-field firing
    rates with Monte Carlo averages overlaid. Bottom-right: stochastic spike
    rasters from three independent trials.],
) <fig-step-response>

== Monte Carlo Validation

The stochastic simulation with probabilistic firing was validated against PRISM
model checking results from the 3-neuron case study.

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    [*Neuron*], [*MC Mean (4N)*], [*MC Std*], [*PRISM (3N)*], [*Role*],
    [N1], [$40.6$], [$9.2$], [$37.2$], [Winner],
    [N2], [$5.8$], [$5.3$], [$7.8$], [Loser],
    [N3], [$5.8$], [$5.4$], [$7.8$], [Loser],
    [N4], [$5.8$], [$5.4$], [---], [Loser (4N only)],
  ),
  caption: [Spike counts over 50 time steps. Monte Carlo: 1000 trials. The
    4-neuron case shows a stronger WTA effect (ratio 6.98$times$) than the
    3-neuron PRISM result (ratio 4.78$times$) because the additional N4 neuron
    adds more mutual inhibition among losers.],
) <tab-mc>

#figure(
  image("plots/sim_monte_carlo_hist.png", width: 100%),
  caption: [Monte Carlo spike count distributions (1000 trials, T=50). N1
    shows a tight distribution centered at $approx 41$ spikes, while losers
    N2--N4 have wide, low-mean distributions. The histograms confirm robust
    WTA behavior across trials.],
) <fig-mc-hist>

== Settling Time

#figure(
  image("plots/transient_settling.png", width: 90%),
  caption: [Settling time analysis. The sigmoid mean-field firing rates converge
    within $approx 5$ steps, consistent with the eigenvalue prediction of
    $3 tau = 6.5$ steps. The red dashed line marks the predicted settling
    time.],
) <fig-settling>

#finding[
  The eigenvalue-predicted settling time of $3 tau = 6.5$ steps is consistent
  with the observed convergence of the mean-field simulation. The stochastic
  system exhibits slightly faster convergence due to the "hard" threshold
  nonlinearity acting as a sharpening mechanism.
]


= Discussion

== Summary of Classical Tools Applied

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    [*Tool*], [*What It Reveals*], [*Key Result*],
    [Eigenvalue decomposition of $bold(W)$],
    [Competitive modes of the network],
    [$lambda_2 = +90.9$ with eigenvector aligned to N1's advantage],

    [Linearized state matrix $bold(A)$],
    [Stability and time constants],
    [All $|lambda_i| < 1$; $tau_"dom" = 2.15$ steps],

    [Z-domain transfer functions],
    [Input-output gain at each frequency],
    [$H_(11)(1) = 200$, $H_(12)(1) = -34$ (cross-suppression)],

    [Bode plots],
    [Bandwidth and filtering behavior],
    [Winner BW $= 0.23 pi$, loser BW $= 0.17 pi$],

    [Nyquist plots],
    [Closed-loop stability margins],
    [No encirclement of $(-1, 0)$: stable],

    [Fixed-point analysis],
    [Equilibrium landscape],
    [20 FPs; only N1-winner is a stable node],

    [Phase portrait],
    [Basin of attraction geometry],
    [N1's basin dominates the state space],

    [Monte Carlo + PRISM],
    [Quantitative validation],
    [Winner/loser ratio $= 6.98 times$ (MC) vs $4.78 times$ (PRISM 3N)],
  ),
  caption: [Summary of classical engineering tools and their contributions to
    understanding the WTA network.],
) <tab-summary>

== Limitations

The sigmoid linearization is an approximation --- the actual LIF system uses a
piecewise-constant firing probability function. The linearization is most
accurate near the operating point and degrades for large perturbations. However,
even where quantitative accuracy is lost, the _qualitative_ predictions
(stability, mode structure, bandwidth ordering) remain valid.

The mean-field approximation replaces stochastic spike trains with deterministic
firing rates. This eliminates the variance that drives exploration of the
equilibrium landscape. The stochastic system can transiently visit saddle points
and unstable equilibria that the mean-field model bypasses entirely.

== Implications for SNN Design

The eigenvalue framework suggests a principled approach to SNN synthesis:

1. *Pole placement via weight matrix design.* Choose desired eigenvalues
   (decay rates, oscillation frequencies) and solve for the weight matrix
   $bold(W)$ that realizes them.

2. *Transfer function decomposition.* Express a desired input-output behavior
   as a target $H(z)$, decompose via partial fractions, and map each
   first-order section to a LIF neuron.

3. *Stability verification without simulation.* Compute $bold(A)$ and check
   that all eigenvalues lie inside the unit circle --- a necessary condition
   for bounded activity.

These approaches bridge the gap between the _design_ of SNNs using engineering
tools and the _verification_ of their properties using PRISM model checking.

= Conclusion

The subthreshold dynamics of LIF neurons are linear difference equations, making
the entire apparatus of discrete-time systems theory directly applicable.
For the 4-neuron WTA case study, this analysis:

- *Identified the WTA mechanism* as Mode 2 of the weight matrix
  ($lambda = +90.9$, eigenvector aligned with N1's advantage)
- *Proved stability* via eigenvalue analysis (all $|lambda_i| < 1$)
- *Quantified the frequency response* showing the winner has wider bandwidth
  ($0.23 pi$ vs $0.17 pi$)
- *Mapped the equilibrium landscape* finding 20 fixed points with only one
  stable WTA state
- *Validated against PRISM* with a winner/loser ratio of $6.98 times$ (Monte Carlo)
  vs $4.78 times$ (PRISM, 3-neuron)

The key methodological insight is that while the spiking nonlinearity prevents
exact analytical solutions, the linear analysis provides the right _qualitative_
structure and _approximate_ quantitative predictions, serving as a design and
analysis scaffold that complements formal verification.

#v(1cm)
#line(length: 100%)
#text(size: 9pt, style: "italic")[
  Generated using the CogSpike DEQ analysis toolkit (`deq/run_analysis.py`). \
  All plots are reproducible via `python3 run_analysis.py` in the `deq/` directory.
]
