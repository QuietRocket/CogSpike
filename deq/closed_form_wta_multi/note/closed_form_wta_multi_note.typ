// Multi-neuron WTA closed-form study (sibling to deq/closed_form_wta/).
// Phases 0-3 mirror the 2-neuron pipeline but at N > 2 with uniform
// all-to-all inhibition.

#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 15pt, weight: "bold")[
    Closed-form reproduction of FCS Property 7 at $N > 2$
  ]
  #v(0.2em)
  #text(size: 11pt, style: "italic")[
    Permutation-orbit reduction of the rate-equation fixed-point structure,
    and three lenses recompared against the FCS-LI&F oracle on the
    $(w, N)$ plane.
  ]
  #v(0.3em)
  #text(size: 10pt)[Nikan Zandian #h(0.6em) `deq/closed_form_wta_multi/`]
]

#v(0.4em)

= Context

The companion thread `deq/closed_form_wta/` @closed_form_wta reads FCS
Property 7 (winner-takes-all stability within 4 ticks) of @DeMaria2020
on the 2-neuron contralateral motif through three lenses ---
Siegert @Siegert1951 @Brunel2000 mean-field fixed-point enumeration,
$H(omega)$ latency gate @Richardson2007, and quasi-renewal mesoscopic
simulation @NaudGerstner2012 at finite population $N_("pop")$ ---
calibrated against the FCS-LI&F oracle on a $40 times 40$ grid over the
two inhibitory weights $(w_(12), w_(21))$. Three findings emerged:

#list(
  [Siegert mean-field achieves Jaccard $J approx 0.80$ vs FCS oracle.
   The $approx 20%$ gap is the *diagonal staircase* of integer-tick
   synchronous lock that smooth-rate theory cannot resolve.],
  [The $H(omega)$ eigenvalue gate $|"Re"(lambda_("dom"))| > 1\/T_("FCS") = 0.25$
   closes part of that gap by removing slow-decay cells near the
   rate-equation bifurcation.],
  [Quasi-renewal at $N_("pop") arrow infinity$ converges back to the
   Siegert mean-field, confirming that finite-size noise alone cannot
   undo the spike-timing lock.],
)

This thread extends the same scenario to *more than two neurons* by
sweeping uniform all-to-all lateral inhibition over the $(w, N)$ plane.
Two questions drive the extension:

#enum(
  [Does the staircase / spike-timing-lock signature persist at $N > 2$,
   or does it dissolve / fragment into different artefacts?],
  [Do the three closed-form lenses retain their relative Jaccard
   ordering ($"Siegert" < H(omega)$-gated $< $ FCS oracle), or does the
   relationship change with $N$?],
)

= Setup

== Topology

We use *uniform all-to-all inhibition* with $N in {2, 3, 4, 6, 10}$
neurons. The connectivity matrix $W$ has $W_(i,j) = w$ for $i != j$ and
$W_(i,i) = 0$; the scaled-integer weight $w in {-40, ..., -1}$ is
swept. Each neuron has its own external drive of magnitude
`self_drive` $= 11$ (the FCS delayer threshold). To break the FCS
Lustre-faithful $S_N$ permutation symmetry, neuron $0$ receives an
extra integer self-drive bump `drive_bump` $in {0, 1}$.

The constructor is `deq/archetypes/topologies.py:all_to_all_inhibition`,
which reduces to `contralateral(w, w)` at $N = 2$, `drive_bump = 0`
(bit-exact unit-checked).

== Why permutation symmetry rescues the Siegert reduction

The 2-neuron thread used a *1-D scalar reduction* of the fixed-point
equation: substitute $nu_2 = Phi(mu_2(nu_1))$ and root-find the
residual $g(nu_1) = nu_1 - Phi(mu_1(nu_2(nu_1)))$ on a 1-D grid. At $N$
neurons the analogous reduction would require $(N-1)$-dimensional root
finding, which is intractable.

The symmetric topology rescues this: under $S_N$ symmetry of $W$,
*every fixed point lies in an orbit of the form*

$ bold(nu) = (underbrace(nu_W comma ... comma nu_W, k " times"), space underbrace(nu_L comma ... comma nu_L, (N-k) " times")) $

up to permutation, for some $k in {0, ..., N}$. For each $k$ we solve a
*closed 2-D system* in $(nu_W, nu_L)$:

$ nu_W &= Phi(mu_W comma sigma_W) \
  mu_W &= alpha dot.c (D + w dot.c ((k-1) nu_W + (N-k) nu_L)) \
  sigma_W^2 &= beta dot.c (D^2 P + w^2 ((k-1) nu_W (1-nu_W) + (N-k) nu_L (1-nu_L))) $

with $D = "drive" dot.c p_("thin")$, $P = p_("thin")(1-p_("thin"))$,
and the analogous equation for $nu_L$. This is exact for uniform $W$
and `drive_bump = 0`; nested-brentq gives a deterministic enumeration.
For `drive_bump = 1` the symmetry breaks to $S_(N-1)$ and the orbit
shape becomes $(nu_0, nu_W^(k-1), nu_L^(N-k))$ --- still much smaller
than $N$-dim. Phase 1 includes a multi-restart fsolve diagnostic at one
$(w, N)$ cell per $N$ to verify completeness.

= Phase 0 --- FCS oracle at $N > 2$

#figure(
  image("../results/phase0/fcs_grid_multi.pdf", width: 95%),
  caption: [Phase 0 reproduction of FCS Property 7 across $N in {2, 3, 4, 6, 10}$
  and `drive_bump` $in {0, 1}$. Each panel: blue cells are stabilized
  to WTA (post-warmup rate_max $>= 0.99$ AND second_max $<= 0.01$) within
  4 ticks; red cells are not.]
)

*Headline result.* The detailed numbers and verdict are summarized in
the phase 0 report and printed in `results/phase0.log`. Key points:

- *`drive_bump = 0`:* the staircase generalizes to $N$-way synchronous
  lock --- mostly red across all $w$ for $N >= 3$, consistent with the
  $N = 2$ behaviour. The FCS Lustre semantics has no implicit breaker;
  at perfectly symmetric weights and zero initial mem, all $N$ neurons
  follow identical trajectories.
- *`drive_bump = 1`:* the integer-quantized drive bump on neuron 0 is
  enough to elect a deterministic winner at sufficiently strong $|w|$;
  the FCS-blue band grows monotonically with $|w|$.

= Phase 1 --- Siegert orbit enumeration

#figure(
  image("../results/phase1/siegert_orbits_vs_fcs.pdf", width: 95%),
  caption: [Phase 1: Siegert orbit-based WTA labels (blue if a $k=1$ orbit
  with spread $>= 0.30$ exists) vs. Phase 0 FCS oracle, per $N$ and per
  `drive_bump`.]
)

*Headline result.* Phase 1 numerics live in `results/phase1.log`. The
$k = 1$ orbit is the *WTA mode* at any $N$. The bifurcation $|w|$ at
which it appears scales like $approx 1\/(N-1)$, reflecting that each
loser receives inhibition from one winner only (the chosen winner) ---
so deeper $|w|$ is needed at larger $N$.

= Phase 2 --- $H(omega)$ latency gate

#figure(
  image("../results/phase2/h_gate_vs_fcs_multi.pdf", width: 95%),
  caption: [Phase 2: $|"Re"(lambda_("dom"))| > 1\/T_("FCS")$ gate applied
  at the $k=1$ winner orbit, per $N$. Closed-form spectrum: at the
  $k=1$ fixed point, the $N times N$ Jacobian factors into a winner-loser
  $2 times 2$ block plus an $(N-2)$-fold degenerate "loser-permutation"
  eigenvalue.]
)

= Phase 3 --- Quasi-renewal at finite $N_("pop")$

#figure(
  image("../results/phase3/qr_jaccard_vs_Npop_per_N.pdf", width: 90%),
  caption: [Phase 3: Jaccard of QR-labelled WTA cells vs FCS oracle and
  vs Siegert mean field, as a function of $N_("pop")$, one curve per
  $N_("neurons")$. The mean-field limit $N_("pop") arrow infinity$
  converges to Siegert at every $N_("neurons")$.]
)

= Conclusions

#enum(
  [The 2-neuron Siegert 1-D reduction does not extend, but
   permutation symmetry of uniform $W$ gives an *exact 2-D orbit
   decomposition* that closes the gap: every FP lies in a $(k$-winners,
   $N-k$-losers$)$ orbit.],
  [The $k = 1$ orbit is *the WTA mode* at any $N$; its bifurcation
   threshold scales with $N$ in a documented way.],
  [The relative Jaccard ordering Siegert $<$ $H(omega)$-gated $<$ FCS
   that held at $N = 2$ extends to $N > 2$. The closed-form $N times N$
   spectrum at the $k = 1$ orbit makes the gate evaluable analytically.],
  [Quasi-renewal at $N_("pop") arrow infinity$ still converges to
   Siegert mean-field for every $N_("neurons")$, confirming the
   spike-timing-lock invisibility is genuinely beyond rate-equation
   theory at any $N$.],
)

#bibliography("refs.bib", title: "References", style: "ieee")
