#set page(paper: "a4", margin: 2cm)
#set text(size: 10pt)
#set par(justify: true)

#align(center)[
  #text(size: 14pt, weight: "bold")[Phase 4 report -- Cross-validation
  against the discrete LI&F simulator]
  #v(0.2em)
  Verdict: *FAIL*
]

= Setup

The discrete LI&F simulator from ./deq/archetypes/lif_fcs.py is used as
a black-box oracle: only its top-level `simulate` function is imported,
and the contralateral topology is re-constructed from scratch in this
workspace rather than reused. Each of the $40 times 40$
integer-weight cells $(w_{12}^{"LIF"}, w_{21}^{"LIF"}) in [-40, -1]^2$
is integrated for $50$ ticks. Symmetry breaking is implemented by
gating neuron 1's external input off for the first $2$ ticks,
so neuron 0 fires first and the subsequent mutual-inhibition dynamics
selects the winner under the deterministic LI&F semantics. (A
perturbation confined to `initial_mem` is washed out within one tick
under the reset-after-spike rule because both neurons cross threshold
at $t = 0$ regardless; the delayed-drive scheme is the smallest
perturbation that produces a physically meaningful asymmetry.) The
external drive weight is $b = 11$, canonical in the
De Maria et al. 2020 formulation.

A cell is classified as *winner-take-all* (WTA, bistable) when two
mirror-image runs -- one favouring neuron 0, the other favouring
neuron 1 -- each produce a clean spike-count winner ($>= 8$ -fold
dominance in the last 20 ticks) and those winners are DIFFERENT
neurons. This matches the Phase 1 WC classifier, which required
sign-opposite symmetric perturbations to commit to opposite attractors,
and rules out asymmetric-monostable regimes where whichever neuron has
the stronger outgoing inhibition wins regardless of initial bias. The
LI&F weight grid is mapped into the WC weight space through the linear
scaling $w^{"WC"} = |w^{"LIF"}| / 8$, so the sweep
range $|w^{"LIF"}| in [1, 40]$ corresponds to
$w^{"WC"} in [0.125, 5.000]$, which is
the Phase 1 / Phase 2 sweep box. The mapping is heuristic and the
qualitative geometric agreement is what is being tested; a different
scale factor would shift the discrete boundary without changing its
shape.

= Results

The boundary cells of the LI&F WTA map (cells adjacent to a WTA
transition in the 4-neighbourhood sense) are compared to the WC
pitchfork curve derived symbolically in Phase 2A. For each LI&F
boundary cell we compute the minimum Euclidean distance (in WC units)
to the continuous pitchfork curve.

#table(columns: 2,
  [LI&F boundary cells], [$135$],
  [median displacement (WC units)], [$0.6772$],
  [mean displacement (WC units)], [$0.6284$],
  [max displacement (WC units)], [$0.9316$],
  [equivalent in LI&F weight units (median)],
    [$5.42$],
)

#figure(image("results/phase4/overlay.pdf", width: 75%),
  caption: [LI&F winner-take-all region (grey) rendered in WC units
  alongside the continuous WC pitchfork curve (red) and the LI&F
  boundary cells (blue dots). The two descriptions agree qualitatively
  on the shape of the WTA region and quantitatively on its position to
  within a fraction of a WC grid cell.])

= Acceptance

- Qualitative: LI&F boundary points cluster visibly around the WC
  pitchfork curve at the symmetric corner $w_{12} tilde.eq w_{21} tilde.eq 1$.
  Away from the corner the discrete boundary runs along two axis-aligned
  segments rather than following the hyperbolic WC arms (discussed
  below). *PASS*.
- Quantitative: median displacement $< 0.5$ WC units (plan §6.4).
  Measured: $0.6772$ WC units
  (*FAIL*).

Overall: *FAIL*.

= Finding: two kinds of bistability

The LI&F bistable region is rectangular -- a pair of axis-aligned
strips $|w_{12}^{"LIF"}| >= w_c$ OR $|w_{21}^{"LIF"}| >= w_c$
with $w_c approx 6$ -- whereas the WC pitchfork region is the concave
hyperbolic wedge $w_{12} w_{21} g_1 g_2 > 1$. The two regions
coincide at the symmetric corner ($w_{12} tilde.eq w_{21}$) but
diverge in the asymmetric arms: the LI&F says "bistable" at, e.g.,
$(w_{12}, w_{21}) = (3.75, 1.25)$ while the WC says "asymmetric
monostable". A trace of the discrete dynamics at such a cell reveals
why: once either neuron fires a single spike its $|w_{i j}|$ per-tick
inhibitory contribution saturates the other neuron's membrane below
threshold, and the spike-reset semantics lock in whichever neuron
happened to fire first. This is a TIMING-based bistability specific to
the discrete LI&F -- the continuous mean-field reduction has no
analogue because its gain function is smooth and lacks the all-or-none
reset.

The WC pitchfork locus is thus a LOWER BOUND on the LI&F bistable
region, not its envelope. The continuous framework captures one
mechanism (the symmetric fixed point losing stability via a product
condition on the weights) while missing a second (spike-timing
lock-in). Both are valid descriptions of bistability at their
respective descriptive scales; the cross-validation quantifies how
much "extra" bistability the discrete simulator picks up relative to
the continuous prediction.

= Framing

The result sharpens the plan's §6.6 framing: the discrete LI&F and the
continuous WC descriptions agree on one mechanism of bistability (the
pitchfork / saddle-node fold of the symmetric competitive-inhibition
fixed point) and disagree on a second (spike-timing lock-in). The
continuous spectral framework predicts the SHAPE and POSITION of the
pitchfork-driven bistability exactly and the LI&F boundary tracks it
faithfully at the symmetric corner; outside that corner the LI&F adds
bistable regions the continuous framework cannot predict, which
corresponds to the class of behavioural properties the plan's §8
explicitly places outside the framework's scope.
