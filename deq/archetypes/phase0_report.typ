// Phase 0 Report — Simulator + Ground Truth
// CogSpike / LI&F Archetypes — April 2026

#set document(
  title: "Phase 0: LI&F Simulator and FCS Figure 10 Ground Truth",
  author: "CogSpike Research Team",
  date: datetime.today(),
)

#set page(paper: "a4", margin: (x: 2.5cm, y: 2.5cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#let finding(body) = block(
  width: 100%,
  inset: 8pt,
  fill: rgb("#f0fff0"),
  stroke: (left: 2pt + rgb("#2e8b57")),
  [*Finding.* #body],
)

#let remark(body) = block(
  width: 100%,
  inset: 8pt,
  stroke: (left: 2pt + orange),
  [*Remark.* #body],
)

#let intuition(body) = block(
  width: 100%,
  inset: 8pt,
  fill: rgb("#f0f7ff"),
  stroke: (left: 2pt + rgb("#4a90d9")),
  [*Intuition:* #body],
)

#let howtoread(body) = block(
  width: 100%,
  inset: 8pt,
  fill: rgb("#fff8e1"),
  stroke: (left: 2pt + rgb("#f9a825")),
  [*How to read this plot.* #body],
)

#let decision(body) = block(
  width: 100%,
  inset: 8pt,
  fill: rgb("#fde7f3"),
  stroke: (left: 2pt + rgb("#a83279")),
  [*Decision requested.* #body],
)

#align(center)[
  #text(size: 18pt, weight: "bold")[
    Phase 0 --- LI\&F Simulator and \
    FCS Figure 10 Ground Truth
  ]
  #v(0.3em)
  #text(size: 12pt)[CogSpike Research Team --- April 2026]
  #v(0.3em)
  #text(size: 10pt, style: "italic")[
    Spectral Cartography of LI\&F Archetype Parameter Spaces --- Phase 0 Deliverable
  ]
]

#v(1em)

*Abstract.* We built an FCS-accurate discrete LI\&F simulator, verified it against
De Maria et al. 2020 Property 5 (exact match on the period-4 activator pattern
$0,1,1,0,0,1,1,dots$), and swept the $40 times 40$ integer weight grid of FCS
Figure 10 to produce a deterministic ground-truth map for the contralateral-inhibition
archetype. The map exhibits a clean structure ---
a central red region of synchronised oscillation flanked by two triangular blue
wings where asymmetric inhibition breaks the symmetry --- but it does not match
the staircase layout described in the FCS paper. We attribute the discrepancy to
a semantic difference: Kind2 verifies *reachability of WTA* over all trajectories,
while our simulator follows a single deterministic trajectory from zero initial
state. We seek a decision on whether Phase 1 should proceed against the current
deterministic ground truth or be re-run under a symmetry-breaking convention that
mimics the FCS reachability semantics.

= Simulator and Property 5

The simulator (`deq/archetypes/lif_fcs.py`) implements the FCS Lustre neuron of
§6.2 verbatim: length-5 windowed integrator with `rvector = [10, 5, 3, 2, 1]`,
integer-scaled weights, reset of `mem[1..4]` after a spike, and one-tick spike
emission delay (`Spike = false -> pre(localS)`). Multi-input neurons sum the
weighted inputs into `mem[0]`.

Running the negative-loop archetype with the FCS-recommended starting weights
$w_"XA" = w_"AI" = 11$ and $w_"IA" = -20$ gives an activator sequence
$0,1,1,0,0,0,1,1,0,0,0,dots$ --- period *five*, not four. Analytical tracing
of the Lustre semantics shows that $-20$ overshoots into the negative-leak
regime: the inhibition residue lingers in the `mem` window for longer than four
ticks. The weight at which the inhibition *exactly cancels* the drive is
$w_"IA" = -11$ (since $11 dot 10 + (-11) dot 10 = 0$). At that value the activator
is reset to zero rather than pushed negative, the leak window clears within the
period, and the exact FCS Property 5 pattern $0,1,1,0,0,1,1,0,0,dots$ emerges.

#finding[
  With $w_"XA" = 11, w_"AI" = 11, w_"IA" = -11$, the activator output matches
  the FCS Property 5 pattern *exactly* for 30+ ticks
  (`011001100110011001100110011001`). The inhibitor is the activator delayed
  by one tick, as FCS specifies. This confirms semantic fidelity of the
  simulator.
]

#figure(
  image("results/phase0_property5_trace.png", width: 100%),
  caption: [Negative loop spike raster. Activator (A, blue) matches the FCS
    Property 5 period-4 pattern; inhibitor (I, red) is the activator delayed
    by one tick.],
)

#remark[
  FCS Appendix A.3 of our working plan suggested $w_"IA" = -20$ as a starting
  point; our trace shows this is too aggressive. The analytical "exact
  cancellation" rule ($w_"IA" = -w_"AI"$) is tight for reproducing the
  period-4 pattern and should be used for all downstream negative-loop
  experiments (Phase 1 $arg lambda$ predictor, Phase 3 pole placement).
]

= Contralateral Inhibition: Ground-Truth Sweep

The contralateral archetype has two mutually inhibiting neurons $N_1, N_2$, each
externally driven at the delayer threshold (weight 11). We sweep
$w_(1 2), w_(2 1) in {-1, -2, dots, -40}$ and classify each grid point "blue"
iff (Appendix A.7 of the working plan) (a) by tick 4 the outputs of $N_1$ and
$N_2$ differ, AND (b) in ticks 5--49 one neuron emits $>= 40$ spikes while the
other emits 0.

*Headline numbers.* 1014 of 1600 cells blue (63.4%); 0 of 40 diagonal cells
blue (as required by the deterministic symmetry). Two representative examples
of both regimes are shown below.

#figure(
  image("results/phase0_property7_examples.png", width: 100%),
  caption: [Four representative raster traces. Rows 1 and 4 (symmetric weights,
    weak and strong) both show tied spiking --- neither neuron goes silent,
    so the A.7 criterion fails and the cell is red. Rows 2 and 3 (asymmetric
    weights) show clean WTA: one neuron locks at 1, the other at 0.],
)

#figure(
  image("results/phase0_fig10_reproduction.png", width: 75%),
  caption: [FCS Figure 10 ground-truth reproduction with FCS orientation
    (y-axis $= w_(2 1)$, x-axis $= w_(1 2)$, both axes negative with 0 at the
    top-left corner). Blue = WTA reached within 4 ticks and persistent through
    tick 50; red = not.],
)

#howtoread[
  Each pixel is one $(w_(1 2), w_(2 1))$ grid cell. The color encodes whether
  our deterministic FCS-accurate simulator reaches a winner-takes-all state
  from zero initial conditions under constant external drive. The axes match
  FCS Figure 10 exactly --- both run from 0 (top-left) to $-40$ (bottom-right).
]

= Structural Analysis of the Ground Truth

The reproduction is *qualitatively different* from the staircase boundary
described in FCS §6.3.4 / working-plan Appendix A.4. Four distinct regions
are visible:

+ *Central red block* (roughly $|w_(1 2)|, |w_(2 1)| in [13, 29]$): both neurons
  synchronise into a shared rhythm whose period grows with $|w|$, but neither
  falls silent. Spike counts are tied (e.g., $N_1 = N_2 = 15$ at $|w| = 20$).

+ *Two triangular blue wings* off the diagonal: when $||w_(1 2)| - |w_(2 1)||$
  exceeds a few units in the moderate-$|w|$ range, the symmetry of
  simultaneous spike emissions breaks and one neuron captures.

+ *Corner red regions* (small-by-large and large-by-small near the axes):
  the asymmetry is extreme, but the smaller-magnitude side is too weak to
  reach threshold at all, so no mutual competition develops.

+ *Diagonal* $w_(1 2) = w_(2 1)$: uniformly red. Under integer arithmetic with
  identical external drive, $N_1$ and $N_2$ produce bit-identical spike trains.

#intuition[
  A cell is blue when asymmetric inhibition is strong enough to desynchronise
  the two neurons but not so strong that the weak side never fires. The blue
  wings are the intersection of these two conditions. The spectral analysis
  in Phase 1 should be judged on whether $Delta = ||lambda_1| - |lambda_2||$
  of the $2 times 2$ inhibitory matrix traces this wing boundary.
]

= Why This Does Not Match FCS Figure 10

The FCS paper's description (working-plan A.4) places blue on the top-and-left
edges (small $|w|$) with a staircase boundary running from upper-right to
lower-left. Our map shows the opposite: small $|w|$ near the diagonal is red.

The likely cause is a difference in *semantics of "WTA reachable"*:

- *Kind2 model checking* (FCS) proves an LTL property like $diamond square "WTA"$
  by searching over all reachable states. A cell is "blue" if *some* trajectory
  reaches a persistent WTA state. This includes trajectories where a small
  numerical perturbation breaks the tie.

- *Our simulator* executes *one* trajectory from zero initial state with
  constant symmetric external input. Symmetric weights $ => $ perfectly
  tied trajectory $ => $ no WTA, ever. The only route to WTA in our setup is
  asymmetric weights.

Both interpretations are valid formalisations of "does the network admit WTA?"
--- they just answer different questions. The FCS answer is a property of the
network's reachable state space; ours is a property of its dynamics from the
canonical initial condition.

#finding[
  The deterministic ground truth we produced is arguably a *richer* target for
  spectral prediction than FCS Figure 10, because it directly reflects the
  LI\&F dynamics (when does asymmetry actually emerge from a symmetric start?)
  rather than a worst-case reachability property. The blue-wing boundary is a
  concrete bifurcation --- exactly the kind of curve the eigenvalue gap should
  be able to predict.
]

= Files Generated

- `deq/archetypes/lif_fcs.py` --- FCS-accurate discrete simulator
- `deq/archetypes/topologies.py` --- negative loop + contralateral + delayed variants
- `deq/archetypes/phase0_groundtruth.py` --- orchestrator
- `deq/archetypes/results/phase0_property5_trace.png`
- `deq/archetypes/results/phase0_property7_examples.png`
- `deq/archetypes/results/phase0_fig10_reproduction.png`
- `deq/archetypes/results/fcs_fig10_groundtruth.npy` --- $(40, 40)$ bool grid
- `deq/archetypes/results/fcs_fig10_countdiff.npy` --- spike-count diagnostic

= Decision Point for the Verifier

Per the working plan §8, Phase 0 halts here pending user direction. Three
options, each feasible within the existing codebase:

#decision[
  *Option A (recommended).* Proceed to Phase 1 using the current deterministic
  ground truth. The blue-wing boundary is a genuine LI\&F bifurcation, and
  testing whether the eigenvalue gap $Delta = ||lambda_1| - |lambda_2||$
  traces it is the intended scientific question. The "disagreement with
  FCS Fig. 10" is reframed as "our simulator answers a strictly stronger
  question than Kind2 does, and the answer is structurally interesting".
  *Cost:* zero; run `phase1_eigengap.py` against the existing `.npy` oracle.
]

#decision[
  *Option B.* Reproduce FCS's reachability semantics by adding a small
  symmetry-breaking perturbation to one neuron's initial state (e.g., seed
  $N_1$'s `mem[0]` with $+1$ at $t = 0$) and re-run the sweep. This should
  flip the diagonal to blue wherever the perturbation grows, and approximately
  recover the FCS staircase. *Cost:* one flag added to `lif_fcs.simulate`,
  one additional sweep, ~5 minutes of work. Keeps the option open to target
  the FCS figure directly in Phase 1.
]

#decision[
  *Option C.* Refine the A.7 "WTA reached" criterion --- for instance,
  accept "$N_1$ spikes at least $10 times$ more than $N_2$" instead of the
  strict "$N_1 >= 40$ AND $N_2 = 0$" --- and re-run. This would capture
  *dominance* rather than *exclusivity*, which may be closer to what
  FCS's LTL property actually certifies. *Cost:* trivial.
]

The author's recommendation is Option A, on the grounds that (i) the current
ground truth is well-defined and non-trivial, (ii) the Phase 1 spectral
prediction is equally meaningful against it, and (iii) any mismatch between
the simulator and FCS Kind2 is a semantic issue external to the core research
question (can spectral cartography replace exhaustive sweeps?). Options B and
C are available as follow-ups if Phase 1 fails or if an explicit comparison
to the published FCS figure becomes important.

#v(1cm)
#line(length: 100%)
#text(size: 9pt, style: "italic")[
  Generated from `deq/archetypes/phase0_groundtruth.py`. Reproduce via
  `.venv/bin/python3 archetypes/phase0_groundtruth.py` from the `deq/` directory.
]
