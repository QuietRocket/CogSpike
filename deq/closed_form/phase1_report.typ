#set page(paper: "a4", margin: 2cm)
#set text(size: 10pt)
#set par(justify: true)

#align(center)[
  #text(size: 14pt, weight: "bold")[Phase 1 report -- Siegert closed form (H1)]
  #v(0.2em)
  Verdict: *PASS*
]

= Hypothesis (H1)

For the contralateral inhibition motif simulated with N = 80 FCS-LI&F
neurons per population (threshold heterogeneity $epsilon = 0$,
Bernoulli input thinning $p_("thin") = 0.7$), the Siegert closed form
$nu = Phi(mu, sigma)$ with parameters calibrated on the single-population
f-I curve predicts the WTA boundary in $(w_(12), w_(21))$ space at least
as well as the population thread's heuristic logistic sigmoid.

= S1A. Siegert calibration

Free parameters: $alpha$ (mean scale), $beta$ (variance scale), $tau_m$
(membrane time constant), $tau_("ref")$ (refractory period). $V_("th") = 1$,
$V_r = 0$ fixed.

Fit on V0.1 dataset ($n = 30$ points across drives in
${2, 4, dots, 20}$ at $p_("thin") in {1.0, 0.7, 0.4}$):

- $alpha = 0.2500$
- $beta = 0.004292$
- $tau_m = 2.3504$
- $tau_("ref") = 0.3611$
- $R^2$ on full dataset: $0.9357$ -- gate $>= 0.90$:
  *PASS*
- $R^2$ on stochastic subset ($p_("thin") < 1.0$): $0.9011$

Note on the two $R^2$ values: at $p_("thin") = 1.0$ the FCS LI&F is fully
deterministic and produces a discrete staircase f-I curve (rate jumps
$0 -> 1/3 -> 1/2 -> 1$ at integer drive thresholds). Siegert's diffusion
approximation cannot reproduce a discrete staircase by construction --
it assumes $sigma > 0$ for finite spectra. The stochastic-subset $R^2$ is
the more relevant performance measure for the Phase-0-locked operating
point ($p_("thin") = 0.7$).

#figure(image("results/phase1/s1a_calibration.pdf", width: 75%),
  caption: [S1A: Siegert with calibrated $(alpha, beta, tau_m, tau_("ref"))$
  vs the V0.1 oracle f-I curves at three thinning levels. Markers are
  oracle measurements; lines are Siegert predictions.])

= S1B-C. Contralateral grid comparison

Setup: $(w_(12), w_(21))$ each on the integer grid
${-40, dots, -1}$ ($n = 12$ values
each, total $144$ cells). For each cell:

- LI&F oracle: $N = 80$ population, $T = 200$, symmetry-broken by
  gating N2's drive for the first $2$ ticks. WTA = tail-rate
  ratio $> 4$.
- Siegert prediction: enumerate self-consistent fixed points of the
  2-population Phi-system; WTA = bistable ($>= 2$ fixed points with
  $|rho_1 - rho_2| > 0.05$).
- Population sigmoid (baseline): same enumeration with logistic sigmoid
  $f(x) = 1 / (1 + e^(-4(x - 1)))$, weights mapped via $w^("WC") = |w^("LIF")| / 8$.

Jaccard agreement of WTA-capable cells:
- Siegert vs LI&F oracle: $0.843$
- Population sigmoid vs LI&F oracle: $0.796$

Median boundary displacement (in grid-cell units):
- Siegert: $0.000$
- Population sigmoid: $0.000$

Improvement: Siegert is better
than population sigmoid on Jaccard;
larger boundary displacement.

#figure(image("results/phase1/s1bc_comparison.pdf", width: 100%),
  caption: [Left to right: LI&F oracle WTA labels, Siegert prediction,
  population-thread sigmoid prediction (baseline), and the number of
  Siegert fixed points (1 = monostable, 3 = bistable). Red = N1 dominant,
  blue = N2 dominant, white = symmetric.])

= Overall verdict

*PASS*.

Phase 1 acceptance criteria:
- S1A full-dataset $R^2 >= 0.90$: met.
- Siegert vs LI&F Jaccard $>= 0.70$: met.
- Siegert $>=$ population sigmoid baseline: met.

= Discussion

The LI&F oracle exhibits a *rectangular* WTA boundary in $(w_(12), w_(21))$:
N1 wins whenever $|w_(21)|$ is large enough to suppress N2 *regardless of*
$|w_(12)|$, and symmetrically. This is the spike-timing-locked bistability
documented in the population-thread Phase 4 (median displacement 0.68
WC-units against the WC pitchfork curve).

Siegert's diffusion approximation -- inheriting the smoothness of WC -- can
*partially* reproduce this rectangular shape because the asymmetric arms
appear here as monostable strongly-asymmetric fixed points (one rate near
saturation, the other near zero) when the inhibition is unbalanced. The
genuine *bistable* corner of the Siegert prediction (yellow in panel 4)
covers only the symmetric strong-inhibition region.

The narrow diagonal "no-winner" band visible in the LI&F oracle but absent
in both rate-model panels is the spike-timing-precise regime where neither
inhibition is fast enough to suppress the other -- a finite-population /
spike-timing-physics phenomenon by construction beyond rate equations. This
is the H3 (quasi-renewal mesoscopic) target.
