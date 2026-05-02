#set page(paper: "a4", margin: 2cm)
#set text(size: 10pt)
#set par(justify: true)

#align(center)[
  #text(size: 14pt, weight: "bold")[Phase 0 report -- Stochastic-LI&F bridge & calibration]
  #v(0.2em)
  Verdict: *PASS*
]

= File inventory

The following modules were created under `./deq/closed_form/`:

- `stochastic_lif.py` -- N-neuron oracle wrapping
  `../archetypes/lif_fcs.py:simulate` with per-copy threshold heterogeneity
  and Bernoulli per-tick input thinning. Returns population-mean firing
  rates and per-copy spike trains.
- `phase0_infrastructure.py` -- this validation script.
- `README.md`, `make_all.sh`, `.gitignore` -- workspace boilerplate.

= Validation V0.1 -- Population f-I curve

Setup: single uncoupled neuron, $N = 100$ copies, threshold jitter
$epsilon = 2$, sweep external drive weight $w_X in {2, 4, dots, 20}$,
$T = 1000$ ticks. Three thinning levels $p_("thin") in {1.0, 0.7, 0.4}$.

Monotonicity per curve (drive vs rate non-decreasing): p=1.0: yes, p=0.7: yes, p=0.4: yes.

Saturation maxima per curve: p=1.0: 1.000, p=0.7: 0.701, p=0.4: 0.401.

Acceptance: each curve monotone, $p = 1.0$ saturating $>= 0.4$, smaller
$p_("thin")$ gives strictly lower saturation. *PASS*.

#figure(image("results/phase0/v01_rate_vs_drive.pdf", width: 80%),
  caption: [V0.1: population f-I curve at three thinning levels. The
  $p_("thin") = 1.0$ curve is the deterministic-input case (variance from
  threshold jitter only); lower $p_("thin")$ injects Bernoulli input
  variance, which both reduces the mean drive and reshapes the curve.])

= Validation V0.2 -- ISI CV calibration

Setup: single neuron at drive = 11 (FCS default), $N = 100$ copies,
$T = 2000$ ticks. Sweep $p_("thin") in {1.0, 0.85, 0.7, 0.5, 0.3}$ and
threshold jitter $epsilon in {0, 2, 5, 10\}$.

Maximum per-cell wallclock: 0.0 s
(threshold $30$ s).

Recommended operating point: p_thin = 0.70, jitter = 0, CV = 0.547, rate = 0.701.

Number of $(p_("thin"), epsilon)$ combinations with CV $>= 0.5$:
12 of 20.

Acceptance: at least one combination achieves CV $>= 0.5$ and total runtime
fits in budget. *PASS*.

#figure(image("results/phase0/v02_cv_heatmap.pdf", width: 90%),
  caption: [V0.2: ISI coefficient of variation (left) and mean firing rate
  (right) across the $(p_("thin"), epsilon)$ grid. CV $approx 1$ corresponds
  to Poisson-like firing where Siegert's diffusion approximation is valid;
  CV $<< 0.5$ means quasi-deterministic firing where the diffusion
  approximation underestimates regularity.])

= Overall verdict

*PASS*.

The recommended operating point (p_thin = 0.70, jitter = 0, CV = 0.547, rate = 0.701) is locked in for Phase 1's
Siegert comparison. The Bernoulli-thinning + threshold-jitter combination
produces input statistics in the diffusion-approximation-valid regime
without breaking the FCS-LI&F oracle's per-tick semantics.
