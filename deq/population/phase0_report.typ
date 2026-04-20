#set page(paper: "a4", margin: 2cm)
#set text(size: 10pt)
#set par(justify: true)

#align(center)[
  #text(size: 14pt, weight: "bold")[Phase 0 report -- Infrastructure validation]
  #v(0.2em)
  Verdict: *PASS*
]

= File inventory

The following modules were created under `./deq/population/`:

- `wilson_cowan.py` -- sigmoid, RHS, simulator, fixed-point solver.
- `topologies.py` -- archetype weight-matrix builders.
- `linearization.py` -- Jacobian, spectrum, spectral gap, stability test.
- `phase0_infrastructure.py` -- this validation script.

Modules reserved as stubs for later phases: `bifurcation.py`,
`pole_placement.py`, `ground_truth.py`, `phase1_spectral_gap.py`,
`phase2_bifurcation.py`, `phase3_pole_placement.py`,
`phase4_cross_validation.py`, `phase5_other_archetypes.py`,
`final_summary.py`.

= Validation V0.1 -- Single uncoupled population

Setup: $W = [0]$, $I = 2$, $tau = 1$, sigmoid $k = 4$, $theta = 1$.

Analytical fixed point: $rho^* = f(2) = 1 / (1 + e^(-4)) approx 0.982013790$.

Numerical fixed point: $0.982013790$; error $0.00e+00$.

Analytical eigenvalue of Jacobian: $-1.000000000$
(the Jacobian formula in the plan §2.5 reduces to $-1 slash tau$ when $W = 0$,
since the sigmoid-slope term is multiplied by the zero self-coupling; a
noted typo in plan §2.6 wrote $-1 slash tau + f'(I) slash tau$, which would
correspond to a unit self-loop instead).
Numerical eigenvalue: $-1.000000000$; error $0.00e+00$.

Trajectory from $rho_0 = 0$ monotone: yes.

Acceptance: fixed-point error $< 10^{-6}$, eigenvalue error $< 10^{-6}$, monotone --
*PASS*.

#figure(image("results/phase0/v01_trajectory.pdf", width: 75%),
  caption: [V0.1: single population trajectory from zero initial condition
  converges monotonically to the analytical fixed point $f(I) = f(2)$.])

= Validation V0.2 -- Symmetric contralateral inhibition, low coupling

Setup: $W = "contralateral_inhibition"(0.5, 0.5)$, $I = [1.5, 1.5]$.

Symmetric fixed point: $(rho_1^*, rho_2^*) = (0.662584193, 0.662584193)$;
symmetry error $0.00e+00$.

Jacobian eigenvalue real parts: min $-1.447133$, max $-0.552867$.
Both below $-0.1$: yes.

Return distance at $t = 10$ from perturbed initial condition $rho_0 = (rho_1^* + 0.01, rho_2^* - 0.01)$:
$5.59e-05$ (threshold $10^{-3}$).

Acceptance: *PASS*.

#figure(image("results/phase0/v02_return_to_symmetry.pdf", width: 75%),
  caption: [V0.2: after a small asymmetric perturbation the two population
  rates return to the symmetric fixed point (dashed line) well within ten
  time constants.])

= Overall verdict

*PASS*.

All Phase 0 acceptance criteria are met.
