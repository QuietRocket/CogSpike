# Population-level spectral analysis of neuronal archetypes

Standalone research workspace. Takes the LI&F archetypes of
De Maria et al. 2020 up to a population-level Wilson–Cowan description,
where the non-smooth spike-reset rule is replaced by a smooth sigmoidal
gain and the full classical linearization / bifurcation toolkit applies.

The work is phase-gated. Each phase tests a prespecified hypothesis
with numerical acceptance criteria and emits a `phaseN_report.pdf`.

- Phase 0: infrastructure validation (V0.1 single population, V0.2 symmetric inhibition).
- Phase 1: spectral gap as WTA proxy on the contralateral archetype.
- Phase 2: symbolic pitchfork + Hopf bifurcation analysis.
- Phase 3: pole-placement inverse design for target oscillation frequencies.
- Phase 4: cross-validation against the discrete LI&F simulator as a black-box oracle.
- Phase 5: generalization to series, parallel, and positive-loop archetypes.
- Phase 6: standalone typst research note.

## Run

```
bash ./make_all.sh
```

Requires Python 3.14 with numpy, scipy, sympy, matplotlib (shared venv at
`../.venv`, symlinked here as `.venv`), plus `typst` for report rendering.

Reproducibility: numpy seed `20260420`; all numerical artifacts in `./results/`.
