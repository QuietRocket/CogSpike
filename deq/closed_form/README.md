# Closed-form finite-population analysis of LI&F archetypes

Standalone research workspace. Builds on the population thread
(`../population/`) by replacing its heuristic logistic sigmoid with three
physically-derived closed-form objects, while keeping the topology
*finite, tangible, and explicit*:

1. **Siegert formula** -- diffusion-approximation static f-I curve
   $\nu = \Phi(\mu, \sigma)$ per population (closed form in `erf`).
2. **Linear-response transfer function** $H_i(\omega)$ (Richardson 2007/2008
   threshold-integration recipe) per population. The motif becomes a block
   diagram $\delta\nu(\omega) = (I - H(\omega) J)^{-1} H(\omega) \delta\mu_{ext}(\omega)$
   amenable to classical control theory.
3. **Quasi-renewal mesoscopic equation** (Naud-Gerstner 2012) for finite-N
   populations -- single-integral hazard-rate update with
   $\sqrt{A/N}$ finite-size noise. Targets the spike-timing-locked
   bistability that population thread's Phase 4 missed.

Phase-gated. Each phase tests a hypothesis with numerical acceptance
criteria and emits `phaseN_report.pdf`.

- Phase 0: stochastic-LI&F bridge (threshold heterogeneity + Bernoulli
  input thinning) wrapping `../archetypes/lif_fcs.py` as black-box oracle.
- Phase 1 (H1): Siegert closed form vs LI&F oracle on contralateral motif.
- Phase 2 (H2A): Richardson transfer function $H_i(\omega)$ and closed-loop
  poles of $(I - H J)$.
- Phase 3 (H2B): linear-response cross-validation against LI&F E-I loop FFT.
- Phase 4 (H3): quasi-renewal mesoscopic at finite N, targeting
  spike-timing-lock recovery.
- Phase 5: integrating standalone note (`note/closed_form_note.typ`).

## Run

```
bash ./make_all.sh
```

Requires Python 3.14 with numpy, scipy, sympy, matplotlib (shared venv at
`../.venv`, symlinked here as `.venv`), plus `typst` for report rendering.

Reproducibility: numpy seed `20260502`; all numerical artifacts in `./results/`.
