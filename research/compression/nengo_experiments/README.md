# nengo_experiments — Spikes as Bits, Learned, *in real spikes*

A graded ladder of [Nengo](https://www.nengo.ai/) experiments that **embody** the
claims of `../unified_spiking_compression.typ` ("Spikes as Bits, Learned") in an
actual spiking neural network, and measure — claim by claim — where the paper's
idealized identities survive in spikes and where they strain.

The paper's existing validators (`../validate.py`, `../learn_validate.py`,
`../critique_checks.py`) are **pure numpy**: they prove the *math* is self-consistent
but run no spiking simulator. This suite replaces each idealization (one clean LIF
ODE, float64 softmax, a literal weight matrix, infinite populations, an infinitely
fine clock) with a real spiking mechanism and quantifies the gap.

## Quick start

```bash
cd research/compression/nengo_experiments
uv sync                                   # create the venv, install nengo
uv run python -c "import nengo; print(nengo.__version__)"
uv run python e01_lif_latency/run.py      # run a single experiment
bash run_all.sh                           # run everything + compile all PDFs
```

## Layout

- `spikecoder/` — shared package (rover source, information measures, latency
  calibration, reusable Nengo network builders, metrics, plotting). Knowledge
  accumulates here; later experiments import earlier builders.
- `eNN_*/` — one folder per experiment: `run.py`, `report.typ`, `report.pdf`,
  `results/`, `README.md`.
- `report/synthesis.{typ,pdf}` — the final experimental report compiling all findings.

## Status: complete — 16/16 experiments, 142/142 acceptance checks

The final compiled findings are in **`report/synthesis.pdf`** ("Spikes as Bits, Learned —
in Real Spikes"). Headline: the paper's exact identities survive in real spikes to the
timing grid and machine precision; the quantitative claims survive with a small overhead
that decomposes into the limits the paper flagged; the latency code is refractory-immune
(cleaner than hoped); and the one open premise (cost-spike = gradient) is empirically
narrowed. In the learned coder, bits/symbol fall 1.985 → 1.026 while accuracy rises to the
Bayes-optimal ceiling (r = −0.98).

## The ladder

| Tier | Exp | Theme | result |
|---|---|---|---|
| 0 | e01 | LIF first-spike latency law | 6/6 · law to <1 dt; refractory-immune; τ_rc=20.018 ms |
| 0 | e02 | calibration t*(q)=−λ log₂ q | 5/5 · exact to ≤1 dt; q_max=0.93; noise tail |
| 1 | e03 | NEF softmax vs divisive norm | 8/8 · RMSE ~1/√N; partition representational |
| 1 | e04 | calibrated readout race | 10/10 · stream identity 0.979/1.752/1.114 |
| 1 | e05 | temporal WTA decode | 8/8 · lossless (perfect & uniform); settled F G |
| 2 | e06 | ring-attractor context | 5/5 · holds to <5.7°; capacity 7.9 bits |
| 2 | e07 | predictive subtraction (sign) | 8/8 · inhibitory silences, additive runs away; r=+0.94 |
| 2 | e08 | emission-premise probe | 10/10 · open premise narrowed; emission robust |
| 3 | e09 | PES learns the rover law | 8/8 · q→P; energy 2.0→1.030; grad id 5.8e−10 |
| 3 | e10 | Lyapunov descent | 9/9 · constant = ln2; sum-mode 9e−17; O(η) ball |
| 4 | e11 | frozen end-to-end coder | 13/13 · 0.979 on the floor; lossless; overhead decomposed |
| 4 | e12 | learned end-to-end coder | 12/12 · **the money plot** (1.985→1.026, acc→0.803) |
| 4 | e13 | non-stationary tracking | 8/8 · re-tracks moving floor; O(η) ball; decr-η freezes |
| 4 | e14 | deeper memory / eligibility | 11/11 · recovers ~100% of I₂ (gated on γ>0) |
| 4 | e15 | lossy graded-WTA | 12/12 · spiking rate–distortion curve |
| 4 | e16 | neuromorphic event stream | 9/9 · 8.2× compression; tracks predictability |

See `../README.md` and the paper for the underlying theory; `report/synthesis.pdf` for the
full cross-cutting analysis.
