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

## The ladder

| Tier | Experiments | Theme |
|---|---|---|
| 0 | e01–e02 | single-neuron latency law + calibration |
| 1 | e03–e05 | NEF softmax, calibrated readout race, temporal WTA decode |
| 2 | e06–e08 | attractor context, predictive subtraction, emission-premise probe |
| 3 | e09–e10 | PES learning of the rover law, Lyapunov descent |
| 4 | e11–e16 | end-to-end coders (frozen + learned) + four open-problem capstones |

See `../README.md` and the paper for the underlying theory.
