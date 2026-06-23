# CogSpike — a spiking compression / learning / prediction playground

An interactive, in-browser playground for the spiking predictive-coding framework where
**first-spike latency literally encodes surprisal** (`t*(q) = -λ·log₂ q`) and a **local
three-factor delta rule** (`ΔW = η·c·(y − q)`) learns to be unsurprised. Built on top of the
existing CogSpike SNN workbench (a Rust / eframe-egui app that compiles native and to WASM).

The playground is meant to give an intuitive, visual feel for how a learned spiking coder
**compresses**, **predicts**, and **functions** — across several contrasting scenarios.

## The views

Launch the app and use the top mode bar:

- **Gym — the money plot.** A symbol stream is predicted online; the running bits/symbol
  descends to the source's entropy-rate floor as the delta rule learns, with accuracy rising
  to the Bayes ceiling. A **scenario selector** switches the source to make
  *predictable = compressible* visible by contrast:
  - **Uniform (random)** — no structure; incompressible (stays at log₂4 = 2 bits).
  - **Biased (skewed)** — a skewed marginal; compresses to `H(π) = 1.75` bits.
  - **Periodic (cycle)** — near-deterministic; compresses to ~0.4 bits.
  - **Rover (momentum)** — sticky Markov memory; compresses to the entropy rate 0.9782.
- **Event camera — predictive video compression.** A bright bar drifts across a pixel strip
  emitting ON/OFF spikes; a motion predictor cancels the anticipated spikes and only the
  *unpredicted* ones survive. Steady motion compresses ~80–95×; raising the velocity jitter
  makes the motion unpredictable and compression collapses toward 1× — compression tracks
  predictability. (Ported from the `e16` event-stream experiment.)
- **Spike latency — surprisal made literal.** Each symbol is a LIF neuron; the more the coder
  expects a symbol, the harder its membrane is driven and the **sooner it fires**. The first
  neuron to cross threshold is the prediction, and the wait is exactly the Shannon code
  length. Watch the four ramps start bunched (a tie = 2 bits) and separate as the coder learns
  (the expected move fires in ~4 ms ≈ 0.2 bits while surprises wait ~90 ms ≈ 4–5 bits).

The **Design / Simulate / Verify** modes are the original CogSpike SNN editor, simulator, and
PRISM/PCTL formal-verification workbench.

## Run it

Native:

```
cargo run --release
```

Web (recommended for the demo — shareable, no install for viewers):

```
cargo install --locked trunk        # once
cd crates/cogspike-app && trunk serve --port 8080
# open http://127.0.0.1:8080/
```

After rebuilding, **hard-reload (Cmd/Ctrl+Shift+R)** — the app is a PWA and its service worker
caches the wasm, so a normal reload can show a stale build.

## Verify

```
./check.sh            # native + wasm + fmt + clippy -D warnings + tests + doctests + trunk build
cargo test --workspace
```

See `VERIFICATION.md` for a per-rung log of what each view proves and the exact observed numbers.

## Research provenance

The math is ported byte-faithfully from the compression study under
`research/compression/` (`unified_spiking_compression.*`, `nengo_experiments/spikecoder/*.py`,
`e16_event_stream/run.py`). The idealized coder is parity-tested against the numpy oracle
(`crates/cogspike-core/tests/`); the scenarios and event-camera identity are validated by
deterministic unit tests against the closed-form / integer oracle.
