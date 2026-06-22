# CogSpike playground — verification log

An autonomous overnight build on the `lazy-gym` branch. North star: an intuitive,
visual, in-browser **playground** for compression / learning / prediction, demo-able to
PhD advisors. Every rung below is an atomic, green-gated commit.

How to re-verify everything at once:

```
./check.sh                         # native + wasm + fmt + clippy -D warnings + tests + doctests + trunk build
cargo test --workspace             # all unit + integration tests
cd crates/cogspike-app && trunk serve --port 8080
# open http://127.0.0.1:8080/  (hard-reload Cmd/Ctrl+Shift+R to bypass the PWA service-worker cache)
```

Status legend: **VERIFIED** = built, gated green, and observed working in-browser.

---

## C0 — clippy cleanup (green baseline) · commit `c27d5de` · VERIFIED

**Claim.** `./check.sh` is fully green so every later commit gates cleanly.

The inherited eframe-template restriction-lint policy was red at HEAD (155 lib lints, plus
~95 more across bins/tests/app that only surfaced once the lib compiled clean). Library
code was fixed genuinely and behavior-preservingly (indexing → `.get()`, unwrap →
`.expect()`/`.ok()`, deterministic sorted iteration in the PRISM generators, match-arm
merges, doc fixes). Idiomatic restriction-lint noise in test/CLI/UI targets (`print_stdout`,
`unwrap` in tests, indexing in egui UI, long codegen fns) is scoped with file-level
`#[expect(..., reason=...)]` per the repo's expect-over-allow policy — no global relaxation.

**Re-verify.** `./check.sh` → exit 0 (clippy reports 0 warnings). Behavior preservation is
backstopped by the full test suite (incl. `paper_benchmark`, which exercises the PRISM
generators) passing.

---

## P2 — multi-scenario playground · commit `6acbe99` · VERIFIED

**Claim.** The money-plot gym becomes a playground that makes "predictable = compressible"
visible by *contrast* across four sources, each a general Markov source `(pi, P)` with
analytic floor/ceiling reference lines.

**Oracle (unit tests, `gym/scenario.rs`).** Entropy-rate floors ordered by compressibility:
Uniform = 2.0000 (= log2 4), Biased = 1.7500 (= H(pi)), Periodic < 0.6, Rover = 0.9782.
IID floors equal their marginals (memory only helps the rover). Bayes ceilings: Uniform
0.2500, Rover 0.8031, Periodic > 0.9. Every transition row is a probability distribution.

**Observed in-browser (Gym mode).**
- **Rover (momentum)** after ~1.7M steps: bits/symbol **0.9830** sitting on the floor
  **0.9782**; accuracy **0.8027** on the Bayes ceiling **0.8031**; max|q − P| = **0.0166**
  (learned-q heatmap matches true P). η decayed to 0.0057.
- **Uniform (random)**: bits/symbol **2.0006** flat on floor **2.0000** (incompressible —
  "nothing to learn"); accuracy **0.26** ≈ Bayes **0.25**; the stickiness slider correctly
  disappears (Uniform has no memory). The Rover→0.978 vs Uniform→2.0 contrast is the demo.

**Re-verify.** `cargo test -p cog_spike --lib scenario::` (3 tests). In-browser: Gym mode →
switch the *Source scenario* selector → Run; watch the floor/ceiling lines and tagline change.

---

## P3 — event-camera (DVS) scenario · commits `7af4fe9`, plus glyph polish · VERIFIED

**Claim.** A visual, hardware-native compressor: a bar drifts across a pixel strip emitting
ON/OFF spikes; a predictor cancels the anticipated spikes; the sparse residual is the
compressed video. Ported from `e16_event_stream/run.py` as the pure-integer identity
`resid = raw AND NOT predicted` (exact because `J_ON = 6 >> theta = 1` with per-frame reset —
e16 separately validated 9/9 that the Nengo soma bank realizes this identity).

**Oracle (unit tests, `dvs.rs`).** `roll` matches `np.roll` semantics; the canonical e16
schedule (41 frames) compresses > 2× and concentrates ≥ 50% of the residual in the motion-
novelty windows; steady motion compresses > 5×; raising jitter strictly lowers compression.

**Observed in-browser (Event camera mode).**
- **Steady motion** (velocity 1, jitter 0): **~80–95× compression** (e.g. 438 raw events →
  5 residual spikes); the RAW raster shows clean diagonal ON/OFF stripes, the RESIDUAL raster
  is nearly empty.
- **Jittered motion** (velocity jitter 0.98): compression collapses to **~1.8×** (284 raw →
  158 residual); the RAW raster goes jagged and the RESIDUAL fills in. "Compression tracks
  predictability," shown live.

**Re-verify.** `cargo test -p cog_spike --lib dvs::` (4 tests). In-browser: Event camera mode
→ watch the rasters; toggle *predictor on* and the *velocity jitter* slider.

**Note.** A glyph-tofu polish removed `→`/`■` characters (egui's default font lacks them);
labels now use ASCII / colored text. Confirmed in-browser after a hard reload (the PWA
service worker had cached the prior wasm — a normal reload shows stale glyphs).

---

## P5 — spike-latency view (`Mode::Spikes`) · VERIFIED

**Claim.** Makes the paper's thesis literal: each symbol is a LIF neuron whose calibration
drive `J = theta/(1 - q^alpha)` makes it fire at `t*(q) = -lambda log2 q`. The most-expected
symbol is driven hardest, crosses threshold FIRST, and the wait IS the code length. The
learner runs fast in the background (batched symbols/frame); the four ramps start bunched
(a tie = log2 4 = 2 bits) and separate as `q` sharpens.

**Observed in-browser (Spike latency mode).** After convergence on the rover: the expected
move **U fires at 3.8 ms = 0.19 bits** (q = 0.876), while the surprising moves wait ~86–98 ms
= 4.3–4.9 bits (q ≈ 0.03–0.05). "fires first · actual = U"; bits/symbol 0.959, accuracy 0.814.
The per-symbol table lists each q, its latency (ms) and bits, with the first-spike (decoded)
and actual symbols tagged. A *Step* button advances one symbol for inspection; a *learn speed*
slider controls background convergence.

**Re-verify.** In-browser: Spike latency mode → watch the U ramp pull ahead; the surviving
ramps asymptote just above threshold (very late / "never" = high surprise). Uses the
`latency.rs` LIF model (`calibration_drive`, `nengo_first_spike_time`) — the faithful latency
coder, not the discretized PRISM-oriented `run_simulation` (whose dynamics are a different
model). No PRISM (formal proof dropped from this run's critical path per the user's steer).

---

## Parity-fixture note (honest scope)

P2/P3 are validated by **deterministic behavioral unit tests** against the closed-form /
integer oracle, not against regenerated numpy `.npz` golden files. The DVS residual identity
`raw AND NOT pred` is exact (proven, J_ON ≫ theta), and e16's own 9/9 acceptance separately
validated that the Nengo spiking realization matches it; a byte-exact numpy golden fixture
(via `uv run`) was deferred to keep the run moving and avoid the slow Nengo path. The scenario
floors/ceilings are closed-form and asserted to ≤ 1e-9.
