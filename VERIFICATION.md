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

## Parity (byte-exact numpy oracle)

- **P3 / DVS** now has a **byte-exact** parity test (`tests/dvs_parity.rs` vs
  `tests/fixtures/dvs_golden.json`, regenerated by `generate_dvs_golden.py` with numpy only —
  no Nengo). It reproduces e16's headline exactly: **raw 98 → residual 12 = 8.17×
  compression**, and the **jitter sweep correlation r = +0.841** (compression tracks
  predictability), with every event/residual map and per-ρ ratio asserted `==` / ≤ 1e-9. This
  validates the Rust port of bar brightness, edge events, `np.roll` prediction, and the
  integer residual identity against numpy's array ops. (The identity itself = the Nengo soma
  bank was separately proven by e16's own 9/9 acceptance; running Nengo was unnecessary.)
- **P2 / scenarios** floors/ceilings are closed-form, asserted to ≤ 1e-9 in unit tests.
- **Idealized coder** (pre-existing) is parity-tested to ≤ 1e-12 vs the numpy golden
  (`tests/coder_parity.rs`).

---

# "Agent learning its world" demos (D1–D3)

Three demos that make the existing machinery *read as an autonomous agent learning its
world*. Each is an atomic, green-gated commit on `lazy-gym`. The **logic** of each is pinned
by unit tests; `./check.sh` is fully green (native + wasm32 + fmt + clippy `-D warnings` +
tests + doctests + trunk build). The **live visual** is served at `:8080` (hard-reload to
bypass the PWA cache) — the wasm loads with no panic (console clean of app errors).

## D0 — clip-enforced surprise bound (shipped with D1) · commit `1d4d954` · VERIFIED

**Claim.** The previously-dead `Q_CLIP_LO = 1e-4` is now wired into the live `bits` path
(`gym/rover.rs`), so per-symbol surprisal is bounded by `-log2(Q_CLIP_LO) = 13.2877` bits =
a `0.266 s` first-spike-latency cap on the idealized substrate. An **honest clip-enforced
bound**, not a PRISM/PCTL machine-check (that stays reserved for the spike view).

**Oracle (`gym/rover.rs` tests).** `surprisal_is_clip_bounded`: a maximally-wrong prediction
(q → 0) yields `bits ≤ 13.2877 + 1e-9` over 32 steps. `typical_surprisal_is_unaffected_by_the_clip`:
uniform `q` → exactly 2 bits. The clip is inactive in normal operation, so **`coder_parity`
(≤1e-12), `paper_benchmark`, and `rover_episode` convergence are unchanged** (re-run green).

**Re-verify.** `cargo test -p cog_spike --lib rover::`.

## D1 — Boredom Meter (`Mode::Gym`) · commit `1d4d954` · VERIFIED (logic) / live-served

**Claim.** The agent *gets bored when it understands, flinches when you change its world, and
re-learns* — fenced by the bound. An instantaneous surprise needle (short-EMA bits) reads
**BORED** (green) near the floor / **SURPRISED** (pulsing red) on a jump; the learned-q
heatmap is recolored as a hot→cold **mastery surface** (`-log2 q`); a **Flip the world** button
switches the hidden rover regime (sticky `s=0.7` ↔ memoryless `s=0`) WITHOUT resetting the
agent (re-warms `agent.t` — surprise-gated plasticity), dropping a "world changed" `VLine`.

**Re-verify.** In-browser: Gym → Run → needle parks BORED, heatmap cools → click *Flip the
world* → needle slams SURPRISED, VLine drops, curve climbs off the floor then re-descends to
the new floor as the heatmap re-cools. Switch source to **Uniform** → never bored.

## D2 — Frozen-vs-Learning control (`Mode::Gym`) · commit `f5bd44e` · VERIFIED

**Claim.** A frozen twin (`eta0 = 0`, never learns), scored on the SAME stream, stays at
exactly `log2(N) = 2` bits; the gap controls for source difficulty (the learner's gain is
learning, not luck). Gray "frozen (no learning)" line beside the blue learner; a "bits saved
vs frozen" odometer accumulates the gap.

**Oracle (`ui/gym.rs` test).** `learner_beats_frozen_baseline`: after 60k steps the **frozen
mean is exactly 2.0 bits** (±1e-9), the **learner is < 1.5 bits**, and **> 5000 bits** have
been banked. **Observed in test:** all three hold.

**Re-verify.** `cargo test -p cogspike-app`. In-browser: the gray line stays flat at 2.0 while
the blue line dives; the "bits saved vs frozen" stat climbs.

## D3 — Dreaming Camera (`Mode::Events`) · commit `b2cc2e3` · VERIFIED (logic) / live-served

**Claim.** A **CUT SENSORY INPUT** toggle runs the predictor open-loop: it stops observing and
glides its own expectation forward at the velocity believed at dream onset. The dreamed bar
(violet) is drawn over the real, unseen bar (faint ghost); flipping the velocity slider makes
the real bar reverse while the dream sails on, *confidently wrong*. The dreamed events ARE the
prediction, so the **residual is identically zero** ("nothing surprises a dreamer") — the
residual raster goes black.

**Oracle (`dvs.rs` tests).** `dream_has_zero_residual_and_peels_away_on_reversal`: every dream
frame's residual is all-zero, `resid_total` is unchanged (no learning signal), and after a
reversal the circular divergence reaches **≥ 8 px** (of a 12-px max on the 24-ring).
`dream_glides_at_frozen_belief_not_the_live_slider`: after `begin_dream`, the dreamed bar
advances by the frozen `+1` belief even when the live slider is set to `-3`. **`dvs_parity`
(byte-exact, raw 98 → residual 12) is unaffected** by the added `is_dream` field.

**Re-verify.** `cargo test -p cog_spike --lib dvs::`. In-browser: Event camera → check *CUT
SENSORY INPUT* → the residual raster goes black and a "DREAMING" banner shows the drift; move
the velocity slider and watch the violet dream peel away from the ghost real bar.

## Frame-rate decoupling (user-reported) · commits `1d4d954`, `b2cc2e3` · VERIFIED (logic)

**Claim.** The gym, spike, and event-camera views stepped a fixed amount *per repaint*, so
extra repaints (mouse movement) sped up the animation. All three now step by **real elapsed
time** (`input.stable_dt`, clamped): gym/spikes scale steps by `dt`, the event camera drains a
`0.07 s` accumulator. Animation rate is now mouse-independent.

**Re-verify.** In-browser: Run any live view and wiggle the mouse — the sim speed no longer
changes.
