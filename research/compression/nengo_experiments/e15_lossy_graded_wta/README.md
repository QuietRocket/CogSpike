# e15 — Learned lossy compression via a graded WTA (rate–distortion)

This capstone relaxes the suite's hard, lossless winner-take-all decode (e05/e11) to a
*graded* / soft argmax — a sub-saturating sum-mode gain (a finite softmax temperature),
i.e. a wide graded bump over the calibrated readout — so the decoder commits to a
*coarsened* symbol, merging near-equiprobable / near-tied-latency outcomes and spending
fewer spikes. The graded-bump width is a rate–distortion knob (a per-context merge
margin δ in bits = latency/λ): sweeping it over the real Nengo readout latency table
produces a clean monotone spiking rate–distortion curve — the lossless point on the
entropy-rate floor (rate 0.9790 bits/symbol, +0.73 mbit dt overhead) at distortion 0,
rate falling to 0.3202 bits/symbol as distortion rises to 0.197. The temperature/gain
knob *is* the merge-radius knob (δ_eff(g) = −log₂r / g); a trainable rate–distortion
Lagrangian ℓ + β·d, descended locally per context, traces the same frontier through
four distinct operating points. The safety contract changes honestly from exact
losslessness to bounded distortion (the structural bound is exhibited and held); the
paper's open problem — a *certifiable* distortion bound at a given spike rate — is left
cleanly stated. Run `uv run python run.py`; 12/12 acceptance checks pass.
