# `deq/closed_form_neg_loop/followups/` — Three follow-up experiments

Experimental answers to the three open follow-ups raised in §8 of the
parent integrating note
[`closed_form_neg_loop_note.pdf`](../note/closed_form_neg_loop_note.pdf):

1. **Experiment A** — *Dynamic-τ calibration.* Fit τ_dyn from a
   sinusoidal-sweep experiment on a single FCS neuron and check
   whether the recalibrated H(ω) closes Phase 2's factor-of-4 period
   gap.
2. **Experiment B** — *Single-neuron renewal predictor.* Replace
   the quasi-renewal population mesoscopic with a single-neuron
   age-PMF tracker (no √(A/N) noise) and check whether the negative
   loop's binary `1100` waveform emerges.
3. **Experiment C** — *3-neuron extension.* Apply the three lenses
   to an A→D→I→A negative loop with one delayer (FCS DeMaria 2020,
   Fig. 3 style) and report how the period predictions scale.

Each experiment has a Python script + a per-experiment typst report.
A single integrating note `note/followups_note.pdf` synthesizes all
three.

## Run

```sh
./make_all.sh
```

Final PDF: `note/followups_note.pdf`.

## Reuse (no reimplementation)

All primitives lifted as in the parent thread; see
[../README.md](../README.md). New code is inline per-experiment.
