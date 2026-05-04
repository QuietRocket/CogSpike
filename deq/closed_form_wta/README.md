# `deq/closed_form_wta/` — Closed-form reproduction of FCS Property 7

Closed-form rate-equation reading of the De Maria et al. 2020 FCS paper's
**winner-takes-all (Property 7)** verification on the 2-neuron
contralateral inhibition motif (their Fig. 10). Three lenses
(Siegert / Richardson H(ω) / Naud–Gerstner quasi-renewal) applied to the
same integer (w_12, w_21) grid, calibrated against the prior
[deq/closed_form/](../closed_form/) thread.

## Layout

- `phase0_fcs_baseline.py` — FCS-accurate oracle on the integer grid;
  produces the ground-truth blue/red WTA-in-4-ticks labels.
- `phase1_siegert_wta.py` — Siegert FP enumeration → WTA-capable labels.
- `phase2_latency_gate.py` — H(ω) Jacobian eigenvalue contour reading
  of FCS's 4-tick gate.
- `phase3_finite_N.py` — quasi-renewal mesoscopic at N ∈ {50, 100,
  500, ∞}.
- `note/closed_form_wta_note.typ` — final integrating note.
- `results/phase{0..3}/` — per-phase numerical outputs and plots.
- `make_all.sh` — runs all phases end-to-end.

## Reuse

Lifts (no reimplementation):

- FCS LI&F oracle: [deq/archetypes/lif_fcs.py](../archetypes/lif_fcs.py)
- CI topology: [deq/archetypes/topologies.py:contralateral](../archetypes/topologies.py)
- Siegert + FP solver: [deq/closed_form/siegert.py](../closed_form/siegert.py)
- H(ω): [deq/closed_form/transfer.py](../closed_form/transfer.py)
- Quasi-renewal: [deq/closed_form/quasi_renewal.py](../closed_form/quasi_renewal.py)
- Calibration constants: from [deq/closed_form/results/phase0_v02.npz](../closed_form/results/phase0_v02.npz)

## Run

```sh
./make_all.sh
```

Final PDF: `note/closed_form_wta_note.pdf`.
