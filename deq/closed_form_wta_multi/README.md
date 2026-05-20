# `deq/closed_form_wta_multi/` — N-neuron WTA closed-form study

Extension of the [2-neuron `deq/closed_form_wta/`](../closed_form_wta/)
thread to **arbitrary N**. Uniform all-to-all lateral inhibition with one
scalar weight `w` (rather than the two weights `w_12`, `w_21` of the
contralateral motif), swept across `N ∈ {2, 3, 4, 6, 10}`.

The same three lenses (Siegert / Richardson H(ω) / Naud–Gerstner
quasi-renewal) are recompared against the FCS LI&F oracle on the new
`(w, N)` plane. The cleanly-symmetric topology admits an **exact
permutation-orbit decomposition** of the rate-equation fixed-point set
(`k` winners, `N − k` losers), which replaces the 2-neuron 1-D scalar
reduction.

## Layout

- `phase0_fcs_baseline_multi.py` — FCS LI&F oracle on `(w, N, drive_bump)` grid.
- `phase1_siegert_orbits.py` — Siegert orbit enumeration; WTA-capable iff a
  `k=1` orbit exists with `ν_W − ν_L ≥ 0.30`.
- `phase2_latency_gate_multi.py` — H(ω) Jacobian-eigenvalue gate at the
  `k=1` orbit; closed-form `(N−2)`-fold loser-symmetric eigenvalue.
- `phase3_finite_N_multi.py` — quasi-renewal at `N_pop ∈ {50, 100, 500, 2000}`
  for each `N_neurons`.
- `note/closed_form_wta_multi_note.typ` — final integrating note.
- `results/phase{0..3}/` — per-phase numerical outputs and plots.
- `make_all.sh` — runs all phases end-to-end.

## Reuse

Lifts (no reimplementation):

- FCS LI&F oracle: [deq/archetypes/lif_fcs.py](../archetypes/lif_fcs.py)
- N-neuron topology: [deq/archetypes/topologies.py:all_to_all_inhibition](../archetypes/topologies.py)
- Siegert + orbit FP solver: [deq/closed_form/siegert.py:find_all_fixed_points_uniform_inhibition](../closed_form/siegert.py)
- H(ω) + N×N Jacobian: [deq/closed_form/transfer.py](../closed_form/transfer.py)
- Quasi-renewal: [deq/closed_form/quasi_renewal.py:simulate_uniform_inhibition](../closed_form/quasi_renewal.py)
- Calibration constants: from [deq/closed_form/results/phase1_grid.npz](../closed_form/results/phase1_grid.npz)

## Run

```sh
./make_all.sh
```

Final PDF: `note/closed_form_wta_multi_note.pdf`.

## Cross-link

The 2-neuron prequel: [`deq/closed_form_wta/`](../closed_form_wta/), final
PDF `closed_form_wta/note/closed_form_wta_note.pdf`. Phase 0 of this
thread at N=2, drive_bump=0 reproduces the diagonal `(w_12, w_21) = (w, w)`
of the 2-neuron Phase 0 (up to the documented gate definition: this
thread uses `second_max ≤ 0.01` while the 2-neuron thread uses
`rate_min ≤ 0.01`, which coincide at N=2).
