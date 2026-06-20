# e07 — Predictive subtraction: the one operation that needs a cycle

**Tier 2.** The paper's claim #5 + FIX F (the sign): the recurrent loop cancels the
*expected* drive via **inhibitory** feedback — not an additive depolarization "equal
to the prediction" — so the soma integrates only the *unpredicted residual*
`J_eff = J_in − J_pred`. A well-predicted symbol meets balanced E/I and is near-silent;
a surprising one fires. The headline is **impossible to see in numpy**: the sign of the
feedback is a bookkeeping choice in algebra but real physics in a spiking soma
(`V≥0`, fire at `θ`). Measured here in a real Nengo LIF: inhibitory feedback *silences*
well-predicted symbols (q≥0.95 → no spike) while the wrong (additive) sign doubles the
drive and fires *earlier* than baseline (1.4 vs 2.70 ms) — runaway, no cancellation.
Over a 120-symbol rover stream the residual spike train becomes a **surprise stream**:
output correlates with surprisal at r=+0.94 but with the raw symbol at only +0.24, and
31% of windows (the well-predicted ones, mean 0.234 bits) are silenced vs 1.453 bits
for the firing ones. The shunting-balance idealization is characterized honestly: a
divisive conductance matched to the subtractive form at the steady state (to 1e-15)
still fires early via the `1/(1+g)` time-constant rescaling — the clean
`input − prediction` arithmetic holds only at small conductance (departure ≤0.10 for
g≤0.10, q≤0.09) and breaks to 0.81 near balance. **8/8 acceptance checks pass.**

Run: `uv run python run.py` → `results/` + four PDFs; `typst compile --root .. report.typ report.pdf`.
