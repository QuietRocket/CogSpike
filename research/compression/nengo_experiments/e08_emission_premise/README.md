# e08 — The emission premise (the paper's central open problem)

Empirically narrows — does not prove — the paper's headline conditional identity
("cost-spike = gradient"). The gradient *algebra* `−∂ℓ/∂W_ij = (1/ln2) c_i (y_j−q_j)`
is unconditional; the *physical* identity requires the circuit to **emit exactly** the
signed residual `y−q`, realized by two rectified ON/OFF error channels
`r⁺=max(0,y−q)`, `r⁻=max(0,q−y)`, `r=r⁺−r⁻`. The paper flags a circuit-level
derivation as **open**. Three parts: **(a)** a real spiking ON/OFF population (two
rectified `intercepts∼U(0,1)`, `encoders=+1` ensembles) emits `y−q` with **RMS error
0.0051**, concentrated 7× at the rectification kink (0.0120 near vs 0.0017 far), with
matched ON/OFF gains (imbalance 0.001). **(b)** Feeding the *corrupted* emitted
residual into a fast numpy delta-rule learner maps a **failure boundary**: convergence
survives gain mismatch down to **g_crit=0.60** and a dead-zone up to **θ_r,crit=0.150**;
loop delay is benign at the working lr (breaks only at aggressive lr=0.6, d_crit=10).
The **real spiking emission lands firmly inside the converges region** — driving the
learner with the *measured* emission curve still reaches q=P (excess KL 0.0381 < fail
0.104). **The gradient identity is robust to realistic emission imperfection; it does
not require exact y−q.** **(c)** The latency code and the ON/OFF error channel are
**distinct observables** with **zero** cross-talk (0.0000 ms) — the paper's
distinct-observables caveat. What remains open: the circuit-level *derivation*; e08
shows only that the fidelity bar it must clear is modest. 10/10 acceptance checks pass.
