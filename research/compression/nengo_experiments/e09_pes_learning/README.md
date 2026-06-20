# e09 — The local delta rule learns the rover law (PES embodiment)

Tests paper claims #9 + #10: the local three-factor rule `ΔW_ij = η c_i (y_j − q_j)`
is the delta / Widrow–Hoff rule, exactly SGD on the per-symbol surprisal, and Nengo's
PES rule *is* it (it updates the **decoders** of a spiking population by
`−lr·error·activity`). We wire a 600-neuron context population whose decoded
prediction `q(·|c)` is taught by the realized symbol's one-hot `y`, stream 10,000
momentum-rover symbols, and watch the rule learn the true conditional law.

**Headline:** the learned `q(·|i)` converges to the true rover rows `P` (max|q−P| =
0.152), and the energy descends monotonically from the uniform init (log₂4 = 2.0
bits) below the marginal 1.7500 into a noise ball at **1.030 bits/symbol** (excess KL
0.052). The gradient identity holds to 5.8e−10 (finite-diff) / 0.0 (exact algebraic),
anchoring that PES embodies the exact rule. The excess-vs-numpy gap (0.052 vs 0.0005)
**is the finding**: a constant learning rate converges to a noise ball (not a point),
and the learned object is a *representational* spiking decoder (not a literal `float64`
matrix) — the direction of the paper's claim is exact; the residual is the price of
physicality. The two-channel ON/OFF rectified-error variant reaches the same fixed
point. 8/8 acceptance checks pass.
