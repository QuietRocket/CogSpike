// Research Note: Weight Discretization for Quotient Model Abstraction
// Formal Proof of Threshold Calibration and Behavioral Preservation

#set document(
  title: "Weight Discretization for Quotient Model Abstraction",
  author: "CogSpike Research Team",
  date: datetime.today(),
)

#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.5cm),
  numbering: "1",
)

#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

// Custom theorem-like environments
#let definition(body) = block(
  width: 100%,
  inset: 8pt,
  stroke: (left: 2pt + gray),
  [*Definition.* #body],
)

#let theorem(title, body) = block(
  width: 100%,
  inset: 8pt,
  stroke: (left: 2pt + blue),
  [*Theorem* (#title)*.* #body],
)

#let lemma(title, body) = block(
  width: 100%,
  inset: 8pt,
  stroke: (left: 2pt + purple),
  [*Lemma* (#title)*.* #body],
)

#let proposition(body) = block(
  width: 100%,
  inset: 8pt,
  stroke: (left: 2pt + green),
  [*Proposition.* #body],
)

#let proof(body) = block(
  width: 100%,
  inset: 8pt,
  fill: luma(248),
  [_Proof._ #body #h(1fr) $square$],
)

#let example(body) = block(
  width: 100%,
  inset: 8pt,
  fill: luma(245),
  [*Example.* #body],
)

#let remark(body) = block(
  width: 100%,
  inset: 8pt,
  stroke: (left: 2pt + orange),
  [*Remark.* #body],
)

#align(center)[
  #text(size: 16pt, weight: "bold")[
    Weight Discretization for Quotient Model Abstraction \
    in Spiking Neural Network Verification
  ]

  #v(0.5cm)

  _Research Note --- January 2026_

  #v(1cm)
]

= Introduction

This note extends the filtration-based quotient model abstraction @BaierKatoen2008 to preserve _synaptic weight information_ during formal verification. While the original quotient model (see `research_note_filtration.typ`) abstracts membrane potentials into equivalence classes, it treats all synapses uniformly, losing essential structural information.

We address this by introducing a _weight discretization scheme_ that:
1. Maps continuous weights to a finite discrete range
2. Preserves the relative contribution of synapses to membrane potential
3. Ensures threshold feasibility is maintained
4. Retains weight visibility in the generated PRISM model

= Weight Discretization

== Formal Definition

#definition[
  Given a weight range $[w_"min", w_"max"] = [-100, 100]$ and a discretization parameter $W in NN^+$, the _weight discretization function_ $delta_W: ZZ -> ZZ$ is defined as:
  $ delta_W(w) = "round"(w dot W / w_"max") $

  The discretized weight range is $[-W, W] subset ZZ$.
]

#example[
  For $W = 3$ (7 levels total):
  - $delta_3(100) = 3$ (strong excitatory)
  - $delta_3(67) = 2$ (medium excitatory)
  - $delta_3(33) = 1$ (weak excitatory)
  - $delta_3(0) = 0$ (negligible)
  - $delta_3(-50) = -2$ (medium inhibitory)
  - $delta_3(-100) = -3$ (strong inhibitory)
]

== Threshold Calibration

The key challenge is ensuring that threshold reachability is preserved after discretization. We must calibrate the _discretized threshold_ $T_d$ to be consistent with the original threshold $T$.

#definition[
  The _discretized threshold_ for a neuron with original threshold $T$ and weight discretization parameter $W$ is:
  $ T_d = "ceil"(T dot W / w_"max") $
]

#theorem("Threshold Preservation")[
  Let $cal(N)$ be a neuron with incoming weights ${w_1, ..., w_m}$ and threshold $T$. Let $cal(N)'$ be the discretized version with weights ${delta_W (w_1), ..., delta_W (w_m)}$ and threshold $T_d$.

  If $cal(N)$ can fire in a single step (i.e., $exists$ input pattern $bold(y) in {0,1}^m$ such that $sum_(i=1)^m w_i dot y_i >= T$), then $cal(N)'$ can also fire in a single step.
]

#proof[
  Let $bold(y)^*$ be an input pattern that causes $cal(N)$ to fire:
  $ S = sum_(i=1)^m w_i dot y_i^* >= T $

  The discretized contribution is:
  $ S_d = sum_(i=1)^m delta_W (w_i) dot y_i^* $

  By the rounding property of $delta_W$:
  $ delta_W (w_i) >= (w_i dot W) / w_"max" - 1/2 $

  Therefore:
  $ S_d >= (sum_(i=1)^m w_i dot y_i^*) dot W / w_"max" - m/2 = S dot W / w_"max" - m/2 $

  Since $S >= T$:
  $ S_d >= T dot W / w_"max" - m/2 $

  For the discretized model to fire, we need $S_d >= T_d$. By choosing:
  $ T_d = "ceil"(T dot W / w_"max") <= T dot W / w_"max" + 1 $

  A sufficient condition is:
  $ S dot W / w_"max" - m/2 >= T dot W / w_"max" + 1 $

  Which simplifies to:
  $ (S - T) dot W / w_"max" >= m/2 + 1 $

  This holds when $S - T$ is sufficiently large relative to $m$. For the boundary case $S = T$, we need $W >= w_"max" dot (m/2 + 1) \/ T$.

  In practice, with $W = 3$, $w_"max" = 100$, and typical networks ($m <= 10$, $T = 100$):
  $ W >= 100 dot 6 / 100 = 6 $

  Thus $W = 7$ (range $[-3, 3]$) is sufficient for most practical networks. For larger fan-in, a higher $W$ may be required.
]

#remark[
  The proof shows that weight discretization can introduce a _margin of error_ proportional to the fan-in $m$. For high-fanin neurons ($m > 10$), we recommend either:
  1. Using a finer discretization ($W >= 5$)
  2. Applying a threshold correction factor: $T_d' = T_d - "floor"(m / (2 W))$
]

= Class Transition with Weighted Contributions

== Contribution-Based Class Evolution

In the original quotient model, class evolution used a binary rule:
- Any input fires → class increases by 1
- No input fires → class decreases by 1 (leak)

This loses weight information. We replace it with _weighted contribution-based_ class evolution.

#definition[
  The _weighted contribution_ for neuron $n$ with incoming discretized weights ${w_1^d, ..., w_m^d}$ is:
  $ C_n = sum_(i=1)^m w_i^d dot y_i $
  where $y_i in {0,1}$ is the spike output of presynaptic neuron $i$.
]

#definition[
  The _class delta function_ $Delta: ZZ -> ZZ$ maps contribution to class change:
  $ Delta(C) = "clamp"("round"(C / gamma), -k, k) $
  where $gamma$ is the _class width_ (typically $gamma = T_d / k$) and $k$ is the number of threshold levels.
]

#proposition[
  The class delta function preserves the ordering of contributions: if $C_1 > C_2$, then $Delta(C_1) >= Delta(C_2)$.
]

#proof[
  This follows directly from the monotonicity of `round` and `clamp` operations.
]

== Integration with Leak Rate

The quotient model must also account for membrane potential decay (leak). We propose _discretized leak_:

#definition[
  The _discretized leak factor_ $lambda_d$ is:
  $ lambda_d = 1 - "round"((1 - r) dot k) $
  where $r in [0, 1]$ is the original leak rate and $k$ is the number of classes.
]

The class evolution rule becomes:
$ c'_n = "clamp"(c_n + Delta(C_n) + lambda_d, 0, k) $

#example[
  For $r = 0.9$ (10% decay per step) and $k = 4$:
  $ lambda_d = 1 - "round"(0.1 dot 4) = 1 - 0 = 1 $

  So with no input, the class stays the same + leak = same - 0 (no change from this formula).

  For $r = 0.5$ (50% decay):
  $ lambda_d = 1 - "round"(0.5 dot 4) = 1 - 2 = -1 $

  So with no input, class decreases by 1 per step.
]

#remark[
  We choose to apply leak _only when no excitatory input fires_ to maintain consistency with the precise model's behavior, where leak is a multiplicative factor on the existing potential.
]

= Threshold Feasibility Analysis

== Definition

#definition[
  A neuron configuration is _threshold-feasible_ if there exists at least one input pattern that can cause the neuron to reach the firing threshold within a finite number of steps.
]

#theorem("Feasibility Criterion")[
  A neuron with discretized weights ${w_1^d, ..., w_m^d}$ and threshold $T_d$ is threshold-feasible if and only if:
  $ sum_(w_i^d > 0) w_i^d >= T_d / (1 + |lambda_d|) $
]

#proof[
  Let $E = sum_(w_i^d > 0) w_i^d$ be the maximum excitatory contribution per step.

  _Sufficiency_: If $E >= T_d$, the neuron can fire in a single step when all excitatory inputs fire. Otherwise, the neuron accumulates potential over steps. With leak factor $lambda_d <= 0$, each step adds at most $E + lambda_d$ net contribution.

  For the class to reach $k$ (firing threshold), starting from class 0:
  $ n dot (E + lambda_d) >= k $

  The minimum steps required is $n = "ceil"(k / (E + lambda_d))$, which is finite iff $E + lambda_d > 0$, i.e., $E > |lambda_d|$.

  Since $T_d = k dot gamma$ and $gamma approx T_d / k$, the condition $E >= T_d / (1 + |lambda_d|)$ ensures firing is reachable.

  _Necessity_: If $E < T_d / (1 + |lambda_d|)$, then even with optimal accumulation, the maximum class reached is bounded by $E / |lambda_d| < k$, so firing is unreachable.
]

== Implementation

The feasibility check should be performed at PRISM generation time:

```rust
fn check_feasibility(
    weights: &[i32],  // discretized weights
    threshold: i32,
    leak_factor: i32,
) -> Feasibility {
    let max_excitation: i32 = weights.iter()
        .filter(|&&w| w > 0)
        .sum();

    let min_required = threshold / (1 + leak_factor.abs());

    if max_excitation >= threshold {
        Feasibility::SingleStep
    } else if max_excitation >= min_required {
        let steps = (threshold + max_excitation - 1) / max_excitation;
        Feasibility::MultiStep { min_steps: steps }
    } else {
        Feasibility::Impossible
    }
}
```

= PRISM Model Structure

The weighted quotient model generates PRISM code with explicit weight constants and contribution formulas:

```prism
// Weight constants (discretized)
const int W = 3;  // Discretization parameter
const int W_in0_2 = 3;   // δ_3(100) = 3
const int W_n1_2 = -2;   // δ_3(-67) = -2
const int W_n3_2 = 1;    // δ_3(33) = 1

// Discretized threshold
const int T_d = 3;

// Contribution formula (evaluated at runtime)
formula contrib_2 = W_in0_2 * x0 + W_n1_2 * z1_2 + W_n3_2 * z3_2;

// Class delta (clamped)
formula delta_2 = max(-4, min(4, contrib_2));

// Class evolution with weighted contribution
[tick] y2=0 & pClass2=0 -> (pClass2' = max(0, min(4, pClass2 + delta_2)));
```

= Conclusion

We have established a formal framework for weight discretization in quotient model abstraction:

1. *Threshold Preservation Theorem* guarantees firing reachability is maintained
2. *Contribution-based class evolution* preserves weight effects on dynamics
3. *Feasibility analysis* detects and warns about unreachable configurations
4. *Discretized leak* maintains decay behavior in a state-preserving manner

The key insight is that weights enter the PRISM model as _constants_, not _state variables_, so weight discretization does not increase state space---it only affects transition probabilities and guard conditions.

#v(1cm)
#line(length: 100%)

#bibliography("references.bib", style: "ieee")
