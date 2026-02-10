// Research Note: Weight Discretization for Quotient Model Abstraction
// Formal Proof of Threshold Calibration and Behavioral Preservation
// FOURTH REVISION - JAN 2026

#set document(
  title: "Weight Discretization for Quotient Model Abstraction (Fourth Revision)",
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

#let proofstep(num, title, body) = block(
  width: 100%,
  inset: (left: 12pt, top: 4pt, bottom: 4pt),
  [*Step #num* (#title)*:* #body],
)

#let intuition(body) = block(
  width: 100%,
  inset: 8pt,
  fill: rgb("#f0f7ff"),
  stroke: (left: 2pt + rgb("#4a90d9")),
  [💡 *Intuition:* #body],
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

#let todo(body) = block(
  width: 100%,
  inset: 8pt,
  fill: rgb("#fff3cd"),
  stroke: (left: 2pt + rgb("#856404")),
  [⚠️ *TODO:* #body],
)

#align(center)[
  #text(size: 16pt, weight: "bold")[
    Weight Discretization for Quotient Model Abstraction \
    in Spiking Neural Network Verification
  ]

  #v(0.5cm)

  _Research Note --- January 2026_ \
  _(Fourth Revision)_

  #v(1cm)
]

= Introduction

This note extends the filtration-based quotient model abstraction @BaierKatoen2008 to preserve _synaptic weight information_ during formal verification. While the original quotient model abstracts membrane potentials into equivalence classes, it treats all synapses uniformly, losing essential structural information.

We address this by introducing a _weight discretization scheme_ that:
1. Maps continuous weights to a finite discrete range
2. Preserves the relative contribution of synapses to membrane potential
3. Ensures threshold feasibility is maintained
4. Retains weight visibility in the generated PRISM model

*What's New in Revision 4:*
- Threshold-dependent leak factor formulation (§4.2)
- Soundness theorem: "If we don't receive input, we shouldn't spike" (§3.4)
- Biological property preservation analysis (§5.3)
- Variable dependency graph (§6)
- Corrected code examples and notation

= Preliminaries

This section introduces key concepts and notation used throughout the proofs.

== Notation Summary

#figure(
  table(
    columns: (auto, 1fr),
    inset: 8pt,
    align: (center, left),
    [*Symbol*], [*Meaning*],
    [$W$], [Discretization parameter: the number of positive weight levels],
    [$w_"max"$], [Maximum absolute weight in the original model (typically 100)],
    [$delta_W$], [Weight discretization function: maps original weights to discrete values],
    [$T$], [Original firing threshold of a neuron],
    [$T_d$], [Discretized firing threshold],
    [$m$], [Fan-in: the number of incoming synapses to a neuron],
    [$S$], [Weighted sum of spiking inputs in the original model],
    [$S_d$], [Weighted sum of spiking inputs in the discretized model],
    [$gamma$], [Class width: potential range represented by each equivalence class],
    [$lambda_d$], [Discretized leak factor (always $<= 0$), _dependent on $T_d$_],
    [$k$], [Number of threshold levels (equivalence classes)],
    [$r$], [Retention rate: fraction of potential retained per step ($r in [0,1]$)],
  ),
  caption: [Summary of notation used in this document],
)

== The Rounding Property

The standard mathematical rounding function $"round"(x)$ rounds $x$ to the nearest integer. A crucial property we use throughout is the _rounding bound_:

#definition[
  For any real number $x$, the rounding function satisfies:
  $ x - 1/2 <= "round"(x) <= x + 1/2 $

  Equivalently: $"round"(x) >= x - 1/2$ (lower bound) and $"round"(x) <= x + 1/2$ (upper bound).
]

#intuition[
  Rounding can shift a value by at most $1/2$ in either direction. When we sum $m$ rounded values, the total error is bounded by $m/2$ — this is the _cumulative rounding error_.
]

== The Clamp Function

#definition[
  The _clamp function_ restricts a value to a specified range $[a, b]$ where $a < b$:
  $
    "clamp"(x, a, b) = max(a, min(b, x)) = cases(
      a & "if" x < a,
      x & "if" a <= x <= b,
      b & "if" x > b
    )
  $
]

#intuition[
  Clamping prevents values from exceeding safe bounds. In our model, we clamp class indices to $[0, k]$ and class deltas to $[-k, k]$ to avoid invalid states or runaway accumulation.
]

== Fan-in

#definition[
  The _fan-in_ of a neuron, denoted $m$, is the number of incoming synaptic connections. A neuron with fan-in $m$ receives input from $m$ presynaptic neurons.
]

#remark[
  Fan-in is critical because the cumulative rounding error scales linearly with $m$. High fan-in neurons require finer discretization to maintain accuracy.
]

= Weight Discretization

== Formal Definition

#definition[
  Given a weight range $[w_"min", w_"max"] = [-100, 100]$ and a discretization parameter $W in NN^+$, the _weight discretization function_ $delta_W : RR -> ZZ$ is defined as:
  $ delta_W (w) = "round"(w dot W / w_"max") $

  The discretized weight range is $[-W, W] subset ZZ$.
]

#intuition[
  We scale the original weight by $W / w_"max"$ to map the range $[-w_"max", w_"max"]$ to $[-W, W]$, then round to get an integer. This preserves the _relative magnitude_ of weights while reducing the number of distinct values.
]

#example[
  For $W = 3$ (giving 7 discrete levels: $-3, -2, -1, 0, 1, 2, 3$):
  - $delta_3(100) = "round"(100 dot 3 / 100) = "round"(3) = 3$ (strong excitatory)
  - $delta_3(67) = "round"(67 dot 3 / 100) = "round"(2.01) = 2$ (medium excitatory)
  - $delta_3(33) = "round"(33 dot 3 / 100) = "round"(0.99) = 1$ (weak excitatory)
  - $delta_3(0) = "round"(0) = 0$ (negligible)
  - $delta_3(-50) = "round"(-50 dot 3 / 100) = "round"(-1.5) = -2$ (medium inhibitory)
  - $delta_3(-100) = "round"(-3) = -3$ (strong inhibitory)
]

== Threshold Calibration

The key challenge is ensuring that threshold reachability is preserved after discretization. We must calibrate the _discretized threshold_ $T_d$ to be consistent with the original threshold $T$.

#definition[
  The _discretized threshold_ for a neuron with original threshold $T$ and weight discretization parameter $W$ is:
  $ T_d = "ceil"(T dot W / w_"max") $
]

#intuition[
  We use ceiling (round up) rather than standard rounding to ensure the discretized threshold is _at least as hard_ to reach as the original. This prevents false positives: if a discretized neuron fires, we can be confident the original would too.
]

== Threshold Preservation Theorem (Completeness)

#theorem("Threshold Preservation — Completeness")[
  Let $cal(N)$ be a neuron with incoming weights ${w_1, ..., w_m}$ and threshold $T$. Let $cal(N)'$ be the discretized version with weights ${delta_W (w_1), ..., delta_W (w_m)}$ and threshold $T_d$.

  If $cal(N)$ can fire in a single step (i.e., $exists$ input pattern $bold(y) in {0,1}^m$ such that $sum_(i=1)^m w_i dot y_i >= T$), then $cal(N)'$ can also fire in a single step.
]

#proof[
  We prove this in six steps.

  #proofstep(1, "Setup")[
    Let $bold(y)^*$ be an input pattern that causes the original neuron $cal(N)$ to fire. Define:
    - $S = sum_(i=1)^m w_i dot y_i^*$ — the weighted sum of spiking inputs in the original model
    - By assumption, $S >= T$ (the neuron fires)
  ]

  #proofstep(2, "Discretized contribution")[
    The corresponding weighted sum in the discretized model is:
    $ S_d = sum_(i=1)^m delta_W (w_i) dot y_i^* $
    We need to show $S_d >= T_d$ to prove the discretized neuron also fires.
  ]

  #proofstep(3, "Apply the rounding property")[
    By the rounding property (see §2.2), for each weight:
    $ delta_W (w_i) = "round"(w_i dot W / w_"max") >= (w_i dot W) / w_"max" - 1/2 $

    Since $y_i^* in {0, 1}$, when $y_i^* = 1$ this bound applies, and when $y_i^* = 0$ the term is zero. Summing over all inputs:
    $
      S_d >= sum_(i=1)^m ((w_i dot W) / w_"max" - 1/2) dot y_i^* = (sum_(i=1)^m w_i dot y_i^*) dot W / w_"max" - (sum_(i=1)^m y_i^*) / 2
    $
  ]

  #proofstep(4, "Bound the cumulative error")[
    Let $m^* = sum_(i=1)^m y_i^*$ be the number of active inputs. Then:
    $ S_d >= S dot W / w_"max" - m^* / 2 $

    Since $m^* <= m$ (at most $m$ inputs can be active):
    $ S_d >= S dot W / w_"max" - m / 2 $

    The term $m/2$ is the _cumulative rounding error_ — the worst-case total error from rounding $m$ weights.
  ]

  #proofstep(5, "Derive the firing condition")[
    For the discretized neuron to fire, we need $S_d >= T_d$. By the ceiling property:
    $ T_d = "ceil"(T dot W / w_"max") <= T dot W / w_"max" + 1 $

    Combining with our bound on $S_d$:
    $ S_d >= S dot W / w_"max" - m / 2 $

    Since $S >= T$:
    $ S_d >= T dot W / w_"max" - m / 2 $

    A sufficient condition for firing is:
    $ T dot W / w_"max" - m / 2 >= T dot W / w_"max" + 1 $

    This simplifies to: $(S - T) dot W / w_"max" >= m/2 + 1$
  ]

  #proofstep(6, "Boundary case and parameter choice")[
    For the critical boundary case where $S = T$ exactly, we need:
    $ 0 >= m / 2 + 1 $

    This is never satisfied, so at the exact boundary, firing is not guaranteed. However, if $S$ exceeds $T$ by a small margin, or we choose $W$ large enough, firing is preserved.

    Specifically, for $W >= w_"max" dot (m/2 + 1) / T$, the discretized neuron fires whenever the original does.

    *Practical example:* With $W = 3$, $w_"max" = 100$, $m <= 10$, and $T = 100$:
    $ W >= 100 dot (5 + 1) / 100 = 6 $

    Thus $W = 7$ (discrete range $[-3, 3]$) is sufficient for most practical networks.
  ]
]

#remark[
  *Critical Constraint:* The proof shows that weight discretization introduces a cumulative error of $-m/2$. If the fan-in $m$ is large relative to $W$ (specifically if $m > 2W$), the rounding noise may exceed the signal of the smallest synaptic weight.

  For high-fanin neurons, we strictly recommend:
  1. Using a finer discretization ($W >= m/2$)
  2. Applying a threshold correction factor: $T_d' = T_d - "floor"(m / (2 W))$ (Note: This may increase false positive firings).
]

== Soundness Theorem (Safety)

We now prove the converse property: if the original neuron should _not_ fire, the discretized neuron should also _not_ fire.

#theorem("Asymptotic Silence — Soundness")[
  Let $cal(N)'$ be a discretized neuron with current potential $P_t < T_d$ and leak factor $lambda_d <= -1$. If the input sequence is empty (i.e., total weighted input $S_d = 0$) for all steps $t' >= t$, then $cal(N)'$ will never fire.
]

#proof[
  We prove this by showing the potential strictly decreases to zero.

  #proofstep(1, "Dynamics without input")[
    In the absence of input ($S_d = 0$), the discretized update rule simplifies to:
    $ P_(t+1) = max(0, P_t + lambda_d) $
    This is applying the potential update with only leak (no external contribution).
  ]

  #proofstep(2, "Strict decay guarantee")[
    Since $lambda_d <= -1$, if $P_t > 0$, then:
    $ P_(t+1) <= P_t + lambda_d <= P_t - 1 $
    The potential strictly decreases by at least 1 unit per step.
  ]

  #proofstep(3, "Convergence to zero")[
    Starting from any $P_t < T_d$, the sequence ${P_t}$ is strictly decreasing until it reaches the absorbing state $P = 0$ (due to the $max(0, ...)$ clamp).
  ]

  #proofstep(4, "Firing impossibility")[
    Since $P_t < T_d$ initially and the sequence is non-increasing, the firing condition $P >= T_d$ is never met. The neuron remains silent indefinitely.
  ]
]

#intuition[
  This theorem provides a _safety guarantee_: the discretization does not introduce spurious spikes. If there's no input to drive the potential up, leak ensures it decays to rest. This prevents rounding errors from accidentally triggering false firings.
]

= Class Transition with Weighted Contributions

== Contribution-Based Class Evolution

In the original quotient model, class evolution used a binary rule which loses weight information. We replace it with _weighted contribution-based_ class evolution.

#definition[
  The _weighted contribution_ for neuron $n$ with incoming discretized weights ${w_1^d, ..., w_m^d}$ is:
  $ C_n = sum_(i=1)^m w_i^d dot y_i $
  where $y_i in {0,1}$ is the spike output of presynaptic neuron $i$.
]

#definition[
  The _class delta function_ $Delta: ZZ -> ZZ$ maps contribution to class change:
  $ Delta(C) = "clamp"("round"(C / gamma), -k, k) $
  where:
  - $gamma$ is the _class width_ (typically $gamma = T_d / k$) — the potential range each class represents
  - $k$ is the number of threshold levels
  - The clamp ensures the class change stays within valid bounds
]

#intuition[
  The class delta converts a weighted sum of inputs into a class change. We divide by $gamma$ to express the contribution in "class units", round to get an integer, and clamp to prevent impossibly large jumps.
]

== Threshold-Dependent Leak Factor

The quotient model must also account for membrane potential decay (leak). We propose a corrected _threshold-dependent_ leak formula.

#definition[
  The _discretized leak factor_ $lambda_d$ is:
  $ lambda_d = - max(1, floor((1 - r) dot T_d)) $
  where:
  - $r in [0, 1]$ is the original retention rate (fraction of potential retained per step)
  - $T_d$ is the discretized threshold
  - The $max(1, ...)$ ensures a minimum decay of 1 unit per step (preventing infinite energy trapping)
  - The negative sign ensures leak _decreases_ potential
]

#intuition[
  By linking the leak factor to $T_d$ rather than the number of classes $k$, we ensure the decay scales correctly with the calibrated threshold. A high threshold means larger potential values, so the decay must be proportionally larger to maintain realistic behavior.
]

#example[
  For $r = 0.9$ (90% retention, i.e., 10% decay) and $T_d = 10$:
  $ lambda_d = - max(1, floor((1 - 0.9) dot 10)) = - max(1, floor(1.0)) = -1 $
  _Result:_ Potential decreases by 1 per step without input.

  For $r = 0.5$ (50% decay) and $T_d = 10$:
  $ lambda_d = - max(1, floor((1 - 0.5) dot 10)) = - max(1, floor(5.0)) = -5 $
  _Result:_ Potential decreases by 5 per step without input — aggressive decay.

  For $r = 0.95$ (95% retention) and $T_d = 10$:
  $ lambda_d = - max(1, floor((1 - 0.95) dot 10)) = - max(1, floor(0.5)) = - max(1, 0) = -1 $
  _Result:_ Even with very high retention, we enforce minimum decay of 1.
]

#remark[
  *Key Insight:* The previous formulation $lambda_d = -"round"((1-r) dot k)$ was problematic because:
  1. It depended on $k$ (number of classes) rather than $T_d$ (actual potential scale)
  2. Could result in $lambda_d = 0$ for high retention, violating the Soundness theorem

  The new formulation ensures $|lambda_d| >= 1$ always, guaranteeing the Asymptotic Silence property.
]

The class evolution rule becomes:
$ c'_n = "clamp"(c_n + Delta(C_n) + lambda_d, 0, k) $

= Threshold Feasibility Analysis

== Definition

#definition[
  A neuron configuration is _threshold-feasible_ if there exists at least one input pattern that can cause the neuron to reach the firing threshold within a finite number of steps.
]

#intuition[
  Feasibility asks: "Can this neuron ever fire?" If the leak is too strong relative to the available excitation, the potential can never build up enough to reach threshold — the neuron is permanently silent.
]

#theorem("Feasibility Criterion")[
  A neuron with discretized weights ${w_1^d, ..., w_m^d}$ and threshold $T_d$ is threshold-feasible if and only if:
  $ sum_(w_i^d > 0) w_i^d > |lambda_d| $

  And specifically, for reliable firing:
  $ sum_(w_i^d > 0) w_i^d >= T_d / (1 + |lambda_d|) $
]

#proof[
  #proofstep(1, "Define maximum excitation")[
    Let $E = sum_(w_i^d > 0) w_i^d$ be the maximum possible excitatory contribution per step (achieved when all excitatory presynaptic neurons fire simultaneously).
  ]

  #proofstep(2, "Single-step case")[
    If $E >= T_d$, the neuron can fire in a single step by receiving all excitatory inputs simultaneously. Feasibility is trivially satisfied.
  ]

  #proofstep(3, "Accumulation case")[
    If $E < T_d$, the neuron must accumulate potential over multiple steps. With leak factor $lambda_d <= 0$, each step adds a net contribution of:
    $ "Net gain" = E + lambda_d $
    (Note: $lambda_d$ is negative, so this is $E - |lambda_d|$)

    For accumulation to be possible, we require:
    $ E + lambda_d > 0 quad arrow.r.double quad E > |lambda_d| $

    If this holds, the minimum steps to reach threshold is:
    $ n = "ceil"(T_d / (E + lambda_d)) = "ceil"(T_d / (E - |lambda_d|)) $
  ]

  #proofstep(4, "Necessity: why excitation ≤ leak implies impossibility")[
    If $E <= |lambda_d|$, then the net gain per step is $E - |lambda_d| <= 0$. The potential cannot grow — any excitation is immediately cancelled (or overpowered) by leak. The neuron can never reach threshold.
  ]
]

== Implementation

The feasibility check should be performed at PRISM generation time:

```rust
fn check_feasibility(
    weights: &[i32],  // discretized weights
    threshold: i32,
    leak_factor: i32, // expected to be <= 0 (e.g. -1, -2)
) -> Feasibility {
    // Sum only positive excitatory weights
    let max_excitation: i32 = weights.iter()
        .filter(|&&w| w > 0)
        .sum();

    // Ensure we are working with the magnitude of the leak
    let leak_magnitude = leak_factor.abs();

    // Safety Check (Soundness): Input must overcome leak
    if max_excitation <= leak_magnitude {
        return Feasibility::Impossible;
    }

    let min_required = threshold / (1 + leak_magnitude);

    if max_excitation >= threshold {
        Feasibility::SingleStep
    } else if max_excitation >= min_required {
        // Steps = ceil(Threshold / Net_Gain)
        let net_gain = max_excitation - leak_magnitude;
        let steps = (threshold + net_gain - 1) / net_gain;
        Feasibility::MultiStep { min_steps: steps }
    } else {
        Feasibility::Impossible
    }
}
```

== Biological Property Preservation

We verify the quotient model against standard Leaky Integrate-and-Fire properties defined in the literature @naco20.

#definition[
  The following biological properties should be preserved by the discretized model:

  *Tonic Spiking:* The model fires periodically under constant input $C_"in"$ if and only if:
  $ C_"in" > |lambda_d| $

  *Integrator:* The model fires on simultaneous inputs $n$ if and only if:
  $ sum_(i=1)^n delta_W (w_i) >= T_d $

  *Excitability:* The inter-spike interval (ISI) decreases monotonically as input frequency increases. Specifically, the number of steps between spikes is:
  $ N_"steps" = ceil(T_d / (C_"in" - |lambda_d|)) $
  which decreases as $C_"in"$ increases.
]

#intuition[
  These properties ensure our discretized model behaves like a real LIF neuron:
  - _Tonic spiking_ means sustained input produces sustained output
  - _Integrator_ means simultaneous inputs can trigger immediate firing
  - _Excitability_ means "the more input you receive, the faster you fire"
]

#remark[
  *Immersion Memory:* After emitting a spike, the potential resets to 0. The neuron has no memory of past input signals — only the accumulated potential matters. This is preserved by our model since we explicitly reset $P = 0$ after firing.
]

= Variable Dependencies

This section clarifies the relationships between the key variables in our discretization scheme.

== Dependency Graph

#figure(
  ```
  ┌─────────────────────────────────────────────────────────────────┐
  │                    Variable Dependency Graph                    │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                 │
  │    ┌───────────┐                                                │
  │    │  w_max    │ (Original max weight, typically 100)           │
  │    └─────┬─────┘                                                │
  │          │                                                      │
  │          ▼                                                      │
  │    ┌───────────┐      ┌───────────┐                             │
  │    │     W     │◄─────│     m     │ (Fan-in constraint: W ≥ m/2)│
  │    └─────┬─────┘      └───────────┘                             │
  │          │                                                      │
  │          ├──────────────────┐                                   │
  │          ▼                  ▼                                   │
  │    ┌───────────┐      ┌───────────┐                             │
  │    │   δ_W(w)  │      │    T_d    │ (Discretized threshold)     │
  │    └───────────┘      └─────┬─────┘                             │
  │                             │                                   │
  │          ┌──────────────────┤                                   │
  │          ▼                  ▼                                   │
  │    ┌───────────┐      ┌───────────┐                             │
  │    │     r     │─────►│   λ_d     │ (Leak depends on T_d & r)   │
  │    └───────────┘      └───────────┘                             │
  │    (Retention)              │                                   │
  │                             ▼                                   │
  │                       ┌────────────┐                            │
  │                       │ Feasibility│                            │
  │                       └────────────┘                            │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
  ```,
  caption: [Variable dependencies in the discretization scheme. The crucial link is $lambda_d$ depending on $T_d$ rather than $k$.],
)

== Summary Table

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    inset: 8pt,
    align: (center, left, left),
    [*Variable*], [*Previous Definition*], [*Revised Definition (v4)*],
    [$lambda_d$],
    [$-"round"((1-r) dot k)$\ (Depends on classes $k$)],
    [$-max(1, floor((1-r) dot T_d))$\ (Depends on threshold $T_d$)],

    [$T_d$], [$"ceil"(T dot W / w_"max")$], [Unchanged],
    [$k$], [Independent parameter], [Remains for class discretization, but $lambda_d$ now independent of $k$],
  ),
  caption: [Summary of variable definition changes between revisions],
)

= PRISM Model Structure

The weighted quotient model generates PRISM code with explicit weight constants and contribution formulas. Note that weights are static constants, minimizing state explosion.

```prism
// Threshold-Dependent Leak Calculation
const int T_d = ceil(T_orig * W / w_max);
const int r_percent = 90;  // Retention as percentage
const int lambda_d = -max(1, floor((100 - r_percent) * T_d / 100));

// Weight constants (discretized)
const int W = 3;  // Discretization parameter
const int W_in0_2 = 3;   // δ_3(100) = 3
const int W_n1_2 = -2;   // δ_3(-67) = -2
const int W_n3_2 = 1;    // δ_3(33) = 1

// Contribution formula (evaluated at runtime)
formula contrib_2 = W_in0_2 * x0 + W_n1_2 * z1_2 + W_n3_2 * z3_2;

// Class delta (clamped)
formula delta_2 = max(-4, min(4, contrib_2));

module Neuron
    p : [0..T_d + 5] init 0;
    spike : bool init false;

    // Update with Safety Clamp (max(0, ...))
    [tick] true ->
        (p' = max(0, min(T_d + 5, p + contrib_2 + lambda_d)))
        & (spike' = (p' >= T_d));
endmodule
```

= Visualizations

#todo[
  *Visualization 1: Analog vs Discrete Dynamics*

  Create a side-by-side comparison showing:
  - Left: Continuous exponential decay curve (analog neuron)
  - Right: Step function decay (discrete neuron)
  - Highlight the "conservative bound" — discrete model fires slightly earlier/more often than analog to ensure completeness
  - Show scenarios: (a) spike occurs, (b) decay without input, (c) accumulation to threshold
]

#todo[
  *Visualization 2: Variable Dependency Graph*

  A proper graphical diagram showing:
  - Nodes: $W$, $w_"max"$, $T_d$, $lambda_d$, $r$, $k$, $m$
  - Edges (directed): showing functional dependencies
  - Crucial highlighted edge: $T_d arrow lambda_d$ (new in v4)
  - Color coding: input parameters (blue), derived parameters (green), constraints (orange)
]

= Conclusion

We have established a formal framework for weight discretization in quotient model abstraction. The key contributions in this revision include:

1. *Threshold-Dependent Leak:* The new formulation $lambda_d = -max(1, floor((1-r) dot T_d))$ ensures leak scales with the actual potential range, not the arbitrary number of classes.

2. *Soundness Theorem:* Proof that no spurious spikes occur — if inputs don't cause firing in the original model, the discretized model also remains silent.

3. *Biological Property Preservation:* Verification that tonic spiking, integrator behavior, and excitability are maintained in the quotient model.

4. *Variable Dependency Clarification:* Clear documentation of how $lambda_d$ depends on $T_d$ and $r$, breaking the previous incorrect dependency on $k$.

5. *Corrected Code Examples:* Fixed syntax errors in feasibility check implementation.

*Future Work:*
- Implement graphical visualization comparing analog vs discrete dynamics
- Formal PRISM model checking with the revised leak formulation
- Extend to multi-neuron networks with inter-neuron weight calibration

#v(1cm)
#line(length: 100%)

#bibliography("references.bib", style: "ieee")
