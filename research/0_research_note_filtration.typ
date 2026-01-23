// Research Note: State Space Reduction for Spiking Neural Network Verification
// via Filtration-Based Model Abstraction

#set document(
  title: "State Space Reduction for SNN Verification via Filtration",
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

#let example(body) = block(
  width: 100%,
  inset: 8pt,
  fill: luma(245),
  [*Example.* #body],
)

#align(center)[
  #text(size: 16pt, weight: "bold")[
    State Space Reduction for Spiking Neural Network Verification \
    via Filtration-Based Model Abstraction
  ]

  #v(0.5cm)

  _Research Note --- January 2026_

  #v(1cm)
]

= Introduction

Probabilistic model checking provides formal guarantees for systems exhibiting stochastic behavior. When applied to Spiking Neural Networks (SNNs), we encode the network dynamics as a Discrete-Time Markov Chain (DTMC) and verify properties expressed in Probabilistic Computation Tree Logic (PCTL). However, the _state explosion problem_ renders verification intractable for networks with more than a handful of neurons.

This note presents an approach based on _filtration_ from modal logic to construct the smallest PCTL-equivalent model, dramatically reducing the state space while preserving all properties of interest.

= Problem Statement

== SNN as a DTMC

Consider an SNN with $n$ neurons. Each neuron $i$ has:
- Membrane potential $p_i in [P_"min", P_"max"] subset ZZ$
- Spike output $y_i in {0, 1}$
- Refractory state $s_i in {0, 1, 2}$ (optional)

The state space of the full model is:
$ |S| = product_(i=1)^n (|P_"max" - P_"min" + 1|) dot 2 dot 3 $

For a 6-neuron network with potential range $[0, 200]$ and refractory periods enabled:
$ |S| = (201 dot 2 dot 3)^6 approx 1.8 times 10^{18} $

Even with reachability analysis reducing this to $tilde 10^6$ states, verification time remains prohibitive for iterative tasks such as weight learning.

== Current Optimizations

We have implemented several engineering optimizations in the CogSpike system:
1. *Per-neuron potential bounds*: Computing $P_"max"^i$ based on incoming weights
2. *Disabled refractory periods*: Removing state variables when not needed
3. *Reduced threshold levels*: Using 4 instead of 10 probabilistic firing bands

These reduce the theoretical state space but do not exploit the fundamental _semantic equivalence_ between states.

= Theoretical Foundation

== Bisimulation Equivalence

Two states $s, s'$ in a Kripke structure $cal(M)$ are _bisimilar_ (written $s tilde.op s'$) if:
1. They satisfy the same atomic propositions: $L(s) = L(s')$
2. For every transition $s arrow.r t$, there exists $s' arrow.r t'$ with $t tilde.op t'$ (and vice versa)
3. The transition probabilities to equivalence classes are preserved

The _bisimulation quotient_ $cal(M) slash tilde$ collapses all bisimilar states, yielding the smallest model that satisfies the same PCTL#super[\*] formulas @BaierKatoen2008.

== Filtration

Filtration, introduced in modal logic @BlackburnEtAl2001, constructs a finite model from a potentially infinite one by considering only distinctions relevant to a finite set of formulas $Sigma$.

#definition[
  Given a Kripke model $cal(M) = (W, R, V)$ and a subformula-closed set $Sigma$, the _filtration_ of $cal(M)$ through $Sigma$ is:
  $ cal(M)^f = (W^f, R^f, V^f) $
  where:
  - $W^f = W \/ approx_Sigma$ with $w approx_Sigma w' <==> forall phi in Sigma: cal(M), w tack.r.double phi <=> cal(M), w' tack.r.double phi$
  - $V^f([w]) = V(w) inter "Props"(Sigma)$
  - $R^f$ satisfies: if $w R w'$ then $[w] R^f [w']$ (smallest filtration)
]

#theorem("Filtration Theorem")[
  For all $phi in Sigma$ and $w in W$:
  $ cal(M), w tack.r.double phi <==> cal(M)^f, [w] tack.r.double phi $

  The size of $cal(M)^f$ is bounded by $2^(|Sigma|)$, independent of $|W|$. #cite(<BlackburnEtAl2001>)
]

= Application to SNN Verification

== Membrane Potential Equivalence

For PCTL properties concerning spike events (e.g., $P_( =?) [F^(<=T) "spike"_i]$), the membrane potential matters only insofar as it determines _firing probability_.

With $k$ threshold levels, we define:
$ p approx_k p' <==> "fire_prob"(p) = "fire_prob"(p') $

This partitions the potential range into $k+1$ equivalence classes:
$
  C_0 & = {p : p < theta_1} quad                && "(no firing)" \
  C_j & = {p : theta_j <= p < theta_(j+1)} quad && "(fires with prob " j\/k")" \
  C_k & = {p : p >= theta_k} quad               && "(certain firing)"
$

== State Space Reduction

#proposition[
  For a network with $n$ neurons and $k$ threshold levels, the filtrated state space has size:
  $ |S^f| = (k+1)^n dot 2^n $
  compared to the original:
  $ |S| = |P|^n dot 2^n dot 3^n $
  where $|P|$ is the potential range size.
]

#example[
  For the 6-neuron double-diamond network with $|P| = 201$ and $k = 4$:
  - Original: $201^6 dot 2^6 dot 3^6 approx 4 times 10^{18}$ (theoretical)
  - With disabled refractory: $201^6 dot 2^6 approx 4 times 10^{15}$
  - With filtration ($k=4$): $5^6 dot 2^6 approx 10^6$

  This is a *reduction factor of $10^9$* over the disabled-refractory model.
]

== Transition Probability Preservation

The critical requirement is that the quotient model preserves transition probabilities. Define $P^f: S^f times S^f arrow.r [0,1]$ as:
$ P^f([s], [t]) = sum_(t' in [t]) P(s, t') $

For our SNN abstraction, if state $s$ has potential class $C_j^i$ for neuron $i$:
- Neuron fires with probability $j\/k$
- Upon firing, potential resets to $C_0$
- Upon not firing, potential evolves based on inputs

The key observation is that the _next potential class_ depends only on:
1. Current potential class (not exact value)
2. Input spike pattern
3. Weight configuration

This ensures the quotient is well-defined.

= Proposed Implementation

== Abstract PRISM Model Generation

Instead of generating:
```prism
p2 : [0..200] init 0;
[tick] p2 >= threshold4 -> 1.0 : (y2'=1) & (p2'=0);
```

Generate the abstracted model:
```prism
pClass2 : [0..4] init 0;  // 5 equivalence classes
[tick] pClass2 = 4 -> 1.0 : (y2'=1) & (pClass2'=0);
[tick] pClass2 = 3 -> 0.75 : (y2'=1) & (pClass2'=0)
                    + 0.25 : (y2'=0) & (pClass2'=nextClass2);
```

== Computing Class Transitions

The function $"nextClass": C times 2^I arrow.r cal(D)(C)$ maps current class and input pattern to a distribution over next classes:

$
  "nextClass"(C_j, overline(x)) = cases(
    C_("clamp"(j + delta)) & "if neuron doesn't fire",
    C_0 & "if neuron fires"
  )
$

where $delta$ is determined by the weighted input sum and leak rate.

= Research Contributions

This approach offers several potential contributions:

1. *Formal framework*: Applying filtration theory to probabilistic SNN models
2. *Correctness proof*: Showing PCTL property preservation for spike-related queries
3. *Implementation*: Automated quotient model generation from SNN specifications
4. *Evaluation*: Empirical comparison of state space size and verification time

= Conclusion

The state explosion problem in SNN verification can be addressed not just through engineering optimizations, but through principled model abstraction based on modal logic filtration. By identifying states equivalent with respect to firing behavior, we can construct quotient models with dramatically reduced state spaces while provably preserving PCTL properties.

Future work includes implementing the quotient generator, proving correctness formally, and evaluating on benchmark networks.

#v(1cm)
#line(length: 100%)

#bibliography("references.bib", style: "ieee")
