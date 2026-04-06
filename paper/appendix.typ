// Appendix — Extended Proofs and Derivations
//
// This file is meant to be #include'd from shorter.typ.
// It contains the complete formal proofs and technical sections
// that accompany the main paper. For brevity, only theorem statements
// appear in the main text; full derivations are provided here.

#import "llncs.typ": *
#set math.equation(numbering: "(1)")


// ═══════════════════════════════════════════════════════════════════════════════
// A. Complete Formal Proofs
// ═══════════════════════════════════════════════════════════════════════════════

= Appendix: Complete Formal Proofs <sec-appendix-proofs>

The main text states two central formal guarantees for the
weight-discretized quotient abstraction: Threshold Preservation
and Asymptotic Silence. This appendix provides the complete proofs.

== Threshold Preservation <sec-app-completeness>

#theorem[\
  *(Threshold Preservation.)*
  Let $cal(N)$ be a neuron with weights ${w_1, ..., w_m}$ and threshold $T$.
  If $cal(N)$ can fire in a single step (i.e., $exists bold(y) in {0,1}^m$ such
  that $sum_(i=1)^m w_i dot.c y_i >= T$), then the discretized neuron $cal(N)'$
  with weights ${delta_W (w_1), ..., delta_W (w_m)}$ and threshold $T_d$ can
  also fire.
]

#proof[\
  Let $bold(y)^*$ be a firing input pattern with weighted sum
  $S = sum w_i y_i^* >= T$, and let $m^* = sum y_i^* <= m$ denote the number of
  active inputs. By definition of rounding to the nearest integer,
  $delta_W(w_i) = lr(⌊ w_i dot.c W \/ w_"max" ⌉)$ satisfies
  $ delta_W(w_i) >= w_i dot.c W \/ w_"max" - 1\/2 $ <eq-app-rounding-lb>
  Multiplying @eq-app-rounding-lb by $y_i^* in {0,1}$ and summing over all inputs:
  $
    S_d = sum_(i=1)^m delta_W(w_i) dot.c y_i^*
    >= sum_(i=1)^m w_i y_i^* dot.c W \/ w_"max" - 1\/2 sum_(i=1)^m y_i^*
    = S dot.c W \/ w_"max" - m^* \/ 2
  $ <eq-app-sd-bound>
  Each active input contributes at most $-1\/2$ of rounding error, yielding
  a cumulative shortfall of $-m^*\/2$. Since
  $T_d = ceil(T dot.c W \/ w_"max") <= T dot.c W \/ w_"max" + 1$,
  the discretized neuron fires ($S_d >= T_d$) whenever:
  $ (S - T) dot.c W \/ w_"max" >= m^* \/ 2 + 1 $
  Since $S >= T$ by hypothesis and $m^* <= m$, this holds for
  $W >= w_"max" dot.c (m\/2 + 1) \/ T$.
]

== Asymptotic Silence <sec-app-soundness>

#theorem[\
  *(Asymptotic Silence.)*
  Let $cal(N)'$ be a discretized neuron with potential $P_t < T_d$ and leak
  factor $r < 1$. If the input is zero for all $t' >= t$ (i.e.,
  $C_n = 0$ henceforth), then $cal(N)'$ will never fire.
]

#proof[\
  Without input, $P_(t+1) = floor(r dot.c P_t)$. Since
  $r < 1$ and $P_t > 0$, we have $floor(r dot.c P_t) < P_t$; hence the potential
  is strictly decreasing. The sequence converges to the absorbing state $P = 0$.
  Since the trajectory is non-increasing and starts below $T_d$, the firing
  condition $P >= T_d$ is never met. At $P = 0$, the threshold level is $L = 0$,
  which maps to firing probability zero, ensuring permanent silence.
]


// ═══════════════════════════════════════════════════════════════════════════════
// B. Biological Property Preservation — Extended Details
// ═══════════════════════════════════════════════════════════════════════════════

= Appendix: Biological Property Preservation — Extended Details <sec-appendix-bio>

The discretized model preserves the core LIF properties formalized by De Maria
et al.~@naco20. The main text states the properties; here we provide the
full quantitative characterizations.

- *Tonic spiking.* Under constant input $C_"in"$, the neuron has non-zero
  firing probability iff the net gain per step overcomes the multiplicative
  decay, i.e., $C_"in" > T_d dot.c (1 - r)$. When satisfied, the potential
  accumulates toward the steady-state $p_"ss" = C_"in" \/ (1 - r)$, yielding
  periodic probabilistic spiking.

- *Integrator.* The probability of immediate firing on simultaneous inputs
  reaches 1.0 iff $sum delta_W(w_i) >= T_d$. Below threshold, the response is
  graded via the threshold level mapping.

- *Excitability.* The expected inter-spike interval decreases monotonically as
  input strength increases, since stronger input yields higher net accumulation
  per step and a faster approach to threshold.


// ═══════════════════════════════════════════════════════════════════════════════
// C. PRISM Module Decomposition
// ═══════════════════════════════════════════════════════════════════════════════

= Appendix: PRISM Module Decomposition <sec-appendix-modules>

The PRISM model for an SNN $G = (V, E)$ consists of four module types:

- *GlobalClock*: a step counter with $T_"max" + 1$ states.
- *Inputs*: $2^(|V_"in"|)$ states (one binary variable per input neuron).
- *Neuron* $M_n$: per-neuron state depends on configuration:
  - _Fast_ ($k = 10$ threshold levels, no refractory):
    $|S_n^"fast"| = 2 dot.c R_n$ where $R_n = P_("max",n) - P_("min",n) + 1$.
  - _Full_ ($k = 10$, ARP=2, RRP=4):
    $|S_n^"full"| = 3 dot.c 3 dot.c 5 dot.c 2 dot.c R_n = 90 dot.c R_n$.
  - _Discretized_ ($W = 3$):
    $|S_n^"disc"| = 2 dot.c (P_("max",n)^d + 1)$ where
    $P_("max",n)^d = T_d + E_n^d$.
- *Transfer* $T_(i,j)$: 2 states per internal edge (representing the intermediate state of a spike traveling along a synapse).

#theorem[\
  *(State Space Product).* The theoretical state space is the Cartesian product
  of all module state spaces:
  $ |S_"theory"| = (T_"max"+1) dot.c 2^(|V_"in"|) dot.c product_(n in V_"proc") f_n (C) dot.c 2^(|E_"int"|) $
  where $f_n (C) = |S_n|$ depends on the model configuration $C$. The $2^(|E_"int"|)$ factor accounts for the binary states of all internal transfer edges.
]

For a chain of $N$ neurons with $P_"rth" = 100$ and weight $w = 80$: the
per-neuron factor is $f_n = 2 dot.c 121 = 242$ (fast precise), dropping to
$f_n = 2 dot.c 7 = 14$ (discretized $W = 3$). The per-neuron reduction factor
is $242 \/ 14 approx 17.3 times$.

#theorem[\
  *(Exponential State Space Reduction).* For a chain of $N$ neurons, the
  state space ratio between precise and discretized models compounds
  exponentially:
  $ frac(|S^"precise"|, |S^"disc"|) = product_(n=1)^N frac(R_n, P_("max",n)^d + 1) approx 17.3^N $
  The factor $17.3$ is the per-neuron reduction ratio $242 slash 14$ from the
  chain example above. For $N = 4$: $17.3^4 approx 89,500 times$.
]


// ═══════════════════════════════════════════════════════════════════════════════
// D. Estimated Maximum Verifiable Network Sizes
// ═══════════════════════════════════════════════════════════════════════════════

= Appendix: Estimated Maximum Verifiable Network Sizes <sec-appendix-limits>

@tab-app-limits summarizes the estimated maximum verifiable network sizes based
on the empirical state counts and the 2~GB CUDD BDD memory limit.

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    [*Configuration*], [*Chain*], [*Diamond*], [*Notes*],
    [Det. precise], [\>10N], [\>10N], [Minimal state space],
    [Fast precise], [5--6N], [~4N], [Per-neuron narrowing],
    [Full precise], [~4N], [\<4N], [ARP/RRP dominate],
    [Fast disc. $W$=3], [\>10N], [7--8N], [17× reduction/neuron],
    [Full disc. $W$=3], [~8N], [5--6N], [Best precision/tractability],
  ),
  caption: [Estimated maximum verifiable network sizes (2~GB CUDD limit).],
) <tab-app-limits>
