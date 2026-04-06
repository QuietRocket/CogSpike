# Contralateral Inhibition: Property Analysis

## Model Validation: ✅ Bit-Identical to Previously Verified Model

I ran PRISM 4.9 on both the **new model** (human-readable names: `y_N1`, `y_N2`, etc.) and the **old fixed model** (numeric IDs: `y5`, `y6`, etc.) from the previous conversation. **All 21 properties produce identical results:**

### LTL Properties (infinite-horizon WTA proof)

| # | Property (new names) | Result | Meaning |
|---|---|---|---|
| 1 | `P=? [ G F "spike_N1" ]` | **1.0** ✅ | Winner N1 fires infinitely often |
| 2 | `P=? [ G F "spike_O1" ]` | **1.0** ✅ | Winner output O1 fires infinitely often |
| 3 | `P=? [ F G (y_N2=0) ]` | **1.0** ✅ | Loser N2 eventually goes permanently silent |
| 4 | `P=? [ F G (y_N3=0) ]` | **1.0** ✅ | Loser N3 eventually goes permanently silent |
| 5 | `P=? [ F G (y_O2=0) ]` | **1.0** ✅ | Loser output O2 goes permanently silent |
| 6 | `P=? [ F G (y_O3=0) ]` | **1.0** ✅ | Loser output O3 goes permanently silent |

### Inverse Properties (should be 0.0 — losers don't win, winner doesn't die)

| # | Property | Result | Meaning |
|---|---|---|---|
| 7 | `P=? [ G F "spike_N2" ]` | **0.0** ✅ | Loser N2 cannot persist |
| 8 | `P=? [ G F "spike_N3" ]` | **0.0** ✅ | Loser N3 cannot persist |
| 9 | `P=? [ F G (y_N1=0) ]` | **0.0** ✅ | Winner N1 never goes silent |

### Structural Properties

| # | Property | Result | Meaning |
|---|---|---|---|
| 10 | `P=? [ F G (y_N1+y_N2+y_N3<=1) ]` | **1.0** ✅ | Mutual exclusion eventually holds |

### Bounded Properties (transient dynamics)

| # | Property | Result | Meaning |
|---|---|---|---|
| 11 | `P=? [ F<=10 "spike_N2" ]` | **0.572** | 57% chance losers fire transiently |
| 12 | `P=? [ F<=10 "spike_N3" ]` | **0.572** | Symmetric with N2 |
| 13 | `P=? [ F<=5 "spike_N1" ]` | **0.547** | Winner fires within 5 steps |
| 14 | `P=? [ F<=10 "spike_O1" ]` | **0.807** | Winner output fires within 10 steps |

### Expected Spike Counts (quantitative dominance)

| # | Reward | Count / 50 steps | Ratio vs Winner |
|---|---|---|---|
| 15 | `R{"spike_N1_count"} C<=50` | **37.16** | 1.00x (winner) |
| 16 | `R{"spike_N2_count"} C<=50` | **7.77** | 0.21x (loser) |
| 17 | `R{"spike_N3_count"} C<=50` | **7.77** | 0.21x (loser) |
| 18 | `R{"spike_O1_count"} C<=50` | **36.23** | 1.00x (winner output) |
| 19 | `R{"spike_O2_count"} C<=50` | **7.71** | 0.21x (loser output) |
| 20 | `R{"spike_O3_count"} C<=50` | **7.71** | 0.21x (loser output) |

**State space**: 3,603 states (both models identical).

---

## Why WTA is Correct: The Weight Asymmetry

The model IS conforming. **N1 is the predetermined winner** due to asymmetric inhibitory weights:

| Edge | Weight | Direction |
|---|---|---|
| N1 → N2 | **-100** | Strong suppression |
| N1 → N3 | **-100** | Strong suppression |
| N2 → N1 | -70 | Weak back-inhibition |
| N3 → N1 | -70 | Weak back-inhibition |
| N2 ↔ N3 | -70 | Weak mutual |

N1 delivers -100 inhibition while only receiving -70 back. When N1 fires:
- N2's `newPotential = 100 - 100 + ... = 0 or negative` → suppressed
- N1's `newPotential = 100 - 70 - 70 = -40`, but with `r=0.5` decay it recovers to fire next tick

---

## Why Bounded Properties Show < 1.0

A subtlety worth noting: `F<=2 spike_N1 = 0.333` — but shouldn't N1 fire deterministically by step 2?

This is caused by the **PRISM Input module nondeterminism**. The Inputs module has 6 `[tick]` commands (2 per input). In PRISM's DTMC semantics, when multiple commands in one module have the same synchronized action with overlapping guards, they create a **uniform nondeterministic choice**. Since `in_S1_g0_fires = true`, `in_S2_g0_fires = true`, and `in_S3_g0_fires = true` are all true, PRISM selects one of the 3 "positive" fire commands uniformly at random per tick.

**This is why the early bounded properties converge slowly (0.333 → 0.547 → 0.856 → 0.995)** — the inputs fire one-at-a-time with equal probability. But the **LTL properties (G F, F G) are unaffected** because they look at infinite-horizon behavior where every path eventually reaches the BSCC.

> [!WARNING]
> This Input module nondeterminism is a modeling artifact. To get deterministic all-at-once inputs (matching simulation exactly), the Inputs module should be restructured to have a **single combined `[tick]` command** that updates all inputs simultaneously:
> ```prism
> [tick] true -> (x_S1' = 1) & (x_S2' = 1) & (x_S3' = 1);
> ```
> However, this does NOT affect the infinite-horizon WTA proof — only the transient timing.

---

## What Properties Should You Test?

Here is the complete recommended test suite, adapted for the new naming:

```prism
// ════════════════════════════════════════════════════════════════
// SECTION 1: WTA PROOF (LTL — all should be 1.0)
// ════════════════════════════════════════════════════════════════

// Winner persists forever
P=? [ G F "spike_N1" ]        // Expected: 1.0
P=? [ G F "spike_O1" ]        // Expected: 1.0

// Losers go permanently silent  
P=? [ F G (y_N2=0) ]          // Expected: 1.0
P=? [ F G (y_N3=0) ]          // Expected: 1.0
P=? [ F G (y_O2=0) ]          // Expected: 1.0
P=? [ F G (y_O3=0) ]          // Expected: 1.0

// ════════════════════════════════════════════════════════════════
// SECTION 2: NEGATIVE PROOF (LTL — all should be 0.0)
// ════════════════════════════════════════════════════════════════

// Losers cannot persist
P=? [ G F "spike_N2" ]        // Expected: 0.0
P=? [ G F "spike_N3" ]        // Expected: 0.0

// Winner cannot go silent
P=? [ F G (y_N1=0) ]          // Expected: 0.0

// ════════════════════════════════════════════════════════════════  
// SECTION 3: STRUCTURAL (should be 1.0)
// ════════════════════════════════════════════════════════════════

// Mutual exclusion in steady state
P=? [ F G (y_N1 + y_N2 + y_N3 <= 1) ]

// ════════════════════════════════════════════════════════════════
// SECTION 4: QUANTITATIVE (spike dominance ratio)
// ════════════════════════════════════════════════════════════════

R{"spike_N1_count"}=? [ C<=50 ]    // ~37.2 (winner)
R{"spike_N2_count"}=? [ C<=50 ]    // ~7.8  (loser, ~5x less)
R{"spike_N3_count"}=? [ C<=50 ]    // ~7.8
R{"spike_O1_count"}=? [ C<=50 ]    // ~36.2
R{"spike_O2_count"}=? [ C<=50 ]    // ~7.7
R{"spike_O3_count"}=? [ C<=50 ]    // ~7.7
```
