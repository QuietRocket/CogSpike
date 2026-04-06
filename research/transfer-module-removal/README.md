# Transfer Module Removal — Formal Proof Models

This directory contains hand-crafted PRISM models that formally prove that
removing transfer modules (`z` variables) from the CogSpike PRISM generator
**improves isomorphism** with the simulation engine.

## Key Finding

> **Transfer modules ADD an extra tick of latency per synapse.** They don't
> preserve timing — they make it worse. Removing them gives the correct
> 1-tick-per-synapse delay that matches the simulation.

### Evidence: Chain Trace Comparison

```
WITHOUT Transfer (correct):         WITH Transfer (too slow):
Step  x_Inp  y_A  y_B  y_C         Step  x_Inp  y_A  z_AB  y_B  z_BC  y_C
  0     0     0    0    0            0     0     0     0    0     0    0
  1     1     0    0    0            1     1     0     0    0     0    0
  2     0    [1]   0    0            2     0    [1]    0    0     0    0
  3     0     0   [1]   0            3     0     0    [1]   0     0    0    ← z copies (delay!)
  4     0     0    0   [1] ✓         4     0     0     0   [1]    0    0    ← B fires 1 tick late
  5     done                         5     0     0     0    0    [1]   0    ← z copies (delay!)
                                     6     0     0     0    0     0   [1]   ← C fires 2 ticks late!
```

**Without transfer: 4 steps (3 synaptic hops × 1 tick each)**
**With transfer: 6 steps (3 synaptic hops × 2 ticks each)**

## Proof Scenarios

### 1. Chain Topology (`chain_no_transfer.prism`)

```
Input → A → B → C
```

| Property | Expected | Result | Proves |
|----------|----------|--------|--------|
| `P=? [F "output_fires"]` | 1.0 | ✅ 1.0 | Spike reaches output |
| `P=? [X X "A_fires"]` | 1.0 | ✅ 1.0 | A fires at step 2 |
| `P=? [X X X "B_fires"]` | 1.0 | ✅ 1.0 | B fires at step 3 |
| `P=? [X X X X "output_fires"]` | 1.0 | ✅ 1.0 | C fires at step 4 (1-tick/synapse) |
| `P=? [X X X !y_C=1]` | 1.0 | ✅ 1.0 | C hasn't fired at step 3 |
| `P=? [F<=5 "output_fires"]` | 1.0 | ✅ 1.0 | Reaches within 5 steps |
| `P=? [F<=3 "output_fires"]` | 0.0 | ✅ 0.0 | Doesn't reach in only 3 steps |

### Same Properties on WITH-Transfer Model

| Property | No Transfer | With Transfer | Notes |
|----------|-------------|---------------|-------|
| `P=? [X X X "B_fires"]` | 1.0 | **0.0** ❌ | B is late! |
| `P=? [X X X X "output_fires"]` | 1.0 | **0.0** ❌ | C is late! |
| `P=? [F<=5 "output_fires"]` | 1.0 | **0.0** ❌ | Needs >5 steps |

### 2. Fork Topology (`fork_no_transfer.prism`)

```
          ┌──→ B1
Input → A ┤
          └──→ B2
```

| Property | Expected | Result | Proves |
|----------|----------|--------|--------|
| `P=? [F "both_fire"]` | 1.0 | ✅ 1.0 | Both targets fire |
| `P=? [X X X "both_fire"]` | 1.0 | ✅ 1.0 | Both fire at step 3 |
| `P=? [G (B1 => B2)]` | 1.0 | ✅ 1.0 | Always synchronized |
| `P=? [G (B2 => B1)]` | 1.0 | ✅ 1.0 | Always synchronized |

### 3. Convergence Topology (`convergence_no_transfer.prism`)

```
Input1 → A ──┐
              ├──→ C    (w=55 each, threshold=100, needs both)
Input2 → B ──┘
```

| Property | Expected | Result | Proves |
|----------|----------|--------|--------|
| `P=? [F "output_fires"]` | 1.0 | ✅ 1.0 | C fires (55+55=110 > 100) |
| `P=? [X X "both_sources"]` | 1.0 | ✅ 1.0 | A,B fire simultaneously |
| `P=? [X X X "output_fires"]` | 1.0 | ✅ 1.0 | C fires from coincidence |
| `R{"spikes_C"}=? [C<=10]` | 1.0 | ✅ 1.0 | Exactly 1 spike total |

## Running the Proofs

```bash
cd transfer_proof/

PRISM=/Users/quietrocket/Documents/PhD/prism-4.9-mac64-arm/bin/prism

# Chain without transfer (all should pass)
$PRISM chain_no_transfer.prism chain_no_transfer.props

# Chain with transfer (timing properties will FAIL — proves transfers add latency)
$PRISM chain_with_transfer.prism chain_no_transfer.props

# Fork (simultaneous delivery)
$PRISM fork_no_transfer.prism fork_no_transfer.props

# Convergence (coincidence detection)
$PRISM convergence_no_transfer.prism convergence_no_transfer.props
```

## Conclusion

Removing transfer modules is not just safe — it's **necessary for correct
timing**. The old transfer modules added 1 extra tick of delay per synapse,
making the PRISM model diverge from the Rust simulation engine (which correctly
implements 1-tick delay via double-buffering in Phase 4).
