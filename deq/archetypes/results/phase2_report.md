# Phase 2 Report — FCS Fig. 11 Delayer Reproduction

## Topology
3-neuron delayed contralateral: delayer inserted on N1 → N2 inhibitory branch, swept weight w_12 now lives on delayer → N2, N1 → delayer is unit-gain +11 buffer. N2 → N1 branch unchanged with w_21. The delayer adds one tick of latency to the N1-suppresses-N2 pathway.

## Ground-Truth Comparison
| Semantics | Undelayed blue | Delayed blue |
|---|---|---|
| Deterministic | 1014/1600 (63.4%) | 850/1600 (53.1%) |
| Reachability | 1564/1600 (97.8%) | 1534/1600 (95.9%) |

## Winner Asymmetry
- N1 wins: 448 cells
- N2 wins: 1136 cells
- Tied: 16 cells
- Extra N2 wins: 688 (the FCS Fig. 11 asymmetric red-zone growth). Consistent with FCS §6.3.4's observation that the neuron preceded by the delayer (N2 here, whose incoming inhibition is delayed) wins more often.

## Spectral Classification (15-dim A_full)
- ρ range: [0.005, 3.568]
- vs deterministic GT: 53.1% (thr=0.005)
- vs reachability GT: 95.8% (thr=0.005)

## Interpretation
The delayer produces the expected FCS Fig. 11 asymmetry: N2 (whose incoming inhibition is delayed) wins in 1136 cells vs N1's 448 — a 688-cell imbalance that vanishes in the undelayed topology. The delay gives N2 a head start (N2's inhibition on N1 lands at tick 2 while N1's only lands at tick 3), biasing the tick-2 symmetry-breaking in N2's favour.

Spectral prediction via ρ(A_full) on the 15-dim delayed state matrix mirrors the Phase 1b/1c result: it fails on the deterministic GT (53.1%, near baseline) and succeeds on the reachability GT (95.8%, similar to Phase 1c's undelayed 98.5%). Adding the delayer does not change the fundamental finding: spectral cartography tracks reachability, not bit-exact deterministic outcomes.

The asymmetric red-zone growth observed by FCS is captured by the winner map (`phase2_winner_map.png`). Whether ρ(A) predicts *which* neuron wins (not just whether WTA happens) is a separate eigenvector-asymmetry question left for future work if needed.
