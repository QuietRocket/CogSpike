#!/usr/bin/env python3
"""
Run All Analyses — Orchestrator
================================

Runs the complete classical engineering analysis of the 4-neuron WTA network:
  1. Network summary
  2. LIF simulation & PRISM validation
  3. Mean-field linearization
  4. Eigenvalue analysis (stability)
  5. Z-domain transfer functions
  6. Frequency response (Bode/Nyquist)
  7. Steady-state / equilibrium analysis
  8. Transient analysis (step/impulse response)

All plots are saved to ./plots/
"""

import os
import sys
import numpy as np

# Ensure we're in the deq directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Use non-interactive backend for plot generation
import matplotlib
matplotlib.use('Agg')

from network import print_network_summary, W, LEAK_R, NUM_NEURONS, NEURON_LABELS

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  CLASSICAL ENGINEERING ANALYSIS OF 4-NEURON WTA SNN        ║")
    print("║  Laplace/Z-Transform · Eigenvalues · Bode · Steady-State   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # 1. Network Summary
    print_network_summary()

    # 2. LIF Simulation & Validation
    print("\n\n" + "▓" * 60)
    print("  SECTION 2: LIF SIMULATION")
    print("▓" * 60)
    from lif_simulation import run_validation
    sim_results = run_validation(PLOT_DIR)

    # 3. Linearization
    print("\n\n" + "▓" * 60)
    print("  SECTION 3: MEAN-FIELD LINEARIZATION")
    print("▓" * 60)
    from linearization import analyze_operating_regimes
    lin_results = analyze_operating_regimes(PLOT_DIR)

    # 4. Eigenvalue Analysis
    print("\n\n" + "▓" * 60)
    print("  SECTION 4: EIGENVALUE ANALYSIS")
    print("▓" * 60)
    from eigenvalue_analysis import run_eigenvalue_analysis
    eig_results = run_eigenvalue_analysis(PLOT_DIR)

    # 5. Transfer Functions
    print("\n\n" + "▓" * 60)
    print("  SECTION 5: Z-DOMAIN TRANSFER FUNCTIONS")
    print("▓" * 60)
    from transfer_functions import run_transfer_analysis
    tf_results = run_transfer_analysis(PLOT_DIR)

    # 6. Frequency Response
    print("\n\n" + "▓" * 60)
    print("  SECTION 6: FREQUENCY RESPONSE (BODE/NYQUIST)")
    print("▓" * 60)
    from frequency_response import run_frequency_analysis
    freq_results = run_frequency_analysis(PLOT_DIR)

    # 7. Steady-State Analysis
    print("\n\n" + "▓" * 60)
    print("  SECTION 7: STEADY-STATE / EQUILIBRIUM")
    print("▓" * 60)
    from steady_state import run_steady_state_analysis
    ss_results = run_steady_state_analysis(PLOT_DIR)

    # 8. Transient Analysis
    print("\n\n" + "▓" * 60)
    print("  SECTION 8: TRANSIENT ANALYSIS")
    print("▓" * 60)
    from transient_analysis import run_transient_analysis
    trans_results = run_transient_analysis(PLOT_DIR)

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print("\n\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    SUMMARY OF RESULTS                      ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Network identification
    print("\n  1. NETWORK IDENTIFICATION")
    print("  ─────────────────────────")
    print("     Type: 4-Neuron Competitive Inhibition (Winner-Take-All)")
    print("     Key Feature: Asymmetric inhibitory weights")
    print("     N1 inhibits at -100 (strong), others at -70 (weak)")
    print("     Predetermined winner: N1 (30-unit inhibitory advantage)")

    # Eigenvalue summary
    print("\n  2. EIGENVALUE ANALYSIS")
    print("  ─────────────────────")
    regime = 'All inputs on (u=[1,1,1,1])'
    if regime in eig_results and 'eigen' in eig_results[regime]:
        eig = eig_results[regime]['eigen']
        print(f"     Regime: {regime}")
        print(f"     Eigenvalues: {np.round(eig['eigenvalues'], 4).tolist()}")
        print(f"     |λ|:         {np.round(eig['magnitudes'], 4).tolist()}")
        print(f"     Stable: {eig['stable']}")
        print(f"     Dominant τ: {eig['time_constants'][0]:.3f} steps")

    # Transfer function summary
    print("\n  3. TRANSFER FUNCTION DC GAINS")
    print("  ────────────────────────────")
    if tf_results and 'tf_info' in tf_results:
        for key in ['H11', 'H22', 'H21', 'H12']:
            if key in tf_results['tf_info']:
                dc = tf_results['tf_info'][key]['dc_gain']
                print(f"     {key}: {dc:.4f}")

    # Monte Carlo validation
    print("\n  4. SPIKE COUNT VALIDATION (Monte Carlo vs PRISM)")
    print("  ───────────────────────────────────────────────")
    mc = sim_results['monte_carlo']
    print(f"     MC Mean (T=50, 1000 trials): {np.round(mc['mean_counts'], 2).tolist()}")
    print(f"     MC Std:                      {np.round(mc['std_counts'], 2).tolist()}")
    print(f"     PRISM 3-neuron ref:          [37.16, 7.77, 7.77, N/A]")
    winner_ratio = mc['mean_counts'][0] / mc['mean_counts'][1] if mc['mean_counts'][1] > 0 else np.inf
    print(f"     Winner/Loser ratio:          {winner_ratio:.2f}x")

    # Steady-state
    print("\n  5. STEADY-STATE EQUILIBRIA")
    print("  ─────────────────────────")
    if ss_results and 'fixed_points' in ss_results:
        for i, fp in enumerate(ss_results['fixed_points']):
            from steady_state import classify_fixed_point
            pattern, stability = classify_fixed_point(fp)
            print(f"     FP{i+1}: {pattern} | {stability}")
            print(f"           rates = {np.round(fp['rates'], 3).tolist()}")

    # Settling time
    print("\n  6. TRANSIENT DYNAMICS")
    print("  ────────────────────")
    if trans_results:
        for name, st in trans_results['settling_times'].items():
            print(f"     {name} settling time: {st} steps")
        print(f"     Eigenvalue prediction (3τ): {trans_results['predicted_settling']:.1f} steps")

    # Generated plots
    print("\n  7. GENERATED PLOTS")
    print("  ─────────────────")
    plot_files = sorted(os.listdir(PLOT_DIR))
    for f in plot_files:
        if f.endswith('.png'):
            size_kb = os.path.getsize(os.path.join(PLOT_DIR, f)) / 1024
            print(f"     {f} ({size_kb:.0f} KB)")

    print(f"\n  Total plots: {sum(1 for f in plot_files if f.endswith('.png'))}")
    print(f"\n{'='*60}")
    print("  Analysis complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
