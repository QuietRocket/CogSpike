"""
Eigenvalue Analysis of the Linearized WTA Network
==================================================

For a discrete-time linear system p(t+1) = A*p(t) + B*u(t):
- Eigenvalues of A determine stability and dynamics
- All |λ_i| < 1 => stable (perturbations decay)
- |λ_i| gives decay rate; arg(λ_i) gives oscillation frequency
- Time constant: τ_i = -1 / ln|λ_i| (in time steps)
- The eigenvectors reveal the modes of the network

Key insight: the asymmetric weight structure should produce eigenvalue
asymmetry that explains WHY N1 wins the competition.
"""

import numpy as np
import matplotlib.pyplot as plt
from network import NUM_NEURONS, NEURON_LABELS, LEAK_R, W
from linearization import find_operating_point, build_state_matrix


def analyze_eigenstructure(A, regime_name=""):
    """Complete eigenvalue/eigenvector analysis of state matrix A.

    Returns dict with eigenvalues, eigenvectors, and derived quantities.
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Sort by magnitude (largest first)
    idx = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    results = {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'magnitudes': np.abs(eigenvalues),
        'angles': np.angle(eigenvalues),
        'stable': np.all(np.abs(eigenvalues) < 1.0),
    }

    # Time constants (only for eigenvalues with |λ| < 1 and |λ| > 0)
    time_constants = np.full_like(np.abs(eigenvalues), np.inf)
    valid = (np.abs(eigenvalues) > 1e-10) & (np.abs(eigenvalues) < 1.0)
    time_constants[valid] = -1.0 / np.log(np.abs(eigenvalues[valid]))
    results['time_constants'] = time_constants

    # Damping ratios (for complex eigenvalues)
    damping_ratios = np.zeros(len(eigenvalues))
    for i, lam in enumerate(eigenvalues):
        if np.abs(lam) > 1e-10:
            if np.isreal(lam):
                damping_ratios[i] = 1.0  # Critically/overdamped
            else:
                # ζ = -ln|λ| / sqrt(ln²|λ| + arg²(λ))
                ln_mag = np.log(np.abs(lam))
                angle = np.angle(lam)
                damping_ratios[i] = -ln_mag / np.sqrt(ln_mag**2 + angle**2)
    results['damping_ratios'] = damping_ratios

    return results


def print_eigenanalysis(results, regime_name):
    """Print formatted eigenvalue analysis."""
    print(f"\n  {'Mode':<6} {'Eigenvalue':<24} {'|λ|':<10} {'τ (steps)':<12} {'ζ':<8} {'Type'}")
    print(f"  {'─'*6} {'─'*24} {'─'*10} {'─'*12} {'─'*8} {'─'*20}")

    for i, (lam, mag, tau, zeta) in enumerate(zip(
        results['eigenvalues'], results['magnitudes'],
        results['time_constants'], results['damping_ratios']
    )):
        # Classify mode
        if mag < 1e-10:
            mode_type = "Dead mode"
        elif np.isreal(lam):
            if lam.real > 0:
                mode_type = "Monotonic decay" if mag < 1 else "Monotonic growth"
            else:
                mode_type = "Alternating decay" if mag < 1 else "Alternating growth"
        else:
            mode_type = "Oscillatory decay" if mag < 1 else "Oscillatory growth"

        lam_str = f"{lam.real:+.6f}"
        if np.abs(lam.imag) > 1e-10:
            lam_str += f" {lam.imag:+.6f}j"
        tau_str = f"{tau:.3f}" if not np.isinf(tau) else "∞"

        print(f"  λ_{i+1:<3} {lam_str:<24} {mag:<10.6f} {tau_str:<12} {zeta:<8.4f} {mode_type}")

    print(f"\n  Overall stability: {'STABLE' if results['stable'] else 'UNSTABLE'}")
    print(f"  Dominant time constant: {results['time_constants'][0]:.3f} steps")
    print(f"  Fastest mode: τ = {np.min(results['time_constants'][results['time_constants'] > 0]):.3f} steps")


def print_eigenvector_analysis(results):
    """Print eigenvector interpretation."""
    print(f"\n  Eigenvector analysis (columns = modes, rows = neurons):")
    print(f"  {'':>8}", end="")
    for i in range(NUM_NEURONS):
        print(f"  {'Mode '+str(i+1):>10}", end="")
    print()

    for i in range(NUM_NEURONS):
        print(f"  {NEURON_LABELS[i]:>8}", end="")
        for j in range(NUM_NEURONS):
            v = results['eigenvectors'][i, j]
            if np.isreal(v):
                print(f"  {v.real:>10.4f}", end="")
            else:
                print(f"  {v:>10.4f}", end="")
        print()

    # Interpret dominant mode
    print(f"\n  Interpretation of dominant mode (λ₁ = {results['eigenvalues'][0]:.6f}):")
    v1 = results['eigenvectors'][:, 0].real
    v1_norm = v1 / np.max(np.abs(v1))
    for i in range(NUM_NEURONS):
        bar = '█' * int(abs(v1_norm[i]) * 20)
        sign = '+' if v1_norm[i] > 0 else '-'
        print(f"    {NEURON_LABELS[i]}: {sign}{bar} ({v1_norm[i]:+.4f})")


def plot_eigenvalues(all_results, save_dir="plots"):
    """Plot eigenvalues on the complex plane for all regimes."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1.5, alpha=0.3,
            label='Unit circle')

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
    markers = ['o', 's', 'D', '^']

    for idx, (name, res) in enumerate(all_results.items()):
        if 'eigen' not in res:
            continue
        eig = res['eigen']
        for i, lam in enumerate(eig['eigenvalues']):
            ax.plot(lam.real, lam.imag, markers[idx], color=colors[idx],
                    markersize=12, markeredgecolor='black', markeredgewidth=1,
                    label=name if i == 0 else None)
            # Label eigenvalue
            ax.annotate(f'λ{i+1}={lam:.3f}',
                       (lam.real, lam.imag),
                       textcoords="offset points", xytext=(10, 5),
                       fontsize=7, color=colors[idx])

    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_xlabel('Real Part', fontsize=12)
    ax.set_ylabel('Imaginary Part', fontsize=12)
    ax.set_title('Eigenvalues of Linearized State Matrix A\n(discrete-time: stable if inside unit circle)', fontsize=14)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Zoom to relevant region
    all_eigs = np.concatenate([res['eigen']['eigenvalues']
                               for res in all_results.values() if 'eigen' in res])
    margin = 0.3
    xlim = [min(all_eigs.real.min(), -1) - margin, max(all_eigs.real.max(), 1) + margin]
    ylim = [min(all_eigs.imag.min(), -1) - margin, max(all_eigs.imag.max(), 1) + margin]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/eigenvalues_complex_plane.png", dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {save_dir}/eigenvalues_complex_plane.png")
    plt.close()


def plot_eigenvectors(results, regime_name, save_dir="plots"):
    """Plot eigenvector bar chart showing neuron contributions to each mode."""
    eig = results['eigen']
    fig, axes = plt.subplots(1, NUM_NEURONS, figsize=(16, 4), sharey=True)
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

    for mode in range(NUM_NEURONS):
        ax = axes[mode]
        v = eig['eigenvectors'][:, mode].real
        bars = ax.bar(NEURON_LABELS, v, color=colors, edgecolor='black')
        ax.axhline(y=0, color='black', linewidth=0.5)
        lam = eig['eigenvalues'][mode]
        ax.set_title(f'Mode {mode+1}\nλ={lam:.4f}\n|λ|={np.abs(lam):.4f}',
                     fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Eigenvector Component')
    fig.suptitle(f'Eigenvector Structure — {regime_name}', fontsize=13)
    plt.tight_layout()
    safe_name = regime_name.replace(' ', '_').replace('=', '').replace('[', '').replace(']', '').replace(',', '')[:30]
    plt.savefig(f"{save_dir}/eigenvectors_{safe_name}.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_dir}/eigenvectors_{safe_name}.png")
    plt.close()


def run_eigenvalue_analysis(save_dir="plots"):
    """Run complete eigenvalue analysis for all operating regimes."""
    print("\n" + "=" * 60)
    print("EIGENVALUE ANALYSIS")
    print("=" * 60)

    regimes = {
        'All inputs on (u=[1,1,1,1])': np.ones(NUM_NEURONS),
        'Only N1 input (u=[1,0,0,0])': np.array([1, 0, 0, 0]),
        'N1+N2 inputs (u=[1,1,0,0])': np.array([1, 1, 0, 0]),
    }

    all_results = {}
    for name, u in regimes.items():
        print(f"\n{'='*50}")
        print(f"  Regime: {name}")
        print(f"{'='*50}")

        p_star, r_star = find_operating_point(u)
        A, G = build_state_matrix(p_star)

        print(f"  Operating point: p* = {np.round(p_star, 2).tolist()}")
        print(f"  Firing rates:    f* = {np.round(r_star, 4).tolist()}")
        print(f"  Gains:           g  = {np.round(G, 6).tolist()}")

        eig = analyze_eigenstructure(A, name)
        print_eigenanalysis(eig, name)
        print_eigenvector_analysis(eig)

        all_results[name] = {
            'u': u, 'p_star': p_star, 'rates': r_star,
            'A': A, 'G': G, 'eigen': eig,
        }
        plot_eigenvectors(all_results[name], name, save_dir)

    # --- Special analysis: the raw inhibitory weight matrix ---
    print(f"\n{'='*50}")
    print(f"  BONUS: Eigenvalues of W (raw weight matrix)")
    print(f"{'='*50}")
    eig_W = np.linalg.eigvals(W)
    print(f"  Eigenvalues of W: {np.round(eig_W, 4).tolist()}")
    print(f"  |λ|: {np.round(np.abs(eig_W), 4).tolist()}")
    print(f"\n  The weight matrix structure:")
    print(f"    - One eigenvalue corresponds to the 'common mode' (all neurons same)")
    print(f"    - Others correspond to 'differential modes' (competition)")
    print(f"    - The asymmetry N1(-100) vs others(-70) breaks the symmetry")
    print(f"      of the differential modes, favoring N1")

    all_results['W_matrix'] = {'eigen': analyze_eigenstructure(W)}

    plot_eigenvalues(all_results, save_dir)

    return all_results


if __name__ == "__main__":
    run_eigenvalue_analysis()
