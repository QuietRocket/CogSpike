"""
Frequency Response Analysis — Bode and Nyquist Plots
=====================================================

For a discrete-time system, the frequency response is evaluated on the
unit circle: H(e^{jω}) for ω ∈ [0, π].

ω = 0 corresponds to DC (steady-state gain).
ω = π corresponds to the Nyquist frequency (fastest possible oscillation).

Physical meaning for the SNN:
  - Low frequencies: slow modulation of input firing rates
  - High frequencies: rapid alternation of input spikes
  - The leak rate r=0.5 creates strong low-pass filtering
  - The inhibitory feedback loop shapes the frequency response

Key questions answered:
  1. What is the network's bandwidth? (How fast can it track input changes?)
  2. Are there resonant frequencies? (Modes the network amplifies)
  3. What are the stability margins? (How far from instability?)
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from network import NUM_NEURONS, NEURON_LABELS, LEAK_R
from linearization import find_operating_point, build_state_space


def compute_frequency_response(A, B, C, D, n_points=1000):
    """Compute frequency response H(e^{jω}) for ω ∈ [0, π].

    Returns dict of frequency responses for key SISO paths.
    """
    omega = np.linspace(0, np.pi, n_points)
    z = np.exp(1j * omega)
    n = A.shape[0]
    I = np.eye(n)

    # Compute H(z) = C(zI - A)^{-1}B + D for each frequency
    H = np.zeros((n, n, n_points), dtype=complex)
    for k, zk in enumerate(z):
        try:
            H[:, :, k] = C @ np.linalg.solve(zk * I - A, B) + D
        except np.linalg.LinAlgError:
            H[:, :, k] = np.full((n, n), np.nan)

    # Extract key SISO paths
    results = {
        'omega': omega,
        'freq_normalized': omega / np.pi,  # 0 to 1 (fraction of Nyquist)
        'H11': H[0, 0, :],  # S1→N1 (winner path)
        'H21': H[1, 0, :],  # S1→N2 (cross-inhibition)
        'H12': H[0, 1, :],  # S2→N1 (competitor→winner)
        'H22': H[1, 1, :],  # S2→N2 (loser path)
        'H_full': H,
    }
    return results


def plot_bode(freq_resp, save_dir="plots"):
    """Generate Bode plots for key transfer functions."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    paths = [
        ('H11', 'S1 → N1 (Winner Path)', '#2196F3'),
        ('H22', 'S2 → N2 (Loser Path)', '#FF9800'),
        ('H21', 'S1 → N2 (Cross-Inhibition)', '#E91E63'),
        ('H12', 'S2 → N1 (Competitor → Winner)', '#4CAF50'),
    ]

    omega = freq_resp['omega']
    freq_norm = freq_resp['freq_normalized']

    # --- Top: Combined Bode magnitude ---
    ax = axes[0, 0]
    for key, label, color in paths:
        H = freq_resp[key]
        mag_db = 20 * np.log10(np.abs(H) + 1e-20)
        ax.plot(freq_norm, mag_db, color=color, linewidth=2, label=label)

    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Bode Magnitude — All Paths')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    # --- Top right: Combined Bode phase ---
    ax = axes[0, 1]
    for key, label, color in paths:
        H = freq_resp[key]
        phase = np.angle(H, deg=True)
        # Unwrap phase
        phase = np.unwrap(np.angle(H)) * 180 / np.pi
        ax.plot(freq_norm, phase, color=color, linewidth=2, label=label)

    ax.set_ylabel('Phase (degrees)')
    ax.set_title('Bode Phase — All Paths')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    # --- Bottom left: Winner vs Loser comparison ---
    ax = axes[1, 0]
    H11_mag = 20 * np.log10(np.abs(freq_resp['H11']) + 1e-20)
    H22_mag = 20 * np.log10(np.abs(freq_resp['H22']) + 1e-20)
    ax.plot(freq_norm, H11_mag, '#2196F3', linewidth=2, label='Winner (H11)')
    ax.plot(freq_norm, H22_mag, '#FF9800', linewidth=2, label='Loser (H22)')
    ax.fill_between(freq_norm, H22_mag, H11_mag, alpha=0.15, color='green',
                    label='Winner advantage')
    ax.set_xlabel('Normalized Frequency (×π rad/sample)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Winner vs Loser — Gain Advantage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    # --- Bottom right: Bandwidth analysis ---
    ax = axes[1, 1]
    for key, label, color in paths[:2]:
        H = freq_resp[key]
        mag = np.abs(H)
        dc = mag[0]
        mag_norm = mag / dc if dc > 1e-10 else mag
        mag_norm_db = 20 * np.log10(mag_norm + 1e-20)
        ax.plot(freq_norm, mag_norm_db, color=color, linewidth=2, label=label)

        # Find -3dB bandwidth
        below_3db = np.where(mag_norm_db < -3)[0]
        if len(below_3db) > 0:
            bw_idx = below_3db[0]
            bw = freq_norm[bw_idx]
            ax.axvline(bw, color=color, linestyle='--', alpha=0.5)
            ax.annotate(f'BW={bw:.3f}π', (bw, -3),
                       textcoords="offset points", xytext=(10, 10),
                       fontsize=9, color=color)

    ax.axhline(-3, color='red', linestyle=':', alpha=0.5, label='-3dB line')
    ax.set_xlabel('Normalized Frequency (×π rad/sample)')
    ax.set_ylabel('Normalized Magnitude (dB)')
    ax.set_title('Bandwidth Analysis (Normalized to DC)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([-40, 5])

    fig.suptitle('Bode Analysis of Linearized WTA Network', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/bode_plots.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_dir}/bode_plots.png")
    plt.close()


def plot_nyquist(freq_resp, A, save_dir="plots"):
    """Generate Nyquist plot for the loop transfer function.

    For stability analysis, we examine the open-loop transfer function
    of the feedback path. The key feedback loop is through the inhibitory
    weight matrix W.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    paths = [
        ('H11', 'S1 → N1 (Winner)', '#2196F3'),
        ('H22', 'S2 → N2 (Loser)', '#FF9800'),
    ]

    for ax_idx, (key, label, color) in enumerate(paths):
        ax = axes[ax_idx]
        H = freq_resp[key]

        ax.plot(H.real, H.imag, color=color, linewidth=2, label=f'{label} (ω: 0→π)')
        ax.plot(H.real, -H.imag, color=color, linewidth=1, linestyle='--',
                alpha=0.5, label=f'{label} (ω: -π→0)')

        # Mark key frequencies
        n = len(H)
        markers = [(0, 'DC'), (n//4, 'π/4'), (n//2, 'π/2'), (3*n//4, '3π/4'), (n-1, 'π')]
        for idx, freq_label in markers:
            ax.plot(H[idx].real, H[idx].imag, 'ko', markersize=5)
            ax.annotate(f'ω={freq_label}', (H[idx].real, H[idx].imag),
                       textcoords="offset points", xytext=(8, 8), fontsize=7)

        # Critical point
        ax.plot(-1, 0, 'r+', markersize=15, markeredgewidth=2, label='Critical point (-1,0)')

        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5)
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_title(f'Nyquist Plot — {label}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    fig.suptitle('Nyquist Diagrams of Linearized WTA Network', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/nyquist_plots.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_dir}/nyquist_plots.png")
    plt.close()


def compute_margins(freq_resp):
    """Compute gain and phase margins for key transfer functions."""
    results = {}
    for key in ['H11', 'H22']:
        H = freq_resp[key]
        omega = freq_resp['omega']
        mag = np.abs(H)
        phase = np.unwrap(np.angle(H))

        # Gain margin: |H| at the frequency where phase = -π
        phase_cross = np.where(np.diff(np.sign(phase + np.pi)))[0]
        if len(phase_cross) > 0:
            idx = phase_cross[0]
            gm = 1.0 / mag[idx]  # Gain margin
            gm_db = 20 * np.log10(gm)
        else:
            gm_db = np.inf

        # Phase margin: phase at the frequency where |H| = 1
        gain_cross = np.where(np.diff(np.sign(mag - 1)))[0]
        if len(gain_cross) > 0:
            idx = gain_cross[0]
            pm = 180 + phase[idx] * 180 / np.pi  # Phase margin in degrees
        else:
            pm = np.inf  # Never crosses unity gain

        results[key] = {'gain_margin_db': gm_db, 'phase_margin_deg': pm}

    return results


def run_frequency_analysis(save_dir="plots"):
    """Run complete frequency response analysis."""
    print("\n" + "=" * 60)
    print("FREQUENCY RESPONSE ANALYSIS")
    print("=" * 60)

    u = np.ones(NUM_NEURONS)
    p_star, r_star = find_operating_point(u)
    A, B, C, D = build_state_space(p_star)

    print(f"\n  Computing H(e^{{jω}}) for ω ∈ [0, π] ...")
    freq_resp = compute_frequency_response(A, B, C, D)

    # DC gains
    print(f"\n  DC Gains (ω=0):")
    for key in ['H11', 'H22', 'H21', 'H12']:
        dc = np.abs(freq_resp[key][0])
        print(f"    {key}: {dc:.4f} ({20*np.log10(dc+1e-20):.2f} dB)")

    # Bandwidth analysis
    print(f"\n  Bandwidth (-3dB point):")
    for key, label in [('H11', 'Winner'), ('H22', 'Loser')]:
        H = freq_resp[key]
        mag = np.abs(H)
        dc = mag[0]
        if dc > 1e-10:
            mag_norm = mag / dc
            below_3db = np.where(20*np.log10(mag_norm+1e-20) < -3)[0]
            if len(below_3db) > 0:
                bw = freq_resp['freq_normalized'][below_3db[0]]
                print(f"    {label} ({key}): {bw:.4f}π rad/sample")
            else:
                print(f"    {label} ({key}): > π (full bandwidth)")

    # Stability margins
    print(f"\n  Stability Margins:")
    margins = compute_margins(freq_resp)
    for key, m in margins.items():
        print(f"    {key}: GM = {m['gain_margin_db']:.2f} dB, PM = {m['phase_margin_deg']:.2f}°")

    # Physical interpretation
    print(f"\n  Physical Interpretation:")
    print(f"    The leak rate r={LEAK_R} gives a single-neuron time constant of")
    print(f"    τ = -1/ln(r) = {-1/np.log(LEAK_R):.3f} time steps")
    print(f"    This corresponds to a -3dB cutoff of approximately")
    cutoff = 1 - LEAK_R  # For a simple first-order system H(z) = 1/(z-r)
    print(f"    ω_c ≈ {cutoff:.3f} rad/sample = {cutoff/np.pi:.3f}π")
    print(f"    The inhibitory feedback modifies this, potentially creating")
    print(f"    resonances or sharper rolloff in specific paths.")

    # Generate plots
    plot_bode(freq_resp, save_dir)
    plot_nyquist(freq_resp, A, save_dir)

    return freq_resp


if __name__ == "__main__":
    run_frequency_analysis()
