#!/usr/bin/env python3
"""
CS-Friendly Visualizations for SNN Spectral Analysis
=====================================================

Generates plots with CS terminology, clear annotations, and
analogies to concepts from discrete math / TCS.

All plots saved to plots/cs_*.png
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

# Ensure we're in the deq directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from network import (
    NUM_NEURONS, NEURON_LABELS, LEAK_R, W, B_IN,
    firing_rate_sigmoid, firing_rate_sigmoid_derivative,
    lif_step_meanfield_sigmoid, SIGMOID_MID
)
from linearization import find_operating_point, build_state_matrix, build_state_space
from frequency_response import compute_frequency_response

PLOT_DIR = "plots"

# CS-friendly plot defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'font.family': 'serif',
})

COLORS = {
    'N1': '#2196F3',
    'N2': '#FF9800',
    'N3': '#4CAF50',
    'N4': '#E91E63',
    'stable': '#2E7D32',
    'unstable': '#C62828',
    'neutral': '#757575',
}


def plot_network_graph():
    """Draw the 4-neuron WTA as a directed graph."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Neuron positions (cardinal points)
    positions = {
        'N1': (0.5, 0.85),
        'N2': (0.15, 0.5),
        'N3': (0.85, 0.5),
        'N4': (0.5, 0.15),
    }

    # Input positions
    input_pos = {
        'S1': (0.25, 0.95),
        'S2': (0.0, 0.65),
        'S3': (1.0, 0.65),
        'S4': (0.25, 0.05),
    }

    # Draw inhibitory edges
    edge_pairs = [
        ('N1', 'N2', -100), ('N1', 'N3', -100), ('N1', 'N4', -100),
        ('N2', 'N1', -70), ('N2', 'N3', -70), ('N2', 'N4', -70),
        ('N3', 'N1', -70), ('N3', 'N2', -70), ('N3', 'N4', -70),
        ('N4', 'N1', -70), ('N4', 'N2', -70), ('N4', 'N3', -70),
    ]

    for src, dst, w in edge_pairs:
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        # Offset for bidirectional edges
        dx, dy = x2 - x1, y2 - y1
        perp_x, perp_y = -dy * 0.03, dx * 0.03

        lw = 2.5 if w == -100 else 1.2
        color = '#C62828' if w == -100 else '#EF9A9A'
        alpha = 0.9 if w == -100 else 0.5

        ax.annotate('', xy=(x2 + perp_x, y2 + perp_y),
                    xytext=(x1 + perp_x, y1 + perp_y),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=lw, alpha=alpha,
                                   connectionstyle='arc3,rad=0.15'))
        # Weight label at midpoint
        mx = (x1 + x2) / 2 + perp_x * 4
        my = (y1 + y2) / 2 + perp_y * 4
        if abs(w) == 100:
            ax.text(mx, my, str(w), fontsize=8, ha='center', va='center',
                    color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))

    # Draw input arrows
    for inp, neuron in [('S1', 'N1'), ('S2', 'N2'), ('S3', 'N3'), ('S4', 'N4')]:
        x1, y1 = input_pos[inp]
        x2, y2 = positions[neuron]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#1565C0',
                                   lw=2, alpha=0.7))
        ax.text(x1, y1, inp, fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round', fc='#BBDEFB', ec='#1565C0'),
                fontweight='bold')

    # Draw neurons
    for name, (x, y) in positions.items():
        color = COLORS['N1'] if name == 'N1' else '#90A4AE'
        size = 0.065 if name == 'N1' else 0.055
        circle = plt.Circle((x, y), size, color=color, ec='black', lw=2, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, name, fontsize=14, ha='center', va='center',
                fontweight='bold', color='white', zorder=6)

    # Legend
    ax.plot([], [], '-', color='#C62828', lw=2.5, label='Strong inhibition (-100, from N1)')
    ax.plot([], [], '-', color='#EF9A9A', lw=1.2, label='Weak inhibition (-70, from N2/N3/N4)')
    ax.plot([], [], '-', color='#1565C0', lw=2, label='Excitatory input (+100)')
    ax.legend(loc='lower right', framealpha=0.9, fontsize=9)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('4-Neuron WTA Network as a Directed Graph\n'
                 'N1 (blue) has stronger outgoing inhibition',
                 fontsize=14, pad=10)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/cs_network_graph.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {PLOT_DIR}/cs_network_graph.png")
    plt.close()


def plot_eigenvalue_spectrum():
    """Bar chart of eigenvalue magnitudes with stability annotation."""
    eigvals_W = np.linalg.eigvals(W).real
    idx = np.argsort(-np.abs(eigvals_W))
    eigvals_W = eigvals_W[idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: Weight matrix eigenvalues ---
    ax = axes[0]
    labels = [f'Mode {i+1}\n$\\lambda$={eigvals_W[i]:.1f}' for i in range(4)]
    interpretations = [
        'Common-mode\ninhibition',
        'N1 advantage\n(WTA mode)',
        'Loser\ncompetition',
        'Degenerate\nwith Mode 3',
    ]
    bar_colors = ['#90A4AE', '#4CAF50', '#90A4AE', '#90A4AE']

    bars = ax.bar(range(4), np.abs(eigvals_W), color=bar_colors, edgecolor='black', width=0.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(interpretations, fontsize=9)
    for i, (bar, val) in enumerate(zip(bars, eigvals_W)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'|{val:.1f}|', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Eigenvalue magnitude $|\\lambda|$')
    ax.set_title('Weight Matrix $\\mathbf{W}$ Spectrum')
    ax.grid(axis='y', alpha=0.3)

    # Annotate the WTA mode
    ax.annotate('This mode predicts\nthe winner (N1)',
                xy=(1, np.abs(eigvals_W[1])),
                xytext=(2.2, np.abs(eigvals_W[1]) + 30),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2),
                fontsize=10, color='#2E7D32', fontweight='bold',
                bbox=dict(boxstyle='round', fc='#E8F5E9', ec='#4CAF50'))

    # --- Right: Linearized A eigenvalues ---
    ax = axes[1]
    u = np.ones(NUM_NEURONS)
    p_star, _ = find_operating_point(u)
    A, G = build_state_matrix(p_star)
    eigvals_A = np.sort(np.linalg.eigvals(A).real)[::-1]

    bar_colors_A = ['#4CAF50' if abs(e) < 1 else '#C62828' for e in eigvals_A]
    bars = ax.bar(range(4), np.abs(eigvals_A), color=bar_colors_A, edgecolor='black', width=0.6)
    ax.axhline(1.0, color='red', linestyle='--', lw=2, label='Stability boundary ($|\\lambda| = 1$)')

    mode_labels = [
        f'$\\lambda$={eigvals_A[i]:.3f}' for i in range(4)
    ]
    ax.set_xticks(range(4))
    ax.set_xticklabels(mode_labels, fontsize=9)

    for bar, val in zip(bars, eigvals_A):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{abs(val):.3f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Eigenvalue magnitude $|\\lambda|$')
    ax.set_title('Linearized State Matrix $\\mathbf{A}$ Spectrum\n(all below 1.0 = stable)')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.3)

    # Annotate stability
    ax.fill_between([-0.5, 3.5], 0, 1.0, alpha=0.05, color='green')
    ax.fill_between([-0.5, 3.5], 1.0, 1.3, alpha=0.05, color='red')
    ax.text(3.3, 0.5, 'STABLE\nREGION', fontsize=9, ha='right', color='#2E7D32', alpha=0.7)
    ax.text(3.3, 1.15, 'UNSTABLE', fontsize=9, ha='right', color='#C62828', alpha=0.7)
    ax.set_xlim(-0.5, 3.5)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/cs_eigenvalue_spectrum.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {PLOT_DIR}/cs_eigenvalue_spectrum.png")
    plt.close()


def plot_eigenvector_heatmap():
    """4x4 heatmap of eigenvector components."""
    eigvals, eigvecs = np.linalg.eig(W)
    idx = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[idx].real
    eigvecs = eigvecs[:, idx].real

    # Normalize each eigenvector to have max component = 1
    for j in range(4):
        eigvecs[:, j] = eigvecs[:, j] / np.max(np.abs(eigvecs[:, j]))

    fig, ax = plt.subplots(figsize=(9, 5))

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(eigvecs, cmap='RdBu_r', norm=norm, aspect='auto')

    # Annotate cells
    for i in range(4):
        for j in range(4):
            val = eigvecs[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            label = 'excited' if val > 0.3 else ('inhibited' if val < -0.3 else '')
            ax.text(j, i, f'{val:+.2f}\n{label}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold' if abs(val) > 0.5 else 'normal')

    ax.set_xticks(range(4))
    col_labels = [
        f'Mode 1\n$\\lambda$={eigvals[0]:.0f}\nCommon-mode',
        f'Mode 2\n$\\lambda$={eigvals[1]:.0f}\nWTA mode',
        f'Mode 3\n$\\lambda$={eigvals[2]:.0f}\nLoser comp.',
        f'Mode 4\n$\\lambda$={eigvals[3]:.0f}\nDegenerate',
    ]
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(NEURON_LABELS, fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Eigenvector component\n(red = excited, blue = inhibited)', fontsize=10)

    # Highlight WTA column
    rect = mpatches.FancyBboxPatch((0.55, -0.5), 0.9, 4, boxstyle="round,pad=0.05",
                                    fill=False, edgecolor='#4CAF50', linewidth=3,
                                    linestyle='--')
    ax.add_patch(rect)
    ax.annotate('WTA Mode:\nN1 excited,\nothers inhibited',
                xy=(1, -0.5), xytext=(2.5, -1.1),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2),
                fontsize=10, color='#2E7D32', fontweight='bold',
                bbox=dict(boxstyle='round', fc='#E8F5E9', ec='#4CAF50'))

    ax.set_title('Eigenvector Heatmap of Weight Matrix $\\mathbf{W}$\n'
                 'Each column is an independent competitive mode',
                 fontsize=13, pad=15)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/cs_eigenvector_heatmap.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {PLOT_DIR}/cs_eigenvector_heatmap.png")
    plt.close()


def plot_convergence_trace():
    """Convergence plot styled like iterative method convergence."""
    u = np.ones(NUM_NEURONS)
    p_star, r_star = find_operating_point(u)

    # Start from a symmetric perturbation
    p = np.ones(NUM_NEURONS) * SIGMOID_MID  # All at midpoint
    rates = firing_rate_sigmoid(p)

    T = 30
    errors = []
    potentials_trace = np.zeros((T + 1, NUM_NEURONS))
    potentials_trace[0] = p

    for t in range(T):
        p, rates = lif_step_meanfield_sigmoid(p, rates, u)
        potentials_trace[t + 1] = p
        error = np.linalg.norm(p - p_star)
        errors.append(error)

    # Predicted convergence rate
    A, G = build_state_matrix(p_star)
    rho = max(np.abs(np.linalg.eigvals(A)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: State convergence ---
    ax = axes[0]
    colors = [COLORS['N1'], COLORS['N2'], COLORS['N3'], COLORS['N4']]
    for i in range(NUM_NEURONS):
        ax.plot(potentials_trace[:, i], color=colors[i], linewidth=2,
                label=NEURON_LABELS[i])
        ax.axhline(p_star[i], color=colors[i], linestyle=':', alpha=0.3)

    ax.set_xlabel('Iteration (time step)')
    ax.set_ylabel('Membrane potential $p_i$')
    ax.set_title('State Vector Convergence\n(cf. Jacobi iteration)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Right: Error convergence (log scale) ---
    ax = axes[1]
    steps = np.arange(1, T + 1)
    ax.semilogy(steps, errors, 'ko-', markersize=4, linewidth=1.5, label='Actual $||p(t) - p^*||$')

    # Predicted rate
    predicted_errors = errors[0] * rho ** np.arange(T)
    ax.semilogy(steps, predicted_errors, 'r--', linewidth=2,
                label=f'Predicted: $\\rho^t$, $\\rho = {rho:.3f}$')

    ax.set_xlabel('Iteration (time step)')
    ax.set_ylabel('Error $||p(t) - p^*||$ (log scale)')
    ax.set_title('Geometric Convergence Rate\n'
                 f'Spectral radius $\\rho(\\mathbf{{A}})$ = {rho:.3f} < 1')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # Annotate
    ax.annotate(f'Same criterion as\nJacobi/Gauss-Seidel:\n$\\rho < 1$ means convergence',
                xy=(T * 0.6, errors[int(T * 0.6)] if int(T*0.6) < len(errors) else errors[-1]),
                xytext=(T * 0.45, errors[0] * 0.5),
                fontsize=9, color='#1565C0',
                bbox=dict(boxstyle='round', fc='#E3F2FD', ec='#1565C0'),
                arrowprops=dict(arrowstyle='->', color='#1565C0'))

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/cs_convergence_trace.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {PLOT_DIR}/cs_convergence_trace.png")
    plt.close()


def plot_basin_simplified():
    """Clean basin-of-attraction map."""
    u = np.ones(NUM_NEURONS)
    grid_size = 60
    p1_range = np.linspace(-50, 200, grid_size)
    p2_range = np.linspace(-50, 200, grid_size)
    basin = np.full((grid_size, grid_size), -1.0)

    for i, p2_init in enumerate(p2_range):
        for j, p1_init in enumerate(p1_range):
            p = np.array([p1_init, p2_init, 30, 30])
            rates = firing_rate_sigmoid(p)
            for _ in range(200):
                p, rates = lif_step_meanfield_sigmoid(p, rates, u)
            winner = np.argmax(rates)
            if np.max(rates) < 0.05:
                basin[i, j] = -1
            else:
                basin[i, j] = winner

    fig, ax = plt.subplots(figsize=(9, 8))

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#E0E0E0', '#2196F3', '#FF9800', '#4CAF50', '#E91E63'])
    im = ax.imshow(basin, origin='lower', aspect='auto',
                   extent=[p1_range[0], p1_range[-1], p2_range[0], p2_range[-1]],
                   cmap=cmap, vmin=-1, vmax=3, interpolation='nearest')

    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1, 2, 3], shrink=0.8)
    cbar.ax.set_yticklabels(['Quiescent', 'N1 wins', 'N2 wins', 'N3 wins', 'N4 wins'])

    ax.set_xlabel('N1 initial potential', fontsize=13)
    ax.set_ylabel('N2 initial potential', fontsize=13)
    ax.set_title('Basin of Attraction: Which Neuron Wins?\n'
                 '(N3, N4 initialized at 30; all inputs active)',
                 fontsize=13)

    # Annotate dominant basin
    ax.annotate("N1's basin dominates:\nweight asymmetry creates\na large capture region",
                xy=(120, 50), xytext=(10, 170),
                fontsize=10, color='white', fontweight='bold',
                bbox=dict(boxstyle='round', fc='#1565C0', ec='white', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/cs_basin_of_attraction.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {PLOT_DIR}/cs_basin_of_attraction.png")
    plt.close()


def plot_bode_annotated():
    """Annotated frequency response with CS-friendly labels."""
    u = np.ones(NUM_NEURONS)
    p_star, _ = find_operating_point(u)
    A, B, C, D = build_state_space(p_star)
    freq_resp = compute_frequency_response(A, B, C, D)

    omega = freq_resp['omega']
    freq_norm = freq_resp['freq_normalized']
    H11 = np.abs(freq_resp['H11'])
    H22 = np.abs(freq_resp['H22'])

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.semilogy(freq_norm, H11, color=COLORS['N1'], linewidth=2.5, label='Winner path (S1 → N1)')
    ax.semilogy(freq_norm, H22, color=COLORS['N2'], linewidth=2.5, label='Loser path (S2 → N2)')

    # Bandwidth markers
    for H, color, label in [(H11, COLORS['N1'], 'Winner'), (H22, COLORS['N2'], 'Loser')]:
        dc = H[0]
        bw_threshold = dc / np.sqrt(2)  # -3dB point
        below = np.where(H < bw_threshold)[0]
        if len(below) > 0:
            bw = freq_norm[below[0]]
            ax.axvline(bw, color=color, linestyle='--', alpha=0.5)
            ax.annotate(f'{label} bandwidth\n= {bw:.2f}π',
                       xy=(bw, bw_threshold), xytext=(bw + 0.08, bw_threshold * 2),
                       fontsize=9, color=color,
                       arrowprops=dict(arrowstyle='->', color=color))

    # Annotate regions
    ax.axvspan(0, 0.08, alpha=0.08, color='green')
    ax.axvspan(0.08, 0.35, alpha=0.08, color='yellow')
    ax.axvspan(0.35, 1.0, alpha=0.08, color='red')

    ax.text(0.04, H11[0] * 1.3, 'Network tracks\ninput faithfully', fontsize=10,
            ha='center', color='#2E7D32', fontweight='bold')
    ax.text(0.21, H11[0] * 0.15, 'Transition\nregion', fontsize=10,
            ha='center', color='#F57F17')
    ax.text(0.65, H11[0] * 0.02, 'Network ignores\nrapid changes', fontsize=10,
            ha='center', color='#C62828')

    # X-axis labels
    ax.set_xlabel('Input modulation speed (normalized frequency, ×π rad/step)', fontsize=12)
    ax.set_ylabel('Network response (gain)', fontsize=12)
    ax.set_title('Frequency Response: How the Network Responds\nto Inputs of Different Speeds',
                 fontsize=14)
    ax.legend(fontsize=11, loc='center right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0, 1)

    # Add top x-axis with intuitive labels
    ax2 = ax.twiny()
    ax2.set_xlim(0, 1)
    ax2.set_xticks([0, 0.15, 0.5, 1.0])
    ax2.set_xticklabels(['Constant\n(DC)', 'Slow\nmodulation', 'Medium\nspeed', 'Alternating\nevery step'],
                        fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/cs_bode_annotated.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {PLOT_DIR}/cs_bode_annotated.png")
    plt.close()


def plot_time_domain_filtering():
    """Show network filtering slow vs fast inputs."""
    T = 60
    u_base = np.ones(NUM_NEURONS)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    scenarios = [
        ('Slow modulation (period = 20 steps)', 20),
        ('Fast modulation (period = 4 steps)', 4),
    ]

    for col, (title, period) in enumerate(scenarios):
        # Create time-varying input for S1 only (others constant)
        t_arr = np.arange(T)
        input_mod = 0.5 + 0.5 * np.sin(2 * np.pi * t_arr / period)

        # Simulate
        p = np.zeros(NUM_NEURONS)
        rates = firing_rate_sigmoid(p)
        potentials = np.zeros((T + 1, NUM_NEURONS))
        rates_trace = np.zeros((T + 1, NUM_NEURONS))

        for t in range(T):
            u = np.ones(NUM_NEURONS)
            u[0] = input_mod[t]  # Modulate S1
            p, rates = lif_step_meanfield_sigmoid(p, rates, u)
            potentials[t + 1] = p
            rates_trace[t + 1] = rates

        # Top: Input signal
        ax = axes[0, col]
        ax.plot(t_arr, input_mod, color='#1565C0', linewidth=2, label='S1 input rate')
        ax.axhline(1.0, color='gray', linestyle=':', alpha=0.3)
        ax.set_ylabel('Input rate')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylim(-0.1, 1.2)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Bottom: Network response (N1 firing rate)
        ax = axes[1, col]
        ax.plot(range(T + 1), rates_trace[:, 0], color=COLORS['N1'], linewidth=2, label='N1 firing rate')
        ax.set_ylabel('N1 firing rate')
        ax.set_xlabel('Time step')
        ax.set_ylim(-0.1, 1.2)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        if col == 0:
            ax.annotate('Network tracks\nthe slow input', xy=(35, rates_trace[35, 0]),
                        xytext=(40, 0.9), fontsize=10, color='#2E7D32', fontweight='bold',
                        bbox=dict(boxstyle='round', fc='#E8F5E9', ec='#4CAF50'),
                        arrowprops=dict(arrowstyle='->', color='#4CAF50'))
        else:
            ax.annotate('Network barely\nresponds to\nfast changes', xy=(30, rates_trace[30, 0]),
                        xytext=(35, 0.8), fontsize=10, color='#C62828', fontweight='bold',
                        bbox=dict(boxstyle='round', fc='#FFEBEE', ec='#C62828'),
                        arrowprops=dict(arrowstyle='->', color='#C62828'))

    fig.suptitle('Low-Pass Filtering: The Network Smooths Rapid Fluctuations',
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/cs_time_domain_filtering.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {PLOT_DIR}/cs_time_domain_filtering.png")
    plt.close()


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print("Generating CS-friendly visualizations...")
    plot_network_graph()
    plot_eigenvalue_spectrum()
    plot_eigenvector_heatmap()
    plot_convergence_trace()
    plot_basin_simplified()
    plot_bode_annotated()
    plot_time_domain_filtering()
    print(f"\nAll plots saved to {PLOT_DIR}/cs_*.png")


if __name__ == "__main__":
    main()
