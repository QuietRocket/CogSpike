"""
Z-Domain Transfer Function Analysis
=====================================

For the linearized discrete-time system:
  p(t+1) = A*p(t) + B*u(t)
  y(t) = C*p(t)

The transfer function matrix is:
  H(z) = C * (zI - A)^{-1} * B

This is a 4x4 matrix of SISO transfer functions. Key entries:
  H_11(z): S1 → N1 (winner's direct path + feedback through inhibition)
  H_21(z): S1 → N2 (cross-inhibition: how S1 suppresses N2)
  H_12(z): S2 → N1 (how competing input affects winner)

The Z-transform is the correct framework for this discrete-time system.
Connection to Laplace: z = e^{sT} where T is the sampling period (1 step).
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from network import NUM_NEURONS, NEURON_LABELS, LEAK_R, W, B_IN, C_STATE, D_ZERO
from linearization import find_operating_point, build_state_space


def compute_transfer_matrix(A, B, C, D):
    """Compute the discrete-time transfer function matrix H(z) = C(zI-A)^{-1}B + D.

    Returns a 4x4 array of scipy.signal.TransferFunction objects (discrete).
    """
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]

    tf_matrix = np.empty((p, m), dtype=object)

    # Build MIMO state-space and extract SISO transfer functions
    for i in range(p):
        for j in range(m):
            # Extract SISO: input j → output i
            # Use state-space to transfer function conversion
            Cij = np.zeros((1, n))
            Cij[0, :] = C[i, :]
            Bij = np.zeros((n, 1))
            Bij[:, 0] = B[:, j]
            Dij = np.array([[D[i, j]]])

            sys = signal.StateSpace(A, Bij, Cij, Dij, dt=1)
            tf = sys.to_tf()
            tf_matrix[i, j] = tf

    return tf_matrix


def analyze_transfer_function(tf, name):
    """Analyze a single SISO transfer function."""
    print(f"\n  {name}:")

    # Numerator and denominator
    num = tf.num.flatten()
    den = tf.den.flatten()
    print(f"    Numerator coeffs:   {np.round(num, 4).tolist()}")
    print(f"    Denominator coeffs: {np.round(den, 4).tolist()}")

    # Poles and zeros
    zeros = np.roots(num)
    poles = np.roots(den)
    print(f"    Poles:  {np.round(poles, 6).tolist()}")
    print(f"    Zeros:  {np.round(zeros, 6).tolist()}")

    # DC gain: H(z=1) = H(1)
    dc_gain = np.polyval(num, 1.0) / np.polyval(den, 1.0) if np.abs(np.polyval(den, 1.0)) > 1e-10 else np.inf
    print(f"    DC gain H(1): {dc_gain:.6f}")

    return {
        'num': num, 'den': den,
        'poles': poles, 'zeros': zeros,
        'dc_gain': dc_gain,
    }


def plot_pole_zero_map(all_tf_info, save_dir="plots"):
    """Plot pole-zero maps for key transfer functions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 200)

    key_tfs = [
        ('H_11 (S1→N1, winner path)', 'H11'),
        ('H_21 (S1→N2, cross-inhibition)', 'H21'),
        ('H_12 (S2→N1, competitor→winner)', 'H12'),
    ]

    for ax, (title, key) in zip(axes, key_tfs):
        info = all_tf_info[key]
        ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1, alpha=0.3)

        # Plot poles
        for p in info['poles']:
            ax.plot(p.real, p.imag, 'rx', markersize=12, markeredgewidth=2)
        # Plot zeros
        for z in info['zeros']:
            ax.plot(z.real, z.imag, 'bo', markersize=10, markerfacecolor='none',
                    markeredgewidth=2)

        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5)
        ax.set_title(f'{title}\nDC gain = {info["dc_gain"]:.4f}')
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Set limits
        all_points = np.concatenate([info['poles'], info['zeros']])
        if len(all_points) > 0:
            margin = 0.3
            xlim = [min(all_points.real.min(), -1) - margin,
                    max(all_points.real.max(), 1) + margin]
            ylim = [min(all_points.imag.min(), -1) - margin,
                    max(all_points.imag.max(), 1) + margin]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    fig.suptitle('Pole-Zero Maps (Z-Domain) — Linearized WTA Network', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/transfer_pole_zero.png", dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {save_dir}/transfer_pole_zero.png")
    plt.close()


def plot_dc_gain_matrix(tf_matrix, A, B, C, D, save_dir="plots"):
    """Plot the DC gain matrix as a heatmap."""
    n = A.shape[0]
    # DC gain matrix: H(1) = C(I - A)^{-1}B + D
    try:
        dc_matrix = C @ np.linalg.inv(np.eye(n) - A) @ B + D
    except np.linalg.LinAlgError:
        print("  Warning: (I - A) is singular, DC gain matrix undefined")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(dc_matrix, cmap='RdBu_r', aspect='equal')
    plt.colorbar(im, ax=ax, label='DC Gain')

    # Annotate
    for i in range(NUM_NEURONS):
        for j in range(NUM_NEURONS):
            color = 'white' if abs(dc_matrix[i, j]) > np.max(np.abs(dc_matrix)) * 0.6 else 'black'
            ax.text(j, i, f'{dc_matrix[i,j]:.2f}', ha='center', va='center',
                    color=color, fontsize=11)

    ax.set_xticks(range(NUM_NEURONS))
    ax.set_xticklabels([f'S{i+1} input' for i in range(NUM_NEURONS)])
    ax.set_yticks(range(NUM_NEURONS))
    ax.set_yticklabels([f'N{i+1} potential' for i in range(NUM_NEURONS)])
    ax.set_title('DC Gain Matrix H(z=1)\n(Steady-State Input→Output Mapping)')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/transfer_dc_gain_matrix.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_dir}/transfer_dc_gain_matrix.png")
    plt.close()

    return dc_matrix


def run_transfer_analysis(save_dir="plots"):
    """Run complete transfer function analysis."""
    print("\n" + "=" * 60)
    print("Z-DOMAIN TRANSFER FUNCTION ANALYSIS")
    print("=" * 60)

    # Use the all-inputs-on regime
    u = np.ones(NUM_NEURONS)
    p_star, r_star = find_operating_point(u)
    A, B, C, D = build_state_space(p_star)

    print(f"\n  Operating point: p* = {np.round(p_star, 2).tolist()}")
    print(f"  Firing rates:    f* = {np.round(r_star, 4).tolist()}")

    # Compute transfer function matrix
    print(f"\n  Computing H(z) = C(zI - A)^{{-1}}B ...")
    tf_matrix = compute_transfer_matrix(A, B, C, D)

    # Analyze key transfer functions
    all_tf_info = {}

    key_pairs = [
        (0, 0, 'H11', 'H_11: S1 → N1 (winner direct path)'),
        (1, 0, 'H21', 'H_21: S1 → N2 (cross-inhibition)'),
        (0, 1, 'H12', 'H_12: S2 → N1 (competitor → winner)'),
        (1, 1, 'H22', 'H_22: S2 → N2 (loser direct path)'),
    ]

    for i, j, key, name in key_pairs:
        info = analyze_transfer_function(tf_matrix[i, j], name)
        all_tf_info[key] = info

    # DC gain comparison
    print(f"\n  DC Gain Summary:")
    print(f"    Winner path  H_11(1) = {all_tf_info['H11']['dc_gain']:.4f}")
    print(f"    Loser path   H_22(1) = {all_tf_info['H22']['dc_gain']:.4f}")
    print(f"    Cross-inhib  H_21(1) = {all_tf_info['H21']['dc_gain']:.4f}")
    print(f"    Comp→winner  H_12(1) = {all_tf_info['H12']['dc_gain']:.4f}")
    ratio = all_tf_info['H11']['dc_gain'] / all_tf_info['H22']['dc_gain'] if all_tf_info['H22']['dc_gain'] != 0 else np.inf
    print(f"    Winner/Loser ratio: {ratio:.4f}")

    # Plot DC gain matrix
    dc_matrix = plot_dc_gain_matrix(tf_matrix, A, B, C, D, save_dir)
    if dc_matrix is not None:
        print(f"\n  Full DC Gain Matrix H(1):")
        print(f"    {dc_matrix}")

    # Plot pole-zero maps
    plot_pole_zero_map(all_tf_info, save_dir)

    return {
        'tf_matrix': tf_matrix,
        'tf_info': all_tf_info,
        'dc_matrix': dc_matrix,
        'A': A, 'B': B, 'C': C, 'D': D,
    }


if __name__ == "__main__":
    run_transfer_analysis()
