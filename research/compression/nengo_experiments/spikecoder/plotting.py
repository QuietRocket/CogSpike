"""Consistent matplotlib figures for the experiment suite (Agg backend).

Style mirrors the DEQ program's plots for visual continuity. Every figure helper
saves to a path and returns the figure so callers can compose or annotate further.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .information import H_RATE, H_MARGINAL  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_MEASURED = "#1f77b4"
C_THEORY = "#d62728"
C_FLOOR = "#2ca02c"


def new_fig(w=6.0, h=4.0):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_latency_vs_q(qs, measured, theory, path, title="Latency = surprisal",
                      ylabel="first-spike latency (s)", logx=False):
    """Overlay measured first-spike latency on the analytic law vs q."""
    fig, ax = new_fig()
    order = np.argsort(qs)
    qs = np.asarray(qs)[order]
    ax.plot(qs, np.asarray(theory)[order], "-", color=C_THEORY, lw=2,
            label=r"theory $-\lambda\log_2 q$")
    ax.plot(qs, np.asarray(measured)[order], "o", color=C_MEASURED, ms=6,
            label="Nengo measured")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel("model probability q")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    return save(fig, path)


def plot_energy_descent(steps, energies, path, title="Energy descent to the floor",
                        learned_ref=None):
    """Plot energy (bits/symbol) vs training step with the entropy-rate floor line."""
    fig, ax = new_fig()
    ax.plot(steps, energies, "-", color=C_MEASURED, lw=2, label="energy E(W)")
    ax.axhline(H_RATE, color=C_FLOOR, ls="--", lw=1.5,
               label=f"entropy-rate floor {H_RATE:.4f}")
    ax.axhline(H_MARGINAL, color="gray", ls=":", lw=1.2,
               label=f"marginal {H_MARGINAL:.4f}")
    if learned_ref is not None:
        ax.axhline(learned_ref, color="orange", ls="-.", lw=1.0, alpha=0.7,
                   label=f"learned ref {learned_ref:.4f}")
    ax.set_xlabel("training step")
    ax.set_ylabel("energy (bits/symbol)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    return save(fig, path)


def plot_raster(trange, spikes_2d, path, labels=None, title="spike raster"):
    """Spike raster for a small population (T, N)."""
    spikes_2d = np.asarray(spikes_2d)
    N = spikes_2d.shape[1]
    fig, ax = new_fig(h=0.5 * N + 1.5)
    for j in range(N):
        ts = np.asarray(trange)[spikes_2d[:, j] > 0]
        ax.vlines(ts, j + 0.6, j + 1.4, color=C_MEASURED, lw=1.2)
    ax.set_yticks(range(1, N + 1))
    ax.set_yticklabels(labels if labels is not None else range(1, N + 1))
    ax.set_xlabel("time (s)")
    ax.set_title(title)
    ax.set_ylim(0.4, N + 0.6)
    return save(fig, path)


def plot_bar_compare(categories, measured, theory, path, ylabel="bits/symbol",
                     title="measured vs theory"):
    """Grouped bar chart comparing measured and theoretical values."""
    x = np.arange(len(categories))
    fig, ax = new_fig()
    ax.bar(x - 0.2, measured, 0.4, color=C_MEASURED, label="measured")
    ax.bar(x + 0.2, theory, 0.4, color=C_THEORY, alpha=0.7, label="theory")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    return save(fig, path)
