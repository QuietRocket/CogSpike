"""Phase 3: Quasi-renewal (finite-N) reading of FCS Property 5.

Mesoscopic two-population simulator for the negative-loop motif using
Naud-Gerstner age-distribution dynamics. Same calibration as Phases
1-2. For N in {50, 100, 500, 2000}, simulate T=400 ticks, drop warmup,
and measure on the population activity trace A(t):

  • dominant period via FFT peak (real-valued period, can be
    non-integer);
  • 1100-template match score: fit the best phase-shifted boxcar with
    period 4 to the trace, return correlation.

Per-cell labels at each N:
  qr_p5_blue(N): |period - 4| <= 0.5  AND  template_score >= 0.5
  qr_osc_blue(N): trace amplitude (std / mean) >= 0.1 (sustained
                  oscillation, period agnostic)

Hypotheses:
  (i)  N small: large sqrt(A/N) noise; ringing may sustain at the
       linearization frequency rather than the FCS period (Phase 2's
       factor-of-4 should manifest).
  (ii) N large: trace converges to deterministic rate equations
       (decaying spiral); no sustained oscillation; qr_osc_blue
       collapses.
  (iii) The FCS strict-P5 region is invisible to QR at any N for the
        same reason as Phases 1-2 — discrete-tick spike timing is
        beyond rate-equation reach.

Output: results/phase3/qr_grid.npz, qr_n_sweep.pdf, qr_jaccard_vs_N.pdf,
period_qr_vs_FCS.pdf.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from deq.closed_form.siegert import Siegert  # noqa: E402
from deq.closed_form.quasi_renewal import QuasiRenewal  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "phase3"
RESULTS.mkdir(parents=True, exist_ok=True)


P_THIN = 0.7
T_TOTAL = 400
T_WARMUP_QR = 100
N_SWEEP = [50, 100, 500, 2000]
SEED = 1234
PERIOD_TARGET = 4.0
TOL_PERIOD = 0.5
AMP_GATE = 0.10            # std/mean threshold for "sustained oscillation"
TEMPLATE_THRESH = 0.5      # 1100-template correlation threshold


def load_calibration() -> dict:
    src = ROOT / "deq" / "closed_form" / "results" / "phase1_grid.npz"
    d = np.load(src, allow_pickle=True)
    return dict(
        alpha=float(d["calib_alpha"]),
        beta=float(d["calib_beta"]),
        tau_m=float(d["calib_tau_m"]),
        tau_ref=float(d["calib_tau_ref"]),
    )


def simulate_neg_loop_qr(w_AI: float, w_IA: float, w_XA: float,
                         qr: QuasiRenewal, alpha: float, beta: float,
                         N: int, T: int, seed: int = 0,
                         init_A=(0.3, 0.1)) -> np.ndarray:
    """Run quasi-renewal mesoscopic simulation of the negative-loop motif.

    Population 0 = A (activator, receives external X and inhibition from I).
    Population 1 = I (inhibitor, receives excitation from A only).

    Returns:
        rates: (2, T) per-tick population activity.
    """
    rng = np.random.default_rng(seed)
    m = np.zeros((2, qr.K_max))
    for i, A0 in enumerate(init_A):
        A0s = max(min(A0, 0.99), 1e-3)
        for k in range(qr.K_max):
            m[i, k] = A0s * (1 - A0s) ** k
        m[i] /= m[i].sum()
    A_prev = np.array(init_A, dtype=float)
    rates = np.zeros((2, T))

    for t in range(T):
        # A: external w_XA Bernoulli plus inhibitory recurrent w_IA * A_I.
        mean_in_A = w_XA * P_THIN + w_IA * A_prev[1]
        var_in_A = ((w_XA ** 2) * P_THIN * (1 - P_THIN)
                    + (w_IA ** 2) * A_prev[1] * (1 - A_prev[1]))
        # I: w_AI * A_A only.
        mean_in_I = w_AI * A_prev[0]
        var_in_I = (w_AI ** 2) * A_prev[0] * (1 - A_prev[0])
        mu = np.array([alpha * mean_in_A, alpha * mean_in_I])
        sigma = np.array([
            math.sqrt(max(beta * var_in_A, 0.0)),
            math.sqrt(max(beta * var_in_I, 0.0)),
        ])
        m, A = qr.step(m, mu, sigma, N, rng=rng)
        rates[:, t] = A
        A_prev = A

    return rates


def dominant_period_fft(trace: np.ndarray,
                        min_period: float = 2.0,
                        max_period: float = 32.0) -> float:
    """Return the dominant FFT period (in ticks) restricted to [min, max].
    Returns 0 if no significant peak.
    """
    n = len(trace)
    if n < 8:
        return 0.0
    x = trace - trace.mean()
    if np.std(x) < 1e-6:
        return 0.0
    F = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0)
    power = np.abs(F) ** 2
    # Restrict to [1/max_period, 1/min_period] range.
    mask = (freqs >= 1.0 / max_period) & (freqs <= 1.0 / min_period)
    if not mask.any():
        return 0.0
    idx_loc = np.argmax(power[mask])
    idx_global = np.where(mask)[0][idx_loc]
    f_peak = freqs[idx_global]
    if f_peak < 1e-8:
        return 0.0
    return 1.0 / f_peak


def template_1100_score(trace: np.ndarray) -> float:
    """Correlation against the best phase-shifted period-4 1100 boxcar."""
    n = len(trace)
    if n < 8:
        return 0.0
    x = trace - trace.mean()
    if np.std(x) < 1e-6:
        return 0.0
    best = -1.0
    for shift in range(4):
        pat = np.array([1.0, 1.0, 0.0, 0.0])
        pat = np.roll(pat, shift)
        tmpl = np.tile(pat, n // 4 + 1)[:n]
        tmpl = tmpl - tmpl.mean()
        denom = np.std(x) * np.std(tmpl)
        if denom < 1e-8:
            continue
        c = float(np.mean(x * tmpl) / denom)
        if c > best:
            best = c
    return max(best, 0.0)


def amplitude_score(trace: np.ndarray) -> float:
    """Std / mean -- a coefficient of variation; oscillation strength."""
    m = float(np.mean(trace))
    s = float(np.std(trace))
    if m < 1e-6:
        return 0.0
    return s / m


def jaccard(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = (a & b).sum(); union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else 1.0


def main():
    calib = load_calibration()
    print("Calibration (locked):", calib)
    print()

    siegert = Siegert(V_th=1.0, V_r=0.0,
                      tau_m=calib["tau_m"], tau_ref=calib["tau_ref"])
    qr = QuasiRenewal(siegert=siegert, K_max=30, tau_ref_ticks=0, dt=1.0)

    W_AI = 11

    p0 = np.load(HERE / "results" / "phase0" / "fcs_grid.npz",
                 allow_pickle=True)
    W_IA = p0["W_IA_VALUES"]
    W_XA = p0["W_XA_VALUES"]
    strict_p5 = p0["strict_p5"].astype(bool)
    broad_osc = p0["broad_osc"].astype(bool)
    fcs_period = p0["period"]

    nIA, nXA = len(W_IA), len(W_XA)

    print(f"Quasi-renewal sweep on {nIA} x {nXA} grid; N sweep: {N_SWEEP}")
    print(f"T={T_TOTAL}, warmup={T_WARMUP_QR}\n")

    qr_period = {}
    qr_template = {}
    qr_amp = {}
    qr_p5_blue = {}
    qr_osc_blue = {}
    sample_trace = {}  # save the default-cell trace at each N for plotting

    i_def = int(np.where(W_IA == -11)[0][0])
    j_def = int(np.where(W_XA == 11)[0][0])

    for N in N_SWEEP:
        t0 = time.time()
        period = np.zeros((nIA, nXA))
        template = np.zeros((nIA, nXA))
        amp = np.zeros((nIA, nXA))
        p5_blue = np.zeros((nIA, nXA), dtype=int)
        osc_blue = np.zeros((nIA, nXA), dtype=int)

        for i, w_IA in enumerate(W_IA):
            for j, w_XA in enumerate(W_XA):
                rates = simulate_neg_loop_qr(
                    w_AI=float(W_AI), w_IA=float(w_IA), w_XA=float(w_XA),
                    qr=qr, alpha=calib["alpha"], beta=calib["beta"],
                    N=N, T=T_TOTAL,
                    seed=SEED + i * nXA + j,
                )
                trace_A = rates[0, T_WARMUP_QR:]
                period[i, j] = dominant_period_fft(trace_A)
                template[i, j] = template_1100_score(trace_A)
                amp[i, j] = amplitude_score(trace_A)
                p5_blue[i, j] = int(
                    (abs(period[i, j] - PERIOD_TARGET) <= TOL_PERIOD)
                    and (template[i, j] >= TEMPLATE_THRESH)
                )
                osc_blue[i, j] = int(amp[i, j] >= AMP_GATE)

                if i == i_def and j == j_def:
                    sample_trace[N] = rates

        qr_period[N] = period
        qr_template[N] = template
        qr_amp[N] = amp
        qr_p5_blue[N] = p5_blue
        qr_osc_blue[N] = osc_blue

        elapsed = time.time() - t0
        j_strict = jaccard(p5_blue, strict_p5)
        j_broad_osc = jaccard(osc_blue, broad_osc)
        print(f"  N={N:4d}: qr_p5_blue={int(p5_blue.sum()):4d}, "
              f"qr_osc_blue={int(osc_blue.sum()):4d}, "
              f"J(p5)={j_strict:.3f}, J(broad)={j_broad_osc:.3f}, "
              f"elapsed {elapsed:.1f}s")

    np.savez(
        RESULTS / "qr_grid.npz",
        W_IA_VALUES=W_IA,
        W_XA_VALUES=W_XA,
        W_AI=W_AI,
        N_SWEEP=np.array(N_SWEEP),
        qr_period=np.stack([qr_period[N] for N in N_SWEEP]),
        qr_template=np.stack([qr_template[N] for N in N_SWEEP]),
        qr_amp=np.stack([qr_amp[N] for N in N_SWEEP]),
        qr_p5_blue=np.stack([qr_p5_blue[N] for N in N_SWEEP]),
        qr_osc_blue=np.stack([qr_osc_blue[N] for N in N_SWEEP]),
        T_total=T_TOTAL, T_warmup=T_WARMUP_QR,
        period_target=PERIOD_TARGET, tol_period=TOL_PERIOD,
        amp_gate=AMP_GATE, template_thresh=TEMPLATE_THRESH,
    )

    # ---------- plots ----------

    # Grid panel: FCS strict-P5 / FCS broad-osc / qr_p5_blue per N.
    n_panels = 2 + len(N_SWEEP)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5))
    panels = [
        (axes[0], strict_p5.astype(int),
            f"FCS strict P5\n{int(strict_p5.sum())}/{strict_p5.size}"),
        (axes[1], broad_osc.astype(int),
            f"FCS broad osc\n{int(broad_osc.sum())}/{broad_osc.size}"),
    ]
    for k, N in enumerate(N_SWEEP):
        j_s = jaccard(qr_p5_blue[N], strict_p5)
        panels.append((
            axes[2 + k], qr_p5_blue[N],
            f"QR N={N}, p5_blue\n"
            f"{int(qr_p5_blue[N].sum())}/{qr_p5_blue[N].size}\n"
            f"J(strict_p5)={j_s:.3f}",
        ))
    for ax, labels, title in panels:
        for i, w_IA in enumerate(W_IA):
            for j, w_XA in enumerate(W_XA):
                color = "tab:blue" if labels[i, j] else "tab:red"
                ax.scatter(int(w_XA), int(w_IA), c=color, s=14,
                           edgecolor="none")
        ax.scatter([11], [-11], facecolors="none", edgecolors="gold",
                   s=120, linewidths=1.6, zorder=5)
        ax.set_xlabel(r"$w_{XA}$"); ax.set_ylabel(r"$w_{IA}$")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3); ax.set_aspect("equal")
    fig.suptitle("Phase 3: Quasi-renewal Property-5 labels vs FCS, sweep over N",
                 fontsize=11)
    plt.tight_layout()
    out_pdf = RESULTS / "qr_n_sweep.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\n  wrote {out_pdf}")
    plt.close(fig)

    # Period-comparison: QR period at N=2000 vs FCS period.
    fig, axes = plt.subplots(1, len(N_SWEEP), figsize=(5 * len(N_SWEEP), 4.5))
    for k, N in enumerate(N_SWEEP):
        ax = axes[k]
        valid = (qr_period[N] > 0) & (fcs_period > 0)
        xs = fcs_period[valid]
        ys = qr_period[N][valid]
        is_p5 = strict_p5[valid]
        rng = np.random.default_rng(0)
        xj = xs + rng.uniform(-0.15, 0.15, size=len(xs))
        ax.scatter(xj[~is_p5], ys[~is_p5], c="lightgray", s=6, alpha=0.4)
        ax.scatter(xj[is_p5], ys[is_p5], c="tab:blue", s=12, alpha=0.85,
                   label="FCS strict P5")
        ax.plot([0, 13], [0, 13], "k--", alpha=0.4, label="y = x")
        ax.axhline(4, color="tab:green", linestyle=":", alpha=0.7,
                   label="QR period 4")
        ax.set_xlim(0.5, 12.5); ax.set_ylim(0, 32)
        ax.set_xlabel("FCS-measured period (ticks)")
        ax.set_ylabel("QR FFT-dominant period (ticks)")
        ax.set_title(f"N = {N}", fontsize=10)
        ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Phase 3: QR-measured period vs FCS-measured period, "
                 "by population size", fontsize=11)
    plt.tight_layout()
    out_pdf = RESULTS / "period_qr_vs_FCS.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    plt.close(fig)

    # Jaccard / oscillation amplitude as functions of N.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    j_p5_vec = [jaccard(qr_p5_blue[N], strict_p5) for N in N_SWEEP]
    j_osc_vec = [jaccard(qr_osc_blue[N], broad_osc) for N in N_SWEEP]
    mean_amp = [float(qr_amp[N].mean()) for N in N_SWEEP]
    axes[0].plot(N_SWEEP, j_p5_vec, "o-", color="tab:purple",
                 label="J(qr_p5_blue, FCS strict_p5)")
    axes[0].plot(N_SWEEP, j_osc_vec, "s-", color="tab:orange",
                 label="J(qr_osc_blue, FCS broad_osc)")
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_xscale("log"); axes[0].set_xlabel("N (population size)")
    axes[0].set_ylabel("Jaccard"); axes[0].set_title("Phase 3 Jaccard vs N", fontsize=10)
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(N_SWEEP, mean_amp, "d-", color="tab:red")
    axes[1].axhline(AMP_GATE, color="gray", linestyle="--",
                    alpha=0.5, label=f"amp gate = {AMP_GATE}")
    axes[1].set_xscale("log"); axes[1].set_xlabel("N (population size)")
    axes[1].set_ylabel("grid-mean amplitude (std/mean)")
    axes[1].set_title("Phase 3 grid-mean oscillation amplitude vs N",
                      fontsize=10)
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_pdf = RESULTS / "qr_jaccard_vs_N.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    plt.close(fig)

    # Default-cell trace at each N.
    fig, axes = plt.subplots(len(N_SWEEP), 1, figsize=(9, 2 * len(N_SWEEP)),
                             sharex=True)
    if len(N_SWEEP) == 1:
        axes = [axes]
    show_window = slice(T_WARMUP_QR, T_WARMUP_QR + 60)
    for k, N in enumerate(N_SWEEP):
        ax = axes[k]
        rates = sample_trace[N]
        ax.plot(rates[0, show_window], label="A", color="tab:blue", lw=1.4)
        ax.plot(rates[1, show_window], label="I", color="tab:orange",
                lw=1.0, alpha=0.7)
        ax.set_ylim(-0.05, 1.05); ax.set_ylabel(f"N = {N}\nrate")
        ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)
        # mark the period-4 grid
        for x in range(0, 60, 4):
            ax.axvline(x, color="gray", linestyle=":", alpha=0.2)
    axes[-1].set_xlabel("tick (post-warmup)")
    fig.suptitle(
        "Phase 3: default-cell (w_IA=-11, w_XA=11) QR traces, 60 ticks",
        fontsize=11,
    )
    plt.tight_layout()
    out_pdf = RESULTS / "default_cell_traces.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    plt.close(fig)

    # Default-cell summary stats.
    print("\nFCS default cell (w_IA=-11, w_XA=11) summary:")
    for N in N_SWEEP:
        print(f"  N={N:4d}: QR period={qr_period[N][i_def, j_def]:.2f}, "
              f"template={qr_template[N][i_def, j_def]:.2f}, "
              f"amp={qr_amp[N][i_def, j_def]:.2f}, "
              f"p5_blue={qr_p5_blue[N][i_def, j_def]}, "
              f"osc_blue={qr_osc_blue[N][i_def, j_def]}")

    print()
    print("Phase 3 verdict:")
    j_n50 = jaccard(qr_osc_blue[50], broad_osc)
    j_n2000 = jaccard(qr_osc_blue[2000], broad_osc)
    print(f"  J(qr_osc, FCS broad_osc) at N=50: {j_n50:.3f}; "
          f"at N=2000: {j_n2000:.3f}")
    if j_n50 > j_n2000:
        print("  PASS: noise at small N sustains oscillation more than at "
              "large N (mean-field collapses spiraling into the FP), so "
              "the QR osc-label tracks FCS broad-osc better at finite N.")
    else:
        print("  PARTIAL: finite-N noise does not preferentially enhance "
              "oscillation in this regime; both N's behave similarly.")


if __name__ == "__main__":
    main()
