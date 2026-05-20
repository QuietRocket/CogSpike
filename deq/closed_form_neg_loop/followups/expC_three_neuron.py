"""Experiment C: 3-neuron negative loop with one delayer (A -> D -> I -> A).

Apply the three lenses (FCS oracle, Siegert FP + Jacobian, quasi-renewal
mesoscopic) to the FCS-Fig.3-style negative loop extended by one
delayer cell D between the activator A and inhibitor I. Compare period
predictions to the 2-neuron case.

Topology (neurons 0=A, 1=D, 2=I):
  A: external X (w_XA) + inhibition from I (w_IA < 0)
  D: excited by A (w_AD)
  I: excited by D (w_DI)
  W = [[0,    0,    w_IA],
       [w_AD, 0,    0   ],
       [0,    w_DI, 0   ]]
  B = [[w_XA], [0], [0]]

Defaults: w_XA = w_AD = w_DI = 11, w_IA = -11.

Output: results/expC/three_neuron.{npz,pdf}
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from deq.archetypes.lif_fcs import simulate, spike_sequence_to_str  # noqa: E402
from deq.closed_form.siegert import Siegert  # noqa: E402
from deq.closed_form.transfer import dphi_dmu, jacobian_eigenvalues  # noqa: E402
from deq.closed_form.quasi_renewal import QuasiRenewal  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "expC"
RESULTS.mkdir(parents=True, exist_ok=True)


P_THIN = 0.7
W_AD_DEFAULT = 11
W_DI_DEFAULT = 11
W_IA_DEFAULT = -11
W_XA_DEFAULT = 11


def load_calibration() -> dict:
    src = ROOT / "deq" / "closed_form" / "results" / "phase1_grid.npz"
    d = np.load(src, allow_pickle=True)
    return dict(
        alpha=float(d["calib_alpha"]),
        beta=float(d["calib_beta"]),
        tau_m=float(d["calib_tau_m"]),
        tau_ref=float(d["calib_tau_ref"]),
    )


def topology_neg_loop_3(w_XA: int, w_AD: int, w_DI: int, w_IA: int, T: int):
    """Build the (W, B, ext) tuple for the 3-neuron negative loop."""
    W = np.array([
        [0,    0,    w_IA],
        [w_AD, 0,    0   ],
        [0,    w_DI, 0   ],
    ], dtype=np.int64)
    B = np.array([[w_XA], [0], [0]], dtype=np.int64)
    ext = np.ones((1, T), dtype=np.int64)
    return W, B, ext


def detect_period(seq: np.ndarray, max_period: int = 12) -> int:
    """Return smallest p in [1, max_period] for which the tail of seq is
    periodic with period p over at least 3 consecutive cycles.
    Returns 0 if no period found.
    """
    n = len(seq)
    if n < 3 * max_period:
        max_period = max(1, n // 3)
    s = seq.astype(int)
    for p in range(1, max_period + 1):
        if 3 * p > n:
            break
        # require 3 consecutive matching cycles in the tail
        tail = s[-3 * p:]
        if np.array_equal(tail[:p], tail[p:2 * p]) \
                and np.array_equal(tail[:p], tail[2 * p:3 * p]):
            if p == 1:
                # Constant period 1: only label if the entire post-warmup is constant.
                if np.all(s == s[0]):
                    return 1
                else:
                    continue
            return p
    return 0


def fft_period(trace: np.ndarray, min_period=2.0, max_period=32.0) -> float:
    n = len(trace)
    if n < 8:
        return 0.0
    x = trace - trace.mean()
    if np.std(x) < 1e-9:
        return 0.0
    F = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0)
    pw = np.abs(F) ** 2
    mask = (freqs >= 1.0 / max_period) & (freqs <= 1.0 / min_period)
    if not mask.any():
        return 0.0
    idx_loc = np.argmax(pw[mask])
    idx_glob = np.where(mask)[0][idx_loc]
    fp = freqs[idx_glob]
    if fp < 1e-8:
        return 0.0
    return 1.0 / fp


def neg3_inputs(nu: np.ndarray, w_AD, w_DI, w_IA, w_XA,
                alpha, beta):
    """Per-population (mu, sigma) for the 3-neuron negative loop.

    nu = [nu_A, nu_D, nu_I].
    """
    nu_A, nu_D, nu_I = float(nu[0]), float(nu[1]), float(nu[2])
    mean_A = w_XA * P_THIN + w_IA * nu_I
    mean_D = w_AD * nu_A
    mean_I = w_DI * nu_D
    var_A = ((w_XA ** 2) * P_THIN * (1 - P_THIN)
             + (w_IA ** 2) * nu_I * (1 - nu_I))
    var_D = (w_AD ** 2) * nu_A * (1 - nu_A)
    var_I = (w_DI ** 2) * nu_D * (1 - nu_D)
    mu = alpha * np.array([mean_A, mean_D, mean_I])
    sigma = np.array([
        math.sqrt(max(beta * var_A, 0.0)),
        math.sqrt(max(beta * var_D, 0.0)),
        math.sqrt(max(beta * var_I, 0.0)),
    ])
    return mu, sigma


def find_fp_neg3(w_AD, w_DI, w_IA, w_XA, siegert, alpha, beta):
    """3-D fixed point of the negative-loop-with-delayer rate equations."""
    def residual(nu):
        nu_c = np.clip(nu, 0, 1)
        mu, sigma = neg3_inputs(nu_c, w_AD, w_DI, w_IA, w_XA, alpha, beta)
        nu_pred = np.array([siegert.phi(mu[i], sigma[i]) for i in range(3)])
        return nu - nu_pred

    guesses = [(0.5, 0.5, 0.5), (0.3, 0.3, 0.3), (0.7, 0.5, 0.3),
               (0.4, 0.4, 0.4), (0.6, 0.4, 0.2)]
    best = None
    best_res = float("inf")
    for g in guesses:
        try:
            sol, _info, ier, _ = fsolve(residual, np.array(g, dtype=float),
                                        full_output=True, xtol=1e-9)
            r = float(np.max(np.abs(residual(sol))))
            if ier == 1 and r < best_res:
                best = np.clip(sol, 0, 1)
                best_res = r
        except Exception:
            continue
    if best is not None and best_res < 1e-6:
        return best
    return None


def jacobian_neg3(fp, w_AD, w_DI, w_IA, w_XA, siegert, alpha, beta, tau_m):
    """3x3 Jacobian eigenvalues of the rate-equation linearization at fp."""
    mu, sigma = neg3_inputs(fp, w_AD, w_DI, w_IA, w_XA, alpha, beta)
    g = np.array([dphi_dmu(siegert, float(mu[i]), float(sigma[i]))
                  for i in range(3)])
    if not np.all(np.isfinite(g)):
        return np.array([np.nan + 0j] * 3)
    # W (Siegert units) - same structure as the FCS W but with alpha scaling.
    W = alpha * np.array([
        [0,    0,    w_IA],
        [w_AD, 0,    0   ],
        [0,    w_DI, 0   ],
    ], dtype=float)
    eigs = jacobian_eigenvalues(W, g, tau_m)
    return eigs


def simulate_qr_neg3(w_AD, w_DI, w_IA, w_XA, qr, alpha, beta, N, T,
                     seed=2026, init_A=(0.4, 0.4, 0.2)):
    """Quasi-renewal 3-pop mesoscopic simulation of the negative loop with delayer."""
    rng = np.random.default_rng(seed)
    m = np.zeros((3, qr.K_max))
    for i, A0 in enumerate(init_A):
        A0s = max(min(A0, 0.99), 1e-3)
        for k in range(qr.K_max):
            m[i, k] = A0s * (1 - A0s) ** k
        m[i] /= m[i].sum()
    A_prev = np.array(init_A, dtype=float)
    rates = np.zeros((3, T))

    for t in range(T):
        mean_A = w_XA * P_THIN + w_IA * A_prev[2]
        var_A = ((w_XA ** 2) * P_THIN * (1 - P_THIN)
                 + (w_IA ** 2) * A_prev[2] * (1 - A_prev[2]))
        mean_D = w_AD * A_prev[0]
        var_D = (w_AD ** 2) * A_prev[0] * (1 - A_prev[0])
        mean_I = w_DI * A_prev[1]
        var_I = (w_DI ** 2) * A_prev[1] * (1 - A_prev[1])

        mu = np.array([alpha * mean_A, alpha * mean_D, alpha * mean_I])
        sigma = np.array([
            math.sqrt(max(beta * var_A, 0.0)),
            math.sqrt(max(beta * var_D, 0.0)),
            math.sqrt(max(beta * var_I, 0.0)),
        ])
        m, A = qr.step(m, mu, sigma, N, rng=rng)
        rates[:, t] = A
        A_prev = A

    return rates


def main():
    calib = load_calibration()
    print("Calibration (locked):", calib)
    siegert = Siegert(V_th=1.0, V_r=0.0,
                      tau_m=calib["tau_m"], tau_ref=calib["tau_ref"])
    qr = QuasiRenewal(siegert=siegert, K_max=30, tau_ref_ticks=0, dt=1.0)
    print()

    # ---------------- FCS oracle at default ----------------
    T_FCS = 64
    W, B, ext = topology_neg_loop_3(
        W_XA_DEFAULT, W_AD_DEFAULT, W_DI_DEFAULT, W_IA_DEFAULT, T_FCS)
    spikes, _ = simulate(W, B, ext, T=T_FCS)
    sA = spikes[0]
    sD = spikes[1]
    sI = spikes[2]
    print(f"FCS oracle, 3-neuron negative loop, default weights "
          f"(w_XA={W_XA_DEFAULT}, w_AD={W_AD_DEFAULT}, "
          f"w_DI={W_DI_DEFAULT}, w_IA={W_IA_DEFAULT}):")
    print(f"  A: {spike_sequence_to_str(sA[:32])}...")
    print(f"  D: {spike_sequence_to_str(sD[:32])}...")
    print(f"  I: {spike_sequence_to_str(sI[:32])}...")
    period_fcs = detect_period(sA[16:], max_period=14)
    print(f"  measured period of A (post-warmup): {period_fcs}")

    # ---------------- Sweep w_IA to find period-stable region ----------------
    w_IA_sweep = np.arange(-30, 0)
    fcs_periods = []
    for w_IA in w_IA_sweep:
        W, B, ext = topology_neg_loop_3(
            W_XA_DEFAULT, W_AD_DEFAULT, W_DI_DEFAULT, int(w_IA), T_FCS)
        spikes_s, _ = simulate(W, B, ext, T=T_FCS)
        fcs_periods.append(detect_period(spikes_s[0, 16:], max_period=14))
    fcs_periods = np.array(fcs_periods)
    print(f"\nw_IA sweep on 3-neuron motif (w_AD=w_DI=11, w_XA=11):")
    for w_IA, p in zip(w_IA_sweep, fcs_periods):
        print(f"  w_IA = {int(w_IA):3d}: FCS period = {int(p)}")

    # ---------------- Siegert FP + Jacobian at default ----------------
    fp = find_fp_neg3(
        W_AD_DEFAULT, W_DI_DEFAULT, W_IA_DEFAULT, W_XA_DEFAULT,
        siegert, calib["alpha"], calib["beta"],
    )
    if fp is None:
        print("\nSiegert FP did not converge!")
        eigs_default = np.array([np.nan + 0j] * 3)
        T_pred_static = float("nan")
    else:
        print(f"\nSiegert FP at default: nu = ({fp[0]:.4f}, "
              f"{fp[1]:.4f}, {fp[2]:.4f})")
        eigs_default = jacobian_neg3(
            fp, W_AD_DEFAULT, W_DI_DEFAULT, W_IA_DEFAULT, W_XA_DEFAULT,
            siegert, calib["alpha"], calib["beta"], calib["tau_m"],
        )
        print(f"  Jacobian eigenvalues: ")
        for i, lam in enumerate(eigs_default):
            print(f"    lambda_{i} = {lam.real:+.4f} + {lam.imag:+.4f}i")
        im_abs = np.abs(np.imag(eigs_default))
        dominant_im = float(im_abs.max())
        if dominant_im > 1e-6:
            T_pred_static = 2 * math.pi / dominant_im
        else:
            T_pred_static = float("inf")
        print(f"  dominant |Im(λ)| = {dominant_im:.4f}, "
              f"T_pred (static tau_m) = {T_pred_static:.2f} ticks")

    # ---------------- T_pred recalibrated with tau_dyn from Experiment A -----
    expA_npz = ROOT / "deq" / "closed_form_neg_loop" / "followups" / \
               "results" / "expA" / "bode.npz"
    T_pred_recal = float("nan")
    tau_dyn = None
    if expA_npz.exists():
        dA = np.load(expA_npz, allow_pickle=True)
        tau_dyn = float(dA["tau_fit"])
        tau_static_calib = float(dA["tau_static"])
        if not math.isnan(T_pred_static) and not math.isinf(T_pred_static):
            T_pred_recal = T_pred_static * (tau_dyn / tau_static_calib)
        print(f"\nFrom Experiment A: tau_fit = {tau_dyn:.3f}, "
              f"tau_static = {tau_static_calib:.3f}")
        print(f"  T_pred recalibrated = {T_pred_recal:.2f} ticks")
    else:
        print("\n(Experiment A bode.npz not found; skipping recalibration)")

    # ---------------- QR mesoscopic at N=500 ----------------
    T_QR = 400
    T_WARMUP_QR = 100
    N = 500
    rates_qr = simulate_qr_neg3(
        W_AD_DEFAULT, W_DI_DEFAULT, W_IA_DEFAULT, W_XA_DEFAULT,
        qr, calib["alpha"], calib["beta"], N=N, T=T_QR,
    )
    qr_period_A = fft_period(rates_qr[0, T_WARMUP_QR:])
    print(f"\nQR mesoscopic at N={N}: A's FFT period = {qr_period_A:.2f} ticks")

    # ---------------- Save ----------------
    np.savez(
        RESULTS / "three_neuron.npz",
        fcs_spikes=spikes,
        period_fcs=period_fcs,
        w_IA_sweep=w_IA_sweep,
        fcs_periods=fcs_periods,
        fp=fp if fp is not None else np.full(3, np.nan),
        eigs=eigs_default,
        T_pred_static=T_pred_static,
        T_pred_recal=T_pred_recal,
        tau_dyn=tau_dyn if tau_dyn is not None else float("nan"),
        rates_qr=rates_qr,
        qr_period_A=qr_period_A,
        N=N,
    )

    # ---------------- Plot ----------------
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.3)

    # Top-left: FCS spike trains for A/D/I.
    ax = fig.add_subplot(gs[0, 0])
    for i, (s, label, color) in enumerate([
        (sA[:32], "A", "tab:blue"),
        (sD[:32], "D (delayer)", "tab:green"),
        (sI[:32], "I", "tab:orange"),
    ]):
        ys = i - 0.4 * s.astype(float)
        ax.step(np.arange(32), ys, where="post", color=color, lw=1.5,
                label=label)
    for x in range(0, 32, 4):
        ax.axvline(x, color="gray", linestyle=":", alpha=0.2)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["A", "D", "I"])
    ax.set_xlabel("tick")
    ax.set_title(
        f"FCS oracle (3-neuron, default weights): period of A = {period_fcs}",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # Top-right: w_IA sweep period.
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(w_IA_sweep, fcs_periods, "o-", color="tab:blue")
    ax.axvline(W_IA_DEFAULT, color="tab:orange", linestyle="--", alpha=0.7,
               label=f"default w_IA = {W_IA_DEFAULT}")
    ax.axhline(4, color="gray", linestyle=":", alpha=0.5, label="2-neuron period = 4")
    ax.set_xlabel(r"$w_{IA}$ (FCS scaled)")
    ax.set_ylabel("FCS-measured period of A")
    ax.set_title(
        "Period vs w_IA sweep (3-neuron negative loop, w_AD=w_DI=11)",
        fontsize=10,
    )
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Middle-left: eigenvalue scatter.
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(np.real(eigs_default), np.imag(eigs_default),
               c=["tab:red", "tab:green", "tab:blue"],
               s=70, zorder=3)
    for lam, name in zip(eigs_default, ["λ0", "λ1", "λ2"]):
        ax.annotate(
            f"{name}\n({lam.real:+.2f}, {lam.imag:+.2f})",
            xy=(lam.real, lam.imag),
            xytext=(8, 8), textcoords="offset points", fontsize=9,
        )
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_xlabel(r"Re($\lambda$)"); ax.set_ylabel(r"Im($\lambda$)")
    ax.set_title("Jacobian eigenvalues at FP (3-neuron)", fontsize=10)
    ax.grid(alpha=0.3)

    # Middle-right: QR rates trace.
    ax = fig.add_subplot(gs[1, 1])
    show = slice(T_WARMUP_QR, T_WARMUP_QR + 80)
    ax.plot(rates_qr[0, show], color="tab:blue", lw=1.5, label="A")
    ax.plot(rates_qr[1, show], color="tab:green", lw=1.0, alpha=0.7, label="D")
    ax.plot(rates_qr[2, show], color="tab:orange", lw=1.0, alpha=0.7, label="I")
    for x in range(0, 80, 4):
        ax.axvline(x, color="gray", linestyle=":", alpha=0.2)
    ax.set_xlabel("tick (post-warmup)"); ax.set_ylabel("QR rate")
    ax.set_title(
        f"QR mesoscopic (N={N}): A FFT period = {qr_period_A:.2f}",
        fontsize=10,
    )
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Bottom: period prediction bar chart, 2-neuron vs 3-neuron.
    ax = fig.add_subplot(gs[2, :])
    # 2-neuron numbers (from main study).
    two_n = dict(fcs=4, sieg_im=15.92, hpred=15.92, qr=4.05)
    # 3-neuron numbers (this experiment).
    three_n = dict(fcs=period_fcs,
                   hpred=T_pred_static if math.isfinite(T_pred_static) else 0,
                   hpred_recal=T_pred_recal if math.isfinite(T_pred_recal) else 0,
                   qr=qr_period_A)
    x = np.arange(4)
    width = 0.32
    ax.bar(x - width/2, [two_n["fcs"], two_n["sieg_im"], two_n["hpred"],
                          two_n["qr"]],
           width, label="2-neuron (main study)", color="tab:blue")
    ax.bar(x + width/2,
           [three_n["fcs"], three_n["hpred"], three_n["hpred_recal"],
            three_n["qr"]],
           width, label="3-neuron (this experiment)", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels([
        "FCS-measured\nperiod",
        "T_pred static\n(H(ω))",
        ("T_pred recalibrated\n(τ_dyn from Exp A)"
         if tau_dyn is not None else "T_pred static\n(H(ω))"),
        f"QR period\n(N=500/2000)",
    ])
    ax.set_ylabel("period (ticks)")
    ax.set_title("Period predictions: 2-neuron vs 3-neuron negative loop",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, max(20, max(two_n.values()) * 1.1))

    fig.suptitle(
        "Experiment C: 3-neuron negative loop (A → D → I → A) closed-form lenses",
        fontsize=12,
    )
    out_pdf = RESULTS / "three_neuron.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\n  wrote {out_pdf}")
    plt.close(fig)

    print()
    print("Experiment C summary:")
    print(f"  FCS period (2-neuron):       4")
    print(f"  FCS period (3-neuron):       {period_fcs}")
    print(f"  H(ω) static T_pred (2-neuron): 15.92")
    print(f"  H(ω) static T_pred (3-neuron): {T_pred_static:.2f}")
    if not math.isnan(T_pred_recal):
        print(f"  H(ω) recal T_pred (3-neuron):  {T_pred_recal:.2f}")
    print(f"  QR period (2-neuron, N=2000):  4.05")
    print(f"  QR period (3-neuron, N={N}):    {qr_period_A:.2f}")

    if period_fcs >= 5 and qr_period_A is not None \
            and abs(qr_period_A - period_fcs) <= 1.5:
        print(f"  Experiment C PASS: 3-neuron motif gives longer period "
              f"({period_fcs}); QR ({qr_period_A:.2f}) tracks FCS within ±1.5.")
    else:
        print(f"  Experiment C PARTIAL: period {period_fcs}, "
              f"QR = {qr_period_A:.2f} -- check details in the report.")


if __name__ == "__main__":
    main()
