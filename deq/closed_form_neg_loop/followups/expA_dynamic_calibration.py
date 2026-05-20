"""Experiment A: dynamic-tau calibration from sinusoidal sweep.

The parent study's Phase 2 found that the locked static calibration
(tau_m = 2.35, fit on the steady-state f-I curve) over-estimates the
H(omega) ringing period by a constant factor of ~4 at the FCS default
cell. Hypothesis: tau_m is the wrong dynamic time constant; refitting
it against an actual dynamic response should close the gap.

Method:
  1. Single isolated FCS neuron, driven by Bernoulli input
     x(t) = Bern(p_thin * (1 + eps * sin(omega * t))),
     p_thin = 0.7, eps = 0.2.
  2. Sweep omega over ~12 frequencies covering [0.05, 1.5] rad/tick.
  3. Per frequency: simulate T=4000 ticks, smooth spike train into
     instantaneous rate r(t), FFT at omega -> |R(omega)|, phase.
  4. Empirical Bode |H(omega)| = |R(omega)| / (DC_rate * eps).
  5. Fit single-pole model |H(omega)|^2 = g^2 / (1 + omega^2 * tau^2)
     via curve_fit -> tau_dyn.
  6. Recalibrate Phase 2: T_pred_new = T_pred_old * (tau_dyn / tau_m_static).
     (Eigenvalues of the single-pole Jacobian scale as 1/tau.)

Outputs:
  results/expA/bode.npz, bode_fit.pdf, recal_T_pred.pdf
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from deq.archetypes.lif_fcs import simulate  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "expA"
RESULTS.mkdir(parents=True, exist_ok=True)


P_THIN = 0.7           # calibration regime
EPS = 0.20             # AC amplitude as fraction of p_thin
W_DRIVE = 11           # external-input weight (FCS scaled units)
T_SIM = 4000           # simulation length per frequency
T_WARMUP = 200         # discard initial transient
RATE_SMOOTH = 11       # window for moving-average of spikes -> rate
SEED = 2026
N_TRIALS = 4           # ensemble average to suppress single-realization noise


def load_static_calibration() -> dict:
    src = ROOT / "deq" / "closed_form" / "results" / "phase1_grid.npz"
    d = np.load(src, allow_pickle=True)
    return dict(
        alpha=float(d["calib_alpha"]),
        beta=float(d["calib_beta"]),
        tau_m=float(d["calib_tau_m"]),
        tau_ref=float(d["calib_tau_ref"]),
    )


def driven_spike_train(omega: float, T: int, rng) -> np.ndarray:
    """Run one isolated FCS neuron with Bern(p_thin*(1+eps*sin(omega t))) input.

    Returns spike train (length T, bool).
    """
    t = np.arange(T)
    p = P_THIN * (1.0 + EPS * np.sin(omega * t))
    p = np.clip(p, 0.0, 1.0)
    x = (rng.uniform(size=T) < p).astype(np.int64)

    # Single-neuron FCS network: W = [[0]], B = [[W_DRIVE]], external = x[None, :]
    W = np.array([[0]], dtype=np.int64)
    B = np.array([[W_DRIVE]], dtype=np.int64)
    ext = x.reshape(1, T)
    spikes, _ = simulate(W, B, ext, T=T)
    return spikes[0]


def measure_bode_point(omega: float, n_trials: int = N_TRIALS,
                       rng_base_seed: int = SEED) -> dict:
    """Average over n_trials of |R(omega)| and phase relative to the input."""
    amps = []
    phases = []
    rates_dc = []

    for trial in range(n_trials):
        rng = np.random.default_rng(rng_base_seed + trial * 13 + int(omega * 1e3))
        spikes = driven_spike_train(omega, T_SIM, rng)
        # Smooth spike train into rate.
        kernel = np.ones(RATE_SMOOTH) / RATE_SMOOTH
        r = np.convolve(spikes.astype(float), kernel, mode="same")
        r_post = r[T_WARMUP:T_SIM - T_WARMUP]
        n_post = len(r_post)
        dc_rate = float(np.mean(r_post))

        # Reference sinusoid at the same omega (use cos and sin to get phase).
        t = np.arange(n_post) + T_WARMUP
        cos_ref = np.cos(omega * t)
        sin_ref = np.sin(omega * t)
        r_ac = r_post - dc_rate
        # Coefficients from least-squares of r_ac = A cos + B sin (amplitude
        # = sqrt(A^2 + B^2)).
        amp_cos = 2 * float(np.mean(r_ac * cos_ref))
        amp_sin = 2 * float(np.mean(r_ac * sin_ref))
        amp = math.sqrt(amp_cos ** 2 + amp_sin ** 2)
        phase = math.atan2(amp_cos, amp_sin)  # phase relative to driving sin(omega t)

        amps.append(amp)
        phases.append(phase)
        rates_dc.append(dc_rate)

    return dict(
        omega=omega,
        amp_mean=float(np.mean(amps)),
        amp_std=float(np.std(amps)),
        phase_mean=float(np.mean(phases)),
        dc_rate=float(np.mean(rates_dc)),
    )


def single_pole_mag_sq(omega, g, tau):
    return (g ** 2) / (1.0 + (omega * tau) ** 2)


def main():
    calib = load_static_calibration()
    print("Static calibration (from closed_form_wta phase1):")
    for k, v in calib.items():
        print(f"  {k} = {v:.6f}")
    tau_static = calib["tau_m"]
    print()

    # Frequency sweep.
    omegas = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.55, 0.70,
                       0.85, 1.00, 1.20, 1.40, 1.60])
    print(f"Bode sweep on isolated FCS neuron, "
          f"{len(omegas)} frequencies, T_sim={T_SIM}, n_trials={N_TRIALS}")
    print(f"  AC input: Bern(p_thin*(1 + eps*sin(omega t))), "
          f"p_thin={P_THIN}, eps={EPS}\n")

    rows = []
    for omega in omegas:
        pt = measure_bode_point(float(omega))
        # Empirical |H(omega)| relative to the input modulation amplitude.
        # Input modulation in rate units: input is Bernoulli with mean p_thin
        # modulated by eps; modulation amplitude in input-rate space is
        # eps * p_thin. The DC firing rate is pt["dc_rate"], so |H| should be
        # interpreted as ratio of output-amplitude to input-modulation-amplitude.
        H_mag = pt["amp_mean"] / (EPS * P_THIN)
        rows.append((float(omega), H_mag, pt["amp_mean"], pt["phase_mean"],
                     pt["dc_rate"], pt["amp_std"]))
        print(f"  omega = {omega:.3f}: |R| = {pt['amp_mean']:.4f} +/- "
              f"{pt['amp_std']:.4f}, DC_rate = {pt['dc_rate']:.3f}, "
              f"|H| = {H_mag:.4f}, phase = {math.degrees(pt['phase_mean']):+.1f} deg")

    omegas_arr = np.array([r[0] for r in rows])
    H_mag_arr = np.array([r[1] for r in rows])
    R_amp_arr = np.array([r[2] for r in rows])
    phases_arr = np.array([r[3] for r in rows])
    dc_arr = np.array([r[4] for r in rows])

    # Fit single-pole model.
    # Use |H|^2 against omega for a clean least-squares.
    popt, pcov = curve_fit(
        single_pole_mag_sq, omegas_arr, H_mag_arr ** 2,
        p0=[float(H_mag_arr[0]), tau_static],
        bounds=([0.0, 0.0], [np.inf, np.inf]),
        maxfev=20000,
    )
    g_fit, tau_fit = popt
    fit_vals = single_pole_mag_sq(omegas_arr, g_fit, tau_fit)
    ss_res = float(np.sum((H_mag_arr ** 2 - fit_vals) ** 2))
    ss_tot = float(np.sum((H_mag_arr ** 2 - np.mean(H_mag_arr ** 2)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    print()
    print(f"Single-pole fit |H(omega)|^2 = g^2 / (1 + omega^2 tau^2):")
    print(f"  g_fit   = {g_fit:.4f}")
    print(f"  tau_fit = {tau_fit:.4f} ticks    (static tau_m = {tau_static:.4f})")
    print(f"  R^2     = {r2:.4f}")
    print(f"  ratio   tau_fit / tau_static = {tau_fit / tau_static:.4f}")
    print()

    # Recalibrate Phase 2 T_pred via T_pred_new = T_pred_old * (tau_fit / tau_static).
    p1 = np.load(ROOT / "deq" / "closed_form_neg_loop" /
                 "results" / "phase1" / "siegert_grid.npz",
                 allow_pickle=True)
    p2 = np.load(ROOT / "deq" / "closed_form_neg_loop" /
                 "results" / "phase2" / "h_grid.npz",
                 allow_pickle=True)
    p0 = np.load(ROOT / "deq" / "closed_form_neg_loop" /
                 "results" / "phase0" / "fcs_grid.npz",
                 allow_pickle=True)

    W_IA = p0["W_IA_VALUES"]
    W_XA = p0["W_XA_VALUES"]
    fcs_period = p0["period"]
    strict_p5 = p0["strict_p5"].astype(bool)
    T_pred_static = p2["T_pred"]
    T_pred_recal = T_pred_static * (tau_fit / tau_static)

    # Default cell.
    i_def = int(np.where(W_IA == -11)[0][0])
    j_def = int(np.where(W_XA == 11)[0][0])
    T_def_static = float(T_pred_static[i_def, j_def])
    T_def_recal = float(T_pred_recal[i_def, j_def])
    print(f"FCS default cell (w_IA=-11, w_XA=11):")
    print(f"  Phase 2 static    T_pred = {T_def_static:.2f} ticks (FCS = 4)")
    print(f"  Recalibrated     T_pred = {T_def_recal:.2f} ticks "
          f"(ratio {T_def_recal/4:.2f})")

    # Mean ratio over FCS-period-4 cells.
    fcs_p4 = (fcs_period == 4) & np.isfinite(T_pred_recal)
    if fcs_p4.any():
        ratio_recal = float(np.mean(T_pred_recal[fcs_p4] / 4.0))
        ratio_static = float(np.mean(T_pred_static[fcs_p4] / 4.0))
        print(f"  Over FCS-period-4 cells ({int(fcs_p4.sum())}):")
        print(f"    static  mean T_pred / 4 = {ratio_static:.2f}")
        print(f"    recal   mean T_pred / 4 = {ratio_recal:.2f}")

    np.savez(
        RESULTS / "bode.npz",
        omegas=omegas_arr, H_mag=H_mag_arr, R_amp=R_amp_arr,
        phases=phases_arr, dc_rates=dc_arr,
        g_fit=g_fit, tau_fit=tau_fit, r2=r2, tau_static=tau_static,
        T_pred_recal=T_pred_recal,
        W_IA=W_IA, W_XA=W_XA,
    )

    # -------- Bode plot --------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    omega_dense = np.linspace(omegas_arr.min() * 0.5,
                              omegas_arr.max() * 1.2, 200)
    H_fit = np.sqrt(single_pole_mag_sq(omega_dense, g_fit, tau_fit))
    H_static = np.sqrt(single_pole_mag_sq(omega_dense, g_fit, tau_static))

    ax = axes[0]
    ax.loglog(omegas_arr, H_mag_arr, "o", color="tab:blue",
              markersize=7, label="FCS Bode data")
    ax.loglog(omega_dense, H_fit, "-", color="tab:purple",
              label=f"single-pole fit, tau_fit = {tau_fit:.2f}")
    ax.loglog(omega_dense, H_static, "--", color="tab:gray",
              alpha=0.7, label=f"static tau_m = {tau_static:.2f}")
    ax.set_xlabel(r"$\omega$ (rad/tick)")
    ax.set_ylabel(r"$|H(\omega)|$")
    ax.set_title(f"Magnitude (R^2 = {r2:.3f})", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(omegas_arr, np.degrees(phases_arr), "s",
            color="tab:blue", markersize=7, label="FCS phase")
    # Single-pole phase: phi = -atan(omega tau)
    phi_fit = -np.degrees(np.arctan(omega_dense * tau_fit))
    phi_static = -np.degrees(np.arctan(omega_dense * tau_static))
    ax.plot(omega_dense, phi_fit, "-", color="tab:purple",
            label=f"tau_fit phase")
    ax.plot(omega_dense, phi_static, "--", color="tab:gray",
            alpha=0.7, label=f"static phase")
    ax.set_xlabel(r"$\omega$ (rad/tick)")
    ax.set_ylabel(r"phase $\angle H(\omega)$ (degrees)")
    ax.set_title("Phase", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle(
        "Experiment A: Bode response of isolated FCS neuron, single-pole fit",
        fontsize=11,
    )
    plt.tight_layout()
    out_pdf = RESULTS / "bode_fit.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\n  wrote {out_pdf}")
    plt.close(fig)

    # -------- Recalibrated T_pred figures --------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Heatmap of recalibrated T_pred clipped at 8.
    ax = axes[0]
    T_clip = np.where(np.isfinite(T_pred_recal),
                      np.clip(T_pred_recal, 0, 8), np.nan)
    T_masked = np.ma.masked_invalid(T_clip)
    im = ax.imshow(
        T_masked, origin="lower", cmap="magma",
        vmin=0, vmax=8,
        extent=[W_XA[0] - 0.5, W_XA[-1] + 0.5,
                W_IA[0] - 0.5, W_IA[-1] + 0.5],
        aspect="auto",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$T_{\rm pred}^{\rm recal}$ (ticks)")
    ax.scatter([11], [-11], facecolors="none", edgecolors="white",
               s=160, linewidths=2.0, zorder=5)
    ax.set_xlabel(r"$w_{XA}$"); ax.set_ylabel(r"$w_{IA}$")
    ax.set_title(
        f"Recalibrated $T_{{\\rm pred}}$ heatmap (tau_fit / tau_static = "
        f"{tau_fit/tau_static:.2f})",
        fontsize=10,
    )

    # Scatter: T_pred_recal vs FCS period.
    ax = axes[1]
    valid = np.isfinite(T_pred_recal) & (fcs_period > 0)
    xs = fcs_period[valid]
    ys = T_pred_recal[valid]
    ys_static = T_pred_static[valid]
    is_p5 = strict_p5[valid]
    rng = np.random.default_rng(0)
    xj = xs + rng.uniform(-0.15, 0.15, size=len(xs))

    ax.scatter(xj[~is_p5], ys[~is_p5], c="lightgray", s=8, alpha=0.5,
               label="non-P5 recal")
    ax.scatter(xj[is_p5], ys[is_p5], c="tab:purple", s=14, alpha=0.85,
               label="strict P5 recal")
    # And the original static values for comparison (dimmed).
    ax.scatter(xj[is_p5], ys_static[is_p5], c="tab:orange", s=10,
               alpha=0.5, marker="x", label="strict P5 static (Phase 2)")
    ax.plot([0, 13], [0, 13], "k--", alpha=0.4, label="y = x")
    ax.set_xlim(0.5, 12.5); ax.set_ylim(0, 30)
    ax.set_xlabel("FCS-measured period of A (ticks)")
    ax.set_ylabel(r"$T_{\rm pred}$ (ticks)")
    ax.set_title("Recalibrated vs static T_pred", fontsize=10)
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Experiment A: recalibrated Phase 2 with tau_dyn", fontsize=11)
    plt.tight_layout()
    out_pdf = RESULTS / "recal_T_pred.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    plt.close(fig)

    print()
    if abs(T_def_recal - 4) <= 0.5:
        print(f"Experiment A PASS: recalibrated default-cell T_pred = "
              f"{T_def_recal:.2f} matches FCS period 4 within +/- 0.5.")
    else:
        print(f"Experiment A PARTIAL: recalibrated default-cell T_pred = "
              f"{T_def_recal:.2f}, off by {abs(T_def_recal - 4):.2f} "
              f"from FCS period 4. tau_fit = {tau_fit:.3f} vs the "
              f"factor-of-4 implied tau_static/4 = {tau_static/4:.3f}.")


if __name__ == "__main__":
    main()
