"""Experiment B: single-neuron renewal PMF predictor for the negative loop.

The parent study's Phase 3 used quasi-renewal at finite N to track the
mesoscopic population activity A(t). For the negative loop the
"population" is exactly one neuron, so the natural single-neuron
analog drops the sqrt(A/N) noise term and tracks a deterministic
*PMF over age* p_k(t) = P(last spike was k ticks ago).

Update rule (single neuron, hazard h_k given input):
  fire_prob(t) = sum_k p_k(t-1) * h_k
  p_0(t)       = fire_prob(t)
  p_{k+1}(t)   = p_k(t-1) * (1 - h_k)        for k >= 0
  renormalize.

For the negative loop, A's PMF p_A and I's PMF p_I are coupled
through expected firing probabilities (used to drive the Siegert
hazard input for the other neuron).

Compare the resulting fire_prob_A(t) trace against FCS's binary
spike train at the default cell. Discretize via threshold > 0.5 and
score against the 1100 template.

Output: results/expB/pmf_trace.npz, pmf_vs_fcs.pdf
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from deq.archetypes.lif_fcs import simulate, spike_sequence_to_str  # noqa: E402
from deq.archetypes.topologies import negative_loop  # noqa: E402
from deq.closed_form.siegert import Siegert  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "expB"
RESULTS.mkdir(parents=True, exist_ok=True)


P_THIN = 0.7
K_MAX = 30
TAU_REF_TICKS = 0  # match closed_form_wta phase 3 setting


def load_calibration() -> dict:
    src = ROOT / "deq" / "closed_form" / "results" / "phase1_grid.npz"
    d = np.load(src, allow_pickle=True)
    return dict(
        alpha=float(d["calib_alpha"]),
        beta=float(d["calib_beta"]),
        tau_m=float(d["calib_tau_m"]),
        tau_ref=float(d["calib_tau_ref"]),
    )


class RenewalNeuronPMF:
    """Track a deterministic age-PMF for a single LI&F neuron.

    State is p_k(t), k = 0..K_max-1. p_k = probability that the most
    recent spike was k ticks ago. At each tick, given input (mu, sigma):
      hazard h_k = Phi(mu, sigma)   for k >= tau_ref, else 0
      fire   = sum_k p_k(t-1) h_k
      p_0(t) = fire
      p_{k+1}(t) = p_k(t-1) (1 - h_k)
    """

    def __init__(self, siegert: Siegert, K_max: int = K_MAX,
                 tau_ref_ticks: int = TAU_REF_TICKS):
        self.siegert = siegert
        self.K_max = K_max
        self.tau_ref_ticks = tau_ref_ticks

    def init_state(self, init_age_mean: float = 5.0) -> np.ndarray:
        """Geometric initialization centered on init_age_mean."""
        p0 = max(min(1.0 / init_age_mean, 0.99), 1e-3)
        p = np.array([p0 * (1 - p0) ** k for k in range(self.K_max)])
        p /= p.sum()
        return p

    def step(self, p: np.ndarray, mu: float, sigma: float) -> tuple:
        """Return (p_new, fire_prob)."""
        h_const = float(self.siegert.phi(mu, sigma))
        h_const = min(max(h_const, 0.0), 1.0)
        h = np.zeros(self.K_max)
        for k in range(self.K_max):
            if k >= self.tau_ref_ticks:
                h[k] = h_const
        fire = float((p * h).sum())
        p_new = np.zeros_like(p)
        p_new[0] = fire
        for k in range(1, self.K_max):
            p_new[k] = p[k - 1] * (1.0 - h[k - 1])
        s = p_new.sum()
        if s > 1e-12:
            p_new /= s
        return p_new, fire


def simulate_neg_loop_pmf(w_AI: float, w_IA: float, w_XA: float,
                          renewal: RenewalNeuronPMF,
                          alpha: float, beta: float, T: int,
                          init_age=(5.0, 5.0)) -> dict:
    """Run a 2-neuron negative-loop renewal-PMF simulation.

    A receives external X (Bernoulli, mean p_thin) + inhibition from I.
    I receives excitation from A only.

    The "input rate" for each neuron's Siegert call uses the OTHER
    neuron's fire_prob from the previous tick.
    """
    p_A = renewal.init_state(init_age[0])
    p_I = renewal.init_state(init_age[1])
    fire_A_prev = float(p_A[0])
    fire_I_prev = float(p_I[0])
    fire_A = np.zeros(T)
    fire_I = np.zeros(T)
    pmf_A = np.zeros((T, renewal.K_max))
    pmf_I = np.zeros((T, renewal.K_max))

    for t in range(T):
        # A's input: external w_XA * Bern(p_thin) + w_IA * fire_I
        mean_A = w_XA * P_THIN + w_IA * fire_I_prev
        var_A = ((w_XA ** 2) * P_THIN * (1 - P_THIN)
                 + (w_IA ** 2) * fire_I_prev * (1 - fire_I_prev))
        mu_A = alpha * mean_A
        sigma_A = math.sqrt(max(beta * var_A, 0.0))

        # I's input: w_AI * fire_A
        mean_I = w_AI * fire_A_prev
        var_I = (w_AI ** 2) * fire_A_prev * (1 - fire_A_prev)
        mu_I = alpha * mean_I
        sigma_I = math.sqrt(max(beta * var_I, 0.0))

        p_A, fA = renewal.step(p_A, mu_A, sigma_A)
        p_I, fI = renewal.step(p_I, mu_I, sigma_I)
        fire_A[t] = fA
        fire_I[t] = fI
        pmf_A[t] = p_A
        pmf_I[t] = p_I
        fire_A_prev = fA
        fire_I_prev = fI

    return dict(
        fire_A=fire_A, fire_I=fire_I,
        pmf_A=pmf_A, pmf_I=pmf_I,
    )


def cyclic_template_match(binary_train: np.ndarray, pattern="1100") -> dict:
    """Return best phase shift and correlation against the cyclic template."""
    n = len(binary_train)
    L = len(pattern)
    if n < 2 * L:
        return dict(best_shift=-1, best_corr=0.0, best_match=0)
    s = binary_train.astype(int)
    best_corr = -1.0
    best_shift = -1
    best_match = 0
    for shift in range(L):
        pat = np.array([int(c) for c in pattern])
        pat = np.roll(pat, shift)
        tmpl = np.tile(pat, n // L + 2)[:n]
        # Pearson correlation.
        ss = s - s.mean()
        tt = tmpl - tmpl.mean()
        denom = np.std(s) * np.std(tmpl)
        if denom < 1e-9:
            continue
        c = float(np.mean(ss * tt) / denom)
        match = int(np.sum(s == tmpl))
        if c > best_corr:
            best_corr = c
            best_shift = shift
            best_match = match
    return dict(best_shift=best_shift, best_corr=best_corr,
                best_match=best_match, n_compared=n)


def detect_period(seq: np.ndarray, max_period: int = 12) -> int:
    """Smallest period in [1, max_period] for which seq is eventually periodic."""
    n = len(seq)
    if n < 2 * max_period:
        return 0
    s = seq.astype(int)
    for p in range(1, max_period + 1):
        tail = s[-2 * p:]
        if np.array_equal(tail[:p], tail[p:]):
            return p
    return 0


def fft_period(trace: np.ndarray, min_period=2.0, max_period=24.0) -> float:
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


def main():
    calib = load_calibration()
    print("Calibration (locked):", calib)
    print()

    siegert = Siegert(V_th=1.0, V_r=0.0,
                      tau_m=calib["tau_m"], tau_ref=calib["tau_ref"])
    renewal = RenewalNeuronPMF(siegert=siegert, K_max=K_MAX, tau_ref_ticks=0)

    # FCS default cell.
    w_XA, w_AI, w_IA = 11, 11, -11
    T_PMF = 80
    print(f"Renewal-PMF run at FCS default cell "
          f"(w_XA={w_XA}, w_AI={w_AI}, w_IA={w_IA}), T={T_PMF}\n")

    result = simulate_neg_loop_pmf(
        w_AI=w_AI, w_IA=w_IA, w_XA=w_XA,
        renewal=renewal,
        alpha=calib["alpha"], beta=calib["beta"], T=T_PMF,
    )
    fire_A = result["fire_A"]
    fire_I = result["fire_I"]

    # FCS reference (longer to allow warmup).
    W, B, ext = negative_loop(w_XA=w_XA, w_AI=w_AI, w_IA=w_IA, T=T_PMF + 20)
    spikes, _ = simulate(W, B, ext, T=T_PMF + 20)
    fcs_A = spikes[0, 20:20 + T_PMF].astype(int)
    fcs_I = spikes[1, 20:20 + T_PMF].astype(int)
    print(f"FCS A: {spike_sequence_to_str(fcs_A[:24])} ...")
    print(f"FCS I: {spike_sequence_to_str(fcs_I[:24])} ...")
    print()

    # Discretize PMF trace via threshold > 0.5.
    warmup = 16
    binary_pred_A = (fire_A[warmup:] > 0.5).astype(int)
    fcs_post_A = fcs_A[warmup:]

    pmf_match = cyclic_template_match(binary_pred_A, pattern="1100")
    fcs_match = cyclic_template_match(fcs_post_A, pattern="1100")
    pmf_period = detect_period(binary_pred_A)
    fcs_period_obs = detect_period(fcs_post_A)
    pmf_fft = fft_period(fire_A[warmup:])

    print(f"After warmup (t >= {warmup}, length {len(binary_pred_A)}):")
    print(f"  FCS A:               period={fcs_period_obs}, "
          f"template match='1100' shift={fcs_match['best_shift']}, "
          f"corr={fcs_match['best_corr']:.3f}")
    print(f"  PMF binary thresh:   period={pmf_period}, "
          f"template match shift={pmf_match['best_shift']}, "
          f"corr={pmf_match['best_corr']:.3f}, "
          f"FFT period of fire_A trace={pmf_fft:.2f}")
    print(f"  PMF fire_A mean = {fire_A[warmup:].mean():.3f}, "
          f"std = {fire_A[warmup:].std():.3f}, "
          f"max = {fire_A.max():.3f}, min = {fire_A.min():.3f}")
    print()
    # Direct binary agreement with FCS post-warmup.
    n_compare = min(len(binary_pred_A), len(fcs_post_A))
    agree = int((binary_pred_A[:n_compare] == fcs_post_A[:n_compare]).sum())
    print(f"  PMF-binary vs FCS-A direct agreement: "
          f"{agree}/{n_compare} = {agree/n_compare:.3f}")

    # Phase-shifted agreement (FCS may have a different starting phase).
    best_phase_agree = 0
    best_phase = 0
    for shift in range(4):
        rolled = np.roll(fcs_post_A, shift)
        a = int((binary_pred_A[:n_compare] == rolled[:n_compare]).sum())
        if a > best_phase_agree:
            best_phase_agree = a
            best_phase = shift
    print(f"  Best phase-shifted PMF vs FCS agreement: "
          f"{best_phase_agree}/{n_compare} = "
          f"{best_phase_agree/n_compare:.3f} at shift={best_phase}")

    np.savez(
        RESULTS / "pmf_trace.npz",
        fire_A=fire_A, fire_I=fire_I,
        pmf_A=result["pmf_A"], pmf_I=result["pmf_I"],
        fcs_A=fcs_A, fcs_I=fcs_I,
        binary_pred_A=binary_pred_A,
        pmf_match=np.array(list(pmf_match.values()), dtype=object),
        fcs_match=np.array(list(fcs_match.values()), dtype=object),
        warmup=warmup,
        agree=agree, best_phase_agree=best_phase_agree, best_phase=best_phase,
    )

    # ---------- Plot ----------
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    t_axis = np.arange(T_PMF)
    show_range = slice(0, 48)

    ax = axes[0]
    ax.plot(t_axis[show_range], fire_A[show_range], "-",
            color="tab:blue", lw=1.6, label=r"PMF $\Pr(\rm fire_A)(t)$")
    ax.plot(t_axis[show_range], fire_I[show_range], "-",
            color="tab:orange", lw=1.0, alpha=0.7,
            label=r"PMF $\Pr(\rm fire_I)(t)$")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6,
               label="threshold 0.5")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("PMF fire prob")
    ax.set_title(
        f"Experiment B: single-neuron renewal-PMF on negative-loop default cell\n"
        f"(w_XA={w_XA}, w_AI={w_AI}, w_IA={w_IA})", fontsize=11)
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    pmf_bin = binary_pred_A
    pmf_full = np.concatenate([np.zeros(warmup, dtype=int), pmf_bin])
    ax.step(t_axis[show_range], pmf_full[show_range], "-",
            color="tab:purple", where="post", lw=1.6,
            label="PMF binary (threshold > 0.5)")
    ax.set_ylim(-0.1, 1.2)
    ax.set_ylabel("PMF binary")
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9)

    ax = axes[2]
    ax.step(t_axis[show_range], fcs_A[show_range], "-",
            color="black", where="post", lw=1.6,
            label="FCS A (oracle)")
    ax.set_ylim(-0.1, 1.2)
    ax.set_ylabel("FCS A")
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9)

    ax = axes[3]
    # FCS I shifted by 1 (the property 5 phase relationship).
    ax.step(t_axis[show_range], fcs_I[show_range], "-",
            color="darkgreen", where="post", lw=1.6,
            label="FCS I (oracle)")
    ax.set_ylim(-0.1, 1.2)
    ax.set_ylabel("FCS I"); ax.set_xlabel("tick t")
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9)

    # Vertical guides every 4 ticks.
    for ax in axes:
        for x in range(0, 48, 4):
            ax.axvline(x, color="gray", linestyle=":", alpha=0.2)

    plt.tight_layout()
    out_pdf = RESULTS / "pmf_vs_fcs.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\n  wrote {out_pdf}")
    plt.close(fig)

    # Print verdict.
    print()
    pmf_corr = pmf_match["best_corr"]
    if pmf_period == 4 and pmf_corr >= 0.7:
        print(f"Experiment B PASS: PMF-binary spike train matches FCS "
              f"period 4 with template correlation {pmf_corr:.2f} >= 0.7.")
    elif pmf_period == 4:
        print(f"Experiment B PARTIAL: PMF-binary has correct period 4, "
              f"but template correlation {pmf_corr:.2f} < 0.7 -- "
              f"the binary pattern does not perfectly match the cyclic '1100'.")
    else:
        print(f"Experiment B FAIL: PMF-binary period {pmf_period} != 4, "
              f"or template match insufficient.")


if __name__ == "__main__":
    main()
