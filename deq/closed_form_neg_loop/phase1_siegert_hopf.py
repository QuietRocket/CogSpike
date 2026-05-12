"""Phase 1: Siegert (static) reading of FCS Property 5 (negative loop).

For each (w_IA, w_XA) cell on the Phase 0 grid:

  1. Solve the 2-pop Siegert fixed point (nu_A, nu_I) on the negative-loop
     weight structure: A receives external drive `w_XA * p_thin` plus
     w_IA * nu_I; I receives w_AI * nu_A only.
  2. Compute the Jacobian eigenvalues of the rate-equation linearization
     at the FP via deq/closed_form/transfer.py (single-pole low-pass).
  3. Classify the FP: stable spiral (Re < 0, Im != 0), stable node
     (Re < 0, Im == 0), or unstable (Re > 0).

Calibration constants (alpha, beta, tau_m, tau_ref) are locked from
deq/closed_form/results/phase1_grid.npz (the closed_form_wta operating
point at p_thin=0.7).

For the standard 2-pop negative-loop weight matrix W = [[0, w_IA],
[w_AI, 0]] with w_IA < 0 < w_AI, the Jacobian A = (1/tau_m)(-I +
diag(g) J) has eigenvalues (-1 +/- sqrt(g_A * g_I * w_AI * w_IA)) / tau_m
where the radicand is negative -> a complex-conjugate pair with
Re(lambda) = -1/tau_m < 0. So the FP is ALWAYS a stable spiral, never
Hopf-unstable. The interesting prediction is the ringing frequency
Im(lambda), interpreted as the natural oscillation frequency of the
linearization. Phase 2 turns this into a predicted period.

Label spiral_blue iff the FP exists AND eigenvalues are complex
(Im(lambda) != 0) — i.e. the linearization at least *rings*. Compare
against FCS broad_osc and strict_p5.

Output: results/phase1/siegert_grid.npz, hopf_vs_fcs.pdf, eig_spectrum.pdf.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from deq.closed_form.siegert import Siegert  # noqa: E402
from deq.closed_form.transfer import (  # noqa: E402
    dphi_dmu,
    jacobian_eigenvalues,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "phase1"
RESULTS.mkdir(parents=True, exist_ok=True)


P_THIN = 0.7   # locked at the closed_form_wta calibration operating point
SIGMA_FLOOR = 1e-3


def load_calibration() -> dict:
    src = ROOT / "deq" / "closed_form" / "results" / "phase1_grid.npz"
    d = np.load(src, allow_pickle=True)
    return dict(
        alpha=float(d["calib_alpha"]),
        beta=float(d["calib_beta"]),
        tau_m=float(d["calib_tau_m"]),
        tau_ref=float(d["calib_tau_ref"]),
        r2=float(d["calib_r2"]),
    )


def neg_loop_inputs(nu: np.ndarray, w_AI: float, w_IA: float,
                    w_XA: float, alpha: float, beta: float) -> tuple:
    """At (nu_A, nu_I), compute (mu, sigma) for both populations.

    A receives external X (Bernoulli, mean p_thin) weighted by w_XA, plus
    inhibitory feedback from I weighted by w_IA (negative).
    I receives nothing external, only excitation from A weighted by w_AI.
    """
    nu_A, nu_I = float(nu[0]), float(nu[1])

    # Mean inputs.
    mean_A = w_XA * P_THIN + w_IA * nu_I
    mean_I = w_AI * nu_A

    # Variance inputs.
    var_A = (w_XA ** 2) * P_THIN * (1 - P_THIN) + (w_IA ** 2) * nu_I * (1 - nu_I)
    var_I = (w_AI ** 2) * nu_A * (1 - nu_A)

    mu = np.array([alpha * mean_A, alpha * mean_I])
    sigma = np.array([
        math.sqrt(max(beta * var_A, 0.0)),
        math.sqrt(max(beta * var_I, 0.0)),
    ])
    return mu, sigma


def find_fixed_point_neg_loop(w_AI: float, w_IA: float, w_XA: float,
                              siegert: Siegert, alpha: float, beta: float,
                              guesses=None) -> tuple:
    """Return (nu_star, success). Tries several initial guesses."""
    if guesses is None:
        guesses = [(0.5, 0.5), (0.9, 0.5), (0.1, 0.5),
                   (0.5, 0.9), (0.5, 0.1)]

    def residual(nu):
        nu_c = np.clip(nu, 0.0, 1.0)
        mu, sigma = neg_loop_inputs(nu_c, w_AI, w_IA, w_XA, alpha, beta)
        nu_pred = np.array([siegert.phi(mu[k], sigma[k]) for k in range(2)])
        return nu - nu_pred

    best_fp = None
    best_res = float("inf")
    for g in guesses:
        try:
            nu_star, _info, ier, _msg = fsolve(
                residual, np.array(g, dtype=float),
                full_output=True, xtol=1e-9,
            )
            r = float(np.max(np.abs(residual(nu_star))))
            if ier == 1 and r < best_res:
                best_fp = np.clip(nu_star, 0.0, 1.0)
                best_res = r
        except Exception:
            continue

    return (best_fp, True) if (best_fp is not None and best_res < 1e-6) \
        else (None, False)


def jacobian_neg_loop(nu_star: np.ndarray, w_AI: float, w_IA: float,
                      w_XA: float, siegert: Siegert,
                      alpha: float, beta: float, tau_m: float) -> np.ndarray:
    """Return the 2 complex eigenvalues of the rate-equation Jacobian at nu_star.

    The Jacobian is A = (1/tau_m)(-I + diag(g) J), with
      J[0,1] = alpha * w_IA  (A receives from I)
      J[1,0] = alpha * w_AI  (I receives from A)
      g_i    = dPhi/dmu at (mu_i*, sigma_i*).
    """
    mu, sigma = neg_loop_inputs(nu_star, w_AI, w_IA, w_XA, alpha, beta)
    g_A = dphi_dmu(siegert, float(mu[0]), float(sigma[0]))
    g_I = dphi_dmu(siegert, float(mu[1]), float(sigma[1]))
    # Saturation / degeneracy: dphi_dmu can blow up when nu is at the
    # boundary (nu -> 0 or 1) because the erfcx bracket diverges.
    if not (math.isfinite(g_A) and math.isfinite(g_I)):
        return np.array([np.nan + 0j, np.nan + 0j])
    J = alpha * np.array([[0.0, w_IA], [w_AI, 0.0]])
    gains = np.array([g_A, g_I])
    eigs = jacobian_eigenvalues(J, gains, tau_m)
    return eigs


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    inter = (a & b).sum(); union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else 1.0


def main():
    calib = load_calibration()
    print("Calibration (locked from closed_form/phase1):")
    for k, v in calib.items():
        print(f"  {k} = {v:.6f}")
    print()

    siegert = Siegert(V_th=1.0, V_r=0.0,
                      tau_m=calib["tau_m"], tau_ref=calib["tau_ref"])

    W_AI = 11
    p0 = np.load(HERE / "results" / "phase0" / "fcs_grid.npz",
                 allow_pickle=True)
    W_IA = p0["W_IA_VALUES"]
    W_XA = p0["W_XA_VALUES"]
    strict_p5 = p0["strict_p5"]
    broad_osc = p0["broad_osc"]
    nIA, nXA = len(W_IA), len(W_XA)

    print(f"Siegert FP + Jacobian on {nIA} x {nXA} grid; "
          f"w_AI={W_AI}, p_thin={P_THIN}\n")

    fp_grid = np.full((nIA, nXA, 2), np.nan)
    eig_re = np.full((nIA, nXA, 2), np.nan)
    eig_im = np.full((nIA, nXA, 2), np.nan)
    spiral_blue = np.zeros((nIA, nXA), dtype=int)
    success_grid = np.zeros((nIA, nXA), dtype=int)

    for i, w_IA in enumerate(W_IA):
        for j, w_XA in enumerate(W_XA):
            fp, ok = find_fixed_point_neg_loop(
                float(W_AI), float(w_IA), float(w_XA),
                siegert, calib["alpha"], calib["beta"],
            )
            success_grid[i, j] = int(ok)
            if not ok:
                continue
            fp_grid[i, j] = fp
            eigs = jacobian_neg_loop(
                fp, float(W_AI), float(w_IA), float(w_XA),
                siegert, calib["alpha"], calib["beta"], calib["tau_m"],
            )
            if np.any(~np.isfinite(eigs)):
                # Degenerate gain (saturated FP); skip.
                continue
            # Sort by descending Im so eigs[0] is the upper conjugate pair member.
            order = np.argsort(-np.imag(eigs))
            eigs = eigs[order]
            eig_re[i, j] = np.real(eigs)
            eig_im[i, j] = np.imag(eigs)
            # Spiral: complex conjugate pair (Im != 0).
            spiral_blue[i, j] = int(abs(np.imag(eigs[0])) > 1e-8)

    print(f"FP convergence: {int(success_grid.sum())}/{success_grid.size} cells\n")

    # Eigenvalue regime stats.
    has_fp = success_grid.astype(bool)
    re_dom = np.where(has_fp, eig_re[..., 0], np.nan)  # eigs[0] has largest Im
    is_complex = np.where(has_fp, np.abs(eig_im[..., 0]) > 1e-8, False)
    is_unstable = np.where(has_fp, re_dom > 0, False)
    print(f"Eigenvalue census (cells with FP):")
    print(f"  complex spiral (Im!=0): {int(is_complex.sum())} / "
          f"{int(has_fp.sum())} ({100*is_complex.sum()/max(has_fp.sum(),1):.1f}%)")
    print(f"  unstable (Re > 0):      {int(is_unstable.sum())} / "
          f"{int(has_fp.sum())} ({100*is_unstable.sum()/max(has_fp.sum(),1):.1f}%)")
    print(f"  stable spiral (Re<0, Im!=0): "
          f"{int((is_complex & ~is_unstable).sum())}")
    print()

    # Jaccards against FCS labels.
    j_strict = jaccard(spiral_blue, strict_p5)
    j_broad = jaccard(spiral_blue, broad_osc)
    print(f"Spiral-blue cells: {int(spiral_blue.sum())} / {spiral_blue.size}")
    print(f"  Jaccard vs FCS strict_p5: {j_strict:.3f}")
    print(f"  Jaccard vs FCS broad_osc: {j_broad:.3f}")
    print()

    np.savez(
        RESULTS / "siegert_grid.npz",
        W_IA_VALUES=W_IA,
        W_XA_VALUES=W_XA,
        W_AI=W_AI,
        fp_grid=fp_grid,
        eig_re=eig_re,
        eig_im=eig_im,
        spiral_blue=spiral_blue,
        success_grid=success_grid,
        jaccard_strict=j_strict,
        jaccard_broad=j_broad,
        **calib,
        p_thin=P_THIN,
    )

    # ---------- plots ----------

    # 3-panel: FCS strict, FCS broad, Siegert spiral.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    panels = [
        (axes[0], strict_p5,
            f"FCS strict Property 5\n{int(strict_p5.sum())}/{strict_p5.size} blue"),
        (axes[1], broad_osc,
            f"FCS broad oscillation\n{int(broad_osc.sum())}/{broad_osc.size} blue"),
        (axes[2], spiral_blue,
            f"Siegert spiral (Im(λ)≠0)\n"
            f"{int(spiral_blue.sum())}/{spiral_blue.size} blue\n"
            f"J(strict)={j_strict:.3f}, J(broad)={j_broad:.3f}"),
    ]
    for ax, labels, title in panels:
        for i, w_IA in enumerate(W_IA):
            for j, w_XA in enumerate(W_XA):
                color = "tab:blue" if labels[i, j] else "tab:red"
                ax.scatter(int(w_XA), int(w_IA), c=color, s=18,
                           edgecolor="none")
        ax.scatter([11], [-11], facecolors="none", edgecolors="gold",
                   s=160, linewidths=2.0, zorder=5)
        ax.set_xlabel(r"$w_{XA}$")
        ax.set_ylabel(r"$w_{IA}$")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
    fig.suptitle("Phase 1: Siegert FP + Jacobian vs FCS Property 5", fontsize=11)
    plt.tight_layout()
    out_pdf = RESULTS / "hopf_vs_fcs.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    plt.close(fig)

    # Eigenvalue spectrum scatter (Re vs Im).
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    re_flat = eig_re[..., 0].ravel()
    im_flat = eig_im[..., 0].ravel()
    fcs_flat = strict_p5.ravel().astype(bool)
    mask = ~np.isnan(re_flat)
    ax.scatter(re_flat[mask & ~fcs_flat], im_flat[mask & ~fcs_flat],
               c="lightgray", s=8, alpha=0.5, label="FCS not P5")
    ax.scatter(re_flat[mask & fcs_flat], im_flat[mask & fcs_flat],
               c="tab:blue", s=14, alpha=0.85, label="FCS strict P5")
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xlabel(r"Re($\lambda$)")
    ax.set_ylabel(r"Im($\lambda$)")
    ax.set_title(
        "Phase 1: Jacobian eigenvalue spectrum across grid\n"
        f"(Re=−1/τ_m={-1/calib['tau_m']:.3f} for all complex cells)",
        fontsize=10,
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_pdf = RESULTS / "eig_spectrum.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    plt.close(fig)

    # Heatmap of Im(lambda) — the natural ringing frequency.
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
    im_dom = eig_im[..., 0]
    im_plot = np.where(np.isnan(im_dom), 0, im_dom)
    im_im = ax.imshow(
        im_plot, origin="lower", cmap="plasma",
        vmin=0, vmax=np.nanmax(im_plot),
        extent=[W_XA[0] - 0.5, W_XA[-1] + 0.5,
                W_IA[0] - 0.5, W_IA[-1] + 0.5],
        aspect="auto",
    )
    cbar = plt.colorbar(im_im, ax=ax)
    cbar.set_label(r"Im($\lambda$): natural ringing rate (1/tick)")
    ax.scatter([11], [-11], facecolors="none", edgecolors="white",
               s=160, linewidths=2.0, zorder=5)
    ax.set_xlabel(r"$w_{XA}$")
    ax.set_ylabel(r"$w_{IA}$")
    ax.set_title(
        "Phase 1: Im(λ) heatmap — predicted ringing frequency at the FP",
        fontsize=10,
    )
    plt.tight_layout()
    out_pdf = RESULTS / "im_lambda_heatmap.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")
    plt.close(fig)

    # Sanity gate: at default cell (w_IA=-11, w_XA=11), Im(lambda) > 0
    # (stable spiral with ringing).
    i_def = int(np.where(W_IA == -11)[0][0])
    j_def = int(np.where(W_XA == 11)[0][0])
    print(f"FCS default cell (w_IA=-11, w_XA=11):")
    print(f"  FP = ({fp_grid[i_def, j_def, 0]:.4f}, "
          f"{fp_grid[i_def, j_def, 1]:.4f})")
    print(f"  eigs = ({eig_re[i_def, j_def, 0]:+.4f} + "
          f"{eig_im[i_def, j_def, 0]:+.4f}i, "
          f"{eig_re[i_def, j_def, 1]:+.4f} + "
          f"{eig_im[i_def, j_def, 1]:+.4f}i)")
    is_spiral = bool(spiral_blue[i_def, j_def])
    print(f"  spiral_blue = {int(is_spiral)}")
    print()
    if is_spiral:
        print("Phase 1 PASS gate: FCS default cell is a stable spiral with "
              "Im(λ) > 0 (decaying ringing). The rate-equation FP is "
              "linearly stable; Property 5 lives beyond what mean-field "
              "rate theory can predict — Phase 2 will check whether the "
              "ringing FREQUENCY matches FCS's period-4 anyway.")
    else:
        print("Phase 1 FAIL gate: default cell is not a spiral (FP is "
              "either non-existent or has real eigenvalues only).")


if __name__ == "__main__":
    main()
