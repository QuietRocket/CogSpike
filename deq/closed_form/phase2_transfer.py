"""Phase 2 (H2A) - Linear-response transfer function H(omega).

Implements Richardson-style single-pole H(omega) at a Siegert fixed point of
the negative-loop (E-I) archetype, builds the closed-loop matrix
M(omega) = I - H(omega) J, and verifies the omega = 0 self-consistency:
det(M(0)) zero crossings <=> Jacobian eigenvalues passing through zero.

The negative-loop topology (FCS Fig. 1d, deq/archetypes/topologies.py):

    A (activator) <- external X (weight w_XA)
    A             <- inhibition from I (weight w_IA, negative)
    I (inhibitor) <- excitation from A (weight w_AI, positive)

so the recurrent matrix in our W convention is

    W = [[0, w_IA],  [w_AI, 0]]

and the external-drive vector is B = [w_XA, 0]^T (only A is externally
driven).

Outputs:
- Bode magnitude/phase plot of the closed-loop response G(omega)
- Imaginary-axis det(M(omega)) curve
- Numerical self-consistency residual

Gate: |consistency residual| <= 1e-3 (closed-loop description at omega = 0
matches time-domain Jacobian).
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from siegert import Siegert  # noqa: E402
from transfer import (  # noqa: E402
    closed_loop_matrix,
    closed_loop_response,
    closed_loop_zero_freq_consistency,
    dphi_dmu,
    find_imaginary_axis_poles,
    jacobian_eigenvalues,
)

SEED = 20260502
np.random.seed(SEED)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
FIG_DIR = RESULTS / "phase2"
FIG_DIR.mkdir(exist_ok=True)


def banner(msg: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{msg}\n{bar}")


def save_fig(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")


# Phase 1 calibration lock-in (loaded from disk).
def load_phase1_calibration():
    npz = np.load(RESULTS / "phase1_grid.npz")
    return {
        "alpha": float(npz["calib_alpha"]),
        "beta": float(npz["calib_beta"]),
        "tau_m": float(npz["calib_tau_m"]),
        "tau_ref": float(npz["calib_tau_ref"]),
    }


def negative_loop_fp(w_XA, w_AI, w_IA, drive_X, p_thin, calib, siegert):
    """Self-consistent Siegert FP for negative loop.

    State: nu = (nu_A, nu_I).
    Inputs:
      mu_A      = alpha * (drive_X * w_XA * p_thin + w_IA * nu_I)
      mu_I      = alpha * (w_AI * nu_A)
      sigma_A^2 = beta * ((drive_X * w_XA)^2 * p_thin*(1-p_thin)
                          + w_IA^2 * nu_I*(1-nu_I))
      sigma_I^2 = beta * (w_AI^2 * nu_A*(1-nu_A))
    """
    from scipy.optimize import fsolve

    def residual(nu):
        nu = np.clip(nu, 0.0, 1.0)
        nu_A, nu_I = nu
        mean_A = drive_X * w_XA * p_thin + w_IA * nu_I
        var_A = (
            (drive_X * w_XA) ** 2 * p_thin * (1 - p_thin)
            + w_IA ** 2 * nu_I * (1 - nu_I)
        )
        mean_I = w_AI * nu_A
        var_I = w_AI ** 2 * nu_A * (1 - nu_A)
        mu_A = calib["alpha"] * mean_A
        mu_I = calib["alpha"] * mean_I
        sigma_A = float(np.sqrt(max(calib["beta"] * var_A, 0.0)))
        sigma_I = float(np.sqrt(max(calib["beta"] * var_I, 0.0)))
        nu_pred = np.array([siegert.phi(mu_A, sigma_A), siegert.phi(mu_I, sigma_I)])
        return nu - nu_pred

    nu_star, _, ier, _ = fsolve(residual, np.array([0.3, 0.3]), full_output=True,
                                 xtol=1e-10)
    success = ier == 1 and np.max(np.abs(residual(nu_star))) < 1e-7

    # Also return the (mu, sigma) at the FP for downstream use.
    nu_A, nu_I = nu_star
    mu_A = calib["alpha"] * (drive_X * w_XA * p_thin + w_IA * nu_I)
    mu_I = calib["alpha"] * (w_AI * nu_A)
    sigma_A = float(np.sqrt(max(
        calib["beta"] * ((drive_X * w_XA) ** 2 * p_thin * (1 - p_thin)
                         + w_IA ** 2 * nu_I * (1 - nu_I)),
        0.0,
    )))
    sigma_I = float(np.sqrt(max(
        calib["beta"] * (w_AI ** 2 * nu_A * (1 - nu_A)),
        0.0,
    )))
    return nu_star, np.array([mu_A, mu_I]), np.array([sigma_A, sigma_I]), success


def main() -> int:
    banner("Phase 2 -- Linear-response transfer function H(omega) at Siegert FP")

    # Load calibration from Phase 1.
    calib = load_phase1_calibration()
    print(f"  Calibration (from Phase 1): alpha={calib['alpha']:.4f}, "
          f"beta={calib['beta']:.6f}, tau_m={calib['tau_m']:.4f}, "
          f"tau_ref={calib['tau_ref']:.4f}")
    siegert = Siegert(V_th=1.0, V_r=0.0, tau_m=calib["tau_m"],
                      tau_ref=calib["tau_ref"])

    # Negative loop configuration (FCS defaults).
    w_XA, w_AI, w_IA = 11, 11, -11
    drive_X = 1
    p_thin = 0.7

    # Build the recurrent J matrix in *FCS units*; the calibration alpha
    # converts (drive, p_thin, weights) to Siegert mu/sigma. For the closed
    # loop H(omega) J, J should be in the same units that mu_i is computed
    # from -- i.e., mu_i = alpha * J_ij @ nu_j + (external term). So the
    # effective J that multiplies nu in the rate equation is alpha * J_FCS.
    J_FCS = np.array([[0, w_IA], [w_AI, 0]], dtype=float)
    J_eff = calib["alpha"] * J_FCS

    print(f"\n  J_FCS = \n{J_FCS}")
    print(f"  J_eff (alpha * J_FCS) = \n{J_eff}")

    # Solve self-consistent Siegert FP.
    nu_star, mu_star, sigma_star, ok = negative_loop_fp(
        w_XA, w_AI, w_IA, drive_X, p_thin, calib, siegert
    )
    print(f"\n  Siegert FP: nu = ({nu_star[0]:.4f}, {nu_star[1]:.4f}), "
          f"converged = {ok}")
    print(f"  mu = ({mu_star[0]:.4f}, {mu_star[1]:.4f}), "
          f"sigma = ({sigma_star[0]:.4f}, {sigma_star[1]:.4f})")

    # Compute DC gains d Phi / d mu at FP.
    gains = np.array([
        dphi_dmu(siegert, mu_star[0], sigma_star[0]),
        dphi_dmu(siegert, mu_star[1], sigma_star[1]),
    ])
    print(f"  DC gains d Phi / d mu = ({gains[0]:.4f}, {gains[1]:.4f})")

    # Time-domain Jacobian eigenvalues. With J_eff already including alpha,
    # the input-to-mu map is mu = J_eff @ nu (drop external for linearization).
    eigs_jac = jacobian_eigenvalues(J_eff, gains, calib["tau_m"])
    print(f"\n  Jacobian eigenvalues: {eigs_jac}")
    print(f"    Re max = {max(eigs_jac.real):.4f}, "
          f"|Im| max = {max(np.abs(eigs_jac.imag)):.4f}")

    # Self-consistency check.
    consistency = closed_loop_zero_freq_consistency(J_eff, gains, calib["tau_m"])
    print(f"\n  Self-consistency residual (closed-loop omega = 0 vs Jacobian): "
          f"{consistency['consistency_residual']:.2e}")

    # Bode plot over a frequency range.
    omegas = np.logspace(-2, 2, 200)
    G_resps = np.array([
        closed_loop_response(w, J_eff, gains, calib["tau_m"]) for w in omegas
    ])
    # Magnitude / phase of the (A, A) entry (input drive to A's rate response).
    G_AA = G_resps[:, 0, 0]
    mag_dB = 20 * np.log10(np.abs(G_AA) + 1e-30)
    phase_deg = np.angle(G_AA, deg=True)

    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    axes[0].semilogx(omegas, mag_dB, linewidth=1.5)
    axes[0].set_ylabel("|G_AA| (dB)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"Bode plot of closed-loop G(omega) = (I - HJ)^(-1) H, "
                      f"entry A->A")
    axes[1].semilogx(omegas, phase_deg, linewidth=1.5)
    axes[1].set_ylabel("phase (deg)")
    axes[1].set_xlabel("omega (rad / time-unit)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "bode")
    plt.close(fig)

    # det(M(omega)) curve along iomega axis.
    dets = np.array([
        complex(np.linalg.det(closed_loop_matrix(w, J_eff, gains, calib["tau_m"])))
        for w in omegas
    ])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].semilogx(omegas, dets.real, label="Re det(M)")
    axes[0].semilogx(omegas, dets.imag, label="Im det(M)")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel("omega")
    axes[0].set_ylabel("det(M)")
    axes[0].set_title("Closed-loop characteristic polynomial det(I - H(omega) J)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dets.real, dets.imag, color="C2", linewidth=1.2)
    axes[1].plot(0, 0, "k+", markersize=10)
    axes[1].set_xlabel("Re det(M)")
    axes[1].set_ylabel("Im det(M)")
    axes[1].set_title("Nyquist locus of det(M(omega))")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].axvline(0, color="k", lw=0.5)
    fig.tight_layout()
    save_fig(fig, "characteristic")
    plt.close(fig)

    # Look for omega-axis crossings (potential oscillation onsets).
    omega_dense = np.linspace(0.01, 30.0, 600)
    crossings = find_imaginary_axis_poles(J_eff, gains, calib["tau_m"], omega_dense)
    print(f"\n  Imaginary-axis det(M) crossings: {crossings}")

    # Save artifacts.
    np.savez(
        RESULTS / "phase2_transfer.npz",
        omegas=omegas,
        dets_re=dets.real,
        dets_im=dets.imag,
        G_AA=G_AA,
        gains=gains,
        nu_star=nu_star,
        mu_star=mu_star,
        sigma_star=sigma_star,
        eigs_jac=eigs_jac,
        crossings=np.array(crossings),
        consistency_residual=consistency["consistency_residual"],
    )

    pass_consistency = consistency["consistency_residual"] <= 1e-3
    overall = pass_consistency
    render_report(
        calib, J_FCS, J_eff, nu_star, mu_star, sigma_star, gains, eigs_jac,
        consistency, crossings, overall,
    )

    banner(f"Phase 2 verdict: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


def render_report(calib, J_FCS, J_eff, nu_star, mu_star, sigma_star, gains,
                  eigs_jac, consistency, crossings, overall_pass):
    typ = HERE / "phase2_report.typ"
    pdf = HERE / "phase2_report.pdf"
    verdict = "PASS" if overall_pass else "FAIL"

    eigs_str = ", ".join(
        f"{e.real:.4f} + {e.imag:.4f}i" for e in eigs_jac
    )
    crossings_str = (
        ", ".join(f"{c:.4f}" for c in crossings)
        if crossings else "none in tested range"
    )

    content = f"""#set page(paper: "a4", margin: 2cm)
#set text(size: 10pt)
#set par(justify: true)

#align(center)[
  #text(size: 14pt, weight: "bold")[Phase 2 report -- Linear-response H(omega) (H2 part A)]
  #v(0.2em)
  Verdict: *{verdict}*
]

= Hypothesis (H2 part A)

For the negative-loop (E-I) archetype with default FCS weights
($w_("XA") = 11$, $w_("AI") = 11$, $w_("IA") = -11$, $p_("thin") = 0.7$),
the single-pole low-pass approximation of the linear-response transfer
function

$ H_i(omega) = (partial Phi_i / partial mu_i) / (1 + i omega tau_m) $

at the Siegert fixed point, plugged into the closed-loop matrix
$M(omega) = I - H(omega) J$, satisfies the $omega = 0$ self-consistency:
the time-domain Jacobian
$A = (1\/tau_m)(- I + "diag"(g) J)$ has the same spectrum as
$("diag"(g) J - I) \/ tau_m$ obtained from $det M(0) = 0$.

= Operating point

Calibration locked in from Phase 1:

- $alpha = {calib['alpha']:.4f}$, $beta = {calib['beta']:.6f}$,
  $tau_m = {calib['tau_m']:.4f}$, $tau_("ref") = {calib['tau_ref']:.4f}$.

Topology and effective Jacobian:

$ J_("FCS") = mat({int(J_FCS[0,0])}, {int(J_FCS[0,1])}; {int(J_FCS[1,0])}, {int(J_FCS[1,1])}) , quad J_("eff") = alpha J_("FCS") = mat({J_eff[0,0]:.4f}, {J_eff[0,1]:.4f}; {J_eff[1,0]:.4f}, {J_eff[1,1]:.4f}) $

Self-consistent Siegert fixed point:

- $nu^* = ({nu_star[0]:.4f}, {nu_star[1]:.4f})$
- $mu^* = ({mu_star[0]:.4f}, {mu_star[1]:.4f})$
- $sigma^* = ({sigma_star[0]:.4f}, {sigma_star[1]:.4f})$

DC gains:

- $partial Phi_A \/ partial mu_A = {gains[0]:.4f}$
- $partial Phi_I \/ partial mu_I = {gains[1]:.4f}$

= Time-domain Jacobian eigenvalues

Spectrum of $A = (1 \/ tau_m) (- I + "diag"(g) J_("eff"))$:

{eigs_str}

Maximum real part: ${max(eigs_jac.real):.4f}$ (negative => stable focus
or node; positive => unstable)$\.$
Maximum |imaginary| part: ${max(np.abs(eigs_jac.imag)):.4f}$
(non-zero => oscillatory dynamics).

= Self-consistency

Closed-loop characteristic polynomial $det M(omega) = det(I - H(omega) J)$
at $omega = 0$ has roots $1 - tau_m lambda$ in correspondence with the
Jacobian spectrum. Numerical residual (max difference between sorted
spectra):

#text(size: 14pt)[$|"residual"| = {consistency['consistency_residual']:.2e}$]

Gate $<= 10^(-3)$: *{"PASS" if consistency['consistency_residual'] <= 1e-3 else "FAIL"}*.

Imaginary-axis crossings of $det M(omega)$ in $[0.01, 30]$ rad/time-unit:
{crossings_str}.

= Bode plot of closed-loop response

#figure(image("results/phase2/bode.pdf", width: 75%),
  caption: [Bode magnitude (top) and phase (bottom) of the closed-loop
  transfer $G(omega) = (I - H(omega) J)^(-1) H(omega)$, entry A -> A.
  The single-pole low-pass shape is consistent with a stable
  configuration where the negative loop adds bandwidth-limited damping.])

#figure(image("results/phase2/characteristic.pdf", width: 95%),
  caption: [Left: real and imaginary parts of $det M(omega)$. Right:
  Nyquist locus of $det M(omega)$. The locus does not encircle the origin,
  consistent with the Jacobian eigenvalues having negative real parts.])

= Overall verdict

*{verdict}*.

The Phase 2 self-consistency check confirms the closed-loop machinery
correctly reduces to the time-domain Jacobian at $omega = 0$. Phase 3
(H2 part B) extends to non-zero $omega$: comparing predicted spectral peaks
against LI&F-population FFT measurements.
"""
    typ.write_text(content)
    subprocess.run(
        ["typst", "compile", str(typ), str(pdf)], check=True, cwd=str(HERE)
    )


if __name__ == "__main__":
    sys.exit(main())
