"""Phase 0 - Infrastructure validation.

Runs two textbook checks (V0.1, V0.2) against the Wilson-Cowan primitives
and emits a typst-rendered phase0_report.pdf with a PASS / FAIL verdict.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from linearization import is_stable, jacobian, spectrum  # noqa: E402
from topologies import contralateral_inhibition  # noqa: E402
from wilson_cowan import Sigmoid, find_fixed_point, simulate  # noqa: E402

SEED = 20260420
np.random.seed(SEED)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
FIG_DIR = RESULTS / "phase0"
FIG_DIR.mkdir(exist_ok=True)

TAU = 1.0


def banner(msg: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{msg}\n{bar}")


def save_fig(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")


def validation_v01(sigmoid: Sigmoid) -> dict:
    banner("V0.1  Single uncoupled population, I = 2")
    W = np.array([[0.0]])
    I = np.array([2.0])

    rho_analytical = float(sigmoid.f(2.0))
    rho_star, success = find_fixed_point(W, I, sigmoid, rho_guess=np.array([0.5]))
    fp_err = abs(float(rho_star[0]) - rho_analytical)

    t, y = simulate(
        W, I, TAU, sigmoid, t_span=(0.0, 20.0), rho0=np.array([0.0]), max_step=0.01
    )
    # monotone in this setup (single pop, positive drive, starts below FP)
    diffs = np.diff(y[0])
    monotone = bool(np.all(diffs >= -1e-10))

    J = jacobian(W, I, rho_star, TAU, sigmoid)
    eigvals, _ = spectrum(J)
    lam_numeric = float(eigvals[0].real)
    # For the self-coupling-free case W = [[0]] the standard WC Jacobian
    # (1/tau) * (-I_n + diag(f'(W rho* + I)) @ W) reduces to -1/tau exactly,
    # because the second term is multiplied by W = 0. The plan's §2.6 noted
    # "-1/tau + f'(I)/tau" which would correspond to W = 1 (a self-loop),
    # not W = 0. We use the value implied by the Jacobian formula in §2.5.
    lam_analytical = -1.0 / TAU
    eig_err = abs(lam_numeric - lam_analytical)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(t, y[0], linewidth=2, label=r"$\rho(t)$")
    ax.axhline(rho_analytical, linestyle="--", color="k", label=r"$f(2)$")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("V0.1 single population converging to $f(I)$")
    ax.legend()
    save_fig(fig, "v01_trajectory")
    plt.close(fig)

    return {
        "rho_analytical": rho_analytical,
        "rho_numerical": float(rho_star[0]),
        "fp_error": fp_err,
        "monotone": monotone,
        "found": success,
        "lambda_numeric": lam_numeric,
        "lambda_analytical": lam_analytical,
        "eig_error": eig_err,
        "pass": (fp_err < 1e-6) and monotone and (eig_err < 1e-6) and success,
    }


def validation_v02(sigmoid: Sigmoid) -> dict:
    banner("V0.2  Symmetric contralateral inhibition at low coupling")
    W, I = contralateral_inhibition(0.5, 0.5, drive=1.5)
    rho_star, success = find_fixed_point(W, I, sigmoid, rho_guess=np.array([0.5, 0.5]))
    symmetry_err = abs(float(rho_star[0]) - float(rho_star[1]))

    J = jacobian(W, I, rho_star, TAU, sigmoid)
    eigvals, _ = spectrum(J)
    stable = is_stable(J, tol=0.1)  # strict: all Re < -0.1

    perturbation = np.array([rho_star[0] + 0.01, rho_star[1] - 0.01])
    t, y = simulate(W, I, TAU, sigmoid, t_span=(0.0, 20.0), rho0=perturbation, max_step=0.01)

    idx10 = int(np.searchsorted(t, 10.0))
    idx10 = min(idx10, len(t) - 1)
    state_at_10 = y[:, idx10]
    return_err = float(np.linalg.norm(state_at_10 - rho_star))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(t, y[0], label=r"$\rho_1(t)$")
    ax.plot(t, y[1], label=r"$\rho_2(t)$")
    ax.axhline(float(rho_star[0]), linestyle="--", color="k", linewidth=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel("population rate")
    ax.set_title("V0.2 symmetric state recovers after small asymmetric kick")
    ax.legend()
    save_fig(fig, "v02_return_to_symmetry")
    plt.close(fig)

    eig_reals = np.sort(eigvals.real)

    return {
        "rho_star": rho_star.tolist(),
        "symmetry_error": symmetry_err,
        "eig_real_min": float(eig_reals[0]),
        "eig_real_max": float(eig_reals[-1]),
        "stable_re_lt_negpoint1": stable,
        "found": success,
        "return_err_at_t10": return_err,
        "pass": (
            success
            and symmetry_err < 1e-6
            and stable
            and return_err < 1e-3
        ),
    }


def render_report(v01: dict, v02: dict, overall_pass: bool) -> None:
    typ = HERE / "phase0_report.typ"
    pdf = HERE / "phase0_report.pdf"

    verdict = "PASS" if overall_pass else "FAIL"

    content = f"""#set page(paper: "a4", margin: 2cm)
#set text(size: 10pt)
#set par(justify: true)

#align(center)[
  #text(size: 14pt, weight: "bold")[Phase 0 report -- Infrastructure validation]
  #v(0.2em)
  Verdict: *{verdict}*
]

= File inventory

The following modules were created under `./deq/population/`:

- `wilson_cowan.py` -- sigmoid, RHS, simulator, fixed-point solver.
- `topologies.py` -- archetype weight-matrix builders.
- `linearization.py` -- Jacobian, spectrum, spectral gap, stability test.
- `phase0_infrastructure.py` -- this validation script.

Modules reserved as stubs for later phases: `bifurcation.py`,
`pole_placement.py`, `ground_truth.py`, `phase1_spectral_gap.py`,
`phase2_bifurcation.py`, `phase3_pole_placement.py`,
`phase4_cross_validation.py`, `phase5_other_archetypes.py`,
`final_summary.py`.

= Validation V0.1 -- Single uncoupled population

Setup: $W = [0]$, $I = 2$, $tau = 1$, sigmoid $k = 4$, $theta = 1$.

Analytical fixed point: $rho^* = f(2) = 1 / (1 + e^(-4)) approx {v01['rho_analytical']:.9f}$.

Numerical fixed point: ${v01['rho_numerical']:.9f}$; error ${v01['fp_error']:.2e}$.

Analytical eigenvalue of Jacobian: ${v01['lambda_analytical']:.9f}$
(the Jacobian formula in the plan §2.5 reduces to $-1 slash tau$ when $W = 0$,
since the sigmoid-slope term is multiplied by the zero self-coupling; a
noted typo in plan §2.6 wrote $-1 slash tau + f'(I) slash tau$, which would
correspond to a unit self-loop instead).
Numerical eigenvalue: ${v01['lambda_numeric']:.9f}$; error ${v01['eig_error']:.2e}$.

Trajectory from $rho_0 = 0$ monotone: {"yes" if v01['monotone'] else "NO"}.

Acceptance: fixed-point error $< 10^{{-6}}$, eigenvalue error $< 10^{{-6}}$, monotone --
*{"PASS" if v01['pass'] else "FAIL"}*.

#figure(image("results/phase0/v01_trajectory.pdf", width: 75%),
  caption: [V0.1: single population trajectory from zero initial condition
  converges monotonically to the analytical fixed point $f(I) = f(2)$.])

= Validation V0.2 -- Symmetric contralateral inhibition, low coupling

Setup: $W = "contralateral_inhibition"(0.5, 0.5)$, $I = [1.5, 1.5]$.

Symmetric fixed point: $(rho_1^*, rho_2^*) = ({v02['rho_star'][0]:.9f}, {v02['rho_star'][1]:.9f})$;
symmetry error ${v02['symmetry_error']:.2e}$.

Jacobian eigenvalue real parts: min ${v02['eig_real_min']:.6f}$, max ${v02['eig_real_max']:.6f}$.
Both below $-0.1$: {"yes" if v02['stable_re_lt_negpoint1'] else "NO"}.

Return distance at $t = 10$ from perturbed initial condition $rho_0 = (rho_1^* + 0.01, rho_2^* - 0.01)$:
${v02['return_err_at_t10']:.2e}$ (threshold $10^{{-3}}$).

Acceptance: *{"PASS" if v02['pass'] else "FAIL"}*.

#figure(image("results/phase0/v02_return_to_symmetry.pdf", width: 75%),
  caption: [V0.2: after a small asymmetric perturbation the two population
  rates return to the symmetric fixed point (dashed line) well within ten
  time constants.])

= Overall verdict

*{verdict}*.

All Phase 0 acceptance criteria {'are met' if overall_pass else 'are NOT met'}.
"""
    typ.write_text(content)
    subprocess.run(
        ["typst", "compile", str(typ), str(pdf)], check=True, cwd=str(HERE)
    )


def main() -> int:
    sigmoid = Sigmoid(k=4.0, theta=1.0)
    v01 = validation_v01(sigmoid)
    v02 = validation_v02(sigmoid)

    print("\nV0.1 results:")
    for k, v in v01.items():
        print(f"  {k}: {v}")
    print("\nV0.2 results:")
    for k, v in v02.items():
        print(f"  {k}: {v}")

    overall = v01["pass"] and v02["pass"]
    render_report(v01, v02, overall)

    banner(f"Phase 0 verdict: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
