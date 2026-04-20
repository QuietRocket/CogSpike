"""Phase 5 - Generalization to other archetypes.

Three subtasks exercise the spectral framework on archetypes beyond the
contralateral-inhibition and negative-loop cases studied in Phases 1-3:

  5A. Simple series of n populations -- steady-state transmission/
      attenuation; analytical product-of-gains prediction vs numerical
      integration.

  5B. Parallel composition of n populations driven by a single source --
      block-diagonal Jacobian structure; verify each downstream population's
      stability is independent.

  5C. Positive loop of two mutually exciting populations -- saddle-node
      bifurcation producing a high-active FP alongside the low-active one;
      contrast with the contralateral case's pitchfork.

Each subtask declares PASS iff its analytical prediction matches the
numerical simulation to within 5 % relative error in the predicted
regime and no free parameter has been tuned.
"""

from __future__ import annotations

import io
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=sys.stdout.encoding, line_buffering=True)

from linearization import jacobian, spectrum  # noqa: E402
from topologies import (  # noqa: E402
    parallel_composition,
    positive_loop,
    simple_series,
)
from wilson_cowan import (  # noqa: E402
    Sigmoid,
    find_fixed_point,
    simulate,
)

SEED = 20260420
np.random.seed(SEED)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
FIG_DIR = RESULTS / "phase5"
FIG_DIR.mkdir(exist_ok=True)

TAU = 1.0
SIGMOID_K = 4.0
SIGMOID_THETA = 1.0


def banner(msg: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


def save_fig(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(
            FIG_DIR / f"{name}.{ext}",
            dpi=300 if ext == "png" else None,
            bbox_inches="tight",
        )


# ----------------------------------------------------------------------------
# 5A. Simple series of n populations
# ----------------------------------------------------------------------------


def series_steady_state_numerical(
    n: int, w: float, drive: float, sigmoid: Sigmoid, tau: float = TAU
) -> np.ndarray:
    """Integrate the chain long enough for steady state and return rho^*.

    Each stage is a Wilson-Cowan population with feed-forward input from the
    previous stage's activity. For a feed-forward chain with no recurrence,
    the steady state is unique and reached monotonically.
    """
    W, Ivec = simple_series([w] * (n - 1), drive=drive)
    rho_star, ok = find_fixed_point(W, Ivec, sigmoid, rho_guess=np.full(n, 0.5))
    if ok:
        return rho_star
    # Fallback: long simulation.
    t, y = simulate(
        W, Ivec, tau, sigmoid,
        t_span=(0.0, 200.0), rho0=np.zeros(n),
        rtol=1e-8, atol=1e-10, max_step=0.1,
    )
    return y[:, -1]


def series_steady_state_analytical(
    n: int, w: float, drive: float, sigmoid: Sigmoid
) -> np.ndarray:
    """Closed-form steady-state for the feed-forward chain.

    At stage 0: rho_0 = f(drive).
    At stage k > 0: rho_k = f(w * rho_{k-1}).

    No recurrence → the cascade is a pure composition of sigmoids.
    """
    out = np.zeros(n)
    out[0] = float(sigmoid.f(drive))
    for k in range(1, n):
        out[k] = float(sigmoid.f(w * out[k - 1]))
    return out


def run_5A(sigmoid: Sigmoid, drive: float = 1.5) -> dict:
    ns = [2, 3, 5, 10]
    ws = np.linspace(0.2, 4.0, 15)
    errs_per_n = {}
    rows = []
    for n in ns:
        stage_errs = []
        for w in ws:
            num = series_steady_state_numerical(n, float(w), drive, sigmoid)
            ana = series_steady_state_analytical(n, float(w), drive, sigmoid)
            # Report max over stages of the relative error.
            denom = np.maximum(np.abs(ana), 1e-9)
            rel = np.max(np.abs(num - ana) / denom)
            stage_errs.append(float(rel))
            rows.append((n, float(w), num[-1], ana[-1], float(rel)))
        errs_per_n[n] = np.array(stage_errs)
        print(
            f"  n={n}: max rel err over w-sweep = "
            f"{float(np.max(stage_errs)):.2e}, median = {float(np.median(stage_errs)):.2e}",
            flush=True,
        )
    # Plot final-stage amplitude vs weight, all n on one axis.
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(ns)))
    for n, c in zip(ns, colors):
        nums = [r[2] for r in rows if r[0] == n]
        anas = [r[3] for r in rows if r[0] == n]
        ax.plot(ws, anas, "-", color=c, linewidth=1.0, label=f"n={n} analytical")
        ax.plot(ws, nums, "o", color=c, markersize=3.5, label=f"n={n} numerical")
    ax.set_xlabel("chain weight w")
    ax.set_ylabel(r"steady-state activity $\rho_{n-1}^*$")
    ax.set_title("5A: series chain final-stage amplitude")
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, "series")
    plt.close(fig)

    max_err = float(max(np.max(v) for v in errs_per_n.values()))
    passed = max_err < 0.05
    return {
        "ns": ns,
        "weights": ws,
        "max_rel_err": max_err,
        "pass": passed,
        "per_n_max": {n: float(np.max(v)) for n, v in errs_per_n.items()},
    }


# ----------------------------------------------------------------------------
# 5B. Parallel composition
# ----------------------------------------------------------------------------


def run_5B(sigmoid: Sigmoid, drive: float = 1.5) -> dict:
    """Verify block-diagonal Jacobian structure of the parallel composition.

    With a driver population (index 0) feeding n downstream populations, the
    Wilson-Cowan Jacobian is block-lower-triangular: the driver block is a
    1x1 scalar -1/tau, the downstream block is diagonal (no cross-coupling),
    and the off-diagonal block captures the driver -> downstream coupling.
    The eigenvalues are therefore the union of {-1/tau} (from the driver)
    and {-1/tau} (one per downstream population), all equal to -1/tau
    exactly. We check this structure to machine precision.
    """
    rng = np.random.default_rng(SEED)
    ns = [2, 4, 8]
    reports = []
    for n in ns:
        w_in = rng.uniform(0.2, 2.5, size=n)
        W, Ivec = parallel_composition(w_in, n, drive=drive)
        rho_star, ok = find_fixed_point(
            W, Ivec, sigmoid, rho_guess=np.full(n + 1, 0.5),
        )
        assert ok, f"FP failed for parallel n={n}"
        # Expected analytical FP: driver rho_0 = f(drive); each downstream
        # rho_k = f(w_in[k-1] * rho_0).
        rho_drv = float(sigmoid.f(drive))
        rho_down = np.array([float(sigmoid.f(w_in[k] * rho_drv)) for k in range(n)])
        rho_pred = np.concatenate(([rho_drv], rho_down))
        fp_err = float(np.max(np.abs(rho_star - rho_pred)))

        # Jacobian structure check.
        J = jacobian(W, Ivec, rho_star, TAU, sigmoid)
        # Downstream-to-downstream block (rows/cols 1..n) should be
        # diagonal: no downstream-downstream cross terms exist in W.
        downstream_block = J[1:, 1:]
        off_diag_mag = float(np.max(np.abs(downstream_block - np.diag(np.diag(downstream_block)))))
        # Upper-triangular block (downstream -> driver) should be zero
        # (no feedback from downstream to driver).
        upper_block = J[0, 1:]
        upper_mag = float(np.max(np.abs(upper_block)))

        eigs, _ = spectrum(J)
        eigs_real = np.sort(eigs.real)[::-1]
        # Every eigenvalue should equal -1/tau exactly because each
        # diagonal block is -1/tau (no self-excitation anywhere).
        eig_err = float(np.max(np.abs(eigs_real - (-1.0 / TAU))))

        reports.append(
            dict(
                n=n,
                w_in=[float(x) for x in w_in],
                fp_err=fp_err,
                off_diag_mag=off_diag_mag,
                upper_mag=upper_mag,
                eig_err=eig_err,
                eigs_real=[float(x) for x in eigs_real],
            )
        )
        print(
            f"  n={n}: FP err={fp_err:.2e}, off-diag={off_diag_mag:.2e}, "
            f"upper={upper_mag:.2e}, eigval err={eig_err:.2e}",
            flush=True,
        )

    # Plot: downstream-block Jacobian heatmap for n=8 to visualize block structure.
    n_vis = 8
    w_in = rng.uniform(0.2, 2.5, size=n_vis)
    W, Ivec = parallel_composition(w_in, n_vis, drive=drive)
    rho_star, _ = find_fixed_point(W, Ivec, sigmoid, rho_guess=np.full(n_vis + 1, 0.5))
    J = jacobian(W, Ivec, rho_star, TAU, sigmoid)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(J, cmap="RdBu", aspect="equal",
                   vmin=-np.max(np.abs(J)), vmax=np.max(np.abs(J)))
    ax.set_title(f"5B: Jacobian of parallel composition (n={n_vis})")
    ax.set_xlabel("source population index")
    ax.set_ylabel("target population index")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save_fig(fig, "parallel_jacobian")
    plt.close(fig)

    overall_err = max(r["fp_err"] for r in reports)
    structure_ok = all(
        r["off_diag_mag"] < 1e-12 and r["upper_mag"] < 1e-12 and r["eig_err"] < 1e-12
        for r in reports
    )
    fp_ok = overall_err < 0.05
    passed = structure_ok and fp_ok
    return {
        "reports": reports,
        "max_fp_err": overall_err,
        "structure_ok": structure_ok,
        "pass": passed,
    }


# ----------------------------------------------------------------------------
# 5C. Positive loop
# ----------------------------------------------------------------------------


def enumerate_positive_loop_fps(
    w: float, drive: float, sigmoid: Sigmoid, n_samples: int = 2001,
) -> list[float]:
    """Enumerate fixed points of the symmetric positive loop via the
    scalar reduction rho = f(w * rho + drive).

    Symmetry forces rho_1 = rho_2 at every FP, so we reduce to a 1D
    problem g(rho) = rho - f(w rho + drive) = 0 and bracket all roots.
    Returns a sorted list of fixed-point activities in (0, 1).
    """
    from scipy.optimize import brentq

    def g(rho: float) -> float:
        return rho - float(sigmoid.f(w * rho + drive))

    xs = np.linspace(0.0, 1.0, n_samples)
    ys = np.array([g(x) for x in xs])
    roots: list[float] = []
    for i in range(len(xs) - 1):
        if ys[i] == 0.0:
            roots.append(float(xs[i]))
        elif ys[i] * ys[i + 1] < 0:
            try:
                r = brentq(g, float(xs[i]), float(xs[i + 1]), xtol=1e-12)
                roots.append(float(r))
            except Exception:
                pass
    return sorted(set(round(r, 10) for r in roots))


def run_5C(sigmoid: Sigmoid) -> dict:
    """Saddle-node bifurcation of the symmetric positive loop.

    Scan w from 0 to 10 with drive = 0 (bare loop). The FP count is 1 at
    weak coupling (low-active attractor) and jumps to 3 (low stable,
    middle saddle, high stable) when w crosses the saddle-node fold.
    The analytical saddle-node condition is tangency of y = rho to
    y = f(w rho), i.e.,

        rho = f(w rho)
        1 = w * f'(w rho) = w * k * f(w rho) * (1 - f(w rho))
          = w * k * rho * (1 - rho)

    The two equations give rho(w) on the fold. The onset weight
    w_c = min w such that two FPs merge.
    """
    from scipy.optimize import brentq

    # Analytical saddle-node: maximum of g(rho, w) = 0 touches zero.
    # At the fold, w and rho satisfy
    #   rho = f(w rho)
    #   w = 1 / (k rho (1 - rho))
    # We parameterize by rho in (0, 1) (excluding 0 and 1 where f' vanishes).
    rhos_grid = np.linspace(0.02, 0.98, 2001)
    fold_ws = 1.0 / (sigmoid.k * rhos_grid * (1 - rhos_grid))
    # Bounded below by w_min = 1 / (k * 0.25) = 4/k = 1 for k=4.
    # For each w >= w_c on this curve, drive must equal f^{-1}(rho) - w*rho.
    fold_drives = (
        sigmoid.theta - (1.0 / sigmoid.k) * np.log(1.0 / rhos_grid - 1.0)
    ) - fold_ws * rhos_grid
    # At drive = 0, two saddle-node points (upper and lower fold). Find w
    # values where fold_drives crosses zero.
    w_bifurcs = []
    for i in range(len(fold_drives) - 1):
        if fold_drives[i] * fold_drives[i + 1] < 0:
            w_star = brentq(
                lambda rho: (
                    sigmoid.theta - (1.0 / sigmoid.k) * np.log(1.0 / rho - 1.0)
                )
                - (1.0 / (sigmoid.k * rho * (1 - rho))) * rho,
                float(rhos_grid[i]),
                float(rhos_grid[i + 1]),
                xtol=1e-10,
            )
            w_bifurcs.append(float(1.0 / (sigmoid.k * w_star * (1 - w_star))))
    w_bifurcs = sorted(w_bifurcs)
    print(f"  Analytical saddle-node weights (drive=0): {w_bifurcs}", flush=True)

    # Numerical scan: FP count along w.
    ws_scan = np.linspace(0.1, 10.0, 400)
    fp_counts = []
    fp_all: list[tuple[float, list[float]]] = []
    for w in ws_scan:
        fps = enumerate_positive_loop_fps(float(w), drive=0.0, sigmoid=sigmoid)
        fp_counts.append(len(fps))
        fp_all.append((float(w), fps))
    fp_counts = np.array(fp_counts)
    # Numerical SN weight: first w where FP count changes from 1 to 3.
    trans_ws = []
    for i in range(len(fp_counts) - 1):
        if fp_counts[i] != fp_counts[i + 1]:
            trans_ws.append(
                0.5 * (float(ws_scan[i]) + float(ws_scan[i + 1]))
            )
    print(f"  Numerical FP-count transitions at w = {trans_ws}", flush=True)

    # Compare the (at least one) analytical vs numerical SN weight.
    if w_bifurcs and trans_ws:
        # Match each analytical value to its nearest numerical transition.
        matches = []
        for wa in w_bifurcs:
            wn = min(trans_ws, key=lambda x: abs(x - wa))
            matches.append((wa, wn, abs(wa - wn) / wa))
        for wa, wn, rel in matches:
            print(
                f"    analytical w_SN = {wa:.4f}, nearest numerical = {wn:.4f}, "
                f"rel err = {rel * 100:.2f}%",
                flush=True,
            )
        max_err = max(m[2] for m in matches)
    else:
        matches = []
        max_err = float("nan")

    # Bifurcation plot: FP activity vs w.
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    for w, fps in fp_all:
        for r in fps:
            ax.plot(w, r, "k.", markersize=1.5, alpha=0.7)
    for wa in w_bifurcs:
        ax.axvline(wa, color="red", linewidth=0.8, linestyle="--", label=f"SN (analytical)" if wa == w_bifurcs[0] else None)
    ax.set_xlabel("loop weight w")
    ax.set_ylabel(r"fixed-point activity $\rho^*$")
    ax.set_title("5C: positive-loop saddle-node (drive = 0)")
    ax.legend(fontsize=9)
    save_fig(fig, "positive_loop")
    plt.close(fig)

    passed = (not np.isnan(max_err)) and max_err < 0.05 and bool(w_bifurcs)
    return {
        "analytical_ws": w_bifurcs,
        "numerical_transitions": trans_ws,
        "matches": matches,
        "max_rel_err": max_err,
        "pass": passed,
    }


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------


def render_report(
    res_a: dict, res_b: dict, res_c: dict, overall_pass: bool,
) -> None:
    verdict = "PASS" if overall_pass else "PARTIAL"
    typ = HERE / "phase5_report.typ"
    pdf = HERE / "phase5_report.pdf"

    per_n = res_a["per_n_max"]
    per_n_rows = "\n".join(
        [f"  [$n = {n}$], [${e:.3e}$]," for n, e in per_n.items()]
    )

    b_rows = "\n".join(
        [
            f"  [$n = {r['n']}$], [${r['fp_err']:.2e}$], "
            f"[${r['off_diag_mag']:.2e}$], [${r['upper_mag']:.2e}$], "
            f"[${r['eig_err']:.2e}$],"
            for r in res_b["reports"]
        ]
    )

    if res_c["matches"]:
        c_rows = "\n".join(
            [
                f"  [${wa:.4f}$], [${wn:.4f}$], [${rel * 100:.3f}%$],"
                for (wa, wn, rel) in res_c["matches"]
            ]
        )
    else:
        c_rows = "  [-], [-], [-],"

    content = rf"""#set page(paper: "a4", margin: 2cm)
#set text(size: 10pt)
#set par(justify: true)

#align(center)[
  #text(size: 14pt, weight: "bold")[Phase 5 report -- Generalization to
  other archetypes]
  #v(0.2em)
  Verdict: *{verdict}*
]

= Subtask 5A -- Simple series chain

An $n$-population feed-forward chain with equal inter-stage weight $w$
and drive only on stage 0 has a recursive steady state
$rho_0 = f(I)$, $rho_k = f(w dot rho_(k-1))$. With no recurrence the
Jacobian is triangular and all eigenvalues equal $-1 slash tau$, so
convergence to the steady state is exponential and free of oscillation;
this is the natural population-level analogue of a feed-forward gain
cascade.

The table below reports the maximum relative error between the
analytical recursion and the numerical steady state over a weight sweep
$w in [0.2, 4.0]$ and chains of lengths $n in {{2, 3, 5, 10}}$.

#table(columns: 2,
  [chain length], [max rel. error over $w$-sweep],
{per_n_rows}
)

Acceptance (5 % relative error): *{"PASS" if res_a["pass"] else "FAIL"}*
(overall max = ${res_a['max_rel_err']:.3e}$).

#figure(image("results/phase5/series.pdf", width: 75%),
  caption: [5A. Final-stage activity $rho_{{n - 1}}^*$ versus chain weight
  $w$ for chains of lengths 2, 3, 5, 10. Solid lines are the analytical
  recursive-sigmoid prediction; open dots are the numerical steady
  states. The curves overlap within line width.])

= Subtask 5B -- Parallel composition

A driver population feeding $n$ independently-weighted downstream
populations produces a block-triangular Jacobian: the driver's $1 times 1$
self-block is $-1 slash tau$, the downstream $n times n$ block is
diagonal (no cross-coupling), and the only non-zero off-diagonal block
is the driver $arrow$ downstream coupling. Every eigenvalue therefore
equals $-1 slash tau$ exactly, independent of the gain vector
$(w_(i))_(i=1)^(n)$. The fixed point is closed-form:
$rho_0 = f(I)$, $rho_k = f(w_k rho_0)$.

#table(columns: 5,
  [$n$], [FP $L^(oo)$ error], [downstream off-diag mag.],
  [downstream $arrow$ driver coupling], [eigenvalue error],
{b_rows}
)

Acceptance (structure to machine precision, FP within 5 %):
*{"PASS" if res_b["pass"] else "FAIL"}*.

#figure(image("results/phase5/parallel_jacobian.pdf", width: 60%),
  caption: [5B. Jacobian of a parallel composition with $n = 8$ downstream
  populations. The block-triangular structure is visible: the driver's
  row (row 0) depends only on itself; each downstream row (rows 1--8) has
  a non-zero entry only on its own diagonal and on column 0.])

= Subtask 5C -- Positive loop saddle-node bifurcation

Two mutually exciting populations with $w_(12) = w_(21) = w$ and zero
drive admit a symmetric scalar reduction $rho = f(w rho)$. The
saddle-node fold of the FP curve occurs when the slope condition
$w k rho(1 - rho) = 1$ is met simultaneously with the FP equation.
Eliminating $rho$ gives the analytical saddle-node weight(s) at
$I = 0$, tabulated below against a numerical FP-count scan.

#table(columns: 3,
  [analytical $w_("SN")$], [numerical transition], [rel. error],
{c_rows}
)

Acceptance (5 % relative error): *{"PASS" if res_c["pass"] else "FAIL"}*.

#figure(image("results/phase5/positive_loop.pdf", width: 75%),
  caption: [5C. Fixed-point branches of the symmetric positive loop as
  functions of the loop weight $w$, at zero drive. A single low-active
  branch persists for small $w$; above the analytical saddle-node weight
  (red dashed line) a high-active branch and a middle saddle appear.])

= Verdict

- 5A series chain: {"PASS" if res_a["pass"] else "FAIL"}
- 5B parallel composition: {"PASS" if res_b["pass"] else "FAIL"}
- 5C positive loop saddle-node: {"PASS" if res_c["pass"] else "FAIL"}

Overall: *{verdict}*.
"""
    typ.write_text(content)
    subprocess.run(["typst", "compile", str(typ), str(pdf)], check=True, cwd=str(HERE))


def main() -> int:
    sigmoid = Sigmoid(k=SIGMOID_K, theta=SIGMOID_THETA)

    banner("Phase 5  generalization to other archetypes")

    banner("Subtask 5A  simple series chain")
    t0 = time.time()
    res_a = run_5A(sigmoid)
    print(f"5A done in {time.time() - t0:.2f}s; pass={res_a['pass']}", flush=True)

    banner("Subtask 5B  parallel composition")
    t0 = time.time()
    res_b = run_5B(sigmoid)
    print(f"5B done in {time.time() - t0:.2f}s; pass={res_b['pass']}", flush=True)

    banner("Subtask 5C  positive-loop saddle-node")
    t0 = time.time()
    res_c = run_5C(sigmoid)
    print(f"5C done in {time.time() - t0:.2f}s; pass={res_c['pass']}", flush=True)

    overall_pass = res_a["pass"] and res_b["pass"] and res_c["pass"]
    render_report(res_a, res_b, res_c, overall_pass)
    banner(f"Phase 5 verdict: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
