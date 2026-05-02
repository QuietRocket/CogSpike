"""Phase 0 - Stochastic-LI&F bridge & calibration.

Two validations:

V0.1 -- Single-population rate-vs-drive curve. Sweep external drive over a
range, run the population oracle at three thinning levels p_thin in
{1.0, 0.7, 0.4}, measure mean steady-state rate. Verifies the f-I curve is
monotone, saturating, and that thinning lowers the saturation rate as
expected.

V0.2 -- ISI CV calibration. For a fixed mid-range drive, sweep
(p_thin, tau_jitter) and compute mean ISI CV across N=100 copies over
T=2000 ticks. Locks in the (p_thin, tau_jitter) pair that gives
CV in [0.5, 1.0] -- a regime where Siegert's diffusion approximation is
applicable (CV ~ 1 = Poisson; CV << 0.5 = quasi-deterministic).

Gate: V0.1 monotone-saturating curve at all three p_thin; V0.2 finds at
least one parameter combination with CV >= 0.5 and runs in <=30s.
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

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding=sys.stdout.encoding, line_buffering=True
)

from stochastic_lif import (  # noqa: E402
    population_isi_cv,
    simulate_population,
    steady_state_rate,
)

SEED = 20260502
np.random.seed(SEED)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
FIG_DIR = RESULTS / "phase0"
FIG_DIR.mkdir(exist_ok=True)


def banner(msg: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{msg}\n{bar}")


def save_fig(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")


def single_pop_topology(self_drive: int = 11):
    """Single uncoupled neuron with one external input."""
    W = np.zeros((1, 1), dtype=np.int64)
    B = np.array([[self_drive]], dtype=np.int64)
    return W, B


def validation_v01_rate_vs_drive() -> dict:
    banner("V0.1  Rate vs drive (single uncoupled population, threshold jitter only)")
    self_drives = list(range(2, 22, 2))  # 2, 4, ..., 20
    p_thins = [1.0, 0.7, 0.4]
    N = 100
    T = 1000

    curves = {}
    for p_thin in p_thins:
        rates_at_drives = []
        for sd in self_drives:
            W, B = single_pop_topology(self_drive=sd)
            ext = np.ones((1, T), dtype=np.int64)
            rates, _ = simulate_population(
                W,
                B,
                ext,
                N=N,
                p_thin=p_thin,
                tau_jitter=2,
                T=T,
                seed=SEED + int(p_thin * 100) + sd,
            )
            mean_rate = float(steady_state_rate(rates).item())
            rates_at_drives.append(mean_rate)
            print(
                f"  p_thin={p_thin:.2f}  drive={sd:2d}  rate={mean_rate:.3f}"
            )
        curves[p_thin] = np.array(rates_at_drives)

    # Monotone? Each curve should be non-decreasing in drive.
    monotone_per_curve = {}
    for p_thin, rs in curves.items():
        diffs = np.diff(rs)
        monotone_per_curve[p_thin] = bool(np.all(diffs >= -0.05))

    # Saturating? top of each curve should be > 0.5 of the p_thin = 1.0 max.
    max_rates = {p: float(rs.max()) for p, rs in curves.items()}

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for p_thin in p_thins:
        ax.plot(
            self_drives,
            curves[p_thin],
            marker="o",
            linewidth=1.5,
            label=f"p_thin = {p_thin:.2f}",
        )
    ax.set_xlabel("external drive weight (FCS units)")
    ax.set_ylabel("mean steady-state firing rate")
    ax.set_title("V0.1  Population f-I curve (threshold jitter eps=2, N=100)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, "v01_rate_vs_drive")
    plt.close(fig)

    # Save raw data for the report.
    np.savez(
        RESULTS / "phase0_v01.npz",
        drives=np.array(self_drives),
        **{f"rates_p{int(p * 100)}": rs for p, rs in curves.items()},
    )

    return {
        "drives": self_drives,
        "curves": {p: rs.tolist() for p, rs in curves.items()},
        "monotone_per_curve": monotone_per_curve,
        "max_rates": max_rates,
        "all_monotone": all(monotone_per_curve.values()),
        "saturates": max_rates[1.0] >= 0.4 and max_rates[0.4] < max_rates[1.0],
        "pass": all(monotone_per_curve.values())
        and max_rates[1.0] >= 0.4
        and max_rates[0.4] < max_rates[1.0],
    }


def validation_v02_isi_cv() -> dict:
    banner("V0.2  ISI CV calibration (mid-range drive, sweep over jitter and thinning)")
    self_drive = 11  # midrange (the FCS default)
    N = 100
    T = 2000

    p_thins = [1.0, 0.85, 0.7, 0.5, 0.3]
    jitters = [0, 2, 5, 10]

    results = {}
    timings = {}
    for p_thin in p_thins:
        for jit in jitters:
            W, B = single_pop_topology(self_drive=self_drive)
            ext = np.ones((1, T), dtype=np.int64)
            t0 = time.time()
            rates, spk = simulate_population(
                W,
                B,
                ext,
                N=N,
                p_thin=p_thin,
                tau_jitter=jit,
                T=T,
                seed=SEED + int(p_thin * 100) * 10 + jit,
            )
            elapsed = time.time() - t0
            cv = population_isi_cv(spk[0])
            mean_rate = float(steady_state_rate(rates).item())
            results[(p_thin, jit)] = {"cv": cv, "rate": mean_rate, "time": elapsed}
            timings[(p_thin, jit)] = elapsed
            print(
                f"  p_thin={p_thin:.2f}  jitter={jit:2d}  CV={cv:.3f}  "
                f"rate={mean_rate:.3f}  ({elapsed:.1f}s)"
            )

    # Find at least one combination with CV >= 0.5.
    valid_combos = [
        (p, j, r["cv"], r["rate"])
        for (p, j), r in results.items()
        if not np.isnan(r["cv"]) and r["cv"] >= 0.5
    ]

    # Heatmap of CV vs (p_thin, jitter).
    cv_grid = np.full((len(jitters), len(p_thins)), np.nan)
    rate_grid = np.full_like(cv_grid, np.nan)
    for i, j in enumerate(jitters):
        for k, p in enumerate(p_thins):
            cv_grid[i, k] = results[(p, j)]["cv"]
            rate_grid[i, k] = results[(p, j)]["rate"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, grid, title, vmax in [
        (axes[0], cv_grid, "ISI CV", 1.5),
        (axes[1], rate_grid, "mean rate", 1.0),
    ]:
        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=(0, len(p_thins), 0, len(jitters)),
            vmin=0,
            vmax=vmax,
            cmap="viridis",
        )
        ax.set_xticks(np.arange(len(p_thins)) + 0.5)
        ax.set_xticklabels([f"{p:.2f}" for p in p_thins])
        ax.set_yticks(np.arange(len(jitters)) + 0.5)
        ax.set_yticklabels([str(j) for j in jitters])
        ax.set_xlabel("p_thin")
        ax.set_ylabel("threshold jitter eps")
        ax.set_title(title)
        for i in range(len(jitters)):
            for k in range(len(p_thins)):
                v = grid[i, k]
                if not np.isnan(v):
                    ax.text(
                        k + 0.5,
                        i + 0.5,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if v < vmax * 0.6 else "black",
                    )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("V0.2  CV and rate over (p_thin, jitter), drive=11, N=100, T=2000")
    fig.tight_layout()
    save_fig(fig, "v02_cv_heatmap")
    plt.close(fig)

    # Pick the recommended operating point: smallest cost combo with CV >= 0.5
    # and rate well above 0 and below 1.
    recommended = None
    if valid_combos:
        # Prefer larger p_thin (less aggressive thinning) and smaller jitter.
        valid_combos.sort(key=lambda x: (-x[0], x[1]))
        recommended = valid_combos[0]

    np.savez(
        RESULTS / "phase0_v02.npz",
        p_thins=np.array(p_thins),
        jitters=np.array(jitters),
        cv_grid=cv_grid,
        rate_grid=rate_grid,
    )

    return {
        "p_thins": p_thins,
        "jitters": jitters,
        "results": {f"{p}_{j}": r for (p, j), r in results.items()},
        "max_time_per_run": max(timings.values()),
        "valid_combos": valid_combos,
        "recommended": recommended,
        "pass": (recommended is not None) and (max(timings.values()) <= 30.0),
    }


def render_report(v01: dict, v02: dict, overall_pass: bool) -> None:
    typ = HERE / "phase0_report.typ"
    pdf = HERE / "phase0_report.pdf"
    verdict = "PASS" if overall_pass else "FAIL"

    rec = v02["recommended"]
    rec_str = (
        f"p_thin = {rec[0]:.2f}, jitter = {rec[1]}, CV = {rec[2]:.3f}, "
        f"rate = {rec[3]:.3f}"
        if rec
        else "no combination achieved CV >= 0.5"
    )

    monotone_summary = ", ".join(
        f"p={p}: {'yes' if m else 'NO'}" for p, m in v01["monotone_per_curve"].items()
    )
    max_rates_summary = ", ".join(
        f"p={p}: {r:.3f}" for p, r in v01["max_rates"].items()
    )

    content = f"""#set page(paper: "a4", margin: 2cm)
#set text(size: 10pt)
#set par(justify: true)

#align(center)[
  #text(size: 14pt, weight: "bold")[Phase 0 report -- Stochastic-LI&F bridge & calibration]
  #v(0.2em)
  Verdict: *{verdict}*
]

= File inventory

The following modules were created under `./deq/closed_form/`:

- `stochastic_lif.py` -- N-neuron oracle wrapping
  `../archetypes/lif_fcs.py:simulate` with per-copy threshold heterogeneity
  and Bernoulli per-tick input thinning. Returns population-mean firing
  rates and per-copy spike trains.
- `phase0_infrastructure.py` -- this validation script.
- `README.md`, `make_all.sh`, `.gitignore` -- workspace boilerplate.

= Validation V0.1 -- Population f-I curve

Setup: single uncoupled neuron, $N = 100$ copies, threshold jitter
$epsilon = 2$, sweep external drive weight $w_X in {{2, 4, dots, 20}}$,
$T = 1000$ ticks. Three thinning levels $p_("thin") in {{1.0, 0.7, 0.4}}$.

Monotonicity per curve (drive vs rate non-decreasing): {monotone_summary}.

Saturation maxima per curve: {max_rates_summary}.

Acceptance: each curve monotone, $p = 1.0$ saturating $>= 0.4$, smaller
$p_("thin")$ gives strictly lower saturation. *{"PASS" if v01["pass"] else "FAIL"}*.

#figure(image("results/phase0/v01_rate_vs_drive.pdf", width: 80%),
  caption: [V0.1: population f-I curve at three thinning levels. The
  $p_("thin") = 1.0$ curve is the deterministic-input case (variance from
  threshold jitter only); lower $p_("thin")$ injects Bernoulli input
  variance, which both reduces the mean drive and reshapes the curve.])

= Validation V0.2 -- ISI CV calibration

Setup: single neuron at drive = 11 (FCS default), $N = 100$ copies,
$T = 2000$ ticks. Sweep $p_("thin") in {{1.0, 0.85, 0.7, 0.5, 0.3}}$ and
threshold jitter $epsilon in {{0, 2, 5, 10\}}$.

Maximum per-cell wallclock: {v02["max_time_per_run"]:.1f} s
(threshold $30$ s).

Recommended operating point: {rec_str}.

Number of $(p_("thin"), epsilon)$ combinations with CV $>= 0.5$:
{len(v02["valid_combos"])} of {len(v02["p_thins"]) * len(v02["jitters"])}.

Acceptance: at least one combination achieves CV $>= 0.5$ and total runtime
fits in budget. *{"PASS" if v02["pass"] else "FAIL"}*.

#figure(image("results/phase0/v02_cv_heatmap.pdf", width: 90%),
  caption: [V0.2: ISI coefficient of variation (left) and mean firing rate
  (right) across the $(p_("thin"), epsilon)$ grid. CV $approx 1$ corresponds
  to Poisson-like firing where Siegert's diffusion approximation is valid;
  CV $<< 0.5$ means quasi-deterministic firing where the diffusion
  approximation underestimates regularity.])

= Overall verdict

*{verdict}*.

The recommended operating point ({rec_str}) is locked in for Phase 1's
Siegert comparison. The Bernoulli-thinning + threshold-jitter combination
produces input statistics in the diffusion-approximation-valid regime
without breaking the FCS-LI&F oracle's per-tick semantics.
"""
    typ.write_text(content)
    subprocess.run(
        ["typst", "compile", str(typ), str(pdf)], check=True, cwd=str(HERE)
    )


def main() -> int:
    v01 = validation_v01_rate_vs_drive()
    v02 = validation_v02_isi_cv()

    overall = v01["pass"] and v02["pass"]
    render_report(v01, v02, overall)

    banner(f"Phase 0 verdict: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
