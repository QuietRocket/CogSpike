// Phase 1 report: Siegert FP + Jacobian on the negative-loop grid
#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 14pt, weight: "bold")[
    Phase 1 — Siegert FP + Jacobian eigenvalues
  ]
  #v(0.2em)
  #text(size: 10pt, style: "italic")[
    `deq/closed_form_neg_loop/`, May 2026
  ]
]

= Goal

For each $(w_(I A), w_(X A))$ cell on the Phase 0 grid, solve the
2-population Siegert fixed point on the negative-loop topology and
compute the Jacobian eigenvalues of the rate-equation linearization at
that FP. Classify cells by FP stability + ringing.

= Setup

- *Calibration* (locked from `closed_form/results/phase1_grid.npz`):
  $alpha = 0.250$, $beta = 0.00429$, $tau_m = 2.350$, $tau_("ref") = 0.361$, $R^2 = 0.936$.
- *Topology* (rate version): $A$ sees external Bernoulli drive
  $w_(X A) dot p_("thin")$ ($p_("thin") = 0.7$ matches calibration) plus
  $w_(I A) dot nu_I$; $I$ sees $w_(A I) dot nu_A$ only.
- *FP solver*: `scipy.optimize.fsolve` on the 2-D self-consistency
  $bold(nu) = bold(Phi)(bold(mu)(bold(nu)), bold(sigma)(bold(nu)))$,
  seeded from 5 initial guesses.
- *Jacobian*: $A = (1 slash tau_m) (-I + "diag"(g) J)$, with
  $J = alpha mat(0, w_(I A); w_(A I), 0)$ and
  $g_i = (d Phi / d mu)|_("FP")$ (from `transfer.dphi_dmu`).

= Closed-form structure

For this 2x2 antidiagonal $J$, the eigenvalues are

$ lambda_(plus.minus) = 1 / tau_m (-1 plus.minus sqrt(g_A g_I w_(A I) w_(I A))) $

Since $w_(I A) < 0 < w_(A I)$ and gains are positive, the radicand is
*negative*, so $lambda_(plus.minus)$ is a complex-conjugate pair with

$ "Re"(lambda) = -1 / tau_m approx -0.426,quad "Im"(lambda) = (1 / tau_m) sqrt(|g_A g_I w_(A I) w_(I A)|). $

The FP is therefore *always a stable spiral* (Re negative, Im nonzero)
unless the gains degenerate at $nu = 0$ or $nu = 1$. *Rate theory
cannot predict sustained oscillation* — Property 5's limit cycle lives
strictly beyond mean-field reach.

= Result

#figure(image("results/phase1/hopf_vs_fcs.pdf", width: 100%),
  caption: [Three labellings on the same grid. *Left:* FCS strict
  Property 5 (445/1600). *Middle:* FCS broad oscillation (946/1600).
  *Right:* Siegert spiral-blue, where the FP exists and Im(λ) ≠ 0
  (1440/1600 = 90%). The smooth-rate envelope is much wider than
  Property 5; almost every cell looks like a ringing spiral to
  mean-field theory. Gold ring: FCS default cell $(-11, 11)$.])

#table(
  columns: (auto, auto, auto),
  table.header([*Label*], [*Cells*], [*Jaccard vs Siegert spiral*]),
  [Siegert spiral (Im ≠ 0)], [1440 / 1600], [—],
  [FCS strict Property 5], [445 / 1600], [0.309],
  [FCS broad oscillation], [946 / 1600], [0.657],
)

The 0.31 Jaccard against strict Property 5 quantifies the
mean-field-over-prediction: rate theory says "everywhere rings", but
only ~28% of cells actually oscillate at exactly period 4. Against the
looser broad-oscillation label, Jaccard climbs to 0.66 — the Siegert
spiral envelope is a decent *upper bound* on where oscillation can
happen, but it can't pick out which specific period FCS will lock.

The 160 non-spiral cells (10%) are saturation regions: $w_(X A)$
strongly dominates $|w_(I A)|$, $A$ fires every tick, gains degenerate
at $nu = 1$ — Phase 1 marks these as "no spiral" because the rate
equation has no analytic linearization there.

#figure(image("results/phase1/im_lambda_heatmap.pdf", width: 80%),
  caption: [Heatmap of Im(λ) — the predicted ringing rate (rad / tick).
  Brighter is faster ringing. The ringing rate grows with both
  $|w_(I A)|$ and $w_(A I) dot w_(X A)$ (which both raise the gain
  product $g_A g_I w_(A I) w_(I A)$).])

#figure(image("results/phase1/eig_spectrum.pdf", width: 75%),
  caption: [Jacobian eigenvalue spectrum across all converged cells.
  Re(λ) is locked at $-1/tau_m approx -0.426$ for every complex cell
  (this is the rigid prediction of single-pole low-pass H(ω)); only
  Im(λ) varies. Blue dots: cells where FCS strict Property 5 holds.])

= Sanity gate

FCS default cell $(w_(I A), w_(X A)) = (-11, 11)$:

#align(center)[
  $bold(nu)^star = (0.3520, 0.1793),quad
   lambda = -0.426 plus.minus 0.395 i$
]

Stable spiral, predicted ringing period
$T_("pred") = 2 pi slash |"Im"(lambda)| approx 15.9$ ticks — about
$4 times$ the FCS-measured period of $4$. *Phase 1 PASS gate* (FP
exists, eigenvalues complex), but Phase 2 will quantify this factor-of-4
period gap across the entire grid: the single-pole rate model predicts
the *existence* of ringing but systematically *over-estimates* its
period relative to discrete-tick FCS dynamics.

= Verdict

*Phase 1 PASS gate.* The rate-equation FP is a stable spiral
everywhere outside the saturation band; mean-field theory cannot
predict Property 5's sustained oscillation but it does provide a
ringing-frequency estimate at every cell. The 0.31 Jaccard against
strict Property 5 documents that smooth-rate theory has the *wrong
class* of attractor here — Property 5 is a discrete-tick limit cycle,
not a continuous-time Hopf bifurcation. Phase 2 turns the Im(λ)
estimates into period predictions and checks them against FCS's
measured period.

#v(0.6em)

#text(size: 9pt, fill: luma(80))[
  Output: `results/phase1/{siegert_grid.npz, hopf_vs_fcs.pdf,
  eig_spectrum.pdf, im_lambda_heatmap.pdf}`, `results/phase1.log`.
]
