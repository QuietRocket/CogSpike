// Phase 3 (multi-N): Quasi-renewal at finite N_pop.
#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 14pt, weight: "bold")[
    Phase 3 (multi-N) --- Quasi-renewal at finite $N_("pop")$
  ]
  #v(0.2em)
  #text(size: 10pt, style: "italic")[
    `deq/closed_form_wta_multi/`, May 2026
  ]
]

= Goal

Test whether the *quasi-renewal mesoscopic equation*
@NaudGerstner2012 at finite population $N_("pop")$ converges to the
Siegert mean-field as $N_("pop") arrow infinity$ for $N_("neurons") > 2$,
and how the finite-size broadening of the no-WTA band depends jointly on
$N_("neurons")$ and $N_("pop")$.

= Method

For each cell $(w, N_("neurons"), N_("pop"))$ with $"drive_bump" = 1$
fixed (the regime where smooth-rate theory is meaningful; $"drive_bump" = 0$
is uniformly red at every $N$ per Phase 0), simulate the QR equation
for $T = 200$ ticks with strong neuron-0-favored initial condition
$bold(A)(0) = (0.95, 0.005, ..., 0.005)$. WTA-blue iff post-warmup
($t in [50, 200]$) `rate_max` $-$ `second_max` $>= 0.30$.

#text(size: 9pt, fill: luma(80))[
  Initial-condition note: an initial-condition asymmetry of $0.4$ (the
  2-neuron `(0.5, 0.1)` from `simulate_contralateral`) is sufficient at
  $N = 2$ but too weak at $N >= 3$. Each loser at higher $N$ feels
  $(N - 1)$ competing winners' inhibition while the winner feels only
  $(N - 1)$ losers; the symmetric basin attracts unless the IC is
  strongly off-axis. Using $(0.95, 0.005, ..., 0.005)$ consistently
  across $N$ both reproduces the 2-neuron behaviour and breaks into
  the asymmetric basin at high $N$.
]

= Result

#figure(
  image("results/phase3/qr_n_sweep_multi.pdf", width: 99%),
  caption: [Phase 3 QR finite-$N_("pop")$ sweep, `drive_bump = 1`. Rows
  are $N_("neurons") in {2, 3, 4, 6, 10}$. Columns left to right: FCS
  oracle (margin), Siegert mean-field, then QR at $N_("pop") in
  {50, 100, 500, 2000}$. At every $N_("neurons")$, QR converges to the
  Siegert label pattern as $N_("pop") arrow infinity$.]
)

#figure(
  image("results/phase3/qr_jaccard_vs_Npop_per_N.pdf", width: 96%),
  caption: [Left: Jaccard(QR, FCS-margin) vs $N_("pop")$, one curve per
  $N_("neurons")$. Right: Jaccard(QR, Siegert) vs $N_("pop")$. The right
  panel shows clean convergence to the Siegert mean-field at every
  $N_("neurons")$; the left panel confirms that mean-field convergence
  does not bridge the FCS staircase / inverse-staircase gap.]
)

== Mean-field convergence headline (drive_bump=1, $N_("pop") = 2000$)

#table(
  columns: 4,
  inset: 5pt,
  table.header([*$N_("neurons")$*], [*QR-blue / 40*], [*J(QR, FCS)*], [*J(QR, Siegert)*]),
  [2],  [30], [0.290], [*1.000*],
  [3],  [26], [0.296], [*0.897*],
  [4],  [24], [0.111], [*0.857*],
  [6],  [22], [0.154], [*0.880*],
  [10], [18], [0.000], [*0.818*],
)

Mean $J("QR"_("N_pop=2000"), "Siegert")$ across $N_("neurons")$ is *0.89*,
above the 0.70 pass-gate.

== $N_("pop")$-broadening: finite-size loss of WTA cells

#table(
  columns: 6,
  inset: 5pt,
  table.header([*$N_("neurons")$*], [*QR N=50*], [*QR N=100*], [*QR N=500*], [*QR N=2000*], [*Siegert*]),
  [2],  [28], [30], [30], [30], [30],
  [3],  [7],  [18], [25], [26], [29],
  [4],  [2],  [14], [22], [24], [28],
  [6],  [0],  [9],  [20], [22], [25],
  [10], [0],  [2],  [17], [18], [22],
)

At $N_("neurons") = 2$, QR sees 28 / 30 blue already at $N_("pop") = 50$
--- the asymmetric basin is wide enough to survive $sqrt(A / 50)$ noise.
At $N_("neurons") = 10$, QR sees 0 blue at $N_("pop") = 50$ and only
2 / 22 at $N_("pop") = 100$. *The required $N_("pop")$ for mean-field
recovery grows with $N_("neurons")$*: each additional competitor
amplifies the noise-induced symmetry-restoration.

= Verdict

*Phase 3 PASS.* QR converges to the Siegert mean-field for every
$N_("neurons") in {2, 3, 4, 6, 10}$ at $N_("pop") = 2000$ (mean
Jaccard $0.89$). The expected finite-$N_("pop")$ broadening of the
no-WTA band is reproduced and shows an additional $N_("neurons")$-dependence:

#enum(
  [Mean-field convergence holds at every $N_("neurons")$ — the
   *rate equation is still the right large-$N_("pop")$ object* at
   any $N_("neurons")$.],
  [Finite-$N_("pop")$ broadening *amplifies with $N_("neurons")$*: at
   $N_("neurons") = 10$ we need $N_("pop") >= 500$ to recover most
   Siegert-blue cells, vs $N_("pop") >= 100$ at $N_("neurons") = 2$.],
  [Neither QR nor Siegert recovers the FCS-integer-tick-only WTA cells
   at $N_("neurons") = 10$ (e.g., $w in {-15, -16, -7}$): rate
   equations and their stochastic mesoscopic approximations *cannot
   see* what integer-tick LI&F semantics resolve. The inverse-staircase
   phenomenon is genuinely beyond rate-equation theory.],
)

#v(0.6em)

#bibliography("note/refs.bib", title: none, style: "ieee")

#v(0.4em)

#text(size: 9pt, fill: luma(80))[
  Output: `results/phase3/qr_grid_multi.npz`,
  `qr_n_sweep_multi.pdf`, `qr_jaccard_vs_Npop_per_N.pdf`,
  `results/phase3.log`.
]
