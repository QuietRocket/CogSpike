#import "../report_style.typ": *
#show: setup

#report_header("e06", "The ring attractor that holds the running context",
  "Tier 2 · recurrent predictor stage · paper §\"Attractors: holding a graded context\" + FIX G / BIO-06")

#claim[
  The running context $c_t$ — *which previous symbol* — is held as a bump of
  *graded persistent activity* on a line/ring attractor (Seung 1996; Ben-Yishai 1995):
  the bump's position along the manifold *is* the stored value. FIX G (BIO-06) is the
  paper's honesty patch: such an attractor does *not* store arbitrarily deep history.
  It is a *single* graded analog value, finite and noise-limited, with capacity
  $ C approx log_2("SNR") quad "bits", quad "SNR" = ("usable range") slash ("positional jitter"). $
  For the $4$-symbol rover we only need $C >= log_2 4 = 2$ bits. The spiking-reality
  gap vs the `numpy` validator (which sets $c_t = e_i$ *exactly and instantaneously*):
  here $c_t$ is an *analog* bump on a real spiking ring that takes *time* to write and
  settle, and *drifts* while it holds. We make all three measurable: hold fidelity,
  write/settle latency, and drift $arrow.r$ effective capacity.
]

= What we built

#method[
  *A 2-D ring attractor.* Symbol $s in {0,1,2,3}$ is the angle $theta_s = s dot 2pi
  slash 4$ on the unit circle, represented in a 2-D NEF ensemble ($700$ LIF neurons,
  radius $1.4$) as the point $(cos theta, sin theta)$. The recurrent feedback decodes
  the *radial-projection* map $f(x) = x slash |x|$ through a relay ensemble
  (`Connection(ens, relay, function=f)` $arrow.r$ `Connection(relay, ens,
  synapse=tau)`, $tau = 100$ ms). Under the NEF integrator identity $dot(x) approx
  (f(x) - x) slash tau$, this map restores the radius to $1$ while leaving the *angle
  untouched* — so *every* unit-norm point is a fixed point: a *continuous ring*.

  *Clear-then-write load.* A brief additive write current cannot overpower an
  already-latched bump — for an antipodal target the new input *cancels* the old bump
  radially, the state passes through the origin (where the ring angle is undefined),
  and the recurrence snaps it back (we observed exactly this failure first). So we
  load a new item the way attractor working-memory models do: a $20$ ms *clear*
  (strong inhibition of memory $+$ relay) wipes the old bump to zero, then a $40$ ms
  *write* builds the new bump *from rest* — precisely the regime in which the ring
  faithfully holds an arbitrary angle. This clear-then-write is the spiking
  realization of the coder's per-symbol `new_window` load of $c_t$. Window layout:
  $20$ ms clear, $40$ ms write, $140$ ms hold $arrow.r$ inter-symbol interval (ISI)
  $= 200$ ms. All code in `e06_attractor/run.py`; $delta t = 10^(-3)$.
]

= The ring is a genuine continuous attractor

#finding[
  With no input, the ring holds an *arbitrary* written angle — not just the four
  symbol vertices — to within a maximum of *$5.7 degree$* (mean $2.4 degree$) over a
  $300$ ms hold, across ten probe angles spanning the circle ($0 degree$ to $340
  degree$). This is the defining property of a *continuous ring of fixed points*
  (graded persistent activity), not a handful of discrete wells: the manifold itself
  is the memory, and the stored value is graded.
]

= (1) Hold fidelity: the written symbol survives the inter-symbol interval

#finding[
  After writing each symbol and holding with *no input*, the decoded context still
  indicates the written symbol for *$100%$* of the hold samples (all four symbols),
  with the bump sitting essentially on its target:

  #table(columns: 4, align: (center,) * 4, stroke: 0.5pt + luma(200),
    table.header[symbol][$theta$][mean $|"err"|$ over hold][decoded-correct fraction],
    [U], [$0 degree$],   [$0.7 degree$], [$1.000$],
    [D], [$90 degree$],  [$1.9 degree$], [$1.000$],
    [L], [$180 degree$], [$0.1 degree$], [$1.000$],
    [R], [$270 degree$], [$1.5 degree$], [$1.000$],
  )

  The half-cell decode margin is $45 degree$ ($= pi slash N$): a held angle decodes to
  the correct symbol while its error stays under $45 degree$. Across all four symbols,
  *$100%$* of hold samples are within margin — the ring holds the written context
  cleanly across the whole $140$ ms ISI hold.
]

#figure(image("results/e06_bump_hold.pdf", width: 86%),
  caption: [The bump-hold trace: a $6$-symbol stream ($"U L D R U D"$) written one symbol
    per window through *one continuous* ring simulation. The decoded bump angle (blue)
    jumps to each newly written symbol (red bars = written target per window) and
    *holds* it flat across the inter-symbol interval. Per-window hold decode error:
    *$0 slash 6$*.])

#intuition[
  This is the recurrent loop the paper draws as "context $c_t$: a bump on a line/ring
  attractor — ONE held value." The bump *is* $c_t$. At each window the clear-then-write
  loads the previous symbol; between windows the ring holds it, so the downstream
  predictor $q(dot mid(|) c_t)$ reads a stable context for the entire symbol.
]

= (2) Write/settle latency is far inside the ISI

#finding[
  After a window starts, the bump arrives at and *stays within* the half-cell margin
  of the new target at $33$–$35$ ms (U $33$, D $34$, L $33$, R $35$) — the write pulse
  ends at $60$ ms and the inter-symbol interval is $200$ ms. The *maximum* settle time,
  $35$ ms, is $5.7times$ shorter than the ISI: the context is loaded and stable long
  before the symbol it conditions is emitted. The write/settle latency is the spiking
  cost of an *analog* context update — invisible to the validator's instantaneous
  $c_t = e_i$ — and it is comfortably affordable.
]

= (3) Drift, and the finite capacity FIX G makes measurable

#figure(image("results/e06_drift.pdf", width: 74%),
  caption: [Drift: write a symbol, then hold for $1$ s with *no input* ($8$ seeds, grey;
    rms across seeds, blue). The bump wanders, but the rms wander stays *far* below the
    $45 degree$ half-cell margin (red dashed) — $1.48 degree$ after one ISI (green dotted),
    $4.90 degree$ even after a full second.])

#finding[
  The held bump *drifts* — graded persistent activity on a finite spiking substrate
  is not a perfect register. The drift is *bounded*: rms wander $= 1.48 degree$ after
  one ISI ($140$ ms), growing slowly at $approx 4.0 degree slash "s"$ to $4.90 degree$
  after a full $1$ s hold. Both are an order of magnitude inside the $45 degree$
  half-cell margin, so the symbol identity is never lost on the timescales the coder
  uses.
]

#gap[
  This drift is exactly the spiking reality the `numpy` validator cannot see. The
  validator's $c_t = e_i$ is a clean, drift-free one-hot. Here $c_t$ is an analog bump
  whose position is corrupted by the ensemble's representational noise and integrated
  by the recurrence — so it *wanders*, and two bump positions are distinguishable only
  if they differ by more than that wander. That is FIX G made literal: the attractor's
  capacity is finite and noise-limited, not arbitrary.
]

#figure(image("results/e06_capacity.pdf", width: 70%),
  caption: [Effective capacity $C = log_2("SNR")$, $"SNR" = 360 degree slash "(positional
    jitter)"$. At the inter-symbol timescale ($1.48 degree$ jitter) the ring resolves
    $approx 243$ states ($C = 7.93$ bits); even after a $1$ s hold ($4.90 degree$
    jitter) it resolves $approx 74$ states ($C = 6.20$ bits). Both *comfortably* exceed
    the $log_2 4 = 2$ bits a $4$-symbol source needs (red dashed).])

#finding[
  *Effective capacity, measured.* With usable range $= 360 degree$ and the
  ISI-timescale positional jitter $1.48 degree$, the signal-to-noise ratio is $243$, so
  the attractor reliably resolves $C = log_2 243 = 7.93$ bits of context — and even
  the $1$ s-hold worst case gives $C = 6.20$ bits. The $4$-symbol first-order rover needs
  only $log_2 4 = 2$ bits, so its context fits with $approx 4$ bits to spare. *Four
  states comfortably fit.*
]

#honest[
  *The capacity is finite, and that is the point (FIX G / BIO-06).* We report $C approx
  log_2("SNR")$ as a *bound*, not a license for unbounded memory. Two honesty caveats.
  (i) Our SNR uses the *drift-only* positional jitter measured at $delta t = 10^(-3)$
  with $700$ neurons; a noisier substrate, a coarser ensemble, or a longer hold all
  *shrink* the SNR and hence $C$ — the paper's claim is precisely that this number is
  finite and substrate-dependent, which our measurement confirms (it *halves* from
  $7.93$ to $6.20$ bits as the hold grows from one ISI to $1$ s). (ii) Capacity here is
  the number of *distinguishable bump positions* on one ring — a *single graded value*.
  It encodes *memory order up to that capacity*, not arbitrarily deep history: a source
  whose statistics depend on more past structure than one ring position can index
  cannot be predicted to its conditional floor by this substrate. For the first-order
  rover, the relevant context is exactly the previous symbol, and it fits. We also note
  the *load* is an engineered clear-then-write, not a self-organized one: the honest
  reading is that a real circuit needs a gating signal to overwrite a latched bump
  (antipodal writes otherwise cancel through the origin) — which is itself the
  `new_window` reset the coder already performs at every symbol.
]

= Acceptance

#accept_table((
  (true, [ring holds a written symbol over an ISI ($100%$ of hold samples within the $45 degree$ half-cell)]),
  (true, [per-window hold decode of a $6$-symbol stream is correct ($0 slash 6$ errors)]),
  (true, [write/settle latency $< $ ISI (max $35$ ms $lt.double$ $200$ ms, all symbols)]),
  (true, [drift is bounded: rms wander after one ISI ($1.48 degree$) $< 45 degree$ half-cell]),
  (true, [effective capacity at ISI ($7.93$ bits) $>= log_2 4 = 2$ bits — $4$ states distinguishable]),
))

#finding[
  *5/5 passed.* A real spiking ring attractor holds the running context $c_t$ as a
  graded bump: it loads an arbitrary symbol in $approx 35$ ms, holds it across the
  $200$ ms inter-symbol interval with $0$ decode errors over a $6$-symbol stream, and
  drifts only $1.48 degree$ rms per ISI — an effective capacity of $approx 8$ bits, far
  above the $2$ bits the $4$-symbol source needs. FIX G is confirmed, not contradicted:
  the capacity is *finite, noise-limited, and substrate-dependent* (it halves under a
  $1$ s hold), exactly $log_2("SNR")$ bits — and that finite budget comfortably holds a
  first-order context. The graded analog bump is the faithful, drifting, settle-limited
  reality behind the validator's instantaneous $c_t = e_i$.
]
