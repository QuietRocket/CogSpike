# Cover note: population_note v2

This is a substantial restructuring of the v1 note for an FCS-native
audience, following the design brief. The original `population_note.typ`
and `population_note.pdf` are untouched; the v2 lives alongside as
`population_note_v2.typ` / `population_note_v2.pdf`.

## What's new vs the v1

**Restructured to ten sections.** The old structure (Introduction →
Setup → Hypothesis A → Hypothesis B → Hypothesis C → Bridge → Other
archetypes → Discussion → Conclusion) became (Introduction → W-C lift
→ Contralateral topology in equations → Pitchfork verification →
Bifurcation loci as loop conditions → Pole placement → Cross-validation
→ Other archetypes → Discussion → Conclusion). The reordering puts the
topology→equation derivation before the numerical verification, so
when the t=4 panel appears the reader has already seen the loop-gain
mechanism behind it.

**§2 added derivation sketch + variable bridge table.** The W-C
reduction is now spelled out in five steps (distributed thresholds →
CDF counting → smoothness emerges → leak averaging → multi-pop ODE),
and Table 1 maps each population-framework symbol back to its FCS
analogue with intuition (ρᵢ ↔ Boolean spike, τ ↔ leak vector window,
f ↔ threshold CDF, etc.). This is the conceptual scaffold the v1
omitted.

**§3 is the new heart.** Three pages on the contralateral archetype
that walk from the topology graph (FCS Fig. 1f) to the pitchfork in
two visible lines of algebra, with intermediate stops at the
symmetric-fixed-point plot, the sum/difference mode decoupling, the
"pencil on its tip" remark, and the loop-gain Barkhausen-criterion
analogy. Elisabetta's question — *"shouldn't the WTA behaviour be
visible directly in the equations?"* — is answered explicitly here.

**§4 disambiguates the WTA property.** The v1's apparent diagonal
disagreement with FCS Fig. 10 was not a colour bug — it was a
property mismatch: FCS §6.3.4 tests "stabilises within four ticks",
while the v1 tested "commits asymptotically by t=50τ". The v2 spells
out the three nested but inequivalent properties and shows both
panels (t=50τ and t=4τ) side-by-side in the FCS teal/red palette.

**§5 frames bifurcation loci as loop conditions.** det J = 0 is now
named explicitly as the loop-gain unity contour (the same w₁₂·w₂₁·g₁·g₂ = 1
from §3), and a new sign-product-around-loops table classifies
contralateral (+ pitchfork), activator-inhibitor (− Hopf), and
positive loop (+ saddle-node) under the Thomas-rule framework. The
brief identified Thomas's rules as a familiar reference from FCS §2.

**§7 added scaling-question answer.** A 200×200 LI&F sweep extends
the original 40×40 Phase 4 figure out to |w^LIF|=200. The
rectangular-strip boundary at |w|≈7 persists flat — no fractal or
new structure emerges. This is the answer to your original "how does
Fig. 10 scale" motivation.

## Deviations from the design brief

**The "WTA region shrinks dramatically at t=4" prediction was wrong.**
The brief expected the t=4 panel to look much more like FCS Fig. 10's
red-dominated diagonal. In practice the WC region only shrinks by
2.4% (1184 vs 1213 cells out of 2500), because the WC linear dynamics
commit fast: starting from a 0.05 perturbation, growth at rate
(wg−1)/τ reaches the 0.3 commitment margin within ~4τ across most of
the bistable region. I reframed this as evidence *for* the framework:
the spectral predictor's 99.96% accuracy is robust to short time
horizons.

I did probe stricter margin variants (`_t4_margin_probe.py`):
- margin=0.3 → 1184 cells
- margin=0.5 → 1154
- margin=0.7 → 1056
- margin=0.85 → 0 (no cell commits hard within 4τ from a 0.05 perturbation)

So FCS-Fig.10-like geometries can be obtained by tightening the
margin instead of just shrinking the time horizon, but I chose to
keep margin=0.3 in the v2 figure for consistency with the rest of
the note's classifier. The v2 §4.3 instead names the time-scale
mismatch directly: FCS's "4 ticks" is not the WC framework's "4τ",
because the discrete LI&F simulator needs many ticks just to
accumulate spike-count statistics whereas the WC linear mode commits
in O(τ) time. The two frameworks are measuring different things.

If you'd rather present the strict-margin variant in the v2 figure,
the .npy files exist at `results/ground_truth_contralateral_t4_m{30,50,70,85}.npy`;
swap the array name in `regen_figs.py`'s p1_t4_panel block.

**E2 (FCS Lustre code) skipped.** The brief flagged the I3S Redmine
URL was deprecated and recommended emailing Daniel Gaffé directly.
Neither I nor the cross-validation §7 has access to the original
FCS Lustre code, so the §7 comparison is to your own LI&F simulator
(`deq/archetypes/lif_fcs.py`) implementing FCS §6.2 semantics
verbatim. Acknowledged in §7.3.

**E4 (delayer-augmented contralateral) skipped per scope decision.**

**Loop-gain heatmap generated but not referenced in v2.** I produced
`note/figs/loop_gain_heatmap.pdf` per the brief's "optional" item.
On reflection, the v1's existing Figure 1's spectral-gap heatmap
(now in v2 as figure references via the symbolic curve) already
shows the same information at the level of the dominant eigenvalue.
If you'd like to reference the loop-gain heatmap directly somewhere
in §3 or §5, the file is there.

## E1 verification: t=4 panel vs FCS Fig. 10

**Geometry**: the v2 t=4 panel and t=50 panel share the same
hyperbolic wedge bounded by the pitchfork curve. *FCS Fig. 10's
red-dominated geometry is qualitatively different:* in the lower-weight
region the FCS figure is uniformly red (no fast WTA), while the WC
panels at t=4 are uniformly teal (asymptotic WTA happens). The
populations differ on this region because they are testing
qualitatively different things — "the bistable mode exists" (WC) vs
"the bistable mode commits within 4 discrete ticks" (FCS).

**Diagonal**: the WC pitchfork apex at w*=1 corresponds to |w^LIF|≈8
under the heuristic |w|/8 scaling. FCS Fig. 10's diagonal teal sets
in at |w^LIF|≈30. The factor-of-roughly-four gap between
*"bistable mode exists"* and *"fast WTA"* is itself informative and
called out in v2 §4.3.

## Engineering deliverables

New code:
- `phase1b_t4_sweep.py` — parallelised 4-tick WTA classifier sweep
- `phase4b_extreme_lif.py` — 200×200 LI&F bistable sweep
- `_t4_margin_probe.py` — one-off margin-sensitivity probe (kept for reference)
- `note/regen_figs.py` extended to emit `p1_t4_panel.pdf`,
  `rho_star_curve.pdf`, `loop_gain_heatmap.pdf`, `p4_lif_extreme.pdf`

New artifacts:
- `results/ground_truth_contralateral_t4.npy` and four margin variants
- `results/lif_extreme_*.npy` (5 arrays: WTA map + spike counts)

New figures (in `note/figs/`):
- `p1_t4_panel.pdf` — t=50τ vs t=4τ side-by-side, FCS palette
- `rho_star_curve.pdf` — symmetric fixed-point branches
- `loop_gain_heatmap.pdf` — loop-gain product on the Phase 1 grid
- `p4_lif_extreme.pdf` — 200×200 LI&F bistable region

Build:
- `make_all.sh` updated to run phase1b + phase4b and to compile both v1 and v2 PDFs
- v1 (`population_note.typ` / `.pdf`) is untouched

The v2 PDF compiles in <1s once .npy artifacts are present (which
they are after the first `make_all.sh` run); from a clean checkout
the full pipeline is roughly the v1 runtime + 5s for E1 + 5s for E3.

## Open questions for you

1. **Strict-margin t=4 figure vs current.** As noted above, the
   stricter-margin variant produces a smaller WTA region that visually
   looks more like FCS Fig. 10. The v2 currently uses margin=0.3 for
   classifier consistency. Worth swapping?

2. **Loop-gain heatmap inclusion.** The figure exists but isn't
   referenced. If you want it in §3.5 or §5.1 it's a one-line image
   include.

3. **Pitchfork-apex unit conversion sentence.** The v2 §4.3 says
   "w*=1 corresponds to |w^LIF|≈8 under the heuristic scaling". This
   uses the §7 scaling factor of 8. If you'd rather pin a different
   conversion factor (FCS's natural unit choice was b=11 for the
   external drive), the relevant numbers in §4.3 and §7 should be
   updated together.
