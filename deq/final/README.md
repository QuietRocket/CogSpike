# `deq/final/` — DEQ research synthesis

This folder consolidates the seven-thread CogSpike **DEQ** research programme —
the multi-week investigation into whether *differential-equation / continuous*
analysis can reveal structure in spiking neural networks that the prior
*discrete* (model-checking) work cannot.

It is the destination called for in the task "compile all this understanding
into `deq/final`". It contains the comprehensive synthesis; a condensed paper
draft distilled from it lives in `paper/continuous_lens.typ`.

## Contents

| File | What it is |
|---|---|
| `synthesis.typ` / `synthesis.pdf` | **The master compendium.** All 7 threads, organised by the methodology ladder, with the capability matrix, the honest dead-ends section, three appendices (chronology, phase scorecards, glossary). Written for the FCS / formal-methods audience — no differential-equations background assumed. |
| `refs.bib` | Merged, de-duplicated bibliography (the per-thread `refs.bib` files plus FCS / formal-methods entries), single `ieee` style. |
| `figs/` | The 17 curated figures used by the two documents, copied from the source threads. Provenance table below. |
| `README.md` | This file. |
| `../../paper/continuous_lens.typ` | **The condensed paper draft**, distilled from `synthesis.typ`. Lives in `paper/` beside the prior CogSpike paper. |

## The central finding

Continuous methods recover the **shape and scale** of LI&F archetype behaviour
— boundaries, envelopes, bifurcation curves, oscillation existence,
reachability — but systematically **miss the integer-tick spike-timing
physics** (the exact winner-take-all staircase, the exact oscillation period,
the binary `1100` waveform). One nonlinearity, the **spike-reset rule**, is the
whole boundary between what these methods reach and what they do not, and
whether it appears in the property being verified is a decidable, a-priori
test. The six methods form a **ladder** of increasing fidelity / decreasing
tractability; the synthesis is organised by it.

## The seven threads

The compendium consolidates seven `deq/` research threads, in research order:

| # | Thread directory | Subject |
|---|---|---|
| 1 | `deq/research_note.typ`, `deq/cs_research_note.typ` (+ toolkit `deq/*.py`) | Foundations: LI&F dynamics as linear recurrences; eigenvalue / transfer-function toolkit |
| 2 | `deq/archetypes/` | Spectral cartography; the two-regime winner-take-all split; the diagnostic principle |
| 3 | `deq/population/` | Wilson–Cowan mean-field lift; exact bifurcation curves |
| 4 | `deq/closed_form/` | Siegert / transfer-function / quasi-renewal closed forms |
| 5 | `deq/closed_form_wta/` | Three lenses on FCS Property 7 (winner-take-all, the staircase) |
| 6 | `deq/closed_form_wta_multi/` | Winner-take-all at N > 2; the inverse staircase |
| 7 | `deq/closed_form_neg_loop/` | Three lenses on FCS Property 5 (negative-loop oscillation) |

Each thread retains its own phase reports and advisor-facing `note/` under its
directory; this folder is the cross-thread synthesis, not a replacement.

## Building

The synthesis uses paths relative to itself, so it builds directly:

```sh
cd deq/final
typst compile synthesis.typ
```

The paper draft references this folder's `figs/` and `refs.bib` with
repository-root-relative paths (the house convention used by the thread
notes), so it is built with `--root` set to the repository root:

```sh
typst compile --root <repo-root> paper/continuous_lens.typ
```

## Figure provenance

All figures in `figs/` are copied unmodified from the source threads' result
directories. Nothing is regenerated.

| `figs/` file | Source | Used in |
|---|---|---|
| `triptych.png` | `archetypes/results/final_triptych.png` | synthesis §3.1, paper §3 |
| `fcs_staircase.pdf` | `closed_form_wta/results/phase0/fcs_grid.pdf` | synthesis §3.1 |
| `wc_wta_panel.pdf` | `population/note/figs/p1_t4_panel.pdf` | synthesis §3.1 |
| `wc_lif_overlay.pdf` | `population/note/figs/p4_overlay.pdf` | synthesis §3.1, paper §4 |
| `pitchfork.pdf` | `population/note/figs/p2_pitchfork.pdf` | paper §4 |
| `siegert_envelope.pdf` | `closed_form_wta/results/phase1/siegert_vs_fcs.pdf` | synthesis §3.1, paper §5 |
| `qr_staircase_jaccard.pdf` | `closed_form_wta/results/phase3/qr_jaccard_vs_N.pdf` | synthesis §3.1 |
| `inverse_staircase.pdf` | `closed_form_wta_multi/results/phase1/siegert_orbits_vs_fcs.pdf` | synthesis §3.1 |
| `bode.pdf` | `closed_form/results/phase2/bode.pdf` | synthesis §2.2 |
| `prop5_trace.png` | `archetypes/results/phase0_property5_trace.png` | synthesis §3.2 |
| `negloop_Hw_period.pdf` | `closed_form_neg_loop/results/phase2/T_pred_vs_FCS_period.pdf` | synthesis §3.2, paper §6 |
| `negloop_qr_period.pdf` | `closed_form_neg_loop/results/phase3/period_qr_vs_FCS.pdf` | synthesis §3.2, paper §6 |
| `negloop_3neuron.pdf` | `closed_form_neg_loop/followups/results/expC/three_neuron.pdf` | synthesis §3.2 |
| `rho_distributions.png` | `archetypes/results/phase1c_rho_distributions.png` | synthesis §3.3 |
| `winner_map.png` | `archetypes/results/phase2_winner_map.png` | synthesis §3.3 |
| `poleplacement.png` | `archetypes/results/final_fig3_poleplacement.png` | synthesis §5 |
| `expA_bode.pdf` | `closed_form_neg_loop/followups/results/expA/bode_fit.pdf` | synthesis §5 |

The four PNG figures are reused as raster (no regeneration); the rest are
vector PDF.

## Open items (for the advisors)

- **Authorship.** `synthesis.typ` is a solo research note (`Nikan Zandian`),
  matching the seven thread notes. `paper/continuous_lens.typ` is drafted solo
  for now; the likely *submission* authorship is the three-author line of the
  prior CogSpike paper (Zandian Jazi, De Maria, Leturc) — flagged as a `TODO`
  in the draft, to be decided with advisors.
- **Venue.** The paper draft is a standalone Typst article kept template-light;
  the target venue (LNCS conference vs. a computational-neuroscience journal)
  and final formatting are advisor decisions.
