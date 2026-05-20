#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=../.venv/bin/python3

mkdir -p results/phase0 results/phase1 results/phase2 results/phase3

$PYTHON phase0_fcs_baseline_multi.py     2>&1 | tee results/phase0.log
$PYTHON phase1_siegert_orbits.py         2>&1 | tee results/phase1.log
$PYTHON phase2_latency_gate_multi.py     2>&1 | tee results/phase2.log
$PYTHON phase3_finite_N_multi.py         2>&1 | tee results/phase3.log

typst compile phase0_report.typ results/phase0_report.pdf
typst compile phase1_report.typ results/phase1_report.pdf
typst compile phase2_report.typ results/phase2_report.pdf
typst compile phase3_report.typ results/phase3_report.pdf
typst compile note/closed_form_wta_multi_note.typ note/closed_form_wta_multi_note.pdf --root .

echo "All phases complete. Final note: note/closed_form_wta_multi_note.pdf"
