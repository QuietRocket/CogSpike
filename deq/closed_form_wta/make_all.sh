#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=../closed_form/.venv/bin/python3

mkdir -p results

$PYTHON phase0_fcs_baseline.py     2>&1 | tee results/phase0.log
$PYTHON phase1_siegert_wta.py      2>&1 | tee results/phase1.log
$PYTHON phase2_latency_gate.py     2>&1 | tee results/phase2.log
$PYTHON phase3_finite_N.py         2>&1 | tee results/phase3.log

typst compile phase0_report.typ results/phase0_report.pdf --root ..
typst compile phase1_report.typ results/phase1_report.pdf --root ..
typst compile phase2_report.typ results/phase2_report.pdf --root ..
typst compile phase3_report.typ results/phase3_report.pdf --root ..
typst compile note/closed_form_wta_note.typ note/closed_form_wta_note.pdf --root ..

echo "All phases complete. Final note: note/closed_form_wta_note.pdf"
