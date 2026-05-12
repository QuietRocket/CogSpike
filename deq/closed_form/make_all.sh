#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=../.venv/bin/python3

mkdir -p results

$PYTHON phase0_infrastructure.py 2>&1 | tee results/phase0.log
$PYTHON phase1_siegert.py        2>&1 | tee results/phase1.log
$PYTHON phase2_transfer.py       2>&1 | tee results/phase2.log
$PYTHON phase3_linresp_xval.py   2>&1 | tee results/phase3.log
$PYTHON phase4_quasi_renewal.py  2>&1 | tee results/phase4.log

typst compile note/closed_form_note.typ note/closed_form_note.pdf --root .

echo "All phases complete. Final note: note/closed_form_note.pdf"
