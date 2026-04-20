#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=.venv/bin/python3

mkdir -p results
$PYTHON phase0_infrastructure.py 2>&1 | tee results/phase0.log
$PYTHON phase1_spectral_gap.py 2>&1 | tee results/phase1.log
$PYTHON phase2_bifurcation.py 2>&1 | tee results/phase2.log
$PYTHON phase3_pole_placement.py 2>&1 | tee results/phase3.log
$PYTHON phase4_cross_validation.py 2>&1 | tee results/phase4.log
$PYTHON phase5_other_archetypes.py 2>&1 | tee results/phase5.log
$PYTHON final_summary.py 2>&1 | tee results/final_summary.log
typst compile note/population_note.typ note/population_note.pdf

echo "All phases complete. Final note: note/population_note.pdf"
