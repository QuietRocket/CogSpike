#!/usr/bin/env bash
# Run all phases of the negative-loop closed-form study end-to-end.
# Mirrors deq/closed_form_wta/make_all.sh.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

ROOT="$(cd ../.. && pwd)"
PYTHON="$ROOT/deq/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python
fi

mkdir -p results/phase0 results/phase1 results/phase2 results/phase3

echo "==> Phase 0: FCS Property 5 baseline"
"$PYTHON" phase0_fcs_baseline.py | tee results/phase0.log

echo "==> Phase 1: Siegert FP + Jacobian"
"$PYTHON" phase1_siegert_hopf.py | tee results/phase1.log

echo "==> Phase 2: H(omega) ringing period"
"$PYTHON" phase2_freq_gate.py | tee results/phase2.log

echo "==> Phase 3: Quasi-renewal finite-N"
"$PYTHON" phase3_finite_N.py | tee results/phase3.log

echo "==> Typst: per-phase reports"
typst compile phase0_report.typ
typst compile phase1_report.typ
typst compile phase2_report.typ
typst compile phase3_report.typ

echo "==> Typst: integrating note"
typst compile note/closed_form_neg_loop_note.typ

echo "==> All done. Final PDF: note/closed_form_neg_loop_note.pdf"
