#!/usr/bin/env bash
# Run all follow-up experiments end-to-end.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

ROOT="$(cd ../../.. && pwd)"
PYTHON="$ROOT/deq/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python
fi

mkdir -p results/expA results/expB results/expC

echo "==> Experiment A: dynamic-tau calibration"
"$PYTHON" expA_dynamic_calibration.py | tee results/expA.log

echo "==> Experiment B: renewal PMF predictor"
"$PYTHON" expB_renewal_neuron.py | tee results/expB.log

echo "==> Experiment C: 3-neuron negative loop"
"$PYTHON" expC_three_neuron.py | tee results/expC.log

echo "==> Typst: per-experiment reports"
typst compile --root "$ROOT" expA_report.typ
typst compile --root "$ROOT" expB_report.typ
typst compile --root "$ROOT" expC_report.typ

echo "==> Typst: integrating note"
typst compile --root "$ROOT" note/followups_note.typ

echo "==> Done. Final PDF: note/followups_note.pdf"
