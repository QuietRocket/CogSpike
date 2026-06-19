#!/usr/bin/env bash
# Run every experiment and compile every report PDF, then the synthesis.
# Usage:  bash run_all.sh            # run all experiments + compile all PDFs
#         bash run_all.sh --reports  # only recompile PDFs (skip running)
set -uo pipefail
cd "$(dirname "$0")"
SUITE="$(pwd)"
ONLY_REPORTS="${1:-}"

echo "== spikecoder package self-test =="
uv run python test_spikecoder.py || echo "  (package test reported failures)"

pass=0; fail=0
for d in e[0-9][0-9]_*/; do
  d="${d%/}"
  if [[ "$ONLY_REPORTS" != "--reports" && -f "$d/run.py" ]]; then
    echo "== running $d =="
    if uv run python "$d/run.py"; then echo "  [$d] run OK"; pass=$((pass+1)); else echo "  [$d] run reported FAIL"; fail=$((fail+1)); fi
  fi
  if [[ -f "$d/report.typ" ]]; then
    echo "== compiling $d/report.pdf =="
    typst compile --root . "$d/report.typ" "$d/report.pdf" && echo "  [$d] PDF OK" || echo "  [$d] PDF FAILED"
  fi
done

if [[ -f report/synthesis.typ ]]; then
  echo "== compiling synthesis =="
  typst compile --root . report/synthesis.typ report/synthesis.pdf && echo "  synthesis PDF OK" || echo "  synthesis PDF FAILED"
fi

echo "== done: $pass experiments ran clean, $fail reported failures =="
