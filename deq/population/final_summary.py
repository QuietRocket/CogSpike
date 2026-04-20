"""Cross-phase summary.

Prints a compact verdict table for each phase by reading the saved logs
and key artifacts. Does not re-run any sweep; called from make_all.sh
after all phase drivers have completed.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def tail(path: Path, n: int = 10) -> list[str]:
    if not path.exists():
        return [f"(missing: {path})"]
    lines = path.read_text().splitlines()
    return lines[-n:]


def extract_verdict(log_path: Path) -> str:
    lines = tail(log_path, 15)
    for line in lines:
        lower = line.lower()
        if "verdict" in lower or "phase" in lower and ("pass" in lower or "fail" in lower or "partial" in lower):
            return line.strip()
    return "(verdict not found)"


def main() -> int:
    print("=" * 72)
    print("Population-level spectral analysis: cross-phase summary")
    print("=" * 72)
    for phase in range(0, 6):
        log = RESULTS / f"phase{phase}.log"
        v = extract_verdict(log)
        print(f"Phase {phase}: {v}")
    print("=" * 72)
    note_pdf = HERE / "note" / "population_note.pdf"
    if note_pdf.exists():
        print(f"Note: {note_pdf} ({note_pdf.stat().st_size // 1024} KB)")
    else:
        print("Note: (not yet compiled)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
