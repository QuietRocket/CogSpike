#!/usr/bin/env bash
# Superset workspace setup for CogSpike.
# Designed to be fast on warm caches and idempotent.

set -euo pipefail

echo "==> Rust: ensuring wasm32-unknown-unknown target"
rustup target add wasm32-unknown-unknown >/dev/null

echo "==> Rust: prefetching cargo dependencies (no compile)"
cargo fetch --locked

REQS="$SUPERSET_WORKSPACE_PATH/deq/requirements.txt"
VENV="$SUPERSET_WORKSPACE_PATH/deq/.venv"

if [[ -f "$REQS" ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
  fi

  if [[ ! -d "$VENV" ]]; then
    echo "==> Python: creating venv at $VENV (via uv)"
    uv venv "$VENV"
  fi

  echo "==> Python: syncing $REQS into $VENV"
  uv pip install --python "$VENV/bin/python" -r "$REQS"
else
  echo "==> Python: no deq/requirements.txt on this branch, skipping venv"
fi

echo "==> Setup complete."
