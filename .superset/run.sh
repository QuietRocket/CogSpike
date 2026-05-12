#!/usr/bin/env bash
# Superset run script: launches the egui app in the browser via trunk.
# Use `cargo run` instead if you want the native build.

set -euo pipefail

if ! command -v trunk >/dev/null 2>&1; then
  echo "trunk not installed. Install with: cargo install --locked trunk"
  echo "Falling back to native: cargo run"
  exec cargo run
fi

exec trunk serve --open
