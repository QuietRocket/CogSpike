#!/bin/bash
# =============================================================================
# OOM Benchmark: Binary-search for maximal SNN topology size before PRISM OOM.
#
# Prerequisite:  cargo test --test benchmark_limits -- --nocapture
#                (generates .pm files in research/limits/benchmark/)
#
# Usage:         bash research/limits/run_benchmark.sh
# Output:        research/limits/benchmark_results.csv
# =============================================================================

set -euo pipefail

# ── PRISM configuration (fixed for all tests) ──────────────────────────────
PRISM="/Users/quietrocket/Documents/PhD/prism-4.9-mac64-arm/bin/prism"
JAVA_MAX_MEM="1g"
JAVA_STACK="4m"
CUDD_MAX_MEM="1g"
TIMEOUT=120   # seconds per PRISM invocation

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/benchmark"
RESULTS="$SCRIPT_DIR/benchmark_results.csv"
MAX_SIZE=30   # upper bound for binary search

# ── Helpers ─────────────────────────────────────────────────────────────────

# Monotonic counter for unique temp file names.
_PROBE_CTR=0

# Try to run PRISM on a model file.
# Returns 0 on success (model built), 1 on OOM / error.
# On success, prints "states transitions time_sec" to stdout.
try_prism() {
    local pm_file="$1"
    _PROBE_CTR=$(( _PROBE_CTR + 1 ))
    local tmp_tra="/tmp/bench_${$}_${_PROBE_CTR}.tra"
    local tmp_sta="/tmp/bench_${$}_${_PROBE_CTR}.sta"

    local start_time
    start_time=$(date +%s)

    local output
    output=$("$PRISM" "$pm_file" \
        -exporttrans "$tmp_tra" \
        -exportstates "$tmp_sta" \
        -javamaxmem "$JAVA_MAX_MEM" \
        -javastack "$JAVA_STACK" \
        -cuddmaxmem "$CUDD_MAX_MEM" \
        -timeout "$TIMEOUT" \
        2>&1) || true
    local exit_code=$?
    local end_time
    end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))

    # Detect OOM / error conditions
    if [ $exit_code -ne 0 ]; then
        rm -f "$tmp_tra" "$tmp_sta"
        return 1
    fi
    # Use specific patterns that don't match normal PRISM output like
    # "Memory limits: cudd=1g" or info lines containing "Error" generically.
    if echo "$output" | grep -qiE "OutOfMemoryError|out of memory|CUDD: out|Exception in thread"; then
        rm -f "$tmp_tra" "$tmp_sta"
        return 1
    fi
    if [ ! -f "$tmp_tra" ] || [ ! -s "$tmp_tra" ]; then
        rm -f "$tmp_tra" "$tmp_sta"
        return 1
    fi

    # Extract state and transition counts from .tra header (first line: "states transitions")
    local header
    header=$(head -1 "$tmp_tra")
    local states transitions
    states=$(echo "$header" | awk '{print $1}')
    transitions=$(echo "$header" | awk '{print $2}')

    rm -f "$tmp_tra" "$tmp_sta"
    echo "$states $transitions $elapsed"
    return 0
}

# Binary search for the maximal topology size that succeeds.
# Arguments: topology preset model_type weight_levels
# Prints result line to stdout.
binary_search() {
    local topo="$1" preset="$2" mtype="$3" wl="$4"

    local lo=1 hi=$MAX_SIZE
    local best_size=0 best_states="—" best_trans="—" best_time="—"

    while [ $lo -le $hi ]; do
        local mid=$(( (lo + hi) / 2 ))

        # Construct filename
        local fname
        if [ "$mtype" = "precise" ]; then
            fname="${topo}${mid}_${preset}_precise.pm"
        else
            fname="${topo}${mid}_${preset}_disc_w${wl}.pm"
        fi

        local pm_path="$MODEL_DIR/$fname"
        if [ ! -f "$pm_path" ]; then
            # Model file does not exist – treat as OOM (beyond generated range)
            hi=$(( mid - 1 ))
            continue
        fi

        echo -n "  [$topo $preset $mtype w=$wl] size=$mid ... " >&2
        local result
        if result=$(try_prism "$pm_path"); then
            local s t e
            read -r s t e <<< "$result"
            echo "OK  (states=$s, trans=$t, ${e}s)" >&2
            best_size=$mid
            best_states=$s
            best_trans=$t
            best_time=$e
            lo=$(( mid + 1 ))
        else
            echo "OOM" >&2
            hi=$(( mid - 1 ))
        fi
    done

    echo "$topo,$preset,$mtype,$wl,$best_size,$best_states,$best_trans,$best_time"
}

# ── Main ────────────────────────────────────────────────────────────────────

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory $MODEL_DIR not found."
    echo "Run:  cargo test --test benchmark_limits -- --nocapture"
    exit 1
fi

echo "OOM Benchmark — PRISM resource budget:"
echo "  Java heap:  $JAVA_MAX_MEM"
echo "  Java stack: $JAVA_STACK"
echo "  CUDD mem:   $CUDD_MAX_MEM"
echo "  Timeout:    ${TIMEOUT}s"
echo ""

# CSV header
echo "topology,preset,model_type,weight_levels,max_size,max_states,max_transitions,time_sec" > "$RESULTS"

PRESETS="deterministic fast full"
W_LEVELS="2 3 5"

for topo in chain fork; do
    for preset in $PRESETS; do
        # Precise model
        binary_search "$topo" "$preset" "precise" "N/A" >> "$RESULTS"

        # Discretized models
        for w in $W_LEVELS; do
            binary_search "$topo" "$preset" "disc" "$w" >> "$RESULTS"
        done
    done
done

echo ""
echo "=== Done. Results in $RESULTS ==="
echo ""
column -t -s, "$RESULTS"
