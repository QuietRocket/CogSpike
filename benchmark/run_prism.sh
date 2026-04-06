#!/usr/bin/env bash
# ============================================================================
# Paper Benchmark: Extract state/transition counts from generated PRISM models.
#
# Prerequisites:
#   1. Run: cargo test --test paper_benchmark -- --nocapture
#   2. Then: bash benchmark/run_prism.sh
#
# Output: benchmark/results.csv
# ============================================================================

set -euo pipefail

PRISM="/Users/quietrocket/Documents/PhD/prism-4.9-mac64-arm/bin/prism"
BENCHMARK_DIR="benchmark"
RESULTS_FILE="$BENCHMARK_DIR/results.csv"
JAVA_MEM="-javamaxmem 4g"
CUDD_MEM="-cuddmaxmem 4g"
TIMEOUT=300  # seconds per PRISM invocation

if [[ ! -x "$PRISM" ]]; then
    echo "ERROR: PRISM not found at $PRISM"
    exit 1
fi

echo "filename,states,transitions,time_sec,status" > "$RESULTS_FILE"

# Count total .pm files
total=$(find "$BENCHMARK_DIR" -name "*.pm" | wc -l | tr -d ' ')
current=0

for pm_file in "$BENCHMARK_DIR"/*.pm; do
    current=$((current + 1))
    fname=$(basename "$pm_file")
    printf "[%d/%d] %-45s " "$current" "$total" "$fname"

    start_time=$(date +%s)

    output=$("$PRISM" "$pm_file" \
        $JAVA_MEM $CUDD_MEM \
        -ex \
        2>&1) || true

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    # Extract states: look for "States:      N (..."
    states=$(echo "$output" | grep "^States:" | tail -1 | sed 's/States:[[:space:]]*//' | sed 's/[[:space:]].*//' | tr -d ',')
    # Extract transitions: look for "Transitions: N"
    transitions=$(echo "$output" | grep "^Transitions:" | tail -1 | sed 's/Transitions:[[:space:]]*//' | sed 's/[[:space:]].*//' | tr -d ',')

    # Check for errors
    if echo "$output" | grep -qiE "out of memory|OutOfMemoryError|Cannot allocate"; then
        echo "OOM (${elapsed}s)"
        echo "$fname,OOM,OOM,$elapsed,oom" >> "$RESULTS_FILE"
    elif [[ $elapsed -ge $TIMEOUT ]]; then
        echo "TIMEOUT (${elapsed}s)"
        echo "$fname,TIMEOUT,TIMEOUT,$elapsed,timeout" >> "$RESULTS_FILE"
    elif [[ -n "$states" && -n "$transitions" ]]; then
        printf "states=%-12s trans=%-16s (%ds)\n" "$states" "$transitions" "$elapsed"
        echo "$fname,$states,$transitions,$elapsed,ok" >> "$RESULTS_FILE"
    else
        echo "PARSE_ERROR (${elapsed}s)"
        echo "  Last 5 lines of output:"
        echo "$output" | tail -5 | sed 's/^/    /'
        echo "$fname,ERROR,ERROR,$elapsed,parse_error" >> "$RESULTS_FILE"
    fi
done

echo ""
echo "=== Results written to $RESULTS_FILE ==="
echo ""
# Print a compact summary table
printf "%-45s %12s %16s\n" "Model" "States" "Transitions"
printf "%-45s %12s %16s\n" "-----" "------" "-----------"
tail -n +2 "$RESULTS_FILE" | sort | while IFS=',' read -r fname states trans time status; do
    printf "%-45s %12s %16s\n" "$fname" "$states" "$trans"
done
