#!/bin/bash
# run_all_benchmarks.sh — Run each benchmark 3 times, report median
set -e

RESULTS_DIR="benchmark_results"
mkdir -p "$RESULTS_DIR"

run_bench() {
    local name=$1
    local binary=$2
    local logfile="$RESULTS_DIR/${name}_raw.txt"
    
    echo "══ $name ══"
    echo "" > "$logfile"
    
    for run in 1 2 3; do
        echo "--- Run $run ---" >> "$logfile"
        ./$binary >> "$logfile" 2>&1
        echo "" >> "$logfile"
    done
    
    # Extract throughput (last numeric line with "checks/sec" or "values/sec" or "dist/sec")
    # Just cat the last run for now; we'll format manually
    tail -20 "$logfile"
    echo ""
}

echo "╔══════════════════════════════════════════════════════╗"
echo "║   FLUX GPU Micro-Benchmarks — RTX 4050 (SM 8.9)    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

run_bench "exact_check" "exact_check"
run_bench "batch_check" "batch_check" 
run_bench "sediment" "sediment"
run_bench "bfs" "bfs"
run_bench "hyperbolic" "hyperbolic"

echo ""
echo "Raw results saved to $RESULTS_DIR/"
echo "Fill README.md with final numbers."
