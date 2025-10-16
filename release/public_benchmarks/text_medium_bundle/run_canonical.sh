#!/bin/bash
# Canonical Benchmark Run Script for text_medium
# Public Benchmark Bundle — SRC Research Lab Phase H.4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$SCRIPT_DIR/dataset"
OUTPUT_DIR="$SCRIPT_DIR/output"
MOCK_BRIDGE="$SCRIPT_DIR/mock_bridge.py"

echo "=== SRC Research Lab Benchmark: text_medium ==="
echo "Timestamp: $(date -Iseconds)"
echo ""

# Check for mock bridge
if [ ! -f "$MOCK_BRIDGE" ]; then
    echo "ERROR: mock_bridge.py not found"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run compression on each dataset file
RESULTS=()
for input_file in "$DATASET_DIR"/*; do
    if [ ! -f "$input_file" ]; then
        continue
    fi

    filename=$(basename "$input_file")
    output_file="$OUTPUT_DIR/$filename.cxe"

    echo "Compressing: $filename"

    # Run mock compression
    result=$(python3 "$MOCK_BRIDGE" compress "$input_file" "$output_file")

    # Extract metrics
    ratio=$(echo "$result" | python3 -c "import sys, json; print(json.load(sys.stdin)['ratio'])")
    cpu_time=$(echo "$result" | python3 -c "import sys, json; print(json.load(sys.stdin)['cpu_time'])")

    echo "  Ratio: $ratio, CPU: $cpu_time s"

    RESULTS+=("$ratio,$cpu_time")
done

# Compute average CAQ
echo ""
echo "Computing CAQ..."

# Simple average computation in bash
total_caq=0
total_ratio=0
total_cpu=0
count=0

for result in "${RESULTS[@]}"; do
    ratio=$(echo "$result" | cut -d',' -f1)
    cpu=$(echo "$result" | cut -d',' -f2)
    caq=$(echo "$ratio / ($cpu + 1.0)" | bc -l)

    total_caq=$(echo "$total_caq + $caq" | bc -l)
    total_ratio=$(echo "$total_ratio + $ratio" | bc -l)
    total_cpu=$(echo "$total_cpu + $cpu" | bc -l)
    count=$((count + 1))
done

if [ $count -gt 0 ]; then
    mean_caq=$(echo "scale=2; $total_caq / $count" | bc -l)
    mean_ratio=$(echo "scale=2; $total_ratio / $count" | bc -l)
    mean_cpu=$(echo "scale=4; $total_cpu / $count" | bc -l)

    echo "Mean CAQ: $mean_caq"
    echo "Mean Ratio: $mean_ratio"
    echo "Mean CPU: $mean_cpu s"
fi

echo ""
echo "✓ Benchmark complete. Check example_submission.json for expected output format."
