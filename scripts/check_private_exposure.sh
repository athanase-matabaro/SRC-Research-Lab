#!/usr/bin/env bash
#
# H5 Private Content Exposure Guard - SRC Research Lab
# Ensures no private content from parent workspace leaks into this repository
#
set -euo pipefail

echo "=== H5 Private Content Exposure Check ==="
echo "Repository: src-research-lab"
echo "Purpose: Validate isolation from parent workspace private content"
echo ""

# Private patterns that must NOT appear in git-tracked files
PRIVATE_PATTERNS=(
    "src_engine_private"
    "results/private_runs"
    "../src_engine_private"
    "/home/.*/compression-lab/src_engine_private"
    "../energy/datasets"
    "../quarantine"
    "../models/proprietary"
)

echo "Checking for private content patterns..."
echo "Patterns monitored:"
for pattern in "${PRIVATE_PATTERNS[@]}"; do
    echo "  - $pattern"
done
echo ""

# Get all git-tracked files
echo "Scanning git-tracked files..."
FILES=$(git ls-files)

# Check for violations
FOUND=()
VIOLATION_DETAILS=()

for pattern in "${PRIVATE_PATTERNS[@]}"; do
    # Use grep to find matches (case-insensitive, show filenames only)
    if matches=$(echo "$FILES" | grep -iF "$pattern" 2>/dev/null); then
        FOUND+=("$pattern")
        while IFS= read -r file; do
            VIOLATION_DETAILS+=("  Pattern '$pattern' found in: $file")
        done <<< "$matches"
    fi
done

# Report results
echo "═══════════════════════════════════════════════════════════"
echo "║           PRIVATE CONTENT EXPOSURE SUMMARY              ║"
echo "═══════════════════════════════════════════════════════════"
echo ""

if [ "${#FOUND[@]}" -gt 0 ]; then
    echo "❌ FAIL: Found references to private paths in git-tracked files"
    echo ""
    echo "Violations detected:"
    for detail in "${VIOLATION_DETAILS[@]}"; do
        echo "$detail"
    done
    echo ""
    echo "ACTION REQUIRED:"
    echo "  1. Remove files that reference private content"
    echo "  2. Update code to avoid parent directory imports"
    echo "  3. Ensure src-research-lab remains self-contained"
    echo "  4. Review docs/ops/PRIVATE_CONTENT.md for guidelines"
    echo ""
    echo "See: docs/ops/PRIVATE_CONTENT.md"
    exit 2
else
    echo "✅ PASS: No private content references detected in git-tracked files"
    echo ""
    echo "Repository is properly isolated from parent workspace."
    echo "Total files scanned: $(echo "$FILES" | wc -l)"
    echo ""
    echo "Isolation verified:"
    echo "  ✓ No references to src_engine_private"
    echo "  ✓ No parent directory escapes (../)"
    echo "  ✓ No absolute paths to compression-lab"
    echo "  ✓ Repository is self-contained"
    exit 0
fi
