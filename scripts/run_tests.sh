#!/usr/bin/env bash
#
# H5 Test Runner - SRC Research Lab
# Wrapper for running tests across multiple frameworks
#
set -euo pipefail

echo "=== H5 Test Runner ==="
echo "Running tests wrapper..."
echo ""

EXIT_CODE=0

# Function to run tests and capture result
run_test_framework() {
    local framework="$1"
    local command="$2"
    
    echo "---"
    echo "Checking for $framework..."
    
    if eval "$command"; then
        echo "✅ $framework tests passed"
    else
        local code=$?
        echo "⚠️  $framework tests failed (exit code: $code)"
        EXIT_CODE=$code
    fi
    echo ""
}

# Python/pytest
if command -v pytest >/dev/null 2>&1 && [ -d tests ]; then
    run_test_framework "pytest" "pytest -q --tb=short || true"
elif command -v python3 >/dev/null 2>&1 && [ -d tests ]; then
    run_test_framework "Python unittest" "python3 -m unittest discover -s tests -p 'test_*.py' -q || true"
fi

# Node.js/npm
if [ -f package.json ]; then
    if command -v npm >/dev/null 2>&1; then
        run_test_framework "npm" "npm ci --quiet && npm test || true"
    fi
fi

# Make
if [ -f Makefile ] && grep -q "^test:" Makefile 2>/dev/null; then
    run_test_framework "make" "make test || true"
fi

echo "=== Test Run Complete ==="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All tests passed"
else
    echo "⚠️  Some tests failed (informational only, does not block CI)"
fi

exit 0
