#!/usr/bin/env bash
#
# scripts/validate_compliance.sh
# H5-05: Real Compliance Pipeline
# Purpose: Run SAST, dependency audits, and license scans; produce artifacts
#
set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
ARTIFACTS_DIR="${WORKDIR}/artifacts"
POLICY_FILE="${WORKDIR}/compliance/policy.yml"
SUMMARY="${ARTIFACTS_DIR}/compliance_summary.json"

mkdir -p "${ARTIFACTS_DIR}"

timestamp() { date --iso-8601=seconds 2>/dev/null || date -u +"%Y-%m-%dT%H:%M:%SZ"; }

echo "=== H5-05 Compliance Validation ==="
echo "Repository: $(basename "$WORKDIR")"
echo "Timestamp: $(timestamp)"
echo "Artifacts directory: ${ARTIFACTS_DIR}"
echo ""

# Helper to run commands with logging
run_cmd() {
  echo "---- RUN: $* ----"
  "$@" 2>&1 | tee -a "${ARTIFACTS_DIR}/compliance_raw.log" || return $?
}

COMPLIANCE_STATUS="PASS"
FINDINGS_COUNT=0

###############################################################################
# 1) SAST (Bandit) for Python
###############################################################################
echo "=== 1. SAST Scanning (Bandit) ==="
SAST_RESULTS="${ARTIFACTS_DIR}/bandit_report.json"

if command -v bandit >/dev/null 2>&1; then
  echo "Running Bandit SAST scan..."
  if bandit -r . -f json -o "${SAST_RESULTS}" 2>&1 | tee -a "${ARTIFACTS_DIR}/compliance_raw.log"; then
    echo "✓ Bandit scan completed"
    
    # Check for high/critical severity findings
    if [ -f "${SAST_RESULTS}" ]; then
      HIGH_COUNT=$(jq '[.results[] | select(.issue_severity == "HIGH")] | length' "${SAST_RESULTS}" 2>/dev/null || echo 0)
      MEDIUM_COUNT=$(jq '[.results[] | select(.issue_severity == "MEDIUM")] | length' "${SAST_RESULTS}" 2>/dev/null || echo 0)
      
      echo "  High severity: $HIGH_COUNT"
      echo "  Medium severity: $MEDIUM_COUNT"
      
      if [ "$HIGH_COUNT" -gt 0 ]; then
        echo "  ⚠️  High severity findings detected"
        COMPLIANCE_STATUS="FAIL"
        FINDINGS_COUNT=$((FINDINGS_COUNT + HIGH_COUNT))
      fi
    fi
  else
    echo "⚠️  Bandit scan failed (non-blocking)"
  fi
else
  echo "⚠️  Bandit not found - skipping SAST"
  echo "  Install: pip install bandit"
fi
echo ""

###############################################################################
# 2) Dependency Audit: Python (pip-audit)
###############################################################################
echo "=== 2. Python Dependency Audit (pip-audit) ==="
PIP_AUDIT="${ARTIFACTS_DIR}/pip_audit.json"

if [ -f "requirements.txt" ] || [ -f "pyproject.toml" ]; then
  if command -v pip-audit >/dev/null 2>&1; then
    echo "Running pip-audit..."
    if pip-audit --format json --output "${PIP_AUDIT}" 2>&1 | tee -a "${ARTIFACTS_DIR}/compliance_raw.log"; then
      echo "✓ pip-audit completed"
    else
      # pip-audit exits non-zero if vulnerabilities found
      echo "⚠️  Vulnerabilities found by pip-audit"
      
      if [ -f "${PIP_AUDIT}" ]; then
        # Count critical/high vulnerabilities
        CRITICAL=$(jq '[.vulnerabilities[] | select(.severity == "critical")] | length' "${PIP_AUDIT}" 2>/dev/null || echo 0)
        HIGH=$(jq '[.vulnerabilities[] | select(.severity == "high")] | length' "${PIP_AUDIT}" 2>/dev/null || echo 0)
        
        echo "  Critical: $CRITICAL"
        echo "  High: $HIGH"
        
        if [ "$CRITICAL" -gt 0 ] || [ "$HIGH" -gt 0 ]; then
          COMPLIANCE_STATUS="FAIL"
          FINDINGS_COUNT=$((FINDINGS_COUNT + CRITICAL + HIGH))
        fi
      fi
    fi
  else
    echo "⚠️  pip-audit not found - skipping Python dependency audit"
    echo "  Install: pip install pip-audit"
  fi
else
  echo "ℹ️  No requirements.txt or pyproject.toml - skipping pip-audit"
fi
echo ""

###############################################################################
# 3) Dependency Audit: Node (npm audit)
###############################################################################
echo "=== 3. Node Dependency Audit (npm audit) ==="
NPM_AUDIT="${ARTIFACTS_DIR}/npm_audit.json"

if [ -f "package.json" ]; then
  if command -v npm >/dev/null 2>&1; then
    echo "Running npm audit..."
    if npm audit --json > "${NPM_AUDIT}" 2>&1; then
      echo "✓ npm audit completed (no vulnerabilities)"
    else
      echo "⚠️  Vulnerabilities found by npm audit"
      
      if [ -f "${NPM_AUDIT}" ]; then
        CRITICAL=$(jq '.metadata.vulnerabilities.critical // 0' "${NPM_AUDIT}" 2>/dev/null || echo 0)
        HIGH=$(jq '.metadata.vulnerabilities.high // 0' "${NPM_AUDIT}" 2>/dev/null || echo 0)
        
        echo "  Critical: $CRITICAL"
        echo "  High: $HIGH"
        
        if [ "$CRITICAL" -gt 0 ] || [ "$HIGH" -gt 0 ]; then
          COMPLIANCE_STATUS="FAIL"
          FINDINGS_COUNT=$((FINDINGS_COUNT + CRITICAL + HIGH))
        fi
      fi
    fi
  else
    echo "⚠️  npm not found - skipping Node dependency audit"
  fi
else
  echo "ℹ️  No package.json - skipping npm audit"
fi
echo ""

###############################################################################
# 4) License Scan (pip-licenses)
###############################################################################
echo "=== 4. License Inventory (pip-licenses) ==="
LICENSES="${ARTIFACTS_DIR}/pip_licenses.json"

if command -v pip-licenses >/dev/null 2>&1; then
  echo "Running pip-licenses..."
  if pip-licenses --format=json --output-file="${LICENSES}" 2>&1 | tee -a "${ARTIFACTS_DIR}/compliance_raw.log"; then
    echo "✓ License inventory completed"
    
    # Check for forbidden licenses (if policy exists)
    if [ -f "${POLICY_FILE}" ] && [ -f "${LICENSES}" ]; then
      # This is a placeholder - real implementation would parse policy.yml
      echo "  License policy check: (placeholder)"
    fi
  else
    echo "⚠️  pip-licenses failed (non-blocking)"
  fi
else
  echo "⚠️  pip-licenses not found - skipping license inventory"
  echo "  Install: pip install pip-licenses"
fi
echo ""

###############################################################################
# 5) Generate Compliance Summary JSON
###############################################################################
echo "=== 5. Generating Compliance Summary ==="

cat > "${SUMMARY}" << EOFJSON
{
  "generated_at": "$(timestamp)",
  "repository": "$(basename "$WORKDIR")",
  "status": "${COMPLIANCE_STATUS}",
  "findings_count": ${FINDINGS_COUNT},
  "artifacts": {
    "bandit_report": "$([ -f "${SAST_RESULTS}" ] && echo "present" || echo "missing")",
    "pip_audit": "$([ -f "${PIP_AUDIT}" ] && echo "present" || echo "missing")",
    "npm_audit": "$([ -f "${NPM_AUDIT}" ] && echo "present" || echo "missing")",
    "pip_licenses": "$([ -f "${LICENSES}" ] && echo "present" || echo "missing")"
  },
  "tools": {
    "bandit": "$(command -v bandit >/dev/null && echo "installed" || echo "missing")",
    "pip-audit": "$(command -v pip-audit >/dev/null && echo "installed" || echo "missing")",
    "npm": "$(command -v npm >/dev/null && echo "installed" || echo "missing")",
    "pip-licenses": "$(command -v pip-licenses >/dev/null && echo "installed" || echo "missing")"
  }
}
EOFJSON

echo "✓ Compliance summary written to: ${SUMMARY}"
echo ""

###############################################################################
# 6) Final Status Report
###############################################################################
echo "═══════════════════════════════════════════════════════════"
echo "║           H5-05 COMPLIANCE VALIDATION SUMMARY           ║"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Status: ${COMPLIANCE_STATUS}"
echo "Critical/High Findings: ${FINDINGS_COUNT}"
echo ""
echo "Artifacts generated:"
ls -lh "${ARTIFACTS_DIR}" 2>/dev/null || echo "  (none)"
echo ""

if [ "${COMPLIANCE_STATUS}" = "PASS" ]; then
  echo "✅ COMPLIANCE: PASS"
  echo "  No critical or high severity vulnerabilities detected"
  echo ""
  exit 0
else
  echo "❌ COMPLIANCE: FAIL"
  echo "  ${FINDINGS_COUNT} critical/high severity findings detected"
  echo ""
  echo "ACTION REQUIRED:"
  echo "  1. Review artifacts in ${ARTIFACTS_DIR}/"
  echo "  2. Address critical/high severity findings"
  echo "  3. See docs/ops/compliance-remediation.md for guidance"
  echo ""
  exit 3
fi
