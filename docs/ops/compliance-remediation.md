# Compliance Remediation Runbook

**Document Version**: 1.0  
**Last Updated**: 2025-10-27  
**Owner**: Security Team / Engineering Lead

---

## Purpose

This runbook provides step-by-step guidance for triaging and remediating compliance issues identified by the H5-05 compliance validation pipeline.

---

## Severity Handling

### Critical Severity

**Action**: ❌ **BLOCK MERGE IMMEDIATELY**

**Process**:
1. Create GitHub issue with label `security-critical`
2. Assign to: Security Team + Engineering Lead
3. Notify via Slack: `#security-incidents`
4. **DO NOT MERGE** until resolved

**Timeline**: Must be resolved within 24 hours

**Typical Remediation**:
- Pin to patched version: `pip install 'package==safe.version'`
- Remove vulnerable dependency entirely
- Apply vendor security patch
- If no fix available: Replace with alternative package

---

### High Severity

**Action**: ⚠️ **BLOCK MERGE** (unless exception approved)

**Process**:
1. Create GitHub issue with label `security-high`
2. Assign to: Engineering Team
3. Request security review for exception (if needed)

**Timeline**: Must be resolved within 3 business days

**Typical Remediation**:
- Upgrade to patched version
- Backport security patch
- Isolate component behind feature flag
- Request temporary exception with mitigation plan

---

### Medium Severity

**Action**: ⚠️ **May proceed with approval**

**Process**:
1. Document in issue tracker
2. Assess risk and impact
3. Create remediation plan
4. May merge with tech lead approval + follow-up ticket

**Timeline**: Remediate within 2 weeks

---

### Low Severity

**Action**: ℹ️ **Informational**

**Process**:
1. Create tracking issue
2. Prioritize in backlog
3. Can merge without immediate remediation

**Timeline**: Best effort

---

## Common Remediation Strategies

### 1. Upgrade Dependency

**Python**:
```bash
# Find safe version
pip install 'package>=safe.version'

# Update requirements.txt
pip freeze | grep package >> requirements.txt

# Or update pyproject.toml
# [tool.poetry.dependencies]
# package = "^safe.version"
```

**Node**:
```bash
# Upgrade to safe version
npm install package@safe.version

# Update lockfile
npm audit fix
```

### 2. Replace Vulnerable Package

**Process**:
1. Research alternatives: Check https://www.npmjs.com/, PyPI
2. Evaluate alternatives for:
   - Active maintenance
   - Security track record
   - License compatibility
   - Feature parity
3. Replace imports and update code
4. Test thoroughly

### 3. Apply Vendor Patch

**If vendor provides patch**:
```bash
# Fork the repository
# Apply patch locally
# Use git submodule or vendored copy
# Document in compliance/exceptions.yml
```

### 4. Isolate Vulnerable Component

**Temporary mitigation**:
- Wrap vulnerable code in feature flag
- Disable functionality until patched
- Add runtime guards
- Monitor for exploits

---

## License Compliance Issues

### Forbidden License Detected

**Action**: ❌ **BLOCK MERGE**

**Process**:
1. Identify package with forbidden license
2. Options:
   - **Option A**: Replace with package under acceptable license
   - **Option B**: Request legal exemption (rare)
   - **Option C**: Remove functionality

**Example**:
```bash
# Find packages with GPL license
pip-licenses | grep GPL

# Replace with MIT-licensed alternative
pip uninstall problematic-package
pip install alternative-package
```

### Unknown License

**Action**: ⚠️ **Requires Review**

**Process**:
1. Research package license on GitHub/PyPI
2. Contact package maintainer if unclear
3. Document findings
4. Request legal review if still uncertain

---

## Adding Compliance Exceptions

### When to Use Exceptions

- False positive from scanner
- Vulnerability not exploitable in our use case
- Waiting for upstream patch
- Legacy code with complex migration

### Exception Format

Edit `compliance/exceptions.yml`:

```yaml
exceptions:
  CVE-2024-12345:
    package: vulnerable-package==1.2.3
    reason: "False positive - we only use safe API subset"
    risk_assessment: "Low - vulnerable function not called"
    mitigation: "Input validation added; monitoring enabled"
    approved_by: "@security-team"
    approved_date: "2025-10-27"
    expires: "2026-01-27"
    tracking_issue: "#123"
```

### Approval Process

1. Engineer documents exception with justification
2. Security team reviews risk assessment
3. Security lead approves with expiry date
4. Exception tracked in issue tracker
5. Automated reminder before expiry

---

## Tools & Commands

### Run Full Compliance Check

```bash
cd /path/to/src-research-lab
./scripts/validate_compliance.sh
```

### Check Specific Tool

**SAST (Bandit)**:
```bash
bandit -r . -f json -o artifacts/bandit_report.json
```

**Python Audit**:
```bash
pip-audit --format json --output artifacts/pip_audit.json
```

**Node Audit**:
```bash
npm audit --json > artifacts/npm_audit.json
```

**License Inventory**:
```bash
pip-licenses --format=json --output-file artifacts/pip_licenses.json
```

### View Artifacts

```bash
# Summary
cat artifacts/compliance_summary.json | jq

# Detailed findings
cat artifacts/bandit_report.json | jq '.results[] | select(.issue_severity == "HIGH")'
cat artifacts/pip_audit.json | jq '.vulnerabilities[] | select(.severity == "critical")'
```

---

## Escalation Paths

### Security Incident

**Triggers**:
- Critical vulnerability in production code
- Active exploit detected
- Data exposure risk

**Action**:
1. Notify: `@security-team` immediately
2. Create incident ticket
3. Follow security incident response plan
4. Consider hotfix deployment

### Legal/License Issue

**Triggers**:
- GPL/AGPL license detected
- Unclear license terms
- Copyright dispute

**Action**:
1. Notify: `@legal-team`
2. Halt deployment
3. Document usage and exposure
4. Await legal guidance

### Blocked Release

**Triggers**:
- Compliance gate failing repeatedly
- No clear remediation path
- Disagreement on risk assessment

**Action**:
1. Escalate to: Engineering Manager
2. Schedule review meeting
3. Document options and tradeoffs
4. Executive decision required

---

## Monitoring & Reporting

### Weekly Review

**Security Team**:
- [ ] Review open compliance issues
- [ ] Check exception expiry dates
- [ ] Audit new dependencies
- [ ] Update policy if needed

### Monthly Metrics

**Track**:
- Compliance gate pass rate
- Time to remediate by severity
- Exception usage trends
- Recurring vulnerabilities

**Report to**: Engineering leadership

---

## Best Practices

### Preventive Measures

1. **Review dependencies before adding**
   - Check security track record
   - Verify license compatibility
   - Assess maintenance activity

2. **Pin dependency versions**
   - Use lockfiles (requirements.txt, package-lock.json)
   - Test before upgrading
   - Review changelogs

3. **Automate updates**
   - Enable Dependabot/Renovate
   - Configure auto-merge for patches
   - Review major version updates

4. **Regular audits**
   - Run compliance checks locally before PR
   - Include in CI/CD pipeline
   - Schedule manual reviews

### Development Workflow

1. Before adding dependency:
   ```bash
   # Check license
   pip show package-name | grep License
   
   # Check for known vulnerabilities
   pip-audit package-name
   ```

2. Before committing:
   ```bash
   ./scripts/validate_compliance.sh
   ```

3. In PR review:
   - Review compliance artifacts
   - Check for new dependencies
   - Verify license compatibility

---

## References

- **H5 Status Board**: [docs/H5-status-board.md](../H5-status-board.md)
- **Compliance Policy**: [compliance/policy.yml](../../compliance/policy.yml)
- **Decision Log**: [docs/decision-log.md](../decision-log.md) (DEC-2025-10-26-001)
- **Private Content Policy**: [docs/ops/PRIVATE_CONTENT.md](PRIVATE_CONTENT.md)
- **CI Workflow**: [.github/workflows/h5-gates.yml](../../.github/workflows/h5-gates.yml)

### External Resources

- **CVE Database**: https://cve.mitre.org/
- **Python Security**: https://pypi.org/project/pip-audit/
- **NPM Security**: https://docs.npmjs.com/auditing-package-dependencies-for-security-vulnerabilities
- **License Guide**: https://choosealicense.com/

---

**Document Status**: Active  
**Next Review**: 2025-11-27  
**Change History**:
- 2025-10-27: Initial version (v1.0) - H5-05 implementation
