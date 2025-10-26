# Private Content Policy - SRC Research Lab

**Document Version**: 1.0
**Last Updated**: 2025-10-26
**Repository**: src-research-lab
**Owner**: Engineering Lead / Security Team

---

## Purpose

This document defines which content and file patterns must NEVER be included in the src-research-lab repository or its releases. This repository is intended for public release and must remain isolated from private components in the parent compression-lab workspace.

---

## Isolation Policy

###Critical Principle
**The src-research-lab repository MUST remain completely isolated from private content in the parent workspace.**

This means:
- ❌ NO file references to parent directories (`../`)
- ❌ NO absolute paths to compression-lab
- ❌ NO git submodules pointing to private repos
- ❌ NO imports from parent repo code
- ✅ All dependencies declared explicitly in requirements.txt
- ✅ Self-contained within src-research-lab directory

---

## Private Patterns to Block

### 1. Parent Repository Private Components

**Patterns**:
- `src_engine_private`
- `../src_engine_private`
- `/home/*/compression-lab/src_engine_private`
- Any path containing `..` that escapes src-research-lab

**Rationale**: Proprietary compression engine implementation

**Action**: Never reference, import, or include

---

### 2. Private Test Data & Results

**Patterns**:
- `results/private_runs`
- `../results/private_runs`
- Any file with `private` in path

**Rationale**: May contain customer data or proprietary test results

**Action**: Use synthetic/anonymized data only

---

### 3. Parent Workspace Artifacts

**Patterns**:
- `../energy/datasets`
- `../quarantine`
- `../models/proprietary`
- Any reference to parent workspace directories

**Rationale**: Private research data and internal artifacts

**Action**: Document as text if needed for context, never include files

---

### 4. Credentials & Secrets

**Patterns**:
- `.env`
- `.env.local`
- `*.key` (private keys)
- `*.pem` (certificates)
- `credentials.json`
- `secrets/`
- `.secrets`

**Rationale**: Security credentials

**Action**: Never commit. Use environment variables or secret management

---

## Enforcement

### Automated Checks

**Script**: `scripts/check_private_exposure.sh`

**What it does**:
1. Scans git-tracked files for private patterns
2. Checks for parent directory references (`../`)
3. Validates no absolute paths to parent workspace
4. Exits non-zero if violations found

**Run before every commit**:
```bash
./scripts/check_private_exposure.sh
```

**CI Integration**: Runs automatically in `h5_compliance_gate` workflow

---

### Manual Review Checklist

Before committing any new file:
- [ ] File is within src-research-lab directory
- [ ] No imports from `../` paths
- [ ] No absolute paths to compression-lab
- [ ] No proprietary algorithms or data
- [ ] No customer information
- [ ] No hardcoded secrets

---

## What CAN Be Included

### ✅ Safe Content

- Public-facing documentation
- Synthetic test data
- Research algorithms (non-proprietary)
- Bridge SDK (if approved for public release)
- Energy profiling framework (generic parts)
- Runtime guards (public implementation)
- CI/CD automation scripts
- Templates and examples

### ✅ References as Documentation

You CAN document private components as TEXT in:
- Architecture diagrams (describing integration points)
- Documentation explaining system context
- Decision logs referencing private components by name

**Example** (ALLOWED):
```markdown
The src-research-lab integrates with the proprietary compression 
engine (src_engine_private) via a bridge SDK. The private engine 
is not included in this repository.
```

**Example** (NOT ALLOWED):
```python
import sys
sys.path.append('../src_engine_private')
from src_engine_private import ProprietaryAlgorithm
```

---

## Release Bundle Creation

### Safe Packaging

When creating release tarballs or distributions:

```bash
# From within src-research-lab directory
tar -czf release.tar.gz \
  --exclude=".git" \
  --exclude="__pycache__" \
  --exclude="*.pyc" \
  --exclude=".pytest_cache" \
  --exclude="node_modules" \
  --exclude=".venv" \
  --exclude="venv" \
  .
```

### Verification

After creating bundle:
```bash
# Extract to temp location
mkdir /tmp/verify
tar -xzf release.tar.gz -C /tmp/verify

# Run exposure check
cd /tmp/verify/src-research-lab
./scripts/check_private_exposure.sh
```

---

## Incident Response

### If Private Content is Accidentally Committed

**Immediate Actions**:
1. **DO NOT PUSH** if not yet pushed
2. **Revert the commit**: `git reset --hard HEAD~1`
3. **If already pushed**:
   - Contact repo admin immediately
   - Prepare to force-push after review
   - Rotate any exposed credentials

**GitHub-Specific**:
- If pushed to public GitHub, use `git filter-branch` or BFG Repo-Cleaner
- Rotate all exposed secrets immediately
- Notify security team

---

## Training & Onboarding

### New Team Members

Before contributing:
- [ ] Read this document
- [ ] Run `scripts/check_private_exposure.sh` locally
- [ ] Understand isolation policy
- [ ] Know what can/cannot be included
- [ ] Set up pre-commit hooks (if available)

---

## Compliance Checklist

Before any public release:
- [ ] Ran `scripts/check_private_exposure.sh` successfully
- [ ] No parent directory references
- [ ] No absolute paths to compression-lab
- [ ] No proprietary algorithms
- [ ] No customer data
- [ ] No hardcoded secrets
- [ ] CI `h5_compliance_gate` passing
- [ ] Security review completed

---

## Related Documents

- [H5 Status Board](../H5-status-board.md)
- [Decision Log](../decision-log.md) (DEC-2025-10-26-003)
- [Branch Protection](branch-protection.md)
- CI Workflow: `.github/workflows/h5-gates.yml`
- Exposure Guard: `scripts/check_private_exposure.sh`

---

## Contact

**For questions about this policy**:
- Security Team: `@security-team`
- Engineering Lead: `@tech-lead`
- Product Owner: `@product-owner`

---

**Document Status**: Active
**Next Review**: 2025-11-26
**Change History**:
- 2025-10-26: Initial version (v1.0)
