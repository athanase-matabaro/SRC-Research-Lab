# Decision Log - SRC Research Lab

## Purpose

Records significant architectural, technical, and process decisions for the src-research-lab repository.

## Decision Template

```markdown
## DEC-YYYY-MM-DD-XXX: [Decision Title]

**Date**: YYYY-MM-DD
**Status**: [Proposed | Accepted | Superseded]
**Deciders**: @person1, @person2
**Context**: [Feature/Phase ID]

### Context
What problem are we solving?

### Decision
What change are we making?

### Alternatives Considered
1. **Alternative 1**: [Description] - [Why rejected]

### Rationale
Why is this the best decision?

### Consequences
- **Positive**: [Benefits]
- **Negative**: [Drawbacks]
- **Neutral**: [Other impacts]

### Related
- Feature Spec: [Link]
- PRs: [Links]
```

---

## Decisions

### DEC-2025-10-26-001: Adopt CI/CD Compliance Gate for H5 Phase

**Date**: 2025-10-26
**Status**: Accepted
**Deciders**: @engineering-team, @product-owner
**Context**: Phase H5 (Robustness Hardening & Compliance)

#### Context
Phase H5 requires automated compliance enforcement before code reaches production. Manual reviews don't scale and are error-prone.

#### Decision
Implement required CI/CD status check named `h5_compliance_gate` that runs on all PRs to main.

The gate includes:
1. Test execution (`scripts/run_tests.sh`)
2. Compliance checks (`scripts/compliance_check.sh`)
3. Private content exposure guard (`scripts/check_private_exposure.sh`)

#### Alternatives Considered

1. **Manual Compliance Reviews Only**
   - Rejected: Too slow, doesn't scale, human error risk

2. **Post-Merge Audits**
   - Rejected: Too late to catch issues, expensive rollbacks

3. **Compliance Gate Only on Release Branches**
   - Rejected: Issues discovered too late in cycle

#### Rationale
- **Prevention over Detection**: Catch issues before merge
- **Automation**: Ensures consistency, reduces review burden
- **Shift-Left**: Earlier feedback, faster fixes, lower cost
- **Compliance**: Required for production deployment

#### Consequences

**Positive**:
- Automated enforcement reduces risk
- Faster PR reviews (pre-checked)
- Clear feedback for developers
- Audit trail for compliance
- Scales with team growth

**Negative**:
- Initial setup effort
- Potential false positives requiring tuning
- Adds 2-5 minutes to PR workflow
- Requires script maintenance

**Neutral**:
- Team must learn gate requirements
- Developer workflow documentation needs updates

#### Related
- Feature Spec: H5-01 (CI/CD Compliance Gate)
- Workflow: `.github/workflows/h5-gates.yml`
- Scripts: `scripts/compliance_check.sh`, `scripts/check_private_exposure.sh`

---

### DEC-2025-10-26-002: Require Feature Spec Linkage in H5 PRs

**Date**: 2025-10-26
**Status**: Accepted
**Deciders**: @product-owner, @tech-lead
**Context**: Phase H5 (Robustness Hardening & Compliance)

#### Context
Need clear traceability from requirements to code for H5 deliverables. Without formal specs:
- Risk building features that don't align with goals
- Incomplete acceptance criteria leading to rework
- Difficulty auditing what was delivered
- Lack of release notes

#### Decision
All PRs for H5 work items must link to approved feature spec (using `docs/templates/feature-spec.md`) in PR description.

**Requirements**:
1. Feature spec in `docs/features/` or `docs/templates/`
2. PR includes `Feature Spec: <link>` or `Feature ID: H5-XX`
3. Spec has "Accepted" status before PR merge
4. Breaking changes called out explicitly

**Exemptions**:
- Hotfixes (must be documented post-merge)
- Internal tooling not user-facing
- Documentation-only changes

#### Alternatives Considered

1. **Optional Feature Specs**
   - Rejected: Creates inconsistency, defeats traceability

2. **Specs Only for "Large" Features**
   - Rejected: "Large" is subjective, small features need criteria too

3. **Inline Spec in GitHub Issues**
   - Rejected: Not structured for compliance audits, hard to search

#### Rationale
- **Traceability**: Clear link from need → spec → code → release
- **Quality**: Forces upfront thinking about requirements
- **Collaboration**: Enables async stakeholder review
- **Compliance**: Required audit trail
- **Release Management**: Specs feed release notes
- **Onboarding**: New members understand features via specs

#### Consequences

**Positive**:
- Reduced rework from clearer requirements
- Better stakeholder alignment
- Automatic release notes generation
- Audit compliance
- Knowledge base for support

**Negative**:
- Upfront time (30-60 min per feature)
- Learning curve for template
- Discipline required to keep specs updated

**Neutral**:
- Cultural shift to documentation-first
- PM workload shifts earlier in cycle

#### Related
- Template: `docs/templates/feature-spec.md`
- PR Template: `.github/PULL_REQUEST_TEMPLATE.md`
- H5 Status Board: `docs/H5-status-board.md`

---

### DEC-2025-10-26-003: Isolate SRC Research Lab from Parent Repository Private Content

**Date**: 2025-10-26
**Status**: Accepted
**Deciders**: @security-team, @product-owner
**Context**: Phase H5 (Privacy & Security Compliance)

#### Context
The src-research-lab repository is intended for public release, but resides within a larger private compression-lab workspace. Must ensure no private content from parent repo leaks into public releases.

#### Decision
Implement strict isolation:
1. All H5 artifacts contained within src-research-lab only
2. No file references to parent directories (no `../`)
3. No absolute paths to compression-lab
4. Private paths listed as TEXT in docs, not included as files
5. `scripts/check_private_exposure.sh` validates no private references in git tree

**Private patterns to block**:
- `src_engine_private`
- `results/private_runs`
- `../src_engine_private`
- `/home/*/compression-lab/src_engine_private`

#### Alternatives Considered

1. **Allow Relative References to Parent**
   - Rejected: Creates dependency on private repo structure

2. **Submodule Links to Private Components**
   - Rejected: Would expose private repo URLs in public git history

3. **Shared Scripts in Parent Repo**
   - Rejected: Breaks isolation, creates dependency

#### Rationale
- **Security**: Prevents accidental exposure of proprietary code
- **Independence**: src-research-lab can be open-sourced safely
- **Compliance**: Required for public release
- **Maintainability**: Clear boundary between public/private

#### Consequences

**Positive**:
- Safe for public release
- Clear ownership boundary
- Independent development velocity
- Audit trail for compliance

**Negative**:
- Some code duplication between repos
- Can't directly import parent repo utilities
- Must coordinate breaking changes

**Neutral**:
- Requires discipline in development
- CI enforces isolation automatically

#### Related
- Script: `scripts/check_private_exposure.sh`
- Documentation: `docs/ops/PRIVATE_CONTENT.md`
- CI Check: `.github/workflows/h5-gates.yml`

---

## Decision Summary Table

| ID | Date | Title | Status | Impact |
|----|------|-------|--------|--------|
| DEC-2025-10-26-003 | 2025-10-26 | Isolate from Parent Repo | Accepted | High |
| DEC-2025-10-26-002 | 2025-10-26 | Require Feature Spec Linkage | Accepted | High |
| DEC-2025-10-26-001 | 2025-10-26 | Adopt CI/CD Compliance Gate | Accepted | High |

---

**Log Version**: 1.0
**Last Updated**: 2025-10-26
**Next Review**: Monthly
