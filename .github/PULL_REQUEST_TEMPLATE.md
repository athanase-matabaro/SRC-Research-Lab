# Pull Request

## Linked Feature

**Feature Spec**: `<Link to feature spec or H5-XX>`
**Feature ID**: `<H5-XX or component ID>`
**Related Issues**: Closes #`<issue number>`

---

## Summary

### What does this PR do?
`<Concise description of changes>`

### Why is this change needed?
`<Problem this PR solves>`

### How does it work?
`<High-level technical approach>`

---

## Type of Change

- [ ] 🐛 Bug fix
- [ ] ✨ New feature
- [ ] 💥 Breaking change
- [ ] 📝 Documentation update
- [ ] 🔧 Configuration change (CI/CD, build scripts)
- [ ] ♻️ Refactoring
- [ ] ⚡ Performance improvement
- [ ] 🔒 Security fix
- [ ] 🧪 Test coverage improvement

---

## Testing Instructions

### Prerequisites
`<Setup requirements>`

### How to Test

**Manual Testing**:
1. Step 1: `<Action>`
2. Step 2: `<Action>`
3. Expected: `<Result>`

**Automated Tests**:
```bash
pytest tests/
```

### Test Coverage
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] E2E tests added/updated

---

## Compliance Checklist

**Required for all PRs**:

### Code Quality
- [ ] Code follows project style guidelines
- [ ] No commented-out code or debug statements
- [ ] Code is self-documenting with clear comments
- [ ] No new warnings introduced

### Security
- [ ] No hardcoded secrets or credentials
- [ ] Input validation for user-facing changes
- [ ] Authentication/authorization properly enforced
- [ ] Dependencies scanned for vulnerabilities

### Privacy & Data
- [ ] No new PII collection without review
- [ ] Data retention policies applied
- [ ] Audit logging for sensitive operations

### Documentation
- [ ] README updated (if user-facing changes)
- [ ] API documentation updated
- [ ] Inline comments for complex logic
- [ ] Migration guide (if breaking changes)

### Testing
- [ ] All existing tests pass
- [ ] New tests for new functionality
- [ ] Test coverage meets standards (≥80%)
- [ ] Edge cases covered

### CI/CD Gates
- [ ] `h5_compliance_gate` status check passes
- [ ] All required CI checks pass
- [ ] No merge conflicts with target branch
- [ ] Branch up-to-date with `main`

---

## Reviewers

**Required Reviewers**:
- [ ] @<owner> - Product Owner
- [ ] @<tech-lead> - Technical review
- [ ] @<security> - Security review (if security-related)

---

## Release Notes

### User-Facing Changes

**Feature**: `<Feature name>`

**What's New**:
- `<User-visible change>`

**Breaking Changes**:
- `<Description>` - Migration: `<Steps>`

---

## Deployment Plan

- [ ] Standard deployment (merge and deploy)
- [ ] Feature flag rollout
- [ ] Requires maintenance window

### Pre-Deployment Checklist
- [ ] Database migrations tested
- [ ] Rollback plan documented
- [ ] Monitoring configured
- [ ] Stakeholders notified

---

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| `<Risk>` | `<H/M/L>` | `<Strategy>` |

---

## Checklist Before Merging

- [ ] All reviewer comments addressed
- [ ] CI/CD pipeline green
- [ ] Feature spec updated
- [ ] Decision log updated (if needed)
- [ ] H5 status board updated
- [ ] Release notes drafted

---

**PR Author**: @`<your-handle>`
**Created**: `<YYYY-MM-DD>`
**Target Branch**: `main`
**Labels**: `<Add relevant labels>`
