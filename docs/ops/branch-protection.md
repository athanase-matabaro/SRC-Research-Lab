# Branch Protection Requirements - SRC Research Lab

**Document Version**: 1.0
**Last Updated**: 2025-10-26
**Repository**: src-research-lab
**Owner**: Engineering Lead / DevOps

---

## Purpose

Defines required branch protection rules for the src-research-lab repository to enforce code quality, security, and compliance for Phase H5.

---

## Protected Branches

- `main` - Primary production branch (MUST be protected)
- `develop` - Integration branch (if using GitFlow)
- `release/*` - Release branches

---

## Required Protection Rules for `main`

### 1. Require Pull Request Reviews

**Configuration**:
- Required approving reviews: **1 minimum**
- Dismiss stale reviews when new commits pushed: **Enabled**
- Require review from Code Owners: **Enabled** (if CODEOWNERS exists)

### 2. Require Status Checks

**Required Checks**:
- ✅ `h5_compliance_gate` - **CRITICAL** - Compliance and security
- ✅ `test` - Unit and integration tests (if separate job)
- ✅ `lint` - Code quality (if configured)

**Configuration**:
- Require branches to be up to date before merging: **Enabled**
- Status checks must pass on latest commit

### 3. Restrict Direct Pushes

**Configuration**:
- Allow force pushes: **Disabled**
- Allow deletions: **Disabled**
- Restrict push access: **Only via pull requests**
- Include administrators: **Enabled** (admins follow same rules)

### 4. Require Linear History

**Configuration**:
- Require linear history: **Enabled**
- Merge strategy: **Squash and merge** or **Rebase and merge**

---

## Implementation Instructions

### Using GitHub Web Interface

1. Navigate to: **Settings** → **Branches** → **Branch protection rules**
2. Click **Add rule** for `main`
3. Configure as specified above
4. Save changes

### Using GitHub CLI

```bash
# Protect main branch
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --field required_status_checks[strict]=true \
  --field required_status_checks[contexts][]=h5_compliance_gate \
  --field required_pull_request_reviews[required_approving_review_count]=1 \
  --field required_pull_request_reviews[dismiss_stale_reviews]=true \
  --field enforce_admins=true \
  --field required_linear_history=true \
  --field allow_force_pushes=false \
  --field allow_deletions=false
```

### Using Terraform

```hcl
resource "github_branch_protection" "main" {
  repository_id = github_repository.src_research_lab.node_id
  pattern       = "main"

  required_pull_request_reviews {
    required_approving_review_count = 1
    dismiss_stale_reviews           = true
  }

  required_status_checks {
    strict = true
    contexts = [
      "h5_compliance_gate"
    ]
  }

  enforce_admins                  = true
  required_linear_history         = true
  allow_force_pushes              = false
  allow_deletions                 = false
  require_conversation_resolution = true
}
```

---

## Pull Request Requirements

All PRs to `main` must satisfy:

### 1. Feature Spec Linkage
- [ ] PR links to feature spec or H5 work item
- [ ] Spec has "Accepted" status
- [ ] Enforcement: Manual review

### 2. Review Requirements
- [ ] At least 1 approving review
- [ ] Reviewers: Tech lead, feature owner, or designated reviewer
- [ ] All comments resolved

### 3. CI/CD Gates
- [ ] `h5_compliance_gate` status check passes
- [ ] All tests passing
- [ ] No merge conflicts
- [ ] Branch up-to-date with `main`

### 4. Documentation
- [ ] README updated (if needed)
- [ ] API docs updated (if needed)
- [ ] Release notes drafted

---

## Compliance Gate Details

The `h5_compliance_gate` status check is **CRITICAL** and runs:

1. **Test Suite** (`scripts/run_tests.sh`)
   - Pytest for Python tests
   - npm test if package.json exists
   - make test if Makefile exists

2. **Compliance Checks** (`scripts/compliance_check.sh`)
   - Security scanning (if configured)
   - Dependency vulnerabilities
   - Code quality checks

3. **Private Exposure Guard** (`scripts/check_private_exposure.sh`)
   - Validates no private content references
   - Checks for parent directory escapes
   - Ensures repository isolation

---

## Exemption Process

### Hotfixes

For urgent fixes:
1. Create branch: `hotfix/description`
2. Add label: `hotfix`
3. Request emergency approval
4. Still require `h5_compliance_gate` to pass
5. Retrospective review within 24 hours

### Infrastructure Changes

For CI/CD changes:
1. Add label: `infrastructure`
2. Document why compliance may not fully apply
3. Still require code review
4. Update compliance scripts in follow-up if needed

---

## Monitoring

### Weekly Review
- Failed merge attempts (why?)
- Bypassed protections (who and why?)
- Compliance gate failures (patterns?)

### Monthly Audit
- All merged PRs
- Compliance gate effectiveness
- Protection rule violations
- Exemption patterns

---

## Troubleshooting

### "Required status check is not passing"

**Problem**: `h5_compliance_gate` failing

**Solutions**:
1. Check GitHub Actions logs
2. Run `scripts/compliance_check.sh` locally
3. Run `scripts/check_private_exposure.sh` locally
4. Address specific failure
5. Push fix and wait for re-run

### "Branch is not up to date"

**Problem**: Branch behind `main`

**Solutions**:
```bash
git fetch origin main
git rebase origin/main
# or
git merge origin/main

git push --force-with-lease
```

---

## Related Documentation

- [H5 Status Board](../H5-status-board.md)
- [Feature Spec Template](../templates/feature-spec.md)
- [PR Template](../../.github/PULL_REQUEST_TEMPLATE.md)
- [Decision Log](../decision-log.md)
- [Private Content Policy](PRIVATE_CONTENT.md)
- CI Workflow: `.github/workflows/h5-gates.yml`

---

**Next Review**: 2025-11-26
**Change History**:
- 2025-10-26: Initial version for H5 phase
