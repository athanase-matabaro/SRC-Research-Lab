# Documentation Directory

This directory contains all documentation for the SRC Research Lab project.

## Structure

### Core Documentation

- **[index.md](index.md)** - Documentation index and navigation
- **[engineeringculture.md](engineeringculture.md)** - Engineering practices and contribution guidelines
- **[bridge_sdk.md](bridge_sdk.md)** - Bridge SDK API, CLI, and security documentation (800+ lines)

### Release Documentation ([release/](release/))

Official release documentation and deliverables:

- **[bridge_release_notes.md](release/bridge_release_notes.md)** - Phase H.1 release notes with deliverables
- **[PHASE_H1_COMPLETION_SUMMARY.md](release/PHASE_H1_COMPLETION_SUMMARY.md)** - Complete implementation summary
- **[PULL_REQUEST.md](release/PULL_REQUEST.md)** - Pull request description (ready to use)
- **[VALIDATION_SUCCESS.md](release/VALIDATION_SUCCESS.md)** - Validation results (9/9 tests passing)

### Validation Artifacts ([validation/](validation/))

Validation logs and test outputs:

- **[bridge_validation.log](validation/bridge_validation.log)** - Final validation run log (9/9 PASS)
- **[bridge_validation_partial.log](validation/bridge_validation_partial.log)** - Partial validation (6/9, no engine)
- **[validation_final.txt](validation/validation_final.txt)** - Full validation output
- **[validation_output.txt](validation/validation_output.txt)** - Intermediate validation output

## Quick Links

### For Users

- **Getting Started:** See [../README.md](../README.md)
- **API Reference:** [bridge_sdk.md](bridge_sdk.md)
- **Installation:** [bridge_sdk.md#installation](bridge_sdk.md#installation)
- **Examples:** [bridge_sdk.md#quickstart](bridge_sdk.md#quickstart)

### For Contributors

- **Contribution Guide:** [engineeringculture.md](engineeringculture.md)
- **Branch Naming:** [engineeringculture.md#branch-naming-conventions](engineeringculture.md#branch-naming-conventions)
- **Commit Guidelines:** [engineeringculture.md#commit-message-guidelines](engineeringculture.md#commit-message-guidelines)
- **PR Process:** [engineeringculture.md#pull-request-process](engineeringculture.md#pull-request-process)

### For Reviewers

- **Release Notes:** [release/bridge_release_notes.md](release/bridge_release_notes.md)
- **Validation Results:** [release/VALIDATION_SUCCESS.md](release/VALIDATION_SUCCESS.md)
- **Completion Summary:** [release/PHASE_H1_COMPLETION_SUMMARY.md](release/PHASE_H1_COMPLETION_SUMMARY.md)

## Documentation Standards

All documentation follows these standards:

1. **Markdown format** - GitHub-flavored markdown
2. **Clear structure** - Table of contents, sections, subsections
3. **Code examples** - Working examples with expected output
4. **Cross-references** - Links to related documentation
5. **Up-to-date** - Synchronized with code changes

## Adding Documentation

When adding new documentation:

1. Place in appropriate directory:
   - Core docs → `docs/`
   - Release docs → `docs/release/`
   - Validation logs → `docs/validation/`

2. Update this README with links

3. Follow naming conventions:
   - Use `lowercase_with_underscores.md` for most docs
   - Use `UPPERCASE.md` for special docs (README, LICENSE)
   - Use descriptive names (not `doc1.md`, `notes.md`)

4. Include in pull request with code changes

## Maintenance

- Review docs quarterly for accuracy
- Update links when files move
- Archive old validation logs
- Keep release/ directory for historical records

---

**Last Updated:** 2025-10-13
**Maintained by:** SRC Research Lab
