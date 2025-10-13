# Engineering Culture & Contribution Guidelines

This document defines the engineering practices, naming conventions, and contribution workflow for the SRC Research Lab project.

## Table of Contents

1. [Branch Naming Conventions](#branch-naming-conventions)
2. [Commit Message Guidelines](#commit-message-guidelines)
3. [Code Organization](#code-organization)
4. [Pull Request Process](#pull-request-process)
5. [Code Quality Standards](#code-quality-standards)
6. [Documentation Requirements](#documentation-requirements)

---

## Branch Naming Conventions

### Format

Use lowercase with hyphens, following this pattern:

```
<type>/<short-description>
```

### Branch Types

- `feature/` - New features or functionality
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring without feature changes
- `test/` - Adding or updating tests
- `experiment/` - Research experiments and prototypes
- `chore/` - Maintenance tasks (dependencies, configs, etc.)

### Examples

```bash
feature/add-zstd-support
fix/memory-leak-in-bridge
docs/update-api-reference
refactor/simplify-caq-metric
experiment/neural-compression
test/add-baseline-benchmarks
chore/update-dependencies
```

### Protected Branches

- `main` - Production-ready code, requires PR and review
- `develop` - Integration branch for features (if used)

---

## Commit Message Guidelines

### Format

Follow the conventional commits specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Commit Types

- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `style` - Code style changes (formatting, no logic change)
- `refactor` - Code refactoring
- `test` - Adding or updating tests
- `chore` - Maintenance tasks
- `perf` - Performance improvements
- `build` - Build system changes
- `ci` - CI/CD changes

### Rules

1. **Subject line**:
   - Use imperative mood ("add" not "added" or "adds")
   - Start with lowercase
   - No period at the end
   - Maximum 50 characters
   - Be concise and descriptive

2. **Body** (optional):
   - Wrap at 72 characters
   - Explain what and why, not how
   - Separate from subject with blank line

3. **Footer** (optional):
   - Reference issues: `Closes #123` or `Relates to #456`
   - Breaking changes: `BREAKING CHANGE: description`

### Examples

```bash
# Simple commit
feat(bridge): add zstd compression support

# With body
fix(caq): correct division by zero in edge cases

Previously, when cpu_seconds was exactly 0, the CAQ calculation
would fail. Now we use log1p to handle this case gracefully.

# With issue reference
docs(readme): add installation instructions

Closes #42

# Breaking change
refactor(api): change compress output format to JSON

BREAKING CHANGE: compress command now outputs JSON instead of
plain text. Update scripts accordingly.
```

---

## Code Organization

### Directory Structure

```
src-research-lab/
├── docs/                    # All documentation
│   ├── index.md
│   ├── engineeringculture.md
│   └── api/                 # API documentation
├── src/                     # Source code (Python modules)
│   └── bridge/              # Bridge-related code
├── scripts/                 # Executable scripts
│   ├── run_baseline.py
│   └── benchmark_*.py
├── metrics/                 # Metric implementations
│   └── caq_metric.py
├── tests/                   # Test files and fixtures
│   ├── fixtures/
│   └── test_*.py
├── results/                 # Benchmark results (gitignored)
├── experiments/             # Research experiments
├── .github/                 # GitHub workflows and templates
├── README.md
├── LICENSE.md
├── .gitignore
└── .claude.md              # AI assistant context
```

### File Placement Rules

1. **Scripts** - Executable Python scripts go in `scripts/`
2. **Source code** - Reusable Python modules go in `src/`
3. **Tests** - All test files go in `tests/`
4. **Documentation** - All `.md` files (except README/LICENSE) go in `docs/`
5. **Test data** - Sample inputs/outputs go in `tests/fixtures/`
6. **Benchmarks** - Generated benchmark files go in `results/` (gitignored)
7. **Experiments** - Research code and notes go in `experiments/`

### Naming Conventions

#### Python Files
- Use `snake_case` for all Python files
- Test files: `test_<module_name>.py`
- Scripts: `<action>_<noun>.py` (e.g., `run_baseline.py`)

#### Python Code
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

#### Documentation
- Use `lowercase.md` for most docs
- Use `PascalCase.md` for special docs (e.g., `CONTRIBUTING.md`)

---

## Pull Request Process

### Before Creating a PR

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with frequent, atomic commits

3. **Test thoroughly**:
   ```bash
   # Run tests
   python -m pytest tests/

   # Run benchmarks if applicable
   ./scripts/run_baseline.py
   ```

4. **Update documentation** if you changed:
   - Public APIs
   - Configuration options
   - User-facing behavior

5. **Ensure clean code**:
   ```bash
   # Format code (if using black)
   black src/ scripts/

   # Lint (if using pylint/ruff)
   pylint src/ scripts/
   ```

### Creating the PR

1. **Push your branch**:
   ```bash
   git push -u origin feature/your-feature-name
   ```

2. **Open PR** with descriptive title and body:

   **Title**: `feat(bridge): add zstd compression support`

   **Body**:
   ```markdown
   ## Summary
   - Added zstd compression to run_baseline.py
   - Updated CAQ metric calculation
   - Added tests for zstd integration

   ## Test Plan
   - [ ] Ran full benchmark suite
   - [ ] Verified CAQ scores are calculated correctly
   - [ ] Tested with various file sizes

   ## Related Issues
   Closes #42
   ```

3. **Wait for review** - Address feedback promptly

4. **Merge** - Use "Squash and merge" for clean history

---

## Code Quality Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use docstrings for all public functions/classes

### Documentation

Every public function should have a docstring:

```python
def compress_file(input_path: Path, output_path: Path, workers: int = 1) -> dict:
    """
    Compress a file using the SRC Engine.

    Args:
        input_path: Path to input file (must exist)
        output_path: Path to output file (will be created)
        workers: Number of parallel workers (default: 1)

    Returns:
        Dictionary with compression results including ratio and elapsed time

    Raises:
        ValueError: If input_path does not exist
        RuntimeError: If compression fails
    """
    pass
```

### Error Handling

- Use specific exception types
- Provide helpful error messages
- Log errors appropriately
- Never expose internal paths or sensitive data

### Security

- No hardcoded credentials
- Validate all file paths (prevent traversal)
- Sanitize error messages before showing to users
- Use subprocess safely (avoid shell=True)

---

## Documentation Requirements

### When to Update Docs

Update documentation when you:
- Add new features or commands
- Change public APIs
- Modify configuration options
- Fix bugs that affect documented behavior
- Add new metrics or benchmarks

### Documentation Locations

- **User guides** → `docs/`
- **API reference** → `docs/api/`
- **Examples** → `README.md` or `docs/examples/`
- **Architecture decisions** → `docs/architecture/`
- **Experiments** → `experiments/*/README.md`

### Documentation Style

- Use clear, concise language
- Include code examples
- Provide context and rationale
- Keep up to date with code changes

---

## Repository Hygiene

### Keep Root Clean

The root directory should contain only:
- Essential config files (`.gitignore`, `LICENSE.md`)
- Entry point documentation (`README.md`)
- Package metadata (`setup.py`, `pyproject.toml` if needed)
- AI assistant context (`.claude.md`)

Everything else belongs in subdirectories.

### Gitignore Rules

Never commit:
- Generated files (`.pyc`, `__pycache__`)
- IDE configs (`.vscode/`, `.idea/`)
- Local results (`results/*.json`)
- Test outputs (`test_*.txt`, `benchmark_*.txt`)
- Secrets or credentials

### Clean Commits

Before committing:
```bash
# Check what's staged
git status

# Review changes
git diff --staged

# Remove untracked junk
git clean -fd --dry-run  # Preview
git clean -fd            # Execute
```

---

## Workflow Example

Complete workflow for adding a new feature:

```bash
# 1. Create feature branch
git checkout -b feature/add-lz4-support

# 2. Make changes
# ... edit files ...

# 3. Test
./scripts/run_baseline.py

# 4. Commit incrementally
git add scripts/run_baseline.py
git commit -m "feat(benchmark): add lz4 compression support"

git add docs/api/compressors.md
git commit -m "docs(api): document lz4 integration"

# 5. Push branch
git push -u origin feature/add-lz4-support

# 6. Create PR on GitHub
# 7. Address review feedback
# 8. Merge when approved
```

---

## Questions?

If you have questions about these guidelines, please:
1. Check existing issues and PRs for examples
2. Ask in discussions or issues
3. Propose changes to this doc via PR

---

## References

- [Conventional Commits](https://www.conventionalcommits.org/)
- [PEP 8 Style Guide](https://pep8.org/)
- [Git Best Practices](https://git-scm.com/book/en/v2)
