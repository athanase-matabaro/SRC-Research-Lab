# Scripts Directory

Executable scripts for the SRC Research Lab project.

## Structure

### User Scripts

- **[src_bridge.py](src_bridge.py)** - Secure interface to SRC Engine (legacy, use bridge_sdk instead)
- **[run_baseline.py](run_baseline.py)** - Baseline benchmarking script
- **[complete_phase_h1.sh](complete_phase_h1.sh)** - Phase H.1 completion helper script

### Validation Scripts ([validation/](validation/))

- **[validate_bridge.py](validation/validate_bridge.py)** - Comprehensive validation suite (9 tests)
  - Unit tests
  - SDK import sanity
  - CLI roundtrip
  - Path traversal prevention
  - Manifest validation
  - Timeout handling
  - Network prevention
  - Benchmark execution
  - Determinism verification

## Usage

### Run Baseline Benchmarks

```bash
python3 scripts/run_baseline.py
```

Output: `results/baseline_benchmark.json`

### Run Bridge Validation

```bash
# Full validation suite (9 tests)
python3 scripts/validation/validate_bridge.py --full

# Individual tests
python3 scripts/validation/validate_bridge.py --no-network-test
python3 scripts/validation/validate_bridge.py --run-timeout-test
```

Output: `bridge_validation.log`

### Phase H.1 Completion Script

```bash
# Interactive helper for completing Phase H.1
./scripts/complete_phase_h1.sh
```

This script will:
1. Verify unit tests
2. Check SDK import
3. Push commits
4. Help create PR
5. Run full validation (optional)

## Adding New Scripts

When adding new scripts:

1. **Placement:**
   - User-facing scripts → `scripts/`
   - Test/validation scripts → `scripts/validation/`
   - Build/deployment scripts → `scripts/build/` (create if needed)

2. **Naming:**
   - Use `verb_noun.py` format (e.g., `run_baseline.py`, `validate_bridge.py`)
   - Use descriptive names
   - Prefix with action verb (run, validate, build, deploy)

3. **Requirements:**
   - Add shebang: `#!/usr/bin/env python3` or `#!/bin/bash`
   - Make executable: `chmod +x script_name.py`
   - Add docstring/header comment
   - Include usage examples in `--help`

4. **Documentation:**
   - Update this README
   - Add usage examples
   - Document required dependencies

## Script Guidelines

### Python Scripts

```python
#!/usr/bin/env python3
"""
Brief description of what this script does.

Usage:
    python3 scripts/script_name.py [options]

Options:
    --option1    Description of option1
    --option2    Description of option2
"""

import argparse

def main():
    parser = argparse.ArgumentParser(description="...")
    # ... argument parsing
    
if __name__ == "__main__":
    main()
```

### Bash Scripts

```bash
#!/bin/bash
# Brief description of what this script does
#
# Usage: ./scripts/script_name.sh [options]

set -e  # Exit on error

# Script content...
```

## Maintenance

- Keep scripts updated with code changes
- Test scripts before committing
- Document breaking changes
- Archive deprecated scripts to `scripts/archive/`

---

**Last Updated:** 2025-10-13
**Maintained by:** SRC Research Lab
