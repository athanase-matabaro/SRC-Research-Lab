# Bridge SDK Documentation (Phase H.1)

**Version:** 1.0.0
**Date:** 2025-10-13
**Status:** Production Ready

## Table of Contents

1. [Overview](#overview)
2. [Quickstart](#quickstart)
3. [Installation](#installation)
4. [API Reference](#api-reference)
5. [CLI Reference](#cli-reference)
6. [Manifest Format](#manifest-format)
7. [Security Model](#security-model)
8. [Reproducibility](#reproducibility)
9. [FAQ & Troubleshooting](#faq--troubleshooting)

---

## Overview

The Bridge SDK provides a secure, manifest-driven Python API and CLI for compression tasks using the SRC Engine and reference codecs (zstd, lz4). It implements comprehensive security controls, resource limits, and CAQ (Compression-Accuracy Quotient) metric integration.

### Key Features

- **Secure by design**: Workspace-relative path validation, no network access, timeout enforcement
- **Manifest-driven**: Schema-based task validation with `bridge_manifest.yaml`
- **Multi-backend**: SRC Engine, zstd, lz4 support
- **CAQ metrics**: Automatic computation of compression quality scores
- **Reproducible**: Deterministic defaults and validation tools
- **Offline-only**: No telemetry, no external downloads, CPU-only execution

### Architecture

```
┌─────────────────┐
│   User Code     │
└────────┬────────┘
         │
    ┌────▼────────────────────┐
    │   Bridge SDK (api.py)   │
    ├─────────────────────────┤
    │ • compress()            │
    │ • decompress()          │
    │ • analyze()             │
    └────────┬────────────────┘
             │
    ┌────────▼──────────┐
    │  Security Layer   │
    │  (security.py)    │
    ├───────────────────┤
    │ • Path validation │
    │ • Timeout control │
    │ • Network prevent │
    └────────┬──────────┘
             │
    ┌────────▼────────────┐
    │  Manifest Validator │
    │  (manifest.py)      │
    └────────┬────────────┘
             │
    ┌────────▼────────────────┐
    │   SRC Engine Binary     │
    │   (../src_engine_private)│
    └─────────────────────────┘
```

---

## Quickstart

### Python API

```python
import bridge_sdk

# Compress a file
result = bridge_sdk.compress(
    "tests/fixtures/test_input.txt",
    "results/output.cxe",
    config={"care": True, "workers": 2}
)

print(f"Compression ratio: {result['ratio']:.2f}x")
print(f"CAQ score: {result['caq']:.6f}")

# Decompress
result = bridge_sdk.decompress(
    "results/output.cxe",
    "results/restored.txt"
)

print(f"Decompression time: {result['runtime_sec']:.4f}s")
```

### Command-Line Interface

```bash
# Compress with CARE enabled
python3 bridge_sdk/cli.py compress \
  --input tests/fixtures/test_input.txt \
  --output results/output.cxe \
  --care \
  --workers 4

# Decompress
python3 bridge_sdk/cli.py decompress \
  --input results/output.cxe \
  --output results/restored.txt

# Compare with zstd
python3 experiments/run_benchmark_zstd.py \
  --input tests/fixtures/ \
  --output results/benchmark.json \
  --backends src_engine_private,zstd,lz4
```

---

## Installation

### Prerequisites

- **Python 3.8+** (tested with 3.11)
- **SRC Engine binary** at `../src_engine_private/src-engine`
- **System packages** (for reference codecs):
  - `zstd` (Zstandard compression)
  - `lz4` (LZ4 compression)
  - `pytest` (for running tests)

### Install Dependencies

```bash
# Install Python dependencies (PyYAML only)
pip3 install pyyaml pytest

# Install system codecs (Ubuntu/Debian)
sudo apt install zstd liblz4-tool

# Verify installations
zstd --version
lz4 --version
pytest --version
```

### Verify Installation

```bash
# Test SDK import
python3 -c "import bridge_sdk; print(bridge_sdk.__version__)"

# Run unit tests
python3 -m pytest tests/ -v

# Run full validation suite
python3 validate_bridge.py --full
```

---

## API Reference

### bridge_sdk.compress()

Compress a file using the SRC Engine or reference codecs.

**Signature:**

```python
def compress(
    input_path: str,
    output_path: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

**Parameters:**

- `input_path` (str): Input file path (workspace-relative, must exist)
- `output_path` (str): Output file path (workspace-relative)
- `config` (dict, optional): Configuration with keys:
  - `backend` (str): `"src_engine_private"` (default), `"reference_zstd"`, `"reference_lz4"`
  - `workers` (int): Number of parallel workers (default: 1)
  - `care` (bool): Enable CARE encoding (default: False)
  - `beam_width` (int): Beam search width (optional)
  - `max_depth` (int): Max recursion depth (optional)

**Returns:**

Dictionary with keys:

- `status` (str): `"ok"` or `"error"`
- `ratio` (float): Compression ratio (input_size / output_size)
- `runtime_sec` (float): Elapsed time in seconds
- `caq` (float): CAQ score
- `backend` (str): Backend used
- `message` (str): Status message
- `input` (str): Input file path (workspace-relative)
- `output` (str): Output file path (workspace-relative)
- `input_size` (int): Input size in bytes
- `output_size` (int): Output size in bytes

**Raises:**

- `SecurityError`: Path traversal or network access attempt
- `ValidationError`: Invalid arguments or missing files
- `TimeoutError`: Operation exceeded time limit
- `EngineError`: Compression failed

**Example:**

```python
result = bridge_sdk.compress(
    "data/input.txt",
    "results/output.cxe",
    config={
        "backend": "src_engine_private",
        "workers": 4,
        "care": True
    }
)

if result["status"] == "ok":
    print(f"Compressed {result['input']} -> {result['output']}")
    print(f"Ratio: {result['ratio']:.2f}x")
    print(f"Time: {result['runtime_sec']:.4f}s")
    print(f"CAQ: {result['caq']:.6f}")
else:
    print(f"Error: {result['message']}")
```

### bridge_sdk.decompress()

Decompress a CXE archive.

**Signature:**

```python
def decompress(
    input_path: str,
    output_path: str
) -> Dict[str, Any]
```

**Parameters:**

- `input_path` (str): Input CXE file path (workspace-relative, must exist)
- `output_path` (str): Output file path (workspace-relative)

**Returns:**

Dictionary with keys:

- `status` (str): `"ok"` or `"error"`
- `runtime_sec` (float): Elapsed time in seconds
- `backend` (str): Backend used
- `message` (str): Status message
- `input` (str): Input file path
- `output` (str): Output file path
- `input_size` (int): Compressed size in bytes
- `output_size` (int): Decompressed size in bytes

**Raises:**

Same exceptions as `compress()`

**Example:**

```python
result = bridge_sdk.decompress(
    "results/archive.cxe",
    "data/restored.txt"
)

if result["status"] == "ok":
    print(f"Decompressed in {result['runtime_sec']:.4f}s")
```

### bridge_sdk.analyze()

Analyze a CXE archive (Phase H.2+, not yet implemented).

**Signature:**

```python
def analyze(
    input_path: str,
    output_path: str,
    mode: str = "summary"
) -> Dict[str, Any]
```

**Status:** Not implemented (raises `ValidationError`)

---

## CLI Reference

### Global Options

```bash
python3 bridge_sdk/cli.py --version  # Show version
python3 bridge_sdk/cli.py --help     # Show help
```

### compress command

```bash
python3 bridge_sdk/cli.py compress [OPTIONS]

Options:
  --input PATH           Input file path (required)
  --output PATH          Output file path (required)
  --backend BACKEND      Backend: src_engine_private, reference_zstd, reference_lz4
                         (default: src_engine_private)
  --workers N            Number of parallel workers (default: 1)
  --care                 Enable CARE encoding
  --beam-width N         Beam search width (advanced)
  --max-depth N          Max recursion depth (advanced)
```

**Output:** JSON to stdout

**Exit codes:**

- `0`: Success
- `400`: Validation error (bad arguments, path issues)
- `408`: Timeout
- `500`: Engine error
- `127`: Engine binary not found

**Example:**

```bash
python3 bridge_sdk/cli.py compress \
  --input tests/fixtures/test_input.txt \
  --output results/output.cxe \
  --backend src_engine_private \
  --workers 2 \
  --care
```

**Output:**

```json
{
  "status": "ok",
  "ratio": 104.5,
  "runtime_sec": 0.26,
  "caq": 0.000821,
  "backend": "src_engine_private",
  "message": "compression completed",
  "input": "tests/fixtures/test_input.txt",
  "output": "results/output.cxe",
  "input_size": 1046,
  "output_size": 10
}
```

### decompress command

```bash
python3 bridge_sdk/cli.py decompress [OPTIONS]

Options:
  --input PATH    Input CXE file path (required)
  --output PATH   Output file path (required)
```

**Example:**

```bash
python3 bridge_sdk/cli.py decompress \
  --input results/output.cxe \
  --output results/restored.txt
```

### run-task command (for testing)

```bash
python3 bridge_sdk/cli.py run-task [OPTIONS]

Options:
  --task TASK     Task name (compress, decompress, analyze)
  --input PATH    Input file path
  --output PATH   Output file path
  --backend BACKEND
  --workers N
  --care
```

Used by validation suite to test manifest unknown task rejection.

---

## Manifest Format

The `bridge_manifest.yaml` file defines available tasks, argument schemas, and resource limits.

### Full Example

```yaml
version: 1

tasks:
  - name: compress
    description: Compress a file using SRC Engine or reference codecs
    args:
      - name: input
        type: path
        required: true
        description: Input file path (workspace-relative)

      - name: output
        type: path
        required: true
        description: Output file path (workspace-relative)

      - name: backend
        type: enum
        required: false
        default: src_engine_private
        choices:
          - src_engine_private
          - reference_zstd
          - reference_lz4
        description: Backend to use for compression

      - name: workers
        type: int
        required: false
        default: 1
        min: 1
        max: 16
        description: Number of parallel workers

      - name: care
        type: bool
        required: false
        default: false
        description: Enable CARE encoding

    time_limit_sec: 300  # 5 minutes
    cpu_limit: null
    memory_limit_mb: null

  - name: decompress
    description: Decompress a CXE archive
    args:
      - name: input
        type: path
        required: true

      - name: output
        type: path
        required: true

    time_limit_sec: 300

security:
  allow_network: false
  allow_traversal: false
  workspace_only: true

determinism:
  default_seed: 42
  reproducible_mode: true
```

### Manifest Schema

**Top-level fields:**

- `version` (int): Manifest schema version (currently 1)
- `tasks` (list): List of task definitions
- `security` (dict): Security policies
- `determinism` (dict): Reproducibility settings

**Task definition:**

- `name` (str): Task name (compress, decompress, analyze)
- `description` (str): Human-readable description
- `args` (list): Argument definitions
- `time_limit_sec` (int): Maximum execution time in seconds (0 = no limit)
- `cpu_limit` (int|null): CPU limit (null = no limit)
- `memory_limit_mb` (int|null): Memory limit in MB (null = no limit)

**Argument definition:**

- `name` (str): Argument name
- `type` (str): Type (path, string, int, float, bool, enum)
- `required` (bool): Whether argument is required
- `default` (any): Default value if not provided
- `choices` (list): Valid values (for enum type)
- `min` (int|float): Minimum value (for int/float)
- `max` (int|float): Maximum value (for int/float)
- `description` (str): Human-readable description

### Validation Rules

1. All task arguments are validated against manifest before execution
2. Unknown tasks are rejected with code 404
3. Missing required arguments raise `ValidationError`
4. Invalid choice values raise `ValidationError`
5. Type conversions are attempted (string → int, string → bool)
6. Time limits are enforced by security layer

---

## Security Model

### Security Principles

1. **Workspace-only access**: All file paths must be within workspace (no `../` traversal)
2. **Offline-only execution**: No network access (proxy vars cleared)
3. **Resource limits**: Timeout enforcement via SIGALRM (Unix)
4. **Sanitized errors**: No internal paths or stack traces exposed
5. **No privilege escalation**: Runs with user permissions only

### Path Validation

```python
from bridge_sdk.security import validate_workspace_path

# Valid paths (workspace-relative)
validate_workspace_path("tests/fixtures/test_input.txt")  # ✓
validate_workspace_path("results/output.cxe")  # ✓

# Invalid paths (rejected)
validate_workspace_path("../foundation_charter.md")  # ✗ SecurityError
validate_workspace_path("/etc/passwd")  # ✗ SecurityError
```

**How it works:**

1. Convert path to absolute using `Path.resolve()`
2. Check if path is within workspace using `Path.relative_to()`
3. Raise `SecurityError` if outside workspace
4. Optionally check file existence

### Network Prevention

```python
from bridge_sdk.security import disallow_network

# Clears proxy environment variables
disallow_network()  # Clears http_proxy, https_proxy, etc.
```

**Note:** This is a soft check. True network isolation requires OS-level sandboxing (containers, network namespaces).

### Timeout Enforcement

```python
from bridge_sdk.security import enforce_timeout

# Execute with 60-second timeout
with enforce_timeout(60, "compression"):
    run_compression()  # Raises TimeoutError if exceeds 60s
```

**Implementation:** Uses `signal.SIGALRM` (Unix-only). For Windows, alternative timeout mechanisms would be needed.

### Error Sanitization

```python
from bridge_sdk.security import sanitize_error_message

error = "Error in /home/user/workspace/file.txt\nTraceback..."
sanitized = sanitize_error_message(error)
# Result: "Error in workspace/file.txt"
```

**Sanitization rules:**

1. Take first line only (no stack traces)
2. Replace absolute workspace paths with "workspace"
3. Truncate to 200 characters

### Security Checklist

- ✓ Path traversal prevention (`validate_workspace_path`)
- ✓ Network access prevention (`disallow_network`)
- ✓ Timeout enforcement (`enforce_timeout`)
- ✓ Error sanitization (no internal details exposed)
- ✓ No hardcoded credentials
- ✓ Subprocess uses list form (no shell injection)
- ✓ Manifest validation (reject unknown tasks)
- ✓ Type checking (prevent injection via args)

---

## Reproducibility

### Reproducibility Checklist

To ensure reproducible results, follow these steps:

#### 1. Verify Environment

```bash
# Check Python version
python3 --version  # Should be 3.8+

# Check system
uname -a  # Note OS and kernel version

# Check codecs
zstd --version
lz4 --version
```

#### 2. Run Validation Suite

```bash
python3 validate_bridge.py --full
```

**Expected output:** `bridge_validation.log` with all tests passing.

#### 3. Run Benchmarks

```bash
# Run benchmark twice
python3 experiments/run_benchmark_zstd.py \
  --input tests/fixtures/ \
  --output results/benchmark_run1.json

python3 experiments/run_benchmark_zstd.py \
  --input tests/fixtures/ \
  --output results/benchmark_run2.json

# Compare CAQ scores (should be within 1% tolerance)
python3 tools/compare_caq.py \
  results/benchmark_run1.json \
  results/benchmark_run2.json \
  --tolerance 0.01
```

**Expected:** Max delta ≤ 1.0% (0.01)

#### 4. Verify Determinism

```bash
# Use compare_caq.py to check reproducibility
python3 tools/compare_caq.py benchmark_1.json benchmark_2.json --verbose
```

**Output example:**

```
=== CAQ Score Comparison ===
Tolerance: 1.0%
Total comparisons: 15
Max delta: 0.0042 (0.42%)

✓ DETERMINISM: PASS
  Max delta (0.42%) within tolerance (1.0%)
```

### Determinism Settings

From `bridge_manifest.yaml`:

```yaml
determinism:
  default_seed: 42
  reproducible_mode: true
```

**Note:** SRC Engine should respect deterministic flags if provided. Variability may occur due to:

- System load (CPU timing)
- Floating-point precision
- Parallel worker scheduling

### Validation Commands (Exact)

```bash
# 1. Unit tests
python3 -m pytest -q

# 2. Import SDK
python3 -c "import bridge_sdk; print(dir(bridge_sdk))"

# 3. CLI roundtrip
python3 bridge_sdk/cli.py compress --input tests/fixtures/test_input.txt --output results/test.cxe
python3 bridge_sdk/cli.py decompress --input results/test.cxe --output results/restored.txt
diff tests/fixtures/test_input.txt results/restored.txt

# 4. Path traversal test
python3 bridge_sdk/cli.py compress --input ../foundation_charter.md --output results/fail.cxe
# Expected: exit code != 0, JSON error with code 400

# 5. Manifest unknown task
python3 bridge_sdk/cli.py run-task --task unknown_task --input tests/fixtures/test_input.txt --output results/x.cxe
# Expected: exit code != 0, JSON error with code 404

# 6. Benchmark run
python3 experiments/run_benchmark_zstd.py --input tests/fixtures/ --output results/benchmark_zstd.json

# 7. Determinism check
python3 tools/compare_caq.py results/benchmark_1.json results/benchmark_2.json

# 8. Full validation
python3 validate_bridge.py --full
```

---

## FAQ & Troubleshooting

### Q: Where is the SRC Engine binary?

**A:** The engine binary must be at `../src_engine_private/src-engine` relative to the workspace root. If it's elsewhere, update `ENGINE_BINARY` in `bridge_sdk/api.py`.

### Q: Why does compress() fail with code 127?

**A:** Code 127 means the engine binary was not found. Check:

1. Binary exists at expected path
2. Binary has execute permissions (`chmod +x src-engine`)
3. Path in `api.py` is correct

### Q: Can I use the Bridge SDK outside the workspace?

**A:** No. All file paths must be workspace-relative for security. This prevents directory traversal attacks.

### Q: How do I add a new backend?

**A:**

1. Add backend to `bridge_manifest.yaml` choices
2. Implement backend logic in `experiments/reference_codecs.py`
3. Update `run_benchmark_zstd.py` to support new backend
4. Run validation suite to ensure compliance

### Q: Why is the timeout test skipped in validation?

**A:** True timeout testing requires a slow operation (large file or `--simulate-slow` flag). The validation suite verifies the timeout framework is in place but doesn't execute a real timeout to avoid slow test runs.

### Q: Can I run this on Windows?

**A:** Partially. The timeout enforcement uses `signal.SIGALRM` which is Unix-only. On Windows, you'd need to implement alternative timeout mechanisms (threading, process pools). Other features (path validation, manifest, API) work cross-platform.

### Q: How do I debug validation failures?

**A:**

1. Check `bridge_validation.log` for detailed error messages
2. Run individual tests:
   ```bash
   python3 -m pytest tests/test_security.py -v
   ```
3. Enable verbose mode in scripts:
   ```bash
   python3 experiments/run_benchmark_zstd.py --input tests/ --output results/debug.json
   ```

### Q: What if my CAQ scores vary by more than 1%?

**A:** Variability can occur due to:

- System load (CPU timing differences)
- Non-deterministic engine behavior
- Parallel worker scheduling

Solutions:

- Run benchmarks on idle system
- Use single worker (`--workers 1`)
- Increase tolerance: `--tolerance 0.05` (5%)
- Check engine for deterministic flags

### Q: Can I use this SDK for production compression workflows?

**A:** Yes, but:

1. Run full validation suite first (`validate_bridge.py --full`)
2. Test with your specific file types and sizes
3. Monitor for resource usage (CPU, memory, disk)
4. Consider implementing additional logging/monitoring
5. Review security model for your threat model

### Q: How do I contribute a new codec?

**A:** See [Engineering Culture & Contribution Guidelines](engineeringculture.md). Steps:

1. Create feature branch: `feature/add-codec-name`
2. Implement codec in `experiments/reference_codecs.py`
3. Update `bridge_manifest.yaml`
4. Add tests in `tests/`
5. Run validation suite
6. Create pull request

---

## References

- [Main README](../README.md)
- [Engineering Culture](engineeringculture.md)
- [CAQ Metric Documentation](../metrics/caq_metric.py)
- [Manifest Schema](../bridge_manifest.yaml)
- [Validation Suite](../validate_bridge.py)

---

**Last updated:** 2025-10-13
**Maintainer:** SRC Research Lab
**Version:** 1.0.0
