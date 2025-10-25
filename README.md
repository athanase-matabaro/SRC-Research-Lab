# 🌐 SRC Research Lab

[![Phase H.1](https://img.shields.io/badge/phase-H.1-brightgreen)](./) [![Offline Verified](https://img.shields.io/badge/offline-verified-blue)](./) [![CPU-Only](https://img.shields.io/badge/platform-CPU--Only-orange)](./) [![Security Tested](https://img.shields.io/badge/security-tested-green)](./) [![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE.md)

**Open research layer for the Semantic Recursive Compression (SRC) Engine**  
_Reproducible, CPU-first, cost-aware research built on a secure offline bridge to a protected SRC Engine core._

---

## 🔎 Overview

SRC Research Lab is the public research and benchmarking layer for the private **SRC Compression Engine** — a closed-core, CPU-first platform for intelligent, adaptive data compression.

This repository provides:
- A **secure, manifest-driven Bridge SDK** to call the proprietary SRC Engine locally (Phase H.1).
- Reproducible benchmarking scripts and the **CAQ (Cost-Adjusted Quality)** metric.
- Reference comparisons vs standard codecs (zstd, LZ4, gzip, xz).
- Documentation, validation logs, and a governance charter for open research built on a closed core.

**Key design principles**
- **Closed core, open science**: engine internals remain private; results, metrics, and experiments are transparent and reproducible.  
- **CPU-first & offline**: runs reproducibly on everyday hardware without GPU or cloud.  
- **Security-first**: path validation, timeout limits, network blocking, and error sanitization.

---

## 📌 Quick Links

- Docs: `docs/`
- Bridge SDK: `bridge_sdk/`
- Benchmarks: `experiments/` & `scripts/run_baseline.py`
- Validation Reports: `docs/release/VALIDATION_SUCCESS.md`
- CAQ metric: `metrics/caq_metric.py`

---

## 🚀 Quick Start

> These commands assume you are in the repository root (`src-research-lab/`) and you have a working `src-engine` binary available under `../src_engine_private/` or in PATH as `src-engine`.

### Compress (SRC Engine)

```bash
# Simple compression
./scripts/src_bridge.py compress --input tests/fixtures/test_input.txt --output test_output.cxe

# CARE (Context-Aware Recursive Encoding)
./scripts/src_bridge.py compress --input tests/fixtures/test_input.txt --output test_output.cxe --care

# Parallel compression (4 workers, local backend)
./scripts/src_bridge.py compress --input tests/fixtures/test_input.txt --output test_output.cxe --workers 4 --backend local
```

### Decompress

```bash
./scripts/src_bridge.py decompress --input test_output.cxe --output test_restored.txt
```

### Run Baseline Benchmarks (reference codecs included)

```bash
./scripts/run_baseline.py
# results written to results/baseline_benchmark.json
```

---

## 🧪 H.0 & H.1 Verification — Verified Results

> These are the official, audited milestones and metrics from the Phase H.0 → H.1 validation campaign.

### Phase H.0 — Foundation & Governance (completed)

* SRC Engine (v0.3.0) installed and validated (local virtualenv).
* Secure bridge stub implemented and security-reviewed.
* Baseline benchmark (H.0) — **average compression ratio on text datasets: 104.3×**. Results saved to `results/baseline_benchmark.json`.

### Phase H.1 — Bridge SDK & Secure Interface (completed)

* Full Bridge SDK implemented: `bridge_sdk/` (API + CLI + security).
* Validation suite: **9/9 validation tests passed**:
  * Unit tests: 38/38 passed
  * SDK import & API checks
  * CLI roundtrip (compress → decompress)
  * Path traversal prevention
  * Unknown task manifest handling
  * Timeout enforcement
  * Network prevention
  * Benchmark execution (zstd / lz4 comparisons)
  * Determinism test (0.28% variance)
* Reported (H.1) metrics (example dataset / configuration):
  * **Compression Ratio:** **5.63×**
  * **CAQ Score:** **4.44** (reference zstd CAQ ≈ 0.99)
  * **Determinism variance:** **0.28%** (target: < 1.5%)

> **Note:** H.0 and H.1 benchmarks reflect different experiments and datasets. H.0 baseline (104.3×) was measured on a text-heavy benchmark suite used for governance validation. H.1 results are the verified SDK benchmark for the target evaluation dataset used in the Phase H.1 validation run. Full raw outputs and logs are available under `results/` and `docs/release/VALIDATION_SUCCESS.md`.

---

## 🧮 CAQ — Cost-Adjusted Quality (canonical metric)

We use **CAQ** to evaluate compression effectiveness while penalizing compute cost:

```
CAQ = compression_ratio / (cpu_seconds + 1)
```

* `compression_ratio`: (original_size / compressed_size)
* `cpu_seconds`: measured CPU time taken for the task
* Higher CAQ → better cost-adjusted performance

The `metrics/caq_metric.py` module contains the canonical implementation used by all benchmark scripts.

---

## 📂 Project Structure

```
src-research-lab/
├── bridge_sdk/            # Bridge SDK (Phase H.1): api.py, cli.py, security.py, manifest handling
├── scripts/               # CLI wrappers and convenience scripts
│   ├── src_bridge.py      # legacy/compat wrapper (kept for examples)
│   └── run_baseline.py    # baseline benchmark script
├── experiments/           # Research experiments & reference codec harnesses
├── metrics/               # CAQ metric implementation
├── results/               # Benchmark outputs & validation logs
├── docs/                  # Documentation & governance (charter, release notes)
├── tests/                 # Unit & integration tests and fixtures
├── README.md
└── LICENSE.md
```

---

## 🛠 Development & Tests

### Requirements

* Python 3.8+
* `PyYAML` (for manifest parsing) — `pip install -r requirements.txt`
* Optional: `zstd` and `lz4` (for reference benchmarks)

### Run unit tests

```bash
python -m pytest -q
```

### Run full validation suite (local)

```bash
python scripts/validation/validate_bridge.py --full
# Output: scripts/validation/bridge_validation.log (PASS/FAIL summary)
```

### Reproduce CAQ benchmark (example)

```bash
python experiments/run_benchmark_zstd.py --input tests/fixtures/ --output results/benchmark_zstd.json
cat results/benchmark_zstd.json | jq
```

---

## 🔒 Security & Privacy

* **Workspace-only paths**: the SDK strictly rejects paths outside the workspace to prevent data leaks.
* **Timeouts enforced**: default task limits (e.g., 300s) are enforced to prevent runaway processes.
* **Network disabled**: the bridge enforces no-socket / no-telemetry behavior during operations.
* **Sanitized outputs**: errors are returned as structured JSON, only no stack traces or private paths are leaked.

---

## 🤝 Contributing

We welcome contributions from researchers, students, and engineers. Please follow these guidelines:

1. Read our governance & contribution guidelines: `docs/engineeringculture.md`.
2. Use feature branches: `feature/your-feature-name`.
3. Follow Conventional Commits for commit messages.
4. Include tests & validation logs for any new feature.
5. Do **not** commit any proprietary binaries or data from `src_engine_private/`.

When ready, open a Pull Request and include:

* Purpose & summary
* Validation steps & outputs
* Any required data or configuration in `tests/fixtures/` (open-licensed or synthetic only)

---

## 📚 Documentation & References

* Bridge SDK docs: `docs/bridge_sdk_docs.md`
* Validation report: `docs/release/VALIDATION_SUCCESS.md`
* CAQ specification: `metrics/caq.md`
* Governance & charter: `docs/foundation_charter.md`

---

## 🛣 Roadmap & Next Steps

Planned short-term priorities:

1. Replace any mock engine in CI with the production `src-engine` binary (locally) and re-run full validation.
2. Extend benchmark suite with additional codecs (brotli, snappy) and larger mixed-content datasets.
3. Implement progress callbacks for long-running tasks and streaming compression support.
4. Launch public CAQ leaderboard (Phase H.2) and invite community benchmark submissions.

---

## 🧑‍💼 Maintainers & Contact

**Athanase Matabaro** — Lead Researcher  
Email: [matabaro.n.athanase@gmail.com](mailto:matabaro.n.athanase@gmail.com)

For urgent security issues: open an issue titled `SECURITY` and email the maintainer directly.

---

## 🔖 License

See `LICENSE.md` for license and distribution details.

---

Thank you for supporting sustainable, reproducible compression research.
