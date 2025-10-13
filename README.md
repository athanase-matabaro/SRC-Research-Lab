# SRC Research Lab

Open research layer for the Semantic Recursive Compression (SRC) Engine. This repository provides benchmarking tools, metrics evaluation, and a secure bridge interface to the proprietary SRC Engine.

## Quick Start

### Compression

Compress a file using the SRC Engine:

```bash
./scripts/src_bridge.py compress --input tests/fixtures/test_input.txt --output test_output.cxe
```

With CARE (Context-Aware Recursive Encoding) enabled:

```bash
./scripts/src_bridge.py compress --input tests/fixtures/test_input.txt --output test_output.cxe --care
```

Parallel compression with multiple workers:

```bash
./scripts/src_bridge.py compress --input tests/fixtures/test_input.txt --output test_output.cxe --workers 4 --backend local
```

### Decompression

Decompress a CXE archive:

```bash
./scripts/src_bridge.py decompress --input test_output.cxe --output test_restored.txt
```

### Running Benchmarks

Run baseline compression benchmarks:

```bash
./scripts/run_baseline.py
```

Results are saved to `results/baseline_benchmark.json`.

## Project Structure

```
src-research-lab/
├── scripts/                  # Executable scripts
│   ├── src_bridge.py        # Secure interface to the SRC Engine
│   └── run_baseline.py      # Baseline benchmarking script
├── metrics/                  # Metric implementations
│   └── caq_metric.py        # CAQ (Compression-Accuracy Quotient)
├── tests/                    # Test files and fixtures
│   └── fixtures/            # Test data files
├── results/                  # Benchmark results
├── experiments/              # Research experiments
├── docs/                     # Documentation
│   ├── index.md
│   └── engineeringculture.md  # Contribution guidelines
├── README.md
└── LICENSE.md
```

## Usage Examples

### Basic Compression Workflow

```bash
# 1. Compress a text file
./scripts/src_bridge.py compress --input tests/fixtures/test_input.txt --output compressed.cxe --care

# 2. Verify the output
ls -lh compressed.cxe

# 3. Decompress to verify integrity
./scripts/src_bridge.py decompress --input compressed.cxe --output restored.txt

# 4. Compare files
diff tests/fixtures/test_input.txt restored.txt
```

### Benchmark Comparison

```bash
# Run baseline benchmarks (includes gzip, bzip2, xz, zstd, LZ4)
./scripts/run_baseline.py

# View results
cat results/baseline_benchmark.json | jq
```

## Requirements

- Python 3.8+
- SRC Engine binary (`src-engine` in PATH)
- Standard compression tools: gzip, bzip2, xz, zstd, lz4

## License

See [LICENSE.md](LICENSE.md) for details.

## Contributing

We welcome contributions! Please read our [Engineering Culture & Contribution Guidelines](docs/engineeringculture.md) before submitting pull requests.

Key points:
- Create feature branches: `feature/your-feature-name`
- Follow conventional commit messages
- Keep the root directory clean
- Update documentation with your changes

## Documentation

For more information, see the [docs/](docs/) directory:
- [Engineering Culture & Guidelines](docs/engineeringculture.md)
- [Documentation Index](docs/index.md)

## ⚙️ Key Principles

- **Closed core, open science:** the SRC Engine remains private, but every published metric, dataset, and experiment is transparent and verifiable.  
- **CPU-only, offline:** reproducible anywhere without GPUs or cloud access.  
- **Ethical & sustainable:** cost-aware benchmarks (CAQ) encourage efficiency and equity in AI research.  

---

## 📈 Current Milestone

| Phase  | Status |
|:-------------|:-------|
| Foundation & Governance | ✅ Completed |
| Bridge SDK & Secure Interface | 🚧 In Progress |

---

## 🌍 Vision

Build a **global research community** exploring the future of adaptive compression — one that’s transparent, sustainable, and scientifically rigorous, while preserving the proprietary intelligence of the SRC Engine core.

---

**Lead Researcher:** *Athanase Matabaro*
