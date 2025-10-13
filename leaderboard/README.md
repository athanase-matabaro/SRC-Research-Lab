# SRC Research Lab Leaderboard

## Purpose & Principles

The SRC Research Lab Leaderboard provides a transparent, reproducible framework for comparing compression algorithms using the CAQ (Compression-Accuracy Quotient) metric.

**Core Principles:**
- **Reproducibility**: All submissions must include at least 3 independent runs with variance ≤ 1.5%
- **Offline Operation**: No network dependencies; all validation runs locally
- **Security**: Strict workspace isolation, no data exfiltration
- **Transparency**: Open submission schema and validation code
- **Ethics**: Public datasets only; SRC Engine internals remain private

## CAQ Metric

CAQ balances compression ratio with computational efficiency:

```
CAQ = compression_ratio / (cpu_seconds + 1)
```

Higher CAQ scores indicate better overall performance. The +1 in the denominator prevents division by zero and ensures even instantaneous compression has a finite score.

## Submission Schema

All submissions must conform to the JSON schema defined in `leaderboard_schema.json`:

```json
{
  "submitter": "your_identifier",
  "date": "YYYY-MM-DD",
  "dataset": "text_medium" | "image_small" | "mixed_stream",
  "codec": "codec_name:version",
  "version": "version_string",
  "compression_ratio": 5.63,
  "cpu_seconds": 0.26,
  "runs": [
    { "ratio": 5.60, "cpu_seconds": 0.25 },
    { "ratio": 5.64, "cpu_seconds": 0.26 },
    { "ratio": 5.65, "cpu_seconds": 0.27 }
  ],
  "variance": 0.89,
  "caq": 4.44,
  "notes": "Description and reproducible command if available"
}
```

**Required Fields:**
- `submitter`: Your identifier (username, institution, etc.)
- `date`: Submission date in YYYY-MM-DD format
- `dataset`: One of the standard datasets (text_medium, image_small, mixed_stream)
- `codec`: Codec identifier with version
- `version`: Version string
- `compression_ratio`: Mean compression ratio across runs
- `cpu_seconds`: Mean CPU time in seconds
- `runs`: Array of at least 3 individual measurements
- `notes`: Description, methodology, and reproducible command

**Computed Fields:**
- `variance`: Computed as (max_ratio - min_ratio) / mean_ratio * 100
- `caq`: Computed using the CAQ formula

## Datasets

Three standard datasets are provided for reproducible benchmarking:

1. **text_medium** (~105 KB): Technical text about compression research
2. **image_small** (~24 KB): Three synthetic binary patterns simulating image data
3. **mixed_stream** (~967 KB): Mixed content with text, binary patterns, repeated sequences

All datasets are in `datasets/` directory with open licenses.

## Contributor Guide

### Step 1: Run Your Experiment

Use the Bridge SDK or your own compression tool to benchmark against one of the standard datasets:

```bash
# Example using Bridge SDK
python3 bridge_sdk/cli.py compress datasets/text_medium/corpus.txt output.cxe

# Run at least 3 times and record:
# - Compression ratio (original_size / compressed_size)
# - CPU time in seconds
```

### Step 2: Create Submission JSON

Create a JSON file following the schema above. See `examples/sample_submission.json` for a complete example.

**Important:**
- Include at least 3 runs
- Compute mean ratio and mean CPU time
- Calculate variance: (max_ratio - min_ratio) / mean_ratio * 100
- Calculate CAQ: ratio / (cpu_seconds + 1)
- Variance must be ≤ 1.5%

### Step 3: Validate Locally

Before submitting, validate your submission locally:

```bash
python3 scripts/validate_submission.py \
  --input your_submission.json \
  --repeat 0
```

Expected output:
```
PASS: Validation successful
Report: leaderboard/reports/your_name_TIMESTAMP.json
CAQ: 4.47
Variance: 0.38%
```

### Step 4: Submit Pull Request

1. Fork the repository
2. Create a new branch: `git checkout -b submission/your_name`
3. Add your submission: `cp your_submission.json leaderboard/submissions/`
4. Commit and push
5. Open a pull request

### Step 5: Maintainer Validation

A maintainer will run the validation suite:

```bash
python3 scripts/validate_submission.py \
  --input leaderboard/submissions/your_submission.json \
  --repeat 3 \
  --timeout 300
```

On PASS, your submission will be included in the leaderboard:

```bash
python3 scripts/leaderboard_update.py \
  --reports-dir leaderboard/reports \
  --out-json leaderboard/leaderboard.json \
  --out-md leaderboard/leaderboard.md
```

## Validation Requirements

Submissions must pass the following checks:

1. **Schema Validation**: JSON conforms to leaderboard_schema.json
2. **Metric Recomputation**: CAQ recomputed from runs matches submitted value (±1.5% tolerance)
3. **Reproducibility**: Variance across runs ≤ 1.5%
4. **Security**: All paths workspace-relative, no network access
5. **Dataset**: Must use one of the standard datasets

## Exit Codes

**validate_submission.py:**
- 0: PASS - submission is valid and reproducible
- 2: Schema validation error
- 3: Computation/reproducibility failure
- 4: Security violation

**leaderboard_update.py:**
- 0: Success
- 5: No PASS reports found
- 6: Verification failure

## Example Commands

```bash
# Validate sample submission (no reproducibility runs)
python3 scripts/validate_submission.py \
  --input examples/sample_submission.json \
  --repeat 0

# Full validation with 3 reproducibility runs
python3 scripts/validate_submission.py \
  --input examples/sample_submission.json \
  --repeat 3 \
  --timeout 300

# Generate leaderboard
python3 scripts/leaderboard_update.py \
  --reports-dir leaderboard/reports \
  --out-json leaderboard/leaderboard.json \
  --out-md leaderboard/leaderboard.md

# Verify top 5 submissions
python3 scripts/leaderboard_update.py \
  --reports-dir leaderboard/reports \
  --out-json leaderboard/leaderboard.json \
  --out-md leaderboard/leaderboard.md \
  --verify-top 5
```

## Ethics Statement

**Public Data Only**: All benchmark datasets must use publicly available data with appropriate licenses. Do not submit results from proprietary or sensitive datasets.

**SRC Engine Privacy**: While the SRC compression engine is used for benchmarking, its internals remain private. The leaderboard framework enables community participation without exposing proprietary algorithms.

**Fair Comparison**: All submissions use identical datasets and validation procedures to ensure fair, reproducible comparisons.

**Reproducibility**: The offline validation framework ensures that results can be independently verified without network dependencies or external services.

## Troubleshooting

**"Schema validation error"**: Check your JSON against the schema. Common issues:
- Missing required fields
- Wrong data types (e.g., string instead of number)
- Invalid dataset name

**"Variance too high"**: Your runs show >1.5% variance. Possible causes:
- Insufficient warmup runs
- Background processes affecting timing
- Non-deterministic compression algorithm

**"CAQ mismatch"**: Recomputed CAQ differs from submitted value by >1.5%. Double-check your calculations:
- mean_ratio = sum(runs.ratio) / len(runs)
- mean_cpu = sum(runs.cpu_seconds) / len(runs)
- caq = mean_ratio / (mean_cpu + 1)

**"Path outside workspace"**: All file paths must be relative to the repository root. Use datasets in the `datasets/` directory.

## Contact

For questions or issues with the leaderboard system:
- Open an issue on GitHub
- Review existing submissions in `leaderboard/submissions/`
- Check validation reports in `leaderboard/reports/`

---

**Last Updated**: 2025-10-13
**Leaderboard Version**: 1.0.0
