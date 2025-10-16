#!/usr/bin/env python3
"""
Release Bundle Validation Script for SRC Research Lab

End-to-end validation of public benchmark bundles.
Verifies checksums, runs canonical scripts, and validates outputs.

Exit codes:
    0: All validations passed
    1: Validation failure
"""

import sys
import json
import hashlib
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()


def verify_checksums(bundle_dir: Path) -> tuple[bool, str]:
    """Verify bundle checksums."""
    checksum_file = bundle_dir / "checksum.sha256"

    if not checksum_file.exists():
        return False, "checksum.sha256 not found"

    print(f"  Verifying checksums for {bundle_dir.name}...")

    with open(checksum_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if not line.strip():
            continue

        hash_expected, file_path_str = line.strip().split(None, 1)
        file_path = bundle_dir / file_path_str

        if not file_path.exists():
            return False, f"Missing file: {file_path_str}"

        with open(file_path, 'rb') as f:
            hash_actual = hashlib.sha256(f.read()).hexdigest()

        if hash_actual != hash_expected:
            return False, f"Checksum mismatch: {file_path_str}"

    return True, "OK"


def run_canonical_script(bundle_dir: Path, workdir: Path) -> tuple[bool, str, Dict]:
    """Run the canonical benchmark script."""
    script_path = bundle_dir / "run_canonical.sh"

    if not script_path.exists():
        return False, "run_canonical.sh not found", {}

    print(f"  Running canonical script for {bundle_dir.name}...")

    # Copy bundle to workdir for isolated execution
    import shutil
    work_bundle = workdir / bundle_dir.name
    if work_bundle.exists():
        shutil.rmtree(work_bundle)
    shutil.copytree(bundle_dir, work_bundle)

    try:
        # Execute the script
        result = subprocess.run(
            ["bash", str(work_bundle / "run_canonical.sh")],
            cwd=work_bundle,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            return False, f"Script failed: {result.stderr}", {}

        # Parse output for CAQ values
        output_lines = result.stdout.split('\n')
        metrics = {}

        for line in output_lines:
            if "Mean CAQ:" in line:
                try:
                    metrics["mean_caq"] = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
            elif "Mean Ratio:" in line:
                try:
                    metrics["mean_ratio"] = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
            elif "Mean CPU:" in line:
                try:
                    cpu_str = line.split(":")[-1].strip().replace("s", "")
                    metrics["mean_cpu"] = float(cpu_str)
                except ValueError:
                    pass

        return True, "OK", metrics

    except subprocess.TimeoutExpired:
        return False, "Script timeout (>300s)", {}
    except Exception as e:
        return False, f"Execution error: {e}", {}


def compare_with_expected(actual: Dict, expected_file: Path, tolerance: float = 0.015) -> tuple[bool, str]:
    """Compare actual results with expected submission."""
    if not expected_file.exists():
        return False, "example_submission.json not found"

    with open(expected_file, 'r') as f:
        expected = json.load(f)

    expected_metrics = expected.get("computed_metrics", {})

    # Compare CAQ
    if "mean_caq" in actual and "computed_caq" in expected_metrics:
        actual_caq = actual["mean_caq"]
        expected_caq = expected_metrics["computed_caq"]

        if expected_caq > 0:
            variance = abs(actual_caq - expected_caq) / expected_caq

            if variance > tolerance:
                return False, f"CAQ variance {variance*100:.2f}% exceeds tolerance {tolerance*100:.2f}%"

    return True, "OK"


def validate_bundle(bundle_dir: Path, workdir: Path) -> Dict[str, Any]:
    """Validate a single bundle."""
    print(f"\nValidating bundle: {bundle_dir.name}")

    validation = {
        "bundle": bundle_dir.name,
        "timestamp": datetime.now().isoformat(),
        "status": "PASS",
        "checks": []
    }

    # Check 1: Verify checksums
    success, msg = verify_checksums(bundle_dir)
    validation["checks"].append({
        "check": "checksums",
        "status": "PASS" if success else "FAIL",
        "message": msg
    })
    if not success:
        validation["status"] = "FAIL"
        return validation

    # Check 2: Run canonical script
    success, msg, actual_metrics = run_canonical_script(bundle_dir, workdir)
    validation["checks"].append({
        "check": "canonical_run",
        "status": "PASS" if success else "FAIL",
        "message": msg,
        "metrics": actual_metrics
    })
    if not success:
        validation["status"] = "FAIL"
        return validation

    # Check 3: Compare with expected
    expected_file = bundle_dir / "example_submission.json"
    success, msg = compare_with_expected(actual_metrics, expected_file, tolerance=0.015)
    validation["checks"].append({
        "check": "expected_match",
        "status": "PASS" if success else "FAIL",
        "message": msg
    })
    if not success:
        validation["status"] = "FAIL"

    return validation


def generate_validation_report(validations: List[Dict], output_path: Path):
    """Generate final validation report."""
    all_passed = all(v["status"] == "PASS" for v in validations)

    report = {
        "generated_at": datetime.now().isoformat(),
        "status": "PASS" if all_passed else "FAIL",
        "bundles": validations,
        "summary": {
            "total": len(validations),
            "passed": sum(1 for v in validations if v["status"] == "PASS"),
            "failed": sum(1 for v in validations if v["status"] == "FAIL")
        },
        "signature": ""
    }

    # Compute signature
    json_str = json.dumps(report, sort_keys=True, separators=(',', ':'))
    report["signature"] = hashlib.sha256(json_str.encode()).hexdigest()[:16]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Validation report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate public benchmark release bundles"
    )
    parser.add_argument(
        "--bundles",
        nargs="+",
        required=True,
        help="Bundle directories to validate"
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/h4_validation"),
        help="Working directory for isolated execution"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT / "release" / "VALIDATION_RELEASE_H4.json",
        help="Output validation report path"
    )

    args = parser.parse_args()

    # Create workdir
    args.workdir.mkdir(parents=True, exist_ok=True)

    # Validate each bundle
    validations = []
    for bundle_path_str in args.bundles:
        bundle_path = Path(bundle_path_str)
        if bundle_path.is_dir():
            validation = validate_bundle(bundle_path, args.workdir)
            validations.append(validation)

    if not validations:
        print("ERROR: No bundles found to validate", file=sys.stderr)
        return 1

    # Generate report
    generate_validation_report(validations, args.output)

    # Print summary
    all_passed = all(v["status"] == "PASS" for v in validations)
    print("\n" + "="*60)
    if all_passed:
        print("VALIDATION: PASS — all bundles reproduced expected results")
        print("="*60)
        return 0
    else:
        print("VALIDATION: FAIL — some bundles failed validation")
        print("="*60)
        for v in validations:
            if v["status"] == "FAIL":
                print(f"  ✗ {v['bundle']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
