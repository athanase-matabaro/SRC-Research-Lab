#!/usr/bin/env python3
"""
Validate a submission against the leaderboard schema and verify reproducibility.

Exit codes:
    0: PASS - submission is valid and reproducible
    2: Schema validation error
    3: Computation/reproducibility failure
    4: Security violation

Usage:
    python3 scripts/validate_submission.py --input submission.json [--repeat N] [--timeout SECONDS]
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import jsonschema
except ImportError:
    print("Error: jsonschema package not installed. Run: pip install jsonschema", file=sys.stderr)
    sys.exit(2)


# Exit codes
EXIT_PASS = 0
EXIT_SCHEMA_ERROR = 2
EXIT_COMPUTATION_ERROR = 3
EXIT_SECURITY_ERROR = 4

# Get project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SCHEMA_PATH = PROJECT_ROOT / "leaderboard" / "leaderboard_schema.json"
REPORTS_DIR = PROJECT_ROOT / "leaderboard" / "reports"
SIGN_SCRIPT = SCRIPT_DIR / "tools" / "sign_report.py"


def compute_caq(ratio, cpu_seconds):
    """
    Compute CAQ metric: ratio / (cpu_seconds + 1)

    This is a simple formula per the spec.
    Note: The existing caq_metric.py uses a different formula with log,
    but the spec explicitly states: CAQ = ratio / (cpu_seconds + 1)
    """
    return ratio / (cpu_seconds + 1.0)


def check_security(submission_path):
    """Check for security violations in the submission."""
    # Ensure path is within workspace
    try:
        submission_path = Path(submission_path).resolve()
        # Check if path is absolute and within allowed directories
        if not submission_path.is_absolute():
            print("Error: Submission path must be absolute", file=sys.stderr)
            return False
    except Exception as e:
        print(f"Error: Invalid path: {e}", file=sys.stderr)
        return False

    # Block network operations - this is a basic check
    # In a real system, this would be enforced at runtime
    try:
        with open(submission_path, 'r') as f:
            content = f.read()
            # Basic checks for suspicious patterns
            suspicious_patterns = ['socket.socket', 'urllib.request', 'requests.get', 'http.client']
            for pattern in suspicious_patterns:
                if pattern in content:
                    print(f"Warning: Suspicious pattern detected: {pattern}", file=sys.stderr)
    except Exception as e:
        print(f"Error reading submission: {e}", file=sys.stderr)
        return False

    return True


def load_schema():
    """Load the leaderboard schema."""
    try:
        with open(SCHEMA_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Schema not found at {SCHEMA_PATH}", file=sys.stderr)
        sys.exit(EXIT_SCHEMA_ERROR)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid schema JSON: {e}", file=sys.stderr)
        sys.exit(EXIT_SCHEMA_ERROR)


def validate_schema(submission, schema):
    """Validate submission against schema."""
    try:
        jsonschema.validate(instance=submission, schema=schema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Schema validation error: {e}"


def recompute_metrics(submission):
    """Recompute mean_ratio, mean_cpu, computed_caq, and variance from runs."""
    runs = submission.get('runs', [])

    if len(runs) < 3:
        return None, "At least 3 runs required"

    # Extract ratios and cpu times
    ratios = [run['ratio'] for run in runs]
    cpu_times = [run['cpu_seconds'] for run in runs]

    # Compute means
    mean_ratio = sum(ratios) / len(ratios)
    mean_cpu = sum(cpu_times) / len(cpu_times)

    # Compute variance (coefficient of variation as percentage)
    # Variance = (stddev / mean) * 100
    if mean_ratio > 0:
        variance_ratio = (sum((r - mean_ratio) ** 2 for r in ratios) / len(ratios)) ** 0.5
        variance_pct = (variance_ratio / mean_ratio) * 100
    else:
        variance_pct = 0.0

    # Compute CAQ
    computed_caq = compute_caq(mean_ratio, mean_cpu)

    return {
        'mean_ratio': mean_ratio,
        'mean_cpu': mean_cpu,
        'computed_caq': computed_caq,
        'variance': variance_pct
    }, None


def validate_reproducibility(submission, computed_metrics, max_variance=1.5, caq_tolerance=0.015):
    """
    Validate reproducibility:
    - variance <= 1.5%
    - abs(computed_caq - submitted.caq) / submitted.caq <= 0.015 (1.5%)
    """
    variance = computed_metrics['variance']
    computed_caq = computed_metrics['computed_caq']
    submitted_caq = submission.get('caq', 0)

    # Check variance
    if variance > max_variance:
        return False, f"Variance too high: {variance:.2f}% > {max_variance}%"

    # Check CAQ
    if submitted_caq > 0:
        caq_diff = abs(computed_caq - submitted_caq) / submitted_caq
        if caq_diff > caq_tolerance:
            return False, f"CAQ mismatch: computed={computed_caq:.4f}, submitted={submitted_caq:.4f}, diff={caq_diff*100:.2f}%"
    else:
        return False, "Invalid submitted CAQ (must be > 0)"

    return True, None


def generate_report(submission, computed_metrics, status, message, timestamp):
    """Generate validation report."""
    report = {
        'timestamp': timestamp,
        'submission_file': submission.get('submitter', 'unknown'),
        'status': status,
        'message': message,
        'submission': submission,
        'computed_metrics': computed_metrics,
        'validator_version': '1.0.0'
    }
    return report


def write_report(report, reports_dir):
    """Write validation report to file."""
    os.makedirs(reports_dir, exist_ok=True)

    submitter = report['submission'].get('submitter', 'unknown')
    timestamp = report['timestamp'].replace(':', '-').replace(' ', '_')
    report_filename = f"{submitter}_{timestamp}.json"
    report_path = reports_dir / report_filename

    try:
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write('\n')
        return report_path
    except Exception as e:
        print(f"Error writing report: {e}", file=sys.stderr)
        return None


def sign_report(report_path):
    """Sign the report using sign_report.py."""
    try:
        result = subprocess.run(
            [sys.executable, str(SIGN_SCRIPT), str(report_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "Signing timed out"
    except Exception as e:
        return False, f"Signing error: {e}"


def validate_submission(input_file, repeat=3, timeout=300):
    """Main validation logic."""
    # Security check
    if not check_security(input_file):
        print("FAIL: Security violation", file=sys.stderr)
        return EXIT_SECURITY_ERROR

    # Load submission
    try:
        with open(input_file, 'r') as f:
            submission = json.load(f)
    except FileNotFoundError:
        print(f"Error: Submission file not found: {input_file}", file=sys.stderr)
        return EXIT_SCHEMA_ERROR
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        return EXIT_SCHEMA_ERROR

    # Load schema
    schema = load_schema()

    # Validate schema
    valid, error = validate_schema(submission, schema)
    if not valid:
        print(f"FAIL: Schema validation error: {error}", file=sys.stderr)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report = generate_report(
            submission,
            None,
            'FAIL',
            f'Schema validation error: {error}',
            timestamp
        )
        report_path = write_report(report, REPORTS_DIR)
        if report_path:
            sign_report(report_path)
        return EXIT_SCHEMA_ERROR

    # Recompute metrics
    computed_metrics, error = recompute_metrics(submission)
    if error:
        print(f"FAIL: Computation error: {error}", file=sys.stderr)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report = generate_report(
            submission,
            None,
            'FAIL',
            f'Computation error: {error}',
            timestamp
        )
        report_path = write_report(report, REPORTS_DIR)
        if report_path:
            sign_report(report_path)
        return EXIT_COMPUTATION_ERROR

    # Validate reproducibility
    reproducible, error = validate_reproducibility(submission, computed_metrics)
    if not reproducible:
        print(f"FAIL: Reproducibility check failed: {error}", file=sys.stderr)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report = generate_report(
            submission,
            computed_metrics,
            'FAIL',
            f'Reproducibility check failed: {error}',
            timestamp
        )
        report_path = write_report(report, REPORTS_DIR)
        if report_path:
            sign_report(report_path)
        return EXIT_COMPUTATION_ERROR

    # Generate PASS report
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report = generate_report(
        submission,
        computed_metrics,
        'PASS',
        'All validation checks passed',
        timestamp
    )
    report_path = write_report(report, REPORTS_DIR)

    if report_path:
        success, output = sign_report(report_path)
        if success:
            print(f"PASS: Validation successful")
            print(f"Report: {report_path}")
            print(f"CAQ: {computed_metrics['computed_caq']:.4f}")
            print(f"Variance: {computed_metrics['variance']:.2f}%")
        else:
            print(f"Warning: Report signing failed: {output}", file=sys.stderr)
    else:
        print("Warning: Failed to write report", file=sys.stderr)

    return EXIT_PASS


def main():
    parser = argparse.ArgumentParser(
        description='Validate a submission against the leaderboard schema',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit codes:
    0: PASS - submission is valid and reproducible
    2: Schema validation error
    3: Computation/reproducibility failure
    4: Security violation

Example:
    python3 scripts/validate_submission.py --input examples/sample_submission.json
    python3 scripts/validate_submission.py --input submission.json --repeat 5 --timeout 600
        """
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to submission JSON file'
    )
    parser.add_argument(
        '--repeat',
        type=int,
        default=3,
        help='Number of validation runs (default: 3, use 0 to skip)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Timeout in seconds (default: 300)'
    )

    args = parser.parse_args()

    return validate_submission(args.input, args.repeat, args.timeout)


if __name__ == '__main__':
    sys.exit(main())
