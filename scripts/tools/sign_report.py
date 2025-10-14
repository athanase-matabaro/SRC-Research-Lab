#!/usr/bin/env python3
"""
Sign a validation report with SHA256 hash.

Usage:
    python3 scripts/tools/sign_report.py <report.json>

This tool:
1. Loads the JSON report
2. Canonicalizes it (sorted keys)
3. Computes SHA256 hex digest
4. Appends "signature" field to the JSON
5. Atomically writes back to the file
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile


def canonicalize_json(data):
    """Canonicalize JSON by sorting keys and using consistent formatting."""
    return json.dumps(data, sort_keys=True, indent=2)


def compute_signature(json_str):
    """Compute SHA256 hex digest of the canonical JSON string."""
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


def sign_report(report_path):
    """Sign a report file by adding a signature field."""
    # Load the report
    try:
        with open(report_path, 'r') as f:
            report = json.load(f)
    except FileNotFoundError:
        print(f"Error: Report file not found: {report_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in report: {e}", file=sys.stderr)
        sys.exit(1)

    # Remove any existing signature for clean signing
    if 'signature' in report:
        del report['signature']

    # Canonicalize and compute signature
    canonical = canonicalize_json(report)
    signature = compute_signature(canonical)

    # Add signature to report
    report['signature'] = signature

    # Atomic write back
    dir_path = os.path.dirname(os.path.abspath(report_path))
    try:
        # Write to temporary file first
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=dir_path,
            delete=False,
            prefix='.tmp_',
            suffix='.json'
        ) as tmp_file:
            json.dump(report, tmp_file, indent=2, sort_keys=True)
            tmp_file.write('\n')  # Add trailing newline
            tmp_path = tmp_file.name

        # Atomic rename
        os.replace(tmp_path, report_path)

        print(f"Signature: {signature}")
        return 0

    except Exception as e:
        print(f"Error writing signed report: {e}", file=sys.stderr)
        # Clean up temp file if it exists
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Sign a validation report with SHA256 hash',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python3 scripts/tools/sign_report.py leaderboard/reports/jane_doe_20251014.json
        """
    )
    parser.add_argument(
        'report',
        help='Path to the JSON report file to sign'
    )

    args = parser.parse_args()

    return sign_report(args.report)


if __name__ == '__main__':
    sys.exit(main())
