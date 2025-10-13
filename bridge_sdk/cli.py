#!/usr/bin/env python3
"""
Bridge SDK command-line interface (Phase H.1)

Provides a user-friendly CLI that wraps the Python API.
All outputs are JSON to stdout; errors go to stderr with non-zero exit codes.
"""

import sys
import json
import argparse
from typing import Dict, Any

from bridge_sdk.api import compress, decompress, analyze
from bridge_sdk.exceptions import BridgeError
from bridge_sdk import __version__


def format_error_result(error: BridgeError) -> Dict[str, Any]:
    """Format exception as error result dictionary."""
    return {
        "status": "error",
        "code": error.code,
        "message": error.message
    }


def compress_command(args) -> int:
    """Handle compress command."""
    try:
        config = {
            "backend": args.backend,
            "workers": args.workers,
            "care": args.care
        }

        if hasattr(args, 'beam_width') and args.beam_width:
            config['beam_width'] = args.beam_width
        if hasattr(args, 'max_depth') and args.max_depth:
            config['max_depth'] = args.max_depth

        result = compress(args.input, args.output, config=config)
        print(json.dumps(result, indent=2))
        return 0

    except BridgeError as e:
        error_result = format_error_result(e)
        print(json.dumps(error_result, indent=2), file=sys.stderr)
        return e.code if e.code else 1

    except Exception as e:
        error_result = {
            "status": "error",
            "code": 1,
            "message": f"Unexpected error: {type(e).__name__}"
        }
        print(json.dumps(error_result, indent=2), file=sys.stderr)
        return 1


def decompress_command(args) -> int:
    """Handle decompress command."""
    try:
        result = decompress(args.input, args.output)
        print(json.dumps(result, indent=2))
        return 0

    except BridgeError as e:
        error_result = format_error_result(e)
        print(json.dumps(error_result, indent=2), file=sys.stderr)
        return e.code if e.code else 1

    except Exception as e:
        error_result = {
            "status": "error",
            "code": 1,
            "message": f"Unexpected error: {type(e).__name__}"
        }
        print(json.dumps(error_result, indent=2), file=sys.stderr)
        return 1


def analyze_command(args) -> int:
    """Handle analyze command."""
    try:
        mode = getattr(args, 'mode', 'summary')
        result = analyze(args.input, args.output, mode=mode)
        print(json.dumps(result, indent=2))
        return 0

    except BridgeError as e:
        error_result = format_error_result(e)
        print(json.dumps(error_result, indent=2), file=sys.stderr)
        return e.code if e.code else 1

    except Exception as e:
        error_result = {
            "status": "error",
            "code": 1,
            "message": f"Unexpected error: {type(e).__name__}"
        }
        print(json.dumps(error_result, indent=2), file=sys.stderr)
        return 1


def run_task_command(args) -> int:
    """Handle generic run-task command (used for manifest validation tests)."""
    task_name = args.task

    # Route to appropriate handler
    if task_name == "compress":
        return compress_command(args)
    elif task_name == "decompress":
        return decompress_command(args)
    elif task_name == "analyze":
        return analyze_command(args)
    else:
        # Unknown task
        from bridge_sdk.manifest import get_manifest
        from bridge_sdk.exceptions import ManifestError

        try:
            manifest = get_manifest()
            available = manifest.list_tasks()
            raise ManifestError(
                f"Unknown task '{task_name}'—see bridge_manifest.yaml. "
                f"Available: {', '.join(available)}"
            )
        except ManifestError as e:
            error_result = format_error_result(e)
            print(json.dumps(error_result, indent=2), file=sys.stderr)
            return 404


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="bridge_sdk",
        description="Bridge SDK CLI - Secure interface to SRC Engine (Phase H.1)"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"Bridge SDK v{__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Compress command
    compress_parser = subparsers.add_parser(
        "compress",
        help="Compress a file"
    )
    compress_parser.add_argument(
        "--input",
        required=True,
        help="Input file path (workspace-relative)"
    )
    compress_parser.add_argument(
        "--output",
        required=True,
        help="Output file path (workspace-relative)"
    )
    compress_parser.add_argument(
        "--backend",
        default="src_engine_private",
        choices=["src_engine_private", "reference_zstd", "reference_lz4"],
        help="Backend to use (default: src_engine_private)"
    )
    compress_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    compress_parser.add_argument(
        "--care",
        action="store_true",
        help="Enable CARE (Context-Aware Recursive Encoding)"
    )
    compress_parser.add_argument(
        "--beam-width",
        type=int,
        dest="beam_width",
        help="Beam search width (optional)"
    )
    compress_parser.add_argument(
        "--max-depth",
        type=int,
        dest="max_depth",
        help="Maximum recursion depth (optional)"
    )
    compress_parser.set_defaults(func=compress_command)

    # Decompress command
    decompress_parser = subparsers.add_parser(
        "decompress",
        help="Decompress a CXE archive"
    )
    decompress_parser.add_argument(
        "--input",
        required=True,
        help="Input CXE file path (workspace-relative)"
    )
    decompress_parser.add_argument(
        "--output",
        required=True,
        help="Output file path (workspace-relative)"
    )
    decompress_parser.set_defaults(func=decompress_command)

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a CXE archive (future feature)"
    )
    analyze_parser.add_argument(
        "--input",
        required=True,
        help="Input CXE file path (workspace-relative)"
    )
    analyze_parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path (workspace-relative)"
    )
    analyze_parser.add_argument(
        "--mode",
        default="summary",
        choices=["summary", "detailed", "tree"],
        help="Analysis mode (default: summary)"
    )
    analyze_parser.set_defaults(func=analyze_command)

    # Run-task command (for generic manifest validation)
    run_task_parser = subparsers.add_parser(
        "run-task",
        help="Run a task by name (for testing)"
    )
    run_task_parser.add_argument(
        "--task",
        required=True,
        help="Task name (compress, decompress, analyze)"
    )
    run_task_parser.add_argument(
        "--input",
        required=True,
        help="Input file path"
    )
    run_task_parser.add_argument(
        "--output",
        required=True,
        help="Output file path"
    )
    run_task_parser.add_argument(
        "--backend",
        default="src_engine_private",
        help="Backend to use"
    )
    run_task_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers"
    )
    run_task_parser.add_argument(
        "--care",
        action="store_true",
        help="Enable CARE"
    )
    run_task_parser.set_defaults(func=run_task_command)

    # Parse arguments
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    # Execute command
    exit_code = args.func(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
