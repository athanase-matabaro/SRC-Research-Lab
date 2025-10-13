#!/usr/bin/env python3
"""
src_bridge.py - Secure interface to the closed-core SRC Engine (Phase H.0)

This bridge provides a safe, offline-only API for the open research layer
to invoke the proprietary SRC Engine without exposing internal implementation.

Security features:
- Offline-only execution (no network calls)
- Path validation (workspace-relative only)
- Sanitized error messages
- Resource limits enforced by subprocess
- JSON-based communication protocol
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional


# Path to the SRC Engine binary (assumes it's in PATH or specify absolute path)
ENGINE_BINARY = "src-engine"

# Workspace root (parent directory of this script)
WORKSPACE_ROOT = Path(__file__).parent.resolve()


def validate_path(path_str: str, must_exist: bool = False) -> Path:
    """
    Validate that a path is workspace-relative and safe.

    Args:
        path_str: Path string to validate
        must_exist: Whether the path must already exist

    Returns:
        Resolved absolute Path object

    Raises:
        ValueError: If path is invalid or unsafe
    """
    try:
        # Convert to Path and resolve
        path = Path(path_str).resolve()

        # Check if path is within workspace (prevent directory traversal)
        try:
            path.relative_to(WORKSPACE_ROOT)
        except ValueError:
            raise ValueError(f"Path must be within workspace: {WORKSPACE_ROOT}")

        # Check existence if required
        if must_exist and not path.exists():
            raise ValueError(f"Path does not exist: {path}")

        return path

    except Exception as e:
        raise ValueError(f"Invalid path '{path_str}': {e}")


def invoke_engine(task: str, input_path: Path, output_path: Path, **kwargs) -> Dict[str, Any]:
    """
    Invoke the SRC Engine via subprocess with security controls.

    Args:
        task: Engine task ("compress" or "decompress")
        input_path: Input file path (validated)
        output_path: Output file path (validated)
        **kwargs: Additional engine parameters

    Returns:
        Parsed JSON response from engine

    Raises:
        RuntimeError: If engine execution fails
    """
    # Build command
    cmd = [ENGINE_BINARY, task, str(input_path), str(output_path)]

    # Add optional parameters
    if "workers" in kwargs:
        cmd.extend(["--workers", str(kwargs["workers"])])
    if "backend" in kwargs:
        cmd.extend(["--backend", kwargs["backend"]])
    if "care" in kwargs and kwargs["care"]:
        cmd.append("--care")
    if "beam_width" in kwargs:
        cmd.extend(["--beam-width", str(kwargs["beam_width"])])
    if "max_depth" in kwargs:
        cmd.extend(["--max-depth", str(kwargs["max_depth"])])

    try:
        # Execute engine as subprocess (offline-only, no network access)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            check=False   # Don't raise on non-zero exit
        )

        # Check for errors
        if result.returncode != 0:
            # Sanitize error message (no stack traces or internal details)
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            # Remove any paths that might leak internal structure
            error_msg = error_msg.split('\n')[0]  # First line only

            return {
                "status": "error",
                "code": result.returncode,
                "message": error_msg[:200]  # Limit error length
            }

        # Parse engine output
        # For now, engine doesn't output JSON directly, so we construct it
        # from the exit code and file existence
        if output_path.exists():
            input_size = input_path.stat().st_size
            output_size = output_path.stat().st_size
            ratio = input_size / output_size if output_size > 0 else 0

            return {
                "status": "ok",
                "task": task,
                "input": str(input_path.relative_to(WORKSPACE_ROOT)),
                "output": str(output_path.relative_to(WORKSPACE_ROOT)),
                "input_size": input_size,
                "output_size": output_size,
                "ratio": round(ratio, 4)
            }
        else:
            return {
                "status": "error",
                "code": 1,
                "message": "Engine completed but output file not created"
            }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "code": 124,
            "message": "Engine execution timeout (300s limit)"
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "code": 127,
            "message": f"Engine binary not found: {ENGINE_BINARY}"
        }
    except Exception as e:
        # Sanitize any unexpected errors
        return {
            "status": "error",
            "code": 1,
            "message": f"Bridge error: {type(e).__name__}"
        }


def compress_cmd(args) -> None:
    """Handle compress command."""
    try:
        # Validate paths
        input_path = validate_path(args.input, must_exist=True)
        output_path = validate_path(args.output, must_exist=False)

        # Build kwargs from args
        kwargs = {}
        if hasattr(args, 'workers') and args.workers:
            kwargs['workers'] = args.workers
        if hasattr(args, 'backend') and args.backend:
            kwargs['backend'] = args.backend
        if hasattr(args, 'care') and args.care:
            kwargs['care'] = True

        # Invoke engine
        result = invoke_engine("compress", input_path, output_path, **kwargs)

        # Output JSON result
        print(json.dumps(result, indent=2))

        # Exit with appropriate code
        sys.exit(0 if result["status"] == "ok" else 1)

    except ValueError as e:
        error_result = {
            "status": "error",
            "code": 400,
            "message": str(e)
        }
        print(json.dumps(error_result, indent=2), file=sys.stderr)
        sys.exit(1)


def decompress_cmd(args) -> None:
    """Handle decompress command."""
    try:
        # Validate paths
        input_path = validate_path(args.input, must_exist=True)
        output_path = validate_path(args.output, must_exist=False)

        # Invoke engine
        result = invoke_engine("decompress", input_path, output_path)

        # Output JSON result
        print(json.dumps(result, indent=2))

        # Exit with appropriate code
        sys.exit(0 if result["status"] == "ok" else 1)

    except ValueError as e:
        error_result = {
            "status": "error",
            "code": 400,
            "message": str(e)
        }
        print(json.dumps(error_result, indent=2), file=sys.stderr)
        sys.exit(1)


def analyze_cmd(args) -> None:
    """Handle analyze command (future feature)."""
    error_result = {
        "status": "error",
        "code": 501,
        "message": "Analyze command not yet implemented"
    }
    print(json.dumps(error_result, indent=2), file=sys.stderr)
    sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="src_bridge.py",
        description="SRC Engine Bridge - Secure interface to proprietary compression engine"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Compress command
    compress_parser = subparsers.add_parser("compress", help="Compress a file")
    compress_parser.add_argument("--input", required=True, help="Input file path (workspace-relative)")
    compress_parser.add_argument("--output", required=True, help="Output file path (workspace-relative)")
    compress_parser.add_argument("--workers", type=int, help="Number of parallel workers")
    compress_parser.add_argument("--backend", choices=["single", "local"], help="Execution backend")
    compress_parser.add_argument("--care", action="store_true", help="Enable CARE (Context-Aware Recursive Encoding)")
    compress_parser.set_defaults(func=compress_cmd)

    # Decompress command
    decompress_parser = subparsers.add_parser("decompress", help="Decompress a file")
    decompress_parser.add_argument("--input", required=True, help="Input file path (workspace-relative)")
    decompress_parser.add_argument("--output", required=True, help="Output file path (workspace-relative)")
    decompress_parser.set_defaults(func=decompress_cmd)

    # Analyze command (future)
    analyze_parser = subparsers.add_parser("analyze", help="Analyze compression results")
    analyze_parser.add_argument("--input", required=True, help="Archive file to analyze")
    analyze_parser.set_defaults(func=analyze_cmd)

    # Parse and execute
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
