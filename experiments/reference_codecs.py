"""
Reference codec wrappers for baseline comparison (Phase H.1)

Provides Python interfaces to standard compression tools:
- zstd (Zstandard)
- lz4 (LZ4)

Uses subprocess calls to system binaries for offline, CPU-only compression.
"""

import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional


class CodecError(Exception):
    """Reference codec execution error."""
    pass


def _run_command(cmd: list, timeout_sec: int = 300) -> subprocess.CompletedProcess:
    """
    Run command with timeout and capture output.

    Args:
        cmd: Command list
        timeout_sec: Timeout in seconds

    Returns:
        CompletedProcess result

    Raises:
        CodecError: On execution failure
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout_sec,
            check=False
        )

        if result.returncode != 0:
            raise CodecError(
                f"Command failed with code {result.returncode}: "
                f"{result.stderr.decode('utf-8', errors='ignore')[:200]}"
            )

        return result

    except subprocess.TimeoutExpired:
        raise CodecError(f"Command timed out after {timeout_sec}s")
    except FileNotFoundError:
        raise CodecError(f"Command not found: {cmd[0]}")
    except Exception as e:
        raise CodecError(f"Command execution failed: {type(e).__name__}")


def compress_zstd(
    input_path: Path,
    output_path: Path,
    level: int = 3,
    timeout_sec: int = 300
) -> Dict[str, Any]:
    """
    Compress file using zstd.

    Args:
        input_path: Input file path
        output_path: Output file path
        level: Compression level (1-22, default: 3)
        timeout_sec: Timeout in seconds

    Returns:
        Result dictionary with keys:
            - status: "ok" or "error"
            - ratio: Compression ratio
            - runtime_sec: Elapsed time
            - backend: "reference_zstd"
            - input_size: Input size in bytes
            - output_size: Output size in bytes

    Raises:
        CodecError: On compression failure
    """
    if not input_path.exists():
        raise CodecError(f"Input file not found: {input_path}")

    # Build command
    cmd = ["zstd", f"-{level}", "-c", str(input_path)]

    try:
        # Get input size
        input_size = input_path.stat().st_size

        # Run compression
        start = time.time()
        result = _run_command(cmd, timeout_sec=timeout_sec)
        elapsed = time.time() - start

        # Write output
        output_path.write_bytes(result.stdout)
        output_size = output_path.stat().st_size

        # Compute ratio
        ratio = input_size / output_size if output_size > 0 else 0.0

        return {
            "status": "ok",
            "backend": "reference_zstd",
            "ratio": round(ratio, 4),
            "runtime_sec": round(elapsed, 4),
            "input_size": input_size,
            "output_size": output_size
        }

    except CodecError:
        raise
    except Exception as e:
        raise CodecError(f"zstd compression failed: {type(e).__name__}")


def decompress_zstd(
    input_path: Path,
    output_path: Path,
    timeout_sec: int = 300
) -> Dict[str, Any]:
    """
    Decompress zstd file.

    Args:
        input_path: Input .zst file path
        output_path: Output file path
        timeout_sec: Timeout in seconds

    Returns:
        Result dictionary

    Raises:
        CodecError: On decompression failure
    """
    if not input_path.exists():
        raise CodecError(f"Input file not found: {input_path}")

    # Build command
    cmd = ["zstd", "-d", "-c", str(input_path)]

    try:
        # Get input size
        input_size = input_path.stat().st_size

        # Run decompression
        start = time.time()
        result = _run_command(cmd, timeout_sec=timeout_sec)
        elapsed = time.time() - start

        # Write output
        output_path.write_bytes(result.stdout)
        output_size = output_path.stat().st_size

        return {
            "status": "ok",
            "backend": "reference_zstd",
            "runtime_sec": round(elapsed, 4),
            "input_size": input_size,
            "output_size": output_size
        }

    except CodecError:
        raise
    except Exception as e:
        raise CodecError(f"zstd decompression failed: {type(e).__name__}")


def compress_lz4(
    input_path: Path,
    output_path: Path,
    level: int = 1,
    timeout_sec: int = 300
) -> Dict[str, Any]:
    """
    Compress file using lz4.

    Args:
        input_path: Input file path
        output_path: Output file path
        level: Compression level (1-12, default: 1)
        timeout_sec: Timeout in seconds

    Returns:
        Result dictionary with keys:
            - status: "ok" or "error"
            - ratio: Compression ratio
            - runtime_sec: Elapsed time
            - backend: "reference_lz4"
            - input_size: Input size in bytes
            - output_size: Output size in bytes

    Raises:
        CodecError: On compression failure
    """
    if not input_path.exists():
        raise CodecError(f"Input file not found: {input_path}")

    # Build command
    cmd = ["lz4", f"-{level}", "-c", str(input_path)]

    try:
        # Get input size
        input_size = input_path.stat().st_size

        # Run compression
        start = time.time()
        result = _run_command(cmd, timeout_sec=timeout_sec)
        elapsed = time.time() - start

        # Write output
        output_path.write_bytes(result.stdout)
        output_size = output_path.stat().st_size

        # Compute ratio
        ratio = input_size / output_size if output_size > 0 else 0.0

        return {
            "status": "ok",
            "backend": "reference_lz4",
            "ratio": round(ratio, 4),
            "runtime_sec": round(elapsed, 4),
            "input_size": input_size,
            "output_size": output_size
        }

    except CodecError:
        raise
    except Exception as e:
        raise CodecError(f"lz4 compression failed: {type(e).__name__}")


def decompress_lz4(
    input_path: Path,
    output_path: Path,
    timeout_sec: int = 300
) -> Dict[str, Any]:
    """
    Decompress lz4 file.

    Args:
        input_path: Input .lz4 file path
        output_path: Output file path
        timeout_sec: Timeout in seconds

    Returns:
        Result dictionary

    Raises:
        CodecError: On decompression failure
    """
    if not input_path.exists():
        raise CodecError(f"Input file not found: {input_path}")

    # Build command
    cmd = ["lz4", "-d", "-c", str(input_path)]

    try:
        # Get input size
        input_size = input_path.stat().st_size

        # Run decompression
        start = time.time()
        result = _run_command(cmd, timeout_sec=timeout_sec)
        elapsed = time.time() - start

        # Write output
        output_path.write_bytes(result.stdout)
        output_size = output_path.stat().st_size

        return {
            "status": "ok",
            "backend": "reference_lz4",
            "runtime_sec": round(elapsed, 4),
            "input_size": input_size,
            "output_size": output_size
        }

    except CodecError:
        raise
    except Exception as e:
        raise CodecError(f"lz4 decompression failed: {type(e).__name__}")


# Codec registry
CODECS = {
    "zstd": {
        "compress": compress_zstd,
        "decompress": decompress_zstd,
        "extension": ".zst"
    },
    "lz4": {
        "compress": compress_lz4,
        "decompress": decompress_lz4,
        "extension": ".lz4"
    }
}


def get_codec(name: str) -> Dict[str, Any]:
    """
    Get codec by name.

    Args:
        name: Codec name ("zstd" or "lz4")

    Returns:
        Codec dictionary with compress/decompress functions

    Raises:
        CodecError: If codec not found
    """
    if name not in CODECS:
        raise CodecError(f"Unknown codec: {name}. Available: {', '.join(CODECS.keys())}")

    return CODECS[name]
