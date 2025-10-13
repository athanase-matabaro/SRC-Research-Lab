"""
Bridge SDK utilities.

Helper functions for file operations, metrics, and result formatting.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional


def get_file_size(path: Path) -> int:
    """Get file size in bytes."""
    return path.stat().st_size


def compute_compression_ratio(input_size: int, output_size: int) -> float:
    """
    Compute compression ratio as input_size / output_size.

    Args:
        input_size: Original file size in bytes
        output_size: Compressed file size in bytes

    Returns:
        Compression ratio (higher is better, 1.0 = no compression)
    """
    if output_size == 0:
        return 0.0
    return input_size / output_size


def format_result(
    status: str,
    task: str,
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    runtime_sec: Optional[float] = None,
    ratio: Optional[float] = None,
    caq: Optional[float] = None,
    backend: Optional[str] = None,
    message: str = "",
    code: int = 0,
    **extra
) -> Dict[str, Any]:
    """
    Format operation result as standardized JSON dictionary.

    Args:
        status: "ok" or "error"
        task: Task name (compress, decompress, analyze)
        input_path: Input file path (workspace-relative)
        output_path: Output file path (workspace-relative)
        runtime_sec: Elapsed time in seconds
        ratio: Compression ratio
        caq: CAQ score
        backend: Backend used (src_engine_private, zstd, etc.)
        message: Status/error message
        code: Error code (0 for success)
        **extra: Additional fields to include

    Returns:
        Formatted result dictionary
    """
    from bridge_sdk.security import WORKSPACE_ROOT

    result = {
        "status": status,
    }

    if code != 0:
        result["code"] = code

    if task:
        result["task"] = task

    if backend:
        result["backend"] = backend

    if input_path:
        try:
            result["input"] = str(input_path.relative_to(WORKSPACE_ROOT))
        except ValueError:
            result["input"] = input_path.name

    if output_path:
        try:
            result["output"] = str(output_path.relative_to(WORKSPACE_ROOT))
        except ValueError:
            result["output"] = output_path.name

    if runtime_sec is not None:
        result["runtime_sec"] = round(runtime_sec, 4)

    if ratio is not None:
        result["ratio"] = round(ratio, 4)

    if caq is not None:
        result["caq"] = round(caq, 6)

    if message:
        result["message"] = message

    # Add any extra fields
    result.update(extra)

    return result


def load_json_file(path: Path) -> Dict[str, Any]:
    """Load and parse JSON file."""
    with path.open('r') as f:
        return json.load(f)


def save_json_file(path: Path, data: Dict[str, Any], indent: int = 2) -> None:
    """Save data as formatted JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(data, f, indent=indent)


class Timer:
    """Simple context manager for timing operations."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        return False

    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.elapsed is not None:
            return self.elapsed
        elif self.start_time is not None:
            return time.time() - self.start_time
        return 0.0
