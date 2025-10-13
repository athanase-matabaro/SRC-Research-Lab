"""
Bridge SDK public API (Phase H.1)

Main interface for compression, decompression, and analysis tasks.
"""

import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from bridge_sdk.security import (
    validate_workspace_path,
    enforce_timeout,
    disallow_network,
    sanitize_error_message,
    WORKSPACE_ROOT
)
from bridge_sdk.exceptions import (
    BridgeError,
    EngineError,
    ValidationError,
    TimeoutError as BridgeTimeoutError
)
from bridge_sdk.utils import (
    get_file_size,
    compute_compression_ratio,
    format_result,
    Timer
)
from bridge_sdk.manifest import get_manifest


# Path to SRC Engine binary (relative to workspace parent)
ENGINE_BINARY = "../src_engine_private/src-engine"


def _invoke_engine(
    task: str,
    input_path: Path,
    output_path: Path,
    backend: str = "src_engine_private",
    timeout_sec: int = 300,
    **kwargs
) -> Dict[str, Any]:
    """
    Internal: Invoke SRC Engine binary with security controls.

    Args:
        task: Engine task (compress, decompress)
        input_path: Input file path (validated)
        output_path: Output file path (validated)
        backend: Backend to use (src_engine_private only)
        timeout_sec: Time limit in seconds
        **kwargs: Additional engine parameters

    Returns:
        Raw engine result dictionary

    Raises:
        EngineError: If engine execution fails
        BridgeTimeoutError: If timeout exceeded
    """
    # Only src_engine_private backend uses the binary
    if backend != "src_engine_private":
        raise ValidationError(f"Backend '{backend}' not supported by engine")

    # Resolve engine binary path
    engine_path = (WORKSPACE_ROOT / ENGINE_BINARY).resolve()

    if not engine_path.exists():
        raise EngineError(
            f"Engine binary not found at expected location. "
            f"Ensure src-engine is available.",
            engine_code=127
        )

    # Build command
    cmd = [str(engine_path), task, str(input_path), str(output_path)]

    # Add optional parameters
    if "workers" in kwargs and kwargs["workers"]:
        cmd.extend(["--workers", str(kwargs["workers"])])
    if "care" in kwargs and kwargs["care"]:
        cmd.append("--care")
    if "beam_width" in kwargs and kwargs["beam_width"]:
        cmd.extend(["--beam-width", str(kwargs["beam_width"])])
    if "max_depth" in kwargs and kwargs["max_depth"]:
        cmd.extend(["--max-depth", str(kwargs["max_depth"])])

    try:
        # Enforce timeout
        with enforce_timeout(timeout_sec, task):
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

        # Check for errors
        if result.returncode != 0:
            error_msg = sanitize_error_message(
                result.stderr.strip() if result.stderr else "Unknown engine error"
            )
            raise EngineError(error_msg, engine_code=result.returncode)

        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except BridgeTimeoutError:
        raise
    except FileNotFoundError:
        raise EngineError(
            f"Engine binary not executable or not found",
            engine_code=127
        )
    except Exception as e:
        raise EngineError(
            f"Engine execution failed: {type(e).__name__}",
            engine_code=1
        )


def compress(
    input_path: str,
    output_path: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compress a file using the SRC Engine.

    Args:
        input_path: Input file path (workspace-relative)
        output_path: Output file path (workspace-relative)
        config: Optional configuration dictionary with keys:
            - backend: "src_engine_private" (default)
            - workers: Number of parallel workers (int, default: 1)
            - care: Enable CARE encoding (bool, default: False)
            - beam_width: Beam search width (int, optional)
            - max_depth: Maximum recursion depth (int, optional)

    Returns:
        Result dictionary with keys:
            - status: "ok" or "error"
            - ratio: Compression ratio (input_size / output_size)
            - runtime_sec: Elapsed time in seconds
            - caq: CAQ score (computed from ratio and runtime)
            - backend: Backend used
            - message: Status message
            - input: Input file path (workspace-relative)
            - output: Output file path (workspace-relative)
            - input_size: Input file size in bytes
            - output_size: Output file size in bytes

    Raises:
        BridgeError: On validation, security, or execution errors

    Example:
        >>> result = compress("tests/input.txt", "results/output.cxe")
        >>> print(f"Ratio: {result['ratio']:.2f}x, CAQ: {result['caq']:.6f}")
    """
    # Network safety check
    disallow_network()

    # Validate manifest
    manifest = get_manifest()
    config = config or {}

    # Merge config into args
    args = {
        "input": input_path,
        "output": output_path,
        "backend": config.get("backend", "src_engine_private"),
        "workers": config.get("workers", 1),
        "care": config.get("care", False),
        "beam_width": config.get("beam_width"),
        "max_depth": config.get("max_depth")
    }

    # Validate args against manifest
    validated_args = manifest.validate_args("compress", args)

    # Validate paths
    input_path_obj = validate_workspace_path(validated_args["input"], must_exist=True)
    output_path_obj = validate_workspace_path(validated_args["output"], must_exist=False)

    backend = validated_args.get("backend", "src_engine_private")
    timeout_sec = manifest.get_time_limit("compress")

    try:
        # Get input size
        input_size = get_file_size(input_path_obj)

        # Time the operation
        with Timer() as timer:
            # Invoke engine
            _invoke_engine(
                "compress",
                input_path_obj,
                output_path_obj,
                backend=backend,
                timeout_sec=timeout_sec,
                workers=validated_args.get("workers"),
                care=validated_args.get("care"),
                beam_width=validated_args.get("beam_width"),
                max_depth=validated_args.get("max_depth")
            )

        runtime_sec = timer.get_elapsed()

        # Check output exists
        if not output_path_obj.exists():
            raise EngineError("Compression completed but output file not created")

        # Get output size and compute metrics
        output_size = get_file_size(output_path_obj)
        ratio = compute_compression_ratio(input_size, output_size)

        # Compute CAQ score
        from metrics.caq_metric import caq_score
        caq = caq_score(ratio, runtime_sec)

        return format_result(
            status="ok",
            task="compress",
            input_path=input_path_obj,
            output_path=output_path_obj,
            runtime_sec=runtime_sec,
            ratio=ratio,
            caq=caq,
            backend=backend,
            message="compression completed",
            input_size=input_size,
            output_size=output_size
        )

    except BridgeError:
        raise
    except Exception as e:
        raise EngineError(f"Unexpected error: {type(e).__name__}")


def decompress(
    input_path: str,
    output_path: str
) -> Dict[str, Any]:
    """
    Decompress a CXE archive.

    Args:
        input_path: Input CXE file path (workspace-relative)
        output_path: Output file path (workspace-relative)

    Returns:
        Result dictionary with keys:
            - status: "ok" or "error"
            - runtime_sec: Elapsed time in seconds
            - backend: Backend used
            - message: Status message
            - input: Input file path
            - output: Output file path
            - input_size: Input file size in bytes
            - output_size: Output file size in bytes

    Raises:
        BridgeError: On validation, security, or execution errors

    Example:
        >>> result = decompress("results/output.cxe", "results/restored.txt")
        >>> print(f"Decompression time: {result['runtime_sec']:.4f}s")
    """
    # Network safety check
    disallow_network()

    # Validate manifest
    manifest = get_manifest()

    args = {
        "input": input_path,
        "output": output_path
    }

    validated_args = manifest.validate_args("decompress", args)

    # Validate paths
    input_path_obj = validate_workspace_path(validated_args["input"], must_exist=True)
    output_path_obj = validate_workspace_path(validated_args["output"], must_exist=False)

    backend = "src_engine_private"
    timeout_sec = manifest.get_time_limit("decompress")

    try:
        # Get input size
        input_size = get_file_size(input_path_obj)

        # Time the operation
        with Timer() as timer:
            # Invoke engine
            _invoke_engine(
                "decompress",
                input_path_obj,
                output_path_obj,
                backend=backend,
                timeout_sec=timeout_sec
            )

        runtime_sec = timer.get_elapsed()

        # Check output exists
        if not output_path_obj.exists():
            raise EngineError("Decompression completed but output file not created")

        # Get output size
        output_size = get_file_size(output_path_obj)

        return format_result(
            status="ok",
            task="decompress",
            input_path=input_path_obj,
            output_path=output_path_obj,
            runtime_sec=runtime_sec,
            backend=backend,
            message="decompression completed",
            input_size=input_size,
            output_size=output_size
        )

    except BridgeError:
        raise
    except Exception as e:
        raise EngineError(f"Unexpected error: {type(e).__name__}")


def analyze(
    input_path: str,
    output_path: str,
    mode: str = "summary"
) -> Dict[str, Any]:
    """
    Analyze a CXE archive (future feature).

    Args:
        input_path: Input CXE file path (workspace-relative)
        output_path: Output JSON file path (workspace-relative)
        mode: Analysis mode ("summary", "detailed", "tree")

    Returns:
        Result dictionary

    Raises:
        BridgeError: Not yet implemented

    Example:
        >>> result = analyze("results/output.cxe", "results/analysis.json")
    """
    raise ValidationError("Analyze command not yet implemented (Phase H.2+)")
