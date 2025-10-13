"""
Bridge SDK security controls (Phase H.1)

Implements:
- Workspace-relative path validation (prevent directory traversal)
- Network access prevention
- Resource limits (timeout enforcement)
- Sanitized error messages
"""

import os
import signal
from pathlib import Path
from typing import Optional
from bridge_sdk.exceptions import SecurityError, ValidationError, TimeoutError as BridgeTimeoutError


# Workspace root (parent of bridge_sdk/)
WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()


def validate_workspace_path(path_str: str, must_exist: bool = False) -> Path:
    """
    Validate that a path is workspace-relative and safe.

    Args:
        path_str: Path string to validate
        must_exist: Whether the path must already exist

    Returns:
        Resolved absolute Path object within workspace

    Raises:
        SecurityError: If path attempts directory traversal
        ValidationError: If path is invalid or doesn't exist when required
    """
    try:
        # Convert to Path and resolve to absolute
        if not path_str:
            raise ValidationError("Path cannot be empty")

        path = Path(path_str).resolve()

        # Check if path is within workspace (prevent traversal attacks)
        try:
            path.relative_to(WORKSPACE_ROOT)
        except ValueError:
            raise SecurityError(
                f"Invalid path: must be within workspace. "
                f"Attempted access outside {WORKSPACE_ROOT.name}/"
            )

        # Check existence if required
        if must_exist and not path.exists():
            raise ValidationError(f"Path does not exist: {path.relative_to(WORKSPACE_ROOT)}")

        return path

    except (SecurityError, ValidationError):
        raise
    except Exception as e:
        raise ValidationError(f"Invalid path '{path_str}': {type(e).__name__}")


def disallow_network() -> None:
    """
    Prevent network access by checking and clearing proxy environment variables.

    Raises:
        SecurityError: If network configuration detected
    """
    # List of common proxy/network environment variables
    network_vars = [
        "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
        "ftp_proxy", "FTP_PROXY", "all_proxy", "ALL_PROXY",
        "no_proxy", "NO_PROXY"
    ]

    detected = [var for var in network_vars if os.environ.get(var)]

    if detected:
        # Clear them to prevent network access
        for var in detected:
            del os.environ[var]

        # Note: This is a soft check; true network isolation requires
        # OS-level sandboxing (containers, network namespaces, etc.)

    # Verify no proxy vars remain
    remaining = [var for var in network_vars if os.environ.get(var)]
    if remaining:
        raise SecurityError(f"Network configuration detected: {remaining}")


class TimeoutHandler:
    """
    Context manager for enforcing time limits on operations.

    Uses SIGALRM (Unix-only) for timeout enforcement.
    """

    def __init__(self, seconds: int, task_name: str = "task"):
        self.seconds = seconds
        self.task_name = task_name
        self.original_handler = None

    def _timeout_handler(self, signum, frame):
        raise BridgeTimeoutError(
            f"Task timed out (limit: {self.seconds}s)",
            limit_sec=self.seconds
        )

    def __enter__(self):
        if self.seconds <= 0:
            return self

        # Set up signal handler for timeout
        self.original_handler = signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seconds <= 0:
            return False

        # Cancel the alarm
        signal.alarm(0)
        # Restore original handler
        if self.original_handler is not None:
            signal.signal(signal.SIGALRM, self.original_handler)

        return False


def enforce_timeout(seconds: int, task_name: str = "task") -> TimeoutHandler:
    """
    Create a timeout handler context manager.

    Usage:
        with enforce_timeout(300, "compression"):
            run_compression()

    Args:
        seconds: Time limit in seconds (0 or negative = no limit)
        task_name: Task name for error messages

    Returns:
        TimeoutHandler context manager
    """
    return TimeoutHandler(seconds, task_name)


def sanitize_error_message(error_msg: str, max_length: int = 200) -> str:
    """
    Sanitize error message by removing internal paths and limiting length.

    Args:
        error_msg: Raw error message
        max_length: Maximum message length

    Returns:
        Sanitized error message
    """
    # Take only first line (avoid stack traces)
    lines = error_msg.strip().split('\n')
    msg = lines[0] if lines else "Unknown error"

    # Replace absolute paths with relative workspace paths
    msg = msg.replace(str(WORKSPACE_ROOT), "workspace")

    # Truncate to max length
    if len(msg) > max_length:
        msg = msg[:max_length - 3] + "..."

    return msg
