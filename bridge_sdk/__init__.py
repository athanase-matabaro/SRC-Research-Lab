"""
Bridge SDK - Secure Python interface to SRC Engine (Phase H.1)

This package provides a stable, manifest-driven API for compression tasks
with security controls, resource limits, and CAQ metric integration.

Public API:
    compress(input_path, output_path, config) -> dict
    decompress(input_path, output_path) -> dict
    analyze(input_path, output_path, mode) -> dict

Security features:
    - Workspace-relative path validation (no traversal)
    - Offline-only execution (no network access)
    - Resource limits (CPU time, memory)
    - Sanitized error messages
    - Manifest-driven task validation

Usage:
    >>> import bridge_sdk
    >>> result = bridge_sdk.compress("input.txt", "output.cxe")
    >>> print(result['caq'])
"""

__version__ = "1.0.0"
__author__ = "SRC Research Lab"

from bridge_sdk.api import compress, decompress, analyze
from bridge_sdk.exceptions import (
    BridgeError,
    SecurityError,
    ValidationError,
    TimeoutError,
    ManifestError
)

__all__ = [
    "compress",
    "decompress",
    "analyze",
    "BridgeError",
    "SecurityError",
    "ValidationError",
    "TimeoutError",
    "ManifestError",
    "__version__"
]
