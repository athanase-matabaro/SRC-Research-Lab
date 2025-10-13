"""
Bridge SDK exception hierarchy.

All exceptions include sanitized error messages with no internal paths
or stack traces exposed to users.
"""


class BridgeError(Exception):
    """Base exception for all Bridge SDK errors."""
    def __init__(self, message: str, code: int = 1):
        self.message = message
        self.code = code
        super().__init__(message)


class SecurityError(BridgeError):
    """Security violation (path traversal, network access, etc.)."""
    def __init__(self, message: str):
        super().__init__(message, code=400)


class ValidationError(BridgeError):
    """Input validation failure (invalid args, missing files, etc.)."""
    def __init__(self, message: str):
        super().__init__(message, code=400)


class TimeoutError(BridgeError):
    """Task exceeded time limit."""
    def __init__(self, message: str, limit_sec: int):
        self.limit_sec = limit_sec
        super().__init__(message, code=408)


class ManifestError(BridgeError):
    """Manifest parsing or task definition error."""
    def __init__(self, message: str):
        super().__init__(message, code=404)


class EngineError(BridgeError):
    """SRC Engine execution failure."""
    def __init__(self, message: str, engine_code: int = 1):
        self.engine_code = engine_code
        super().__init__(message, code=500)
