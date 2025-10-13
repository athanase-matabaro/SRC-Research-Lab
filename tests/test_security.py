"""
Unit tests for Bridge SDK security module (Phase H.1)

Tests path validation, timeout enforcement, and network prevention.
"""

import pytest
import sys
import os
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bridge_sdk.security import (
    validate_workspace_path,
    disallow_network,
    enforce_timeout,
    sanitize_error_message,
    WORKSPACE_ROOT
)
from bridge_sdk.exceptions import SecurityError, ValidationError, TimeoutError


class TestPathValidation:
    """Test workspace path validation."""

    def test_valid_workspace_path(self):
        """Test validation of valid workspace-relative path."""
        path = validate_workspace_path("tests/fixtures/test_input.txt", must_exist=True)
        assert path.is_absolute()
        assert path.exists()
        assert path.is_relative_to(WORKSPACE_ROOT)

    def test_valid_nonexistent_path(self):
        """Test validation of non-existent path when not required."""
        path = validate_workspace_path("results/new_file.txt", must_exist=False)
        assert path.is_absolute()
        assert path.is_relative_to(WORKSPACE_ROOT)

    def test_nonexistent_path_required(self):
        """Test validation fails for non-existent path when required."""
        with pytest.raises(ValidationError):
            validate_workspace_path("results/nonexistent_file.txt", must_exist=True)

    def test_traversal_attack_dotdot(self):
        """Test path traversal prevention with ../."""
        with pytest.raises(SecurityError):
            validate_workspace_path("../foundation_charter.md")

    def test_traversal_attack_absolute(self):
        """Test path traversal prevention with absolute path outside workspace."""
        with pytest.raises(SecurityError):
            validate_workspace_path("/etc/passwd")

    def test_empty_path(self):
        """Test empty path validation."""
        with pytest.raises(ValidationError):
            validate_workspace_path("")

    def test_none_path(self):
        """Test None path validation."""
        with pytest.raises(ValidationError):
            validate_workspace_path(None)


class TestNetworkPrevention:
    """Test network access prevention."""

    def test_disallow_network_clears_proxies(self):
        """Test that disallow_network() clears proxy environment variables."""
        # Set some proxy vars
        os.environ["http_proxy"] = "http://example.com:8080"
        os.environ["https_proxy"] = "https://example.com:8080"

        # Call disallow_network
        disallow_network()

        # Verify proxies are cleared
        assert "http_proxy" not in os.environ
        assert "https_proxy" not in os.environ

    def test_disallow_network_no_proxies(self):
        """Test disallow_network() when no proxies are set."""
        # Clear all proxy vars first
        for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
            os.environ.pop(var, None)

        # Should not raise
        disallow_network()


class TestTimeoutEnforcement:
    """Test timeout enforcement."""

    def test_timeout_context_manager_success(self):
        """Test timeout context manager with fast operation."""
        with enforce_timeout(5, "test"):
            time.sleep(0.1)  # Fast operation
        # Should complete without timeout

    def test_timeout_context_manager_timeout(self):
        """Test timeout context manager with slow operation."""
        with pytest.raises(TimeoutError):
            with enforce_timeout(1, "test"):
                time.sleep(5)  # Slow operation should timeout

    def test_timeout_zero_no_limit(self):
        """Test timeout=0 means no limit."""
        with enforce_timeout(0, "test"):
            time.sleep(0.1)
        # Should complete without timeout

    def test_timeout_negative_no_limit(self):
        """Test timeout<0 means no limit."""
        with enforce_timeout(-1, "test"):
            time.sleep(0.1)
        # Should complete without timeout


class TestErrorSanitization:
    """Test error message sanitization."""

    def test_sanitize_multiline_error(self):
        """Test sanitization of multiline error (stack trace)."""
        error = "Error message\nTraceback line 1\nTraceback line 2"
        sanitized = sanitize_error_message(error)

        assert "Traceback" not in sanitized
        assert "\n" not in sanitized
        assert sanitized == "Error message"

    def test_sanitize_long_error(self):
        """Test sanitization truncates long messages."""
        error = "A" * 300
        sanitized = sanitize_error_message(error, max_length=200)

        assert len(sanitized) <= 200
        assert sanitized.endswith("...")

    def test_sanitize_replaces_workspace_path(self):
        """Test sanitization replaces workspace absolute paths."""
        error = f"Error in {WORKSPACE_ROOT}/some/file.txt"
        sanitized = sanitize_error_message(error)

        assert str(WORKSPACE_ROOT) not in sanitized
        assert "workspace" in sanitized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
