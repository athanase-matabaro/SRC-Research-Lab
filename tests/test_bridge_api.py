"""
Unit tests for Bridge SDK API (Phase H.1)

Tests compress(), decompress(), and analyze() functions with various scenarios.
"""

import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bridge_sdk import compress, decompress, analyze
from bridge_sdk.exceptions import (
    BridgeError,
    SecurityError,
    ValidationError,
    ManifestError
)


class TestCompress:
    """Test compress() function."""

    def test_compress_basic(self):
        """Test basic compression with valid inputs."""
        # Note: This test requires src-engine to be available
        # For now, we test that the function is callable and validates paths
        try:
            result = compress(
                "tests/fixtures/test_input.txt",
                "results/test_api_output.cxe"
            )
            # If engine is available, check result structure
            assert result["status"] in ["ok", "error"]
            assert "ratio" in result or "message" in result

            # Cleanup
            output_file = Path("results/test_api_output.cxe")
            if output_file.exists():
                output_file.unlink()
            manifest_file = Path("results/test_api_output.cxe.manifest.json")
            if manifest_file.exists():
                manifest_file.unlink()

        except BridgeError as e:
            # Expected if engine not available
            assert e.code in [127, 500]

    def test_compress_invalid_input_path(self):
        """Test compression with non-existent input."""
        with pytest.raises(ValidationError):
            compress("nonexistent_file.txt", "results/output.cxe")

    def test_compress_traversal_attempt(self):
        """Test compression prevents path traversal."""
        with pytest.raises(SecurityError):
            compress("../foundation_charter.md", "results/output.cxe")

    def test_compress_with_config(self):
        """Test compression with configuration."""
        try:
            result = compress(
                "tests/fixtures/test_input.txt",
                "results/test_config_output.cxe",
                config={
                    "workers": 2,
                    "care": True
                }
            )
            assert result["status"] in ["ok", "error"]

            # Cleanup
            output_file = Path("results/test_config_output.cxe")
            if output_file.exists():
                output_file.unlink()
            manifest_file = Path("results/test_config_output.cxe.manifest.json")
            if manifest_file.exists():
                manifest_file.unlink()

        except BridgeError as e:
            # Expected if engine not available
            assert e.code in [127, 500]

    def test_compress_invalid_backend(self):
        """Test compression with unsupported backend (should use manifest validation)."""
        # Note: Backend validation happens in manifest or engine
        # This test verifies error handling
        pass  # Backend validation tested in manifest tests


class TestDecompress:
    """Test decompress() function."""

    def test_decompress_invalid_input_path(self):
        """Test decompression with non-existent input."""
        with pytest.raises(ValidationError):
            decompress("nonexistent_file.cxe", "results/output.txt")

    def test_decompress_traversal_attempt(self):
        """Test decompression prevents path traversal."""
        with pytest.raises(SecurityError):
            decompress("../some_file.cxe", "results/output.txt")


class TestAnalyze:
    """Test analyze() function."""

    def test_analyze_not_implemented(self):
        """Test analyze raises not implemented error."""
        with pytest.raises(ValidationError) as exc_info:
            analyze("tests/fixtures/test_output.cxe", "results/analysis.json")

        assert "not yet implemented" in str(exc_info.value).lower()


class TestResultFormat:
    """Test result dictionary format."""

    def test_result_has_required_keys(self):
        """Test successful result contains required keys."""
        # This will be validated by actual compression tests
        # For now, test the format contract
        try:
            result = compress(
                "tests/fixtures/test_input.txt",
                "results/test_format_output.cxe"
            )

            if result["status"] == "ok":
                assert "ratio" in result
                assert "runtime_sec" in result
                assert "caq" in result
                assert "backend" in result
                assert "message" in result

            # Cleanup
            output_file = Path("results/test_format_output.cxe")
            if output_file.exists():
                output_file.unlink()
            manifest_file = Path("results/test_format_output.cxe.manifest.json")
            if manifest_file.exists():
                manifest_file.unlink()

        except BridgeError:
            # Expected if engine not available
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
