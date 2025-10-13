"""
Unit tests for Bridge SDK manifest module (Phase H.1)

Tests manifest loading, task validation, and argument validation.
"""

import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bridge_sdk.manifest import ManifestLoader, get_manifest
from bridge_sdk.exceptions import ManifestError, ValidationError


class TestManifestLoader:
    """Test ManifestLoader class."""

    def test_load_manifest_success(self):
        """Test successful manifest loading."""
        manifest = get_manifest()
        assert manifest.manifest is not None
        assert "version" in manifest.manifest
        assert "tasks" in manifest.manifest

    def test_get_task_compress(self):
        """Test getting compress task definition."""
        manifest = get_manifest()
        task = manifest.get_task("compress")

        assert task["name"] == "compress"
        assert "args" in task
        assert "time_limit_sec" in task

    def test_get_task_decompress(self):
        """Test getting decompress task definition."""
        manifest = get_manifest()
        task = manifest.get_task("decompress")

        assert task["name"] == "decompress"
        assert "args" in task

    def test_get_task_unknown(self):
        """Test getting unknown task raises error."""
        manifest = get_manifest()

        with pytest.raises(ManifestError) as exc_info:
            manifest.get_task("unknown_task")

        assert "unknown_task" in str(exc_info.value).lower()

    def test_list_tasks(self):
        """Test listing available tasks."""
        manifest = get_manifest()
        tasks = manifest.list_tasks()

        assert "compress" in tasks
        assert "decompress" in tasks
        assert "analyze" in tasks


class TestArgValidation:
    """Test argument validation against manifest."""

    def test_validate_compress_args_minimal(self):
        """Test validation of minimal compress arguments."""
        manifest = get_manifest()
        args = {
            "input": "tests/fixtures/test_input.txt",
            "output": "results/output.cxe"
        }

        validated = manifest.validate_args("compress", args)

        assert validated["input"] == "tests/fixtures/test_input.txt"
        assert validated["output"] == "results/output.cxe"
        assert validated["backend"] == "src_engine_private"  # Default
        assert validated["workers"] == 1  # Default

    def test_validate_compress_args_full(self):
        """Test validation of full compress arguments."""
        manifest = get_manifest()
        args = {
            "input": "tests/fixtures/test_input.txt",
            "output": "results/output.cxe",
            "backend": "reference_zstd",
            "workers": 4,
            "care": True
        }

        validated = manifest.validate_args("compress", args)

        assert validated["input"] == "tests/fixtures/test_input.txt"
        assert validated["output"] == "results/output.cxe"
        assert validated["backend"] == "reference_zstd"
        assert validated["workers"] == 4
        assert validated["care"] is True

    def test_validate_missing_required_arg(self):
        """Test validation fails for missing required argument."""
        manifest = get_manifest()
        args = {
            "output": "results/output.cxe"
            # Missing required "input"
        }

        with pytest.raises(ValidationError) as exc_info:
            manifest.validate_args("compress", args)

        assert "input" in str(exc_info.value).lower()

    def test_validate_invalid_choice(self):
        """Test validation fails for invalid choice value."""
        manifest = get_manifest()
        args = {
            "input": "tests/fixtures/test_input.txt",
            "output": "results/output.cxe",
            "backend": "invalid_backend"
        }

        with pytest.raises(ValidationError) as exc_info:
            manifest.validate_args("compress", args)

        assert "invalid" in str(exc_info.value).lower() or "choices" in str(exc_info.value).lower()

    def test_validate_int_type_conversion(self):
        """Test validation converts string to int."""
        manifest = get_manifest()
        args = {
            "input": "tests/fixtures/test_input.txt",
            "output": "results/output.cxe",
            "workers": "4"  # String instead of int
        }

        validated = manifest.validate_args("compress", args)

        assert validated["workers"] == 4
        assert isinstance(validated["workers"], int)

    def test_validate_bool_type_conversion(self):
        """Test validation converts string to bool."""
        manifest = get_manifest()
        args = {
            "input": "tests/fixtures/test_input.txt",
            "output": "results/output.cxe",
            "care": "true"  # String instead of bool
        }

        validated = manifest.validate_args("compress", args)

        assert validated["care"] is True
        assert isinstance(validated["care"], bool)


class TestResourceLimits:
    """Test resource limit retrieval."""

    def test_get_time_limit_compress(self):
        """Test getting time limit for compress task."""
        manifest = get_manifest()
        time_limit = manifest.get_time_limit("compress")

        assert isinstance(time_limit, int)
        assert time_limit > 0

    def test_get_cpu_limit(self):
        """Test getting CPU limit."""
        manifest = get_manifest()
        cpu_limit = manifest.get_cpu_limit("compress")

        # May be None (no limit)
        assert cpu_limit is None or isinstance(cpu_limit, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
