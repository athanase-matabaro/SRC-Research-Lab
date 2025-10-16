#!/usr/bin/env python3
"""
Unit tests for release bundle preparation script.
"""

import json
import pytest
from pathlib import Path
import sys
import tempfile
import hashlib

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.release_prepare import (
    create_sample_dataset,
    create_run_script,
    create_example_submission,
    create_readme,
    compute_checksums,
    create_bundle
)


class TestDatasetCreation:
    """Test sample dataset generation."""

    def test_create_text_medium(self):
        """Test text_medium dataset creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            dataset_dir = create_sample_dataset("text_medium", tmpdir)

            assert dataset_dir.exists()
            assert (dataset_dir / "sample_1.txt").exists()
            assert (dataset_dir / "sample_2.txt").exists()
            assert (dataset_dir / "sample_3.txt").exists()

    def test_create_image_small(self):
        """Test image_small dataset creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            dataset_dir = create_sample_dataset("image_small", tmpdir)

            assert dataset_dir.exists()
            assert (dataset_dir / "image_1.bin").exists()
            assert (dataset_dir / "image_2.bin").exists()

    def test_create_mixed_stream(self):
        """Test mixed_stream dataset creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            dataset_dir = create_sample_dataset("mixed_stream", tmpdir)

            assert dataset_dir.exists()
            assert (dataset_dir / "data.txt").exists()
            assert (dataset_dir / "binary.dat").exists()

    def test_unknown_dataset(self):
        """Test error on unknown dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            with pytest.raises(ValueError, match="Unknown dataset"):
                create_sample_dataset("unknown_dataset", tmpdir)


class TestScriptCreation:
    """Test run script generation."""

    def test_create_run_script(self):
        """Test run script creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            script_path = create_run_script("text_medium", tmpdir)

            assert script_path.exists()
            assert script_path.name == "run_canonical.sh"

            # Check executable
            assert script_path.stat().st_mode & 0o111  # Has execute bit

            # Check content
            content = script_path.read_text()
            assert "#!/bin/bash" in content
            assert "text_medium" in content

    def test_script_contains_benchmark_logic(self):
        """Test script contains benchmark logic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            script_path = create_run_script("text_medium", tmpdir)

            content = script_path.read_text()
            assert "mock_bridge.py" in content
            assert "compress" in content
            assert "CAQ" in content


class TestSubmissionCreation:
    """Test example submission generation."""

    def test_create_submission(self):
        """Test example submission creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            submission_path = create_example_submission("text_medium", tmpdir)

            assert submission_path.exists()
            assert submission_path.name == "example_submission.json"

            # Load and validate JSON
            with open(submission_path, 'r') as f:
                submission = json.load(f)

            assert "submitter" in submission
            assert "dataset" in submission
            assert submission["dataset"] == "text_medium"
            assert "computed_metrics" in submission

    def test_submission_has_metrics(self):
        """Test submission contains required metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            submission_path = create_example_submission("text_medium", tmpdir)

            with open(submission_path, 'r') as f:
                submission = json.load(f)

            metrics = submission["computed_metrics"]
            assert "computed_caq" in metrics
            assert "mean_ratio" in metrics
            assert "mean_cpu" in metrics


class TestReadmeCreation:
    """Test README generation."""

    def test_create_readme(self):
        """Test README creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            readme_path = create_readme("text_medium", tmpdir)

            assert readme_path.exists()
            assert readme_path.name == "README.md"

            content = readme_path.read_text()
            assert "text_medium" in content
            assert "CAQ" in content
            assert "Quick Start" in content

    def test_readme_structure(self):
        """Test README has required sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            readme_path = create_readme("text_medium", tmpdir)

            content = readme_path.read_text()
            assert "## Overview" in content
            assert "## Requirements" in content
            assert "## Quick Start" in content
            assert "## CAQ Metric" in content


class TestChecksumComputation:
    """Test checksum generation."""

    def test_compute_checksums(self):
        """Test checksum file creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create some test files
            (tmpdir / "file1.txt").write_text("test1")
            (tmpdir / "file2.txt").write_text("test2")

            checksum_path = compute_checksums(tmpdir)

            assert checksum_path.exists()
            assert checksum_path.name == "checksum.sha256"

            # Verify content
            content = checksum_path.read_text()
            assert "file1.txt" in content
            assert "file2.txt" in content

    def test_checksum_format(self):
        """Test checksum file format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            (tmpdir / "test.txt").write_text("hello")
            checksum_path = compute_checksums(tmpdir)

            lines = checksum_path.read_text().strip().split('\n')
            for line in lines:
                parts = line.split()
                assert len(parts) == 2
                assert len(parts[0]) == 64  # SHA256 is 64 hex chars


class TestBundleCreation:
    """Test complete bundle creation."""

    def test_create_complete_bundle(self):
        """Test creating a complete bundle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            bundle_dir = create_bundle("text_medium", tmpdir)

            # Check bundle directory exists
            assert bundle_dir.exists()
            assert bundle_dir.name == "text_medium_bundle"

            # Check all required files
            assert (bundle_dir / "dataset").exists()
            assert (bundle_dir / "run_canonical.sh").exists()
            assert (bundle_dir / "example_submission.json").exists()
            assert (bundle_dir / "README.md").exists()
            assert (bundle_dir / "checksum.sha256").exists()
            assert (bundle_dir / "mock_bridge.py").exists()

    def test_bundle_integrity(self):
        """Test bundle checksum integrity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            bundle_dir = create_bundle("text_medium", tmpdir)
            checksum_file = bundle_dir / "checksum.sha256"

            # Verify all checksums
            with open(checksum_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                if not line.strip():
                    continue

                hash_expected, file_path_str = line.strip().split(None, 1)
                file_path = bundle_dir / file_path_str

                assert file_path.exists(), f"Missing file: {file_path_str}"

                with open(file_path, 'rb') as f:
                    hash_actual = hashlib.sha256(f.read()).hexdigest()

                assert hash_actual == hash_expected, f"Checksum mismatch: {file_path_str}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
