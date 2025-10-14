#!/usr/bin/env python3
"""
Unit tests for leaderboard validation and aggregation.
"""

import json
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.caq_metric import compute_caq, compute_variance


class TestCAQMetric:
    """Test CAQ metric computation."""
    
    def test_compute_caq_basic(self):
        """Test basic CAQ computation."""
        ratio = 5.63
        cpu_seconds = 0.26
        expected_caq = 5.63 / (0.26 + 1.0)
        
        caq = compute_caq(ratio, cpu_seconds)
        assert abs(caq - expected_caq) < 0.01
    
    def test_compute_caq_zero_time(self):
        """Test CAQ with zero CPU time."""
        caq = compute_caq(10.0, 0.0)
        assert caq == 10.0  # 10 / (0 + 1)
    
    def test_compute_caq_invalid_ratio(self):
        """Test CAQ with invalid ratio."""
        with pytest.raises(ValueError):
            compute_caq(0.0, 0.5)
        
        with pytest.raises(ValueError):
            compute_caq(-1.0, 0.5)
    
    def test_compute_caq_invalid_time(self):
        """Test CAQ with invalid CPU time."""
        with pytest.raises(ValueError):
            compute_caq(5.0, -0.1)


class TestVariance:
    """Test variance computation."""
    
    def test_compute_variance_basic(self):
        """Test basic variance computation."""
        values = [5.60, 5.64, 5.65]
        variance = compute_variance(values)
        
        # (5.65 - 5.60) / mean(5.60, 5.64, 5.65) * 100
        mean = sum(values) / len(values)
        expected = (5.65 - 5.60) / mean * 100
        
        assert abs(variance - expected) < 0.01
    
    def test_compute_variance_identical(self):
        """Test variance with identical values."""
        values = [5.0, 5.0, 5.0]
        variance = compute_variance(values)
        assert variance == 0.0
    
    def test_compute_variance_empty(self):
        """Test variance with empty list."""
        with pytest.raises(ValueError):
            compute_variance([])


class TestSubmissionSchema:
    """Test submission schema validation."""
    
    def test_sample_submission_valid(self):
        """Test that sample submission is valid JSON."""
        sample_path = Path(__file__).parent.parent / "examples" / "sample_submission.json"
        
        with open(sample_path, 'r') as f:
            submission = json.load(f)
        
        # Check required fields
        required_fields = ["submitter", "date", "dataset", "codec", "version",
                          "compression_ratio", "cpu_seconds", "runs", "notes"]
        
        for field in required_fields:
            assert field in submission, f"Missing field: {field}"
        
        # Check runs structure
        assert len(submission["runs"]) >= 3, "Need at least 3 runs"
        for run in submission["runs"]:
            assert "ratio" in run
            assert "cpu_seconds" in run
    
    def test_schema_file_valid_json(self):
        """Test that schema file is valid JSON."""
        schema_path = Path(__file__).parent.parent / "leaderboard" / "leaderboard_schema.json"
        
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        assert "$schema" in schema
        assert "type" in schema
        assert schema["type"] == "object"


class TestDatasets:
    """Test dataset files exist and have expected properties."""
    
    def test_text_medium_exists(self):
        """Test text_medium dataset exists."""
        dataset_path = Path(__file__).parent.parent / "datasets" / "text_medium" / "corpus.txt"
        assert dataset_path.exists()
        
        # Check size is ~100KB
        size = dataset_path.stat().st_size
        assert 90_000 < size < 120_000, f"Expected ~100KB, got {size} bytes"
    
    def test_image_small_exists(self):
        """Test image_small dataset exists."""
        dataset_dir = Path(__file__).parent.parent / "datasets" / "image_small"
        assert dataset_dir.exists()

        # Check at least 3 files (*.bin or *.dat)
        files = list(dataset_dir.glob("*.bin")) + list(dataset_dir.glob("*.dat"))
        assert len(files) >= 3, f"Expected at least 3 image files, found {len(files)}"

    def test_mixed_stream_exists(self):
        """Test mixed_stream dataset exists."""
        dataset_dir = Path(__file__).parent.parent / "datasets" / "mixed_stream"
        assert dataset_dir.exists()

        # Check for either mixed_1mb.dat or mixed_content.bin
        files = list(dataset_dir.glob("*.bin")) + list(dataset_dir.glob("*.dat"))
        assert len(files) > 0, "No mixed stream files found"

        # Check size is ~1MB
        total_size = sum(f.stat().st_size for f in files)
        assert 900_000 < total_size < 1_200_000, f"Expected ~1MB, got {total_size} bytes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
