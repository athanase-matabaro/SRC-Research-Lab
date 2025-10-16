#!/usr/bin/env python3
"""
Unit tests for adaptive results integration script.
"""

import json
import pytest
from pathlib import Path
import sys
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.integrate_adaptive import (
    validate_adaptive_schema,
    compute_delta_vs_baseline,
    compute_signature,
    create_validation_report,
    integrate_adaptive_runs
)


class TestSchemaValidation:
    """Test adaptive result schema validation."""

    def test_valid_schema(self):
        """Test validation of valid adaptive result."""
        data = {
            "timestamp": "2025-10-16T12:00:00",
            "dataset": "synthetic_gradients",
            "epochs": 10,
            "mean_baseline_caq": 1.33,
            "mean_adaptive_caq": 1.60,
            "mean_gain_percent": 20.14,
            "results": [
                {
                    "status": "PASS",
                    "adaptive_caq": 1.58,
                    "baseline_caq": 1.33,
                    "gain_percent": 19.41
                }
            ]
        }

        valid, msg = validate_adaptive_schema(data)
        assert valid
        assert msg == "OK"

    def test_missing_field(self):
        """Test validation fails with missing field."""
        data = {
            "timestamp": "2025-10-16T12:00:00",
            "dataset": "synthetic_gradients"
            # Missing required fields
        }

        valid, msg = validate_adaptive_schema(data)
        assert not valid
        assert "Missing required field" in msg

    def test_empty_results(self):
        """Test validation fails with empty results."""
        data = {
            "timestamp": "2025-10-16T12:00:00",
            "dataset": "synthetic_gradients",
            "epochs": 10,
            "mean_baseline_caq": 1.33,
            "mean_adaptive_caq": 1.60,
            "mean_gain_percent": 20.14,
            "results": []
        }

        valid, msg = validate_adaptive_schema(data)
        assert not valid
        assert "non-empty list" in msg

    def test_invalid_result_status(self):
        """Test validation fails with non-PASS status."""
        data = {
            "timestamp": "2025-10-16T12:00:00",
            "dataset": "synthetic_gradients",
            "epochs": 10,
            "mean_baseline_caq": 1.33,
            "mean_adaptive_caq": 1.60,
            "mean_gain_percent": 20.14,
            "results": [
                {
                    "status": "FAIL",
                    "adaptive_caq": 1.58,
                    "baseline_caq": 1.33
                }
            ]
        }

        valid, msg = validate_adaptive_schema(data)
        assert not valid
        assert "status not PASS" in msg


class TestDeltaComputation:
    """Test delta vs baseline computation."""

    def test_positive_delta(self):
        """Test positive improvement."""
        delta = compute_delta_vs_baseline(1.60, 1.33)
        assert abs(delta - 20.30) < 0.1  # ~20.3%

    def test_negative_delta(self):
        """Test negative change."""
        delta = compute_delta_vs_baseline(1.20, 1.50)
        assert delta < 0

    def test_zero_baseline(self):
        """Test zero baseline handling."""
        delta = compute_delta_vs_baseline(1.60, 0.0)
        assert delta == 0.0

    def test_equal_values(self):
        """Test no change."""
        delta = compute_delta_vs_baseline(1.50, 1.50)
        assert delta == 0.0


class TestSignatureComputation:
    """Test signature generation."""

    def test_deterministic_signature(self):
        """Test signature is deterministic."""
        data = {"foo": "bar", "baz": 123}
        sig1 = compute_signature(data)
        sig2 = compute_signature(data)
        assert sig1 == sig2

    def test_different_data_different_signature(self):
        """Test different data produces different signature."""
        data1 = {"foo": "bar"}
        data2 = {"foo": "baz"}
        sig1 = compute_signature(data1)
        sig2 = compute_signature(data2)
        assert sig1 != sig2

    def test_signature_length(self):
        """Test signature is 16-character hex."""
        data = {"test": "data"}
        sig = compute_signature(data)
        assert len(sig) == 16
        assert all(c in "0123456789abcdef" for c in sig)


class TestValidationReport:
    """Test validation report creation."""

    def test_create_report(self):
        """Test report creation from adaptive data."""
        adaptive_data = {
            "timestamp": "2025-10-16T12:00:00",
            "dataset": "synthetic_gradients",
            "epochs": 10,
            "mean_baseline_caq": 1.33,
            "mean_adaptive_caq": 1.60,
            "mean_gain_percent": 20.14,
            "caq_variance": 4.37,
            "entropy_loss": 0.0074,
            "notes": "Test run"
        }

        source_file = Path("test_run.json")
        report = create_validation_report(adaptive_data, source_file)

        assert report["status"] == "PASS"
        assert report["adaptive_flag"] is True
        assert report["submission"]["dataset"] == "synthetic_gradients"
        assert report["computed_metrics"]["computed_caq"] == 1.60
        assert report["computed_metrics"]["delta_vs_src_baseline"] == 20.3
        assert "signature" in report

    def test_report_submitter(self):
        """Test report uses correct submitter."""
        adaptive_data = {
            "dataset": "test",
            "epochs": 1,
            "mean_baseline_caq": 1.0,
            "mean_adaptive_caq": 1.2,
            "mean_gain_percent": 20.0
        }

        report = create_validation_report(adaptive_data, Path("test.json"))
        assert report["submission"]["submitter"] == "athanase_lab"


class TestIntegration:
    """Test full integration workflow."""

    def test_integrate_single_file(self):
        """Test integrating a single adaptive result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test input
            input_file = tmpdir / "test_adaptive.json"
            adaptive_data = {
                "timestamp": "2025-10-16T12:00:00",
                "dataset": "synthetic_gradients",
                "epochs": 10,
                "mean_baseline_caq": 1.33,
                "mean_adaptive_caq": 1.60,
                "mean_gain_percent": 20.14,
                "results": [
                    {
                        "status": "PASS",
                        "adaptive_caq": 1.58,
                        "baseline_caq": 1.33
                    }
                ]
            }

            with open(input_file, 'w') as f:
                json.dump(adaptive_data, f)

            # Create output directory
            out_reports_dir = tmpdir / "reports"

            # Integrate
            count = integrate_adaptive_runs([input_file], out_reports_dir)

            assert count == 1
            assert len(list(out_reports_dir.glob("*.json"))) == 1

            # Verify report content
            report_file = list(out_reports_dir.glob("*.json"))[0]
            with open(report_file, 'r') as f:
                report = json.load(f)

            assert report["status"] == "PASS"
            assert report["adaptive_flag"] is True
            assert report["computed_metrics"]["computed_caq"] == 1.60

    def test_integrate_multiple_files(self):
        """Test integrating multiple adaptive results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create two test inputs
            for i in range(2):
                input_file = tmpdir / f"test_adaptive_{i}.json"
                adaptive_data = {
                    "timestamp": "2025-10-16T12:00:00",
                    "dataset": "synthetic_gradients",
                    "epochs": 10,
                    "mean_baseline_caq": 1.33,
                    "mean_adaptive_caq": 1.60 + i * 0.05,
                    "mean_gain_percent": 20.0 + i * 2.0,
                    "results": [
                        {
                            "status": "PASS",
                            "adaptive_caq": 1.58,
                            "baseline_caq": 1.33
                        }
                    ]
                }

                with open(input_file, 'w') as f:
                    json.dump(adaptive_data, f)

            out_reports_dir = tmpdir / "reports"

            # Integrate files one at a time to ensure unique timestamps
            import time
            input_files = sorted(tmpdir.glob("test_adaptive_*.json"))
            for input_file in input_files:
                integrate_adaptive_runs([input_file], out_reports_dir)
                time.sleep(0.01)  # Small delay to ensure unique timestamps

            reports = list(out_reports_dir.glob("*.json"))
            assert len(reports) == 2

    def test_invalid_input_file(self):
        """Test handling of invalid input file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create invalid JSON
            input_file = tmpdir / "invalid.json"
            with open(input_file, 'w') as f:
                f.write("{invalid json")

            out_reports_dir = tmpdir / "reports"

            # Should return error code
            result = integrate_adaptive_runs([input_file], out_reports_dir)
            assert result == 2  # Schema error code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
