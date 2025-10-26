"""
H5 Smoke Tests - SRC Research Lab
Basic smoke tests for H5 infrastructure
"""

import os
import subprocess
from pathlib import Path


def test_h5_status_board_exists():
    """Verify H5 status board exists."""
    assert Path("docs/H5-status-board.md").exists()
    content = Path("docs/H5-status-board.md").read_text()
    assert "H5-01" in content
    assert "H5-05" in content


def test_feature_spec_template_exists():
    """Verify feature spec template exists."""
    assert Path("docs/templates/feature-spec.md").exists()


def test_pr_template_exists():
    """Verify PR template exists."""
    assert Path(".github/PULL_REQUEST_TEMPLATE.md").exists()


def test_decision_log_exists():
    """Verify decision log with H5 decisions."""
    assert Path("docs/decision-log.md").exists()
    content = Path("docs/decision-log.md").read_text()
    assert "DEC-2025-10-26-001" in content
    assert "DEC-2025-10-26-002" in content
    assert "DEC-2025-10-26-003" in content


def test_h5_scripts_exist_and_executable():
    """Verify H5 scripts are present and executable."""
    scripts = [
        "scripts/run_tests.sh",
        "scripts/compliance_check.sh",
        "scripts/check_private_exposure.sh",
    ]
    
    for script in scripts:
        path = Path(script)
        assert path.exists(), f"{script} must exist"
        assert os.access(path, os.X_OK), f"{script} must be executable"


def test_h5_workflow_exists():
    """Verify H5 GitHub Actions workflow exists."""
    workflow = Path(".github/workflows/h5-gates.yml")
    assert workflow.exists()
    content = workflow.read_text()
    assert "h5_compliance_gate" in content


def test_private_content_policy_exists():
    """Verify private content policy exists."""
    assert Path("docs/ops/PRIVATE_CONTENT.md").exists()


def test_branch_protection_docs_exist():
    """Verify branch protection docs exist."""
    assert Path("docs/ops/branch-protection.md").exists()


def test_scripts_syntax_valid():
    """Verify bash scripts have valid syntax."""
    scripts = [
        "scripts/run_tests.sh",
        "scripts/compliance_check.sh",
        "scripts/check_private_exposure.sh",
    ]
    
    for script in scripts:
        result = subprocess.run(
            ["bash", "-n", script],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"{script} has syntax errors: {result.stderr}"


def test_no_parent_directory_references():
    """Verify no git-tracked files reference parent directories."""
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True
    )
    
    files = result.stdout.splitlines()
    
    # Check that no files are outside src-research-lab
    for file in files:
        assert not file.startswith("../"), f"File escapes directory: {file}"
        assert not file.startswith("/home/"), f"Absolute path detected: {file}"


def test_h5_focus_checklist_exists():
    """Verify H5 focus checklist exists."""
    assert Path("docs/h5-focus-checklist.md").exists()


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
