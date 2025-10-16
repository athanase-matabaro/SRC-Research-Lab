#!/usr/bin/env python3
"""
Root Directory Cleanliness Checker

Verifies that the repository root directory contains only allowed files
according to the engineering culture guidelines.

Exit codes:
    0: Root directory is clean
    1: Violations found
"""

import sys
from pathlib import Path


# Allowed files in root directory
ALLOWED_ROOT_FILES = {
    "README.md",
    "LICENSE.md",
    ".gitignore",
    ".claude.md",
    ".git-workflow-checklist.md",
    ".ai-assistant-protocol.md",
    "requirements.txt",
    "setup.py",
    "pyproject.toml",
    "setup.cfg",
}

# Allowed directories in root
ALLOWED_ROOT_DIRS = {
    ".git",
    ".github",
    ".pytest_cache",
    ".claude",
    "__pycache__",
    "adaptive_model",
    "bridge_sdk",
    "datasets",
    "docs",
    "examples",
    "experiments",
    "leaderboard",
    "metrics",
    "release",
    "results",
    "scripts",
    "src",
    "tests",
    "tools",
}


def check_root_directory(root_path: Path) -> tuple[bool, list[str]]:
    """
    Check if root directory is clean.

    Args:
        root_path: Path to repository root

    Returns:
        Tuple of (is_clean, violations_list)
    """
    violations = []

    for item in root_path.iterdir():
        # Skip hidden files (except explicitly allowed ones)
        if item.name.startswith(".") and item.name not in ALLOWED_ROOT_FILES:
            if item.is_dir() and item.name not in ALLOWED_ROOT_DIRS:
                violations.append(f"Hidden directory: {item.name} (should it be in .gitignore?)")
            continue

        if item.is_file():
            if item.name not in ALLOWED_ROOT_FILES:
                # Suggest where it should go
                suggestion = get_suggestion(item.name)
                violations.append(f"File: {item.name} {suggestion}")

        elif item.is_dir():
            if item.name not in ALLOWED_ROOT_DIRS:
                violations.append(f"Directory: {item.name} (not in allowed directories list)")

    return len(violations) == 0, violations


def get_suggestion(filename: str) -> str:
    """Get suggestion for where file should be moved."""
    if filename.startswith("PHASE_"):
        return "→ should be in docs/phases/"
    elif "VALIDATION" in filename.upper():
        return "→ should be in docs/validation/"
    elif "SUMMARY" in filename.upper() or "COMPLETE" in filename.upper():
        return "→ should be in docs/phases/"
    elif filename.endswith(".md"):
        return "→ should be in docs/"
    elif filename.endswith(".py"):
        return "→ should be in scripts/ or tests/ or src/"
    elif filename.endswith(".yaml") or filename.endswith(".yml"):
        return "→ should be in appropriate module directory"
    elif filename.endswith(".txt") and filename != "requirements.txt":
        return "→ should be in docs/ or docs/validation/"
    else:
        return "→ move to appropriate subdirectory"


def main():
    """Main entry point."""
    # Get repository root (parent of scripts/ directory)
    root_path = Path(__file__).parent.parent.resolve()

    print("=" * 70)
    print("ROOT DIRECTORY CLEANLINESS CHECK")
    print("=" * 70)
    print(f"Checking: {root_path}")
    print()

    is_clean, violations = check_root_directory(root_path)

    if is_clean:
        print("✅ ROOT DIRECTORY IS CLEAN")
        print()
        print("All files and directories in root comply with engineering culture.")
        print()
        print("Allowed in root:")
        print("  Files:")
        for f in sorted(ALLOWED_ROOT_FILES):
            if (root_path / f).exists():
                print(f"    ✓ {f}")
        print("  Directories:")
        for d in sorted(ALLOWED_ROOT_DIRS):
            if (root_path / d).exists() and not d.startswith("_"):
                print(f"    ✓ {d}/")
        return 0

    else:
        print("❌ ROOT DIRECTORY VIOLATIONS FOUND")
        print()
        print(f"Found {len(violations)} violation(s):")
        print()
        for violation in violations:
            print(f"  • {violation}")
        print()
        print("=" * 70)
        print("ACTION REQUIRED:")
        print("=" * 70)
        print()
        print("Move violating files to appropriate directories:")
        print()
        print("  git mv <file> <destination>")
        print()
        print("Common destinations:")
        print("  - Phase summaries → docs/phases/")
        print("  - Validation reports → docs/validation/")
        print("  - Documentation → docs/")
        print("  - Scripts → scripts/")
        print("  - Tests → tests/")
        print()
        print("See .ai-assistant-protocol.md for complete file placement rules.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
