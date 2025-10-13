"""
Bridge manifest parser and validator (Phase H.1)

Implements schema-driven task validation based on bridge_manifest.yaml.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from bridge_sdk.exceptions import ManifestError, ValidationError


class ManifestLoader:
    """
    Load and validate bridge_manifest.yaml.

    The manifest defines:
    - Available tasks (compress, decompress, analyze)
    - Argument schemas (type, required, defaults, choices)
    - Resource limits (timeout, CPU, memory)
    """

    def __init__(self, manifest_path: Optional[Path] = None):
        """
        Initialize manifest loader.

        Args:
            manifest_path: Path to bridge_manifest.yaml (default: workspace root)
        """
        if manifest_path is None:
            # Default to workspace root
            from bridge_sdk.security import WORKSPACE_ROOT
            manifest_path = WORKSPACE_ROOT / "bridge_manifest.yaml"

        self.manifest_path = manifest_path
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        """Load and parse manifest YAML file."""
        if not self.manifest_path.exists():
            raise ManifestError(
                f"Manifest not found: {self.manifest_path.name}. "
                "Run setup or check installation."
            )

        try:
            with self.manifest_path.open('r') as f:
                data = yaml.safe_load(f)

            if not data:
                raise ManifestError("Manifest is empty")

            # Validate required fields
            if "version" not in data:
                raise ManifestError("Manifest missing 'version' field")

            if "tasks" not in data or not isinstance(data["tasks"], list):
                raise ManifestError("Manifest missing 'tasks' list")

            return data

        except yaml.YAMLError as e:
            raise ManifestError(f"Invalid YAML syntax: {e}")
        except Exception as e:
            raise ManifestError(f"Failed to load manifest: {type(e).__name__}")

    def get_task(self, task_name: str) -> Dict[str, Any]:
        """
        Get task definition from manifest.

        Args:
            task_name: Task name (compress, decompress, analyze)

        Returns:
            Task definition dictionary

        Raises:
            ManifestError: If task not found
        """
        tasks = self.manifest.get("tasks", [])

        for task in tasks:
            if task.get("name") == task_name:
                return task

        # Task not found
        available = [t.get("name", "unknown") for t in tasks]
        raise ManifestError(
            f"Unknown task '{task_name}'—see bridge_manifest.yaml. "
            f"Available: {', '.join(available)}"
        )

    def validate_args(self, task_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize arguments for a task.

        Args:
            task_name: Task name
            args: User-provided arguments

        Returns:
            Validated and normalized arguments (with defaults applied)

        Raises:
            ValidationError: If validation fails
        """
        task = self.get_task(task_name)
        task_args = task.get("args", [])

        validated = {}

        # Check each defined argument
        for arg_def in task_args:
            arg_name = arg_def.get("name")
            arg_type = arg_def.get("type", "string")
            required = arg_def.get("required", False)
            default = arg_def.get("default")
            choices = arg_def.get("choices")

            # Get provided value or default
            value = args.get(arg_name, default)

            # Check required
            if required and value is None:
                raise ValidationError(f"Missing required argument: {arg_name}")

            # Skip if no value and not required
            if value is None:
                continue

            # Type validation
            if arg_type == "int":
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    raise ValidationError(f"Argument '{arg_name}' must be integer")

            elif arg_type == "float":
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    raise ValidationError(f"Argument '{arg_name}' must be float")

            elif arg_type == "bool":
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes")
                else:
                    value = bool(value)

            elif arg_type in ("path", "string", "enum"):
                value = str(value)

            # Choice validation
            if choices and value not in choices:
                raise ValidationError(
                    f"Invalid value for '{arg_name}': {value}. "
                    f"Choices: {', '.join(map(str, choices))}"
                )

            validated[arg_name] = value

        return validated

    def get_time_limit(self, task_name: str) -> int:
        """
        Get time limit for a task (seconds).

        Args:
            task_name: Task name

        Returns:
            Time limit in seconds (0 = no limit)
        """
        task = self.get_task(task_name)
        return task.get("time_limit_sec", 0)

    def get_cpu_limit(self, task_name: str) -> Optional[int]:
        """
        Get CPU limit for a task.

        Args:
            task_name: Task name

        Returns:
            CPU limit or None
        """
        task = self.get_task(task_name)
        return task.get("cpu_limit")

    def list_tasks(self) -> List[str]:
        """Get list of available task names."""
        tasks = self.manifest.get("tasks", [])
        return [t.get("name", "unknown") for t in tasks]


# Global manifest instance (lazy-loaded)
_manifest_loader = None


def get_manifest() -> ManifestLoader:
    """Get or create global manifest loader instance."""
    global _manifest_loader
    if _manifest_loader is None:
        _manifest_loader = ManifestLoader()
    return _manifest_loader
