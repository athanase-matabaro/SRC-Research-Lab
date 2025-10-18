"""
Profiler Metadata Schema & Version Migration

Provides metadata schema definitions, version management, and backward compatibility
for energy profiler outputs across all phases (H5.x → C2 → B1 → B2 → B3 → A3).

Schema Versions:
    v1.0: Initial schema (H5.x, C2) - basic energy/time measurements
    v2.0: Added method field (B1, B2)
    v2.1: Full metadata (A3) - method, error_pct, calibration_ref

Security: Offline-only, no network access, local schema validation only.

Author: Athanase Nshombo (Matabaro)
Date: 2025-10-18
Phase: A.3 - Profiler Metadata & Method Consistency
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Schema version constants
SCHEMA_VERSION_1_0 = "1.0"
SCHEMA_VERSION_2_0 = "2.0"
SCHEMA_VERSION_2_1 = "2.1"
CURRENT_SCHEMA_VERSION = SCHEMA_VERSION_2_1

# Profiling method identifiers
METHOD_RAPL = "rapl_cpu_energy_v2"
METHOD_CONSTANT = "constant_power_model"
METHOD_MOCK = "mock_bridge_emulation"
METHOD_UNKNOWN = "unknown"

# Default values for missing metadata
DEFAULT_ERROR_PCT = 0.0
DEFAULT_CALIBRATION_REF = "none"


@dataclass
class ProfilerMetadata:
    """
    Metadata for energy profiler measurements.

    Attributes:
        method: Profiling method used (e.g., 'rapl_cpu_energy_v2', 'constant_power_model')
        error_pct: Measured relative error percentage (0.0 if unknown)
        calibration_ref: Hash or UUID linking to calibration source
        schema_version: Schema version identifier (default: 2.1)
        timestamp: ISO 8601 timestamp of measurement
        cpu_model: CPU model identifier (optional)
        system_id: System identifier for reproducibility (optional)
    """
    method: str
    error_pct: float = DEFAULT_ERROR_PCT
    calibration_ref: str = DEFAULT_CALIBRATION_REF
    schema_version: str = CURRENT_SCHEMA_VERSION
    timestamp: Optional[str] = None
    cpu_model: Optional[str] = None
    system_id: Optional[str] = None

    def __post_init__(self):
        """Validate metadata fields after initialization."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

        # Validate method
        if not isinstance(self.method, str) or len(self.method) == 0:
            raise ValueError(f"Invalid method: {self.method}")

        # Validate error_pct
        if not isinstance(self.error_pct, (int, float)):
            raise ValueError(f"error_pct must be numeric, got {type(self.error_pct)}")
        if not (-100.0 <= self.error_pct <= 100.0):
            raise ValueError(f"error_pct must be in range [-100, 100], got {self.error_pct}")

        # Validate calibration_ref
        if not isinstance(self.calibration_ref, str):
            raise ValueError(f"calibration_ref must be string, got {type(self.calibration_ref)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProfilerMetadata':
        """Create metadata from dictionary."""
        return cls(**data)


@dataclass
class ProfilerResult:
    """
    Complete profiler result with measurements and metadata.

    Attributes:
        energy_joules: Energy consumed in joules
        elapsed_seconds: Time elapsed in seconds
        avg_power_watts: Average power in watts
        metadata: Profiler metadata
    """
    energy_joules: float
    elapsed_seconds: float
    avg_power_watts: float
    metadata: ProfilerMetadata

    def __post_init__(self):
        """Validate result fields."""
        # Validate all fields are finite
        if not all(isinstance(x, (int, float)) and x >= 0 for x in
                   [self.energy_joules, self.elapsed_seconds, self.avg_power_watts]):
            raise ValueError("All measurement values must be non-negative finite numbers")

        if not isinstance(self.metadata, ProfilerMetadata):
            raise ValueError(f"metadata must be ProfilerMetadata, got {type(self.metadata)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "energy_joules": self.energy_joules,
            "elapsed_seconds": self.elapsed_seconds,
            "avg_power_watts": self.avg_power_watts,
            "metadata": self.metadata.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProfilerResult':
        """Create result from dictionary."""
        metadata_dict = data.get('metadata', {})
        metadata = ProfilerMetadata.from_dict(metadata_dict)
        return cls(
            energy_joules=data['energy_joules'],
            elapsed_seconds=data['elapsed_seconds'],
            avg_power_watts=data['avg_power_watts'],
            metadata=metadata
        )


class SchemaMigrator:
    """
    Handles schema version migration for backward compatibility.

    Supports migration from v1.0 → v2.0 → v2.1
    """

    @staticmethod
    def detect_version(data: Dict[str, Any]) -> str:
        """
        Detect schema version from profiler result data.

        Args:
            data: Profiler result dictionary

        Returns:
            Schema version string (1.0, 2.0, or 2.1)
        """
        # Check for explicit schema_version in metadata
        if 'metadata' in data and 'schema_version' in data['metadata']:
            return data['metadata']['schema_version']

        # Check for v2.1 fields
        if 'metadata' in data:
            metadata = data['metadata']
            if all(k in metadata for k in ['method', 'error_pct', 'calibration_ref']):
                return SCHEMA_VERSION_2_1

            # Check for v2.0 (has method but not full metadata)
            if 'method' in metadata:
                return SCHEMA_VERSION_2_0

        # Check for v1.0 (no metadata at all, or minimal metadata)
        if 'method' in data:
            return SCHEMA_VERSION_2_0

        return SCHEMA_VERSION_1_0

    @staticmethod
    def migrate_v1_to_v2(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate v1.0 schema to v2.0.

        v1.0: Basic measurements only
        v2.0: Adds 'method' field

        Args:
            data: v1.0 profiler result

        Returns:
            v2.0 profiler result with method field
        """
        migrated = data.copy()

        # Add method field based on available data
        if 'metadata' not in migrated:
            migrated['metadata'] = {}

        if 'method' not in migrated['metadata']:
            # Infer method from data characteristics
            if migrated.get('rapl_available', False):
                migrated['metadata']['method'] = METHOD_RAPL
            else:
                migrated['metadata']['method'] = METHOD_CONSTANT

        migrated['metadata']['schema_version'] = SCHEMA_VERSION_2_0

        return migrated

    @staticmethod
    def migrate_v2_to_v2_1(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate v2.0 schema to v2.1.

        v2.0: Has method field
        v2.1: Full metadata (method, error_pct, calibration_ref)

        Args:
            data: v2.0 profiler result

        Returns:
            v2.1 profiler result with complete metadata
        """
        migrated = data.copy()

        if 'metadata' not in migrated:
            migrated['metadata'] = {}

        metadata = migrated['metadata']

        # Ensure method exists
        if 'method' not in metadata:
            metadata['method'] = METHOD_UNKNOWN

        # Add missing v2.1 fields with defaults
        if 'error_pct' not in metadata:
            metadata['error_pct'] = DEFAULT_ERROR_PCT

        if 'calibration_ref' not in metadata:
            # Generate calibration ref from method + timestamp
            method = metadata['method']
            timestamp = metadata.get('timestamp', datetime.utcnow().isoformat() + "Z")
            cal_string = f"{method}:{timestamp}"
            metadata['calibration_ref'] = hashlib.sha256(cal_string.encode()).hexdigest()[:16]

        if 'timestamp' not in metadata:
            metadata['timestamp'] = datetime.utcnow().isoformat() + "Z"

        metadata['schema_version'] = SCHEMA_VERSION_2_1

        return migrated

    @classmethod
    def migrate_to_latest(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate profiler result to latest schema version (v2.1).

        Args:
            data: Profiler result in any version

        Returns:
            Profiler result in v2.1 schema
        """
        version = cls.detect_version(data)

        # Already latest version
        if version == CURRENT_SCHEMA_VERSION:
            return data

        # Migrate v1.0 → v2.0 → v2.1
        if version == SCHEMA_VERSION_1_0:
            data = cls.migrate_v1_to_v2(data)
            data = cls.migrate_v2_to_v2_1(data)
        # Migrate v2.0 → v2.1
        elif version == SCHEMA_VERSION_2_0:
            data = cls.migrate_v2_to_v2_1(data)

        return data


class CalibrationManager:
    """
    Manages calibration references for profiler measurements.

    Generates unique calibration identifiers and maintains provenance chain.
    """

    @staticmethod
    def generate_calibration_ref(method: str,
                                 system_info: Optional[Dict[str, Any]] = None,
                                 seed: Optional[str] = None) -> str:
        """
        Generate unique calibration reference.

        Args:
            method: Profiling method used
            system_info: System information dictionary (CPU, OS, etc.)
            seed: Optional seed for deterministic generation

        Returns:
            16-character hex calibration reference
        """
        if seed:
            return hashlib.sha256(seed.encode()).hexdigest()[:16]

        # Build calibration string from method + system info + timestamp
        cal_data = {
            "method": method,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "system": system_info or {}
        }

        cal_string = json.dumps(cal_data, sort_keys=True)
        return hashlib.sha256(cal_string.encode()).hexdigest()[:16]

    @staticmethod
    def validate_calibration_ref(ref: str) -> bool:
        """
        Validate calibration reference format.

        Args:
            ref: Calibration reference string

        Returns:
            True if valid, False otherwise
        """
        if ref == DEFAULT_CALIBRATION_REF:
            return True

        # Check if 16-char hex string
        if len(ref) == 16:
            try:
                int(ref, 16)
                return True
            except ValueError:
                pass

        # Check if valid UUID
        try:
            uuid.UUID(ref)
            return True
        except ValueError:
            pass

        return False


def create_profiler_metadata(method: str,
                             error_pct: float = DEFAULT_ERROR_PCT,
                             calibration_ref: Optional[str] = None,
                             cpu_model: Optional[str] = None,
                             system_id: Optional[str] = None) -> ProfilerMetadata:
    """
    Convenience function to create ProfilerMetadata.

    Args:
        method: Profiling method identifier
        error_pct: Measured relative error percentage
        calibration_ref: Calibration reference (generated if None)
        cpu_model: CPU model identifier
        system_id: System identifier

    Returns:
        ProfilerMetadata instance
    """
    if calibration_ref is None:
        system_info = {"cpu_model": cpu_model, "system_id": system_id} if cpu_model or system_id else None
        calibration_ref = CalibrationManager.generate_calibration_ref(method, system_info)

    return ProfilerMetadata(
        method=method,
        error_pct=error_pct,
        calibration_ref=calibration_ref,
        cpu_model=cpu_model,
        system_id=system_id
    )


def validate_profiler_result(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate profiler result data structure.

    Args:
        data: Profiler result dictionary

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check required measurement fields
    required_fields = ['energy_joules', 'elapsed_seconds', 'avg_power_watts']
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(data[field], (int, float)):
            errors.append(f"Field {field} must be numeric")
        elif data[field] < 0:
            errors.append(f"Field {field} must be non-negative")

    # Check metadata
    if 'metadata' not in data:
        errors.append("Missing 'metadata' field")
    else:
        metadata = data['metadata']

        # Check required metadata fields (v2.1)
        if 'method' not in metadata:
            errors.append("Missing metadata.method")
        elif not isinstance(metadata['method'], str) or len(metadata['method']) == 0:
            errors.append("metadata.method must be non-empty string")

        if 'error_pct' not in metadata:
            errors.append("Missing metadata.error_pct")
        elif not isinstance(metadata['error_pct'], (int, float)):
            errors.append("metadata.error_pct must be numeric")

        if 'calibration_ref' not in metadata:
            errors.append("Missing metadata.calibration_ref")
        elif not isinstance(metadata['calibration_ref'], str):
            errors.append("metadata.calibration_ref must be string")

    return (len(errors) == 0, errors)
