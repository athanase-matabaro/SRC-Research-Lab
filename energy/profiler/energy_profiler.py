"""
Energy Profiler - CPU Energy Measurement Module

Provides offline CPU energy measurement using Intel RAPL (Running Average Power Limit)
or fallback to constant power model. No network access, fully offline operation.

Usage:
    with EnergyProfiler() as profiler:
        # Run compression task
        compress_data()

    joules, seconds = profiler.read()
    caq_e = compute_caqe(ratio, seconds, joules)

Author: Athanase Nshombo (Matabaro)
Date: 2025-10-16
Phase: H.5 - Energy-Aware Compression
"""

import os
import sys
import time
import warnings
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

# Add runtime to path for profiler_metadata import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "runtime"))

try:
    from profiler_metadata import (
        ProfilerMetadata, ProfilerResult, create_profiler_metadata,
        METHOD_RAPL, METHOD_CONSTANT, CalibrationManager,
        CURRENT_SCHEMA_VERSION
    )
    METADATA_AVAILABLE = True
except ImportError:
    METADATA_AVAILABLE = False
    ProfilerMetadata = None
    ProfilerResult = None

# Security: No network imports allowed
# RAPL interface: /sys/class/powercap/intel-rapl/

# Configure logger
logger = logging.getLogger(__name__)

class EnergyProfiler:
    """
    Measures CPU energy consumption using Intel RAPL or fallback models.

    Hierarchy of energy measurement approaches:
    1. Intel RAPL (preferred): /sys/class/powercap/intel-rapl/
    2. Constant power model (fallback): 35W TDP default
    3. User-specified constant (override)

    Attributes:
        method (str): Energy measurement method used
        start_energy (float): Starting energy reading in microjoules
        start_time (float): Starting timestamp
        end_energy (float): Ending energy reading in microjoules
        end_time (float): Ending timestamp
    """

    # Default constant power draw for fallback (Watts)
    DEFAULT_POWER_WATTS = 35.0

    # RAPL sysfs paths
    RAPL_BASE = Path("/sys/class/powercap/intel-rapl")

    def __init__(self, constant_power: Optional[float] = None, quiet: bool = False,
                 error_pct: float = 0.0, calibration_ref: Optional[str] = None):
        """
        Initialize energy profiler.

        Args:
            constant_power: Override constant power (Watts) for fallback mode.
                          If None, uses DEFAULT_POWER_WATTS.
            quiet: If True, suppress RAPL fallback warnings (default: False).
            error_pct: Measured relative error percentage for this profiling method.
            calibration_ref: Calibration reference identifier (generated if None).
        """
        self.constant_power = constant_power or self.DEFAULT_POWER_WATTS
        self.quiet = quiet
        self.method = None
        self.start_energy = None
        self.start_time = None
        self.end_energy = None
        self.end_time = None
        self.error_pct = error_pct
        self.calibration_ref = calibration_ref
        self._profiler_metadata = None

        # Detect available measurement method
        self._detect_method()

    def _detect_method(self) -> None:
        """
        Detect which energy measurement method is available.

        Priority:
        1. RAPL (if /sys/class/powercap/intel-rapl exists and readable)
        2. Constant power fallback
        """
        if self._rapl_available():
            self.method = "rapl"
        else:
            self.method = "constant"
            if not self.quiet:
                # Use logger.info instead of warnings.warn
                # This is informational, not a warning
                logger.info(
                    f"RAPL not available. Using constant power model: "
                    f"{self.constant_power}W"
                )

    def _rapl_available(self) -> bool:
        """
        Check if Intel RAPL interface is available and readable.

        Returns:
            True if RAPL is accessible, False otherwise.
        """
        if not self.RAPL_BASE.exists():
            return False

        try:
            # Try to find package-0 energy file
            package_path = self.RAPL_BASE / "intel-rapl:0"
            if not package_path.exists():
                return False

            energy_file = package_path / "energy_uj"
            if not energy_file.exists():
                return False

            # Try to read it
            with open(energy_file, 'r') as f:
                _ = int(f.read().strip())

            return True

        except (PermissionError, ValueError, OSError):
            return False

    def _read_rapl_energy(self) -> float:
        """
        Read current energy from RAPL interface.

        Returns:
            Energy in microjoules.

        Raises:
            RuntimeError: If RAPL is not available or read fails.
        """
        if self.method != "rapl":
            raise RuntimeError("RAPL not available")

        try:
            package_path = self.RAPL_BASE / "intel-rapl:0"
            energy_file = package_path / "energy_uj"

            with open(energy_file, 'r') as f:
                energy_uj = int(f.read().strip())

            return float(energy_uj)

        except (OSError, ValueError) as e:
            raise RuntimeError(f"Failed to read RAPL energy: {e}")

    def __enter__(self):
        """
        Start energy measurement (context manager entry).

        Returns:
            self for context manager.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stop energy measurement (context manager exit).

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception value if raised.
            exc_tb: Exception traceback if raised.

        Returns:
            False to propagate exceptions.
        """
        self.stop()
        return False

    def start(self) -> None:
        """
        Start energy measurement.

        Records start time and start energy reading.
        """
        self.start_time = time.time()

        if self.method == "rapl":
            try:
                self.start_energy = self._read_rapl_energy()
            except RuntimeError:
                # Fall back to constant if RAPL fails during measurement
                warnings.warn(
                    "RAPL read failed, falling back to constant power model",
                    RuntimeWarning
                )
                self.method = "constant"
                self.start_energy = None
        else:
            self.start_energy = None  # Not used for constant method

    def stop(self) -> None:
        """
        Stop energy measurement.

        Records end time and end energy reading.
        """
        self.end_time = time.time()

        if self.method == "rapl":
            try:
                self.end_energy = self._read_rapl_energy()
            except RuntimeError:
                # Fall back to constant if RAPL fails during measurement
                warnings.warn(
                    "RAPL read failed, falling back to constant power model",
                    RuntimeWarning
                )
                self.method = "constant"
                self.end_energy = None
        else:
            self.end_energy = None  # Not used for constant method

    def read(self) -> Tuple[float, float]:
        """
        Read energy consumption and elapsed time.

        Returns:
            Tuple of (joules, seconds).

        Raises:
            RuntimeError: If start() hasn't been called or measurement failed.
        """
        if self.start_time is None or self.end_time is None:
            raise RuntimeError(
                "Must call start() and stop() before read(). "
                "Or use context manager: with EnergyProfiler() as ep: ..."
            )

        elapsed_seconds = self.end_time - self.start_time

        if self.method == "rapl":
            if self.start_energy is None or self.end_energy is None:
                # Fallback occurred during measurement
                joules = self.constant_power * elapsed_seconds
            else:
                # Convert microjoules to joules
                energy_uj = self.end_energy - self.start_energy

                # Handle RAPL counter overflow (wraps at max_energy_range_uj)
                if energy_uj < 0:
                    try:
                        package_path = self.RAPL_BASE / "intel-rapl:0"
                        max_range_file = package_path / "max_energy_range_uj"
                        with open(max_range_file, 'r') as f:
                            max_range = int(f.read().strip())
                        energy_uj += max_range
                    except (OSError, ValueError):
                        # If we can't read max range, use constant fallback
                        warnings.warn(
                            "RAPL counter overflow detected but max range unknown. "
                            "Using constant power fallback.",
                            RuntimeWarning
                        )
                        joules = self.constant_power * elapsed_seconds
                        return (joules, elapsed_seconds)

                joules = energy_uj / 1_000_000.0  # microjoules to joules
        else:
            # Constant power model: Energy = Power × Time
            joules = self.constant_power * elapsed_seconds

        return (joules, elapsed_seconds)

    def _create_metadata(self) -> Optional[ProfilerMetadata]:
        """
        Create ProfilerMetadata for current measurement.

        Returns:
            ProfilerMetadata instance or None if metadata module unavailable.
        """
        if not METADATA_AVAILABLE:
            return None

        # Map internal method names to metadata constants
        method_name = METHOD_RAPL if self.method == "rapl" else METHOD_CONSTANT

        # Get CPU info for system identification
        cpu_info = self.get_cpu_info()
        cpu_model = cpu_info.get("model", "unknown")

        # Generate or use provided calibration ref
        if self.calibration_ref is None:
            system_info = {"cpu_model": cpu_model, "method": method_name}
            calibration_ref = CalibrationManager.generate_calibration_ref(method_name, system_info)
        else:
            calibration_ref = self.calibration_ref

        return create_profiler_metadata(
            method=method_name,
            error_pct=self.error_pct,
            calibration_ref=calibration_ref,
            cpu_model=cpu_model
        )

    def get_info(self) -> Dict[str, any]:
        """
        Get profiler information and current readings with v2.1 metadata.

        Returns:
            Dictionary with profiler status, measurements, and metadata.
        """
        # Map internal method to schema method
        method_name = METHOD_RAPL if self.method == "rapl" else METHOD_CONSTANT

        info = {
            "method": method_name,
            "constant_power_watts": self.constant_power,
            "rapl_available": self._rapl_available(),
        }

        if self.start_time is not None and self.end_time is not None:
            joules, seconds = self.read()
            avg_power = joules / seconds if seconds > 0 else 0

            info.update({
                "energy_joules": joules,
                "elapsed_seconds": seconds,
                "avg_power_watts": avg_power,
            })

            # Add v2.1 metadata fields
            if self._profiler_metadata is None:
                self._profiler_metadata = self._create_metadata()

            if self._profiler_metadata:
                info["metadata"] = self._profiler_metadata.to_dict()
            else:
                # Fallback: include minimal metadata without module
                info["metadata"] = {
                    "method": method_name,
                    "error_pct": self.error_pct,
                    "calibration_ref": self.calibration_ref or "none",
                    "schema_version": "2.1"
                }

        return info

    def get_result(self) -> Optional[ProfilerResult]:
        """
        Get complete ProfilerResult with measurements and metadata.

        Returns:
            ProfilerResult instance or None if metadata module unavailable.
        """
        if not METADATA_AVAILABLE or self.start_time is None or self.end_time is None:
            return None

        joules, seconds = self.read()
        avg_power = joules / seconds if seconds > 0 else 0

        if self._profiler_metadata is None:
            self._profiler_metadata = self._create_metadata()

        return ProfilerResult(
            energy_joules=joules,
            elapsed_seconds=seconds,
            avg_power_watts=avg_power,
            metadata=self._profiler_metadata
        )

    @staticmethod
    def get_cpu_info() -> Dict[str, str]:
        """
        Get CPU information for reproducibility.

        Returns:
            Dictionary with CPU model, cores, frequency, etc.
        """
        cpu_info = {
            "model": "unknown",
            "cores": "unknown",
            "threads": "unknown",
            "base_freq_mhz": "unknown",
        }

        try:
            # Read /proc/cpuinfo for CPU details
            with open("/proc/cpuinfo", 'r') as f:
                lines = f.readlines()

            for line in lines:
                if line.startswith("model name"):
                    cpu_info["model"] = line.split(":")[1].strip()
                    break

            # Count cores (physical cores)
            core_ids = set()
            for line in lines:
                if line.startswith("core id"):
                    core_ids.add(line.split(":")[1].strip())
            cpu_info["cores"] = str(len(core_ids)) if core_ids else "unknown"

            # Count threads (logical processors)
            processor_count = sum(1 for line in lines if line.startswith("processor"))
            cpu_info["threads"] = str(processor_count)

            # Get base frequency
            for line in lines:
                if line.startswith("cpu MHz"):
                    cpu_info["base_freq_mhz"] = line.split(":")[1].strip()
                    break

        except (OSError, IndexError):
            pass  # Return defaults if unable to read

        return cpu_info


# Convenience function for simple measurements
def measure_energy(func, *args, quiet: bool = False, **kwargs) -> Tuple[any, float, float]:
    """
    Measure energy consumption of a function call.

    Args:
        func: Function to measure.
        *args: Positional arguments for func.
        quiet: If True, suppress RAPL fallback warnings (default: False).
        **kwargs: Keyword arguments for func.

    Returns:
        Tuple of (result, joules, seconds).

    Example:
        result, joules, seconds = measure_energy(compress_data, data, output_path)
        caq_e = ratio / (seconds * (joules/seconds) + 1)
    """
    with EnergyProfiler(quiet=quiet) as profiler:
        result = func(*args, **kwargs)

    joules, seconds = profiler.read()
    return (result, joules, seconds)


# CLI interface for testing and sample generation
if __name__ == "__main__":
    import argparse
    import json
    import random

    parser = argparse.ArgumentParser(description="Energy Profiler Test & Sample Generation")
    parser.add_argument("--sample", type=int, default=10,
                       help="Number of sample measurements to generate")
    parser.add_argument("--save", type=str, default=None,
                       help="Save results to JSON file")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress warnings")
    parser.add_argument("--error-pct", type=float, default=0.0,
                       help="Simulated error percentage for testing")

    args = parser.parse_args()

    print(f"Generating {args.sample} profiler test samples...")
    print(f"Error PCT: {args.error_pct}%")

    results = []

    for i in range(args.sample):
        # Simulate work with random sleep
        work_duration = random.uniform(0.01, 0.1)

        profiler = EnergyProfiler(quiet=args.quiet, error_pct=args.error_pct)
        profiler.start()

        # Simulate work
        time.sleep(work_duration)

        profiler.stop()

        # Get info with metadata
        info = profiler.get_info()
        results.append(info)

        print(f"  Sample {i+1}/{args.sample}: {info['energy_joules']:.4f}J, "
              f"{info['elapsed_seconds']:.3f}s, {info['avg_power_watts']:.2f}W")

    # Print summary
    print(f"\nGenerated {len(results)} samples")
    print(f"All samples include metadata: method, error_pct, calibration_ref")

    # Verify metadata
    metadata_count = sum(1 for r in results if 'metadata' in r)
    print(f"Metadata present in {metadata_count}/{len(results)} samples")

    # Save to file if requested
    if args.save:
        output_data = {
            "profiler_version": "2.1",
            "num_samples": len(results),
            "samples": results
        }

        with open(args.save, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nSaved {len(results)} samples to {args.save}")
        print(f"File size: {os.path.getsize(args.save)} bytes")

    print("\nTest complete.")
