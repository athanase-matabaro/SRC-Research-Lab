# Warning Elimination - Phase H.5

**Date:** 2025-10-17
**Issue:** 176 tests passed with 9 RuntimeWarnings
**Resolution:** All warnings eliminated step-by-step
**Final Result:** 176 tests, 0 warnings ✅

---

## Problem Analysis

### Warning Details

**Type:** `RuntimeWarning`
**Message:** `RAPL not available. Using constant power model: 35.0W` (or 50.0W)
**Source:** `energy/profiler/energy_profiler.py:82`
**Occurrences:** 9 warnings across 8 test functions

**Affected Tests:**
1. `test_constant_mode_basic`
2. `test_context_manager`
3. `test_multiple_reads`
4. `test_custom_power`
5. `test_rapl_detection`
6. `test_rapl_measurement`
7. `test_measure_energy_basic`
8. `test_measure_energy_with_args`
9. `test_energy_measurement_sanity`

### Root Cause

The `EnergyProfiler` class was using `warnings.warn()` to notify users when RAPL wasn't available and it fell back to constant power mode. While informational, this triggered warnings in test output.

---

## Solution: Step-by-Step Fixes

### Step 1: Add Logging Infrastructure

**File:** `energy/profiler/energy_profiler.py`

```python
import logging

# Configure logger
logger = logging.getLogger(__name__)
```

**Why:** Use Python's logging module instead of warnings for informational messages.

---

### Step 2: Add `quiet` Parameter

**Changes to `EnergyProfiler.__init__()`:**

```python
def __init__(self, constant_power: Optional[float] = None, quiet: bool = False):
    """
    Initialize energy profiler.

    Args:
        constant_power: Override constant power (Watts) for fallback mode.
        quiet: If True, suppress RAPL fallback warnings (default: False).
    """
    self.constant_power = constant_power or self.DEFAULT_POWER_WATTS
    self.quiet = quiet
    # ...
```

**Changes to `_detect_method()`:**

```python
def _detect_method(self) -> None:
    if self._rapl_available():
        self.method = "rapl"
    else:
        self.method = "constant"
        if not self.quiet:
            # Use logger.info instead of warnings.warn
            logger.info(
                f"RAPL not available. Using constant power model: "
                f"{self.constant_power}W"
            )
```

**Changes to `measure_energy()` function:**

```python
def measure_energy(func, *args, quiet: bool = False, **kwargs):
    """
    Args:
        quiet: If True, suppress RAPL fallback warnings.
    """
    with EnergyProfiler(quiet=quiet) as profiler:
        result = func(*args, **kwargs)
    # ...
```

**Why:**
- Tests can suppress expected warnings with `quiet=True`
- Production code still sees informational messages (default `quiet=False`)
- Changed to `logger.info()` because this is informational, not a warning

---

### Step 3: Update Tests One-by-One

#### Test 1-3: Constant Mode Tests
```python
# Before
profiler = EnergyProfiler()

# After
profiler = EnergyProfiler(quiet=True)
```

**Files changed:**
- `test_constant_mode_basic()`
- `test_context_manager()`
- `test_multiple_reads()`

#### Test 4: Custom Power Test
```python
# Before
profiler = EnergyProfiler(constant_power=50.0)

# After
profiler = EnergyProfiler(constant_power=50.0, quiet=True)
```

**File changed:** `test_custom_power()`

#### Test 5-6: RAPL Mode Tests
```python
# Before
profiler = EnergyProfiler()

# After
profiler = EnergyProfiler(quiet=True)
```

**Files changed:**
- `test_rapl_detection()`
- `test_rapl_measurement()`

#### Test 7-8: Measure Energy Function Tests
```python
# Before
result, joules, seconds = measure_energy(func, ...)

# After
result, joules, seconds = measure_energy(func, ..., quiet=True)
```

**Files changed:**
- `test_measure_energy_basic()`
- `test_measure_energy_with_args()`

#### Test 9: Robustness Test
```python
# Before
with EnergyProfiler() as profiler:

# After
with EnergyProfiler(quiet=True) as profiler:
```

**File changed:** `test_energy_measurement_sanity()`

---

## Results

### Before
```
176 passed, 9 warnings in 2.71s
```

**Warnings:**
```
tests/test_energy_profiler.py::TestEnergyProfilerConstantMode::test_constant_mode_basic
  RuntimeWarning: RAPL not available. Using constant power model: 35.0W

(8 more similar warnings...)
```

### After
```
176 passed in 2.70s
```

**No warnings! ✓**

---

## Key Improvements

1. **Clean Test Output**
   - No more warning clutter
   - Easier to spot real issues
   - Professional test reports

2. **Flexible Warning Control**
   - Tests suppress expected warnings
   - Production code still informed
   - Configurable per instance

3. **Better Message Classification**
   - `logger.info()` for informational messages
   - `warnings.warn()` reserved for actual warnings
   - Proper severity levels

4. **No Performance Impact**
   - Runtime: 2.70s (was 2.71s)
   - Minimal overhead from logging

---

## Usage Examples

### In Tests (Suppress Warnings)
```python
# Clean test output
with EnergyProfiler(quiet=True) as profiler:
    # ... test code ...
    pass
```

### In Production (See Messages)
```python
# Users see informational messages
with EnergyProfiler() as profiler:  # quiet=False by default
    compress_data()
# Console: INFO: RAPL not available. Using constant power model: 35.0W
```

### With Custom Logging Configuration
```python
import logging
logging.basicConfig(level=logging.INFO)

# Now logger.info() messages appear
profiler = EnergyProfiler()  # Will log RAPL status
```

---

## Files Modified

1. **energy/profiler/energy_profiler.py** (26 lines)
   - Added `import logging`
   - Added `logger` instance
   - Added `quiet` parameter
   - Changed `warnings.warn()` to `logger.info()`

2. **tests/test_energy_profiler.py** (9 changes)
   - Updated 7 test methods with `quiet=True`

3. **tests/test_robustness_h5.py** (1 change)
   - Updated 1 test method with `quiet=True`

**Total:** 3 files, 36 lines modified

---

## Lessons Learned

### Don't Use `warnings.warn()` for Informational Messages

**Problem:** `warnings.warn()` triggers test warnings
**Solution:** Use `logging.info()` for informational messages
**Benefit:** Tests stay clean, users still informed

### Provide Quiet Mode for Tests

**Pattern:**
```python
def __init__(self, quiet: bool = False):
    if not self.quiet:
        logger.info("Informational message")
```

**Why:**
- Tests control verbosity
- Production sees messages
- Flexible configuration

### Fix Warnings One-by-One

**Approach:**
1. Identify all warning sources
2. Fix infrastructure (logging, quiet param)
3. Update each test individually
4. Verify incrementally

**Result:** Systematic, traceable elimination

---

## Verification

### Run All Tests
```bash
pytest tests/ -q
# Output: 176 passed in 2.70s (no warnings)
```

### Check Specific Test Files
```bash
pytest tests/test_energy_profiler.py -v
# All tests pass, no warnings

pytest tests/test_robustness_h5.py -v
# All tests pass, no warnings
```

### Verify Logging Still Works
```python
import logging
logging.basicConfig(level=logging.INFO)

from energy.profiler import EnergyProfiler
profiler = EnergyProfiler()  # quiet=False
# Output: INFO:energy.profiler.energy_profiler:RAPL not available...
```

---

## Conclusion

**Problem:** 9 RuntimeWarnings cluttered test output
**Solution:** Step-by-step warning elimination
**Result:** 176 tests, 0 warnings, clean output ✅

**Impact:**
- Professional test reports
- Better message classification
- Flexible warning control
- Zero performance impact

**Status:** COMPLETE - All warnings resolved

---

**Prepared by:** Athanase Nshombo (Matabaro)
**Date:** 2025-10-17
**Phase:** H.5 - Warning Elimination
**Tests:** 176/176 passing, 0 warnings
