# ProxylFit Testing Plan

## Overview

This document outlines the testing strategy for the planned features (T002, T003, T005, T006). Testing is divided into:

1. **Automated tests** - Run via pytest, can be part of CI
2. **Manual testing checkpoints** - UI/UX validation at key milestones
3. **Regression tests** - Ensure existing functionality isn't broken

## Test Data Requirements

### Minimum Test Dataset
- One T1 DICOM file (multi-frame, time series)
- One T2 DICOM file (single volume)
- Expected dimensions: ~128×128×9×26 (x, y, z, t)

### Test Data Location
```
tests/
├── data/
│   ├── sample_t1.dcm      # Small test T1 (or synthetic)
│   ├── sample_t2.dcm      # Small test T2 (or synthetic)
│   └── expected/          # Expected outputs for regression
│       ├── registered_4d.npz
│       └── kinetic_results.json
```

**Note**: If real DICOM files are too large for repo, create synthetic test data or use a small subset.

---

## Automated Tests

### Unit Tests (pytest)

#### T002: Averaged Images
```python
# tests/test_averaging.py

def test_compute_averaged_image_basic():
    """Average over known range produces expected result."""
    # Create synthetic 4D data with known values
    image_4d = np.ones((10, 10, 3, 20))
    image_4d[:, :, :, 5:10] = 2.0  # Frames 5-9 have value 2

    result = compute_averaged_image(image_4d, start_idx=5, end_idx=10)

    assert result.shape == (10, 10, 3)
    assert np.allclose(result, 2.0)

def test_compute_averaged_image_single_frame():
    """Single frame average returns that frame."""
    image_4d = np.random.rand(10, 10, 3, 20)
    result = compute_averaged_image(image_4d, start_idx=5, end_idx=6)

    assert np.allclose(result, image_4d[:, :, :, 5])

def test_compute_averaged_image_invalid_range():
    """Invalid range raises appropriate error."""
    image_4d = np.ones((10, 10, 3, 20))

    with pytest.raises(ValueError):
        compute_averaged_image(image_4d, start_idx=10, end_idx=5)  # start > end

def test_compute_averaged_image_out_of_bounds():
    """Out of bounds range is handled gracefully."""
    image_4d = np.ones((10, 10, 3, 20))

    # Should either raise error or clip to valid range
    with pytest.raises(IndexError):
        compute_averaged_image(image_4d, start_idx=0, end_idx=30)
```

#### T003: Difference Images
```python
# tests/test_difference.py

def test_compute_difference_image_basic():
    """Difference of two regions produces expected result."""
    image_4d = np.ones((10, 10, 3, 20))
    image_4d[:, :, :, 0:5] = 1.0    # Region A: value 1
    image_4d[:, :, :, 10:15] = 3.0  # Region B: value 3

    result = compute_difference_image(
        image_4d,
        region_a=(0, 5),
        region_b=(10, 15)
    )

    assert result.shape == (10, 10, 3)
    assert np.allclose(result, 2.0)  # 3 - 1 = 2

def test_compute_difference_image_negative():
    """Negative differences are preserved (B < A)."""
    image_4d = np.ones((10, 10, 3, 20))
    image_4d[:, :, :, 0:5] = 5.0    # Region A: value 5
    image_4d[:, :, :, 10:15] = 2.0  # Region B: value 2

    result = compute_difference_image(
        image_4d,
        region_a=(0, 5),
        region_b=(10, 15)
    )

    assert np.allclose(result, -3.0)  # 2 - 5 = -3

def test_compute_difference_image_same_region():
    """Same region for A and B produces zeros."""
    image_4d = np.random.rand(10, 10, 3, 20)

    result = compute_difference_image(
        image_4d,
        region_a=(5, 10),
        region_b=(5, 10)
    )

    assert np.allclose(result, 0.0)
```

#### T005: Pixel-Level Maps
```python
# tests/test_pixel_maps.py

def test_pixel_level_maps_shape():
    """Output maps have correct shape."""
    image_4d = np.random.rand(10, 10, 3, 20) * 100 + 50
    time = np.arange(20) * 1.167  # minutes

    maps = fit_pixel_level_maps(image_4d, time, injection_idx=2)

    assert maps['kb'].shape == (10, 10, 3)
    assert maps['kd'].shape == (10, 10, 3)
    assert maps['r_squared'].shape == (10, 10, 3)

def test_pixel_level_maps_nan_handling():
    """Failed fits produce NaN, not errors."""
    # Create data that will fail to fit (constant signal)
    image_4d = np.ones((5, 5, 2, 20)) * 100
    time = np.arange(20) * 1.167

    maps = fit_pixel_level_maps(image_4d, time, injection_idx=2)

    # Should complete without error
    # Failed fits should be NaN
    assert not np.all(np.isfinite(maps['kb']))  # Some NaNs expected
```

#### T006: Menu State Management
```python
# tests/test_menu_state.py

def test_reset_state_clears_all():
    """Reset state clears T2, ROI, and signal."""
    # This would test the _reset_state method
    pass  # Requires mock or actual dialog instance

def test_load_previous_session():
    """Loading previous session restores registered data."""
    # Create temp directory with saved registration
    # Call load function
    # Verify data loaded correctly
    pass
```

### Integration Tests

```python
# tests/test_integration.py

def test_full_averaging_workflow():
    """End-to-end: load data → select region → save average."""
    # Load test DICOM
    image_4d, spacing = load_dicom_series("tests/data/sample_t1.dcm")

    # Register
    registered_4d, metrics = register_timeseries(
        image_4d, spacing,
        show_quality_window=False
    )

    # Compute average (frames 5-15)
    avg_image = compute_averaged_image(registered_4d, 5, 15)

    # Verify output
    assert avg_image.shape == registered_4d.shape[:3]
    assert np.all(np.isfinite(avg_image))

def test_cli_batch_mode():
    """CLI batch mode completes without UI prompts."""
    import subprocess

    result = subprocess.run([
        "python", "proxyl_analysis/run_analysis.py",
        "--dicom", "tests/data/sample_t1.dcm",
        "--batch",
        "--roi-mode", "rectangle",
        "--z", "4",
        "--skip-registration"
    ], capture_output=True, timeout=60)

    assert result.returncode == 0
```

### Regression Tests

```python
# tests/test_regression.py

def test_kinetic_fitting_unchanged():
    """Kinetic fitting produces same results as baseline."""
    # Load known test data
    # Run fitting
    # Compare to expected results within tolerance
    pass

def test_registration_quality_unchanged():
    """Registration produces similar quality metrics."""
    pass
```

---

## Manual Testing Checkpoints

### Checkpoint 1: After T006 Phase 1 (Basic Menu)

**What to test:**
- [ ] Menu appears after registration completes
- [ ] "Load New T1 DICOM" opens file dialog
- [ ] "Load Previous Session" opens folder dialog and loads data
- [ ] Loading new experiment resets state (T2 status, etc.)
- [ ] "Load T2 Volume" button works, shows registration quality
- [ ] ROI Source defaults to T2 when T2 is loaded
- [ ] ROI Source falls back to T1 when T2 not loaded
- [ ] ROI Method radio buttons work (rectangle/contour/segment)
- [ ] Z-slice spinbox respects bounds
- [ ] "Start ROI Analysis" launches existing workflow
- [ ] Parameter Maps section launches existing sliding window workflow
- [ ] Export buttons work (registered data, report, time series)
- [ ] Exit button closes application
- [ ] App can launch without --dicom argument

**Test scenarios:**
1. Fresh launch (no args) → load T1 from menu → complete ROI analysis
2. Launch with --dicom → verify menu appears after registration
3. Load T1 → Load T2 → verify T2 becomes default ROI source
4. Load experiment A → do partial work → load experiment B → verify state reset
5. Load previous session from saved registration folder

### Checkpoint 2: After T002 (Averaged Images)

**What to test:**
- [ ] "Averaged Image" button enabled only after ROI selection
- [ ] Time curve displays correctly with ROI signal
- [ ] Click-drag selects a region on the time curve
- [ ] Selected region is visually highlighted
- [ ] Region selection info updates (frame range, count)
- [ ] Preview shows averaged image for selected region
- [ ] "Save Average" creates NPZ file with correct data
- [ ] Selecting different regions updates preview
- [ ] Cancel returns to menu without saving
- [ ] Edge cases: single frame, full range

**Test scenarios:**
1. Complete ROI Analysis → Image Tools → Averaged Image → select early frames → save
2. Select region at very start of scan (edge case)
3. Select region at very end of scan (edge case)
4. Select single frame
5. Verify saved NPZ contains expected data structure

### Checkpoint 3: After T003 (Difference Images)

**What to test:**
- [ ] Mode toggle: "Single Average" vs "Two Regions - Difference"
- [ ] In difference mode, first drag creates Region A (blue)
- [ ] Second drag creates Region B (red)
- [ ] Preview shows difference image with diverging colormap
- [ ] Colormap centered at zero
- [ ] Region info shows both regions and "B - A" description
- [ ] "Clear Regions" resets both selections
- [ ] Can re-select either region independently
- [ ] "Save Difference" creates NPZ with both region info
- [ ] Overlapping regions handled gracefully

**Test scenarios:**
1. Select pre-injection (A) and post-injection (B) → verify contrast effect
2. Select early baseline (A) and end-of-scan (B) → verify proxyl effect
3. Select B before A (reversed order) → verify still works
4. Select overlapping regions → verify behavior
5. Switch between single/difference modes → verify state handled

### Checkpoint 4: After T005 (Pixel-Level Maps)

**What to test:**
- [ ] "Pixel-Level" radio button in Parameter Maps section
- [ ] Progress bar shows voxel count and ETA
- [ ] Can cancel with button or Ctrl+C
- [ ] Partial results saved if cancelled
- [ ] All parameter maps generated (kb, kd, knt, A0, A1, r_squared)
- [ ] NaN values for failed fits (not crashes)
- [ ] Visualization displays all maps
- [ ] Maps saved to output directory
- [ ] Within-ROI scope limits computation to ROI voxels

**Test scenarios:**
1. Run on small ROI (quick test)
2. Run on whole image (verify progress/cancel works)
3. Cancel mid-way → verify graceful handling
4. Compare sliding window vs pixel-level on same data

---

## Test Execution Schedule

| Phase | Automated Tests | Manual Checkpoint |
|-------|-----------------|-------------------|
| Before starting | Set up test infrastructure, create synthetic test data | — |
| After T006 Phase 1 | Menu state tests | Checkpoint 1 |
| After T002 | Averaging unit tests, integration test | Checkpoint 2 |
| After T003 | Difference unit tests | Checkpoint 3 |
| After T005 | Pixel map tests | Checkpoint 4 |
| Final | Full regression suite | End-to-end walkthrough |

---

## Test Infrastructure Setup

### Prerequisites
```bash
pip install pytest pytest-cov pytest-timeout
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=proxyl_analysis --cov-report=html

# Run specific test file
pytest tests/test_averaging.py

# Run tests matching pattern
pytest tests/ -k "averaging"

# Skip slow tests (pixel-level fitting)
pytest tests/ -m "not slow"
```

### Test Markers
```python
# In conftest.py
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "ui: marks tests requiring UI")
```

---

## Known Testing Limitations

1. **UI Tests**: Qt dialogs are difficult to automate. Manual testing required for:
   - Visual appearance
   - Mouse interactions (click-drag)
   - Keyboard shortcuts

2. **Real DICOM Data**: Tests may need real data for full validation. Consider:
   - Synthetic data for unit tests
   - Small real dataset for integration tests (not committed to repo)

3. **Platform Differences**: Qt rendering may differ across macOS/Windows/Linux. Manual testing on target platform recommended.

---

## Future: CI Integration

Once tests are stable, can add GitHub Actions workflow:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -e ".[test]"
      - run: pytest tests/ -m "not slow and not ui"
```
