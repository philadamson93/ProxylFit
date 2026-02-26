# Ralph's Feedback — proxylfit_firstlook.pdf

Tracking document for feedback items from Ralph's initial review.

## Summary

- **7 bug fixes**: All resolved and tested
- **3 feature requests**: Open (Items 6-8)
- **1 non-issue**: Operator error (Item 2)

## Bug Fixes (Complete)

### Item 1 — Time Axis: 70s → 33s per Timepoint

**Problem:** `create_time_array()` hardcoded 70s per timepoint. Actual temporal resolution is 33.408s, computable from DICOM metadata: `AcquisitionDuration / (NumberOfFrames / 9)`.

**Fix:**
- Added `extract_temporal_resolution()` to `proxyl_analysis/io.py` — reads AcquisitionDuration and NumberOfFrames from source DICOM
- Updated `create_time_array()` in `proxyl_analysis/run_analysis.py` — added `temporal_resolution_s` parameter, default changed from 70s to 33s
- All call sites extract and pass the DICOM-derived resolution

**Tests:** `TestTimeAxis` in `tests/test_user_feedback_fixes.py` (4 tests)

---

### Item 3 — Grayscale Colormap for Difference Images

**Problem:** Difference images used RdBu_r (red-blue diverging) colormap.

**Fix:** Changed to `cmap='gray'` in `proxyl_analysis/ui/image_tools.py`.

**Tests:** `TestC2_GrayscaleColormap` (2 tests)

---

### Item 4 — Difference Image Wording

**Problem:** Region labels included color names ("Region A (Blue)"), filename order didn't match computation, formula unclear.

**Fix:**
- Simplified labels to "Region A" / "Region B"
- Filename uses B-minus-A order: `diff_t{b}_minus_t{a}`
- Formula states: `Result = mean(Region B) − mean(Region A)`

**Tests:** `TestC1_DifferenceImageLabels` (3 tests)

---

### Item 5 — %Enhancement and %NTE in CSV Export

**Problem:** Fit results export missing derived parameters.

**Fix:** Added %Enhancement (A1/A0 × 100) and %NTE (A2/A0 × 100) to the fit results dialog and CSV export in `proxyl_analysis/ui/fitting.py`.

**Tests:** `TestC3_FitResultsExport` (5 tests)

---

### Item 9 — Closing Window Exits Python

**Problem:** Closing any dialog window killed the entire Python process.

**Fix:** Added `app.setQuitOnLastWindowClosed(False)` in `proxyl_analysis/ui/styles.py`.

**Tests:** `TestA3_QuitOnLastWindowClosed` (2 tests)

---

### Item 10.1 — DICOM Export JSON Serialization Error

**Problem:** numpy int64/float64 values in parameter map metadata caused `json.dumps()` to crash.

**Fix:** Added `_convert_numpy()` helper in `proxyl_analysis/io.py` that converts numpy types to native Python before serialization.

**Tests:** `TestA2_NumpyJsonSerialization` (3 tests)

---

### Item 10.2 — PNG Export show_roi_cb AttributeError

**Problem:** Code referenced `show_roi_cb` but the widget was named `roi_checkbox`.

**Fix:** Updated all references to `roi_checkbox` in `proxyl_analysis/ui/parameter_map_options.py`.

**Tests:** `TestA1_RoiCheckboxAttribute` (2 tests)

---

## Non-Issue

### Item 2 — Difficulty Loading Registered DICOMs

Operator error — user was selecting the wrong folder. No code change needed. Session loading already includes helpful error messages with folder hints (see `TestD1_SessionLoadingFeedback`).

---

## Feature Requests (Open)

### Item 6 — Parameter Map Improvements

Requested improvements to parameter map visualization. Details TBD.

### Item 7 — Read Pixel Values from Parameter Maps

Ability to hover/click on parameter maps and read individual pixel values. Details TBD.

### Item 8 — Image Decimation (32×32 / 16×16)

Support for downsampled parameter mapping at coarser resolutions. Details TBD.
