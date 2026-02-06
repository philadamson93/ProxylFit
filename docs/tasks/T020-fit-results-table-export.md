Reference: docs/claude_ops.md

# T020: Fit Results Table Export with Derived Parameters

**Status**: planned
**Priority**: medium
**Created**: 2026-02-06
**Source**: User feedback items #5, #5.1

## Description

Add the ability to export fit results as a table (CSV), and add two derived parameters: %enhancement and %NTE.

### New Derived Parameters

- **%Enhancement** = `(A1 / A0) * 100` — tracer signal amplitude relative to baseline
- **%NTE** = `(A2 / A0) * 100` — non-tracer effect amplitude relative to baseline

### Changes

**File:** `proxyl_analysis/ui/fitting.py`

1. Add %enhancement and %NTE to the parameters panel in the UI (lines 68-79)
2. Add a "Save Results Table" button alongside the existing "Save Plot" button (line 97)
3. The CSV should include all fit parameters, errors, fit quality metrics, and the two derived parameters

**File:** `proxyl_analysis/model.py`

4. Compute %enhancement and %NTE in the `fit_results` dict returned by `fit_proxyl_kinetics()` (around line 343)

## Verification

- Run a kinetic fit → verify %enhancement and %NTE appear in the UI panel
- Click "Save Results Table" → verify CSV contains all parameters including derived ones
