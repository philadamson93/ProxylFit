Reference: docs/claude_ops.md

# T017: Fix Parameter Map Close Exits Python

**Status**: planned
**Priority**: critical
**Created**: 2026-02-06
**Source**: User feedback item #9

## Description

Closing the parameter map results dialog exits the entire application instead of returning to the main menu.

## Root Cause

`QApplication.quitOnLastWindowClosed` defaults to `True`. When `ParameterMapResultsDialog` is the last visible window and gets closed, Qt quits the process.

## Fix

**File:** `proxyl_analysis/ui/styles.py` lines 132-140 (`init_qt_app`)

Add `app.setQuitOnLastWindowClosed(False)` in `init_qt_app()`.

## Verification

- Open parameter maps, close the dialog → app returns to main menu loop, does not exit
