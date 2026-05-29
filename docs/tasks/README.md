# ProxylFit Tasks

This folder tracks planned features, enhancements, and bug fixes.

## Task Status Legend

| Status | Description |
|--------|-------------|
| `planned` | Documented, not yet started |
| `backlog` | Low priority, not yet started |
| `in-progress` | Currently being implemented |
| `blocked` | Waiting on questions/decisions |
| `deferred` | Postponed, needs further scoping |
| `completed` | Done and merged |

## Current Tasks

| ID | Task | Status | Priority |
|----|------|--------|----------|
| T001 | [T2 to T1 Registration](T001-t2-t1-registration.md) | completed | high |
| T002 | [Averaged Image Generation](T002-averaged-images.md) | completed | medium |
| T003 | [Difference Images & Percent Contrast](T003-difference-images.md) | completed | medium |
| T004 | [Running Average Dynamic Images](T004-running-average.md) | backlog | medium |
| T005 | [Pixel-Level Parameter Maps](T005-pixel-level-params.md) | backlog | high |
| T006 | [Tools Menu](T006-tools-menu.md) | completed | medium |
| T007 | [Registration Progress UI](T007-registration-progress-ui.md) | completed | medium |
| T008 | [UI Bugfixes](T008-ui-bugfixes.md) | completed | high |
| T009 | [UI Refactoring](T009-ui-refactoring.md) | completed | medium |
| T010 | [DICOM Export](T010-dicom-export.md) | completed | medium |
| T011 | [Self-Contained Datasets](T011-self-contained-datasets.md) | completed | high |
| T012 | [DICOM Export Derived Images](T012-dicom-export-derived-images.md) | completed | medium |
| T013 | [Contour Metrics Derived Images](T013-contour-metrics-derived-images.md) | completed | medium |
| T014 | [Parameter Map ROI Options](T014-parameter-map-roi-options.md) | completed | medium |
| T015 | [Application Distribution](T015-application-distribution.md) | planned | low |
| T016 | [Parameter Map Export Crashes](T016-parameter-map-export-crashes.md) | completed | high |
| T017 | [Parameter Map Close Exits Python](T017-parameter-map-close-exits-python.md) | completed | high |
| T018 | [Time Axis Investigation](T018-time-axis-investigation.md) | completed | medium |
| T019 | [Difference Image UX](T019-difference-image-ux.md) | completed | medium |
| T020 | [Fit Results Table Export](T020-fit-results-table-export.md) | completed | medium |
| T021 | [Session Loading UX](T021-session-loading-ux.md) | completed | medium |
| T022 | [Deferred User Feedback](T022-deferred-user-feedback.md) | deferred | medium |
| T023 | [Scanner Registered Output](T023-scanner-registered-output.md) | completed | high |
| T024 | [STEAM VOIs as ROI Source](T024-steam-voi-roi.md) | planned | high |

## Quick Overview

### Planned (next up)
- **T024**: STEAM VOIs as ROI source — auto-load tumor + contralateral voxels from the Bruker raw tree, expose as a one-click ROI option on the kinetics and parameter-map pages

### Backlog (not yet started)
- **T004**: Running average (2-3 point) for noise reduction
- **T005**: Pixel-level kb/kd maps (may extend existing parameter mapping)
- **T015**: Application distribution for non-technical users
- **T022**: Deferred user feedback (parameter map improvements, pixel value readout, decimation)

## Adding New Tasks

1. Create a new file: `TXXX-short-name.md`
2. Use the template below
3. Update this README with the new task

### Task Template

```markdown
# TXXX: Task Title

**Status**: planned | in-progress | blocked | completed
**Priority**: high | medium | low
**Created**: YYYY-MM-DD

## Description
Brief description of the feature/fix.

## Requirements
- Requirement 1
- Requirement 2

## Open Questions
- [ ] Question needing answer

## Implementation Notes
Technical notes, approach, etc.

## Acceptance Criteria
- [ ] Criteria 1
- [ ] Criteria 2
```
