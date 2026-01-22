# ProxylFit Tasks

This folder tracks planned features, enhancements, and bug fixes.

## Task Status Legend

| Status | Description |
|--------|-------------|
| `planned` | Documented, not yet started |
| `in-progress` | Currently being implemented |
| `blocked` | Waiting on questions/decisions |
| `completed` | Done and merged |

## Current Tasks

| ID | Task | Status | Priority |
|----|------|--------|----------|
| T001 | [T2 to T1 Registration](T001-t2-t1-registration.md) | planned | high |
| T002 | [Averaged Image Generation](T002-averaged-images.md) | planned | medium |
| T003 | [Difference Images & Percent Contrast](T003-difference-images.md) | planned | medium |
| T004 | [Running Average Dynamic Images](T004-running-average.md) | planned | medium |
| T005 | [Pixel-Level Parameter Maps](T005-pixel-level-params.md) | planned | high |

## Quick Overview

### High Priority
- **T001**: T2-T1 registration for better tumor segmentation (RANO)
- **T005**: Pixel-level kb/kd maps (may extend existing parameter mapping)

### Medium Priority
- **T002**: Averaged images (full series, pre-proxyl, post-proxyl, contrast-enhanced)
- **T003**: Difference images and percent contrast/effect calculations
- **T004**: Running average (2-3 point) for noise reduction

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
