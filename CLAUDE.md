# CLAUDE.md

## Before You Do Anything

**Read `docs/claude_ops.md` for operating standards.** Key points:

1. **Always start in Plan mode** before writing code
2. **Re-enter plan mode** when direction changes or issues arise
3. **Ask clarifying questions** for functional requirements; use judgement for implementation details

---

## Project Overview

Analysis software for time-resolved PROXYL MRI data. Performs image registration, ROI-based kinetic modeling, and parameter mapping using a Qt-based (PySide6) interface.

## Documentation

| Document | Purpose |
|----------|---------|
| **`docs/claude_ops.md`** | **Operating standards - READ FIRST** |
| `docs/claude_learnings.md` | Mistakes and lessons learned |
| `docs/tutorial.md` | User-facing tutorial with screenshots |
| `docs/installation.md` | Setup and troubleshooting guide |
| `docs/quickstart.md` | 5-minute quick start |
| `docs/changelog.md` | Version history |
| `docs/ui.md` | Qt UI interface guide |
| `docs/testing-plan.md` | Testing documentation |

## Key Entry Points

- **CLI / Main**: `proxyl_analysis/run_analysis.py`
- **Kinetic Model**: `proxyl_analysis/model.py`
- **DICOM I/O**: `proxyl_analysis/io.py`
- **Registration**: `proxyl_analysis/registration.py`
- **ROI Selection**: `proxyl_analysis/roi_selection.py`
- **Parameter Mapping**: `proxyl_analysis/parameter_mapping.py`
- **UI Components**: `proxyl_analysis/ui/`

## Running

```bash
uv run python -m proxyl_analysis
```
