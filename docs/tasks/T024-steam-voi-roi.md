# T024: STEAM VOIs as ROI Source

**Status**: planned
**Priority**: high
**Created**: 2026-05-14

## Description

Load the prescribed STEAM spectroscopy voxels (tumor + contralateral) from the
Bruker raw tree and expose them as an ROI source on both the per-ROI kinetics
page and the parameter-map fitting page. The user gets a one-click, fully
geometric ROI that matches exactly what the scanner prescribed — no manual
drawing, no error from hand-traced contours.

## Rationale

- Every study in this protocol acquires two STEAM voxels per subject: one in
  tumor, one contralateral.
- Those voxels are the canonical region the MR spectroscopy data was sampled
  from — analyzing T1 kinetics in the exact same region is the natural
  multimodal pairing.
- The contralateral voxel doubles as a built-in control for tumor kinetic
  parameters.
- Geometry comes for free from the Bruker `method` file
  (`PVM_VoxArrPosition` / `PVM_VoxArrSize` / `PVM_VoxArrGradOrient`); no
  segmentation needed.

## Answered Questions

| Question | Answer |
|----------|--------|
| Where does the STEAM geometry come from? | Bruker raw tree, `<study>/<expno>/method`, keys `PVM_VoxArrPosition` / `PVM_VoxArrSize` / `PVM_VoxArrGradOrient` |
| How many VOIs per subject? | Always 2 (tumor + contralateral); list-typed schema keeps it extensible |
| Where does the sidecar JSON live? | `output/{dataset}/steam_voi.json` (per-subject; DICOM folder stays multi-subject) |
| Coordinate auto-fix? | Try identity + axis-flip combinations, pick best overlap with T1 brain mask, prompt to confirm before saving the chosen `scanner_to_patient` 4×4 |
| Paired tumor-vs-contralateral plots? | Stretch (Phase 6); single-VOI flow ships first |
| Multi-subject DICOM/Bruker folders? | T1 combo selection auto-follows to the matching Bruker subject; auto-match by PatientID → StudyDate/Time → ordinal |
| Subject-link behavior? | Auto-follow on T1 combo change (matches 99% of workflows) |

## Open Questions

- [ ] Confirm `scanner_to_patient` defaults on this rig — once identified for
      one dataset, it should be the same across all subjects from the same
      scanner+coil configuration. May want a per-scanner preference, not
      per-dataset.
- [ ] Decide on filename for tumor/contralateral label-swap dialog state
      (in-memory only vs. persisted in JSON; persisting is simpler).
- [ ] **Phase 2 design need** — the reference dataset
      (`20250923_..._B2_D16_...`) contains **12 STEAM acquisitions** at
      **6 distinct prescribed positions** (each position acquired twice,
      likely for averaging or pre/post pairs across multiple imaging
      timepoints), not just the two voxels we initially modeled. The
      loader UI must cluster acquisitions by position (within tolerance,
      e.g. <0.1 mm), let the user pick which pair to use as tumor +
      contralateral, and optionally group by timepoint based on
      sequential ExpNo blocks.

## Data Model

### Sidecar JSON

Location: `output/{dataset}/steam_voi.json`. One JSON per analyzed subject.

```json
{
  "schema_version": 1,
  "subject_id": "B16",
  "scanner_to_patient": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
  "voi": [
    {
      "label": "tumor",
      "method": "Bruker:STEAM",
      "frame": "bruker_scanner",
      "position_mm": [-2.5019681, -7.4787888, 0.8929790],
      "size_mm":     [3.0, 3.0, 3.0],
      "orientation": [
        [ 0.66529950,  0.28308262, -0.69082618],
        [-0.34423973,  0.93740690,  0.05260525],
        [ 0.66247686,  0.20281157,  0.72110463]
      ],
      "source": {
        "kind": "bruker_method",
        "path": "../20250923_072207_Recht_..._B2_D16_.../34/method",
        "keys_used": ["PVM_VoxArrPosition", "PVM_VoxArrSize", "PVM_VoxArrGradOrient"]
      }
    },
    {
      "label": "contralateral",
      "method": "Bruker:STEAM",
      "...": "..."
    }
  ]
}
```

Schema notes:

- `frame` is the source coordinate frame of `position_mm`/`orientation`
  (`bruker_scanner` or `patient_lps`). Determines whether
  `scanner_to_patient` is applied during mask generation.
- `voi` is a list, not a scalar; ordering is irrelevant since each entry is
  labeled.
- `source` is bookkeeping; everything needed for analysis is captured in the
  numeric fields. Regeneration from `source.path` is supported if it exists.

### Coordinate math

For each T1 voxel center `p_T1` in patient (LPS) coordinates, compute
`q = orientation @ (scanner_to_patient @ p_T1 − position)` and test
`all(|q_i| ≤ size_mm[i] / 2)`. Vectorized over the full 3D grid; sub-second
for typical T1 volumes.

## New module: `proxyl_analysis/steam_voi.py`

| Function | Purpose |
|---|---|
| `parse_bruker_method(path) -> VoiDict` | Read `PVM_VoxArrPosition/Size/GradOrient` + `##$Method` from one method file |
| `scan_bruker_study(subject_root) -> list[VoiDict]` | Walk numbered subdirs of one subject folder, return all VOIs whose `Method == Bruker:STEAM` |
| `scan_bruker_root(root) -> dict[subject_id, list[VoiDict]]` | Walk every subject folder under a multi-subject root, return per-subject VOI lists |
| `load_voi_json(path) -> StudyVOIs` | Read sidecar JSON |
| `save_voi_json(path, study_vois)` | Write sidecar JSON |
| `voi_to_mask(voi, t1_geometry, scanner_to_patient) -> np.ndarray` | Rasterize one VOI to a 3D boolean mask in T1 voxel space |
| `voi_to_polygon(voi, z, t1_geometry, scanner_to_patient) -> np.ndarray` | Project one VOI onto one z-slice as a polygon for 2D overlay rendering |
| `auto_detect_transform(vois, t1_volume, t1_geometry) -> 4x4` | Try identity + axis-flip combinations, return best by overlap with T1 brain mask |

## UI Changes

### `ui/main_menu.py` — DicomScanResultsDialog

Add a third selector row beside `t1_combo` / `t2_combo`:

```
T1 (PROXYL): [ Subject_B16 — PROXYL_T1   (series 12, 30 frames) ▼ ]
T2:          [ Subject_B16 — T2_TurboRARE (series 8, 25 slices) ▼ ]
STEAM:       [ Subject_B16 — 2 voxels (ExpNo 24, 32)            ▼ ]  [ Labels… ]
```

Plus a Bruker-tree path field at the dialog header level:

```
DICOM folder:   /path/to/dicoms              [Change…]
Bruker tree:    /path/to/20250923_..._B2_D16 [Change…]   (optional)
```

Bruker tree path is remembered per-DICOM-folder in a small settings file so
the user only picks it once per study series.

Auto-follow: when `t1_combo` changes, the STEAM combo jumps to the matching
subject. Match cascade:

| Priority | Match key | DICOM side | Bruker side |
|---|---|---|---|
| 1 | PatientID | tag `(0010,0020)` | `<subject_root>/subject` file |
| 2 | StudyDate + StudyTime (±2 hrs) | `(0008,0020)/(0008,0030)` | folder timestamp prefix |
| 3 | Ordinal position | series_number sort | folder name sort |
| 4 | Manual override | user selects from combo |

Visual hint: green check (priority 1), yellow "verify" (2/3), red "manual"
(4).

`Labels…` button opens a small modal showing both voxels overlaid on the T1
mid-slice with a radio selection to swap which is tumor vs. contralateral.
Default assignment uses laterality (configurable per-rig).

### `ui/main_menu.py` — MainMenuDialog ROI Analysis section

Extend the existing ROI Source row:

```
ROI Source: ○ T2   ○ T1   ◉ STEAM VOI
             └─ when STEAM selected:
                 Voxel: [ Tumor ▼ ]   [ Manage STEAM VOIs… ]
```

When `roi_source == "steam"`, hide the ROI Method row (Rectangle / Contour /
Segment) — there's no drawing step.

A new status line under the ROI status: `STEAM VOI: loaded (2 voxels)` or
`STEAM VOI: not loaded`.

### `ui/fitting.py` — KineticFitDialog

No code changes for Phase 1–5; the dialog already accepts `roi_mask`. The
upstream menu call computes the mask via `voi_to_mask(...)` and passes it
in.

Phase 6 (stretch): when invoked with both VOIs, render a side-by-side
"Tumor vs. Contralateral" panel using the same fit code run twice. Add a
ratio panel for kinetic parameters.

### `ui/parameter_map_options.py`

Two integration points:

1. **Fitting-region ROI** — extend the "ROI Processing" group:

   ```
   ◉ Process whole image
   ○ Process within ROI only
     ROI: ○ Reuse existing  ○ Draw new  ○ Use STEAM VOI [Tumor ▼]
   ```

   Bounds pixel-level fitting to the prescribed voxel. Much faster than
   whole-image fitting.

2. **Measurement ROI** — add a sibling button to the existing
   measurement-ROI flow: "Measure within STEAM VOI…". Picks tumor or
   contralateral and computes stats from the 3D mask. Modify
   `_save_pm_metric_bundle` to accept a label string (e.g.
   `STEAM_Tumor`) so output filenames are self-identifying instead of just
   integer-indexed.

### `ui/steam_voi.py` (new) — STEAM VOI dialog

Standalone modal for the "Manage STEAM VOIs…" / "Load STEAM VOI…" actions.
Reuses ProxylFit styling. Sections:

- Source picker (Bruker study folder vs. single method file)
- Found-VOIs table (Use / ExpNo / position / label)
- Overlay preview on T1 mid-slice
- Scanner→patient transform editor (auto-detect / manual 4×4)
- Save & Use button

## Persistence

| Artifact | Location | Lifetime |
|---|---|---|
| `steam_voi.json` | `output/{dataset}/` | Permanent; per-analyzed-subject |
| Cached masks | `output/{dataset}/steam_voi/{label}_mask.npz` | Regenerated if missing |
| Bruker-tree path mapping | `output/.steam_voi_bruker_paths.json` | Per-DICOM-folder; remembers picker choice |
| `scanner_to_patient` 4×4 | inside `steam_voi.json` | Per-dataset; once tuned, never re-touched |

JSON is the source of truth; everything else is a cache.

## Implementation Phases

| Phase | Scope | Done when |
|---|---|---|
| 1. Core math + parser | `steam_voi.py` + unit tests | `voi_to_mask` round-trips through a synthetic T1 geometry; Bruker method parser matches the values extracted by hand from `34/method` |
| 2. Loader dialog | `ui/steam_voi.py` + folder scan | Pointing at `20250923_…` auto-finds both STEAM voxels, writes JSON, shows preview |
| 3. Menu integration | `DicomScanResultsDialog` STEAM combo + auto-follow; `MainMenuDialog` ROI source option | "STEAM VOI → Tumor" runs kinetic fit on tumor VOI mask |
| 4. Parameter-map fitting ROI | `parameter_map_options.py` | Fit-within-VOI runs and produces correctly-bounded parameter maps |
| 5. Measurement-ROI integration | `parameter_map_options.py` + `fitting.py` summary bundle | "Measure within STEAM VOI" produces summary bundle with `STEAM_Tumor` / `STEAM_Contralateral` labels |
| 6. Stretch: paired tumor vs. contralateral | `KineticFitDialog` | Side-by-side panel + ratio metrics |

## Acceptance Criteria

- [ ] Loading a Bruker study root populates `output/{dataset}/steam_voi.json`
      with both VOIs, no manual editing required
- [ ] Both VOIs render correctly as overlay boxes on the T1 (visually
      verified)
- [ ] Kinetic fit on STEAM tumor VOI matches kinetic fit on a hand-drawn ROI
      of the same region within tolerance
- [ ] Parameter-map fitting bounded by STEAM VOI runs in a small fraction of
      the time of whole-image fitting
- [ ] Session reopens without re-loading the Bruker tree (JSON sidecar is the
      only requirement)
- [ ] Multi-subject DICOM folders: T1 combo selection auto-follows STEAM
      combo to the matching subject by PatientID / StudyDate / ordinal
- [ ] `scanner_to_patient` auto-detect succeeds on the reference dataset
      (`20250923_..._B2_D16_...`); manual override path documented

## Files Affected

```
proxyl_analysis/steam_voi.py            # NEW: parser + math + persistence
proxyl_analysis/ui/steam_voi.py         # NEW: loader dialog
proxyl_analysis/ui/main_menu.py         # STEAM combo in DicomScanResultsDialog;
                                        # STEAM source option in MainMenuDialog
proxyl_analysis/ui/parameter_map_options.py  # STEAM as fitting ROI + measurement ROI
proxyl_analysis/ui/fitting.py           # Phase 6: paired tumor/contralateral panel
proxyl_analysis/io.py                   # Discover steam_voi.json on session load
tests/test_steam_voi.py                 # NEW: unit tests for parser, math, JSON
docs/changelog.md                       # Version entry
```
