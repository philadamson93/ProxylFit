"""
Diagnostic: report Z-spacing for every DICOM series in a folder, including
enhanced multi-frame DICOMs (where slice positions live inside
PerFrameFunctionalGroupsSequence rather than as top-level tags).

For each series it prints:
  - SeriesDescription (and a heuristic T1 / T2 classification)
  - NumberOfFrames and where the spacing tags were found (top-level,
    SharedFunctional, or per-frame)
  - SliceThickness, SpacingBetweenSlices
  - What `_extract_robust_spacing` currently returns
  - The TRUE z step computed from ImagePositionPatient[2] across slices —
    this does NOT depend on tags, so it shows the geometry of the volume
    even when the DICOM is light on metadata.

Usage:
    uv run python scripts/check_spacing.py /path/to/dicom_folder
    uv run python scripts/check_spacing.py /path/to/file.dcm
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pydicom

from proxyl_analysis.io import _extract_robust_spacing


def _attr(ds, name, default="—"):
    val = getattr(ds, name, None)
    return default if val is None else val


def _classify(description: str) -> str:
    """Loose T1 / T2 classification — broader than dicom_scanner's heuristics."""
    desc = (description or "").lower()
    if "t2" in desc:
        return "T2"
    if "t1" in desc or "proxyl" in desc or "dyn" in desc:
        return "T1 / dynamic"
    return "(unknown)"


def _gather_dicoms(folder: Path):
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), force=True, stop_before_pixels=True)
        except Exception:
            continue
        if not hasattr(ds, "SOPInstanceUID"):
            continue
        yield path, ds


def _group_by_series(folder: Path):
    groups = defaultdict(list)
    for path, ds in _gather_dicoms(folder):
        sid = getattr(ds, "SeriesInstanceUID", None) or "(no SeriesInstanceUID)"
        groups[sid].append((path, ds))
    return groups


def _ipp_z_positions_singleframe(ds_list):
    """Z positions across a series of single-frame DICOMs."""
    zs = []
    for ds in ds_list:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            zs.append(float(ipp[2]))
    return zs


def _ipp_z_positions_multiframe(ds):
    """Z positions across the frames of an enhanced multi-frame DICOM."""
    zs = []
    pfs = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if pfs is None:
        return zs
    for frame in pfs:
        ipp = None
        # Standard location for enhanced multi-frame DICOMs
        pps = getattr(frame, "PlanePositionSequence", None)
        if pps and len(pps) > 0:
            ipp = getattr(pps[0], "ImagePositionPatient", None)
        if ipp is None:
            ipp = getattr(frame, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            zs.append(float(ipp[2]))
    return zs


def _summarize_z_steps(zs):
    """Return (unique_count, full_step, summary_text) for a sequence of z-positions."""
    if len(zs) < 2:
        return 0, None, "  -> (need >= 2 z-positions for analysis)"

    # Many multi-frame DICOMs store time outside the slice axis, so the same
    # z repeats. Get the unique z values in acquisition order.
    seen = []
    seen_set = set()
    for z in zs:
        # Round to 4 decimals to dedupe near-identical floats
        key = round(z, 4)
        if key not in seen_set:
            seen_set.add(key)
            seen.append(z)
    seen.sort()

    unique_count = len(seen)
    if unique_count < 2:
        return unique_count, None, (
            f"  -> All {len(zs)} frames share one z-position. "
            "Probably 2D/single-slice acquisition."
        )

    diffs = [seen[i + 1] - seen[i] for i in range(unique_count - 1)]
    step = sum(diffs) / len(diffs)
    span = seen[-1] - seen[0]

    lines = [
        f"  -> Z geometry from ImagePositionPatient:",
        f"       frames seen:        {len(zs)}",
        f"       unique z-positions: {unique_count}",
        f"       avg Δz step:        {step:.4f} mm",
        f"       total span:         {span:.4f} mm  (z={seen[0]:.4f} … {seen[-1]:.4f})",
    ]
    return unique_count, step, "\n".join(lines)


def _report_series(series_uid, files):
    # Sort files by InstanceNumber so [0] is the first slice
    files.sort(key=lambda item: getattr(item[1], "InstanceNumber", 0) or 0)
    first_path, first_ds = files[0]

    desc = _attr(first_ds, "SeriesDescription", default="(no description)")
    kind = _classify(str(desc))
    nframes = getattr(first_ds, "NumberOfFrames", None)
    is_multiframe = nframes is not None and int(nframes) > 1

    print(f"  Series: {desc}    [{kind}]")
    print(f"    UID:                  {series_uid}")
    print(f"    Files in series:      {len(files)}")
    print(f"    NumberOfFrames:       {nframes if nframes is not None else '—'}"
          f"    {'(multi-frame)' if is_multiframe else ''}")

    # Show top-level tags
    print(f"    Top-level PixelSpacing:         {_attr(first_ds, 'PixelSpacing')}")
    print(f"    Top-level SliceThickness:       {_attr(first_ds, 'SliceThickness')}")
    print(f"    Top-level SpacingBetweenSlices: {_attr(first_ds, 'SpacingBetweenSlices')}")

    # Show shared functional group tags if multi-frame
    if is_multiframe:
        sfgs = getattr(first_ds, "SharedFunctionalGroupsSequence", None)
        if sfgs and len(sfgs) > 0:
            pms = getattr(sfgs[0], "PixelMeasuresSequence", None)
            if pms and len(pms) > 0:
                pm = pms[0]
                print(f"    Shared PixelSpacing:         {_attr(pm, 'PixelSpacing')}")
                print(f"    Shared SliceThickness:       {_attr(pm, 'SliceThickness')}")
                print(f"    Shared SpacingBetweenSlices: {_attr(pm, 'SpacingBetweenSlices')}")

    # What does our reader return today?
    try:
        spacing = _extract_robust_spacing(str(first_path))
        print(f"    -> _extract_robust_spacing returns (x, y, z) = {spacing}")
    except Exception as e:
        print(f"    -> _extract_robust_spacing FAILED: {e}")

    # Compute the TRUE z geometry from ImagePositionPatient — independent of tags.
    if is_multiframe:
        zs = _ipp_z_positions_multiframe(first_ds)
    else:
        zs = _ipp_z_positions_singleframe([d for _, d in files])

    _, step, summary = _summarize_z_steps(zs)
    print(summary)

    # Compare the ground-truth step against what the reader reported
    if step is not None:
        try:
            reported_z = float(spacing[2])
            if abs(abs(step) - reported_z) > 1e-3:
                print(
                    f"    !! Reader reports z={reported_z} mm but actual z step "
                    f"is {abs(step):.4f} mm — mismatch."
                )
            else:
                print(f"    OK: reader's z spacing matches IPP-derived step.")
        except Exception:
            pass

    print()


def report_path(path_str: str):
    p = Path(path_str)
    if not p.exists():
        print(f"\nDoes not exist: {p}")
        return
    print(f"\n=== {p} ===")
    if p.is_file():
        # Treat as a single (possibly multi-frame) DICOM
        try:
            ds = pydicom.dcmread(str(p), force=True, stop_before_pixels=True)
        except Exception as e:
            print(f"  ERROR reading DICOM: {e}")
            return
        sid = getattr(ds, "SeriesInstanceUID", "(no SeriesInstanceUID)")
        _report_series(sid, [(p, ds)])
        return

    groups = _group_by_series(p)
    if not groups:
        print("  No DICOM files found.")
        return
    print(f"  Found {sum(len(v) for v in groups.values())} DICOM files in "
          f"{len(groups)} series.\n")
    for series_uid, files in groups.items():
        _report_series(series_uid, files)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for path in sys.argv[1:]:
        report_path(path)


if __name__ == "__main__":
    main()
