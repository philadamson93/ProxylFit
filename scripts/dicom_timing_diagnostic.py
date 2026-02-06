#!/usr/bin/env python3
"""
Diagnostic script to extract timing information from DICOM files.

Usage:
    python scripts/dicom_timing_diagnostic.py /path/to/dicom_file.dcm

Reads timing-related DICOM tags to determine the actual frame interval
for multi-frame PROXYL MRI datasets. This helps verify or correct the
hardcoded 70-second assumption in create_time_array().
"""

import sys
from pathlib import Path

import pydicom


# Tags of interest for timing
TIMING_TAGS = [
    ('AcquisitionTime', 0x00080032),
    ('ContentTime', 0x00080033),
    ('TriggerTime', 0x00181060),
    ('RepetitionTime', 0x00180080),
    ('TemporalPositionIdentifier', 0x00200100),
    ('NumberOfTemporalPositions', 0x00200105),
    ('TemporalResolution', 0x00200110),
    ('AcquisitionDuration', 0x00180073),
    ('FrameReferenceTime', 0x00541300),
    ('ActualFrameDuration', 0x00181242),
    ('NumberOfFrames', 0x00280008),
]


def parse_dicom_time(time_str: str) -> float:
    """Convert DICOM time string (HHMMSS.ffffff) to seconds."""
    time_str = time_str.strip()
    hours = int(time_str[0:2])
    minutes = int(time_str[2:4])
    seconds = float(time_str[4:])
    return hours * 3600 + minutes * 60 + seconds


def analyze_timing(filepath: str):
    """Analyze timing information from a DICOM file or directory."""
    path = Path(filepath)

    if path.is_dir():
        dcm_files = sorted(path.glob('*.dcm'))
        if not dcm_files:
            dcm_files = sorted(path.glob('*'))
            dcm_files = [f for f in dcm_files if f.is_file()]
        if not dcm_files:
            print(f"No DICOM files found in {path}")
            return
    else:
        dcm_files = [path]

    print(f"Analyzing {len(dcm_files)} file(s)...\n")

    # Read first file for general info
    ds = pydicom.dcmread(str(dcm_files[0]))

    print("=== General Info ===")
    for attr in ['PatientName', 'StudyDescription', 'SeriesDescription',
                 'Modality', 'Manufacturer', 'InstitutionName']:
        val = getattr(ds, attr, 'N/A')
        print(f"  {attr}: {val}")

    print(f"\n=== Timing Tags (from first file) ===")
    for name, tag in TIMING_TAGS:
        val = getattr(ds, name, None)
        if val is not None:
            print(f"  {name} ({hex(tag)}): {val}")
        else:
            print(f"  {name} ({hex(tag)}): NOT PRESENT")

    # Check for multi-frame
    n_frames = getattr(ds, 'NumberOfFrames', None)
    if n_frames:
        print(f"\n=== Multi-frame DICOM ({n_frames} frames) ===")
        # Check for per-frame functional groups
        if hasattr(ds, 'PerFrameFunctionalGroupsSequence'):
            print("  Has PerFrameFunctionalGroupsSequence")
            for i, fg in enumerate(ds.PerFrameFunctionalGroupsSequence[:5]):
                print(f"  Frame {i}: {[str(item.keyword) for item in fg]}")
        if hasattr(ds, 'SharedFunctionalGroupsSequence'):
            print("  Has SharedFunctionalGroupsSequence")

    # If directory with multiple files, compute time intervals
    if len(dcm_files) > 1:
        print(f"\n=== Time Intervals (from {min(len(dcm_files), 20)} files) ===")
        times = []
        for f in dcm_files[:20]:
            try:
                d = pydicom.dcmread(str(f), stop_before_pixels=True)
                acq_time = getattr(d, 'AcquisitionTime', None)
                content_time = getattr(d, 'ContentTime', None)
                temporal_pos = getattr(d, 'TemporalPositionIdentifier', None)

                t = acq_time or content_time
                if t:
                    t_sec = parse_dicom_time(str(t))
                    times.append((f.name, t_sec, temporal_pos))
                    print(f"  {f.name}: time={t} ({t_sec:.1f}s), temporal_pos={temporal_pos}")
            except Exception as e:
                print(f"  {f.name}: ERROR - {e}")

        if len(times) >= 2:
            # Sort by time and compute intervals
            times.sort(key=lambda x: x[1])
            intervals = [times[i+1][1] - times[i][1] for i in range(len(times)-1)]
            if intervals:
                print(f"\n  Time intervals (seconds): {[f'{dt:.1f}' for dt in intervals]}")
                print(f"  Mean interval: {sum(intervals)/len(intervals):.1f} seconds")
                print(f"  Min interval:  {min(intervals):.1f} seconds")
                print(f"  Max interval:  {max(intervals):.1f} seconds")

    # Check for private tags that might contain timing info
    print(f"\n=== Private Tags (potential timing info) ===")
    found_private = False
    for elem in ds:
        if elem.tag.is_private and elem.VR not in ('OB', 'OW', 'UN'):
            val_str = str(elem.value)[:80]
            if any(kw in val_str.lower() for kw in ['time', 'interval', 'duration', 'temporal', 'dynamic']):
                print(f"  {elem.tag} ({elem.VR}): {val_str}")
                found_private = True
    if not found_private:
        print("  No timing-related private tags found")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/dicom_timing_diagnostic.py <dicom_file_or_directory>")
        sys.exit(1)

    analyze_timing(sys.argv[1])
