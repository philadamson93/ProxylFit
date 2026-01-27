"""
DICOM folder scanner utility.

Scans a folder of DICOM files and extracts metadata to help identify
PROXYL and T2 images for analysis.
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

try:
    import pydicom
except ImportError:
    pydicom = None


def scan_dicom_folder(folder_path: str, recursive: bool = True) -> List[Dict]:
    """
    Scan a folder for DICOM files and extract metadata.

    Parameters
    ----------
    folder_path : str
        Path to folder containing DICOM files
    recursive : bool
        Whether to scan subdirectories

    Returns
    -------
    List[Dict]
        List of dictionaries with metadata for each unique series
    """
    if pydicom is None:
        raise ImportError("pydicom is required. Install with: pip install pydicom")

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Collect all DICOM files
    if recursive:
        dicom_files = list(folder.rglob("*.dcm")) + list(folder.rglob("*.DCM"))
    else:
        dicom_files = list(folder.glob("*.dcm")) + list(folder.glob("*.DCM"))

    # Also check for files without extension (common in DICOM)
    for f in folder.iterdir() if not recursive else folder.rglob("*"):
        if f.is_file() and f.suffix == "" and not f.name.startswith("."):
            # Try to read as DICOM
            try:
                pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                dicom_files.append(f)
            except Exception:
                pass

    # Group by SeriesInstanceUID
    series_data = defaultdict(lambda: {
        'files': [],
        'study_description': '',
        'series_description': '',
        'series_number': 0,
        'modality': '',
        'rows': 0,
        'cols': 0,
        'num_frames': 0,
        'num_slices': 0,
        'patient_id': '',
        'study_date': '',
        'series_uid': '',
        'is_proxyl': False,
        'is_t2': False,
        'sample_file': ''
    })

    for fpath in dicom_files:
        try:
            ds = pydicom.dcmread(str(fpath), stop_before_pixels=True)

            # Use SeriesInstanceUID as key, fallback to series number + description
            series_uid = str(ds.get('SeriesInstanceUID', ''))
            if not series_uid:
                series_uid = f"{ds.get('SeriesNumber', 0)}_{ds.get('SeriesDescription', 'Unknown')}"

            info = series_data[series_uid]
            info['files'].append(str(fpath))
            info['series_uid'] = series_uid
            info['study_description'] = str(ds.get('StudyDescription', ''))
            info['series_description'] = str(ds.get('SeriesDescription', ''))
            info['series_number'] = int(ds.get('SeriesNumber', 0))
            info['modality'] = str(ds.get('Modality', ''))
            info['rows'] = int(ds.get('Rows', 0))
            info['cols'] = int(ds.get('Columns', 0))
            info['patient_id'] = str(ds.get('PatientID', ''))
            info['study_date'] = str(ds.get('StudyDate', ''))
            info['sample_file'] = str(fpath)

            # Track frames per file
            num_frames = int(ds.get('NumberOfFrames', 1))
            info['num_frames'] = max(info['num_frames'], num_frames)

        except Exception as e:
            # Skip files that can't be read
            continue

    # Convert to list and calculate derived fields
    results = []
    for series_uid, info in series_data.items():
        # Number of slices is number of files (for 2D) or frames (for 4D)
        info['num_slices'] = len(info['files']) if info['num_frames'] <= 1 else info['num_frames']

        # Detect PROXYL and T2 series
        desc_lower = info['series_description'].lower()
        info['is_proxyl'] = 'proxyl' in desc_lower or (
            'flash' in desc_lower and info['num_frames'] > 100
        )
        info['is_t2'] = 't2' in desc_lower and 'turbo' in desc_lower

        # Remove files list for cleaner output (keep sample_file)
        info['num_files'] = len(info['files'])
        del info['files']

        results.append(info)

    # Sort by series number
    results.sort(key=lambda x: x['series_number'])

    return results


def save_scan_to_csv(scan_results: List[Dict], output_path: str) -> str:
    """
    Save scan results to CSV file.

    Parameters
    ----------
    scan_results : List[Dict]
        Results from scan_dicom_folder()
    output_path : str
        Path for output CSV file

    Returns
    -------
    str
        Path to saved CSV file
    """
    if not scan_results:
        raise ValueError("No scan results to save")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define column order
    columns = [
        'series_number',
        'series_description',
        'study_description',
        'modality',
        'rows',
        'cols',
        'num_slices',
        'num_frames',
        'num_files',
        'is_proxyl',
        'is_t2',
        'patient_id',
        'study_date',
        'sample_file'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(scan_results)

    return str(output_path)


def find_proxyl_series(scan_results: List[Dict]) -> List[Dict]:
    """Find series that appear to be PROXYL acquisitions."""
    return [s for s in scan_results if s['is_proxyl']]


def find_t2_series(scan_results: List[Dict]) -> List[Dict]:
    """Find series that appear to be T2 acquisitions."""
    return [s for s in scan_results if s['is_t2']]


def print_scan_summary(scan_results: List[Dict]) -> None:
    """Print a summary of the scan results."""
    print(f"\nFound {len(scan_results)} series:\n")
    print(f"{'Series#':>8} | {'Description':<40} | {'Size':>10} | {'Slices':>6} | {'Type':<10}")
    print("-" * 85)

    for s in scan_results:
        size_str = f"{s['rows']}x{s['cols']}"
        type_str = ""
        if s['is_proxyl']:
            type_str = "PROXYL"
        elif s['is_t2']:
            type_str = "T2"

        print(f"{s['series_number']:>8} | {s['series_description'][:40]:<40} | {size_str:>10} | {s['num_slices']:>6} | {type_str:<10}")

    # Summary
    proxyl = find_proxyl_series(scan_results)
    t2 = find_t2_series(scan_results)
    print(f"\nSummary: {len(proxyl)} PROXYL series, {len(t2)} T2 series")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scan DICOM folder and extract metadata")
    parser.add_argument("folder", help="Path to DICOM folder")
    parser.add_argument("-o", "--output", help="Output CSV path", default=None)
    parser.add_argument("--no-recursive", action="store_true", help="Don't scan subdirectories")

    args = parser.parse_args()

    results = scan_dicom_folder(args.folder, recursive=not args.no_recursive)
    print_scan_summary(results)

    if args.output:
        csv_path = save_scan_to_csv(results, args.output)
        print(f"\nSaved to: {csv_path}")
    else:
        # Default output next to the folder
        folder_name = Path(args.folder).name
        csv_path = save_scan_to_csv(results, f"{folder_name}_dicom_scan.csv")
        print(f"\nSaved to: {csv_path}")
