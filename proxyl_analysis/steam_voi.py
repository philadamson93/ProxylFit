"""
STEAM voxel-of-interest (VOI) handling for ProxylFit.

Loads prescribed STEAM spectroscopy voxels from a Bruker ParaVision raw tree
(the ``method`` file inside each numbered ExpNo folder) or from a sidecar
JSON, and rasterizes them to 3D boolean masks aligned with a T1 voxel grid
so they can be used as ROIs in the existing kinetics and parameter-map
pipelines.

This is the Phase 1 deliverable of T024 — pure math + parsing + I/O. UI
integration lives in ``proxyl_analysis/ui/steam_voi.py`` and the existing
main-menu and parameter-map dialogs.

Key Bruker keys consumed
------------------------
- ``##$Method`` — must equal ``<Bruker:STEAM>`` for a scan to be loaded as a
  STEAM VOI.
- ``##$PVM_VoxArrPosition`` — VOI center, in scanner coordinates (mm).
- ``##$PVM_VoxArrSize`` — VOI edge lengths along (Read, Phase, Slice) (mm).
- ``##$PVM_VoxArrGradOrient`` — 3×3 rotation. Rows are the logical
  (Read, Phase, Slice) axes expressed in the scanner frame.

Coordinate frames
-----------------
- **bruker_scanner** — what ``PVM_VoxArrPosition`` is expressed in. Bore
  fixed; axes depend on the magnet hardware orientation.
- **patient_lps** — what DICOM ``ImagePositionPatient`` is expressed in.
  Standard DICOM convention.

The two frames usually differ by axis sign flips for rodent scanners. The
``scanner_to_patient`` 4×4 (default identity) captures the mapping for one
rig; once tuned it stays in ``steam_voi.json`` per dataset.

A voxel at patient-frame point ``p`` is inside the VOI iff::

    p_scanner = inv(scanner_to_patient) @ p          (in homogeneous coords)
    q_logical = orientation @ (p_scanner - position)
    all(|q_logical_i| <= size_mm[i] / 2)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class T1Geometry:
    """Voxel-grid geometry of a T1 volume, in DICOM patient (LPS) coords.

    The patient-coordinate location of voxel ``(i, j, k)`` (zero-based) is::

        p = origin + i * spacing[0] * direction[:, 0]
                   + j * spacing[1] * direction[:, 1]
                   + k * spacing[2] * direction[:, 2]

    Parameters
    ----------
    shape : (int, int, int)
        ``(nx, ny, nz)`` — number of voxels along each grid axis.
    origin : np.ndarray, shape (3,)
        Patient-coordinate position of voxel ``(0, 0, 0)``. Equivalent to
        DICOM ``ImagePositionPatient`` of the first slice (in mm).
    spacing : np.ndarray, shape (3,)
        Voxel edge lengths ``(sx, sy, sz)`` in mm.
    direction : np.ndarray, shape (3, 3)
        Columns are unit vectors along the voxel ``i``, ``j``, ``k`` axes
        expressed in patient coordinates. Equivalent to a 3×3 rotation
        built from DICOM ``ImageOrientationPatient`` (first two columns)
        plus their cross product (third column).
    """

    shape: Tuple[int, int, int]
    origin: np.ndarray
    spacing: np.ndarray
    direction: np.ndarray


@dataclass
class SteamVOI:
    """One prescribed STEAM voxel.

    Stored as a plain dataclass (not frozen) so labels can be swapped via
    UI without rebuilding the whole object.

    Parameters
    ----------
    label : str
        User-facing name. Conventional values: ``"tumor"``,
        ``"contralateral"``. Free-form otherwise.
    method : str
        Acquisition method as parsed from the source. For Bruker STEAM
        this is ``"Bruker:STEAM"``.
    frame : str
        Coordinate frame of ``position_mm`` and ``orientation``. One of
        ``"bruker_scanner"`` or ``"patient_lps"``.
    position_mm : np.ndarray, shape (3,)
        VOI center in the frame named by ``frame``.
    size_mm : np.ndarray, shape (3,)
        Edge lengths along the VOI's logical (Read, Phase, Slice) axes.
    orientation : np.ndarray, shape (3, 3)
        Rotation whose rows are the logical (Read, Phase, Slice) unit
        vectors expressed in ``frame``. Multiplying this matrix against
        a frame-coordinate offset yields the logical-frame coordinate.
    source : dict
        Bookkeeping describing how this VOI was obtained
        (``kind="bruker_method"`` / ``"bruker_dicom"`` / ``"json"``,
        ``path``, ``keys_used``, etc.).
    """

    label: str
    method: str
    frame: str
    position_mm: np.ndarray
    size_mm: np.ndarray
    orientation: np.ndarray
    source: Dict[str, Any] = field(default_factory=dict)
    # Per-VOI scanner→patient 4×4 affine. When ``None`` the
    # ``StudyVOIs.scanner_to_patient`` default is used. Storing per-VOI
    # accommodates Bruker quirks where different prescribed voxels in
    # the same session end up needing different sign-flips even after
    # gradient-order normalization.
    transform: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict matching the schema."""
        d = {
            "label": self.label,
            "method": self.method,
            "frame": self.frame,
            "position_mm": [float(x) for x in self.position_mm],
            "size_mm": [float(x) for x in self.size_mm],
            "orientation": [[float(x) for x in row] for row in self.orientation],
            "source": dict(self.source),
        }
        if self.transform is not None:
            d["transform"] = [
                [float(x) for x in row] for row in self.transform
            ]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SteamVOI":
        transform = None
        if d.get("transform") is not None:
            transform = np.asarray(d["transform"], dtype=float).reshape(4, 4)
        return cls(
            label=str(d["label"]),
            method=str(d["method"]),
            frame=str(d["frame"]),
            position_mm=np.asarray(d["position_mm"], dtype=float).reshape(3),
            size_mm=np.asarray(d["size_mm"], dtype=float).reshape(3),
            orientation=np.asarray(d["orientation"], dtype=float).reshape(3, 3),
            source=dict(d.get("source", {})),
            transform=transform,
        )

    def effective_transform(self, default: np.ndarray) -> np.ndarray:
        """Return ``transform`` if set, else the supplied ``default``."""
        return self.transform if self.transform is not None else default


@dataclass
class StudyVOIs:
    """All VOIs (typically tumor + contralateral) for one analyzed subject.

    Parameters
    ----------
    schema_version : int
        Schema version. Increment when the on-disk JSON format breaks
        compatibility.
    subject_id : str or None
        Free-form subject identifier (Bruker ``SUBJECT_id``, DICOM
        ``PatientID``, or user-assigned). May be ``None`` when not known.
    scanner_to_patient : np.ndarray, shape (4, 4)
        Affine taking a point from the Bruker scanner frame to the DICOM
        patient (LPS) frame. Identity by default; per-rig overrides live
        in this field once calibrated.
    voi : list[SteamVOI]
        Two VOIs in the standard protocol (tumor + contralateral); the
        list is open-ended for extension to multi-voxel grids.
    """

    schema_version: int = 1
    subject_id: Optional[str] = None
    scanner_to_patient: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=float)
    )
    voi: List[SteamVOI] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "subject_id": self.subject_id,
            "scanner_to_patient": [
                [float(x) for x in row] for row in self.scanner_to_patient
            ],
            "voi": [v.to_dict() for v in self.voi],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StudyVOIs":
        m = np.asarray(
            d.get("scanner_to_patient", np.eye(4).tolist()), dtype=float
        ).reshape(4, 4)
        return cls(
            schema_version=int(d.get("schema_version", 1)),
            subject_id=d.get("subject_id"),
            scanner_to_patient=m,
            voi=[SteamVOI.from_dict(v) for v in d.get("voi", [])],
        )

    def by_label(self, label: str) -> Optional[SteamVOI]:
        """Return the VOI with the given label, or ``None`` if absent."""
        for v in self.voi:
            if v.label == label:
                return v
        return None


# ---------------------------------------------------------------------------
# T1 geometry construction from DICOM
# ---------------------------------------------------------------------------

def _ifg_get(ds, attr: str):
    """Pull a geometry attribute from an Enhanced-MR DICOM.

    Multi-frame DICOMs (SOP class Enhanced MR Image Storage) move
    geometry tags into ``SharedFunctionalGroupsSequence`` and
    ``PerFrameFunctionalGroupsSequence``. This helper walks the
    standard sequence path for the three tags we care about
    (``PixelSpacing``, ``ImageOrientationPatient``,
    ``ImagePositionPatient``) and returns the value if found, else
    ``None``. Falls back to the top-level attribute when neither
    sequence path matches.
    """
    # Top-level (Standard DICOM)
    val = getattr(ds, attr, None)
    if val is not None:
        return val

    def _from_seq(seq):
        if seq is None or len(seq) == 0:
            return None
        item = seq[0]
        for sub_name in (
            "PixelMeasuresSequence",
            "PlaneOrientationSequence",
            "PlanePositionSequence",
        ):
            sub_seq = getattr(item, sub_name, None)
            if sub_seq is not None and len(sub_seq) > 0:
                v = getattr(sub_seq[0], attr, None)
                if v is not None:
                    return v
        return None

    val = _from_seq(getattr(ds, "SharedFunctionalGroupsSequence", None))
    if val is not None:
        return val
    val = _from_seq(getattr(ds, "PerFrameFunctionalGroupsSequence", None))
    return val


def build_t1_geometry_from_dicom(dicom_path) -> T1Geometry:
    """Build a ``T1Geometry`` from a registered T1 DICOM series.

    Reads ``ImagePositionPatient``, ``ImageOrientationPatient``,
    ``PixelSpacing``, and per-slice ``SliceLocation`` from the DICOM
    headers to construct the patient-frame voxel grid.

    Handles three input layouts:

    1. **Per-slice directory** — one single-frame DICOM per slice
       (the standard ProxylFit ``registered/dicoms/`` layout).
    2. **Standard multi-frame DICOM** — one file with
       ``NumberOfFrames > 1`` and geometry tags at the top level.
    3. **Enhanced-MR multi-frame DICOM** — one file with
       ``SharedFunctionalGroupsSequence`` /
       ``PerFrameFunctionalGroupsSequence`` carrying the geometry
       (typical Bruker DICOM export of a multi-time-point series).

    Parameters
    ----------
    dicom_path : Path or str
        Either a single DICOM file (multi-frame) or a directory
        containing one DICOM per slice.

    Returns
    -------
    T1Geometry
        Voxel-grid geometry with ``shape``, ``origin``, ``spacing``,
        and ``direction`` in DICOM patient (LPS) coordinates.

    Raises
    ------
    ImportError
        If pydicom is not installed.
    FileNotFoundError
        If ``dicom_path`` does not exist.
    ValueError
        If the DICOMs don't have the expected geometry tags.
    """
    try:
        import pydicom
    except ImportError as e:  # pragma: no cover
        raise ImportError("pydicom is required for build_t1_geometry_from_dicom") from e

    p = Path(dicom_path)
    if not p.exists():
        raise FileNotFoundError(p)

    # Collect DICOM files. Sort by SliceLocation when available.
    if p.is_dir():
        candidates = sorted([f for f in p.iterdir() if f.is_file()])
        if not candidates:
            raise ValueError(f"no DICOM files in {p}")
    else:
        candidates = [p]

    # Read the first DICOM to determine geometry attributes.
    ds0 = pydicom.dcmread(str(candidates[0]), force=True)

    px_spacing = _ifg_get(ds0, "PixelSpacing")
    if px_spacing is None:
        raise ValueError(
            f"DICOM at {dicom_path} has no PixelSpacing (checked top-level, "
            "SharedFunctionalGroupsSequence, and PerFrameFunctionalGroupsSequence)"
        )
    # DICOM PixelSpacing = [row_spacing, col_spacing]. ProxylFit's numpy
    # convention is (x, y, z) = (col_index, row_index, slice_index), so:
    #   spacing[0] (numpy X) = col spacing = PixelSpacing[1]
    #   spacing[1] (numpy Y) = row spacing = PixelSpacing[0]
    spacing_xy = [float(x) for x in px_spacing]

    iop = _ifg_get(ds0, "ImageOrientationPatient")
    if iop is None:
        raise ValueError(
            f"DICOM at {dicom_path} has no ImageOrientationPatient"
        )
    # DICOM ImageOrientationPatient = [first-row dir cosines, first-col dir cosines]
    # "first row" = top row of pixel data = direction of increasing COLUMN index
    #            = ProxylFit's numpy X axis
    # "first col" = left col of pixel data = direction of increasing ROW index
    #            = ProxylFit's numpy Y axis
    orientation = [float(x) for x in iop]
    x_cosine = np.array(orientation[:3], dtype=float)
    y_cosine = np.array(orientation[3:], dtype=float)
    slice_cosine = np.cross(x_cosine, y_cosine)

    nrows = int(ds0.Rows)
    ncols = int(ds0.Columns)

    # Three cases.
    if p.is_file() and hasattr(ds0, "NumberOfFrames") and int(ds0.NumberOfFrames) > 1:
        # Multi-frame DICOM. ProxylFit time series may have many frames
        # per slice; collapse repeats to unique z positions by taking
        # frames whose ImagePositionPatient differs.
        n_frames = int(ds0.NumberOfFrames)
        per_frame = getattr(ds0, "PerFrameFunctionalGroupsSequence", None)
        positions: List[np.ndarray] = []
        if per_frame is not None:
            seen = set()
            for i in range(n_frames):
                item = per_frame[i]
                pps = getattr(item, "PlanePositionSequence", None)
                if pps is None or len(pps) == 0:
                    continue
                ipp = pps[0].ImagePositionPatient
                key = tuple(round(float(x), 4) for x in ipp)
                if key not in seen:
                    seen.add(key)
                    positions.append(np.array(ipp, dtype=float))
        if positions:
            # Sort by projection onto slice normal so origin is the first slice.
            keys = [float(np.dot(pos, slice_cosine)) for pos in positions]
            order = np.argsort(keys)
            positions = [positions[i] for i in order]
            origin = positions[0]
            n_slices = len(positions)
            if n_slices > 1:
                z_spacing = float(np.linalg.norm(positions[1] - positions[0]))
            else:
                z_spacing = float(
                    getattr(ds0, "SpacingBetweenSlices",
                            getattr(ds0, "SliceThickness", 1.0))
                )
        else:
            # No per-frame positions — fall back to top-level.
            top_ipp = _ifg_get(ds0, "ImagePositionPatient")
            if top_ipp is None:
                raise ValueError(
                    f"multi-frame DICOM at {dicom_path} has no ImagePositionPatient"
                )
            origin = np.array(top_ipp, dtype=float)
            z_spacing = float(
                getattr(ds0, "SpacingBetweenSlices",
                        getattr(ds0, "SliceThickness", 1.0))
            )
            n_slices = n_frames
    else:
        # Per-slice DICOM directory or single-frame file.
        slice_infos = []
        for f in candidates:
            d = pydicom.dcmread(str(f), force=True, stop_before_pixels=True)
            ipp = _ifg_get(d, "ImagePositionPatient")
            if ipp is None:
                continue
            ipp_arr = np.array(ipp, dtype=float)
            loc = float(getattr(d, "SliceLocation", np.dot(ipp_arr, slice_cosine)))
            slice_infos.append((loc, ipp_arr, f))
        if not slice_infos:
            raise ValueError(
                f"no DICOMs with ImagePositionPatient found under {dicom_path}"
            )
        slice_infos.sort(key=lambda t: t[0])
        n_slices = len(slice_infos)
        origin = slice_infos[0][1]
        if n_slices > 1:
            z_spacing = float(np.linalg.norm(slice_infos[1][1] - slice_infos[0][1]))
        else:
            z_spacing = float(
                getattr(ds0, "SpacingBetweenSlices",
                        getattr(ds0, "SliceThickness", 1.0))
            )

    # Build the 3x3 direction matrix. Columns are the patient-frame
    # directions of ProxylFit's numpy (x, y, z) axes.
    direction = np.column_stack([x_cosine, y_cosine, slice_cosine])
    spacing = np.array([spacing_xy[1], spacing_xy[0], z_spacing], dtype=float)
    shape = (ncols, nrows, n_slices)

    return T1Geometry(
        shape=shape,
        origin=origin,
        spacing=spacing,
        direction=direction,
    )


# ---------------------------------------------------------------------------
# Bruker method-file parser
# ---------------------------------------------------------------------------

_PARAM_LINE_RE = re.compile(r"^##\$([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_ARRAY_SHAPE_RE = re.compile(r"^\(\s*([\d\s,]+)\s*\)\s*$")
_STRING_VALUE_RE = re.compile(r"^<(.*)>\s*$")


def _parse_bruker_params(path: Path) -> Dict[str, Any]:
    """Parse a Bruker ParaVision parameter file (``method``, ``acqp``,
    ``subject``, etc.) into a dict.

    Handles three value shapes that show up in the keys we care about:

    1. **String** — ``##$KEY=<value>`` on one line. Returned as ``"value"``.
    2. **Scalar / enum** — ``##$KEY=Bruker:STEAM`` or ``##$KEY=42``.
       Returned as the raw token string; numeric tokens are not auto-cast
       because some enum names look numeric.
    3. **Array** — ``##$KEY=( shape )`` on one line, followed by
       whitespace-separated values on subsequent lines until the next
       ``##$`` or ``$$`` directive. Returned as a flat list of strings
       (caller reshapes/casts). For a single-line string-typed array such
       as ``##$KEY=( 60 )\\n<value>``, the value is returned as
       ``"value"``.

    Structured-object values (lines beginning with ``(((``) are skipped;
    we don't need them for VOI extraction.

    Parameters
    ----------
    path : Path
        Path to the parameter file.

    Returns
    -------
    dict
        Map from parameter name (without the ``##$`` prefix) to either a
        string (cases 1, 2, and string-typed arrays) or a flat list of
        string tokens (case 3, numeric arrays).
    """
    path = Path(path)
    text = path.read_text(encoding="latin-1")
    # Split into logical lines but keep array bodies together.
    out: Dict[str, Any] = {}

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _PARAM_LINE_RE.match(line)
        if not m:
            i += 1
            continue

        key, rest = m.group(1), m.group(2).strip()

        # String value on the same line: ##$KEY=<value>
        sm = _STRING_VALUE_RE.match(rest)
        if sm:
            out[key] = sm.group(1)
            i += 1
            continue

        # Array declaration: ##$KEY=( shape )
        am = _ARRAY_SHAPE_RE.match(rest)
        if am:
            tokens: List[str] = []
            i += 1
            while i < len(lines):
                body = lines[i]
                if body.startswith("##$") or body.startswith("##") or body.startswith("$$"):
                    break
                stripped = body.strip()
                # String-typed array: a single <...> follows the shape.
                sm2 = _STRING_VALUE_RE.match(stripped)
                if sm2:
                    out[key] = sm2.group(1)
                    i += 1
                    break
                # Structured-object array (PVM_VoxelGeoCub etc.) — skip
                # the whole block until the next directive.
                if stripped.startswith("(") and not stripped.replace(
                    " ", ""
                ).replace(",", "").lstrip("(").rstrip(")").replace(
                    ".", ""
                ).replace("-", "").replace("e", "").replace("E", "").replace(
                    "+", ""
                ).isdigit():
                    # Not pure numeric — skip block.
                    while i < len(lines) and not (
                        lines[i].startswith("##")
                        or lines[i].startswith("$$")
                    ):
                        i += 1
                    out[key] = None
                    break
                tokens.extend(stripped.split())
                i += 1
            else:
                pass  # ran off end
            if key not in out:
                out[key] = tokens
            continue

        # Plain scalar / enum on the same line.
        out[key] = rest
        i += 1

    return out


def _floats(tokens: List[str], n: int) -> np.ndarray:
    """Convert a token list to a length-``n`` float array."""
    if tokens is None:
        raise ValueError("expected numeric array, got None")
    if len(tokens) < n:
        raise ValueError(
            f"expected {n} numeric tokens, got {len(tokens)}: {tokens!r}"
        )
    return np.array([float(x) for x in tokens[:n]], dtype=float)


# ---------------------------------------------------------------------------
# Per-VOI excitation-order normalization
# ---------------------------------------------------------------------------

# Canonical Bruker scanner frame after normalization:
#   X axis = R→L  (positive = patient-left)
#   Y axis = A→P  (positive = patient-posterior)
#   Z axis = H→F  (positive = patient-foot)
# So a vanilla rodent rig in Head_Prone position maps canonical →
# DICOM LPS via diag(+1, +1, -1) (Z flip only). Different scanners
# can require additional axis permutations / sign flips, captured in
# the user-tuned ``scanner_to_patient`` 4×4.
_EXC_TOKEN_TO_CANONICAL: Dict[str, Tuple[int, int]] = {
    "RL": (0, +1), "LR": (0, -1),
    "AP": (1, +1), "PA": (1, -1),
    "HF": (2, +1), "FH": (2, -1),
}


def _exc_order_to_perm(
    order_str: str,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Parse a Bruker ``PVM_VoxExcOrder`` token (e.g. ``AP_RL_HF``).

    Returns ``(perm, signs)`` such that
    ``canonical[k] = signs[k] * recorded[perm[k]]`` maps the recorded
    position triplet to the canonical (RL, AP, HF) ordering used
    downstream. Unknown / malformed tokens return identity.
    """
    parts = order_str.split("_") if order_str else []
    if len(parts) != 3:
        return (0, 1, 2), (1, 1, 1)
    perm = [0, 0, 0]
    signs = [1, 1, 1]
    for recorded_idx, token in enumerate(parts):
        if token not in _EXC_TOKEN_TO_CANONICAL:
            return (0, 1, 2), (1, 1, 1)
        canonical_idx, token_sign = _EXC_TOKEN_TO_CANONICAL[token]
        perm[canonical_idx] = recorded_idx
        signs[canonical_idx] = token_sign
    return tuple(perm), tuple(signs)


def parse_bruker_method(method_path: Path) -> Dict[str, Any]:
    """Parse one Bruker ``method`` file and return a STEAM-VOI descriptor.

    Reads only the keys needed for VOI geometry. Returns a dict matching
    the JSON schema's per-VOI entry shape (without ``label``, which is
    assigned downstream by the loader / UI).

    Parameters
    ----------
    method_path : Path
        Path to a Bruker ``method`` file. Typically lives at
        ``<study_root>/<expno>/method``.

    Returns
    -------
    dict
        ::

            {
                "method": "Bruker:STEAM",  # or whatever ##$Method says
                "frame": "bruker_scanner",
                "position_mm": [x, y, z],
                "size_mm":     [sx, sy, sz],
                "orientation": [[...], [...], [...]],
                "source": {
                    "kind": "bruker_method",
                    "path": "<absolute path>",
                    "keys_used": [
                        "PVM_VoxArrPosition",
                        "PVM_VoxArrSize",
                        "PVM_VoxArrGradOrient",
                    ],
                },
            }

    Raises
    ------
    FileNotFoundError
        If ``method_path`` does not exist.
    KeyError
        If any of the required ``PVM_VoxArr*`` keys are missing.
    ValueError
        If the array shapes don't match the expected counts.
    """
    method_path = Path(method_path)
    if not method_path.is_file():
        raise FileNotFoundError(method_path)

    params = _parse_bruker_params(method_path)

    method_str = params.get("Method", "")
    if not isinstance(method_str, str):
        method_str = ""

    required = ("PVM_VoxArrPosition", "PVM_VoxArrSize", "PVM_VoxArrGradOrient")
    for k in required:
        if k not in params:
            raise KeyError(
                f"{k} not found in {method_path} — not a STEAM VOI file?"
            )

    position_raw = _floats(params["PVM_VoxArrPosition"], 3)
    size = _floats(params["PVM_VoxArrSize"], 3)
    orient_flat = _floats(params["PVM_VoxArrGradOrient"], 9)
    orientation_raw = orient_flat.reshape(3, 3)

    # Per-VOI normalization: PVM_VoxArrPosition is recorded in the
    # gradient-axis order specified by PVM_VoxExcOrder (Read_Phase_Slice
    # → body axis tokens). Two STEAM voxels in the same session can use
    # different orderings (e.g. tumor=RL_AP_HF, contralateral=AP_RL_HF).
    # Normalizing both to a canonical (RL, AP, HF) frame here means one
    # user-tuned ``scanner_to_patient`` works for every voxel.
    exc_order = params.get("PVM_VoxExcOrder", "") or ""
    if isinstance(exc_order, list):  # array form (unlikely but defensive)
        exc_order = ""
    perm, signs = _exc_order_to_perm(exc_order)
    P = _signed_perm_matrix(perm, signs)[:3, :3]
    position = P @ position_raw
    # orientation maps scanner → RPS; we replace scanner with canonical
    # scanner via post-multiply by P^T:
    #   q_RPS = orientation_raw @ delta_recorded_scanner
    #         = orientation_raw @ (P^T @ delta_canonical_scanner)
    #         = (orientation_raw @ P^T) @ delta_canonical_scanner
    orientation = orientation_raw @ P.T

    return {
        "method": method_str or "Bruker:STEAM",
        "frame": "bruker_scanner",
        "position_mm": position.tolist(),
        "size_mm": size.tolist(),
        "orientation": orientation.tolist(),
        "source": {
            "kind": "bruker_method",
            "path": str(method_path),
            "keys_used": list(required),
            "voxel_exc_order": exc_order,
        },
    }


def _is_steam_method(params: Dict[str, Any]) -> bool:
    """Return True if a parsed method file represents a STEAM acquisition."""
    m = params.get("Method", "")
    return isinstance(m, str) and "STEAM" in m.upper()


# ---------------------------------------------------------------------------
# Bruker study / root scanners
# ---------------------------------------------------------------------------

def _iter_expnos(subject_root: Path):
    """Yield ``(expno_int, expno_path)`` for every numbered subdirectory."""
    for child in sorted(subject_root.iterdir(), key=lambda p: p.name):
        if child.is_dir() and child.name.isdigit():
            yield int(child.name), child


def scan_bruker_study(subject_root: Path) -> List[Dict[str, Any]]:
    """Scan one Bruker subject folder and return all STEAM VOI descriptors.

    Walks every numbered ExpNo subdirectory; for each, reads only enough
    of ``method`` to check ``##$Method`` and, if it matches STEAM,
    extracts the VOI geometry.

    Parameters
    ----------
    subject_root : Path
        A single subject's Bruker folder, e.g.
        ``.../20250923_..._B2_D16_.../``. Contains numbered ExpNo
        subdirectories.

    Returns
    -------
    list[dict]
        One VOI descriptor per STEAM acquisition, with an extra
        ``source.expno`` field set to the integer ExpNo. Empty list if
        no STEAM scans were found (or the folder doesn't look like a
        Bruker study).
    """
    subject_root = Path(subject_root)
    if not subject_root.is_dir():
        return []

    results: List[Dict[str, Any]] = []
    for expno, expno_dir in _iter_expnos(subject_root):
        method_path = expno_dir / "method"
        if not method_path.is_file():
            continue
        try:
            params = _parse_bruker_params(method_path)
        except (OSError, UnicodeDecodeError):
            continue
        if not _is_steam_method(params):
            continue
        try:
            voi = parse_bruker_method(method_path)
        except (KeyError, ValueError):
            continue
        voi["source"]["expno"] = expno
        results.append(voi)
    return results


def scan_bruker_root(root: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Scan a multi-subject Bruker root and return per-subject STEAM VOIs.

    For each immediate subdirectory that contains a ``subject`` file
    plus numbered ExpNo dirs, read the subject identifier and collect
    every STEAM VOI found in that subject's experiments.

    Parameters
    ----------
    root : Path
        Parent directory that contains multiple per-subject Bruker
        folders.

    Returns
    -------
    dict[str, list[dict]]
        Map from subject identifier (the Bruker ``SUBJECT_id`` if
        readable, else the folder basename) to a list of VOI
        descriptors. Subjects with no STEAM scans are omitted.
    """
    root = Path(root)
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not root.is_dir():
        return out

    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        subj_file = child / "subject"
        # Heuristic: a Bruker subject folder either has a subject file
        # or has numbered ExpNo subdirectories.
        if not subj_file.is_file() and not any(
            c.is_dir() and c.name.isdigit() for c in child.iterdir()
        ):
            continue

        vois = scan_bruker_study(child)
        if not vois:
            continue

        subject_id = child.name
        if subj_file.is_file():
            try:
                subj = _parse_bruker_params(subj_file)
                sid = subj.get("SUBJECT_id")
                if isinstance(sid, str) and sid:
                    subject_id = sid
            except (OSError, UnicodeDecodeError):
                pass

        out[subject_id] = vois

    return out


def parse_bruker_subject(subject_file: Path) -> Dict[str, Any]:
    """Extract identification fields from a Bruker ``subject`` file.

    Returns a small dict with the keys most useful for matching a Bruker
    subject folder to a DICOM study: ``subject_id`` (Bruker
    ``SUBJECT_id``), ``study_name`` (``SUBJECT_study_name``),
    ``study_instance_uid`` (``SUBJECT_study_instance_uid``), and
    ``instrument_position`` (``SUBJECT_study_instrument_position``).

    Missing keys yield ``None`` in the returned dict (rather than raising)
    so the caller can fall back to other match keys.
    """
    params = _parse_bruker_params(Path(subject_file))
    return {
        "subject_id": params.get("SUBJECT_id") or None,
        "study_name": params.get("SUBJECT_study_name") or None,
        "study_instance_uid": params.get("SUBJECT_study_instance_uid") or None,
        "instrument_position": params.get(
            "SUBJECT_study_instrument_position"
        )
        or None,
    }


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def load_voi_json(path: Path) -> StudyVOIs:
    """Read a ``steam_voi.json`` sidecar from disk.

    Parameters
    ----------
    path : Path
        Path to the JSON file.

    Returns
    -------
    StudyVOIs

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    ValueError
        If the schema version is unsupported.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    sv = StudyVOIs.from_dict(d)
    if sv.schema_version != 1:
        raise ValueError(
            f"unsupported steam_voi.json schema_version={sv.schema_version}"
        )
    return sv


def save_voi_json(path: Path, study_vois: StudyVOIs) -> None:
    """Write a ``steam_voi.json`` sidecar, pretty-printed.

    Parameters
    ----------
    path : Path
        Output path; parent directories are created if missing.
    study_vois : StudyVOIs
        VOIs to persist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(study_vois.to_dict(), f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Geometry: VOI → T1 voxel mask
# ---------------------------------------------------------------------------

def _grid_patient_coords(geom: T1Geometry) -> np.ndarray:
    """Return an ``(nx, ny, nz, 3)`` array of patient-frame voxel centers."""
    nx, ny, nz = geom.shape
    ii, jj, kk = np.meshgrid(
        np.arange(nx, dtype=float),
        np.arange(ny, dtype=float),
        np.arange(nz, dtype=float),
        indexing="ij",
    )
    # Each voxel center: origin + i*sx*d0 + j*sy*d1 + k*sz*d2
    d = geom.direction  # columns are voxel axes in patient coords
    s = geom.spacing
    px = (
        geom.origin[0]
        + ii * s[0] * d[0, 0]
        + jj * s[1] * d[0, 1]
        + kk * s[2] * d[0, 2]
    )
    py = (
        geom.origin[1]
        + ii * s[0] * d[1, 0]
        + jj * s[1] * d[1, 1]
        + kk * s[2] * d[1, 2]
    )
    pz = (
        geom.origin[2]
        + ii * s[0] * d[2, 0]
        + jj * s[1] * d[2, 1]
        + kk * s[2] * d[2, 2]
    )
    return np.stack([px, py, pz], axis=-1)


def _apply_transform(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Apply a 4×4 affine to a (..., 3) array of points."""
    R = affine[:3, :3]
    t = affine[:3, 3]
    return points @ R.T + t


def voi_to_mask(
    voi: SteamVOI,
    geom: T1Geometry,
    scanner_to_patient: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Rasterize a VOI to a 3D boolean mask aligned with a T1 grid.

    Parameters
    ----------
    voi : SteamVOI
        VOI to rasterize. Read from ``voi.frame``:

        - ``"bruker_scanner"`` — ``position_mm`` and ``orientation`` are
          in the Bruker scanner frame; ``scanner_to_patient`` is applied.
        - ``"patient_lps"`` — already in DICOM patient coords;
          ``scanner_to_patient`` is ignored.
    geom : T1Geometry
        Target voxel grid. Mask shape will equal ``geom.shape``.
    scanner_to_patient : np.ndarray or None
        4×4 affine taking points from Bruker scanner coordinates to
        DICOM patient (LPS). Default: identity.

    Returns
    -------
    np.ndarray, shape ``geom.shape``, dtype bool
        ``True`` for voxels whose center lies inside the VOI cube.

    Notes
    -----
    The interior test uses an inclusive half-edge bound
    (``|q| <= size/2``), so a voxel center sitting exactly on a face is
    counted as inside. This matches typical scanner behavior where the
    prescribed voxel is inclusive on all bounding faces.
    """
    if scanner_to_patient is None:
        scanner_to_patient = np.eye(4)
    scanner_to_patient = np.asarray(scanner_to_patient, dtype=float).reshape(4, 4)
    # Per-VOI override takes precedence when set.
    effective = voi.effective_transform(scanner_to_patient)

    pts_patient = _grid_patient_coords(geom)  # (nx, ny, nz, 3)

    if voi.frame == "patient_lps":
        pts_frame = pts_patient
    elif voi.frame == "bruker_scanner":
        # Need patient → scanner.
        inv = np.linalg.inv(effective)
        pts_frame = _apply_transform(pts_patient, inv)
    else:
        raise ValueError(f"unknown VOI frame: {voi.frame!r}")

    delta = pts_frame - voi.position_mm  # (nx, ny, nz, 3) in frame coords
    # Logical-frame offset: rows of orientation are logical axes in frame.
    # q_logical[i] = orientation[i, :] · delta
    q = np.einsum("ij,xyzj->xyzi", voi.orientation, delta)

    half = voi.size_mm / 2.0
    inside = np.all(np.abs(q) <= half, axis=-1)
    return inside


def cluster_voi_acquisitions(
    vois: List[Dict[str, Any]],
    tolerance_mm: float = 0.1,
) -> List[Dict[str, Any]]:
    """Group VOI acquisitions by prescribed position.

    A single subject often has the same VOI re-acquired several times
    (averaging, pre/post comparisons, multiple imaging timepoints). The
    Bruker tree records each acquisition separately under its own ExpNo
    even when the geometry is identical. This function groups
    acquisitions whose ``position_mm`` agree within ``tolerance_mm``
    (Euclidean distance) and returns one cluster per distinct prescribed
    voxel.

    Parameters
    ----------
    vois : list[dict]
        VOI descriptors as returned by ``scan_bruker_study`` / ``parse_bruker_method``.
    tolerance_mm : float, default 0.1
        Maximum Euclidean distance between two ``position_mm`` values
        for them to be considered the same prescribed voxel.

    Returns
    -------
    list[dict]
        One entry per cluster, with keys::

            {
                "position_mm":  [x, y, z],     # mean of cluster members
                "size_mm":      [sx, sy, sz],  # from first member
                "orientation":  [[...]],       # from first member
                "method":       "Bruker:STEAM",
                "frame":        "bruker_scanner",
                "members":      [voi_dict, ...],  # original acquisitions
                "expnos":       [int, ...],       # for members with source.expno
            }

        Clusters are sorted by the smallest member ExpNo so the order
        is stable and the natural acquisition order is preserved.
    """
    clusters: List[Dict[str, Any]] = []
    for voi in vois:
        pos = np.asarray(voi["position_mm"], dtype=float)
        # Find an existing cluster whose center is within tolerance.
        matched = None
        for c in clusters:
            cpos = np.asarray(c["position_mm"], dtype=float)
            if np.linalg.norm(pos - cpos) <= tolerance_mm:
                matched = c
                break
        if matched is None:
            matched = {
                "position_mm": list(voi["position_mm"]),
                "size_mm": list(voi["size_mm"]),
                "orientation": [list(row) for row in voi["orientation"]],
                "method": voi.get("method", "Bruker:STEAM"),
                "frame": voi.get("frame", "bruker_scanner"),
                "members": [],
                "expnos": [],
            }
            clusters.append(matched)
        matched["members"].append(voi)
        expno = voi.get("source", {}).get("expno")
        if expno is not None:
            matched["expnos"].append(int(expno))
        # Update cluster center to running mean (stabilizes against noise).
        all_pos = np.array(
            [m["position_mm"] for m in matched["members"]], dtype=float
        )
        matched["position_mm"] = list(all_pos.mean(axis=0))

    # Stable sort: by smallest ExpNo in cluster (acquisition order).
    def _sort_key(c):
        return min(c["expnos"]) if c["expnos"] else float("inf")

    clusters.sort(key=_sort_key)
    return clusters


def cluster_to_voi(cluster: Dict[str, Any], label: str) -> SteamVOI:
    """Build a ``SteamVOI`` from a cluster, using the cluster's mean
    position and the first member's size + orientation.

    Records the contributing ExpNos in ``source.expnos`` so the
    provenance is preserved in the saved JSON.
    """
    first_member = cluster["members"][0]
    return SteamVOI(
        label=label,
        method=cluster["method"],
        frame=cluster["frame"],
        position_mm=np.asarray(cluster["position_mm"], dtype=float),
        size_mm=np.asarray(cluster["size_mm"], dtype=float),
        orientation=np.asarray(cluster["orientation"], dtype=float),
        source={
            "kind": "bruker_method_cluster",
            "expnos": list(cluster["expnos"]),
            "n_members": len(cluster["members"]),
            "paths": [m.get("source", {}).get("path") for m in cluster["members"]],
        },
    )


def group_clusters_by_timepoint(
    clusters: List[Dict[str, Any]],
    max_expno_gap: int = 4,
) -> List[List[Dict[str, Any]]]:
    """Heuristic: group clusters into per-timepoint blocks by ExpNo.

    Acquisitions in the same imaging timepoint are usually performed
    back-to-back, so their ExpNos form a contiguous run. We sort the
    clusters by minimum ExpNo and start a new timepoint whenever the
    gap exceeds ``max_expno_gap``.

    Parameters
    ----------
    clusters : list[dict]
        Output of ``cluster_voi_acquisitions``.
    max_expno_gap : int, default 4
        Gap in ExpNo numbering that separates one timepoint from the
        next. Tune per scanner — 4 leaves room for one or two
        non-STEAM scans (e.g. a T2 between STEAM pairs).

    Returns
    -------
    list[list[dict]]
        Ordered list of timepoint groups; each group is a list of
        clusters that fall within one timepoint block.
    """
    if not clusters:
        return []

    # Sort by minimum ExpNo in each cluster.
    items = sorted(clusters, key=lambda c: min(c["expnos"]) if c["expnos"] else 0)
    groups: List[List[Dict[str, Any]]] = [[items[0]]]
    prev_max = max(items[0]["expnos"], default=0)
    for c in items[1:]:
        c_min = min(c["expnos"]) if c["expnos"] else prev_max + 1
        if c_min - prev_max > max_expno_gap:
            groups.append([c])
        else:
            groups[-1].append(c)
        prev_max = max(prev_max, max(c["expnos"], default=prev_max))
    return groups


# ---------------------------------------------------------------------------
# Scanner-to-patient transform auto-detection
# ---------------------------------------------------------------------------

# Eight candidate sign-flip vectors on the (x, y, z) axes.
_AXIS_FLIPS: List[Tuple[int, int, int]] = [
    (+1, +1, +1),
    (-1, +1, +1),
    (+1, -1, +1),
    (+1, +1, -1),
    (-1, -1, +1),
    (-1, +1, -1),
    (+1, -1, -1),
    (-1, -1, -1),
]

# Six axis permutations. ``perm = (a, b, c)`` means
# ``patient[k] = scanner[perm[k]]`` after applying signs. Bruker's
# scanner XYZ axes do not always coincide with DICOM LPS XYZ — the
# ``PVM_VoxExcOrder`` field (e.g. ``AP_RL_HF`` vs. ``RL_AP_HF``) tells
# us which permutation maps Bruker → patient.
_AXIS_PERMS: List[Tuple[int, int, int]] = [
    (0, 1, 2),  # XYZ
    (0, 2, 1),  # XZY
    (1, 0, 2),  # YXZ   ← e.g. PVM_VoxExcOrder=AP_RL_HF
    (1, 2, 0),  # YZX
    (2, 0, 1),  # ZXY
    (2, 1, 0),  # ZYX
]


def _flip_matrix(signs: Tuple[int, int, int]) -> np.ndarray:
    """Convenience: diagonal sign-flip 4×4 affine (no permutation)."""
    return _signed_perm_matrix((0, 1, 2), signs)


def _signed_perm_matrix(
    perm: Tuple[int, int, int],
    signs: Tuple[int, int, int],
) -> np.ndarray:
    """Build a 4×4 affine that permutes and sign-flips scanner axes.

    For input ``(x_s, y_s, z_s)`` in scanner coords, the result satisfies
    ``patient[k] = signs[k] * scanner[perm[k]]``.

    Example — Bruker scanner X is actually AP (patient Y), scanner Y is
    RL (patient X), Z is HF (patient Z), with no sign flips::

        M = _signed_perm_matrix((1, 0, 2), (+1, +1, +1))
        # patient_X = scanner_Y, patient_Y = scanner_X, patient_Z = scanner_Z
    """
    M = np.zeros((4, 4), dtype=float)
    for i in range(3):
        M[i, perm[i]] = float(signs[i])
    M[3, 3] = 1.0
    return M


_PERM_LABELS = {p: name for p, name in zip(
    _AXIS_PERMS, ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]
)}


def _perm_from_matrix(M: np.ndarray) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Extract ``(perm, signs)`` from a signed-permutation 4×4 affine.

    Returns ``((perm_x, perm_y, perm_z), (sign_x, sign_y, sign_z))``
    such that ``_signed_perm_matrix(perm, signs)`` reproduces the
    rotation/scale part of ``M`` (translation ignored).
    """
    R = np.asarray(M, dtype=float)[:3, :3]
    perm = [0, 0, 0]
    signs = [1, 1, 1]
    for i in range(3):
        # Find which column of row i is non-zero.
        col = int(np.argmax(np.abs(R[i, :])))
        perm[i] = col
        signs[i] = 1 if R[i, col] >= 0 else -1
    return tuple(perm), tuple(signs)


def _build_brain_mask(t1: np.ndarray) -> np.ndarray:
    """Coarse anatomy mask separating tissue from air.

    Uses an Otsu-like split between the dark and bright populations
    (works well for high-contrast preclinical MRI where most of the
    field-of-view is air). The threshold is the value that maximises
    inter-class variance; a percentile fallback is used when the data
    is too uniform for Otsu to converge meaningfully.

    Returns a 3D boolean mask the same shape as ``t1``.
    """
    flat = t1[np.isfinite(t1)].astype(float)
    if flat.size == 0:
        return np.zeros_like(t1, dtype=bool)

    # Otsu's method on a 256-bin histogram.
    lo, hi = float(flat.min()), float(flat.max())
    if hi <= lo:
        return np.zeros_like(t1, dtype=bool)
    hist, edges = np.histogram(flat, bins=256, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = hist.sum()
    if total == 0:
        return np.zeros_like(t1, dtype=bool)
    p = hist.astype(float) / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    threshold_idx = int(np.argmax(sigma_b))
    threshold = float(centers[threshold_idx])
    return t1 > threshold


def auto_detect_scanner_to_patient(
    vois: List[SteamVOI],
    t1_volume: np.ndarray,
    t1_geometry: T1Geometry,
    brain_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[Tuple[Tuple[int, int, int], float]]]:
    """Find the axis-flip transform that maximises VOI/anatomy overlap.

    For each of the 8 axis-sign combinations, rasterize all candidate
    VOIs onto the T1 grid using that flip as ``scanner_to_patient``,
    and compute the fraction of VOI voxels whose T1 intensity is above
    a coarse anatomy threshold. The best-scoring flip is returned.

    Parameters
    ----------
    vois : list[SteamVOI]
        VOIs whose ``frame`` is ``"bruker_scanner"``. VOIs with
        ``frame == "patient_lps"`` are ignored (they don't depend on
        the transform).
    t1_volume : np.ndarray, shape == ``t1_geometry.shape``
        T1 volume in patient-grid coordinates. Used to derive a coarse
        anatomy mask.
    t1_geometry : T1Geometry
    brain_mask : np.ndarray or None
        Optional precomputed mask. If ``None``, computed from
        ``t1_volume`` at the 30th-percentile threshold.

    Returns
    -------
    best_transform : np.ndarray, shape (4, 4)
        4×4 affine for the winning axis-flip.
    scores : list[((int, int, int), float)]
        All eight candidates with their overlap scores
        ``[((sx, sy, sz), score), ...]``, sorted descending. Useful
        for the UI to display "this is the best, here are the others".
    """
    if brain_mask is None:
        brain_mask = _build_brain_mask(t1_volume)
    scanner_vois = [v for v in vois if v.frame == "bruker_scanner"]
    if not scanner_vois:
        return np.eye(4), [((+1, +1, +1), 1.0)]

    # Score: overlap with brain mask, penalised heavily when the VOI
    # falls entirely outside the field-of-view. We blend (a) fraction
    # of VOI voxels inside the brain mask with (b) mean T1 intensity
    # under the VOI normalised by the global mean — both reward
    # transforms that land the cuboid on tissue, but (b) keeps a
    # signal even when the brain mask is itself imperfect.
    t1_mean = float(t1_volume[brain_mask].mean()) if brain_mask.any() else 1.0
    if t1_mean <= 0:
        t1_mean = 1.0

    # Try all 6 axis permutations × 8 sign-flip combinations (48 total).
    # Bruker's scanner XYZ axes can be aligned with patient anatomy in
    # any of these — the actual mapping is determined by the bore
    # orientation and the gradient axis assignment.
    scores: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], float]] = []
    for perm in _AXIS_PERMS:
        for signs in _AXIS_FLIPS:
            M = _signed_perm_matrix(perm, signs)
            combined = np.zeros(t1_geometry.shape, dtype=bool)
            for v in scanner_vois:
                combined |= voi_to_mask(v, t1_geometry, scanner_to_patient=M)
            total = int(combined.sum())
            if total == 0:
                scores.append((perm, signs, 0.0))
                continue
            inside_frac = (
                float(np.logical_and(combined, brain_mask).sum()) / total
            )
            intensity_ratio = float(t1_volume[combined].mean()) / t1_mean
            scores.append((perm, signs, 0.7 * inside_frac + 0.3 * intensity_ratio))

    scores.sort(key=lambda item: item[2], reverse=True)
    best_perm, best_signs, _ = scores[0]
    return _signed_perm_matrix(best_perm, best_signs), scores


def voi_to_polygon(
    voi: SteamVOI,
    z_index: int,
    geom: T1Geometry,
    scanner_to_patient: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Analytic intersection of the VOI cube with a single T1 z-slice.

    Computes the polygon where the prescribed cuboid crosses the plane
    ``k = z_index`` in T1 voxel coordinates. For a rotated cube the
    cross-section is a polygon with 3–6 sides (not a rectangle), so the
    returned outline matches the actual cube geometry exactly rather
    than approximating with a bounding rectangle.

    Parameters
    ----------
    voi : SteamVOI
    z_index : int
        Slice index within the T1 grid.
    geom : T1Geometry
    scanner_to_patient : np.ndarray or None
        Study-default 4×4 affine; ``voi.transform`` overrides when set.

    Returns
    -------
    np.ndarray, shape (N+1, 2), or None
        Closed polygon vertices ``(i, j)`` in T1 voxel coordinates.
        Last vertex equals the first. Returns ``None`` if the cube
        does not intersect the slice plane.
    """
    if scanner_to_patient is None:
        scanner_to_patient = np.eye(4)
    transform = voi.effective_transform(
        np.asarray(scanner_to_patient, dtype=float).reshape(4, 4)
    )

    # 1) Build 8 cube corners in the VOI's *frame* (Bruker scanner or
    #    patient LPS). The cube's local edges are the rows of
    #    ``orientation`` scaled by ``size_mm / 2``.
    half = voi.size_mm / 2.0
    R = voi.orientation  # rows = (Read, Phase, Slice) axes in frame coords
    signs = np.array(
        [[e0, e1, e2] for e0 in (-1, 1) for e1 in (-1, 1) for e2 in (-1, 1)],
        dtype=float,
    )  # shape (8, 3)
    # corner = position + e0 * half[0] * R[0] + e1 * half[1] * R[1] + e2 * half[2] * R[2]
    corners_frame = (
        voi.position_mm
        + signs[:, 0:1] * half[0] * R[0, :]
        + signs[:, 1:2] * half[1] * R[1, :]
        + signs[:, 2:3] * half[2] * R[2, :]
    )  # shape (8, 3)

    # 2) Transform corners frame → patient → T1 voxel coordinates.
    if voi.frame == "patient_lps":
        corners_patient = corners_frame
    elif voi.frame == "bruker_scanner":
        M = transform[:3, :3]
        t = transform[:3, 3]
        corners_patient = corners_frame @ M.T + t
    else:
        raise ValueError(f"unknown VOI frame: {voi.frame!r}")

    # patient → voxel-index: project (patient - origin) onto each voxel
    # axis (column of direction) and divide by spacing.
    delta = corners_patient - geom.origin
    corners_voxel = np.column_stack([
        delta @ geom.direction[:, i] / geom.spacing[i] for i in range(3)
    ])  # shape (8, 3); columns = (i_voxel, j_voxel, k_voxel)

    # 3) Enumerate the 12 cube edges (pairs of corners differing in
    #    exactly one sign component) and find each edge's intersection
    #    with the plane k = z_index.
    z_target = float(z_index)
    poly_pts: List[np.ndarray] = []
    for a in range(8):
        for b in range(a + 1, 8):
            diff = int(np.abs(signs[a] - signs[b]).sum())
            if diff != 2:  # one coord differs by 2 (from -1 to +1)
                continue
            za = corners_voxel[a, 2]
            zb = corners_voxel[b, 2]
            if za == zb:
                # Edge parallel to slice plane — include both endpoints
                # if they lie exactly on the plane.
                if za == z_target:
                    poly_pts.append(corners_voxel[a, :2])
                    poly_pts.append(corners_voxel[b, :2])
                continue
            t = (z_target - za) / (zb - za)
            if -1e-9 <= t <= 1.0 + 1e-9:
                pt = corners_voxel[a] + t * (corners_voxel[b] - corners_voxel[a])
                poly_pts.append(pt[:2])

    if len(poly_pts) < 3:
        return None

    # 4) Deduplicate and sort by angle around centroid to get a
    #    well-formed convex polygon outline.
    poly = np.array(poly_pts, dtype=float)
    # Snap-and-unique to half-voxel resolution.
    keys = np.round(poly * 2.0).astype(int)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    poly = poly[np.sort(unique_idx)]
    if poly.shape[0] < 3:
        return None
    centroid = poly.mean(axis=0)
    angles = np.arctan2(poly[:, 1] - centroid[1], poly[:, 0] - centroid[0])
    order = np.argsort(angles)
    poly = poly[order]

    # Close the polygon.
    return np.vstack([poly, poly[:1]])
