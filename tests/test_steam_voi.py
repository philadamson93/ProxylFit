"""
Tests for ``proxyl_analysis.steam_voi`` — Phase 1 of T024.

Covers:
- Bruker ``method`` file parsing (synthetic + real reference dataset)
- Multi-subject scanning
- JSON schema round-trip
- ``voi_to_mask`` geometry on a synthetic T1 grid
- Edge cases: missing keys, non-STEAM method, malformed input
"""

import json
from pathlib import Path

import numpy as np
import pytest

from proxyl_analysis.steam_voi import (
    StudyVOIs,
    SteamVOI,
    T1Geometry,
    auto_detect_scanner_to_patient,
    cluster_to_voi,
    cluster_voi_acquisitions,
    group_clusters_by_timepoint,
    load_voi_json,
    parse_bruker_method,
    parse_bruker_subject,
    save_voi_json,
    scan_bruker_root,
    scan_bruker_study,
    voi_to_mask,
    voi_to_polygon,
)


# ---------------------------------------------------------------------------
# Reference values from the real dataset's 34/method (hand-extracted)
# ---------------------------------------------------------------------------

REF_POSITION = np.array(
    [-2.5019681032489505, -7.4787887915866866, 0.89297899944217862]
)
REF_SIZE = np.array([3.0, 3.0, 3.0])
REF_ORIENTATION = np.array(
    [
        [0.66529949845527558, 0.28308262156683794, -0.69082617692296922],
        [-0.34423973069218666, 0.93740689973916491, 0.052605248163846915],
        [0.66247685632641573, 0.2028115718795524, 0.72110462565678579],
    ]
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_method_file(tmp_path: Path) -> Path:
    """Minimal Bruker method file matching the real 34/method's STEAM block."""
    content = (
        "##TITLE=Parameter List, ParaVision 360 V3.6\n"
        "##JCAMPDX=4.24\n"
        "##DATATYPE=Parameter Values\n"
        "##$Method=<Bruker:STEAM>\n"
        "##$PVM_VoxArrSize=( 1, 3 )\n"
        "3 3 3\n"
        "##$PVM_VoxArrPosition=( 1, 3 )\n"
        f"{REF_POSITION[0]:.16f} {REF_POSITION[1]:.16f} {REF_POSITION[2]:.16f}\n"
        "##$PVM_VoxArrPositionRPS=( 1, 3 )\n"
        f"{REF_POSITION[0]:.16f} {REF_POSITION[1]:.16f} {REF_POSITION[2]:.16f}\n"
        "##$PVM_VoxArrGradOrient=( 1, 3, 3 )\n"
        f"{REF_ORIENTATION[0,0]:.17f} {REF_ORIENTATION[0,1]:.17f} {REF_ORIENTATION[0,2]:.17f} \n"
        f"{REF_ORIENTATION[1,0]:.17f} {REF_ORIENTATION[1,1]:.17f} {REF_ORIENTATION[1,2]:.18f}\n"
        f"{REF_ORIENTATION[2,0]:.17f} {REF_ORIENTATION[2,1]:.16f} {REF_ORIENTATION[2,2]:.17f}\n"
        "##$PVM_NVoxels=1\n"
        "##END=\n"
    )
    p = tmp_path / "method"
    p.write_text(content)
    return p


@pytest.fixture
def synthetic_t2_method_file(tmp_path: Path) -> Path:
    """A non-STEAM method file used to confirm scanners filter correctly."""
    content = (
        "##TITLE=Parameter List, ParaVision 360 V3.6\n"
        "##$Method=<Bruker:RARE>\n"
        "##$PVM_NVoxels=0\n"
        "##END=\n"
    )
    p = tmp_path / "method"
    p.write_text(content)
    return p


@pytest.fixture
def synthetic_subject_file(tmp_path: Path) -> Path:
    content = (
        "##TITLE=Parameter List, ParaVision 360 V3.6\n"
        "##$SUBJECT_id=( 60 )\n"
        "<Test_Subject_42>\n"
        "##$SUBJECT_study_name=( 64 )\n"
        "<Synthetic-MRS-Study>\n"
        "##$SUBJECT_study_instrument_position=Head_Prone\n"
        "##END=\n"
    )
    p = tmp_path / "subject"
    p.write_text(content)
    return p


@pytest.fixture
def synthetic_bruker_study(tmp_path: Path, synthetic_subject_file: Path):
    """A Bruker subject folder with three ExpNos: one STEAM, one T2, one
    STEAM. Mimics the real dataset where two STEAM voxels are acquired
    per subject."""
    study = tmp_path / "20250923_synthetic_study"
    study.mkdir()
    (study / "subject").write_text(synthetic_subject_file.read_text())

    # ExpNo 9 — STEAM #1 (tumor)
    expno9 = study / "9"
    expno9.mkdir()
    pos9 = REF_POSITION
    _write_steam_method(expno9 / "method", pos9, REF_SIZE, REF_ORIENTATION)

    # ExpNo 11 — T2 (non-STEAM, should be ignored)
    expno11 = study / "11"
    expno11.mkdir()
    (expno11 / "method").write_text(
        "##$Method=<Bruker:RARE>\n##END=\n"
    )

    # ExpNo 14 — STEAM #2 (contralateral): mirror across X
    expno14 = study / "14"
    expno14.mkdir()
    pos14 = REF_POSITION * np.array([-1.0, 1.0, 1.0])
    _write_steam_method(expno14 / "method", pos14, REF_SIZE, REF_ORIENTATION)

    return study


def _write_steam_method(
    path: Path,
    position: np.ndarray,
    size: np.ndarray,
    orient: np.ndarray,
) -> None:
    """Helper: write a minimal STEAM method file with the given values."""
    content = (
        "##$Method=<Bruker:STEAM>\n"
        "##$PVM_VoxArrSize=( 1, 3 )\n"
        f"{size[0]} {size[1]} {size[2]}\n"
        "##$PVM_VoxArrPosition=( 1, 3 )\n"
        f"{position[0]:.17f} {position[1]:.17f} {position[2]:.17f}\n"
        "##$PVM_VoxArrGradOrient=( 1, 3, 3 )\n"
        f"{orient[0,0]:.17f} {orient[0,1]:.17f} {orient[0,2]:.17f}\n"
        f"{orient[1,0]:.17f} {orient[1,1]:.17f} {orient[1,2]:.17f}\n"
        f"{orient[2,0]:.17f} {orient[2,1]:.17f} {orient[2,2]:.17f}\n"
        "##END=\n"
    )
    path.write_text(content)


def _real_method_path() -> Path:
    """Return the path to the real reference method file, if present."""
    return Path(
        "/Users/ralphhurd/Downloads/"
        "20250923_072207_Recht_ICglioma_Survival_MRS_and_3CP_B2_D16_09232025_2_49/"
        "34/method"
    )


# ---------------------------------------------------------------------------
# Method-file parser
# ---------------------------------------------------------------------------

class TestParseBrukerMethod:

    def test_parses_synthetic_method(self, synthetic_method_file: Path):
        voi = parse_bruker_method(synthetic_method_file)
        assert voi["method"] == "Bruker:STEAM"
        assert voi["frame"] == "bruker_scanner"
        np.testing.assert_allclose(voi["position_mm"], REF_POSITION, rtol=1e-12)
        np.testing.assert_allclose(voi["size_mm"], REF_SIZE)
        np.testing.assert_allclose(
            voi["orientation"], REF_ORIENTATION, rtol=1e-12
        )
        assert voi["source"]["kind"] == "bruker_method"
        assert voi["source"]["path"] == str(synthetic_method_file)
        assert "PVM_VoxArrPosition" in voi["source"]["keys_used"]

    def test_real_dataset_matches_hand_extracted_values(self):
        """Smoke test against the actual 34/method file used in T024 design."""
        real = _real_method_path()
        if not real.is_file():
            pytest.skip(f"reference Bruker method file not present at {real}")
        voi = parse_bruker_method(real)
        assert voi["method"].upper().endswith("STEAM")
        np.testing.assert_allclose(voi["position_mm"], REF_POSITION, rtol=1e-12)
        np.testing.assert_allclose(voi["size_mm"], REF_SIZE)
        np.testing.assert_allclose(
            voi["orientation"], REF_ORIENTATION, rtol=1e-12
        )

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parse_bruker_method(tmp_path / "no_such_method")

    def test_missing_required_key_raises(self, tmp_path: Path):
        broken = tmp_path / "method"
        broken.write_text("##$Method=<Bruker:STEAM>\n##END=\n")
        with pytest.raises(KeyError):
            parse_bruker_method(broken)


# ---------------------------------------------------------------------------
# Study scanner
# ---------------------------------------------------------------------------

class TestScanBrukerStudy:

    def test_finds_two_steam_voxels(self, synthetic_bruker_study: Path):
        vois = scan_bruker_study(synthetic_bruker_study)
        assert len(vois) == 2
        expnos = sorted(v["source"]["expno"] for v in vois)
        assert expnos == [9, 14]
        # Both are STEAM
        assert all("STEAM" in v["method"].upper() for v in vois)

    def test_skips_non_steam_methods(
        self, tmp_path: Path, synthetic_t2_method_file: Path
    ):
        study = tmp_path / "study_with_only_t2"
        study.mkdir()
        expno = study / "5"
        expno.mkdir()
        (expno / "method").write_text(synthetic_t2_method_file.read_text())
        assert scan_bruker_study(study) == []

    def test_missing_folder_returns_empty(self, tmp_path: Path):
        assert scan_bruker_study(tmp_path / "nope") == []

    def test_real_dataset_finds_at_least_one_steam(self):
        real_root = _real_method_path().parent.parent
        if not real_root.is_dir():
            pytest.skip("reference Bruker study not present")
        vois = scan_bruker_study(real_root)
        assert len(vois) >= 1
        assert any("STEAM" in v["method"].upper() for v in vois)


class TestScanBrukerRoot:

    def test_finds_subject_by_subject_id(self, tmp_path: Path, synthetic_bruker_study: Path):
        # Place the synthetic study under a multi-subject root.
        root = tmp_path / "multi_subject_root"
        root.mkdir()
        target = root / synthetic_bruker_study.name
        # Move the study into the root.
        synthetic_bruker_study.rename(target)
        result = scan_bruker_root(root)
        # Subject id parsed from the subject file
        assert "Test_Subject_42" in result
        assert len(result["Test_Subject_42"]) == 2

    def test_skips_irrelevant_subfolders(self, tmp_path: Path):
        root = tmp_path / "mixed"
        root.mkdir()
        # A folder that doesn't look like a Bruker subject (no subject, no expnos)
        junk = root / "not_a_subject"
        junk.mkdir()
        (junk / "random_file.txt").write_text("hello")
        result = scan_bruker_root(root)
        assert result == {}


# ---------------------------------------------------------------------------
# Subject-file parser
# ---------------------------------------------------------------------------

def test_parse_bruker_subject(synthetic_subject_file: Path):
    info = parse_bruker_subject(synthetic_subject_file)
    assert info["subject_id"] == "Test_Subject_42"
    assert info["study_name"] == "Synthetic-MRS-Study"
    assert info["instrument_position"] == "Head_Prone"


# ---------------------------------------------------------------------------
# JSON I/O round-trip
# ---------------------------------------------------------------------------

class TestJsonRoundtrip:

    def test_voi_dataclass_to_from_dict(self):
        voi = SteamVOI(
            label="tumor",
            method="Bruker:STEAM",
            frame="bruker_scanner",
            position_mm=REF_POSITION.copy(),
            size_mm=REF_SIZE.copy(),
            orientation=REF_ORIENTATION.copy(),
            source={"kind": "bruker_method", "path": "/tmp/method", "expno": 34},
        )
        d = voi.to_dict()
        # Must be JSON-serializable
        s = json.dumps(d)
        roundtrip = SteamVOI.from_dict(json.loads(s))
        np.testing.assert_allclose(roundtrip.position_mm, REF_POSITION)
        np.testing.assert_allclose(roundtrip.size_mm, REF_SIZE)
        np.testing.assert_allclose(roundtrip.orientation, REF_ORIENTATION)
        assert roundtrip.label == "tumor"

    def test_study_save_load(self, tmp_path: Path):
        sv = StudyVOIs(
            subject_id="B16",
            scanner_to_patient=np.diag([1.0, -1.0, 1.0, 1.0]),
            voi=[
                SteamVOI(
                    label="tumor",
                    method="Bruker:STEAM",
                    frame="bruker_scanner",
                    position_mm=REF_POSITION.copy(),
                    size_mm=REF_SIZE.copy(),
                    orientation=REF_ORIENTATION.copy(),
                ),
                SteamVOI(
                    label="contralateral",
                    method="Bruker:STEAM",
                    frame="bruker_scanner",
                    position_mm=(REF_POSITION * np.array([-1, 1, 1])).copy(),
                    size_mm=REF_SIZE.copy(),
                    orientation=REF_ORIENTATION.copy(),
                ),
            ],
        )
        out = tmp_path / "steam_voi.json"
        save_voi_json(out, sv)

        loaded = load_voi_json(out)
        assert loaded.schema_version == 1
        assert loaded.subject_id == "B16"
        np.testing.assert_allclose(
            loaded.scanner_to_patient, np.diag([1.0, -1.0, 1.0, 1.0])
        )
        assert len(loaded.voi) == 2
        assert loaded.by_label("tumor") is not None
        assert loaded.by_label("contralateral") is not None
        assert loaded.by_label("nope") is None


# ---------------------------------------------------------------------------
# Geometry: voi_to_mask
# ---------------------------------------------------------------------------

class TestVoiToMask:

    def test_axis_aligned_cube_in_patient_frame(self):
        """An axis-aligned 4 mm cube at origin should produce a 4×4×4
        voxel mask on a 1 mm grid centered at origin."""
        # 21×21×21 grid, 1 mm spacing, centered so voxel (10,10,10) is origin
        geom = T1Geometry(
            shape=(21, 21, 21),
            origin=np.array([-10.0, -10.0, -10.0]),
            spacing=np.array([1.0, 1.0, 1.0]),
            direction=np.eye(3),
        )
        voi = SteamVOI(
            label="center",
            method="test",
            frame="patient_lps",
            position_mm=np.zeros(3),
            size_mm=np.array([4.0, 4.0, 4.0]),
            orientation=np.eye(3),
        )
        mask = voi_to_mask(voi, geom)
        # Volume should be ~4×4×4 voxels. With inclusive-bound semantics
        # the count is 5×5×5 because voxel centers at ±0, ±1, ±2 mm all
        # satisfy |q| ≤ 2.
        assert mask.shape == geom.shape
        assert mask.sum() == 5 * 5 * 5
        # Centered: voxel (10,10,10) is inside
        assert mask[10, 10, 10]
        # Far corner: outside
        assert not mask[0, 0, 0]

    def test_scanner_to_patient_flip(self):
        """A negative X flip in scanner_to_patient should move the mask
        to the mirror image of the un-flipped result."""
        geom = T1Geometry(
            shape=(21, 21, 21),
            origin=np.array([-10.0, -10.0, -10.0]),
            spacing=np.array([1.0, 1.0, 1.0]),
            direction=np.eye(3),
        )
        # VOI center at +3 mm in scanner X
        voi = SteamVOI(
            label="off-center",
            method="test",
            frame="bruker_scanner",
            position_mm=np.array([3.0, 0.0, 0.0]),
            size_mm=np.array([2.0, 2.0, 2.0]),
            orientation=np.eye(3),
        )
        mask_identity = voi_to_mask(voi, geom)
        flip_x = np.diag([-1.0, 1.0, 1.0, 1.0])
        mask_flipped = voi_to_mask(voi, geom, scanner_to_patient=flip_x)

        # Identity: VOI sits at patient X=+3 → voxel i=13 is center
        assert mask_identity[13, 10, 10]
        # Flipped: VOI sits at patient X=−3 → voxel i=7 is center
        assert mask_flipped[7, 10, 10]
        # The flip should be a mirror image about i=10
        np.testing.assert_array_equal(mask_identity, mask_flipped[::-1, :, :])

    def test_rotated_orientation(self):
        """A 45° rotation about Z should produce a cube extended along
        the XY diagonals: a point at (2.5, 0, 0) is OUTSIDE the
        axis-aligned 4 mm cube but INSIDE the rotated one (the rotated
        cube reaches √2·2 ≈ 2.83 mm along the X axis)."""
        # 0.1 mm grid, ±5 mm — fine enough that rasterization error is
        # small but still cheap.
        n = 101
        geom = T1Geometry(
            shape=(n, n, n),
            origin=np.array([-5.0, -5.0, -5.0]),
            spacing=np.full(3, 0.1),
            direction=np.eye(3),
        )
        c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
        R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

        voi_rot = SteamVOI(
            label="rot",
            method="test",
            frame="patient_lps",
            position_mm=np.zeros(3),
            size_mm=np.array([4.0, 4.0, 4.0]),
            orientation=R,
        )
        voi_ax = SteamVOI(
            label="ax",
            method="test",
            frame="patient_lps",
            position_mm=np.zeros(3),
            size_mm=np.array([4.0, 4.0, 4.0]),
            orientation=np.eye(3),
        )
        mask_rot = voi_to_mask(voi_rot, geom)
        mask_ax = voi_to_mask(voi_ax, geom)

        # Both masks non-empty and centered.
        assert mask_rot.sum() > 0
        assert mask_ax.sum() > 0
        center = n // 2
        assert mask_rot[center, center, center]
        assert mask_ax[center, center, center]

        # Point (2.5, 0, 0) → grid index (75, 50, 50).
        # Outside axis-aligned (X = 2.5 > size/2 = 2)
        # Inside rotated (rotated cube reaches √2·2 ≈ 2.83 along X)
        assert not mask_ax[75, 50, 50], "axis-aligned cube should not include (2.5, 0, 0)"
        assert mask_rot[75, 50, 50], "rotated cube should include (2.5, 0, 0)"

        # Point (0, 0, 2.5): outside both (rotation about Z doesn't help)
        assert not mask_ax[50, 50, 75]
        assert not mask_rot[50, 50, 75]

        # At a fine enough grid (0.1 mm vs. 4 mm cube edge), rasterized
        # volumes should agree to within a couple percent.
        rel_err = abs(mask_rot.sum() - mask_ax.sum()) / mask_ax.sum()
        assert rel_err < 0.05, (
            f"rotated mask volume {mask_rot.sum()} vs. unrotated "
            f"{mask_ax.sum()} (rel err {rel_err:.3f})"
        )

    def test_unknown_frame_raises(self):
        geom = T1Geometry(
            shape=(5, 5, 5),
            origin=np.zeros(3),
            spacing=np.ones(3),
            direction=np.eye(3),
        )
        voi = SteamVOI(
            label="bad",
            method="test",
            frame="martian_coordinates",
            position_mm=np.zeros(3),
            size_mm=np.ones(3),
            orientation=np.eye(3),
        )
        with pytest.raises(ValueError):
            voi_to_mask(voi, geom)


# ---------------------------------------------------------------------------
# Acquisition clustering / timepoint grouping
# ---------------------------------------------------------------------------

class TestClusterVoiAcquisitions:

    def _voi_dict(self, position, expno, size=(3, 3, 3), orient=None):
        return {
            "method": "Bruker:STEAM",
            "frame": "bruker_scanner",
            "position_mm": list(position),
            "size_mm": list(size),
            "orientation": (
                orient.tolist() if orient is not None else np.eye(3).tolist()
            ),
            "source": {"kind": "bruker_method", "expno": expno},
        }

    def test_empty_input(self):
        assert cluster_voi_acquisitions([]) == []

    def test_groups_duplicates_within_tolerance(self):
        vois = [
            self._voi_dict([1.0, 2.0, 3.0], expno=13),
            self._voi_dict([1.001, 2.0, 3.0], expno=15),  # within 0.1 mm
            self._voi_dict([5.0, 5.0, 5.0], expno=20),    # different
        ]
        clusters = cluster_voi_acquisitions(vois, tolerance_mm=0.1)
        assert len(clusters) == 2
        # First cluster contains the two close acquisitions
        first = clusters[0]
        assert len(first["members"]) == 2
        assert sorted(first["expnos"]) == [13, 15]
        # Cluster position is the mean of the members
        np.testing.assert_allclose(first["position_mm"], [1.0005, 2.0, 3.0])

    def test_does_not_merge_beyond_tolerance(self):
        vois = [
            self._voi_dict([0.0, 0.0, 0.0], expno=1),
            self._voi_dict([0.5, 0.0, 0.0], expno=2),  # outside 0.1 tolerance
        ]
        clusters = cluster_voi_acquisitions(vois, tolerance_mm=0.1)
        assert len(clusters) == 2

    def test_sort_order_by_expno(self):
        vois = [
            self._voi_dict([10.0, 0.0, 0.0], expno=30),
            self._voi_dict([0.0, 0.0, 0.0], expno=10),
            self._voi_dict([5.0, 0.0, 0.0], expno=20),
        ]
        clusters = cluster_voi_acquisitions(vois)
        first_expnos = [min(c["expnos"]) for c in clusters]
        assert first_expnos == sorted(first_expnos)


class TestClusterToVoi:

    def test_round_trips_position_and_geometry(self):
        cluster = {
            "position_mm": [1.0, 2.0, 3.0],
            "size_mm": [3.0, 3.0, 3.0],
            "orientation": np.eye(3).tolist(),
            "method": "Bruker:STEAM",
            "frame": "bruker_scanner",
            "members": [
                {"source": {"kind": "bruker_method", "expno": 13, "path": "/p/13/method"}},
                {"source": {"kind": "bruker_method", "expno": 15, "path": "/p/15/method"}},
            ],
            "expnos": [13, 15],
        }
        voi = cluster_to_voi(cluster, label="tumor")
        assert voi.label == "tumor"
        assert voi.frame == "bruker_scanner"
        np.testing.assert_allclose(voi.position_mm, [1.0, 2.0, 3.0])
        assert voi.source["expnos"] == [13, 15]
        assert voi.source["n_members"] == 2


class TestGroupClustersByTimepoint:

    def _make_cluster(self, expnos):
        return {"expnos": list(expnos)}

    def test_single_block(self):
        clusters = [self._make_cluster([9, 11]), self._make_cluster([12])]
        groups = group_clusters_by_timepoint(clusters, max_expno_gap=4)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_two_timepoints_split_on_large_gap(self):
        # Block 1: expnos 9-11; block 2: expnos 23-27 (gap of 12 > 4)
        clusters = [
            self._make_cluster([9, 11]),
            self._make_cluster([23, 25]),
            self._make_cluster([26, 27]),
        ]
        groups = group_clusters_by_timepoint(clusters, max_expno_gap=4)
        assert len(groups) == 2
        assert len(groups[0]) == 1
        assert len(groups[1]) == 2

    def test_empty_input(self):
        assert group_clusters_by_timepoint([]) == []


# ---------------------------------------------------------------------------
# Scanner-to-patient auto-detect
# ---------------------------------------------------------------------------

class TestAutoDetectTransform:

    def _make_t1_with_voi_anatomy(self, voi_center, grid_n=51, spacing=0.5):
        """Build a synthetic T1 volume that has 'anatomy' (high intensity)
        in a sphere centered at ``voi_center``. Returns (volume, geometry).
        """
        geom = T1Geometry(
            shape=(grid_n, grid_n, grid_n),
            origin=np.full(3, -(grid_n - 1) / 2 * spacing),
            spacing=np.full(3, spacing),
            direction=np.eye(3),
        )
        ii, jj, kk = np.meshgrid(
            np.arange(grid_n), np.arange(grid_n), np.arange(grid_n), indexing="ij"
        )
        # patient coords of each voxel
        px = geom.origin[0] + ii * spacing
        py = geom.origin[1] + jj * spacing
        pz = geom.origin[2] + kk * spacing
        radius = (
            (px - voi_center[0]) ** 2
            + (py - voi_center[1]) ** 2
            + (pz - voi_center[2]) ** 2
        )
        # Spherical anatomy with high intensity; air outside is low.
        volume = np.where(radius < 10.0**2, 1000.0, 100.0).astype(np.float32)
        return volume, geom

    def test_picks_correct_flip_when_y_is_flipped(self):
        """If the VOI is prescribed at +Y=5 in scanner coords but the
        anatomy actually sits at -Y=5 in patient coords (i.e. scanner_Y
        and patient_Y differ by sign), the auto-detect should pick the
        Y-flip transform."""
        anatomy_center_patient = np.array([0.0, -5.0, 0.0])
        t1, geom = self._make_t1_with_voi_anatomy(anatomy_center_patient)

        # VOI's prescribed position in scanner coords is +Y=5.
        voi = SteamVOI(
            label="probe",
            method="test",
            frame="bruker_scanner",
            position_mm=np.array([0.0, 5.0, 0.0]),
            size_mm=np.array([4.0, 4.0, 4.0]),
            orientation=np.eye(3),
        )
        best, scores = auto_detect_scanner_to_patient([voi], t1, geom)
        # The winning transform should map +Y in scanner to -Y in patient.
        # With the new permutation-aware result that means M[1, 1] is
        # negative (or some perm/sign combo that lands the VOI at -Y=5).
        assert best[1, 1] == -1.0
        # Scores are (perm, signs, score) tuples now — best > worst.
        assert scores[0][2] > scores[-1][2]

    def test_identity_when_signs_already_match(self):
        anatomy_center_patient = np.array([0.0, 5.0, 0.0])
        t1, geom = self._make_t1_with_voi_anatomy(anatomy_center_patient)
        voi = SteamVOI(
            label="probe",
            method="test",
            frame="bruker_scanner",
            position_mm=np.array([0.0, 5.0, 0.0]),
            size_mm=np.array([4.0, 4.0, 4.0]),
            orientation=np.eye(3),
        )
        best, _ = auto_detect_scanner_to_patient([voi], t1, geom)
        # Identity wins.
        np.testing.assert_allclose(best, np.eye(4))

    def test_ignores_patient_lps_vois(self):
        t1 = np.ones((11, 11, 11), dtype=np.float32) * 100.0
        geom = T1Geometry(
            shape=(11, 11, 11),
            origin=np.zeros(3),
            spacing=np.ones(3),
            direction=np.eye(3),
        )
        voi = SteamVOI(
            label="lps",
            method="test",
            frame="patient_lps",
            position_mm=np.zeros(3),
            size_mm=np.array([3.0, 3.0, 3.0]),
            orientation=np.eye(3),
        )
        best, scores = auto_detect_scanner_to_patient([voi], t1, geom)
        np.testing.assert_allclose(best, np.eye(4))
        assert len(scores) == 1


def test_voi_to_polygon_returns_outline_when_slice_intersects():
    geom = T1Geometry(
        shape=(11, 11, 11),
        origin=np.array([-5.0, -5.0, -5.0]),
        spacing=np.ones(3),
        direction=np.eye(3),
    )
    voi = SteamVOI(
        label="center",
        method="test",
        frame="patient_lps",
        position_mm=np.zeros(3),
        size_mm=np.array([3.0, 3.0, 3.0]),
        orientation=np.eye(3),
    )
    poly = voi_to_polygon(voi, z_index=5, geom=geom)
    assert poly is not None
    # Closed polygon
    np.testing.assert_array_equal(poly[0], poly[-1])
    # Within bounds
    assert poly.shape == (5, 2)


def test_voi_to_polygon_returns_none_when_slice_misses():
    geom = T1Geometry(
        shape=(11, 11, 11),
        origin=np.array([-5.0, -5.0, -5.0]),
        spacing=np.ones(3),
        direction=np.eye(3),
    )
    voi = SteamVOI(
        label="far",
        method="test",
        frame="patient_lps",
        position_mm=np.array([0.0, 0.0, 20.0]),  # way out of grid
        size_mm=np.array([1.0, 1.0, 1.0]),
        orientation=np.eye(3),
    )
    poly = voi_to_polygon(voi, z_index=5, geom=geom)
    assert poly is None
