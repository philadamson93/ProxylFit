"""
Qt smoke tests for SteamVoiLoaderDialog.

Verifies the dialog can be instantiated, scans a synthetic Bruker tree,
populates the acquisitions table, and writes a valid ``steam_voi.json``
when the workflow runs end-to-end programmatically.

These tests require PySide6 + pytest-qt; they're skipped automatically
if either is unavailable.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from proxyl_analysis.steam_voi import T1Geometry, load_voi_json
from proxyl_analysis.ui.steam_voi import SteamVoiLoaderDialog


REF_POSITION_1 = np.array([1.0, 2.0, 3.0])
REF_POSITION_2 = np.array([-1.0, 2.0, 3.0])
REF_SIZE = np.array([3.0, 3.0, 3.0])
REF_ORIENT = np.eye(3)


def _write_steam_method(path: Path, pos, size, orient) -> None:
    content = (
        "##$Method=<Bruker:STEAM>\n"
        "##$PVM_VoxArrSize=( 1, 3 )\n"
        f"{size[0]} {size[1]} {size[2]}\n"
        "##$PVM_VoxArrPosition=( 1, 3 )\n"
        f"{pos[0]:.17f} {pos[1]:.17f} {pos[2]:.17f}\n"
        "##$PVM_VoxArrGradOrient=( 1, 3, 3 )\n"
        f"{orient[0,0]:.17f} {orient[0,1]:.17f} {orient[0,2]:.17f}\n"
        f"{orient[1,0]:.17f} {orient[1,1]:.17f} {orient[1,2]:.17f}\n"
        f"{orient[2,0]:.17f} {orient[2,1]:.17f} {orient[2,2]:.17f}\n"
        "##END=\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def synthetic_subject(tmp_path: Path) -> Path:
    """Two-timepoint Bruker subject: 4 ExpNos forming 2 timepoints
    with 2 STEAM voxels each."""
    study = tmp_path / "subj"
    study.mkdir()
    (study / "subject").write_text(
        "##$SUBJECT_id=( 60 )\n<TestSubject>\n##END=\n"
    )
    # Timepoint 1 — ExpNos 9, 10
    _write_steam_method(study / "9" / "method", REF_POSITION_1, REF_SIZE, REF_ORIENT)
    _write_steam_method(study / "10" / "method", REF_POSITION_2, REF_SIZE, REF_ORIENT)
    # Timepoint 2 (gap of 10) — ExpNos 20, 21
    _write_steam_method(study / "20" / "method", REF_POSITION_1, REF_SIZE, REF_ORIENT)
    _write_steam_method(study / "21" / "method", REF_POSITION_2, REF_SIZE, REF_ORIENT)
    return study


@pytest.fixture
def t1_volume_and_geom():
    """A synthetic T1 cube whose anatomy sits at both reference VOI
    positions, so auto-detect picks the identity transform."""
    n = 41
    spacing = 0.5
    geom = T1Geometry(
        shape=(n, n, n),
        origin=np.full(3, -(n - 1) / 2 * spacing),
        spacing=np.full(3, spacing),
        direction=np.eye(3),
    )
    vol = np.full((n, n, n), 100.0, dtype=np.float32)
    # Mark anatomy at both VOI locations
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    px = geom.origin[0] + ii * spacing
    py = geom.origin[1] + jj * spacing
    pz = geom.origin[2] + kk * spacing
    for ref in (REF_POSITION_1, REF_POSITION_2):
        dist = (px - ref[0]) ** 2 + (py - ref[1]) ** 2 + (pz - ref[2]) ** 2
        vol[dist < 4.0] = 1000.0
    return vol, geom


class TestSteamVoiLoaderDialog:

    def test_constructs_with_empty_state(self, qtbot, t1_volume_and_geom, tmp_path):
        vol, geom = t1_volume_and_geom
        dlg = SteamVoiLoaderDialog(vol, geom, output_dir=tmp_path)
        qtbot.addWidget(dlg)
        assert dlg.tumor_voi is None
        assert dlg.contralateral_voi is None
        assert not dlg.save_btn.isEnabled()

    def test_scan_populates_table_and_combos(
        self, qtbot, t1_volume_and_geom, tmp_path, synthetic_subject
    ):
        vol, geom = t1_volume_and_geom
        dlg = SteamVoiLoaderDialog(
            vol, geom, output_dir=tmp_path, bruker_root_hint=synthetic_subject
        )
        qtbot.addWidget(dlg)
        # Two timepoints (gap > 4), each with 2 unique positions = 2 clusters
        assert len(dlg.timepoint_groups) == 2
        assert dlg.timepoint_combo.count() == 2
        # Table row count = total clusters across timepoints (2 + 2)
        assert dlg.table.rowCount() == 4
        # Defaults assign tumor + contralateral on first timepoint
        assert dlg.tumor_voi is not None
        assert dlg.contralateral_voi is not None
        assert dlg.tumor_voi.label == "tumor"
        assert dlg.contralateral_voi.label == "contralateral"
        assert dlg.save_btn.isEnabled()

    def test_save_writes_json(
        self, qtbot, t1_volume_and_geom, tmp_path, synthetic_subject
    ):
        vol, geom = t1_volume_and_geom
        dlg = SteamVoiLoaderDialog(
            vol, geom, output_dir=tmp_path, bruker_root_hint=synthetic_subject
        )
        qtbot.addWidget(dlg)
        dlg._on_save()  # bypass user click but go through full save path

        out = tmp_path / "steam_voi.json"
        assert out.is_file()
        loaded = load_voi_json(out)
        assert loaded.subject_id is None or isinstance(loaded.subject_id, str)
        assert len(loaded.voi) == 2
        labels = {v.label for v in loaded.voi}
        assert labels == {"tumor", "contralateral"}
        assert dlg.result_vois is not None
