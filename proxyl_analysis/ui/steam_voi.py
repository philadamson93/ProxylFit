"""
STEAM VOI loader dialog.

User-facing modal for loading prescribed STEAM voxels from a Bruker raw
tree (or a pre-existing ``steam_voi.json``) and assigning tumor vs.
contralateral labels. The dialog writes ``<output_dir>/steam_voi.json``
on save and exposes the chosen ``StudyVOIs`` via :attr:`result_vois`.

Workflow
--------
1. User points at a Bruker subject folder (or a pre-existing JSON).
2. Dialog scans the tree, clusters acquisitions by position, and
   groups clusters into timepoint blocks.
3. User picks a timepoint and assigns which cluster is tumor / which
   is contralateral.
4. Auto-detect button finds the best axis-flip
   ``scanner_to_patient`` transform by overlap with T1 anatomy.
5. Preview shows both VOIs overlaid on the T1 mid-slice; user can
   z-scroll to verify alignment.
6. Save & Use writes the JSON and closes; the parent dialog reads
   :attr:`result_vois` for downstream use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon

from ..steam_voi import (
    StudyVOIs,
    SteamVOI,
    T1Geometry,
    auto_detect_scanner_to_patient,
    cluster_to_voi,
    cluster_voi_acquisitions,
    group_clusters_by_timepoint,
    load_voi_json,
    parse_bruker_method,
    save_voi_json,
    scan_bruker_study,
    voi_to_polygon,
)
from .styles import PROXYLFIT_STYLE


class SteamVoiLoaderDialog(QDialog):
    """Modal dialog for loading and labeling STEAM VOIs.

    Parameters
    ----------
    t1_volume : np.ndarray, shape ``(nx, ny, nz)``
        T1 volume used for overlay preview and transform auto-detect.
    t1_geometry : T1Geometry
        DICOM-derived voxel-grid geometry of ``t1_volume``.
    output_dir : Path or str
        Per-subject output directory (e.g. ``output/{dataset}/``).
        ``steam_voi.json`` is written here on Save.
    parent : QWidget, optional
    bruker_root_hint : Path or str, optional
        If provided, the Bruker source path is pre-filled with this
        value and the initial scan runs automatically.
    existing_json : Path or str, optional
        If a ``steam_voi.json`` already exists at this path, it is
        loaded as the dialog's initial state.
    """

    def __init__(
        self,
        t1_volume: np.ndarray,
        t1_geometry: T1Geometry,
        output_dir,
        parent=None,
        bruker_root_hint=None,
        existing_json=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("ProxylFit — STEAM VOIs")
        self.setMinimumSize(1100, 750)

        self.t1_volume = t1_volume
        self.t1_geometry = t1_geometry
        self.output_dir = Path(output_dir)

        # State
        self.clusters: List[Dict[str, Any]] = []
        self.timepoint_groups: List[List[Dict[str, Any]]] = []
        self.scanner_to_patient = np.eye(4)
        self.tumor_voi: Optional[SteamVOI] = None
        self.contralateral_voi: Optional[SteamVOI] = None
        self.subject_id: Optional[str] = None

        # Output (set on accept)
        self.result_vois: Optional[StudyVOIs] = None

        self._setup_ui()
        self.setStyleSheet(PROXYLFIT_STYLE)

        # Seed initial state. Always populate the Bruker tree field
        # when a hint is provided — even when a previously-saved JSON
        # is being loaded — so the user can re-scan or pick a different
        # subject without having to re-browse to the tree.
        if bruker_root_hint:
            self.bruker_path_edit.setText(str(bruker_root_hint))
        if existing_json and Path(existing_json).is_file():
            self._load_existing_json(Path(existing_json))
        elif bruker_root_hint:
            self._on_scan_bruker()

        self._update_preview()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(15, 15, 15, 15)

        # Title
        title = QLabel("STEAM VOI Loader")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)

        # Section 1: source picker
        layout.addWidget(self._build_source_section())

        # Section 2: acquisitions table
        layout.addWidget(self._build_acquisitions_section(), stretch=1)

        # Two-column: assignment + transform / preview
        bottom = QHBoxLayout()
        bottom.addWidget(self._build_assignment_section(), stretch=1)
        bottom.addWidget(self._build_preview_section(), stretch=2)
        layout.addLayout(bottom, stretch=2)

        # Action buttons
        layout.addLayout(self._build_action_buttons())

    def _build_source_section(self) -> QWidget:
        group = QGroupBox("Source")
        gl = QHBoxLayout(group)
        gl.addWidget(QLabel("Bruker subject folder:"))
        self.bruker_path_edit = QLineEdit()
        self.bruker_path_edit.setPlaceholderText(
            "Path to a Bruker subject folder containing numbered ExpNo dirs"
        )
        gl.addWidget(self.bruker_path_edit, stretch=1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse_bruker)
        gl.addWidget(browse)
        scan = QPushButton("Scan")
        scan.clicked.connect(self._on_scan_bruker)
        gl.addWidget(scan)
        load_json = QPushButton("Load JSON…")
        load_json.setToolTip("Load a previously-saved steam_voi.json")
        load_json.clicked.connect(self._on_browse_json)
        gl.addWidget(load_json)
        return group

    def _build_acquisitions_section(self) -> QWidget:
        group = QGroupBox("STEAM Acquisitions")
        gl = QVBoxLayout(group)
        self.status_label = QLabel(
            "No acquisitions loaded. Choose a Bruker subject folder above and click Scan."
        )
        self.status_label.setStyleSheet("color: #666;")
        gl.addWidget(self.status_label)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Timepoint", "Cluster", "ExpNos", "Position (mm)", "Size (mm)", "Acqs"]
        )
        self.table.horizontalHeader().setStretchLastSection(False)
        gl.addWidget(self.table, stretch=1)
        return group

    def _build_assignment_section(self) -> QWidget:
        group = QGroupBox("Tumor / Contralateral")
        gl = QVBoxLayout(group)

        # Timepoint selector
        tp_row = QHBoxLayout()
        tp_row.addWidget(QLabel("Timepoint:"))
        self.timepoint_combo = QComboBox()
        self.timepoint_combo.currentIndexChanged.connect(
            self._on_timepoint_changed
        )
        tp_row.addWidget(self.timepoint_combo, stretch=1)
        gl.addLayout(tp_row)

        # Tumor / contralateral combos
        for label, attr in (("Tumor", "tumor_combo"), ("Contralateral", "contra_combo")):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{label}:"))
            combo = QComboBox()
            combo.currentIndexChanged.connect(self._on_assignment_changed)
            row.addWidget(combo, stretch=1)
            gl.addLayout(row)
            setattr(self, attr, combo)

        gl.addSpacing(8)

        # Scanner-to-patient transform. Each VOI can carry its own
        # override; the "Adjust transform for" combo selects which
        # transform the order/flip controls operate on.
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Adjust transform for:"))
        self.transform_target_combo = QComboBox()
        self.transform_target_combo.addItem("Both VOIs (shared)", "both")
        self.transform_target_combo.addItem("Tumor only", "tumor")
        self.transform_target_combo.addItem("Contralateral only", "contralateral")
        self.transform_target_combo.currentIndexChanged.connect(
            self._on_transform_target_changed
        )
        target_row.addWidget(self.transform_target_combo, stretch=1)
        gl.addLayout(target_row)

        gl.addWidget(QLabel("<b>Scanner → Patient transform</b>"))
        self.transform_label = QLabel("Identity (no axis flips)")
        self.transform_label.setStyleSheet("font-family: monospace;")
        gl.addWidget(self.transform_label)

        # Manual axis-order + flip controls. Bruker scanner XYZ axes
        # are not always aligned with DICOM LPS XYZ: the gradient axis
        # assignment (PVM_VoxExcOrder, e.g. ``AP_RL_HF``) determines
        # which permutation maps Bruker → patient. The user toggles
        # order + signs live and watches the preview update.
        from ..steam_voi import _AXIS_PERMS, _PERM_LABELS
        order_row = QHBoxLayout()
        order_row.addWidget(QLabel("Axis order:"))
        self.order_combo = QComboBox()
        for perm in _AXIS_PERMS:
            self.order_combo.addItem(_PERM_LABELS[perm], perm)
        self.order_combo.currentIndexChanged.connect(
            self._on_flip_check_changed
        )
        order_row.addWidget(self.order_combo)
        order_row.addStretch()
        gl.addLayout(order_row)

        flip_row = QHBoxLayout()
        flip_row.addWidget(QLabel("Flip axes:"))
        self.flip_checks = {}
        for axis in ("X", "Y", "Z"):
            cb = QCheckBox(axis)
            cb.toggled.connect(self._on_flip_check_changed)
            flip_row.addWidget(cb)
            self.flip_checks[axis] = cb
        flip_row.addStretch()
        gl.addLayout(flip_row)

        auto = QPushButton("Auto-detect")
        auto.setToolTip(
            "Try all axis-flip combinations and pick the one whose VOIs "
            "best overlap T1 anatomy."
        )
        auto.clicked.connect(self._on_auto_detect)
        gl.addWidget(auto)

        gl.addStretch()
        return group

    def _on_flip_check_changed(self, _checked=False):
        """Live-update the targeted transform from axis-order + flips."""
        from ..steam_voi import _signed_perm_matrix
        signs = tuple(
            -1 if self.flip_checks[a].isChecked() else 1 for a in ("X", "Y", "Z")
        )
        perm = self.order_combo.currentData() or (0, 1, 2)
        M = _signed_perm_matrix(perm, signs)
        self._apply_transform_to_target(M)
        self._refresh_transform_label()
        self._update_preview()

    def _apply_transform_to_target(self, M: np.ndarray):
        """Apply ``M`` to whichever VOI(s) the target combo points at."""
        target = self.transform_target_combo.currentData() if hasattr(
            self, "transform_target_combo"
        ) else "both"
        if target == "tumor":
            if self.tumor_voi is not None:
                self.tumor_voi.transform = M.copy()
        elif target == "contralateral":
            if self.contralateral_voi is not None:
                self.contralateral_voi.transform = M.copy()
        else:  # both / shared
            self.scanner_to_patient = M
            # Clear any per-VOI overrides so the global takes effect.
            for v in (self.tumor_voi, self.contralateral_voi):
                if v is not None:
                    v.transform = None

    def _on_transform_target_changed(self, _index: int):
        """When the user picks a different target, sync the controls
        to that target's current transform."""
        self._refresh_transform_label()

    def _build_preview_section(self) -> QWidget:
        group = QGroupBox("T1 Overlay Preview")
        gl = QVBoxLayout(group)

        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.fig.set_facecolor("#f5f5f5")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        gl.addWidget(self.canvas, stretch=1)

        z_row = QHBoxLayout()
        z_row.addWidget(QLabel("Z-slice:"))
        self.z_spin = QSpinBox()
        self.z_spin.setRange(0, max(0, self.t1_geometry.shape[2] - 1))
        self.z_spin.setValue(self.t1_geometry.shape[2] // 2)
        self.z_spin.valueChanged.connect(lambda _v: self._update_preview())
        z_row.addWidget(self.z_spin)
        z_row.addStretch()
        gl.addLayout(z_row)
        return group

    def _build_action_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setObjectName("cancelButton")
        cancel.clicked.connect(self.reject)
        row.addWidget(cancel)
        self.save_btn = QPushButton("Save && Use")
        self.save_btn.setObjectName("acceptButton")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save)
        row.addWidget(self.save_btn)
        return row

    # ------------------------------------------------------------------
    # Source actions
    # ------------------------------------------------------------------

    def _on_browse_bruker(self):
        d = QFileDialog.getExistingDirectory(
            self, "Pick Bruker subject folder", self.bruker_path_edit.text() or ""
        )
        if d:
            self.bruker_path_edit.setText(d)

    def _on_browse_json(self):
        f, _ = QFileDialog.getOpenFileName(
            self,
            "Load steam_voi.json",
            str(self.output_dir),
            "JSON Files (*.json)",
        )
        if f:
            self._load_existing_json(Path(f))

    def _on_scan_bruker(self):
        path = Path(self.bruker_path_edit.text())
        if not path.is_dir():
            QMessageBox.warning(self, "Not found", f"Folder does not exist:\n{path}")
            return
        raw = scan_bruker_study(path)
        if not raw:
            QMessageBox.information(
                self,
                "No STEAM",
                f"No STEAM acquisitions found under {path}.\n\n"
                "Looked at every numbered ExpNo subdirectory for "
                "##$Method=<Bruker:STEAM>.",
            )
            self.clusters = []
            self.timepoint_groups = []
            self._refresh_table()
            self._refresh_combos()
            return
        self.clusters = cluster_voi_acquisitions(raw, tolerance_mm=0.1)
        self.timepoint_groups = group_clusters_by_timepoint(
            self.clusters, max_expno_gap=4
        )
        self._refresh_table()
        self._refresh_combos()
        self._update_preview()

    def _load_existing_json(self, path: Path):
        try:
            sv = load_voi_json(path)
        except Exception as e:
            QMessageBox.warning(self, "Load failed", f"Could not read {path}:\n{e}")
            return
        self.scanner_to_patient = sv.scanner_to_patient
        self.subject_id = sv.subject_id
        self.tumor_voi = sv.by_label("tumor")
        self.contralateral_voi = sv.by_label("contralateral")
        self._refresh_transform_label()
        self._update_preview()
        self.status_label.setText(
            f"Loaded existing JSON: {path}  "
            f"(subject: {self.subject_id or 'unknown'}, "
            f"{len(sv.voi)} VOI(s))"
        )
        self.save_btn.setEnabled(
            self.tumor_voi is not None or self.contralateral_voi is not None
        )

    # ------------------------------------------------------------------
    # Table + combo refresh
    # ------------------------------------------------------------------

    def _refresh_table(self):
        # Flatten clusters with timepoint annotations
        rows: List[Dict[str, Any]] = []
        for ti, group in enumerate(self.timepoint_groups):
            for ci, cluster in enumerate(group):
                rows.append({"tp": ti, "ci": ci, "cluster": cluster})

        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            c = r["cluster"]
            pos = c["position_mm"]
            sz = c["size_mm"]
            items = [
                QTableWidgetItem(f"T{r['tp'] + 1}"),
                QTableWidgetItem(chr(ord("A") + r["ci"])),
                QTableWidgetItem(", ".join(str(e) for e in c["expnos"])),
                QTableWidgetItem(
                    f"({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})"
                ),
                QTableWidgetItem(f"{sz[0]:.1f}×{sz[1]:.1f}×{sz[2]:.1f}"),
                QTableWidgetItem(str(len(c["members"]))),
            ]
            for col, it in enumerate(items):
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(i, col, it)
        self.table.resizeColumnsToContents()

        total_clusters = sum(len(g) for g in self.timepoint_groups)
        total_acqs = sum(
            len(c["members"]) for g in self.timepoint_groups for c in g
        )
        if total_clusters:
            self.status_label.setText(
                f"Found {total_acqs} acquisitions → {total_clusters} unique "
                f"VOIs → {len(self.timepoint_groups)} timepoint(s)"
            )

    def _refresh_combos(self):
        self.timepoint_combo.blockSignals(True)
        self.timepoint_combo.clear()
        for i, group in enumerate(self.timepoint_groups):
            expnos = sorted(e for c in group for e in c["expnos"])
            label = f"T{i + 1}   (ExpNos {expnos[0]}–{expnos[-1]}, {len(group)} VOIs)"
            self.timepoint_combo.addItem(label, i)
        self.timepoint_combo.blockSignals(False)
        if self.timepoint_groups:
            self.timepoint_combo.setCurrentIndex(0)
            self._on_timepoint_changed(0)

    def _on_timepoint_changed(self, _ix: int):
        tp_idx = self.timepoint_combo.currentData()
        if tp_idx is None or tp_idx >= len(self.timepoint_groups):
            return
        clusters = self.timepoint_groups[tp_idx]
        for combo in (self.tumor_combo, self.contra_combo):
            combo.blockSignals(True)
            combo.clear()
            for ci, c in enumerate(clusters):
                pos = c["position_mm"]
                label = (
                    f"{chr(ord('A') + ci)}  "
                    f"({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})  "
                    f"[ExpNos {','.join(str(e) for e in c['expnos'])}]"
                )
                combo.addItem(label, ci)
            combo.blockSignals(False)
        # Sensible defaults: cluster A=tumor, cluster B=contralateral.
        if self.tumor_combo.count() > 0:
            self.tumor_combo.setCurrentIndex(0)
        if self.contra_combo.count() > 1:
            self.contra_combo.setCurrentIndex(1)
        elif self.contra_combo.count() == 1:
            self.contra_combo.setCurrentIndex(0)
        self._on_assignment_changed()

    def _on_assignment_changed(self):
        tp_idx = self.timepoint_combo.currentData()
        if tp_idx is None or tp_idx >= len(self.timepoint_groups):
            return
        clusters = self.timepoint_groups[tp_idx]
        tumor_ci = self.tumor_combo.currentData()
        contra_ci = self.contra_combo.currentData()

        self.tumor_voi = (
            cluster_to_voi(clusters[tumor_ci], "tumor")
            if tumor_ci is not None and tumor_ci < len(clusters)
            else None
        )
        self.contralateral_voi = (
            cluster_to_voi(clusters[contra_ci], "contralateral")
            if contra_ci is not None and contra_ci < len(clusters)
            else None
        )
        self.save_btn.setEnabled(
            self.tumor_voi is not None and self.contralateral_voi is not None
        )
        self._update_preview()

    # ------------------------------------------------------------------
    # Transform auto-detect
    # ------------------------------------------------------------------

    def _on_auto_detect(self):
        # Auto-detect operates on whichever target the user has selected.
        # When target=both, optimize against the union of all VOIs as
        # before. When tumor or contralateral, optimize that VOI alone
        # and write the result into that VOI's per-VOI transform.
        target = self.transform_target_combo.currentData() if hasattr(
            self, "transform_target_combo"
        ) else "both"
        if target == "tumor":
            active = [self.tumor_voi] if self.tumor_voi else []
        elif target == "contralateral":
            active = [self.contralateral_voi] if self.contralateral_voi else []
        else:
            active = [v for v in (self.tumor_voi, self.contralateral_voi) if v]
        if not active:
            QMessageBox.information(
                self,
                "No VOIs",
                "Assign tumor / contralateral first, then auto-detect.",
            )
            return
        best, scores = auto_detect_scanner_to_patient(
            active, self.t1_volume, self.t1_geometry
        )
        from ..steam_voi import _PERM_LABELS, _perm_from_matrix

        # Show top-5 in a small confirmation.
        top = scores[:5]
        report = "\n".join(
            f"  order={_PERM_LABELS[perm]}  flips=({sx:+d},{sy:+d},{sz:+d})"
            f"   score={s * 100:5.1f}%"
            for perm, (sx, sy, sz), s in top
        )
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Auto-detect transform")
        msg.setText("Best scanner→patient transform found:")
        best_perm, best_signs = _perm_from_matrix(best)
        msg.setInformativeText(
            f"Axis order: {_PERM_LABELS[best_perm]}   "
            f"Flips: ({best_signs[0]:+d}, {best_signs[1]:+d}, {best_signs[2]:+d})\n\n"
            f"Top candidates (anatomy overlap × intensity):\n{report}\n\n"
            "Apply this transform?"
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        if msg.exec() == QMessageBox.Yes:
            self._apply_transform_to_target(best)
            self._refresh_transform_label()
            self._update_preview()

    def _refresh_transform_label(self):
        """Read whichever transform the target combo currently selects,
        and reflect it in the label + control state."""
        from ..steam_voi import _PERM_LABELS, _perm_from_matrix
        target = self.transform_target_combo.currentData() if hasattr(
            self, "transform_target_combo"
        ) else "both"
        if target == "tumor" and self.tumor_voi is not None:
            current = self.tumor_voi.effective_transform(self.scanner_to_patient)
        elif target == "contralateral" and self.contralateral_voi is not None:
            current = self.contralateral_voi.effective_transform(
                self.scanner_to_patient
            )
        else:
            current = self.scanner_to_patient
        perm, signs = _perm_from_matrix(current)
        order = _PERM_LABELS.get(perm, "?")
        if perm == (0, 1, 2) and signs == (1, 1, 1):
            self.transform_label.setText("Identity (no permutation, no flips)")
        else:
            flipped = [a for a, s in zip("XYZ", signs) if s < 0]
            self.transform_label.setText(
                f"Order: {order}   "
                f"Flips: ({signs[0]:+d}, {signs[1]:+d}, {signs[2]:+d}) "
                f"— flipped axes: {','.join(flipped) or 'none'}"
            )
        # Keep manual controls in sync without re-triggering handlers.
        if hasattr(self, "flip_checks"):
            for axis, s in zip("XYZ", signs):
                cb = self.flip_checks[axis]
                cb.blockSignals(True)
                cb.setChecked(s < 0)
                cb.blockSignals(False)
        if hasattr(self, "order_combo"):
            for i in range(self.order_combo.count()):
                if self.order_combo.itemData(i) == perm:
                    self.order_combo.blockSignals(True)
                    self.order_combo.setCurrentIndex(i)
                    self.order_combo.blockSignals(False)
                    break

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _update_preview(self):
        self.ax.clear()
        self.ax.axis("off")
        z = int(self.z_spin.value())
        if 0 <= z < self.t1_volume.shape[2]:
            sl = self.t1_volume[:, :, z].T
            self.ax.imshow(sl, cmap="gray", origin="lower")
            self.ax.set_title(f"T1 slice z={z}", fontsize=10)

        for voi, color in (
            (self.tumor_voi, "red"),
            (self.contralateral_voi, "cyan"),
        ):
            if voi is None:
                continue
            transform = voi.effective_transform(self.scanner_to_patient)
            poly = voi_to_polygon(
                voi, z, self.t1_geometry, scanner_to_patient=transform
            )
            if poly is None:
                continue
            # voi_to_polygon returns grid (i, j) where i is numpy axis 0
            # (X) and j is numpy axis 1 (Y). The slice is displayed via
            # ``imshow(slice.T, origin='lower')`` so array element [j,i]
            # appears at matplotlib data coords (x=i, y=j). Use the
            # vertices as-is — no axis swap.
            patch = MplPolygon(
                poly, closed=True, fill=False, edgecolor=color, linewidth=2
            )
            self.ax.add_patch(patch)
            self.ax.text(
                poly[0, 0], poly[0, 1] - 1, voi.label,
                color=color, fontsize=9, fontweight="bold",
            )

        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Save & accept
    # ------------------------------------------------------------------

    def _on_save(self):
        if self.tumor_voi is None or self.contralateral_voi is None:
            QMessageBox.warning(
                self, "Incomplete", "Assign both tumor and contralateral first."
            )
            return
        sv = StudyVOIs(
            schema_version=1,
            subject_id=self.subject_id,
            scanner_to_patient=self.scanner_to_patient,
            voi=[self.tumor_voi, self.contralateral_voi],
        )
        out_path = self.output_dir / "steam_voi.json"
        try:
            save_voi_json(out_path, sv)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"{e}")
            return
        self.result_vois = sv
        self.accept()
