"""
Main workflow menu dialog for ProxylFit.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame,
    QGroupBox, QSpinBox, QRadioButton, QFileDialog, QMessageBox,
    QScrollArea, QWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QLineEdit
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor

from .styles import init_qt_app
from .components import HeaderWidget, ButtonBar


class DicomScanResultsDialog(QDialog):
    """Dialog to display DICOM scan results with ability to load selected series."""

    # Color constants
    COLOR_PROXYL_VALID = QColor(220, 240, 220)      # Light green for valid PROXYL
    COLOR_PROXYL_SELECTED = QColor(100, 200, 100)   # Bright green for selected PROXYL
    COLOR_T2_VALID = QColor(220, 220, 240)          # Light blue for valid T2
    COLOR_T2_SELECTED = QColor(100, 150, 220)       # Bright blue for selected T2
    COLOR_DEFAULT = QColor(255, 255, 255)           # White for other rows

    def __init__(self, scan_results: list, folder_path: str, parent=None):
        super().__init__(parent)
        self.scan_results = scan_results
        self.folder_path = folder_path
        self.result = None  # Will hold the load action result

        # Filter series by type
        self.proxyl_series = [s for s in scan_results if s.get('is_proxyl')]
        self.t2_series = [s for s in scan_results if s.get('is_t2')]

        # Map sample_file paths to row indices for highlighting
        self.path_to_row = {}

        # STEAM VOI state — populated when user picks a Bruker tree.
        # Maps subject_id → {
        #     "bruker_root": Path,
        #     "subject_folder": Path,
        #     "subject_meta": dict,
        #     "clusters": list,
        # }
        self.steam_index: dict = {}
        self.bruker_root_path: str = ""

        self.setWindowTitle("DICOM Scan Results")
        self.setMinimumSize(900, 600)
        self.resize(1000, 700)

        self._setup_ui()
        self._update_row_highlights()

    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)

        # Header
        header = HeaderWidget("DICOM Scan Results", f"Found {len(self.scan_results)} series")
        layout.addWidget(header)

        # Summary
        summary_label = QLabel(
            f"PROXYL series: {len(self.proxyl_series)}  |  "
            f"T2 series: {len(self.t2_series)}  |  "
            f"Folder: {self.folder_path}"
        )
        summary_label.setStyleSheet("color: #666; font-size: 11px;")
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)

        # Table
        self.table = QTableWidget()
        columns = ['Series#', 'Description', 'Size', 'Slices', 'Type', 'Study Date', 'File']
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setRowCount(len(self.scan_results))

        for row, s in enumerate(self.scan_results):
            # Store path to row mapping
            sample_file = s.get('sample_file', '')
            if sample_file:
                self.path_to_row[sample_file] = row

            # Determine type string
            type_str = ''
            if s.get('is_proxyl'):
                type_str = 'PROXYL'
            elif s.get('is_t2'):
                type_str = 'T2'

            # Create items (colors will be set by _update_row_highlights)
            items = [
                QTableWidgetItem(str(s.get('series_number', ''))),
                QTableWidgetItem(s.get('series_description', '')[:50]),
                QTableWidgetItem(f"{s.get('rows', 0)}x{s.get('cols', 0)}"),
                QTableWidgetItem(str(s.get('num_slices', 0))),
                QTableWidgetItem(type_str),
                QTableWidgetItem(s.get('study_date', '')),
                QTableWidgetItem(Path(sample_file).name if sample_file else '')
            ]

            for col, item in enumerate(items):
                self.table.setItem(row, col, item)

        # Resize columns
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        for i in [0, 2, 3, 4, 5]:
            self.table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)

        layout.addWidget(self.table)

        # Load Selection section
        load_group = QGroupBox("Load Selection")
        load_layout = QVBoxLayout(load_group)

        # T1 (PROXYL) selector
        t1_layout = QHBoxLayout()
        t1_label = QLabel("T1 (PROXYL):")
        t1_label.setMinimumWidth(80)
        t1_layout.addWidget(t1_label)

        self.t1_combo = QComboBox()
        self.t1_combo.addItem("-- None --", None)
        for s in self.proxyl_series:
            desc = s.get('series_description', 'Unknown')[:40]
            series_num = s.get('series_number', 0)
            frames = s.get('num_frames', 0)
            label = f"{desc} (series {series_num}, {frames} frames)"
            self.t1_combo.addItem(label, s.get('sample_file'))
        if self.proxyl_series:
            self.t1_combo.setCurrentIndex(1)  # Select first PROXYL
        self.t1_combo.currentIndexChanged.connect(self._update_row_highlights)
        t1_layout.addWidget(self.t1_combo, stretch=1)
        load_layout.addLayout(t1_layout)

        # T2 selector
        t2_layout = QHBoxLayout()
        t2_label = QLabel("T2:")
        t2_label.setMinimumWidth(80)
        t2_layout.addWidget(t2_label)

        self.t2_combo = QComboBox()
        self.t2_combo.addItem("-- None --", None)
        for s in self.t2_series:
            desc = s.get('series_description', 'Unknown')[:40]
            series_num = s.get('series_number', 0)
            slices = s.get('num_slices', 0)
            label = f"{desc} (series {series_num}, {slices} slices)"
            self.t2_combo.addItem(label, s.get('sample_file'))
        if self.t2_series:
            self.t2_combo.setCurrentIndex(1)  # Select first T2
        self.t2_combo.currentIndexChanged.connect(self._update_row_highlights)
        t2_layout.addWidget(self.t2_combo, stretch=1)
        load_layout.addLayout(t2_layout)

        # STEAM VOI selector (optional). Points at a Bruker raw tree
        # whose per-subject folders are matched against the DICOM
        # PROXYL series. See T024 for design rationale.
        steam_path_layout = QHBoxLayout()
        steam_path_label = QLabel("Bruker tree:")
        steam_path_label.setMinimumWidth(80)
        steam_path_label.setToolTip(
            "Optional. Root folder containing one or more per-subject "
            "Bruker ParaVision study folders. Used to extract STEAM VOIs."
        )
        steam_path_layout.addWidget(steam_path_label)
        self.bruker_root_edit = QLineEdit()
        self.bruker_root_edit.setPlaceholderText(
            "Optional — path to Bruker root containing per-subject folders"
        )
        steam_path_layout.addWidget(self.bruker_root_edit, stretch=1)
        bruker_browse = QPushButton("Browse…")
        bruker_browse.clicked.connect(self._on_browse_bruker_root)
        steam_path_layout.addWidget(bruker_browse)
        bruker_scan = QPushButton("Scan")
        bruker_scan.clicked.connect(self._on_scan_bruker_root)
        steam_path_layout.addWidget(bruker_scan)
        load_layout.addLayout(steam_path_layout)

        steam_layout = QHBoxLayout()
        steam_label = QLabel("STEAM VOIs:")
        steam_label.setMinimumWidth(80)
        steam_layout.addWidget(steam_label)
        self.steam_combo = QComboBox()
        self.steam_combo.addItem("-- None --", None)
        self.steam_combo.setEnabled(False)
        steam_layout.addWidget(self.steam_combo, stretch=1)
        # Indicator (•) reflects auto-match confidence:
        #   green  = PatientID match
        #   yellow = StudyDate or ordinal match
        #   red    = manual / no match
        self.steam_match_label = QLabel("")
        self.steam_match_label.setMinimumWidth(20)
        self.steam_match_label.setToolTip("Auto-match confidence")
        steam_layout.addWidget(self.steam_match_label)
        load_layout.addLayout(steam_layout)

        # Auto-follow: when T1 changes, jump STEAM combo to the
        # matching subject (PatientID → StudyDate → ordinal).
        self.t1_combo.currentIndexChanged.connect(self._on_t1_changed_auto_follow)

        layout.addWidget(load_group)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        save_btn = QPushButton("Save to CSV")
        save_btn.clicked.connect(self._save_csv)
        btn_layout.addWidget(save_btn)

        btn_layout.addSpacing(20)

        load_btn = QPushButton("Load Selected")
        load_btn.setMinimumSize(120, 35)
        load_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; "
            "padding: 8px 16px; border-radius: 4px; border: none; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        load_btn.clicked.connect(self._load_selected)
        btn_layout.addWidget(load_btn)

        btn_layout.addSpacing(10)

        close_btn = QPushButton("Cancel")
        close_btn.clicked.connect(self.reject)
        btn_layout.addWidget(close_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _save_csv(self):
        """Save scan results to CSV."""
        from ..dicom_scanner import save_scan_to_csv

        folder_name = Path(self.folder_path).name
        default_name = f"{folder_name}_dicom_scan.csv"

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Scan Results",
            str(Path(self.folder_path).parent / default_name),
            "CSV Files (*.csv)"
        )

        if filepath:
            try:
                save_scan_to_csv(self.scan_results, filepath)
                QMessageBox.information(self, "Saved", f"Scan results saved to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

    def _update_row_highlights(self):
        """Update row colors based on type and selection status."""
        # Get currently selected paths
        t1_selected_path = self.t1_combo.currentData() if hasattr(self, 't1_combo') else None
        t2_selected_path = self.t2_combo.currentData() if hasattr(self, 't2_combo') else None

        for row, s in enumerate(self.scan_results):
            sample_file = s.get('sample_file', '')
            is_proxyl = s.get('is_proxyl', False)
            is_t2 = s.get('is_t2', False)

            # Determine color based on type and selection
            if sample_file == t1_selected_path and t1_selected_path:
                color = self.COLOR_PROXYL_SELECTED
            elif sample_file == t2_selected_path and t2_selected_path:
                color = self.COLOR_T2_SELECTED
            elif is_proxyl:
                color = self.COLOR_PROXYL_VALID
            elif is_t2:
                color = self.COLOR_T2_VALID
            else:
                color = self.COLOR_DEFAULT

            # Apply color to all cells in the row
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item:
                    item.setBackground(color)

    def _load_selected(self):
        """Load the selected T1 and T2 series."""
        t1_path = self.t1_combo.currentData()
        t2_path = self.t2_combo.currentData()

        if not t1_path and not t2_path:
            QMessageBox.warning(self, "No Selection", "Please select at least one series to load.")
            return

        # STEAM info: pass whatever the user has set in the Bruker
        # tree field — even if they didn't pick a STEAM subject yet —
        # so the downstream loader dialog can pre-populate the same
        # path and the user doesn't have to re-browse.
        steam_subject_id = self.steam_combo.currentData()
        steam_info = None
        if self.bruker_root_path:
            entry = (
                self.steam_index.get(steam_subject_id)
                if steam_subject_id else None
            )
            steam_info = {
                'bruker_root': self.bruker_root_path,
                'subject_id': steam_subject_id,
                'subject_folder': (
                    str(entry['subject_folder']) if entry else None
                ),
                'auto_load': bool(entry),
            }

        self.result = {
            'action': 'load_from_scan',
            't1_path': t1_path,
            't2_path': t2_path,
            'steam_info': steam_info,
        }
        self.accept()

    def get_result(self):
        """Get the dialog result."""
        return self.result

    # ------------------------------------------------------------------
    # Bruker / STEAM auto-follow (T024 Phase 3)
    # ------------------------------------------------------------------

    def _on_browse_bruker_root(self):
        """Open a folder picker seeded at the current Bruker root."""
        seed = self.bruker_root_edit.text() or str(Path(self.folder_path).parent)
        d = QFileDialog.getExistingDirectory(self, "Pick Bruker root folder", seed)
        if d:
            self.bruker_root_edit.setText(d)

    def _on_scan_bruker_root(self):
        """Walk every subject folder under the Bruker root, find STEAM
        VOIs, and populate the STEAM combo."""
        from ..steam_voi import (
            cluster_voi_acquisitions,
            parse_bruker_subject,
            scan_bruker_study,
        )

        root = self.bruker_root_edit.text().strip()
        if not root:
            QMessageBox.warning(self, "No path", "Pick a Bruker root folder first.")
            return
        root_path = Path(root)
        if not root_path.is_dir():
            QMessageBox.warning(self, "Not found", f"Folder does not exist:\n{root}")
            return

        # Two cases: root IS a single subject folder, OR root contains
        # multiple subject folders. Treat both uniformly by including
        # the root itself plus its immediate subdirectories.
        candidates = [root_path] + [
            c for c in sorted(root_path.iterdir()) if c.is_dir()
        ]

        self.steam_index = {}
        for subj_folder in candidates:
            vois = scan_bruker_study(subj_folder)
            if not vois:
                continue
            subj_meta = {}
            subj_file = subj_folder / "subject"
            if subj_file.is_file():
                try:
                    subj_meta = parse_bruker_subject(subj_file)
                except Exception:  # pragma: no cover - corrupt subject file
                    pass
            subject_id = (
                subj_meta.get("subject_id")
                or subj_folder.name
            )
            self.steam_index[subject_id] = {
                'bruker_root': root_path,
                'subject_folder': subj_folder,
                'subject_meta': subj_meta,
                'clusters': cluster_voi_acquisitions(vois, tolerance_mm=0.1),
                'raw_acquisitions': vois,
            }

        self.bruker_root_path = root
        self._refresh_steam_combo()

        n_subjects = len(self.steam_index)
        if n_subjects == 0:
            QMessageBox.information(
                self,
                "No STEAM",
                "No STEAM acquisitions found anywhere under this folder.",
            )
        else:
            self._auto_follow_to_t1_subject()

    def _refresh_steam_combo(self):
        """Repopulate steam_combo from steam_index."""
        self.steam_combo.blockSignals(True)
        self.steam_combo.clear()
        self.steam_combo.addItem("-- None --", None)
        for subj_id, entry in self.steam_index.items():
            n_clusters = len(entry['clusters'])
            n_acqs = len(entry['raw_acquisitions'])
            label = f"{subj_id}  —  {n_clusters} VOIs / {n_acqs} acqs"
            self.steam_combo.addItem(label, subj_id)
        self.steam_combo.setEnabled(self.steam_combo.count() > 1)
        self.steam_combo.blockSignals(False)

    def _on_t1_changed_auto_follow(self, _index: int):
        """When the user picks a different T1, follow the STEAM combo
        to the subject that matches it."""
        if not self.steam_index:
            return
        self._auto_follow_to_t1_subject()

    def _auto_follow_to_t1_subject(self):
        """Run the PatientID → StudyDate → ordinal cascade and select
        the matching subject in the STEAM combo.

        Updates ``steam_match_label`` with a coloured indicator:
        • = green (PatientID match),
        • = yellow (StudyDate match),
        • = orange (ordinal match), • = grey (no match).
        """
        # Find the T1 series record for the current combo selection.
        t1_path = self.t1_combo.currentData() if hasattr(self, 't1_combo') else None
        t1_series = None
        if t1_path:
            for s in self.scan_results:
                if s.get('sample_file') == t1_path:
                    t1_series = s
                    break

        match_subj, match_kind = None, "none"
        if t1_series:
            t1_pid = t1_series.get('patient_id', '')
            t1_date = t1_series.get('study_date', '')

            # Priority 1: PatientID
            if t1_pid:
                for sid, entry in self.steam_index.items():
                    meta = entry['subject_meta']
                    if meta.get('subject_id') and meta['subject_id'] == t1_pid:
                        match_subj, match_kind = sid, "patient_id"
                        break

            # Priority 2: StudyDate (YYYYMMDD substring on either side)
            if match_subj is None and t1_date:
                for sid, entry in self.steam_index.items():
                    folder_name = entry['subject_folder'].name
                    if t1_date and t1_date in folder_name:
                        match_subj, match_kind = sid, "study_date"
                        break

            # Priority 3: ordinal — match by sort order of T1 series
            # vs. STEAM subject folders.
            if match_subj is None and self.proxyl_series:
                t1_ix = self.proxyl_series.index(t1_series) if t1_series in self.proxyl_series else 0
                subj_ids = list(self.steam_index.keys())
                if 0 <= t1_ix < len(subj_ids):
                    match_subj, match_kind = subj_ids[t1_ix], "ordinal"

        # Apply match.
        if match_subj is None:
            self.steam_combo.setCurrentIndex(0)
            self.steam_match_label.setText("")
            self.steam_match_label.setToolTip("No auto-match")
        else:
            for i in range(self.steam_combo.count()):
                if self.steam_combo.itemData(i) == match_subj:
                    self.steam_combo.setCurrentIndex(i)
                    break
            color, text, tip = {
                "patient_id": ("#4CAF50", "●", "Matched by PatientID"),
                "study_date": ("#FFC107", "●", "Matched by StudyDate substring"),
                "ordinal":    ("#FF9800", "●", "Matched by ordinal position (verify!)"),
            }.get(match_kind, ("#888", "●", "Manual"))
            self.steam_match_label.setText(text)
            self.steam_match_label.setStyleSheet(f"color: {color}; font-size: 16px;")
            self.steam_match_label.setToolTip(tip)


class MainMenuDialog(QDialog):
    """
    Main workflow menu shown after registration completes.

    Provides access to:
    - Load new experiment / Load previous session
    - ROI Analysis (with T2 as default source)
    - Parameter Maps (sliding window, pixel-level)
    - Image Tools (averaged, difference images)
    - Export options
    """

    # Signals for workflow actions
    roi_analysis_requested = Signal(dict)  # Emits ROI analysis settings
    parameter_maps_requested = Signal(dict)  # Emits parameter map settings
    export_requested = Signal(str)  # Emits export type

    def __init__(self,
                 registered_4d: Optional[np.ndarray] = None,
                 spacing: Optional[Tuple] = None,
                 time_array: Optional[np.ndarray] = None,
                 dicom_path: str = "",
                 output_dir: str = './output',
                 registered_t2: Optional[np.ndarray] = None,
                 roi_state: Optional[dict] = None,
                 steam_info: Optional[dict] = None,
                 parent=None):
        super().__init__(parent)
        self.registered_4d = registered_4d
        self.spacing = spacing
        self.time_array = time_array
        self.dicom_path = dicom_path
        self.output_dir = output_dir
        self.registered_t2 = registered_t2

        # STEAM VOI hint carried in from DicomScanResultsDialog — has
        # bruker_root, subject_folder, subject_id. Used when the user
        # clicks "Manage STEAM VOIs…" to seed the loader dialog.
        self.steam_info = steam_info or {}
        # Loaded VOIs (StudyVOIs) — None until user runs the loader.
        self.study_vois = None
        # Cached T1Geometry — lazily computed when first needed.
        self._t1_geometry = None

        # State - persisted across menu returns via roi_state dict
        if roi_state:
            self.roi_mask = roi_state.get('roi_mask')
            self.roi_signal = roi_state.get('roi_signal')
            self.injection_idx = roi_state.get('injection_idx')
            self.injection_time = roi_state.get('injection_time')
        else:
            self.roi_mask = None
            self.roi_signal = None
            self.injection_idx = None
            self.injection_time = None
        self.result = None  # Stores the user's action

        # Determine max z-slice
        if registered_4d is not None:
            self.max_z = registered_4d.shape[2] - 1
        else:
            self.max_z = 8  # Default

        self.setWindowTitle("ProxylFit - Analysis Menu")
        self.setMinimumSize(700, 800)
        self.resize(750, 850)

        self._setup_ui()
        self._update_data_status()

        # Eagerly load any previously-saved STEAM VOIs for this dataset
        # so downstream dialogs (parameter-map options, results) see
        # them as available without requiring the user to first switch
        # to the STEAM ROI method.
        self._try_load_steam_voi_json()

    def _setup_ui(self):
        """Build the menu UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create scroll area to handle overflow
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        # Content widget inside scroll area
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Header
        header = HeaderWidget("ProxylFit Analysis Menu", "Select an analysis workflow")
        layout.addWidget(header)

        # Experiment section
        self._create_experiment_section(layout)

        # Data status
        self._create_data_status_section(layout)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # ROI Analysis section
        self._create_roi_section(layout)

        # Parameter Maps section
        self._create_param_maps_section(layout)

        # Image Tools section
        self._create_image_tools_section(layout)

        # Export section
        self._create_export_section(layout)

        # Spacer at bottom of scrollable content
        layout.addStretch()

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

        # Footer with Exit button (outside scroll area, always visible)
        footer = ButtonBar()
        footer.add_button("exit", "Exit", self._on_exit, "cancel")
        main_layout.addWidget(footer)

    def _create_experiment_section(self, parent_layout):
        """Create the Experiment section for loading data."""
        group = QGroupBox("Experiment")
        layout = QVBoxLayout(group)

        # Buttons row
        btn_layout = QHBoxLayout()

        scan_btn = QPushButton("Load from DICOM Folder...")
        scan_btn.clicked.connect(self._scan_dicom_folder)
        scan_btn.setToolTip("Load T1/T2 series from a DICOM folder")
        btn_layout.addWidget(scan_btn)

        load_new_btn = QPushButton("Load T1 DICOM...")
        load_new_btn.clicked.connect(self._load_new_experiment)
        btn_layout.addWidget(load_new_btn)

        load_prev_btn = QPushButton("Load Previous Session...")
        load_prev_btn.clicked.connect(self._load_previous_session)
        btn_layout.addWidget(load_prev_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Current experiment info
        self.experiment_info_label = QLabel("No data loaded")
        self.experiment_info_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.experiment_info_label)

        parent_layout.addWidget(group)

    def _create_data_status_section(self, parent_layout):
        """Create the data status display."""
        status_layout = QHBoxLayout()

        # T1 status
        self.t1_status_label = QLabel("T1 Data: Not loaded")
        status_layout.addWidget(self.t1_status_label)

        status_layout.addSpacing(20)

        # Registration status
        self.reg_status_label = QLabel("Registration: —")
        status_layout.addWidget(self.reg_status_label)

        status_layout.addStretch()

        # T2 status and load button
        self.t2_status_label = QLabel("T2 Data: Not loaded")
        status_layout.addWidget(self.t2_status_label)

        self.load_t2_btn = QPushButton("Load T2 Volume...")
        self.load_t2_btn.clicked.connect(self._load_t2_volume)
        self.load_t2_btn.setEnabled(self.registered_4d is not None)
        status_layout.addWidget(self.load_t2_btn)

        parent_layout.addLayout(status_layout)

    def _create_roi_section(self, parent_layout):
        """Create the ROI Analysis section."""
        group = QGroupBox("ROI Analysis")
        layout = QVBoxLayout(group)

        description = QLabel("Draw ROI to extract time series and set injection time")
        description.setStyleSheet("color: #666;")
        layout.addWidget(description)

        # ROI Source — which anatomy is shown for drawing or as the
        # overlay backdrop for a STEAM VOI.
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("ROI Source:"))

        self.t2_source_radio = QRadioButton("T2")
        self.t1_source_radio = QRadioButton("T1")
        self.t2_source_radio.setChecked(True)  # T2 is default

        # Disable T2 option if not loaded
        self.t2_source_radio.setEnabled(self.registered_t2 is not None)
        if self.registered_t2 is None:
            self.t1_source_radio.setChecked(True)

        source_layout.addWidget(self.t2_source_radio)
        source_layout.addWidget(self.t1_source_radio)
        source_layout.addStretch()
        layout.addLayout(source_layout)

        # ROI Method — how the ROI is produced. STEAM VOI sits here as
        # a fourth option: instead of drawing on the image, use the
        # prescribed Bruker voxel directly. Z-slice and Source are both
        # irrelevant for STEAM (the box is 3D and prescribed), so they
        # collapse when STEAM is picked.
        self.method_row = QWidget()
        method_layout = QHBoxLayout(self.method_row)
        method_layout.setContentsMargins(0, 0, 0, 0)
        method_layout.addWidget(QLabel("ROI Method:"))

        self.rect_radio = QRadioButton("Rectangle")
        self.contour_radio = QRadioButton("Manual Contour")
        self.segment_radio = QRadioButton("Segment")
        self.steam_method_radio = QRadioButton("STEAM VOI")
        self.contour_radio.setChecked(True)  # Default

        method_layout.addWidget(self.rect_radio)
        method_layout.addWidget(self.contour_radio)
        method_layout.addWidget(self.segment_radio)
        method_layout.addWidget(self.steam_method_radio)
        method_layout.addStretch()
        layout.addWidget(self.method_row)

        # STEAM-VOI sub-row: voxel picker + manage button. Shown only
        # when the STEAM Method radio is selected.
        self.steam_subrow = QWidget()
        steam_sub_layout = QHBoxLayout(self.steam_subrow)
        steam_sub_layout.setContentsMargins(20, 0, 0, 0)
        steam_sub_layout.addWidget(QLabel("Voxel:"))
        self.steam_voxel_combo = QComboBox()
        self.steam_voxel_combo.addItem("-- (load VOIs first) --", None)
        self.steam_voxel_combo.setEnabled(False)
        steam_sub_layout.addWidget(self.steam_voxel_combo, stretch=1)
        self.manage_steam_btn = QPushButton("Manage STEAM VOIs…")
        self.manage_steam_btn.clicked.connect(self._on_manage_steam)
        steam_sub_layout.addWidget(self.manage_steam_btn)
        layout.addWidget(self.steam_subrow)
        self.steam_subrow.setVisible(False)

        # Hook the STEAM radio after the subrow is in place so the
        # initial signal handler can find it.
        self.steam_method_radio.toggled.connect(self._on_roi_source_changed)

        # Z-slice (hidden when STEAM is the method — picked automatically)
        self.z_row = QWidget()
        z_layout = QHBoxLayout(self.z_row)
        z_layout.setContentsMargins(0, 0, 0, 0)
        z_layout.addWidget(QLabel("Z-slice:"))

        self.z_spinbox = QSpinBox()
        self.z_spinbox.setMinimum(0)
        self.z_spinbox.setMaximum(self.max_z)
        self.z_spinbox.setValue(min(4, self.max_z))
        z_layout.addWidget(self.z_spinbox)

        self.z_max_label = QLabel(f"/ {self.max_z}")
        z_layout.addWidget(self.z_max_label)

        z_layout.addStretch()
        layout.addWidget(self.z_row)

        # ROI status line
        self.roi_status_label = QLabel("ROI: Not drawn")
        self.roi_status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.roi_status_label)

        # Buttons row - Draw ROI and Run Kinetic Fit
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        # Draw ROI button - green
        self.start_roi_btn = QPushButton("Draw ROI")
        self.start_roi_btn.setMinimumSize(120, 40)
        self.start_roi_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; "
            "padding: 8px 20px; font-size: 13px; border-radius: 5px; border: none; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.start_roi_btn.clicked.connect(self._draw_roi)
        self.start_roi_btn.setEnabled(self.registered_4d is not None)
        btn_layout.addWidget(self.start_roi_btn)

        btn_layout.addSpacing(15)

        # Set/Change Injection Time button — reopens the injection
        # time selector with the existing ROI signal so the user can
        # adjust the injection index or toggle the Fix A2 fit option
        # and re-run the kinetic fit on the same ROI for comparison.
        # Disabled until an ROI has been drawn.
        self.set_injection_btn = QPushButton("Set Injection Time")
        self.set_injection_btn.setMinimumSize(160, 40)
        self.set_injection_btn.setStyleSheet(
            "QPushButton { background-color: #607D8B; color: white; font-weight: bold; "
            "padding: 8px 20px; font-size: 13px; border-radius: 5px; border: none; }"
            "QPushButton:hover { background-color: #455A64; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.set_injection_btn.clicked.connect(self._set_injection_time)
        self.set_injection_btn.setEnabled(False)
        self.set_injection_btn.setToolTip(
            "Reopen the injection time dialog with the current ROI's "
            "signal. Lets you change injection index or toggle Fix A2 "
            "and re-run the kinetic fit on the same ROI."
        )
        btn_layout.addWidget(self.set_injection_btn)

        btn_layout.addSpacing(15)

        # Run Kinetic Fit button - orange, requires ROI + injection time
        self.kinetic_fit_btn = QPushButton("Run Kinetic Fit")
        self.kinetic_fit_btn.setMinimumSize(140, 40)
        self.kinetic_fit_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-weight: bold; "
            "padding: 8px 20px; font-size: 13px; border-radius: 5px; border: none; }"
            "QPushButton:hover { background-color: #F57C00; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.kinetic_fit_btn.clicked.connect(self._run_kinetic_fit)
        self.kinetic_fit_btn.setEnabled(False)  # Enabled when ROI + injection time set
        self.kinetic_fit_btn.setToolTip("Draw ROI and select injection time first")
        btn_layout.addWidget(self.kinetic_fit_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        parent_layout.addWidget(group)

    def _create_param_maps_section(self, parent_layout):
        """Create the Parameter Maps section."""
        group = QGroupBox("Parameter Maps")
        layout = QVBoxLayout(group)

        description = QLabel("Generate spatial parameter maps across the image")
        description.setStyleSheet("color: #666;")
        layout.addWidget(description)

        # Create button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.create_maps_btn = QPushButton("Create Parameter Maps")
        self.create_maps_btn.setMinimumSize(180, 40)
        self.create_maps_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; "
            "padding: 8px 20px; font-size: 13px; border-radius: 5px; border: none; }"
            "QPushButton:hover { background-color: #1976D2; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.create_maps_btn.clicked.connect(self._create_parameter_maps)
        self.create_maps_btn.setEnabled(self.registered_4d is not None)
        btn_layout.addWidget(self.create_maps_btn)
        btn_layout.addStretch()

        layout.addLayout(btn_layout)

        parent_layout.addWidget(group)

    def _create_image_tools_section(self, parent_layout):
        """Create the Image Tools section."""
        group = QGroupBox("Image Tools")
        layout = QVBoxLayout(group)

        description = QLabel("Select time ranges on signal curve to generate processed images. Requires ROI + injection time.")
        description.setStyleSheet("color: #666;")
        description.setWordWrap(True)
        layout.addWidget(description)

        # Buttons
        btn_layout = QHBoxLayout()

        self.averaged_btn = QPushButton("Averaged Image")
        self.averaged_btn.clicked.connect(self._create_averaged_image)
        self.averaged_btn.setEnabled(False)  # Enabled after ROI + injection time
        self.averaged_btn.setToolTip("Draw ROI and select injection time first")
        btn_layout.addWidget(self.averaged_btn)

        self.difference_btn = QPushButton("Difference Image")
        self.difference_btn.clicked.connect(self._create_difference_image)
        self.difference_btn.setEnabled(False)  # Enabled after ROI + injection time
        self.difference_btn.setToolTip("Draw ROI and select injection time first")
        btn_layout.addWidget(self.difference_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        parent_layout.addWidget(group)

    def _create_export_section(self, parent_layout):
        """Create the Export section."""
        group = QGroupBox("Export")
        layout = QHBoxLayout(group)

        self.export_data_btn = QPushButton("Registered 4D Data")
        self.export_data_btn.clicked.connect(lambda: self._export("registered_data"))
        self.export_data_btn.setEnabled(self.registered_4d is not None)
        layout.addWidget(self.export_data_btn)

        self.export_report_btn = QPushButton("Registration Report")
        self.export_report_btn.clicked.connect(lambda: self._export("registration_report"))
        self.export_report_btn.setEnabled(self.registered_4d is not None)
        layout.addWidget(self.export_report_btn)

        self.export_timeseries_btn = QPushButton("Time Series CSV")
        self.export_timeseries_btn.clicked.connect(lambda: self._export("timeseries"))
        self.export_timeseries_btn.setEnabled(self.roi_signal is not None)
        layout.addWidget(self.export_timeseries_btn)

        layout.addStretch()

        parent_layout.addWidget(group)

    def _update_data_status(self):
        """Update all data status displays."""
        # T1 status
        if self.registered_4d is not None:
            shape = self.registered_4d.shape
            self.t1_status_label.setText(f"T1 Data: {shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}")
            self.reg_status_label.setText("Registration: Complete")
            self.reg_status_label.setStyleSheet("color: green;")

            # Update experiment info
            dicom_name = Path(self.dicom_path).name if self.dicom_path else "Unknown"
            self.experiment_info_label.setText(f"Current: {dicom_name}\nOutput: {self.output_dir}")
        else:
            self.t1_status_label.setText("T1 Data: Not loaded")
            self.reg_status_label.setText("Registration: —")
            self.reg_status_label.setStyleSheet("")
            self.experiment_info_label.setText("No data loaded")

        # T2 status
        if self.registered_t2 is not None:
            self.t2_status_label.setText("T2 Data: Loaded")
            self.t2_status_label.setStyleSheet("color: green;")
            self.t2_source_radio.setEnabled(True)
            self.t2_source_radio.setChecked(True)
        else:
            self.t2_status_label.setText("T2 Data: Not loaded")
            self.t2_status_label.setStyleSheet("")
            self.t2_source_radio.setEnabled(False)
            self.t1_source_radio.setChecked(True)

        # Update button states
        has_data = self.registered_4d is not None
        self.load_t2_btn.setEnabled(has_data)
        self.start_roi_btn.setEnabled(has_data)
        self.create_maps_btn.setEnabled(has_data)
        self.export_data_btn.setEnabled(has_data)

        # Update ROI status (enables kinetic fit, image tools if ROI exists)
        self._update_roi_status()
        self.export_report_btn.setEnabled(has_data)

        # Update z-slice bounds
        if has_data:
            self.max_z = self.registered_4d.shape[2] - 1
            self.z_spinbox.setMaximum(self.max_z)
            self.z_max_label.setText(f"/ {self.max_z}")

    def _load_new_experiment(self):
        """Load a new T1 DICOM and run registration."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load T1 DICOM", "",
            "DICOM Files (*.dcm);;All Files (*)"
        )
        if not file_path:
            return

        # Store result for caller to handle
        self.result = {
            'action': 'load_new',
            'dicom_path': file_path
        }
        self.accept()

    def _load_previous_session(self):
        """Load a previous session from saved registration data."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Previous Session Folder",
            str(Path(self.output_dir).parent) if self.output_dir else "."
        )
        if not folder_path:
            return

        # Check if valid registration data exists. Accept the current
        # T1-nested per-slice layout (`T1/z00/t000.dcm`), the
        # transitional per-slice layout (`z00/t000.dcm`), and the
        # legacy flat layout (`z00_t000.dcm`). The first two updates
        # to the registration output folder broke this check, forcing
        # users to redo registration on every load.
        p = Path(folder_path)
        dicom_dir = p / "registered" / "dicoms"

        def _has_dicoms(d):
            return d.exists() and (
                (d / "T1" / "z00" / "t000.dcm").exists()
                or (d / "z00" / "t000.dcm").exists()
                or (d / "z00_t000.dcm").exists()
            )

        if _has_dicoms(dicom_dir):
            # Valid session found
            self.result = {
                'action': 'load_previous',
                'session_path': folder_path
            }
            self.accept()
            return

        # Not found — try to give helpful diagnostics
        hints = []

        # Check if user selected the parent of a valid session
        for child in p.iterdir():
            if child.is_dir():
                if _has_dicoms(child / "registered" / "dicoms"):
                    hints.append(f"  • {child.name}/")

        # Check if the selected folder itself contains "registered" but wrong structure
        if (p / "registered").exists() and not dicom_dir.exists():
            hints.append("  • Found 'registered/' folder but no 'dicoms/' subfolder inside it.")
        elif dicom_dir.exists():
            dcm_files = list(dicom_dir.rglob("*.dcm"))
            if dcm_files:
                hints.append(
                    f"  • Found {len(dcm_files)} DICOM files in registered/dicoms/ "
                    "but no T1/z00/t000.dcm, z00/t000.dcm, or z00_t000.dcm was present."
                )

        msg = f"No valid session found in:\n{folder_path}\n\n"
        msg += "A valid session folder should contain one of:\n"
        msg += "  registered/dicoms/T1/z00/t000.dcm  (current layout)\n"
        msg += "  registered/dicoms/z00/t000.dcm     (transitional)\n"
        msg += "  registered/dicoms/z00_t000.dcm     (legacy)\n\n"

        if hints:
            msg += "Possible issues:\n" + "\n".join(hints) + "\n\n"

        msg += "Tip: Select the output folder that was created when you\n"
        msg += "originally ran registration (e.g., 'output_MyExperiment/')."

        QMessageBox.warning(self, "Invalid Session", msg)

    def _load_t2_volume(self):
        """Open file dialog and load T2 volume."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load T2 DICOM", "",
            "DICOM Files (*.dcm);;All Files (*)"
        )
        if not file_path:
            return

        # Store result for caller to handle T2 registration
        self.result = {
            'action': 'load_t2',
            't2_path': file_path
        }
        self.accept()

    def _scan_dicom_folder(self):
        """Scan a DICOM folder and show available series."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select DICOM Folder",
            str(Path.home())
        )
        if not folder_path:
            return

        try:
            from ..dicom_scanner import scan_dicom_folder

            # Scan the folder
            results = scan_dicom_folder(folder_path)

            if not results:
                QMessageBox.warning(self, "No DICOM Files", "No DICOM files found in the selected folder.")
                return

            # Show results dialog
            dialog = DicomScanResultsDialog(results, folder_path, self)
            if dialog.exec() and dialog.get_result():
                # User clicked "Load Selected" - pass result to main menu caller
                self.result = dialog.get_result()
                self.accept()

        except ImportError as e:
            QMessageBox.critical(self, "Missing Package", f"Required package not found: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Scan Error", f"Error scanning folder:\n{e}")

    def _draw_roi(self):
        """Launch ROI drawing workflow (just ROI + injection time, no fitting)."""
        # STEAM VOI is a geometric ROI — no drawing step.
        if self.steam_method_radio.isChecked():
            voi = self.steam_voxel_combo.currentData()
            if voi is None or self.study_vois is None:
                QMessageBox.warning(
                    self,
                    "No STEAM VOI",
                    "Click 'Manage STEAM VOIs…' to load tumor + contralateral "
                    "voxels first, then pick one from the Voxel dropdown."
                )
                return
            try:
                steam_mask = self._build_steam_mask(voi)
            except Exception as e:
                QMessageBox.critical(self, "Mask build failed", f"{e}")
                return
            self.result = {
                'action': 'draw_roi',
                'roi_source': 'steam',
                'roi_mode': 'steam_voi',
                'z_slice': self.z_spinbox.value(),
                'steam_voi_label': voi.label,
                'steam_mask': steam_mask,
            }
            self.accept()
            return

        # Gather settings
        if self.t2_source_radio.isChecked() and self.registered_t2 is not None:
            roi_source = 't2'
        else:
            roi_source = 't1'

        if self.rect_radio.isChecked():
            roi_mode = 'rectangle'
        elif self.segment_radio.isChecked():
            roi_mode = 'segment'
        else:
            roi_mode = 'contour'

        self.result = {
            'action': 'draw_roi',
            'roi_source': roi_source,
            'roi_mode': roi_mode,
            'z_slice': self.z_spinbox.value()
        }
        self.accept()

    # ------------------------------------------------------------------
    # STEAM VOI source (T024 Phase 3)
    # ------------------------------------------------------------------

    def _on_roi_source_changed(self, _checked: bool):
        """Show/hide STEAM sub-row + Z-slice row based on the method.

        When STEAM is the method, the voxel sub-picker appears and the
        Z-slice control is hidden — the slice is picked automatically
        from the slice with the largest VOI cross-section.
        """
        is_steam = self.steam_method_radio.isChecked()
        self.steam_subrow.setVisible(is_steam)
        if hasattr(self, 'z_row'):
            self.z_row.setVisible(not is_steam)
        if is_steam:
            self._try_load_steam_voi_json()

    def _try_load_steam_voi_json(self):
        """Look for an existing steam_voi.json in the dataset's output dir."""
        if self.study_vois is not None:
            return
        from ..steam_voi import load_voi_json
        candidate = Path(self.output_dir) / "steam_voi.json"
        if candidate.is_file():
            try:
                self.study_vois = load_voi_json(candidate)
                self._refresh_steam_voxel_combo()
            except Exception:  # pragma: no cover
                pass

    def _refresh_steam_voxel_combo(self):
        """Populate the Voxel combo with the loaded VOIs."""
        self.steam_voxel_combo.blockSignals(True)
        self.steam_voxel_combo.clear()
        if self.study_vois is None or not self.study_vois.voi:
            self.steam_voxel_combo.addItem("-- (load VOIs first) --", None)
            self.steam_voxel_combo.setEnabled(False)
        else:
            for v in self.study_vois.voi:
                pos = v.position_mm
                label = (
                    f"{v.label.capitalize()}  "
                    f"({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})"
                )
                self.steam_voxel_combo.addItem(label, v)
            self.steam_voxel_combo.setEnabled(True)
        self.steam_voxel_combo.blockSignals(False)

    def _on_manage_steam(self):
        """Open the STEAM VOI loader dialog seeded with current state."""
        if self.registered_4d is None:
            QMessageBox.warning(self, "Load data", "Load a T1 series first.")
            return
        from ..steam_voi import build_t1_geometry_from_dicom, load_voi_json
        from .steam_voi import SteamVoiLoaderDialog

        # Lazily compute / cache the T1Geometry.
        if self._t1_geometry is None:
            try:
                self._t1_geometry = build_t1_geometry_from_dicom(self.dicom_path)
            except Exception as e:
                QMessageBox.warning(
                    self, "Geometry unavailable",
                    f"Could not build T1 geometry from DICOM:\n{e}\n\n"
                    "STEAM VOI overlay requires DICOM ImagePositionPatient / "
                    "ImageOrientationPatient tags. Try re-loading the dataset."
                )
                return

        # T1 mid-slice volume (3D) for the preview.
        t1_volume = self.registered_4d[:, :, :, 0]

        # Prefer the subject-specific folder if the DICOM scan dialog
        # already narrowed it down, otherwise fall back to the
        # multi-subject Bruker root the user picked.
        bruker_hint = None
        if self.steam_info:
            bruker_hint = (
                self.steam_info.get('subject_folder')
                or self.steam_info.get('bruker_root')
            )
        json_path = Path(self.output_dir) / "steam_voi.json"
        existing = json_path if json_path.is_file() else None

        dlg = SteamVoiLoaderDialog(
            t1_volume,
            self._t1_geometry,
            output_dir=Path(self.output_dir),
            parent=self,
            bruker_root_hint=bruker_hint,
            existing_json=existing,
        )
        if dlg.exec():
            self.study_vois = dlg.result_vois
            self._refresh_steam_voxel_combo()
            # Pre-select tumor.
            for i in range(self.steam_voxel_combo.count()):
                v = self.steam_voxel_combo.itemData(i)
                if v is not None and v.label == "tumor":
                    self.steam_voxel_combo.setCurrentIndex(i)
                    break

    def _build_steam_mask(self, voi):
        """Rasterize a SteamVOI to a T1-grid mask. Caches geometry."""
        from ..steam_voi import build_t1_geometry_from_dicom, voi_to_mask
        if self._t1_geometry is None:
            self._t1_geometry = build_t1_geometry_from_dicom(self.dicom_path)
        # Pass the study default; voi_to_mask honors voi.transform if set.
        s2p = (
            self.study_vois.scanner_to_patient
            if self.study_vois is not None else np.eye(4)
        )
        t2_to_t1 = (
            self.study_vois.t2_to_t1_translation_mm
            if self.study_vois is not None else None
        )
        return voi_to_mask(
            voi, self._t1_geometry,
            scanner_to_patient=s2p,
            t2_to_t1_translation_mm=t2_to_t1,
        )

    def _run_kinetic_fit(self):
        """Launch kinetic fitting on existing ROI data."""
        if self.roi_mask is None or self.roi_signal is None or self.injection_idx is None:
            QMessageBox.warning(
                self, "Missing Data",
                "Please draw an ROI and select injection time first."
            )
            return

        self.result = {
            'action': 'kinetic_fit',
            'roi_mask': self.roi_mask,
            'roi_signal': self.roi_signal,
            'injection_idx': self.injection_idx,
            'injection_time': self.injection_time
        }
        self.accept()

    def _set_injection_time(self):
        """Reopen the injection time dialog on the current ROI signal.

        Emits a 'reopen_injection' action so run_analysis.py can launch
        select_injection_time_qt with the existing roi_state. Lets the
        user adjust injection index or the Fix A2 fit option and then
        re-run the kinetic fit on the same ROI for comparison without
        having to redraw it.
        """
        if self.roi_mask is None or self.roi_signal is None:
            QMessageBox.warning(
                self, "Missing ROI",
                "Please draw an ROI first."
            )
            return

        self.result = {
            'action': 'reopen_injection',
        }
        self.accept()

    def set_roi_data(self, roi_mask: np.ndarray, roi_signal: np.ndarray,
                     injection_idx: int, injection_time: float):
        """Set ROI data after drawing (called by run_analysis.py)."""
        self.roi_mask = roi_mask
        self.roi_signal = roi_signal
        self.injection_idx = injection_idx
        self.injection_time = injection_time
        self._update_roi_status()

    def _update_roi_status(self):
        """Update ROI status display and button states."""
        if self.roi_mask is not None and self.injection_idx is not None:
            num_pixels = int(np.sum(self.roi_mask))
            self.roi_status_label.setText(
                f"ROI: {num_pixels} pixels | Injection: t={self.injection_idx}"
            )
            self.roi_status_label.setStyleSheet("color: green; font-weight: bold;")
            # Enable kinetic fit button
            self.kinetic_fit_btn.setEnabled(True)
            self.kinetic_fit_btn.setToolTip("")
            # Enable Set Injection Time (revisit) — needs ROI signal,
            # which is always set alongside roi_mask in this branch.
            self.set_injection_btn.setEnabled(True)
            self.set_injection_btn.setToolTip(
                "Reopen the injection time dialog with the current "
                "ROI's signal. Adjust injection index or toggle "
                "Fix A2, then re-run the kinetic fit for comparison."
            )
            # Enable image tools
            self.averaged_btn.setEnabled(True)
            self.averaged_btn.setToolTip("")
            self.difference_btn.setEnabled(True)
            self.difference_btn.setToolTip("")
            # Enable time series export
            self.export_timeseries_btn.setEnabled(True)
        elif self.roi_mask is not None:
            num_pixels = int(np.sum(self.roi_mask))
            self.roi_status_label.setText(f"ROI: {num_pixels} pixels | Injection: Not set")
            self.roi_status_label.setStyleSheet("color: #FF9800;")
            # ROI exists but no injection yet — let the user set it
            # without redrawing.
            self.set_injection_btn.setEnabled(True)
        else:
            self.roi_status_label.setText("ROI: Not drawn")
            self.roi_status_label.setStyleSheet("color: #666; font-style: italic;")

    def _create_parameter_maps(self):
        """Launch parameter mapping workflow."""
        self.result = {
            'action': 'parameter_maps',
            # Carry loaded STEAM VOIs + transform forward so the
            # parameter-map options dialog can offer them as a
            # fitting-region choice and/or as a measurement ROI.
            'study_vois': self.study_vois,
        }
        self.accept()

    def _create_averaged_image(self):
        """Launch averaged image tool (T002)."""
        if self.roi_signal is None:
            QMessageBox.warning(
                self, "Missing Data",
                "Please draw an ROI and select injection time first."
            )
            return

        self.result = {
            'action': 'image_tools',
            'mode': 'average',
            'roi_signal': self.roi_signal
        }
        self.accept()

    def _create_difference_image(self):
        """Launch difference image tool (T003)."""
        if self.roi_signal is None:
            QMessageBox.warning(
                self, "Missing Data",
                "Please draw an ROI and select injection time first."
            )
            return

        self.result = {
            'action': 'image_tools',
            'mode': 'difference',
            'roi_signal': self.roi_signal
        }
        self.accept()

    def _export(self, export_type: str):
        """Handle export requests."""
        self.result = {
            'action': 'export',
            'export_type': export_type
        }
        self.accept()

    def _on_exit(self):
        """Handle exit button."""
        self.result = {'action': 'exit'}
        self.reject()

    def get_result(self) -> Optional[dict]:
        """Get the result after dialog closes."""
        return self.result


def show_main_menu(registered_4d: Optional[np.ndarray] = None,
                   spacing: Optional[Tuple] = None,
                   time_array: Optional[np.ndarray] = None,
                   dicom_path: str = "",
                   output_dir: str = './output',
                   registered_t2: Optional[np.ndarray] = None,
                   roi_state: Optional[dict] = None,
                   steam_info: Optional[dict] = None) -> Optional[dict]:
    """
    Show the main workflow menu.

    Parameters
    ----------
    registered_4d : np.ndarray, optional
        Registered 4D image data [x, y, z, t]
    spacing : tuple, optional
        Voxel spacing (x, y, z)
    time_array : np.ndarray, optional
        Time array for the data
    dicom_path : str
        Path to the source DICOM file
    output_dir : str
        Output directory path
    registered_t2 : np.ndarray, optional
        Registered T2 volume (if loaded)
    roi_state : dict, optional
        Preserved ROI state with keys: roi_mask, roi_signal, injection_idx, injection_time

    Returns
    -------
    dict or None
        User's action and settings, or None if cancelled
    """
    app = init_qt_app()

    dialog = MainMenuDialog(
        registered_4d=registered_4d,
        spacing=spacing,
        time_array=time_array,
        dicom_path=dicom_path,
        output_dir=output_dir,
        registered_t2=registered_t2,
        roi_state=roi_state,
        steam_info=steam_info,
    )

    result = dialog.exec()

    if result == QDialog.Accepted:
        return dialog.get_result()
    else:
        return dialog.get_result()  # May contain 'exit' action
