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
    QComboBox
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

        self.result = {
            'action': 'load_from_scan',
            't1_path': t1_path,
            't2_path': t2_path
        }
        self.accept()

    def get_result(self):
        """Get the dialog result."""
        return self.result


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
                 parent=None):
        super().__init__(parent)
        self.registered_4d = registered_4d
        self.spacing = spacing
        self.time_array = time_array
        self.dicom_path = dicom_path
        self.output_dir = output_dir
        self.registered_t2 = registered_t2

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

        # ROI Source
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

        # ROI Method
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("ROI Method:"))

        self.rect_radio = QRadioButton("Rectangle")
        self.contour_radio = QRadioButton("Manual Contour")
        self.segment_radio = QRadioButton("Segment")
        self.contour_radio.setChecked(True)  # Default

        method_layout.addWidget(self.rect_radio)
        method_layout.addWidget(self.contour_radio)
        method_layout.addWidget(self.segment_radio)
        method_layout.addStretch()
        layout.addLayout(method_layout)

        # Z-slice
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z-slice:"))

        self.z_spinbox = QSpinBox()
        self.z_spinbox.setMinimum(0)
        self.z_spinbox.setMaximum(self.max_z)
        self.z_spinbox.setValue(min(4, self.max_z))
        z_layout.addWidget(self.z_spinbox)

        self.z_max_label = QLabel(f"/ {self.max_z}")
        z_layout.addWidget(self.z_max_label)

        z_layout.addStretch()
        layout.addLayout(z_layout)

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

        # Check if valid registration data exists (2D DICOM slice format)
        p = Path(folder_path)
        dicom_dir = p / "registered" / "dicoms"

        # Check for 2D slice format: z00_t000.dcm
        if not (dicom_dir.exists() and (dicom_dir / "z00_t000.dcm").exists()):
            QMessageBox.warning(
                self, "Invalid Session",
                f"No registration data found in:\n{folder_path}\n\n"
                "Expected: registered/dicoms/z00_t000.dcm, z00_t001.dcm, ...\n"
                "If you have old data, please re-run registration."
            )
            return

        # Store result for caller to handle
        self.result = {
            'action': 'load_previous',
            'session_path': folder_path
        }
        self.accept()

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
        else:
            self.roi_status_label.setText("ROI: Not drawn")
            self.roi_status_label.setStyleSheet("color: #666; font-style: italic;")

    def _create_parameter_maps(self):
        """Launch parameter mapping workflow."""
        self.result = {
            'action': 'parameter_maps'
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
                   roi_state: Optional[dict] = None) -> Optional[dict]:
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
        roi_state=roi_state
    )

    result = dialog.exec()

    if result == QDialog.Accepted:
        return dialog.get_result()
    else:
        return dialog.get_result()  # May contain 'exit' action
