# T007: Registration Progress UI

**Status**: planned
**Priority**: high (blocking T006 testing)
**Created**: 2025-01-21

## Description

Add a progress dialog during registration so the UI remains responsive and users can see registration progress. Currently, registration blocks the main Qt thread, freezing the UI until complete.

## Problem

When loading a new dataset without prior registration:
1. The main menu dialog closes
2. The UI freezes completely
3. Progress is only visible in the terminal
4. User has no indication the app is working (appears hung)

## Solution

Run registration in a background QThread with a progress dialog that shows:
- Current timepoint being registered (e.g., "Registering timepoint 7/125")
- Progress bar
- Estimated time remaining
- Cancel button (optional)

## Implementation Steps

### 1. Add progress callback to `register_timeseries`

```python
# registration.py
def register_timeseries(image_4d: np.ndarray, spacing: Tuple[float, float, float],
                       output_dir: Optional[str] = None,
                       show_quality_window: bool = True,
                       dicom_path: Optional[str] = None,
                       progress_callback: Optional[Callable[[int, int, str], None]] = None
                       ) -> Tuple[np.ndarray, List[RegistrationMetrics]]:
    """
    ...
    progress_callback : callable, optional
        Function called with (current, total, status_message) for progress updates
    """
    # In the loop:
    for timepoint in range(1, t):
        if progress_callback:
            progress_callback(timepoint, t-1, f"Registering timepoint {timepoint}/{t-1}")
        # ... existing registration code ...
```

### 2. Create RegistrationWorker QThread

```python
# ui.py
class RegistrationWorker(QThread):
    """Background worker for registration."""
    progress = Signal(int, int, str)  # current, total, message
    finished = Signal(object, object)  # registered_4d, metrics
    error = Signal(str)

    def __init__(self, image_4d, spacing, output_dir, dicom_path):
        super().__init__()
        self.image_4d = image_4d
        self.spacing = spacing
        self.output_dir = output_dir
        self.dicom_path = dicom_path

    def run(self):
        try:
            registered_4d, metrics = register_timeseries(
                self.image_4d, self.spacing,
                output_dir=self.output_dir,
                show_quality_window=False,  # We'll show after dialog closes
                dicom_path=self.dicom_path,
                progress_callback=self._emit_progress
            )
            self.finished.emit(registered_4d, metrics)
        except Exception as e:
            self.error.emit(str(e))

    def _emit_progress(self, current, total, message):
        self.progress.emit(current, total, message)
```

### 3. Create RegistrationProgressDialog

```python
# ui.py
class RegistrationProgressDialog(QDialog):
    """Progress dialog for registration."""

    def __init__(self, image_4d, spacing, output_dir, dicom_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Registration in Progress")
        self.setModal(True)
        self.setMinimumWidth(400)

        self._setup_ui()
        self._start_worker(image_4d, spacing, output_dir, dicom_path)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self.status_label = QLabel("Starting registration...")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.detail_label = QLabel("")
        layout.addWidget(self.detail_label)

    def _start_worker(self, image_4d, spacing, output_dir, dicom_path):
        self.worker = RegistrationWorker(image_4d, spacing, output_dir, dicom_path)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, current, total, message):
        self.status_label.setText(message)
        self.progress_bar.setValue(int(100 * current / total))

    def _on_finished(self, registered_4d, metrics):
        self.registered_4d = registered_4d
        self.metrics = metrics
        self.accept()

    def _on_error(self, error_msg):
        QMessageBox.critical(self, "Registration Error", error_msg)
        self.reject()
```

### 4. Create helper function

```python
# ui.py
def run_registration_with_progress(image_4d, spacing, output_dir, dicom_path, parent=None):
    """Run registration with a progress dialog. Returns (registered_4d, metrics) or (None, None) on cancel."""
    dialog = RegistrationProgressDialog(image_4d, spacing, output_dir, dicom_path, parent)
    if dialog.exec() == QDialog.Accepted:
        return dialog.registered_4d, dialog.metrics
    return None, None
```

### 5. Integrate into run_analysis.py

Replace direct `register_timeseries` call with `run_registration_with_progress` when running in UI mode (not batch).

## Files to Modify

| File | Changes |
|------|---------|
| `proxyl_analysis/registration.py` | Add `progress_callback` parameter |
| `proxyl_analysis/ui.py` | Add `RegistrationWorker`, `RegistrationProgressDialog`, helper function |
| `proxyl_analysis/run_analysis.py` | Use progress dialog in UI mode |

## Acceptance Criteria

- [ ] Registration runs in background thread
- [ ] Progress dialog shows current timepoint and progress bar
- [ ] UI remains responsive during registration
- [ ] Progress dialog closes when registration completes
- [ ] Quality visualization window shows after progress dialog
- [ ] Error handling for registration failures
- [ ] Works when loading new DICOM from main menu

## Dependencies

- None (standalone improvement)

## Blocks

- T006 Phase 1 testing (main menu UX)
