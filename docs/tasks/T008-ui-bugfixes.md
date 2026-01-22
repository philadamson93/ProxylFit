# T008: UI Bug Fixes for Image Viewers

**Status**: complete
**Priority**: high
**Created**: 2025-01-22

## Description

Fix several UI bugs in the image viewing components:

1. Registration results viewer: images change size when scrolling through timepoints
2. Image Tools (difference/average): first click point not visible until second point selected
3. Image Tools (difference/average): images appear upside down
4. Image Tools: preview requires clicking "Update Preview" instead of showing automatically

---

## Bug 1: Image Size Changes When Scrolling Registration Results

**Location**: `proxyl_analysis/ui.py`, `RegistrationReviewDialog._update_display()` (lines 614-647)

**Problem**: When navigating through timepoints, the image plots change size slightly. This is caused by calling `tight_layout()` on every update, which recalculates padding and margins.

**Current Code**:
```python
def _update_display(self):
    t = self.current_timepoint
    z = self.z_slice

    # Clear and re-plot each time
    self.ax_ref.clear()
    self.ax_ref.imshow(ref_slice, cmap='gray', origin='lower')
    # ... other axes ...

    self.fig_images.tight_layout()  # Called every update!
    self.canvas_images.draw()
```

**Fix**:
1. Store the imshow objects during initial setup
2. Use `set_data()` to update image data instead of clearing and re-plotting
3. Only call `tight_layout()` once during initialization

**Fixed Code**:
```python
def _setup_ui(self):
    # ... existing setup code ...

    # Initialize image objects (will be updated later)
    self.im_ref = self.ax_ref.imshow(np.zeros((10,10)), cmap='gray', origin='lower')
    self.im_registered = self.ax_registered.imshow(np.zeros((10,10)), cmap='gray', origin='lower')
    self.im_diff = self.ax_diff.imshow(np.zeros((10,10)), cmap='hot', origin='lower')
    self.im_unregistered = self.ax_unregistered.imshow(np.zeros((10,10)), cmap='gray', origin='lower')

    # Set up axes once
    for ax, title in [(self.ax_ref, 'Reference (t=0)'), ...]:
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    self.fig_images.tight_layout()  # Call once only

def _update_display(self):
    t = self.current_timepoint
    z = self.z_slice

    # Get slices
    ref_slice = self.original_4d[:, :, z, 0].T
    orig_slice = self.original_4d[:, :, z, t].T
    reg_slice = self.registered_4d[:, :, z, t].T
    diff_slice = np.abs(ref_slice - reg_slice)

    # Update image data using set_data() - no tight_layout needed
    self.im_ref.set_data(ref_slice)
    self.im_ref.set_clim(vmin=ref_slice.min(), vmax=ref_slice.max())

    self.im_registered.set_data(reg_slice)
    self.im_registered.set_clim(vmin=reg_slice.min(), vmax=reg_slice.max())

    self.im_diff.set_data(diff_slice)
    self.im_diff.set_clim(vmin=diff_slice.min(), vmax=diff_slice.max())

    self.im_unregistered.set_data(orig_slice)
    self.im_unregistered.set_clim(vmin=orig_slice.min(), vmax=orig_slice.max())

    # Update titles if they change
    self.ax_registered.set_title(f'Registered t={t}', fontsize=10)
    self.ax_unregistered.set_title(f'Original t={t} (unregistered)', fontsize=10)

    self.canvas_images.draw()
```

---

## Bug 2: First Click Point Not Visible Until Second Point Selected

**Location**: `proxyl_analysis/ui.py`, `ImageToolsDialog._on_plot_click()` and `_update_plot()` (lines 2851-2994)

**Problem**: When clicking to select a region, the first point (start) is not shown visually until the second point (end) is also selected. User has no feedback that their first click was registered.

**Current Code** (`_update_plot`):
```python
# Only draws when BOTH start and end are set
if self.region_a_start is not None and self.region_a_end is not None:
    self.ax.axvspan(...)
```

**Fix**: Add visual markers when only the start point is selected:

```python
def _update_plot(self):
    self.ax.clear()

    # Plot signal
    self.ax.plot(self.time_array, self.roi_signal, 'k-', linewidth=1.5)

    # Show start point marker while selecting (before end is set)
    if self.region_a_start is not None and self.region_a_end is None:
        self.ax.axvline(self.time_array[self.region_a_start], color='blue',
                        linestyle='-', linewidth=2, alpha=0.8, label='Region A start')
        self.ax.plot(self.time_array[self.region_a_start],
                     self.roi_signal[self.region_a_start],
                     'bo', markersize=10)

    # Highlight region A (blue) - when both are set
    if self.region_a_start is not None and self.region_a_end is not None:
        # ... existing axvspan code ...

    # Same for region B in difference mode
    if self.mode == 'difference':
        if self.region_b_start is not None and self.region_b_end is None:
            self.ax.axvline(self.time_array[self.region_b_start], color='red',
                            linestyle='-', linewidth=2, alpha=0.8, label='Region B start')
            self.ax.plot(self.time_array[self.region_b_start],
                         self.roi_signal[self.region_b_start],
                         'ro', markersize=10)

        if self.region_b_start is not None and self.region_b_end is not None:
            # ... existing axvspan code ...
```

---

## Bug 3: Images Upside Down in Difference/Average Viewer

**Location**: `proxyl_analysis/ui.py`, `ImageToolsDialog._show_preview()` (lines 3023-3045)

**Problem**: Images in the ImageToolsDialog appear upside down compared to the registration viewer. The `origin` parameter is not set in `imshow()`.

**Current Code**:
```python
def _show_preview(self):
    slice_data = self.preview_image[:, :, self.current_z].T

    if self.mode == 'average':
        im = self.preview_ax.imshow(slice_data, cmap='gray', aspect='equal')
        # No origin parameter!
    else:
        im = self.preview_ax.imshow(slice_data, cmap='RdBu_r', aspect='equal',
                                    vmin=vmin, vmax=vmax)
        # No origin parameter!
```

**Fix**: Add `origin='lower'` to match the registration viewer orientation:

```python
def _show_preview(self):
    slice_data = self.preview_image[:, :, self.current_z].T

    if self.mode == 'average':
        im = self.preview_ax.imshow(slice_data, cmap='gray', aspect='equal', origin='lower')
    else:
        vmax = np.max(np.abs(slice_data))
        vmin = -vmax
        im = self.preview_ax.imshow(slice_data, cmap='RdBu_r', aspect='equal',
                                    vmin=vmin, vmax=vmax, origin='lower')
```

---

## Bug 4: Preview Should Show Automatically

**Location**: `proxyl_analysis/ui.py`, `ImageToolsDialog` class

**Problem**: After selecting both start and end points for a region, the user must manually click "Update Preview" to see the image. The preview should appear automatically.

**Current Code** (`_check_can_preview`):
```python
def _check_can_preview(self):
    """Check if we have enough data to preview."""
    if self.mode == 'average':
        can_preview = (self.region_a_start is not None and
                      self.region_a_end is not None)
    else:
        can_preview = (self.region_a_start is not None and
                      self.region_a_end is not None and
                      self.region_b_start is not None and
                      self.region_b_end is not None)

    self.preview_btn.setEnabled(can_preview)
    self.save_btn.setEnabled(can_preview)
    # No auto-preview!
```

**Fix**: Automatically compute and show preview when selection is complete:

```python
def _check_can_preview(self):
    """Check if we have enough data to preview, and auto-show if ready."""
    if self.mode == 'average':
        can_preview = (self.region_a_start is not None and
                      self.region_a_end is not None)
    else:
        can_preview = (self.region_a_start is not None and
                      self.region_a_end is not None and
                      self.region_b_start is not None and
                      self.region_b_end is not None)

    self.preview_btn.setEnabled(can_preview)
    self.save_btn.setEnabled(can_preview)

    # Auto-update preview when selection is complete
    if can_preview:
        self._update_preview()
```

Additionally, rename the button from "Update Preview" to "Refresh Preview" since it will auto-update:
```python
self.preview_btn = QPushButton("Refresh Preview")
```

---

## Implementation Order

1. Bug 3 (upside down images) - Simple one-line fix
2. Bug 4 (auto-preview) - Small change to `_check_can_preview`
3. Bug 2 (first click marker) - Moderate change to `_update_plot`
4. Bug 1 (registration size changes) - Larger refactor of `_update_display`

## Testing

- Verify registration viewer shows consistent image sizes when navigating timepoints
- Verify clicking first point in Image Tools shows a marker immediately
- Verify images in Image Tools match orientation in registration viewer
- Verify preview updates automatically when region selection is complete
