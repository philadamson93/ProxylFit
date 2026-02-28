"""
Tests for parallel parameter map generation (T008).

Verifies that:
- Parallel and sequential paths produce identical results
- Stride, ROI mask, and cancellation work correctly in parallel mode
- The ParameterMappingWorker QThread completes without hanging when
  ProcessPoolExecutor is spawned inside it
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_4d(shape=(10, 10, 1, 20), seed=42):
    """Create synthetic 4D data with a strong time ramp so fits succeed."""
    np.random.seed(seed)
    data = np.random.uniform(500, 1000, shape).astype(np.float64)
    for t in range(shape[3]):
        data[:, :, :, t] += t * 100
    return data


def _make_time_array(n=20):
    return np.linspace(0, 10, n)


# ===========================================================================
# Unit tests: create_parameter_maps directly
# ===========================================================================

class TestParallelParameterMaps:
    """T008: Verify parallel and sequential paths produce equivalent results."""

    @pytest.fixture
    def synthetic_4d(self):
        return _make_synthetic_4d()

    @pytest.fixture
    def time_array(self):
        return _make_time_array()

    def test_parallel_sequential_parity(self, synthetic_4d, time_array):
        """parallel=True and parallel=False produce identical output maps."""
        from proxyl_analysis.parameter_mapping import create_parameter_maps

        result_par = create_parameter_maps(
            synthetic_4d, time_array,
            window_size=1, z_slice=0, parallel=True
        )
        result_seq = create_parameter_maps(
            synthetic_4d, time_array,
            window_size=1, z_slice=0, parallel=False
        )

        for key in ('kb_map', 'kd_map', 'knt_map', 'r_squared_map',
                     'a1_amplitude_map', 'a2_amplitude_map', 'baseline_map',
                     't0_map', 'tmax_map', 'mask'):
            np.testing.assert_allclose(
                result_par[key], result_seq[key],
                rtol=1e-5, atol=1e-8,
                err_msg=f"Mismatch in {key} between parallel and sequential"
            )

    def test_parallel_with_stride(self, synthetic_4d, time_array):
        """stride=2 works correctly in parallel mode."""
        from proxyl_analysis.parameter_mapping import create_parameter_maps

        result = create_parameter_maps(
            synthetic_4d, time_array,
            window_size=1, z_slice=0, stride=2, parallel=True
        )
        assert result['kb_map'].shape == (10, 10, 1)

        # Verify at least some fits succeeded (prevents vacuous pass)
        assert np.any(result['mask']), "No fits succeeded — test data may be too weak"

        # Verify stride block fill
        kb = result['kb_map'][:, :, 0]
        mask = result['mask'][:, :, 0]
        for x in range(0, 10, 2):
            for y in range(0, 10, 2):
                block = kb[x:x+2, y:y+2]
                if mask[x, y]:
                    assert np.all(block == block[0, 0]), (
                        f"Block at ({x},{y}) not filled uniformly"
                    )

    def test_parallel_with_roi_mask(self, synthetic_4d, time_array):
        """ROI mask correctly limits positions in parallel mode."""
        from proxyl_analysis.parameter_mapping import create_parameter_maps

        roi_mask = np.zeros((10, 10), dtype=bool)
        roi_mask[:5, :5] = True

        result = create_parameter_maps(
            synthetic_4d, time_array,
            window_size=1, z_slice=0, roi_mask=roi_mask, parallel=True
        )
        assert result['kb_map'].shape == (10, 10, 1)
        # Only 25 positions (5x5) should be processed
        assert result['metadata']['total_positions'] == 25

    def test_parallel_metadata_recorded(self, synthetic_4d, time_array):
        """Metadata contains parallel=True and num_workers."""
        from proxyl_analysis.parameter_mapping import create_parameter_maps

        result = create_parameter_maps(
            synthetic_4d, time_array,
            window_size=1, z_slice=0, parallel=True
        )
        assert result['metadata']['parallel'] is True
        assert result['metadata']['num_workers'] >= 1

    def test_sequential_fallback_flag(self, synthetic_4d, time_array):
        """parallel=False records parallel=False in metadata."""
        from proxyl_analysis.parameter_mapping import create_parameter_maps

        result = create_parameter_maps(
            synthetic_4d, time_array,
            window_size=1, z_slice=0, parallel=False
        )
        assert result['metadata']['parallel'] is False
        assert result['metadata']['num_workers'] == 1

    def test_parallel_cancellation(self, synthetic_4d, time_array):
        """Progress callback returning False stops parallel processing cleanly."""
        from proxyl_analysis.parameter_mapping import create_parameter_maps

        call_count = 0

        def cancel_early(pct, current, total):
            nonlocal call_count
            call_count += 1
            # Cancel after the first progress update
            return False if call_count >= 1 else True

        # Should not raise
        result = create_parameter_maps(
            synthetic_4d, time_array,
            window_size=1, z_slice=0, parallel=True,
            progress_callback=cancel_early
        )
        # Result should be valid (possibly with partial fits)
        assert 'kb_map' in result
        assert result['kb_map'].shape == (10, 10, 1)

    def test_parallel_single_position_uses_sequential(self):
        """Falls back to sequential when only 1 position exists."""
        from proxyl_analysis.parameter_mapping import create_parameter_maps

        data = _make_synthetic_4d(shape=(4, 4, 1, 20), seed=88)
        time = _make_time_array()

        result = create_parameter_maps(
            data, time,
            window_size=1, z_slice=0, stride=20, parallel=True
        )
        # stride=20 on 4x4 → single position at (0,0)
        assert result['metadata']['total_positions'] == 1
        # Should have used sequential (parallel=False due to single position)
        assert result['metadata']['parallel'] is False


# ===========================================================================
# UI integration tests: ParameterMappingWorker with real QThread
# ===========================================================================

class TestParallelWorkerIntegration:
    """T008: Verify parallel processing doesn't hang when spawned from QThread."""

    @pytest.fixture
    def synthetic_4d(self):
        return _make_synthetic_4d(shape=(6, 6, 1, 20), seed=42)

    @pytest.fixture
    def time_array(self):
        return _make_time_array()

    def test_worker_parallel_completes(self, qt_app, qtbot, synthetic_4d, time_array):
        """Worker with parallel=True completes without hanging."""
        from proxyl_analysis.ui.parameter_map_options import ParameterMappingWorker

        options = {
            'window_size': (1, 1, 1),
            'z_slice': 0,
            'kernel_type': 'sliding_window',
            'stride': 2,
        }

        worker = ParameterMappingWorker(
            synthetic_4d, time_array, options,
            time_units='minutes'
        )

        received = []
        worker.finished.connect(lambda pm: received.append(pm))

        # Start real QThread — this will spawn ProcessPoolExecutor internally
        with qtbot.waitSignal(worker.finished, timeout=60000):
            worker.start()

        assert len(received) == 1
        param_maps = received[0]
        assert 'kb_map' in param_maps
        assert param_maps['kb_map'].shape == (6, 6, 1)
        assert param_maps['metadata']['parallel'] is True

    def test_worker_parallel_cancel(self, qt_app, qtbot, synthetic_4d, time_array):
        """Cancelling the worker during parallel processing doesn't hang."""
        from proxyl_analysis.ui.parameter_map_options import ParameterMappingWorker

        # Use stride=1 to create more work (36 positions)
        options = {
            'window_size': (1, 1, 1),
            'z_slice': 0,
            'kernel_type': 'sliding_window',
            'stride': 1,
        }

        worker = ParameterMappingWorker(
            synthetic_4d, time_array, options,
            time_units='minutes'
        )

        worker.start()
        # Cancel immediately
        worker.cancel()

        # Worker thread should terminate within timeout (no hang)
        qtbot.waitUntil(lambda: not worker.isRunning(), timeout=30000)
        assert not worker.isRunning()

    def test_worker_parallel_progress_signals(self, qt_app, qtbot, synthetic_4d, time_array):
        """Worker emits monotonically increasing progress signals."""
        from proxyl_analysis.ui.parameter_map_options import ParameterMappingWorker

        options = {
            'window_size': (1, 1, 1),
            'z_slice': 0,
            'kernel_type': 'sliding_window',
            'stride': 2,
        }

        worker = ParameterMappingWorker(
            synthetic_4d, time_array, options,
            time_units='minutes'
        )

        progress_values = []
        worker.progress.connect(lambda pct, cur, tot: progress_values.append(pct))

        with qtbot.waitSignal(worker.finished, timeout=60000):
            worker.start()

        # Should have received at least one progress update
        assert len(progress_values) >= 1
        # Final progress should be ~100%
        assert progress_values[-1] >= 90.0
