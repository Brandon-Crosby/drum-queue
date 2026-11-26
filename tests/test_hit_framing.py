"""The hit detector must use the same frames within a 32-bit array-index budget."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'kitmatch'))
import hits


class FramingTests(unittest.TestCase):
    def test_selected_frames_preserve_existing_fft_input(self):
        x = np.random.default_rng(4).normal(0, .01, (5000, 2)).astype(np.float32)
        pad = np.pad(x.mean(1), (hits.N_FFT // 2, hits.N_FFT // 2))
        expected = np.lib.stride_tricks.sliding_window_view(pad, hits.N_FFT)[::hits.HOP] * np.hanning(hits.N_FFT)
        original = np.fft.rfft
        seen = []
        def capture(value, *args, **kwargs):
            seen.append(value.copy())
            return original(value, *args, **kwargs)
        before = x.copy()
        with patch('numpy.fft.rfft', side_effect=capture):
            hits.onsets(x)
        np.testing.assert_array_equal(seen[0], expected)
        np.testing.assert_array_equal(x, before)

    def test_nine_second_input_works_with_32bit_view_limits(self):
        original = np.lib.stride_tricks.sliding_window_view
        def constrained(value, size, *args, **kwargs):
            if (len(value) - size + 1) * size * value.dtype.itemsize > 2**31 - 1:
                raise ValueError('32-bit view exceeds index limit')
            return original(value, size, *args, **kwargs)
        x = np.zeros((9 * hits.SR, 2), dtype=np.float32)
        with patch('numpy.lib.stride_tricks.sliding_window_view', side_effect=constrained):
            self.assertEqual(len(hits.onsets(x)), 0)


if __name__ == '__main__':
    unittest.main()
