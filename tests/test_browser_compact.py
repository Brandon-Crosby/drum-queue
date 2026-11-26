"""Precision experiments must not alter tensor types or exceptional values."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('compact_model', Path(__file__).resolve().parents[1] / 'tools/browser/compact_model.py')
compact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compact)


class CompactTests(unittest.TestCase):
    def test_ties_round_to_even_and_preserve_sign(self):
        bits = np.array([0x3f800400, 0x3f800c00, 0xbf800400, 0xbf800c00], dtype='<u4')
        result = compact.round_weights(bits.view('<f4'), 12)
        np.testing.assert_array_equal(result.view('<u4'), [0x3f800000, 0x3f801000, 0xbf800000, 0xbf801000])
        self.assertEqual(result.dtype, np.dtype('<f4'))

    def test_special_values_and_overflow_are_unchanged(self):
        bits = np.array([0x7f800000, 0xff800000, 0x7fc01234, 0x7f7fffff, 0x80000000, 0], dtype='<u4')
        np.testing.assert_array_equal(compact.round_weights(bits.view('<f4'), 12).view('<u4'), bits)

    def test_full_precision_is_an_independent_exact_copy(self):
        original = np.array([1.234, -0.14], dtype='<f4')
        result = compact.round_weights(original, 23)
        np.testing.assert_array_equal(result, original)
        self.assertFalse(np.shares_memory(result, original))
        for invalid in [0, 24]:
            with self.assertRaises(ValueError):
                compact.round_weights(original, invalid)


if __name__ == '__main__':
    unittest.main()
