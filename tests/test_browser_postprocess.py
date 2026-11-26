"""Keep the browser adapter's download contract aligned with existing artifacts."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('browser_postprocess', ROOT / 'tools/browser/postprocess_job.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class PostprocessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / 'input'
        self.inputs.mkdir()
        self.reference = self.root / 'reference.npz'
        self.reference.touch()
        for name in ['kick', 'snare', 'perc', 'hh', 'drums', 'bass', 'vocals', 'other']:
            sf.write(self.inputs / (name + '.wav'), np.zeros((4410, 2)), 44100, subtype='PCM_24')

    def finish(self, **kwargs):
        return adapter.finish('test.wav', self.inputs, self.root / 'output', self.reference, **kwargs)

    def test_requires_reference_and_full_mix_by_default(self):
        self.reference.unlink()
        with self.assertRaisesRegex(ValueError, 'reference'):
            self.finish()
        self.reference.touch()
        (self.inputs / 'bass.wav').unlink()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.finish()

    def test_rejects_mismatched_stems_before_processing(self):
        sf.write(self.inputs / 'hh.wav', np.zeros((100, 2)), 44100)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            self.finish()

    def test_optional_sample_failure_preserves_stems(self):
        with patch('kitmatch.clean.write_kit', side_effect=RuntimeError('test failure')):
            result = self.finish()
        self.assertEqual(len(result['stems']), 8)
        self.assertTrue(result['fullMix'])
        self.assertTrue(result['note'])
        self.assertEqual(result['shots'], [])

    def test_artifact_groups_and_zip_members(self):
        def samples(source, target, **kwargs):
            target.mkdir()
            sf.write(target / 'test - Kick 1.wav', np.zeros((100, 2)), 44100)
            sf.write(target / 'test - Kick loop 120 BPM (2 bars).wav', np.zeros((100, 2)), 44100)
            (target / 'test - Groove.mid').write_bytes(b'MThd')
            return [dict(name='test - Kick 1.wav', label='Kick', piece='kick', seconds=.1, count=1)]
        with patch('kitmatch.clean.write_kit', side_effect=samples):
            result = self.finish()
        self.assertEqual(result['groove'], 'test - Groove.mid')
        self.assertEqual(result['loops'][0]['bpm'], 120)
        self.assertEqual(result['stems'][2]['label'], 'percussion')
        for kind, count in [('kit', 4), ('all', 8), ('shots', 3)]:
            artifact = next(x for x in result['artifacts'] if x['kind'] == 'zip' and x['name'] == kind)
            with zipfile.ZipFile(artifact['path']) as archive:
                self.assertEqual(len(archive.namelist()), count)
                if kind == 'shots':
                    self.assertIn('test - one-shots/Loops/test - Kick loop 120 BPM (2 bars).wav', archive.namelist())
                else:
                    self.assertIn('test/kick.wav', archive.namelist())


if __name__ == '__main__':
    unittest.main()
