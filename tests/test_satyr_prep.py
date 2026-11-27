"""Satyr (6 drum stems): toms/cymbals by track name, and prep's 7-stem layout next to Helyx's 5."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'datagen'))
sys.path.insert(0, str(ROOT / 'train'))
import drumroles  # noqa: E402
import prep  # noqa: E402


class DrumRolesTest(unittest.TestCase):
    def test_names_from_real_sessions(self):
        cases = {'Lo Tom': 'toms', '- Tom': 'toms', 'MOONBOY - Tom 7': 'toms', 'PML_MH4_Perc_003_Tom_Alone': 'toms',
                 'TT1': 'toms', 'Floor Tom 1': 'toms', 'Crash 2': 'cymbals', 'Ride': 'cymbals', 'cym_1': 'cymbals',
                 'reverse crash': 'cymbals', 'LilGeniuz_Perc_TomZay': 'perc', 'custom perc': 'perc',
                 'atomic': 'perc', 'override fx': 'perc', 'Shaker': 'perc', 'Perc (Zay-1)': 'perc'}
        for name, want in cases.items():
            self.assertEqual(drumroles.kind(name), want, name)

    def test_split_key_only_for_other_drums(self):
        self.assertIsNone(drumroles.split_key('kick', 'Tom'))
        self.assertEqual(drumroles.split_key('other_drums', 'Perc 2', 'Tom low.wav'), 'toms')
        self.assertEqual(drumroles.split_key('other_drums', 'Conga'), 'perc_only')


class PrepSixTest(unittest.TestCase):
    def song(self, root, name, others, split=False):
        d = root / name
        d.mkdir(parents=True)
        for k in ['kick', 'snare', 'hh', 'other_drums', 'rest'] + (['toms', 'cymbals', 'perc_only'] if split else []):
            sf.write(str(d / f'{k}.flac'), (np.random.randn(44100 * 6, 2) * 0.05).astype('float32'), 44100,
                     subtype='PCM_24')
        (d / 'roles.json').write_text(json.dumps({'roles': {'other_drums': others}}))
        return d

    def test_six_and_helyx_layouts(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            plain = self.song(t / 'src', 'plain', ['Conga'])
            old = self.song(t / 'src', 'old', ['Lo Tom'])
            new = self.song(t / 'src', 'new', ['Lo Tom', 'Crash'], split=True)
            six = sorted(f'{k}.flac' for k in prep.INSTRUMENTS6)
            self.assertTrue(prep.real_song(plain, t / 'o6' / 'plain', False, six=True).startswith('train'))
            self.assertEqual(sorted(p.name for p in (t / 'o6' / 'plain').iterdir()), six)
            self.assertTrue(prep.real_song(old, t / 'o6' / 'old', False, six=True).startswith('skip'))
            prep.real_song(new, t / 'o6' / 'new', False, six=True)
            self.assertEqual(sorted(p.name for p in (t / 'o6' / 'new').iterdir()), six)
            prep.real_song(new, t / 'o5' / 'new', False)          # Helyx: unchanged 5 stems
            self.assertEqual(sorted(p.name for p in (t / 'o5' / 'new').iterdir()),
                             sorted(f'{k}.flac' for k in prep.INSTRUMENTS))


if __name__ == '__main__':
    unittest.main()
