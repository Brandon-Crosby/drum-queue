"""Loops: listed on the job, playable through the shots endpoint, nothing else on disk is."""
import sys, time, unittest, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'web' / 'backend')]

from fastapi.testclient import TestClient
import service


class LoopPreviewTest(unittest.TestCase):
    def test_loops_listed_and_served(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / 'jobs' / '0123456789abcdef' / 'out' / 'song' / 'one-shots'
            d.mkdir(parents=True)
            for n in ['song - Kick 1.wav', 'song - Snare-Perc loop (4 bars).wav', 'song - secret.wav']:
                (d / n).write_bytes(b'RIFF')
            store = service.MemoryStore()
            store.put('job:0123456789abcdef', dict(status='done', name='song.mp3', out_dir='song', created=time.time(),
                                     shots=[dict(name='song - Kick 1.wav', label='Kick 1', piece='kick', seconds=.5, count=9)],
                                     loops=[dict(name='song - Snare-Perc loop (4 bars).wav', label='Snare-Perc loop', bars=4)]))
            c = TestClient(service.create_app(td, store, submit=lambda jid: None))
            j = c.get('/api/jobs/0123456789abcdef').json()
            self.assertEqual(j['loops'], [dict(name='song - Snare-Perc loop (4 bars).wav', label='Snare-Perc loop', bars=4, bpm=None)])
            self.assertEqual(c.get('/api/jobs/0123456789abcdef/shots/song - Snare-Perc loop (4 bars).wav').status_code, 200)
            self.assertEqual(c.get('/api/jobs/0123456789abcdef/shots/song - Kick 1.wav').status_code, 200)
            self.assertEqual(c.get('/api/jobs/0123456789abcdef/shots/song - secret.wav').status_code, 404)

    def test_engine_lists_loops(self):
        import re
        src = (ROOT / 'engine.py').read_text()
        pat = re.search(r"re\.search\(r'(.+?)', f\.name\)", src).group(1)
        m = re.search(pat, 'OMAY - KUIKI - Snare-Perc loop (4 bars).wav')          # splits made before the tempo
        self.assertEqual((m.group(1), m.group(2), m.group(3)), ('Snare-Perc', None, '4'))
        m = re.search(pat, 'OMAY - KUIKI - Snare-Perc loop 113 BPM (4 bars).wav')
        self.assertEqual((m.group(1), m.group(2), m.group(3)), ('Snare-Perc', '113', '4'))
        m = re.search(pat, 'idea - 127 - Hats loop 127.5 BPM (2 bars).wav')
        self.assertEqual((m.group(1).strip(), m.group(2), m.group(3)), ('Hats', '127.5', '2'))   # engine strips

    def test_bpm_label(self):
        sys.path.insert(0, str(ROOT / 'kitmatch'))
        import clean
        self.assertEqual([clean.bpm_label(b) for b in (113.02, 127.5, 69.0, 89.96)], ['113', '127.5', '69', '90'])


if __name__ == '__main__':
    unittest.main()
