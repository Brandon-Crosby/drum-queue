"""Team mode can pick stage 1 (Demucs vs BS-RoFormer) per upload; the public always gets the default."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'web/backend')]
import service  # noqa: E402


class MixToggleTests(unittest.TestCase):
    def upload(self, team='', mix=''):
        from fastapi.testclient import TestClient
        env = {'MIX_CHOICES': 'htdemucs_ft,bs_roformer_4stem', 'TEAM_CODE': 'team-code-123'}
        with patch.dict(os.environ, env), tempfile.TemporaryDirectory() as d:
            store = service.MemoryStore()
            client = TestClient(service.create_app(d, store, submit=lambda jid: None))
            self.assertEqual(client.get('/api/mixes').json()['choices'], ['htdemucs_ft', 'bs_roformer_4stem'])
            r = client.post('/api/jobs', files={'file': ('song.wav', b'RIFF0000WAVE', 'audio/wav')},
                            data={'team': team, 'mix': mix})
            self.assertEqual(r.status_code, 200, r.text)
            jid = r.json()['id']
            return store.get(f'job:{jid}'), client.get(f'/api/jobs/{jid}').json()

    def test_team_picks_bs_roformer(self):
        job, pub = self.upload(team='team-code-123', mix='bs_roformer_4stem')
        self.assertEqual(job['mix'], 'bs_roformer_4stem')
        self.assertEqual(pub['mix'], 'bs_roformer_4stem')

    def test_public_gets_default_even_if_it_asks(self):
        job, pub = self.upload(mix='bs_roformer_4stem')
        self.assertEqual(job['mix'], 'htdemucs_ft')
        self.assertEqual(pub['mix'], 'htdemucs_ft')

    def test_unknown_choice_falls_back(self):
        job, _ = self.upload(team='team-code-123', mix='something_else')
        self.assertEqual(job['mix'], 'htdemucs_ft')

    def test_run_job_uses_the_jobs_choice(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            src = root / 'jobs/0123456789abcdef/input/song.wav'
            src.parent.mkdir(parents=True)
            src.write_bytes(b'x')
            store = service.MemoryStore()
            store.put('job:0123456789abcdef', {'status': 'queued', 'name': 'song.wav', 'created': 1,
                                               'mix': 'bs_roformer_4stem'})
            with patch('engine.process', return_value={'duration': 9, 'stems': [], 'out_dir': 'song'}) as p:
                service.run_job('0123456789abcdef', root, store, object(), 'htdemucs_ft')
            self.assertEqual(p.call_args[0][0]['opts']['model'], 'bs_roformer_4stem')


if __name__ == '__main__':
    unittest.main()
