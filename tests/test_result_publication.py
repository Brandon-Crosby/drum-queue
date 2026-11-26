"""A cloud result must not become downloadable before its files are published."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'web/backend')]
import service


class PublicationTests(unittest.TestCase):
    def run_case(self, fail=False, cancel=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'jobs/0123456789abcdef/input/song.wav'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'input')
            store = service.MemoryStore()
            store.put('job:0123456789abcdef', {'status': 'queued', 'name': 'song.wav', 'created': 1})
            calls = []

            def publish():
                calls.append(store.get('job:0123456789abcdef')['status'])
                if cancel:
                    service.update_job(store, '0123456789abcdef', cancelled=True, status='cancelled')
                if fail:
                    raise OSError('storage unavailable')

            with patch('engine.process', return_value={'duration': 9, 'stems': [], 'out_dir': 'song'}):
                service.run_job('0123456789abcdef', root, store, object(), 'htdemucs_ft', publish=publish)
            return calls, store.get('job:0123456789abcdef')

    def test_cancel_during_publish_stays_cancelled(self):
        calls, job = self.run_case(cancel=True)
        self.assertEqual(calls, ['running'])
        self.assertEqual(job['status'], 'cancelled')

    def test_missing_preview_errors_are_not_cached(self):
        from fastapi.testclient import TestClient
        with tempfile.TemporaryDirectory() as directory:
            store = service.MemoryStore()
            store.put('job:0123456789abcdef', {'status': 'done', 'out_dir': 'song',
                      'stems': [{'name': 'hh.wav', 'preview': 'hh.mp3'}]})
            client = TestClient(service.create_app(directory, store, submit=lambda jid: None))
            response = client.get('/api/jobs/0123456789abcdef/preview/hh.mp3')
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.headers['cache-control'], 'no-store')

    def test_publish_before_announcing_done(self):
        calls, job = self.run_case()
        self.assertEqual(calls, ['running'])
        self.assertEqual(job['status'], 'done')

    def test_failed_publish_does_not_announce_ready_downloads(self):
        calls, job = self.run_case(fail=True)
        self.assertEqual(calls, ['running'])
        self.assertEqual(job['status'], 'error')


if __name__ == '__main__':
    unittest.main()
