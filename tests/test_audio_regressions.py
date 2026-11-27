"""Synthetic regressions: no GPU, model downloads, cloud calls or private audio.

Run: python -m unittest discover -s tests -v
"""
import asyncio
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave
from types import SimpleNamespace
from unittest.mock import Mock

import engine

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'web' / 'backend'))
sys.path.insert(0, str(ROOT / 'kitmatch'))
import clean
import groove
import service


class OneShotTests(unittest.TestCase):
    def test_selected_hit_does_not_include_next_onset(self):
        for channels in (1, 2):
            with self.subTest(channels=channels):
                # Identical attacks tie: previously the first, overlapping hit won.
                on = np.array([0, 4500, 14500, 24500])
                nxt = np.array([4500, 14500, 24500, 34500])
                x = np.zeros(35000, dtype=np.float32)
                for i in on:
                    x[i:i + 500] = np.sin(np.arange(500) * .1)
                if channels == 2:
                    x = np.column_stack([x, x])
                before = x.copy()
                seg, pick = clean.typical_hit(x, on, nxt, 8000)
                self.assertIsNotNone(seg)
                self.assertLessEqual(on[pick] + len(seg), nxt[pick])
                np.testing.assert_array_equal(x, before)

    def test_too_few_isolated_hits_keep_averaged_fallback(self):
        on = np.arange(4) * 1000
        seg, pick = clean.typical_hit(np.zeros(6000), on, on + 1000, 2000)
        self.assertIsNone(seg)
        self.assertIsNone(pick)

    def test_isolated_hit_retains_attack_and_fades_tail(self):
        on = np.array([0, 4000, 8000])
        x = np.ones((12000, 2), dtype=np.float32)
        seg, pick = clean.typical_hit(x, on, on + 4000, 2500)
        self.assertIn(pick, range(3))
        np.testing.assert_array_equal(seg[:100], x[:100])
        np.testing.assert_array_equal(seg[-1], [0, 0])


class GrooveTests(unittest.TestCase):
    def test_display_tempo_uses_filename_like_midi_export(self):
        fake = SimpleNamespace(get_duration=Mock(return_value=12.3), load=Mock())
        with patch.dict(sys.modules, {'librosa': fake}):
            bpm, duration = engine.analyze(Path('127.5 bpm - swing.wav'))
        self.assertEqual(bpm, 127.5)
        self.assertEqual(duration, 12.3)
        fake.load.assert_not_called()

    def test_filename_tempo_keeps_decimal(self):
        self.assertEqual(groove.bpm_from_name('beat 127.5 bpm.wav'), 127.5)
        self.assertEqual(groove.bpm_from_name('breakdown136'), 136)
        self.assertIsNone(groove.bpm_from_name('beat 201 bpm'))

    def test_alignment_preserves_pickup_intervals(self):
        times = np.array([.12, .25, .375, .5, .625, .75])
        aligned, _ = groove.align([dict(label='Hat', times=times)], 120)
        self.assertGreaterEqual(aligned[0]['times'].min(), 0)
        # .12 is only 5 ms before the .125 grid line: it lands on the downbeat (clamped, at most 20 ms);
        # every later interval keeps its detected timing. A real pickup: test_alignment_gives_a_real_pickup...
        np.testing.assert_allclose(np.diff(aligned[0]['times'])[1:], np.diff(times)[1:])
        self.assertLessEqual(abs(np.diff(aligned[0]['times'])[0] - np.diff(times)[0]), 0.02)

    def test_alignment_keeps_slightly_early_downbeat_in_bar_one(self):
        # 120 BPM: a 16th is 0.125 s. The first kick lands 5 ms early; the rest sit on the grid.
        kick = np.array([1.995, 2.5, 3.0, 3.5, 4.0])
        hat = np.array([2.0, 2.25, 2.5, 2.75, 3.0])
        aligned, start = groove.align([dict(label='Kick 1', times=kick), dict(label='Hat', times=hat)], 120)
        self.assertGreaterEqual(min(p['times'].min() for p in aligned), 0)
        self.assertAlmostEqual(aligned[0]['times'][0], 0.0, places=6)      # no near-empty bar in front
        self.assertAlmostEqual(aligned[1]['times'][0], 0.0, places=2)
        np.testing.assert_allclose(np.diff(aligned[0]['times'][1:]), np.diff(kick[1:]))

    def test_alignment_gives_a_real_pickup_its_own_bar(self):
        # A hat a full 16th before the first kick is a pickup: the song starts a bar earlier.
        kick = np.array([2.0, 2.5, 3.0, 3.5])
        hat = np.array([1.875, 2.25, 2.75])
        aligned, start = groove.align([dict(label='Kick 1', times=kick), dict(label='Hat', times=hat)], 120)
        bar = 16 * 60 / 120 / 4
        self.assertAlmostEqual(aligned[0]['times'][0], bar, places=2)
        self.assertAlmostEqual(aligned[1]['times'][0], bar - 0.125, places=2)

    def test_full_song_preserves_detected_timing_loop_stays_quantized(self):
        on = np.array([0, 7000, 11000, 18000, 22000])
        parts = [dict(role='hh', label='Hat', onsets=on)]
        written = []
        def capture(path, parts, bpm, title=''):
            written.append((Path(path).name, parts, bpm))
        with tempfile.TemporaryDirectory() as td, \
             patch.object(clean.H, 'load', return_value=np.ones((25000, 2))), \
             patch.object(groove, 'write', side_effect=capture):
            clean.write_groove(td, td, parts, 'beat 127.5 - ', {'hh': 'unused'}, lambda _: None)
        full, loop = written
        self.assertIn('full song', full[0])
        np.testing.assert_allclose(full[1][0]['times'], on / clean.SR)   # same clock as the audio file
        self.assertEqual(full[2], 127.5)
        aligned, _ = groove.align(full[1], full[2])
        quantized = groove.quantize(aligned, full[2])
        np.testing.assert_allclose(loop[1][0]['times'], quantized[0]['times'])

    def test_full_song_midi_lines_up_with_audio_after_a_pickup(self):
        # a hat pickup, then the first kick 0.885 s in: the full-song MIDI must not move by a pickup bar
        sr = clean.SR
        parts = [dict(role='hh', label='Hat 1', onsets=(np.array([0.025, 0.24, 0.455, 0.67, 1.1]) * sr).astype(int)),
                 dict(role='kick', label='Kick 1', onsets=(np.array([0.885, 2.62]) * sr).astype(int))]
        written = []
        with tempfile.TemporaryDirectory() as td, \
             patch.object(clean.H, 'load', return_value=np.ones((4 * sr, 2))), \
             patch.object(groove, 'write', side_effect=lambda path, parts, bpm, title='': written.append((Path(path).name, parts))), \
             patch.object(clean, 'write_loops'):
            clean.write_groove(td, td, parts, 'idea 69 - ', {'hh': 'x', 'kick': 'x'}, lambda _: None)
        full = dict((p['label'], p['times']) for p in written[0][1])
        np.testing.assert_allclose(full['Kick 1'], [0.885, 2.62], atol=1 / sr)
        np.testing.assert_allclose(full['Hat 1'][0], 0.025, atol=1 / sr)


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        settings = service.Settings()
        settings.turnstile_secret = settings.proxy_secret = settings.team_code = ''
        settings.per_ip_per_day = settings.daily_cap = 100
        self.submitted = []
        self.store = service.MemoryStore()
        self.app = service.create_app(self.root, self.store, self.submitted.append, settings=settings)
        wav = io.BytesIO()
        with wave.open(wav, 'wb') as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(22050)
            f.writeframes(b'\0\0' * 2205)
        self.audio = wav.getvalue()

    def upload(self, name):
        boundary = 'audio-regression'
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"'
                '\r\nContent-Type: audio/wav\r\n\r\n').encode() + self.audio
        body += f'\r\n--{boundary}--\r\n'.encode()
        messages = []
        async def receive():
            return {'type': 'http.request', 'body': body, 'more_body': False}
        async def send(message):
            messages.append(message)
        scope = dict(type='http', asgi={'version': '3.0'}, http_version='1.1', scheme='http',
                     method='POST', path='/api/jobs', raw_path=b'/api/jobs', query_string=b'',
                     headers=[(b'content-type', f'multipart/form-data; boundary={boundary}'.encode())],
                     client=('127.0.0.1', 10000), server=('localhost', 80))
        asyncio.run(self.app(scope, receive, send))
        status = next(m['status'] for m in messages if m['type'] == 'http.response.start')
        data = b''.join(m.get('body', b'') for m in messages if m['type'] == 'http.response.body')
        return status, json.loads(data)

    def test_long_audio_filename_keeps_extension_and_original_bytes(self):
        for name in ('a' * 140 + '.wav', 'b' * 116 + '.AIFF', 'normal.wav'):
            with self.subTest(name=name):
                status, result = self.upload(name)
                self.assertEqual(status, 200, result)
                saved = list((self.root / 'jobs' / result['id'] / 'input').iterdir())
                self.assertEqual(len(saved), 1)
                self.assertLessEqual(len(saved[0].name), 120)
                self.assertEqual(saved[0].suffix, Path(name).suffix)
                self.assertEqual(saved[0].read_bytes(), self.audio)
                self.assertIn(result['id'], self.submitted)

    def test_unsupported_long_filename_is_rejected_without_job(self):
        self.assertEqual(self.upload('a' * 140 + '.exe')[0], 400)
        self.assertEqual(self.submitted, [])
        self.assertFalse((self.root / 'jobs').exists())


if __name__ == '__main__':
    unittest.main()


class SilentStemTest(unittest.TestCase):
    def test_bleed_only_and_sparse_stems(self):
        sr = clean.SR
        rng = np.random.default_rng(0)
        def hits(level_db, every, n_sec=20):
            x = np.zeros(n_sec * sr, np.float32)
            for i in range(0, len(x) - 2000, int(every * sr)):
                x[i:i + 2000] = rng.standard_normal(2000) * 10 ** (level_db / 20) * np.exp(-np.arange(2000) / 400)
            return x
        stems = {'kick': hits(-52, 0.25),          # faint hat ticks only: bleed
                 'snare': hits(-6, 1.0), 'perc': hits(-12, 7.0),   # perc: three hits in 20 s, sparse but real
                 'hh': hits(-10, 0.25)}
        self.assertEqual(clean.silent_stems(stems), {'kick'})
        stems['kick'] = np.zeros(20 * sr, np.float32)
        self.assertEqual(clean.silent_stems(stems), {'kick'})
        stems['kick'] = hits(-3, 2.0)
        self.assertEqual(clean.silent_stems(stems), set())
