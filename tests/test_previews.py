"""Preview encoding must retain audible high frequencies and stereo separation."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf
from mutagen.mp3 import MP3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine


class PreviewTests(unittest.TestCase):
    def test_stereo_high_frequency_survives_mp3_and_wav(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rate = 44100
            time = np.arange(rate * 2) / rate
            high = 0.15 * np.sin(2 * np.pi * 16000 * time)
            # Opposite-polarity high end would disappear in a mono downmix.
            stereo = np.stack([high + 0.1 * np.sin(2 * np.pi * 500 * time),
                               -high + 0.1 * np.sin(2 * np.pi * 700 * time)], axis=1)
            source = root / 'stem.wav'
            sf.write(source, stereo, rate, subtype='PCM_24')
            original = source.read_bytes()
            for extension in ['mp3', 'wav']:
                destination = root / ('preview.' + extension)
                engine.save_preview(source, destination)
                decoded, output_rate = sf.read(destination, always_2d=True)
                self.assertEqual(output_rate, rate)
                self.assertEqual(decoded.shape[1], 2)
                self.assertLess(abs(len(decoded) - len(stereo)), rate * 0.05)
                # Measure the actual 16 kHz signal, not just the file header.
                projection = np.exp(-2j * np.pi * 16000 * np.arange(len(decoded)) / rate)
                amplitude = 2 * abs(np.sum(decoded * projection[:, None], axis=0)) / len(decoded)
                self.assertTrue(np.all(amplitude > 0.12), amplitude)
                self.assertGreater(np.sqrt(np.mean((decoded[:, 0] - decoded[:, 1]) ** 2)), 0.15)
            self.assertEqual(source.read_bytes(), original, 'Preview generation changed the downloadable stem')
            self.assertAlmostEqual(MP3(root / 'preview.mp3').info.bitrate, 192000, delta=1000)
            self.assertEqual(sf.info(root / 'preview.wav').subtype, 'PCM_24')

    def test_mono_input_remains_mono(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            time = np.arange(22050) / 22050
            source = root / 'mono.wav'
            sf.write(source, 0.1 * np.sin(2 * np.pi * 1000 * time), 22050)
            destination = root / 'preview.wav'
            engine.save_preview(source, destination)
            info = sf.info(destination)
            self.assertEqual(info.channels, 1)
            self.assertEqual(info.samplerate, 44100)
            self.assertAlmostEqual(info.duration, 1, places=3)


if __name__ == '__main__':
    unittest.main()
