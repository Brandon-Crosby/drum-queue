import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('compress_models', Path(__file__).resolve().parents[1] / 'tools/browser/compress_models.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CompressionTests(unittest.TestCase):
    def fixture(self, root):
        assets = root / 'assets'
        assets.mkdir()
        manifest = {'schema': 1, 'models': {}}
        for name, data in [('demucs', b'\x00\x01' * 600000), ('helyx', bytes(range(256)) * 5000)]:
            sha = hashlib.sha256(data).hexdigest()
            (assets / f'{name}-{sha}.onnx').write_bytes(data)
            manifest['models'][name] = {'sha256': sha, 'bytes': len(data)}
        (assets / 'manifest.json').write_text(json.dumps(manifest))
        return assets, manifest

    def test_sidecars_round_trip_without_changing_models_or_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            assets, manifest = self.fixture(root)
            before = {path.name: path.read_bytes() for path in assets.iterdir()}
            report = module.compress(assets, root / 'output')
            self.assertEqual(len(report), 2)
            for entry in report:
                raw = (assets / entry['model']).read_bytes()
                self.assertEqual(gzip.decompress((root / 'output' / (entry['model'] + '.gz')).read_bytes()), raw)
                self.assertLess(entry['gzipBytes'], entry['rawBytes'])
            self.assertEqual({path.name: path.read_bytes() for path in assets.iterdir()}, before)
            self.assertFalse(list((root / 'output').glob('*.tmp')))

    def test_mismatched_model_is_not_compressed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            assets, manifest = self.fixture(root)
            entry = manifest['models']['demucs']
            (assets / f"demucs-{entry['sha256']}.onnx").write_bytes(b'X' * entry['bytes'])
            with self.assertRaisesRegex(ValueError, 'hash does not match'):
                module.compress(assets, root / 'output')
            self.assertFalse(list((root / 'output').glob('*.gz')))


if __name__ == '__main__':
    unittest.main()
