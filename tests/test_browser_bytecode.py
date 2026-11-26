"""Precompiled runtime caches must match the runtime and stay inside cache paths."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location('postprocess_benchmark', Path(__file__).resolve().parents[1] / 'tools/browser/benchmark_postprocess.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BytecodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'pyodide-lock.json').write_text('{}')

    def prepare(self, name='lib/python3.13/site-packages/numpy/__pycache__/test.cpython-313.pyc', flags=3):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr(name, b'\x00' * 4 + flags.to_bytes(4, 'little') + b'\x00' * 8)
        data = stream.getvalue()
        (self.root / 'bytecode.zip').write_bytes(data)
        manifest = dict(runtime='Pyodide 0.29.0', invalidation='checked-hash', bytes=len(data),
                        sha256=hashlib.sha256(data).hexdigest(), runtimeLockSha256=hashlib.sha256(b'{}').hexdigest())
        (self.root / 'manifest.json').write_text(json.dumps(manifest))
        return data

    def test_accepts_matching_checked_hash_cache(self):
        data = self.prepare()
        self.assertEqual(benchmark.verified_bytecode(self.root, self.root)[0], data)

    def test_rejects_stale_runtime_and_corrupted_archive(self):
        self.prepare()
        (self.root / 'pyodide-lock.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'different runtime'):
            benchmark.verified_bytecode(self.root, self.root)
        (self.root / 'pyodide-lock.json').write_text('{}')
        (self.root / 'bytecode.zip').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'verification'):
            benchmark.verified_bytecode(self.root, self.root)

    def test_rejects_archive_escape_and_unchecked_code(self):
        for name in ['../escape.pyc', '/lib/python3.13/__pycache__/test.pyc', 'lib/python3.13/../../escape.pyc', 'lib/python3.13/__pycache__/test.py']:
            self.prepare(name)
            with self.assertRaisesRegex(ValueError, 'path'):
                benchmark.verified_bytecode(self.root, self.root)
        self.prepare(flags=1)
        with self.assertRaisesRegex(ValueError, 'checked-hash'):
            benchmark.verified_bytecode(self.root, self.root)


if __name__ == '__main__':
    unittest.main()
