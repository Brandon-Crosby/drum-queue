#!/usr/bin/env python3
"""Stage authorized float32 ONNX exports and pinned ORT assets outside the site tree."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demucs', type=Path, required=True)
    parser.add_argument('--helyx', type=Path, required=True)
    parser.add_argument('--ort', type=Path, required=True, help='onnxruntime-web package directory (1.23.2)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    site = Path(__file__).resolve().parents[2] / 'web'
    target = args.output.resolve()
    if target == site or site in target.parents:
        parser.error('Keep model assets outside web/; deploy them separately from the source package.')
    package = json.loads((args.ort / 'package.json').read_text())
    if package['version'] != '1.23.2':
        parser.error('Expected onnxruntime-web 1.23.2')
    target.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 1, 'runtime': 'onnxruntime-web@1.23.2', 'models': {}}
    for name, source in [('demucs', args.demucs), ('helyx', args.helyx)]:
        size = source.stat().st_size
        if not 0 < size <= 512 * 1024 * 1024:
            parser.error(f'Invalid {name} model size')
        with source.open('rb') as stream:
            hasher = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(block)
            digest = hasher.hexdigest()
        shutil.copyfile(source, target / f'{name}-{digest}.onnx')
        manifest['models'][name] = {'sha256': digest, 'bytes': size}
    # Keep matching WASM + JS modules together, including ORT's asyncify dependency.
    shutil.copytree(args.ort / 'dist', target / 'ort-1.23.2', dirs_exist_ok=True)
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(target)


if __name__ == '__main__':
    main()
