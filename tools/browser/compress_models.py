#!/usr/bin/env python3
"""Create lossless HTTP gzip sidecars for verified browser models outside web/.

Serve each sidecar at the model's canonical .onnx URL with Content-Encoding:
gzip. Browser decoding restores the original bytes before the worker's SHA-256
check. This command does not publish models or change the site's server setup.
"""
import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path


def digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()


def compress(assets, output):
    manifest = json.loads((assets / 'manifest.json').read_text())
    if manifest.get('schema') != 1 or set(manifest.get('models', {})) != {'demucs', 'helyx'}:
        raise ValueError('Expected the two browser models in a schema-1 manifest')
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for name, entry in manifest['models'].items():
        if not re.fullmatch('[a-f0-9]{64}', entry.get('sha256', '')):
            raise ValueError('Invalid model hash')
        source = assets / f"{name}-{entry['sha256']}.onnx"
        if source.stat().st_size != entry['bytes']:
            raise ValueError('Model size does not match manifest')
        with source.open('rb') as stream:
            if digest(stream) != entry['sha256']:
                raise ValueError('Model hash does not match manifest')
        destination = output / (source.name + '.gz')
        temporary = destination.with_suffix('.gz.tmp')
        try:
            with source.open('rb') as src, temporary.open('wb') as dst:
                with gzip.GzipFile(filename='', fileobj=dst, mode='wb', compresslevel=6, mtime=0) as compressed:
                    for block in iter(lambda: src.read(1024 * 1024), b''):
                        compressed.write(block)
            with gzip.open(temporary, 'rb') as stream:
                if digest(stream) != entry['sha256']:
                    raise ValueError('Compressed model failed lossless verification')
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        results.append({'model': source.name, 'rawBytes': entry['bytes'],
                        'gzipBytes': destination.stat().st_size,
                        'roundTripSha256': entry['sha256']})
    (output / 'report.json').write_text(json.dumps(results, indent=2) + '\n')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True, help='Directory containing manifest.json and .onnx files')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    site = Path(__file__).resolve().parents[2] / 'web'
    output = args.output.resolve()
    if output == site or site in output.parents:
        parser.error('Keep model files outside web/ and the desktop source package')
    print(json.dumps(compress(args.assets.resolve(), output), indent=2))


if __name__ == '__main__':
    main()
