#!/usr/bin/env python3
"""Stage the CPU worker's pinned runtime, naming reference and Python sources."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile
from benchmark_postprocess import verified_bytecode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--bytecode', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='Asset directory containing v1/')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output, web = args.output.resolve(), root / 'web'
    if output == web or web in output.parents:
        parser.error('Keep runtime assets outside web/ and the source package')
    target = output / 'v1'
    runtime = args.runtime.resolve()
    lock = json.loads((runtime / 'pyodide-lock.json').read_text())
    packages = set()
    def include(name):
        if name in packages:
            return
        packages.add(name)
        for dependency in lock['packages'][name]['depends']:
            include(dependency)
    for name in ['numpy', 'scipy', 'soundfile']:
        include(name)
    files = ['pyodide.js', 'pyodide.asm.js', 'pyodide.asm.wasm', 'python_stdlib.zip', 'pyodide-lock.json']
    for name in sorted(packages):
        entry = lock['packages'][name]
        source = runtime / entry['file_name']
        if hashlib.sha256(source.read_bytes()).hexdigest() != entry['sha256']:
            parser.error('Runtime package verification failed: ' + name)
        files.append(entry['file_name'])
    (target / 'runtime').mkdir(parents=True, exist_ok=True)
    for name in files:
        shutil.copyfile(runtime / name, target / 'runtime' / name)
    def save(name, data):
        (target / name).write_bytes(data)
        return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(root / 'engine.py', 'engine.py')
        archive.write(root / 'tools/browser/postprocess_job.py', 'postprocess_job.py')
        for source in sorted((root / 'kitmatch').glob('*.py')):
            archive.write(source, 'kitmatch/' + source.name)
    manifest = dict(schema=1, runtime='pyodide@0.29.0', python=save('python.zip', bundle.getvalue()),
                    reference=save('reference.npz', args.reference.read_bytes()))
    if args.bytecode:
        data, _ = verified_bytecode(args.bytecode, runtime)
        manifest['bytecode'] = save('bytecode.zip', data)
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(output=str(target), bytes=sum(p.stat().st_size for p in target.rglob('*') if p.is_file()))))


if __name__ == '__main__':
    main()
