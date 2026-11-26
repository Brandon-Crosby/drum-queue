#!/usr/bin/env python3
"""Build the small DSP WebAssembly module with Zig 0.14.1's C compiler."""
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--zig', default='zig', help='Path to Zig 0.14.1 executable')
parser.add_argument('--output', type=Path, help='Separate diagnostic module output')
parser.add_argument('--simd', action='store_true', help='Test SIMD compilation without changing arithmetic precision')
args = parser.parse_args()
if args.simd and not args.output:
    parser.error('Use --output for a separate SIMD candidate')
if subprocess.check_output([args.zig, 'version'], text=True).strip() != '0.14.1':
    parser.error('Expected Zig 0.14.1 for a reproducible module')
root = Path(__file__).resolve().parents[2]
output = args.output or root / 'web/site/browser/dsp.wasm'
if args.simd and output.resolve() == (root / 'web/site/browser/dsp.wasm').resolve():
    parser.error('Keep the SIMD candidate separate from the product module')
output.parent.mkdir(parents=True, exist_ok=True)
subprocess.run([args.zig, 'cc', '-target', 'wasm32-freestanding', '-O3',
                *(['-msimd128'] if args.simd else []),
                '-ffp-contract=off', '-fno-builtin', '-nostdlib',
                '-Wl,--no-entry', '-Wl,--export-memory',
                '-Wl,--initial-memory=41943040', '-Wl,--max-memory=41943040',
                str(root / 'tools/browser/dsp.c'), '-o',
                str(output)], check=True)

output.chmod(0o644)
