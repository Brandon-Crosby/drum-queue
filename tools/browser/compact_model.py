#!/usr/bin/env python3
"""Prepare an experimental reduced-precision weight export for local quality tests.

Computation and tensor types remain float32. This is lossy: it must pass audio
and feature checks before deployment. It does not change the live manifest.
Use compress_models.py after staging to measure its HTTP download size.
"""
import argparse
import hashlib
import json
from pathlib import Path


def round_weights(values, mantissa_bits):
    import numpy as np
    if not 1 <= mantissa_bits <= 23:
        raise ValueError('Mantissa bits must be between 1 and 23')
    original = np.asarray(values, dtype='<f4')
    if mantissa_bits == 23:
        return original.copy()
    drop = 23 - mantissa_bits
    bits = original.view('<u4')
    low = bits & ((1 << drop) - 1)
    high = bits >> drop
    halfway = 1 << (drop - 1)
    increment = (low > halfway) | ((low == halfway) & ((high & 1) != 0))
    rounded = ((high + increment.astype('<u4')) << drop).astype('<u4').view('<f4')
    # Preserve nonfinite payloads and avoid overflowing a finite weight.
    valid = np.isfinite(original) & np.isfinite(rounded)
    return np.where(valid, rounded, original).astype('<f4')


def compact(source, output, mantissa_bits):
    import numpy as np
    import onnx
    model = onnx.load(source)
    changed, maximum = 0, 0.0
    for tensor in model.graph.initializer:
        if tensor.data_type != onnx.TensorProto.FLOAT or not tensor.raw_data:
            continue
        values = np.frombuffer(tensor.raw_data, dtype='<f4')
        if values.size < 2:
            continue
        rounded = round_weights(values, mantissa_bits)
        changed += int(np.count_nonzero(values.view('<u4') != rounded.view('<u4')))
        finite = np.isfinite(values)
        if finite.any():
            maximum = max(maximum, float(np.max(np.abs(values[finite] - rounded[finite]))))
        tensor.raw_data = rounded.tobytes()
    onnx.checker.check_model(model)
    onnx.save(model, output)
    digest = hashlib.sha256()
    with output.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(mantissaBits=mantissa_bits, changedWeights=changed,
                maxWeightAbsoluteError=maximum, bytes=output.stat().st_size,
                sha256=digest.hexdigest(), experimental=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mantissa-bits', type=int, choices=range(1, 24), default=12)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    web = Path(__file__).resolve().parents[2] / 'web'
    if source == output or output == web or web in output.parents:
        parser.error('Use a separate experimental output outside web/')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = compact(source, output, args.mantissa_bits)
    output.with_suffix('.report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
