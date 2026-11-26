#!/usr/bin/env python3
"""Export authorized Demucs drums + Helyx-4 cores; STFT stays outside ONNX.

Run in the existing pinned model environment with torch 2.8 and onnx installed.
Only load checkpoints you trust: PyTorch checkpoint loading can execute code.
"""
import argparse
import inspect
import json
import sys
import textwrap
import types
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True,
                        help='Installed DrumSplitter root containing runtime/, models/, assets/')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.runtime.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(root / 'runtime/msst'),
                   str(root / 'runtime/vendor/audio_separator/separator/uvr_lib_v5')]
    import torch
    from audio_separator.separator.uvr_lib_v5.demucs.pretrained import get_model
    from utils.settings import get_model_from_config

    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.mha.set_fastpath_enabled(False)
    report = {'precision': 'float32', 'demucs': 'f7e0c4bc', 'helyx': 'helyx-4'}
    model = get_model('f7e0c4bc', repo=root / 'models').cpu().eval()
    length = int(model.segment * model.samplerate)
    if length != 343980 or model.sources[0] != 'drums':
        raise ValueError('Unsupported Demucs shape/source order')
    wave = torch.randn(1, 2, length) * .05
    with torch.inference_mode():
        spectrum = model._magnitude(model._spec(wave))
        original = model(wave)
    # Export real-valued network branches. The complex STFT/iSTFT is in dsp.js.
    source = textwrap.dedent(inspect.getsource(model.forward))
    source = source.replace('def forward(self, mix):', 'def forward(self, mix, spectrum):')
    source = source.replace('z = self._spec(mix)', 'z = None')
    source = source.replace('mag = self._magnitude(z).to(mix.device)', 'mag = spectrum')
    marker = '    # to cpu as non-cuda'
    if marker not in source:
        raise ValueError('Demucs source changed; review the export adapter first')
    source = source[:source.index(marker)] + '''    xt = xt.view(B, S, -1, mix.shape[-1])
    xt = xt * stdt[:, None] + meant[:, None]
    return x, xt
'''
    namespace = dict(model.forward.__globals__)
    exec(source, namespace)
    model.forward = types.MethodType(namespace['forward'], model)
    with torch.inference_mode():
        spectral, temporal = model(wave, spectrum)
        rebuilt = model._ispec(model._mask(None, spectral), length) + temporal
        error = float((rebuilt - original).abs().max())
        if error > 1e-6:
            raise ValueError(f'Demucs export adapter changed output: {error}')
        report['demucs_adapter_max_error'] = error
        torch.onnx.export(model, (wave, spectrum), str(output / 'demucs.onnx'),
                          input_names=['wave', 'spectrum'], output_names=['spectral', 'temporal'],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    del model

    model, config = get_model_from_config('mdx23c', str(root / 'assets/config.yaml'))
    if list(config.training.instruments) != ['kick', 'snare', 'perc', 'hh', 'other'] or config.audio.chunk_size != 130560:
        raise ValueError('Expected Helyx-4 config and source order')
    state = torch.load(root / 'assets/last_mdx23c.ckpt', map_location='cpu', weights_only=False)
    for key in ('state_dict', 'model_state_dict', 'state'):
        if isinstance(state, dict) and key in state:
            state = state[key]
    model.load_state_dict({key.replace('_orig_mod.', ''): value for key, value in state.items()})
    model.eval()
    wave = torch.randn(1, 2, 130560) * .05
    with torch.inference_mode():
        spectrum = model.stft(wave)
        original = model(wave)
    stft = model.stft

    class IdentitySTFT:
        def __call__(self, value):
            return value

        def inverse(self, value):
            return value

    model.stft = IdentitySTFT()
    with torch.inference_mode():
        predicted = model(spectrum)
        rebuilt = stft.inverse(predicted)
        error = float((rebuilt - original).abs().max())
        if error > 1e-6:
            raise ValueError(f'Helyx export adapter changed output: {error}')
        report['helyx_adapter_max_error'] = error
        torch.onnx.export(model, spectrum, str(output / 'helyx.onnx'),
                          input_names=['spectrum'], output_names=['stems'],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    (output / 'export.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
