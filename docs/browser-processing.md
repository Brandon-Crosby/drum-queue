# Experimental browser processing

Status: experimental backend under review. The live site does not select it yet.
First-use speed parity and the full website result contract remain release gates.

The existing picker, mixer, EQ, one-shot, loop and download controls remain the
UI contract. The background processor never renders controls or changes page DOM;
this change adds no alternate interface.

`processor.js` currently exposes a UI-independent `DrumBrowserProcessor.process`
API, returning progress events and WAV buffers. It is not wired into the public
app yet. That integration must preserve existing features and meet the first-use
performance gate; a reduced replacement interface is not an acceptable shortcut.

The processor decodes mono/stereo audio at 44.1 kHz and transfers PCM to a dedicated
Web Worker. That worker runs float32 Demucs `f7e0c4bc` (the drums member of
`htdemucs_ft`), then Helyx-4 through ONNX Runtime Web 1.23.2 with WebGPU. Float64 WebAssembly
STFT/iSTFT surrounds the real-valued exported networks; dsp.js remains the
numerical reference. The small dsp.wasm module is built from tools/browser/dsp.c
with Zig 0.14.1, without fast-math. Some bookkeeping and
operators can run on CPU; this is GPU acceleration, not a guarantee that every
operation is on GPU.

Demucs uses deterministic zero-shift inference, normalization and 25% overlap;
Helyx uses overlap 4, reflection padding and linear crossfades. Cloud settings
with random shifts/ensembles can produce different results. Four PCM24 WAVs are
returned: kick, snare, percussion and hi-hat. The separation API returns only these four stems. An optional CPU pipeline
prepares one-shots, loops, MIDI and ZIPs from them; full-mix outputs and the
existing website result adapter remain unimplemented.

## Limits and deployment

- 30 seconds / 30 MB per clip. This is a conservative product bound, not a
  measured maximum supported by every device. Long songs, phones and other GPUs
  are not validated.
- Requires a secure context and working WebGPU. Keep the tab open; sleep, browser
  throttling, memory pressure or closing the tab can interrupt work. A worker
  keeps computation off the page thread; it does not persist after closing it.
- Float32 models total 611,957,925 bytes, plus runtime assets. Cache Storage avoids
  re-downloading verified models when available; the browser can evict them.
  Each new worker still creates fresh inference sessions.
- Cancellation terminates the worker. Failed/missing/corrupt assets never cause
  an audio upload or paid cloud fallback.
- No auth tokens, model weights, third-party runtime binaries, recordings or inference
  endpoints are included in the source package. Model distribution permissions
  and hosting must be reviewed separately before public availability.
- Serve models under `/browser-assets/v1/` on the site's origin, using object
  storage and an appropriate route/proxy. These files exceed a normal Pages
  static-asset limit; do not place them in `web/site` or commit them to git.
  Static asset bandwidth/storage still costs resources even without cloud GPU
  inference. Serve `.wasm` as `application/wasm` and matching ORT JS/modules.
- Model filenames include SHA-256. The worker checks size and digest even on
  cache hits. Publish models/runtime before switching the manifest; keep old
  immutable filenames available. `/browser/*` is revalidated to avoid stale
  worker dependencies. The manifest is trusted same-origin configuration, not a
  signed third-party update mechanism.

## Reproduce locally

The committed DSP module can be rebuilt with `python tools/browser/build_dsp.py
--zig /path/to/zig` using Zig 0.14.1. Node tests compare its output with the
JavaScript reference for both model shapes. No model precision is reduced.

Use the existing pinned MSST commit `1d4b3c431b37de696b3795ed2018493aacba946f`,
Helyx-4 checkpoint/config, Demucs `f7e0c4bc` and torch 2.8. Run the export with the
managed environment's vendor modules on `PYTHONPATH`, plus `onnx`. Only load
trusted PyTorch checkpoints. `--runtime` points to an authorized installed root
containing `runtime/msst`, `runtime/vendor`, `assets`, and `models`.

```sh
python tools/browser/export_models.py --runtime /path/to/DrumSplitter --output /tmp/browser-models
npm install --prefix /tmp/browser-ort onnxruntime-web@1.23.2
python tools/browser/prepare_assets.py --demucs /tmp/browser-models/demucs.onnx --helyx /tmp/browser-models/helyx.onnx --ort /tmp/browser-ort/node_modules/onnxruntime-web --output /tmp/browser-assets/v1
python web/build_site.py
python tools/browser/serve.py --assets /tmp/browser-assets
```

The static server exposes worker assets only. It has no POST or inference
endpoint, and does not provide a replacement application page. For development,
load `/browser/processor.js` in an isolated test harness or the existing app's
future processing adapter:

```js
const controller = new AbortController();
const result = await DrumBrowserProcessor.process(file, {
  signal: controller.signal,
  onProgress: event => updateExistingJobProgress(event),
});
// Return result.files to the existing result adapter; no controls are rendered here.
```

CI runs the existing audio/API/UI and desktop package checks plus browser input,
cancellation/retry, WAV encoding, cache/hash validation, tensor/session cleanup,
and no-fallback tests. Actual GPU model runs are manual because CI has neither
licensed weights nor representative physical GPU hardware.

## Sample preparation and combined pipeline

`postprocess.js` owns a reusable Pyodide 0.29.0 worker. It runs the existing Python
sample, loop and MIDI algorithms; `pipeline.js` overlaps setup with separation.
It removes each job's temporary files and releases the worker on cancellation,
failure, disposal or five minutes of inactivity. Only one job runs at a time.

Stage the runtime, naming reference and optional checked-hash bytecode cache
outside the source tree:

```sh
node tools/browser/prepare_bytecode.cjs /path/to/pyodide-runtime /path/to/bytecode
python tools/browser/prepare_postprocess.py --runtime /path/to/pyodide-runtime --reference /path/to/soundid_ref.npz --bytecode /path/to/bytecode --output /path/to/post-assets
```

The combined prototype requires `DrumBrowserPipeline.create({allowKitOnly: true})`.
Without that explicit opt-in, the adapter requires the full stem set. It does not
silently substitute partial output for the website's existing features.

## Existing preview recovery

A live test returned `File is gone` for a listed hi-hat preview while the mixer
remained on Loading. The backend fix publishes files before announcing completion,
prevents caching preview errors, and lets the existing mixer play available stems
and retry missing previews. Tests cover publication ordering and failure recovery;
the production cause was not confirmed from deployment logs.

## Performance and verification

See [pipeline measurements](browser-pipeline-reuse.md) for the combined speed
comparison and reproduction commands, and [regression checks](browser-regression-review.md)
for tested behavior. Both retain the limits of the evidence. Original per-run
reports remain available in the linked commit history; they are not required
runtime assets.
