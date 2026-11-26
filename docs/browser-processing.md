# Experimental browser processing

Status: local candidate. **PR on hold:** the requested first-use speed target
(including model download and setup) has not been met. The live site is not
updated by this work.

The standalone experimental page, renderer and opt-in link were removed at the
user's request. `web/site/index.html` and `styles.css` match the base revision
`c9d765e`. Existing picker, mixer, EQ, one-shot, loop and download controls are the
UI contract. The background processor never renders controls or changes page DOM.

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
returned: kick, snare, percussion and hi-hat. Full mix stems, one-shots, loops,
MIDI and the full existing result contract are not implemented in this processor.
The existing website retains those features.

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

## Measured on the development Mac

Apple M1 Pro / 16 GB / Chrome WebGPU. These measurements are observations, not
speed or hardware-support guarantees. Other development work ran on this Mac;
these are not controlled live-service benchmarks.

A 9-second stereo clip completed in 126.85 seconds with cached weights, including
fresh sessions and DSP. The same zero-shift/overlap Python CPU reference took
54.22 seconds. A later live-site check is described below. First use additionally
requires the ~612 MB model transfer. This browser path is currently a slower,
experimental alternative, not a replacement for the live service.

| Output | Frames per channel | SNR vs equivalent Python | Maximum sample error |
| --- | ---: | ---: | ---: |
| Kick | 396900 | 79.76 dB | 1.48e-7 |
| Snare | 396900 | 102.66 dB | 1.15e-6 |
| Percussion | 396900 | 103.89 dB | 1.09e-6 |
| Hi-hat | 396900 | 82.26 dB | 1.35e-6 |

All samples were finite. The comparison includes PCM24 encoding error and checks
full overlap-add output, not only one neural block. It establishes numerical
agreement with equivalent settings; it is not a listening-quality comparison
with the deployed cloud pipeline. Exporting with the supplied tool reproduced
the tested ONNX files byte-for-byte; both PyTorch export adapters had zero maximum
round-trip error before ONNX export.

The page's 100 ms heartbeat continued during the worker run (1185 ticks; largest
observed gap 1.19 seconds, including decode/setup/output work). This demonstrates
continued page scheduling, not a guarantee of zero UI latency or a peak GPU-memory
measurement. Cache reads were observed for both models on the second run.

A 30-second clip also completed with cached weights in 243.82 seconds, producing
four finite stereo WAVs of 1,323,000 frames per channel (7,938,044 bytes each).
The 100 ms heartbeat ticked 2367 times, with an observed maximum gap of 826 ms.
A 31-second clip was rejected through the actual file picker before model work.
This checks the advertised cap on this machine, not a general memory ceiling.

In the candidate UI, playback time advanced, starting another stem paused the
previous player, and Stop paused/reset all players. A WAV saved through Chrome's
native Save dialog was byte-identical to the captured worker output. Cancel
during model preparation restored controls; retry started a new worker.

## Live-site baseline and preview recovery

An authenticated test on drumsplitter.com uploaded a generated 9-second drum
fixture. The live job reported 11.28 seconds queued and 33.30 seconds running
(44.57 seconds from server creation to completion, excluding upload). Its result
listed a hi-hat preview that returned `File is gone`; the live mixer remained on
Loading even after other previews loaded. This was observed before these changes
were deployed.

The cloud worker currently writes the done status before its final volume commit.
The proposed fix publishes output files before setting done, prevents caching
error responses, and lets the mixer play loaded stems and retry missing previews.
The storage race is consistent with the live response; the production cause
cannot be confirmed without deployment logs. Tests verify publication ordering
and failure handling. The candidate UI was tested with a real MP3 response and an
injected initial 404: mute/unmute loaded the missing stem, playback advanced to
0:08, and Stop returned it to 0:00. This was a local served candidate, not a
claim that the production backend has already been repaired.

## WebAssembly optimization and remaining speed gate

Profiling a later 9-second run measured 139.11 seconds total: about 86.22 seconds
of JavaScript transforms, 20.72 seconds model creation and 23.52 seconds inference.
The compiled transform replacement reduced a complete run to 59.34 seconds.
Its four PCM24 outputs were byte-identical to the earlier worker outputs and
retained the same Python-reference agreement above. The transform-only Node
tests also passed within a maximum 1e-7 sample difference.

This still does **not** establish first-use parity with the live site. The
59.34-second run used cached model files, and the live cloud job's server interval
was 44.57 seconds. Initial model transfer is additional. At 100 Mbit/s, transmitting
611,957,925 bytes alone has a theoretical lower bound of about 49 seconds before
protocol overhead, setup or inference. A localhost transfer is not a meaningful
internet download benchmark. No public weight hosting has been configured.

Only one generated 9-second clip was submitted to the live cloud service. All
other model tests ran on the development Mac. The PR remains unsubmitted pending
the first-use performance requirement; passing regression tests does not waive
that requirement or establish support for every browser and GPU.

Final end-to-end candidate UI check: the same generated 9-second clip used for the
live cloud test completed in **154 seconds**, as reported by the page's own timer
from Split through returned WAVs, including setup with cached models. Its snare
preview advanced to the end and Stop reset all four players. This slower UI
observation supersedes any suggestion that the 59.34-second harness result proves
normal-use speed. Timing varies substantially in this environment. First-download
parity remains unproven and the observed cached run already misses the target.

Final local checks: 21 Python tests, 30 browser/UI tests, 4 source-build tests and
1 deployment-watch test pass. The compiled DSP binary was rebuilt with pinned
Zig and its SHA-256 remained unchanged. No regression was found in these tested
cases; this is not a claim about all hardware or all production conditions.

## Existing UI preserved

The earlier standalone page and screenshots above are historical prototype QA;
that page and its renderer are removed from the proposed source changes. The
processor is tested separately for file limits, cancellation during decode and
inference, retries, concurrency and error cleanup. A DOM assertion confirms that
processing leaves the surrounding application's body unchanged. No cloud GPU
jobs were started for this refactor.

After removing the alternate UI, all 33 browser/UI checks pass, including the
existing mixer flows and the UI-independent processor tests. The desktop source
package builds successfully. A direct diff against c9d765e confirms no changes to
the existing index.html or styles.css. The visual check used a local result fixture
and did not run any cloud inference.

The optimized first-use check is recorded in `browser-speed-check.md` and
`benchmarks/browser-model-loading.json`. Removing full-model buffer copies and
moving cache writes off the inference critical path reduced uncapped cold-asset
runs to 29.07–35.94 seconds, with byte-identical WAVs. At simulated 100 Mbit/s,
first-use still takes 109.26 seconds and misses the target. The current loader
has six additional failure-mode tests; all 39 browser/UI checks pass. The PR
remains on hold for first-use speed and the incomplete full-site feature contract.

The follow-up audit, including the retained long-run outlier, WAV preflight fix
and lossless download tests, is in [browser-performance-audit.md](browser-performance-audit.md).
