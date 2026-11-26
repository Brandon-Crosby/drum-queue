# Smaller browser models and artifact parity

This is an experiment, not a live-site change or a completed feature-parity
release. The website's interface and cloud processing path are unchanged.
No cloud GPU jobs were submitted for these checks.

## Model download and audio

The candidate rounds Helyx's float32 weights to 12 mantissa bits while keeping
tensor types, activations, graph shapes and computation float32. It is lossy;
it is not the earlier rejected float16-activation export. Demucs remains the
original model because rounding it offers negligible download savings.

With ordinary gzip, Helyx decreases from 405,740,336 to 292,020,230 bytes.
Together with unchanged Demucs, the model download decreases from 505,600,227
to 391,880,121 bytes: **22.49% less than the gzip baseline**. Decoded models
still occupy the same space, so this does not reduce model memory or guarantee
faster inference. Model verification uses the candidate's own SHA-256 hash.

On one authorized 9-second music excerpt, the real Chrome/WebGPU output differed
from the original browser output as follows:

| Stem | Signal-to-error ratio | Largest absolute sample difference |
| --- | ---: | ---: |
| Kick | 58.77 dB | 0.00000942 |
| Snare | 78.60 dB | 0.00003278 |
| Percussion | 83.48 dB | 0.00002253 |
| Hi-hat | 63.69 dB | 0.00001252 |

These are numerical comparisons, not listening-test results or proof across
genres. Quieter stems can have lower SNR despite a small absolute difference.
The experiment is not enabled in a published manifest.

A second generated 9-second fixture also produced finite output in every stem;
its per-stem signal-to-error ratios ranged from 59.50 to 91.92 dB.

| Same generated 9 s clip, simulated shared 100 Mbps | Original gzip weights | Smaller Helyx candidate |
| --- | ---: | ---: |
| Cold model cache | 88.70 s | 99.63 s |
| Verified cached models | 49.16 s | 71.13 s |

**The candidate did not win this speed comparison.** Smaller transfer size did
not offset variable session preparation and processing time. These sequential
pairs used the same current worker; GPU driver caches were not reset. They
include model transfer, verification, session setup and four-stem processing,
but not the remaining full-site features or postprocessing. Runtime assets were
fetched again by the diagnostic's no-store server. A smaller download alone is
insufficient evidence to promote this candidate.

Prepare a candidate separately, then stage it with `prepare_assets.py` and
create verified HTTP sidecars with `compress_models.py`:

```sh
python tools/browser/compact_model.py --source /path/to/helyx.onnx --output /path/to/experiments/helyx-m12.onnx --mantissa-bits 12
```

## GPU lifecycle

The worker now releases its session and destroys its owned GPU device before
reporting completion. Queue draining is bounded to avoid hanging on a lost
device. Tests cover successful output, inference failure, release failure,
rejected queue completion and a stalled queue.

Three original-weight 9-second runs before this change took 33.70, 49.16 and
52.78 seconds. Three after it took 29.60, 48.99 and 46.45 seconds. All 24 WAVs
matched the original hashes. These runs do not isolate the cause of the
previously recorded 320-second outlier or prove that it cannot recur.

## Existing sample-extraction code in WASM

The proof loads the repository's Python sample-extraction code and naming
reference into pinned Pyodide 0.29.0. On the same four existing stems, native
Python and Node-hosted WASM returned identical metadata, three one-shots, two
loops, MIDI files and ZIP member contents. ZIP timestamps are excluded from
the byte comparison. The staged runtime and dependencies total 31,539,400
bytes, additional to the model download.

The same extraction then ran in a real Chrome dedicated worker. It matched
native metadata and all 14 artifact checks, including the contents of each
ZIP. Runtime initialization took 13.84 seconds and extraction/packaging took
41.60 seconds, for **55.58 seconds total** on the local no-store server.
This is substantially slower than the Node-hosted proof. These are measured
wall times on a shared machine, not isolated CPU benchmarks. This route
demonstrates compatibility but is not a speed-qualified replacement.
Subsequent startup work is measured separately in
[the bytecode cache report](browser-runtime-startup.md).

This exposed a 32-bit array-index limit in hit detection: creating all sliding
windows before selecting hop-spaced frames produces an oversized intermediate
view. The shared code now constructs only the required frames. Tests verify
that FFT inputs remain exactly the same and that a 9-second clip works under
the simulated 32-bit limit.

`postprocess_job.py` adapts those existing algorithms into the artifact contract.
It requires the naming reference and full mix stem set by default. The current
proof explicitly opts into a four-stem test; it is not a replacement for the
website's full-mix processing. Its BPM metadata comes from the filename or
generated MIDI tempo, rather than the server's separate librosa analysis.

## Remaining release checks

- Full-mix model selection and quality comparison against the website.
- Optimize browser-hosted extraction, then end-to-end testing in the original UI.
- Longer clips, cancellation during each stage, and memory pressure.
- Listening checks on representative music and testing on other hardware.

The current 30-second limit remains. One M1 Pro/16 GB/Chrome 155 machine cannot
establish performance or compatibility for all users. The historical 44.57-second
cloud result is a reference from one job, not a universal deadline or a new paid
control run. Evidence is in [the measurement JSON](benchmarks/browser-compact-quality.json).

Validation passed: 53 JavaScript tests, 32 Python tests, four source-package
tests and the deployment-watch test. Neither a passing suite nor two short
audio fixtures establishes zero regressions across arbitrary music and devices.
