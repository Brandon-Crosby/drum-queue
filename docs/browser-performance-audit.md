# Local browser performance audit

**The performance gate remains open. No PR or deployment is ready under the
first-use speed and full-feature requirements.**

Testing found a WAV metadata timeout and an intermittent long-run slowdown.
The WAV preflight now reads a bounded header for supported mono/stereo PCM and
float WAVs, preserving duration limits and cancellation. Unusual WAV layouts and
other containers retain browser metadata loading. The audio decoder, models,
overlap, transforms and output encoding are unchanged.

## Measurements

Apple M1 Pro, 16 GB RAM, Chrome 155, WebGPU, AC power. All jobs ran locally and
sequentially; no additional cloud GPU job was submitted. Inputs were the same
generated 9-second fixture and a 30-second repetition of it.

| Case | Previous | Candidate |
| --- | ---: | ---: |
| 9 s, cold models, two sequences | 127.78 / 52.90 s | 34.22 / 32.83 s |
| 9 s, cached models, same sequences | 54.78 / 51.69 s | 49.98 / 54.61 s |
| 30 s, first comparison, cold | 109.39 s | 97.09 s |
| 30 s, first comparison, cached | 156.69 s | **320.24 s** |
| 30 s, final browser-path comparison, cold | 153.88 s | 81.68 s |
| 30 s, final browser-path comparison, cached | 107.72 s | 94.46 s |

The previous revision is e79ff02; the loader candidate is 9470aac. The final
30-second candidate included the WAV header change. Its final conservative
fallback for unusual trailing chunks is committed in 3a1bb74 and covered by unit
tests and the subsequent real 9-second transport runs. Full per-case provenance,
phase timings and output hashes are in
[the measurement JSON](benchmarks/browser-performance-audit.json).

The **320.24-second run is retained**, not discarded. It recorded a 37.06-second
main-thread heartbeat gap, with slower transforms and inference as well as
session preparation. It did not recur in the later pair, but its cause has not
been fully isolated. Background application load was substantial during parts of
the session. Neither this sample size nor the passing repeats proves zero speed
regressions. Profile timings are wall time, not isolated GPU execution time;
model-load timings include session creation and must not be added to it again.

A separate pre-fix test became hidden and its second WAV failed the media-element
metadata preflight after 15 seconds. The corrected header path completed both
30-second runs. Native window automation also returned stale page state and did
not execute a reload; that attempted reload produced no measurement and is not
counted. Subsequent tests used the live browser connection, with focus and
visibility recorded. The earlier timing comparisons did not all record focus.

## Lossless download compression

| Same 9 s input, simulated shared 100 Mbit/s | Model response-body bytes | Elapsed |
| --- | ---: | ---: |
| Identity encoding | 611,957,925 | 122.77 s |
| Gzip encoding | 505,600,227 | 95.74 s |
| Subsequent gzip-origin cached run | 0 | 52.85 s |

The payload is **17.38% smaller**. Decompression restores the original model
SHA-256 hashes, and all returned WAVs match the identity-encoding baseline.
The timing difference also includes variable GPU setup/inference; it cannot all
be credited to compression. The cached run verifies storage reuse, without a
separate identity-encoding cached control at that bandwidth.

This still misses the recorded cloud interval of 44.57 seconds for the 9-second
fixture. That cloud job produced more features than this prototype. HTTP body
transfers here include runtime assets, and first-use timing includes model
transfer, verification, session creation and processing. File selection is
outside timing, as before. GPU-driver/compiled-code caches were not reset. The
100 Mbit/s cap is a localhost simulation, not an internet/CDN measurement.

Compression is an optional deployment artifact; **it is not enabled on the live
site**. Generate verified sidecars outside web/:

```sh
python tools/browser/compress_models.py --assets /path/to/browser-assets/v1 --output /path/to/compressed-models
```

A host must serve the sidecar at the canonical .onnx URL with
`Content-Encoding: gzip` and `Content-Type: application/octet-stream`, negotiating
encoding appropriately. Cached decoded bytes retain the original model hash.
The local benchmark implements this mapping with `--gzip-models`.

## Validation and scope

- **20 completed splits / 80 WAVs** matched the appropriate baseline byte for
  byte; one pre-fix metadata failure is recorded separately.
- **48 Node tests**, **23 Python tests**, four source-package tests and the
  deployment-watch test passed locally: **76 automated checks**.
- The built source package's seven browser assets match the tested source bytes.
- DSP WebAssembly rebuilt identically with pinned Zig 0.14.1.
- Product index.html and styles.css still match base c9d765e. No UI controls or
  layout were added, and the browser processor is not wired into the live site.

Only one physical Mac was tested. Intel Macs, M4 Macs, other macOS versions and
other browsers remain unverified; emulating a viewport would not test their GPU
performance. Full mix stems, one-shots, loops, MIDI and clips longer than 30
seconds remain outside this browser prototype's implemented contract.

Reproduce a local comparison using `tools/browser/benchmark.py`: use `--scripts`
with a locally extracted browser/ directory from the previous revision,
`--profile` for phase timings, `--reuse-cache` for cold/cached sequences, and a
fresh invocation with `--mbps 100` for the shared cap. A default invocation tests
the current code. Diagnostic pages and profiling hooks live under tools/, not
in the product UI. Models, generated audio and raw reports remain under ignored
build/ directories and are not shipped in the source package.

Temporary model caches were cleared for 16 benchmark origins after verification.
Future benchmark runs clear their model cache after the complete series; cache
reuse within a series is still measured before cleanup.
