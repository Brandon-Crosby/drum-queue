# Browser first-use speed check

**Result: failed. No PR or deployment is ready under the first-use speed requirement.**
The existing product HTML/CSS is unchanged. This diagnostic tests the background
processing API; it does not add or replace the website interface.

The same generated 9-second stereo WAV was used for these local runs and the
previously recorded live job. No additional cloud GPU job was submitted.

| Scenario | Time through returned WAV buffers |
| --- | ---: |
| Recorded live cloud job, queue + processing | 44.57 s |
| Previous cold assets, uncapped localhost | 73.96 s / 69.41 s |
| Optimized cold assets, uncapped localhost | **29.07 s / 35.94 s / 37.91 s** |
| Subsequent cached-model run, same origin as 37.91 s run | 68.23 s |
| Previous cold assets, simulated shared 100 Mbit/s | 203.87 s |
| Optimized cold assets, simulated shared 100 Mbit/s | **109.26 s** |

Each browser run transferred 611,957,925 model bytes and approximately 637.6 MB
of timed response bodies including the runtime and worker assets. Cache Storage
was cleared before each cold run and HTTP responses used no-store. Timing starts with a
selected file already in memory and includes metadata/decode, all worker model
and runtime transfers, verification, session creation, inference,
audio transforms and PCM24 WAV return. It excludes subsequent diagnostic hashing.
Browser compiled-code and GPU driver caches were not reset; these are favorable
cold-asset checks, not clean-machine or actual internet/CDN measurements.

The optimized loader writes directly into one size-bounded buffer instead of
assembling a Blob and copying it again for Cache Storage. Verified cache writes
now overlap inference; completion waits at most 1.5 seconds for outstanding
writes. Cache access/write failures cannot prevent processing, and a stalled
cache stream is canceled before a verified network retry. SHA-256 validation is
still required before creating a GPU session. No model weights, precision,
overlap, transform or audio export settings changed.

The previous 100 Mbit/s run had an 86.65-second interval between the final sampled
Helyx download progress and model preparation. The optimized interval was 1.15
seconds. Its maximum diagnostic heartbeat gap fell from 10.90 to 0.42 seconds.
This is consistent with removing buffer/cache work from the critical path; the
benchmark does not isolate every cause of timing variation.

The first-use target remains unmet at 100 Mbit/s. The model payload alone has a
48.96-second ideal transfer floor at that bandwidth, excluding runtime bytes,
latency and processing. Uncapped localhost results cannot establish real internet
speed or full application parity. No new cloud GPU job was used for this check.

A separate cache-reuse sequence took 37.91 seconds cold and 68.23 seconds warm.
The warm run transferred zero model bytes, verified both cached hashes and
returned identical WAVs. Cache reading took about 0.11 seconds for Demucs and
0.30 seconds for Helyx before verification; the slower warm run spent longer in
session preparation and inference. Its cause is not isolated. Do not present
these measurements as guaranteed latency or claim cached use is always faster.

All five optimized 9-second runs returned four valid 44.1 kHz stereo PCM24 WAVs of 9 seconds each.
Their SHA-256 hashes were identical to all three baseline runs. This confirms stable
output for this fixture; it is not proof of quality or feature parity on all songs.
The cloud reference excludes upload and produced more outputs than the browser
prototype: full mix stems, one-shots, loops and MIDI. No full-application speed or
feature-parity claim is being made.

Hardware: Apple M1 Pro, 16 GB RAM, Chrome 155 with WebGPU. Other applications were
running. The observed times are not a guarantee for another Mac or browser.

## Reproduce

Stage the authorized models/runtime as described in `browser-processing.md`, then:

```sh
python tools/browser/benchmark.py --assets /path/to/browser-assets --input /path/to/authorized-9-second-fixture.wav --output /tmp/browser-speed --runs 2
```

Open the printed localhost diagnostic URL. It only serves local assets and writes
local timing reports. For the shared bandwidth simulation, use a fresh invocation
with `--runs 1 --mbps 100` and a separate output directory. Do not run GPU cases
concurrently. The diagnostic page lives in this developer tool, not web/site.

Compact measurements and output hashes are in
[`benchmarks/browser-first-use.json`](benchmarks/browser-first-use.json).
Optimized measurements are in
[`benchmarks/browser-model-loading.json`](benchmarks/browser-model-loading.json).
The local raw reports are retained under build/speed-check and
build/speed-optimized and excluded from git.

## Regression checks for the loading optimization

- 39 Node tests passed against the built source package: existing UI startup,
  loop display, preview retry, processor cancellation/limits, engine behavior,
  worker cleanup, native DSP equivalence and six cache/loading failure cases.
- A real 30-second input completed in 143.95 seconds with four valid stereo
  44.1 kHz PCM24 WAVs. This checks the duration limit; it has no identical
  30-second pre-optimization output baseline.
- 21 Python audio/API/publication tests passed.
- Four source-package build tests and the deployment-watch test passed.
- Rebuilt DSP WebAssembly from pinned Zig 0.14.1; no binary difference.
- Existing product index.html and styles.css match base c9d765e.
- Existing mixer UI smoke check on the local fixture: Play advanced to 0:04,
  Stop reset to 0:00, and kick EQ opened, changed Low to +2 dB and reset.
  This verifies the existing UI separately; the new processor API is not wired
  into the live site.

These checks cover the prototype and existing tested behavior. They do not prove
all songs, browsers or full-site browser processing: bass/vocals/other mix stems,
one-shots, loops, MIDI and inputs longer than 30 seconds remain outside the
browser processor's implemented contract. The PR remains on hold.

The follow-up audit, including the retained long-run outlier, WAV preflight fix
and lossless download tests, is in [browser-performance-audit.md](browser-performance-audit.md).
