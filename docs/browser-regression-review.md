# Regression review

The checked paths pass after two fixes. This does not establish full website
feature parity or zero regressions across every device and input.

## Found and fixed

- A job joining an already-starting CPU runtime did not observe its own abort
  signal until initialization finished. It now cancels immediately and releases
  the shared worker. The new test failed before the fix and passes afterward.
- Preview recovery could show “unavailable” while the initial requests were still
  pending. Loading now redraws when requests start. Both the partial-failure and
  all-failure tests reproduced the incorrect label before the fix.
- CI omitted the newest CPU-worker and pipeline tests. It now discovers every
  `tests/browser-*.cjs` and `tests/ui-*.cjs` file.

The fixes are in `373bb7b`. Upstream main `6d4b4e3` was merged locally, then the
checks were rerun on `7e64b06`. Its additional changes affect training files.

## Verification

| Check | Result |
| --- | --- |
| JavaScript UI, worker, model-loading and lifecycle tests | 66 passed |
| Python audio, export, publication and adapter tests | 35 passed |
| Source package tests | 4 passed |
| Deployment-watch test | Passed |
| Actual Chrome OfflineAudioContext checks | 20 passed |
| Native onset detection against the original code | 14 exact matches |
| Compiled DSP rebuilt with pinned Zig | Identical |
| Packaged browser assets versus current source | All 10 match |

The local Python dependency versions match the CI workflow. GitHub-hosted CI
has not been run for these new commits yet.

The real Chrome UI test used the existing public website build with previously
generated audio. Preview requests were deliberately held, failed, then allowed
to succeed. The interface showed Loading, then unavailable, and recovered using
its existing mute/unmute controls. Playback reached `0:01 / 0:09`; the EQ dial
changed and reset normally. Separate Web Audio renders verified gain, stereo
polarity, EQ bypass, independent stem settings and invalid-value handling.

The shared hit-detector change was compared against the old code using float32
and float64 input at seven lengths, from one sample to nine seconds. Every
detected onset matched exactly. The audio-computation modules also retain the
same hashes as the earlier combined-pipeline measurement; the new changes affect
cancellation and preview status rather than audio math.

## Remaining limits

No production cloud inference job was submitted. Publication ordering and error
handling were tested through local service fixtures. The new browser pipeline
still requires explicit four-stem prototype opt-in and remains limited to
30-second clips. Full mix processing, connecting it to the original website
flow, and broader device testing remain release gates. The experimental backend
has not been published as a replacement.

Machine-readable evidence is in
[the review JSON](benchmarks/browser-regression-review.json). The earlier timing
measurements remain historical measurements of their recorded source revision;
they were not relabeled as new speed tests.
