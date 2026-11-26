# Overlapping setup and reusing the CPU worker

The browser prototype can now prepare Python/SciPy while the audio models load
and run, and retain that initialized CPU worker for the next job. Each job's
temporary audio files are removed before completion. Cancellation, worker
failure, explicit disposal and five minutes of inactivity release the worker.
GPU sessions still run sequentially and are released as before.

The existing website does not select this backend yet. Its UI and cloud path
are unchanged. No cloud GPU jobs were submitted.

## Combined measurements

M1 Pro, 16 GB RAM, Chrome 155/WebGPU; same generated 9-second clip and original
float32 model weights. Both configurations used the prepared bytecode cache.
All HTTP response bodies shared a simulated 100 Mbps connection cap.

| Decode through WAVs, samples, loops, MIDI and ZIPs | Serial setup; fresh CPU worker per job | Overlap setup; reuse CPU worker |
| --- | ---: | ---: |
| First use, cold model cache | 139.32 s | 108.49 s |
| Next job, verified cached models | 108.93 s | 63.21 s |

The measured differences are **22% on first use** and **42% on the repeat**.
These are sequential pairs on one shared machine, not a statistically established
speed guarantee. Driver caches were not reset. First use includes model and
sample-runtime download, verification, initialization and processing. The input
fixture was already selected before timing; file reading and decoding are timed.
Diagnostic output hashing and saving copies to disk occur after the timer.

In the first overlap run, sample tools were ready at 35.39 seconds, before model
processing ended at 89.57 seconds. The first job transferred 55,516,830 bytes of
sample assets; the repeat transferred **zero**, confirming runtime reuse. Model
bodies were downloaded only on the first job in each pair. The diagnostic's
no-store server still fetched ORT assets on each job.

All four jobs returned the original four WAV hashes. Metadata and all 13
artifact checks per job matched native Python: stem WAVs, one-shots, loops,
MIDI and ZIP member contents. ZIP timestamps are excluded from comparisons.

## Implementation and reproduction

- `postprocess.js` owns the reusable CPU worker and cancellation lifecycle.
- `postprocess-worker.js` verifies its assets, prepares imports, processes one
  job at a time and clears the job filesystem before returning artifacts.
- `pipeline.js` coordinates GPU separation and CPU sample preparation. Full-mix
  processing is not implemented here, so creating this pipeline requires an
  explicit four-stem test opt-in. The sample worker separately requires the
  complete stem set by default.

Prepare assets outside the site source tree:

```sh
python tools/browser/prepare_postprocess.py --runtime /path/to/pyodide-runtime --reference /path/to/soundid_ref.npz --bytecode /path/to/bytecode --output /path/to/post-assets
```

Run `tools/browser/benchmark.py` with the existing model and fixture arguments,
adding `--post-assets /path/to/post-assets --post-mode serial` for the control,
or `--post-mode overlap --post-reuse` for the candidate. Both pairs here used
`--runs 2 --reuse-cache --mbps 100 --profile --save-wavs`.

Current automated and UI results are in [the regression review](browser-regression-review.md).

## Scope

This measures a combined **four-stem prototype**, not the full website workflow.
Full mix separation, original-UI integration, longer clips and other devices
remain release checks. These numbers do not establish parity with the historical
44.57-second cloud job, which returned more features. No new cloud timing job was
run. The 30-second prototype limit remains.

The source hashes, manifests, timings and output hashes are recorded in
[the evidence JSON](benchmarks/browser-pipeline-reuse.json).

## Earlier screening and rejected experiment

Earlier separation-only tests included a 320.24-second cached 30-second run and
37.06-second heartbeat gap. Its cause was not fully isolated; later passing runs
do not prove it cannot recur. A pre-fix WAV metadata timeout is also retained in
the JSON summary. Those timings exclude sample preparation and cannot be compared
directly with the combined table above.

Reduced-precision Helyx weights saved 22.49% of the gzip model transfer but were
slower in the measured pair (99.63 s cold / 71.13 s cached versus 88.70 / 49.16 s).
The unused lossy export experiment was removed; this backend retains float32 models.

[Full historical reports](https://github.com/Brandon-Crosby/drum-queue/tree/48aa4ea101c3b1c76c3002e8eee7abb11c5970a1/docs/benchmarks)
remain accessible for provenance. The current JSON keeps source/model hashes,
all four combined timings, one copy of output references and earlier screening
results, without repeated per-stage traces and artifact hashes.
