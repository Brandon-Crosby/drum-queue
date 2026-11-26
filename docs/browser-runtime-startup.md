# Preparing Python bytecode before browser startup

Profiling the browser sample-extraction stage found 19.40 seconds in Python
imports, including 10.72 seconds compiling source. An optional prepared cache
now compiles those sources with the same Pyodide 0.29.0 interpreter ahead of
time. It does not change NumPy/SciPy algorithms, model weights, or the website UI.

| Same 9-second stems, real Chrome worker | Without cache | Prepared bytecode |
| --- | ---: | ---: |
| Profiled pair, total runtime setup + extraction + packaging | 54.18 s | 37.45 s |
| Unprofiled pair, reverse test order | 40.83 s | 35.79 s |

The unprofiled pair improved by **5.04 seconds (12.3%)**. The larger difference
in the profiled pair should not be generalized: timings vary, the profiler adds
overhead, and these are sequential tests on one shared M1 Pro/16 GB Mac. This
is a sample-preparation improvement, not proof of a faster end-to-end split.
Model separation is outside these timings. All assets came from a local
no-store server; internet first-use performance has not been measured here.

The cache adds **6,063,786 bytes** to setup. It contains 729 checked-hash Python
cache files. The benchmark verifies the archive hash, runtime lock hash and
allowed cache paths before serving it; the browser checks the archive hash and
interpreter version before loading it. Python also checks each cached module
against its source. A changed runtime requires a newly prepared archive.

All four runs matched native Python metadata and all 14 artifact checks per
run: stems, sample WAVs, loop WAVs, MIDI and ZIP contents. ZIP timestamps are
excluded from comparisons. The full Python suite passed 35 tests and the
JavaScript suite passed 53 tests.

Generate and test the cache outside the site source tree:

```sh
node tools/browser/prepare_bytecode.cjs /path/to/pyodide-runtime /path/to/bytecode
python tools/browser/benchmark_postprocess.py --runtime /path/to/pyodide-runtime --bytecode /path/to/bytecode --stems /path/to/stems --reference /path/to/soundid_ref.npz --output /path/to/results
```

The benchmark requires the complete stem set by default. Its explicit
`--kit-only-prototype` option was used for these four-stem artifact tests; this
does not establish full-site feature parity. `--profile` adds Python hot-spot
measurements.

A separate SIMD DSP build preserved exact transform bytes but showed no useful
gain in alternating local tests (approximately 83 ms and 93 ms for the two
transform shapes). It remains a diagnostic build option and is not selected by
the product. No published model manifest, product WASM module, website control,
or cloud processing path changed.

Overlapping runtime preparation and retaining the CPU worker have since been
tested in [the combined pipeline report](browser-pipeline-reuse.md). Concurrent
CPU work can also delay inference, so that report measures the combined path.
Measurements and profiles are in [the evidence JSON](benchmarks/browser-runtime-startup.json).
