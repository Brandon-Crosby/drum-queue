# Audio export regression checks

Run `python -m unittest discover -s tests -v`. The audio-regressions workflow runs these checks on pull requests with NumPy, SciPy, SoundFile, FastAPI, python-multipart and mutagen. No model downloads, GPU, private audio or cloud calls are needed.

The fixtures cover:

- A representative one-shot with another onset inside its requested tail. Only candidates that fit before the supplied next onset may replace the averaged sample; fewer than three isolated candidates retain the existing averaged fallback. Both mono and stereo attacks are checked, including unchanged source audio and the existing tail fade.
- Full-song MIDI keeping the detected inter-hit spacing, including off-grid notes. `Groove.mid` remains the quantized compact loop; `Groove (full song).mid`, included in the one-shot ZIP, preserves detected timing. These are alternative patterns, not audio recordings or guarantees of correct onset detection.
- A pickup just before the estimated grid phase. Alignment moves the start back a bar instead of leaving negative times that the MIDI writer would clamp to zero.
- A filename tempo of 127.5 BPM retaining its fractional part.
- Real multipart uploads of synthetic WAV bytes with long and mixed-case filenames. Filename shortening preserves the validated extension, and unsupported types still return 400 without creating a job.

These checks prove specific code behavior. They do not measure Helyx separation quality, correct instrument labels, kick fidelity, vocal/808 bleed, or reconstruction accuracy on Afrobeat/Afro house. Those require licensed reference audio, truth stems where available, and listening/A-B evaluation (see `eval/kit_score.py`). No model weights, dependencies used by inference, hosted deployment or desktop runtime are changed by these fixes.
