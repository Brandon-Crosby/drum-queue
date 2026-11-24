# Audio export regression checks

Run `python -m unittest discover -s tests -v`. The audio-regressions workflow runs these checks on pull requests with NumPy, SciPy, SoundFile, FastAPI, python-multipart and mutagen. No model downloads, GPU, private audio or cloud calls are needed.

The fixtures cover:

- A representative one-shot with another onset inside its requested tail. Only candidates that fit before the supplied next onset may replace the averaged sample; fewer than three isolated candidates retain the existing averaged fallback. Both mono and stereo attacks are checked, including unchanged source audio and the existing tail fade.
- Full-song MIDI keeping the detected inter-hit spacing, including off-grid notes. `Groove.mid` remains the quantized compact loop; `Groove (full song).mid`, included in the one-shot ZIP, preserves detected timing. These are alternative patterns, not audio recordings or guarantees of correct onset detection.
- A pickup just before the estimated grid phase. Alignment moves the start back a bar instead of leaving negative times that the MIDI writer would clamp to zero.
- A filename tempo of 127.5 BPM retaining its fractional part, with the displayed result tempo using the same filename value as the MIDI export.
- Real multipart uploads of synthetic WAV bytes with long and mixed-case filenames. Filename shortening preserves the validated extension, and unsupported types still return 400 without creating a job.

These checks prove specific code behavior. They do not measure Helyx separation quality, correct instrument labels, kick fidelity, vocal/808 bleed, or reconstruction accuracy on Afrobeat/Afro house. Those require licensed reference audio, truth stems where available, and listening/A-B evaluation (see `eval/kit_score.py`). No model weights, dependencies used by inference, hosted deployment or desktop runtime are changed by these fixes.

## UI checks on this Mac

A 159-character WAV filename was uploaded through the file picker and processed by the real Demucs/Helyx pipeline on MPS. The user confirmed audible preview playback and an advancing timer. The one-shot ZIP passed its integrity check and contained two WAVs and both MIDI variants (58 notes). A second split showed 127.5 BPM after correcting the displayed tempo.

Reopening a result exposed an obsolete footer selector in the desktop patch, which stopped page initialization. Local copy now lives in the main script, tolerates missing decorative elements and preserves the footer links. The theme handler also stays inside the mixer closure. `tests/ui-startup.cjs` checks the built page with local/hosted settings, result reload, a missing footer, theme changes and the MIDI download attribute. HTTP responses and audio hardware are substituted in this test; it is not a listening test.

FL Studio 24.2.2 imported the full-song MIDI into separate Kick and Hat channels with notes visible in the piano roll. Its import set the project to **128 BPM**, although the MIDI tempo is **127.5 BPM**. Set FL's tempo to the exported tempo when using fractional BPM. Import used FL's default FLEX channels; loading the matching WAVs into samplers and comparing the reconstructed beat remains a separate check.
