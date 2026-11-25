# To-dos

Ideas and planned work for drumsplitter.com. Add new items at the bottom of a section.

## Planned

### Album kits (multi-song upload) - owner: Dalyn
- Upload several songs at once (e.g. a whole album), queue them, and get one album-based drum kit.
- Organize the kit into folders by type: `Kick/`, `Snare/`, `Hats/`, `Percs/` (plus `Loops/` and MIDI as now).
- Remove duplicates across songs: producers often reuse the same sound across an album. Use the
  soundmatch fingerprints (same matching as `eval/ab_kits.py`) to keep the cleanest copy of each sound.
- GPU cost grows with the number of songs, so this may be a higher membership tier. Logistics TBD.

Suggested order:
1. Sort single-song kits into the type folders in the zip (`get_zip` in `web/backend/service.py`).
   Small, ships on its own, and the album kit reuses it.
2. Multi-file upload + queue in the UI and `create_job`: one job per song, grouped under an album id.
   Each song runs through the existing per-song pipeline unchanged.
3. Album step after every song finishes: a new `kitmatch/album.py` that takes the per-song one-shots,
   groups same sounds with soundmatch, keeps the cleanest, and writes the album zip.
4. Tier/limits for album uploads (later).

Working in parallel with the model and kit-maker tuning:
- Brando and Claude are changing the model (`train/`, `eval/`) and how one-shots are cut
  (`kitmatch/clean.py`, `hits.py`, `groove.py`). Treat the per-song kit as a black box: call it, don't edit it.
- The contract the album step can rely on: per-song shots named like `Kick 1.wav`, `Snare-Perc 2.wav`,
  `Hat 1.wav`, loops as `* loop (* bars).wav`, plus the groove MIDI. If that naming needs to change, ask first.
- Use eval beats or your own test songs only; public uploads are never kept or trained on.
