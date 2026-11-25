# To-dos

Ideas and planned work for drumsplitter.com. Add new items at the bottom of a section.

## Planned

### Album kits (multi-song upload)
- Upload several songs at once (e.g. a whole album) and get one album-based drum kit.
- Organize the kit into folders by type: `Kick/`, `Snare/`, `Hats/`, `Percs/` (plus `Loops/` and MIDI as now).
- Remove duplicates across songs: producers often reuse the same sound across an album. Use the
  soundmatch fingerprints (same matching as `eval/ab_kits.py`) to keep the cleanest copy of each sound.
- GPU cost grows with the number of songs, so this may be a higher membership tier. Logistics TBD.
- Small first step: sort single-song kits into the same type folders in the zip (`web/backend/service.py`).
