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

### Re-split and version history (team mode first) - owner: Dalyn
Why: after every model or kit-maker deploy we re-upload the same songs to compare kits (KUIKI, Nov 2026).
- Version stamp on every split: model (e.g. Helyx 4) + kit-maker commit + date, stored on the job, shown
  as a small line on the results page and in the zip name. Public splits get this too, so a bad-kit report
  tells us exactly what made it.
- Team mode only: a "Re-split with latest" button. Team uploads are already kept (`keep_team`), so it
  re-runs the saved upload on the current model + kit maker without re-uploading.
- Version history on the team results page: earlier versions of the same song stay listed, switch between
  them to A/B the pads, loops and MIDI.
- Not for public splits: their uploads are deleted after the split (privacy rule), and the same model gives
  the same result, so a public user just re-uploads after an update.

### Kit maker follow-ups (Brando + Claude)
- Snare-Perc one-shots in busy perc patterns come out short (KUIKI Snare-Perc 2, ~0.1 s): the cut stops at
  the next hit. Let the tail ring past it (from a hit with space after it, or the averaged tail).
- Hats loop can carry noise from the hat stem (KUIKI). Options to A/B: a gentle gate/expander between hits,
  a light denoise, or a high-pass/EQ on the loop only. One-shots and stems stay untouched.

## Toms and cymbals stems (owner: Dalyn, after Satyr eval)
If Satyr (6 stems, see train/NEXT_RUN.md) ships: the site and kit maker handle two more stems.
- mixer: toms and cymbals channels (solo/mute/EQ like the others), previews, downloads
- kit maker: a Tom slot (one-shots + a Toms loop), cymbals one-shots; Groove MIDI gets tom notes
- engine: stem list from the model, not hard-coded to kick/snare/perc/hh

## Inference orchestrator (later: local app, or when the site runs more than one model)
A small local inference server (helper process on a local socket) the app/plugins talk to: hello/submit/
progress/result/cancel; pipelines as recipes (`drums6: [stage1, satyr]`, best model per stem as config);
a model catalog (arch, file, checksum, stems, chunk/hop, memory); a memory budget with idle-model eviction and
a clear refusal instead of a crash; cancellation between chunks. PyTorch first, Core ML / ONNX later.
Pinned Nov 28, 2026: architecture, not quality; stay on Satyr / stage 1 / kit maker first.
