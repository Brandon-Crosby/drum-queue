# Drum Queue

Drop in songs (or whole album folders) and get the drum kit back as separate WAV stems:
**kick, snare, toms, hi-hat, ride, crash**. Runs locally on your Mac; nothing is uploaded.

Two stages, the same chain Xport Studio's drum-kit add-on uses:

1. **Demucs** pulls the drums out of the full mix (and also gives bass / vocals / other).
2. **MDX23C DrumSep** (aufr33 & jarredou) splits the drum stem into kit pieces.

Both run through [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator),
on the Apple Silicon GPU (MPS) when available.

## Set up (once)

    git clone <this repo> drum-queue
    cd drum-queue
    ./setup.sh

Uses [uv](https://github.com/astral-sh/uv) to make a Python 3.11 environment in `.venv`
(no Homebrew needed). The first song downloads Demucs (~300 MB) and DrumSep (~440 MB)
into `~/.drumsplit/models`. If Xport Studio already downloaded DrumSep, that copy is reused.

## Run

    ./run.sh

Opens http://127.0.0.1:8766. Drag songs or folders onto the page, or drop audio into
`inbox/`. Stems land in `~/DrumSplits/<song>/`. Every finished song has a mixer: mute/solo
each kit piece, click a waveform to jump, Space to play/pause.

### Options on the page
- **Drums from mix**: Best (`htdemucs_ft`), Faster (`htdemucs`, about 4x quicker), or
  6-stem (`htdemucs_6s`, also writes guitar and piano).
- **Kit split**: Normal (overlap 4, like Xport) or Extra clean (overlap 8, 2x slower).
- **Also keep bass / vocals / other**: uncheck to keep only drums + kit pieces.

### Flags
`--out ~/DrumSplits` `--inbox ~/Drum_Inbox` `--port 9000` `--keep-uploads`
`--model htdemucs` `--overlap 8` `--drums-only` (these three set the defaults for inbox files)

## Command-line batch version

`drumsplit.py` runs the same pipeline without the web app:

    .venv/bin/python drumsplit.py "/path/to/Album Folder" -o ~/DrumSplits
    .venv/bin/python drumsplit.py song.mp3 --drums-only --overlap 8

## Notes
- The queue is saved to disk; quit any time and unfinished songs resume next launch.
- Kit pieces that come out silent (for example, no ride in the song) are marked "silent".
- ffmpeg: uses one on your PATH, else Xport Studio's bundled copy, else the one from the
  `imageio-ffmpeg` package that `setup.sh` installs.
- If a model download fails with an SSL certificate error, turn off any VPN or web filter
  and hit Retry.

## Credits
- [Demucs](https://github.com/facebookresearch/demucs) (Meta AI)
- MDX23C DrumSep model by aufr33 and jarredou
- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)

Model weights are downloaded at runtime and are not part of this repo; they carry their
own licenses.
