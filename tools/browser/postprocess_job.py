"""Browser/WASM postprocessing adapter using the repository's existing algorithms.

Inputs are 44.1 kHz PCM WAV stems in a virtual filesystem. This module does not
load models, contact an API, or replace the website interface.
"""
import json
import re
import zipfile
from pathlib import Path


def finish(name, split_dir, output_dir, reference, *, require_full_mix=True, log=lambda message: None):
    import soundfile as sf
    import engine
    from kitmatch import clean, groove

    split_dir, output_dir, reference = map(Path, (split_dir, output_dir, reference))
    kit = ['kick', 'snare', 'perc', 'hh']
    mix = ['drums', 'bass', 'vocals', 'other']
    required = kit + (mix if require_full_mix else [])
    if not reference.is_file():
        raise ValueError('The sound-naming reference is required')
    if any(not (split_dir / (stem + '.wav')).is_file() for stem in required):
        raise ValueError('The required stem set is incomplete')
    info = sf.info(split_dir / 'kick.wav')
    if info.samplerate != 44100 or info.channels != 2 or not 0 < info.frames <= 30 * 44100:
        raise ValueError('Unsupported browser stem format or duration')
    for stem in kit + mix:
        path = split_dir / (stem + '.wav')
        if path.exists():
            other = sf.info(path)
            if (other.samplerate, other.channels, other.frames) != (44100, 2, info.frames):
                raise ValueError('Stem audio dimensions differ')
    output_dir.mkdir(parents=True, exist_ok=True)
    shots_dir = output_dir / 'one-shots'
    song = engine.safe_name(Path(name).stem)[:60]
    log('Making one-shots')
    note = ''
    try:
        made = clean.write_kit(split_dir, shots_dir, prefix=song + ' - ', ref=reference, log=log)
    except Exception as error:
        # Match the server: optional sample extraction must not discard stems.
        made = []
        note = 'One-shots could not be created for this audio.'
        log('One-shot processing failed: ' + type(error).__name__)
    shots = [{key: item[key] for key in ('name', 'label', 'piece', 'seconds', 'count')} for item in made]
    loops = []
    for path in sorted(shots_dir.glob('* loop *bars).wav')):
        match = re.search(r'(Kick|Snare-Perc|Hats|[^-]+?) loop (?:([\d.]+) BPM )?\((\d+) bars?\)\.wav$', path.name)
        if match:
            loops.append(dict(name=path.name, label=match[1].strip(), bars=int(match[3]), bpm=float(match[2]) if match[2] else None))
    midi = next(iter(sorted(shots_dir.glob('*Groove.mid'))), None)
    bpm = groove.bpm_from_name(song)
    if bpm is None and midi:
        data = midi.read_bytes()
        offset = data.find(b'\xff\x51\x03')
        if offset >= 0 and offset + 6 <= len(data):
            tempo = int.from_bytes(data[offset + 3:offset + 6], 'big')
            if tempo:
                bpm = 60000000 / tempo
    stems, artifacts = [], []
    for stem in kit + engine.MIX_ORDER:
        path = split_dir / (stem + '.wav')
        if not path.is_file():
            continue
        db = engine.rms_db(path)
        stems.append(dict(name=path.name, preview=path.name, stem=stem, label=engine.KIT_LABEL.get(stem, stem),
                          group='kit' if stem in kit else 'mix', quiet=db is not None and db < -60, db=db))
        artifacts.append(dict(kind='stem', name=path.name, path=str(path)))
    for path in sorted(shots_dir.iterdir()) if shots_dir.exists() else []:
        if path.is_file() and path.suffix in ('.wav', '.mid'):
            artifacts.append(dict(kind='shot', name=path.name, path=str(path)))
    for kind in ['kit', 'all', 'shots']:
        path = output_dir / (kind + '.zip')
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as archive:
            if kind == 'shots':
                wanted = {item['name'] for item in shots}
                for artifact in artifacts:
                    if artifact['kind'] != 'shot':
                        continue
                    if artifact['name'] in wanted or artifact['name'].endswith('.mid'):
                        archive.write(artifact['path'], song + ' - one-shots/' + artifact['name'])
                    elif ' loop ' in artifact['name']:
                        archive.write(artifact['path'], song + ' - one-shots/Loops/' + artifact['name'])
            else:
                for stem in stems:
                    if kind == 'kit' and stem['group'] != 'kit':
                        continue
                    archive.write(split_dir / stem['name'], song + '/' + stem['name'])
        artifacts.append(dict(kind='zip', name=kind, path=str(path)))
    return dict(name=name, status='done', stage='Done', stems=stems, shots=shots, loops=loops,
                groove=midi.name if midi else None, bpm=bpm, duration=round(info.frames / 44100, 1), note=note,
                fullMix=all((split_dir / (stem + '.wav')).is_file() for stem in mix), artifacts=artifacts)
