"""engine.py - full mix -> drums (Demucs) -> kick/snare/toms/hh/ride/crash (MDX23C DrumSep).

Same two-stage chain Xport Studio's "drum kit" add-on uses, via python-audio-separator.
Pure functions plus a small model cache, no web code. app.py calls process() once per job.
"""
import logging
import re
import shutil
import sys
from pathlib import Path

AUDIO_EXTS = {'.mp3', '.wav', '.flac', '.m4a', '.aiff', '.aif', '.ogg', '.opus'}

MIX_MODELS = {                       # option value -> audio-separator model file
    'htdemucs_ft': 'htdemucs_ft.yaml',   # best drums, slowest (4-model bag)
    'htdemucs':    'htdemucs.yaml',      # ~4x faster, slightly bleedier
    'htdemucs_6s': 'htdemucs_6s.yaml',   # also splits guitar + piano
}
DRUMSEP = 'MDX23C-DrumSep-aufr33-jarredou.ckpt'
DRUMSEP_YAML = 'config_drumsep_mdx23c.yaml'
KIT = ['kick', 'snare', 'toms', 'hh', 'ride', 'crash']
KIT_LABEL = {'hh': 'hi-hat'}
MIX_ORDER = ['drums', 'bass', 'vocals', 'guitar', 'piano', 'other']
PREVIEW_SR = 22050

MODEL_DIR = Path.home() / '.drumsplit' / 'models'
XPORT_MODELS = Path.home() / 'XportStudio' / 'Cache' / 'SeparatorModels'

log = logging.getLogger('drumqueue')


def safe_name(s):
    s = re.sub(r"[^\w\-\. ()&']+", '_', s).strip(' ._')
    return s or 'track'


def _label_of(filename):
    """'input_(Drums)_htdemucs_ft.wav' -> 'drums'  (last '(...)' group)."""
    a, b = filename.rfind('('), filename.rfind(')')
    return filename[a + 1:b].strip().lower() if 0 <= a < b else None


def link_xport_drumsep():
    """Reuse the DrumSep checkpoint Xport Studio already downloaded (437 MB)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for f in (DRUMSEP, DRUMSEP_YAML):
        src, dst = XPORT_MODELS / f, MODEL_DIR / f
        if src.exists() and not dst.exists():
            try:
                dst.symlink_to(src)
            except OSError:
                shutil.copy2(src, dst)


def ensure_ffmpeg():
    """audio-separator refuses to start without an `ffmpeg` on PATH. Use, in order: one already
    on PATH, Xport Studio's bundled copy, or the one from the imageio-ffmpeg package."""
    import os
    if shutil.which('ffmpeg'):
        return shutil.which('ffmpeg')
    cands = [XPORT_MODELS.parent / 'bin' / 'ffmpeg']
    try:
        import imageio_ffmpeg
        cands.append(Path(imageio_ffmpeg.get_ffmpeg_exe()))
    except Exception:
        pass
    for exe in cands:
        if exe.is_file() and os.access(exe, os.X_OK):
            bindir = Path.home() / '.drumsplit' / 'bin'      # PATH needs a file literally named ffmpeg
            bindir.mkdir(parents=True, exist_ok=True)
            link = bindir / 'ffmpeg'
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(exe)
            os.environ['PATH'] = f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"
            return str(link)
    return None


def friendly_error(e):
    msg = str(e).strip() or type(e).__name__
    if 'CERTIFICATE_VERIFY_FAILED' in msg or 'SSLError' in msg or 'SSL:' in msg:
        return ('Model download failed with an SSL certificate error (same thing that broke Xport\'s '
                'drum split). Turn off any VPN / web filter and retry, or run '
                '"/Applications/Python 3.11/Install Certificates.command" if you have python.org Python.')
    if 'ffmpeg' in msg.lower():
        return ('ffmpeg not found. Run "./setup.sh" again (it now installs one), then restart ./run.sh and Retry.')
    if any(k in msg for k in ('ConnectionError', 'Max retries', 'NameResolution', 'timed out')):
        return 'Could not download the model (no internet?). It only needs to download once. ' + msg[:200]
    return msg


# ------------------------------------------------------------- model cache
class Models:
    """Keeps the loaded separators between jobs, so only the first song pays the load time.
    Only touched from the single worker thread."""

    def __init__(self, work_dir):
        self.work = Path(work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.mix = self.mix_key = None
        self.kit = self.kit_key = None

    def _make(self, overlap=4):
        ensure_ffmpeg()
        from audio_separator.separator import Separator
        return Separator(output_dir=str(self.work), model_file_dir=str(MODEL_DIR), output_format='WAV',
                         log_level=logging.WARNING,
                         mdxc_params={'segment_size': 256, 'override_model_segment_size': False,
                                      'batch_size': 1, 'overlap': overlap, 'pitch_shift': 0})

    def get_mix(self, key):
        if self.mix_key != key:
            self.mix = None
            s = self._make()
            s.load_model(model_filename=MIX_MODELS[key])
            self.mix, self.mix_key = s, key
        return self.mix

    def get_kit(self, overlap):
        if self.kit_key != overlap:
            self.kit = None
            link_xport_drumsep()
            s = self._make(overlap)
            s.load_model(model_filename=DRUMSEP)
            self.kit, self.kit_key = s, overlap
        return self.kit

    def run(self, sep, audio):
        for f in self.work.iterdir():              # leftovers from a crashed job
            if f.is_file():
                f.unlink()
        out = {}
        for o in sep.separate(str(audio)):
            p = Path(o)
            if not p.is_absolute():
                p = self.work / p
            lab = _label_of(p.name)
            if lab:
                out[lab] = p
        return out


# ------------------------------------------------------------------ helpers
def analyze(path):
    """(bpm | None, duration seconds). Tempo folded into 70-150 like MIDI Queue."""
    bpm, dur = None, 0.0
    try:
        import librosa
        import numpy as np
        dur = float(librosa.get_duration(path=str(path)))
        y, sr = librosa.load(str(path), sr=22050, mono=True, offset=min(15, dur / 4), duration=120)
        t, _ = librosa.beat.beat_track(y=y, sr=sr)
        b = float(np.atleast_1d(t)[0])
        if b > 0:
            while b < 70:
                b *= 2
            while b >= 150:
                b /= 2
            bpm = round(b, 1)
    except Exception:
        pass
    return bpm, dur


def to_wav(src, dst):
    """Hand the separator a plain 44.1k stereo WAV so mp3/m4a never need ffmpeg."""
    import librosa
    import numpy as np
    import soundfile as sf
    y, sr = librosa.load(str(src), sr=44100, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    sf.write(str(dst), y.T, sr, subtype='PCM_16')


def save_preview(src, dst):
    """Small mono copy for the in-browser mixer (full-quality files are the downloads)."""
    import librosa
    import soundfile as sf
    y, _ = librosa.load(str(src), sr=PREVIEW_SR, mono=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y, PREVIEW_SR, subtype='PCM_16')


def rms_db(path):
    try:
        import numpy as np
        import soundfile as sf
        y, _ = sf.read(str(path), dtype='float32')
        r = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
        return -120.0 if r <= 0 else round(float(20 * np.log10(r)), 1)
    except Exception:
        return None


def unique_dir(root, name):
    d, n = root / name, 2
    while d.exists():
        d, n = root / f'{name} ({n})', n + 1
    return d


# ------------------------------------------------------------------ the job
def process(job, out_root, models, stage):
    """Split one song. Returns fields to merge into the job record."""
    opts = job['opts']
    src = Path(job['src'])
    out_root = Path(out_root)
    out_dir = Path(job['out_dir']) if job.get('out_dir') else unique_dir(out_root, safe_name(Path(job['name']).stem))
    out_dir = out_root / out_dir.name
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / 'preview').mkdir(parents=True)

    stage('Reading audio')
    bpm, dur = analyze(src)
    wav = models.work.parent / 'input.wav'
    to_wav(src, wav)

    stage('Loading mix model' if models.mix_key != opts['model'] else 'Splitting drums from the mix')
    mix = models.get_mix(opts['model'])
    stage('Splitting drums from the mix')
    stems = models.run(mix, wav)
    wav.unlink(missing_ok=True)
    if 'drums' not in stems:
        raise RuntimeError(f'Demucs gave no drums stem (got {sorted(stems)})')
    shutil.move(str(stems.pop('drums')), out_dir / 'drums.wav')
    for lab, p in stems.items():
        if opts.get('keep_full', True):
            shutil.move(str(p), out_dir / f'{lab}.wav')
        else:
            p.unlink(missing_ok=True)

    stage('Loading kit model' if models.kit_key != opts['overlap'] else 'Splitting kit pieces')
    kit = models.get_kit(opts['overlap'])
    stage('Splitting kit pieces')
    pieces = models.run(kit, out_dir / 'drums.wav')
    for lab, p in pieces.items():
        shutil.move(str(p), out_dir / f'{lab}.wav')

    stage('Making previews')
    out = []
    for name in KIT + MIX_ORDER:
        f = out_dir / f'{name}.wav'
        if not f.exists():
            continue
        save_preview(f, out_dir / 'preview' / f.name)
        level = rms_db(f)
        out.append(dict(name=f.name, stem=name, label=KIT_LABEL.get(name, name),
                        group='kit' if name in KIT else 'mix',
                        quiet=bool(level is not None and level < -60), db=level))
    missing = [k for k in KIT if not (out_dir / f'{k}.wav').exists()]
    note = f'Missing kit pieces: {", ".join(missing)}' if missing else ''
    return dict(stems=out, out_dir=out_dir.name, bpm=bpm, duration=round(dur, 1), note=note)


def separator_available():
    import importlib.util
    try:
        return importlib.util.find_spec('audio_separator') is not None
    except Exception:
        return False


if __name__ == '__main__':
    print('engine ok; python', sys.version.split()[0])
