#!/usr/bin/env python3
"""
drumsplit — batch drum-kit separation, same chain Xport Studio uses.

  Stage 1: full mix  -> drums / bass / vocals / other   (Demucs, default htdemucs_ft)
  Stage 2: drums     -> kick / snare / toms / hh / ride / crash   (MDX23C-DrumSep)

Usage:
  python3 drumsplit.py "/Volumes/T7/Bad Bunny Albums/Un Verano Sin Ti" -o ~/DrumSplits
  python3 drumsplit.py song1.mp3 song2.wav -o ~/DrumSplits --drums-only

Output, one folder per track (album subfolders preserved):
  ~/DrumSplits/<album>/<track>/drums.wav bass.wav vocals.wav other.wav
                               kick.wav snare.wav toms.wav hh.wav ride.wav crash.wav

Tracks that already have all kit stems are skipped, so you can stop and re-run.
"""
import argparse
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".aiff", ".aif", ".opus"}
DRUMSEP_MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
DRUMSEP_CONFIG = "config_drumsep_mdx23c.yaml"
KIT_STEMS = ["kick", "snare", "toms", "hh", "ride", "crash"]
XPORT_MODELS = Path.home() / "XportStudio" / "Cache" / "SeparatorModels"

log = logging.getLogger("drumsplit")


def find_audio(inputs):
    """Yield (file, base_dir) for every audio file in the given files/folders."""
    for raw in inputs:
        p = Path(raw).expanduser().resolve()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p, p.parent
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in AUDIO_EXTS and not f.name.startswith("._"):
                    yield f, p.parent
        else:
            log.warning("Skipping (not audio or not found): %s", raw)


def stem_label(filename):
    """'Song_(Drums)_htdemucs_ft.wav' -> 'drums'. Uses the last '(...)' group."""
    name = Path(filename).name
    start, end = name.rfind("("), name.rfind(")")
    if start == -1 or end < start:
        return None
    return name[start + 1:end].strip().lower()


def link_xport_drumsep(model_dir):
    """Reuse the DrumSep checkpoint Xport Studio already downloaded (saves 437 MB)."""
    for fname in (DRUMSEP_MODEL, DRUMSEP_CONFIG):
        src, dst = XPORT_MODELS / fname, model_dir / fname
        if src.exists() and not dst.exists():
            try:
                dst.symlink_to(src)
                log.info("Reusing Xport Studio's %s", fname)
            except OSError:
                shutil.copy2(src, dst)


def make_separator(output_dir, model_dir, overlap):
    from audio_separator.separator import Separator
    return Separator(
        output_dir=str(output_dir),
        model_file_dir=str(model_dir),
        output_format="WAV",
        log_level=logging.WARNING,
        mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                     "batch_size": 1, "overlap": overlap, "pitch_shift": 0},
    )


def load(sep, model_filename):
    try:
        sep.load_model(model_filename=model_filename)
    except Exception as e:
        msg = str(e)
        hint = ""
        if "CERTIFICATE_VERIFY_FAILED" in msg or "SSL" in msg:
            hint = ("\n  SSL error while downloading the model. If you use python.org's Python, run\n"
                    "  '/Applications/Python 3.11/Install Certificates.command' once, then retry.")
        sys.exit(f"Could not load {model_filename}: {msg}{hint}")


def run_model(sep, audio_path, work_dir):
    """Separate one file; return {label: absolute_path} of the stems written."""
    outs = sep.separate(str(audio_path))
    stems = {}
    for o in outs:
        p = Path(o)
        if not p.is_absolute():
            p = Path(work_dir) / p
        label = stem_label(p.name)
        if label:
            stems[label] = p
    return stems


def main():
    ap = argparse.ArgumentParser(description="Batch drum-kit stem separation (Demucs -> DrumSep).")
    ap.add_argument("inputs", nargs="+", help="audio files and/or folders (searched recursively)")
    ap.add_argument("-o", "--out", default="~/DrumSplits", help="output folder (default ~/DrumSplits)")
    ap.add_argument("--stage1", default="htdemucs_ft.yaml",
                    help="stage-1 model (default htdemucs_ft.yaml; htdemucs_6s.yaml adds guitar/piano)")
    ap.add_argument("--model-dir", default="~/.drumsplit/models", help="where models are cached")
    ap.add_argument("--overlap", type=int, default=4, help="DrumSep overlap, 2-50 (Xport uses 4; 8 is cleaner, slower)")
    ap.add_argument("--drums-only", action="store_true", help="keep only drums.wav + kit stems")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:                                   # same ffmpeg lookup as the web app
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from engine import ensure_ffmpeg
        if not ensure_ffmpeg():
            log.warning("ffmpeg not found; run ./setup.sh (it installs one).")
    except ImportError:
        pass
    out_root = Path(args.out).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    link_xport_drumsep(model_dir)

    jobs = []
    for f, base in find_audio(args.inputs):
        dest = out_root / f.parent.relative_to(base) / f.stem
        if all((dest / f"{s}.wav").exists() for s in KIT_STEMS):
            log.info("✓ already done: %s", f.name)
            continue
        jobs.append((f, dest))
    if not jobs:
        log.info("Nothing to do.")
        return
    log.info("%d track(s) to split -> %s\n", len(jobs), out_root)

    work = Path(tempfile.mkdtemp(prefix="drumsplit_"))
    t0 = time.time()
    failed = []
    try:
        # Stage 1 — load Demucs once, run every track.
        log.info("Stage 1/2: %s", args.stage1)
        sep = make_separator(work, model_dir, args.overlap)
        load(sep, args.stage1)
        drum_stems = []
        for i, (f, dest) in enumerate(jobs, 1):
            log.info("  [%d/%d] %s", i, len(jobs), f.name)
            try:
                dest.mkdir(parents=True, exist_ok=True)
                stems = run_model(sep, f, work)
                if "drums" not in stems:
                    raise RuntimeError(f"no drums stem in output: {sorted(stems)}")
                for label, p in stems.items():
                    if args.drums_only and label != "drums":
                        p.unlink(missing_ok=True)
                        continue
                    shutil.move(str(p), dest / f"{label}.wav")
                drum_stems.append((f, dest))
            except Exception as e:  # keep going on bad files
                log.error("    ✗ stage 1 failed: %s", e)
                failed.append(f.name)
        del sep

        # Stage 2 — load DrumSep once, run every drums stem.
        log.info("\nStage 2/2: %s", DRUMSEP_MODEL)
        sep = make_separator(work, model_dir, args.overlap)
        load(sep, DRUMSEP_MODEL)
        for i, (f, dest) in enumerate(drum_stems, 1):
            log.info("  [%d/%d] %s", i, len(drum_stems), f.name)
            try:
                stems = run_model(sep, dest / "drums.wav", work)
                for label, p in stems.items():
                    shutil.move(str(p), dest / f"{label}.wav")
                missing = [s for s in KIT_STEMS if not (dest / f"{s}.wav").exists()]
                if missing:
                    log.warning("    missing kit stems: %s (got %s)", missing, sorted(stems))
            except Exception as e:
                log.error("    ✗ stage 2 failed: %s", e)
                failed.append(f.name)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    mins = (time.time() - t0) / 60
    log.info("\nDone in %.1f min. %d ok, %d failed. Output: %s",
             mins, len(jobs) - len(set(failed)), len(set(failed)), out_root)
    for name in sorted(set(failed)):
        log.info("  failed: %s", name)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
