"""msst_patch.py - adds a per-stem loss to the MSST trainer (run once on the cloned repo at image build).

    python3 msst_patch.py /msst

Why: MSST's default loss (masked_loss) is MSE on the raw waveform. A loud kick makes big errors in
absolute terms and a quiet snare or hat makes small ones, so the model is barely penalized for a
bad snare - and our snare scores have stayed flat across three runs. MSST's own L1-SNR losses don't
fix it: they flatten all stems of a sample into one signal before computing the ratio.

`stem_l1_snr_loss` scores each stem of each sample on its own, in dB (L1-SNR, like the SDR we grade
with), so a snare that's 20 dB under the kick counts as much as the kick. A silent target stem
scores its leakage against a small floor, so bleed into an empty stem is punished. Optional per-stem
weights (--stem_loss_weights, in the config's instrument order) let snare count extra.
"""
import sys
from pathlib import Path

LOSS = '''

def stem_l1_snr_loss(y_: torch.Tensor, y: torch.Tensor, weights=None, eps: float = 3e-4) -> torch.Tensor:
    """Per-stem L1-SNR in dB: every (sample, stem) pair is scored on its own, so a quiet snare or
    hat counts as much as a loud kick. weights: one per stem (config instrument order), or None."""
    if y_.dim() == 3:                      # [B, C, T]: a single target stem
        y_, y = y_.unsqueeze(1), y.unsqueeze(1)
    b, s = y.shape[0], y.shape[1]
    err = torch.mean(torch.abs(y_.float() - y.float()).reshape(b, s, -1), dim=-1)
    ref = torch.mean(torch.abs(y.float()).reshape(b, s, -1), dim=-1)
    d = 10.0 * torch.log10((err + eps) / (ref + eps))           # [B, S]; lower is better
    if weights is not None:
        w = torch.as_tensor(weights, dtype=d.dtype, device=d.device)[:s]
        return (d * w).sum(dim=1).mean() / w.sum()
    return d.mean()
'''

CHOICE = '''
    if 'stem_l1_snr_loss' in args.loss:
        _w = getattr(args, 'stem_loss_weights', None) or None
        loss_fns.append(
            lambda y_pred, y_true, x=None: stem_l1_snr_loss(y_pred, y_true, _w) * args.stem_l1_snr_loss_coef
        )
'''


def patch(root: Path):
    lf = root / 'utils' / 'losses.py'
    s = lf.read_text()
    if 'def stem_l1_snr_loss' not in s:
        anchor = '\ndef choice_loss('
        assert anchor in s, 'losses.py: choice_loss not found'
        s = s.replace(anchor, LOSS + '\n' + anchor, 1)
        anchor = "    if 'masked_loss' in args.loss:"
        assert anchor in s, 'losses.py: masked_loss branch not found'
        s = s.replace(anchor, CHOICE.lstrip('\n') + '\n' + anchor, 1)
        lf.write_text(s)
    sf = root / 'utils' / 'settings.py'
    s = sf.read_text()
    if "'stem_l1_snr_loss'" not in s:
        anchor = "'fullness_penalty_loss', 'bleedless_penalty_loss'"
        assert anchor in s, 'settings.py: loss choices not found'
        s = s.replace(anchor, anchor + ", 'stem_l1_snr_loss'", 1)
        anchor = '    parser.add_argument("--masked_loss_coef"'
        assert anchor in s, 'settings.py: --masked_loss_coef not found'
        s = s.replace(anchor,
                      '    parser.add_argument("--stem_l1_snr_loss_coef", type=float, default=1.)\n'
                      '    parser.add_argument("--stem_loss_weights", type=float, nargs="+", default=None,\n'
                      '                        help="per-stem weights for stem_l1_snr_loss, config instrument order")\n'
                      + anchor, 1)
        sf.write_text(s)
    patch_dataset(root)
    patch_tripwire(root)
    patch_silent_valid(root)
    patch_sdpa_priority(root)
    print('msst patched: stem_l1_snr_loss, mixaug (aligned grooves, restack, master chain), NaN tripwire, '
          'no score for a stem a validation song does not have')


NAN_LIMIT = 50          # this many training steps in a row with a NaN/inf loss: the model is broken, stop


def patch_tripwire(root: Path):
    """Stop training as soon as the loss stays NaN/inf, instead of burning GPU hours on a broken model
    (helyx-8: the bleedless penalty went infinite on silent stems under fp16 and the run trained on NaN
    for ~3 h). A lone bad step can happen with mixed precision (the grad scaler skips it); 50 in a row
    can't."""
    tf = root / 'train.py'
    s = tf.read_text()
    if 'NaN tripwire' in s:
        return
    a = """            li = loss.item() * gradient_accumulation_steps
            all_losses[f'epoch_{epoch}'].append(li)"""
    assert s.count(a) == 1, 'train.py: loss.item() line not found'
    s = s.replace(a, """            li = loss.item() * gradient_accumulation_steps
            globals()['_bad_steps'] = 0 if math.isfinite(li) else globals().get('_bad_steps', 0) + 1
            if globals()['_bad_steps'] >= %d:
                raise SystemExit(f'NaN tripwire: {globals()["_bad_steps"]} training steps in a row with loss={li}; '
                                 'stopping (the model weights are broken; see train/NEXT_RUN.md)')
            all_losses[f'epoch_{epoch}'].append(li)""" % NAN_LIMIT, 1)
    if '\nimport math' not in s:
        s = 'import math\n' + s
    tf.write_text(s)


def patch_sdpa_priority(root: Path):
    """MSST's RoFormer attention calls sdpa_kernel(..., set_priority=True) at inference (validation), which
    needs torch >= 2.6; our image pins 2.5.1 (Helyx is tuned on it). satyr-1b crashed there on its first
    validation. Without set_priority torch 2.5 still picks among the same fast kernels, just in its own order."""
    import torch
    if tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) >= (2, 6):
        return
    for f in (root / 'models').rglob('*.py'):
        s = f.read_text()
        if 'set_priority=True' in s:
            f.write_text(s.replace(', set_priority=True)', ')'))


def patch_silent_valid(root: Path):
    """A validation song with no toms (most songs) scored Satyr's toms at about -48 dB SDR, which swamped the
    average that picks the best checkpoint (CPU test, Nov 27 2026). A stem the song doesn't have isn't scored
    for that song. Helyx's validation songs have every drum part, so its scores don't change."""
    vf = root / 'valid.py'
    s = vf.read_text()
    if 'silent in this song' in s:
        return
    a = """                track, sr1 = read_audio_transposed(f"{folder}/{instr}.{extension}", instr, skip_err=True)
                if track is None:
                    continue"""
    assert s.count(a) == 1, 'valid.py: reference read not found'
    s = s.replace(a, a + """
                if np.abs(track).max() < 1e-4:          # silent in this song (e.g. no toms): nothing to score
                    continue""", 1)
    vf.write_text(s)


def patch_dataset(root: Path):
    """helyx-4: training mixes from real grooves (some of the time), re-stacked snares, master chain.
    See mixaug.py; all three are off unless the config's augmentations set their probabilities."""
    import shutil
    shutil.copy(Path(__file__).resolve().parent / 'mixaug.py', root / 'utils' / 'mixaug.py')
    df = root / 'utils' / 'dataset.py'
    s = df.read_text()
    if 'mixaug' in s:
        return
    a = "import warnings"
    assert a in s, 'dataset.py: imports not found'
    s = s.replace(a, a + "\ntry:\n    from utils import mixaug\nexcept ImportError:\n    import mixaug", 1)
    a = """        elif self.dataset_type in [1, 2, 3]:
            res = self.load_random_mix()"""
    assert a in s, 'dataset.py: random-mix branch not found'
    s = s.replace(a, """        elif self.dataset_type in [1, 2, 3]:
            _aug = self.config['augmentations'] if self.aug else {}
            if self.aug and random.random() < mixaug._get(_aug, 'aligned_prob'):
                res, _ = self.load_aligned_data()           # one song's stems as the producer placed them
            else:
                res = self.load_random_mix()
            if self.aug:
                res = mixaug.restack(res, list(self.instruments), _aug)
                res = mixaug.darken(res, list(self.instruments), _aug)
                res = mixaug.onbeat(res, list(self.instruments), _aug)
                res = mixaug.nokick(res, list(self.instruments), _aug)
                res = mixaug.quiethats(res, list(self.instruments), _aug)
                res = mixaug.layerkick(res, list(self.instruments), _aug)
                res = mixaug.noisebed(res, list(self.instruments), _aug)""", 1)
    a = """        if self.dataset_type != 6 and self.dataset_type!=7:
            mix = res.sum(0)
"""
    assert a in s, 'dataset.py: mix sum not found'
    s = s.replace(a, a + """        if self.aug and self.dataset_type in [1, 2, 3]:
            res, mix = mixaug.master(res, self.config['augmentations'])
""", 1)
    df.write_text(s)


if __name__ == '__main__':
    patch(Path(sys.argv[1] if len(sys.argv) > 1 else '/msst'))
