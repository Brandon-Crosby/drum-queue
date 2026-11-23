# Next Helyx run: what it has to fix, and what's ready

Nothing here has been trained yet. Check with Brando before spending (the Modal balance is shared
with the live site).

## What's wrong with helyx-4 (measured, Nov 2026)

| Problem | Evidence | Fix in the data (ready) |
|---|---|---|
| Dark snares go to the kick | hasta luego: 1% of snare hits right (stock DrumSep 96%). Only 1.1% of 26k real training snare hits are that dark; every synth snare was bright | `synth.dark_snare`; `mixaug.darken` (darken_prob 0.3) |
| Music on the beat goes into the snare stem | stunyun137: a synth (Serum/Zenology) and a vox chop became "snare" one-shots; snare stem only 59% pure. Training only mixed music in quietly (-26 to -16 dB) at random times | `mixaug.onbeat` (onbeat_prob 0.3): chops of the music laid on the snare/perc hits, louder, kept in `other` |
| No-kick, 808-led beats | stunyun137 has no kick; the 808's attack ends up in the snare/hat stems | `mixaug.nokick` (nokick_prob 0.1) |
| Stacked snare+kick | fails for every tool (Helyx 0-4% on TABLAaLAS / church's chicken) | `mixaug.restack` (since helyx-5); still unsolved |
| Percussion lost to the music | stunyun137: 42% of perc hits stayed in Demucs' "other" | Stage 1 (Demucs), not Helyx: needs its own fix (see below) |

## Data to add before the run
- `~/DrumSplits/training-candidates (dark + stacked snares).json`: 9 dark-snare and 13 stacked beats.
- Perc-heavy beats with stems (several percs, congas/clav/bongos).
- 808-led beats without a kick.
- Fix or drop MOONBOY KAOS (its "kick" track is mostly not kicks).

## Never train on
The test beats (17 + breakdown136 + stunyun 137). Test folders carry `test_only` in roles.json and
`make_testset.build(full=True)` refuses them.

## Start point
Stock DrumSep handles dark snares far better than helyx-4, so start this run from stock DrumSep (the
default), not from helyx-4.

## How to judge it
`modal run eval/batch_eval.py --ckpt <run>/last_mdx23c.ckpt --label <run>` and compare with:
- snare routing (alone/stacked) and hit cleanliness: `eval/baselines/helyx-4-*.json`
- stunyun 137: snare 2.6 dB SDR, 59% pure, 91% of hits found, -4.6 dB clean; hats -14.6 dB
- hasta luego snare: helyx-4 1% / stock DrumSep 96%
Only ship if it beats helyx-4 overall *and* by ear on A/B one-shots.

## Stage 1 (Demucs) percussion
Percs staying in the music is Demucs' call, before Helyx sees anything. Options to test (cheap, no
training): run Helyx on drums + a little of "other" for perc-heavy songs, or a light separation
of "other" for transients. Measure with the perc routing on stunyun 137.
