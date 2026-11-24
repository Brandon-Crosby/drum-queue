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
helyx-4 (helyx-1 -> 4 were each trained on top of the last, so starting over from stock DrumSep would
throw away hours of learning the kick/snare/perc/hh/other split). Same MSE loss as helyx-4, so the new
data and mixes are the only change. If the dark-snare habit survives, try stock DrumSep next.

## Ready (Nov 22, 2026)
18 candidate beats (41 min) prepped as training copies in ~/DrumSplits/trainset-candidates
(roles reviewed: doubtful tracks, drum loops/buses and mislabeled-sounding tracks left out; roles in
~/DrumSplits/roles-candidates). Upload, then launch:
    modal volume put drumsplit-train ~/DrumSplits/trainset-candidates /trainset-candidates
    modal deploy train/modal_train.py
    modal run train/modal_train.py::go --run helyx-6 --hours 2 --synth 80 --init runs/helyx-4/last_mdx23c.ckpt
Note: tools/zone-six/build.py checks that the site still uses helyx-4; switching the site to a new
model needs that guard updated too.

## How to judge it
`modal run eval/batch_eval.py --ckpt <run>/last_mdx23c.ckpt --label <run>` and compare with:
- snare routing (alone/stacked) and hit cleanliness: `eval/baselines/helyx-4-*.json`
- stunyun 137: snare 2.6 dB SDR, 59% pure, 91% of hits found, -4.6 dB clean; hats -14.6 dB
- hasta luego snare: helyx-4 1% / stock DrumSep 96%
Only ship if it beats helyx-4 overall *and* by ear on A/B one-shots.

## helyx-6 results and the run after it (Nov 23, 2026)
helyx-6 (4 h from helyx-4, darken/onbeat/nokick, +18 beats): kit score 67% -> 73% real one-shots
(snare/perc 64 -> 75%, hats 69 -> 63%); snare SDR 1.3 -> 3.0. Producer's ears on church's chicken:
helyx-4 kick deeper, helyx-6 hats are noise, helyx-6 found a perc helyx-4 missed.

Found on church's chicken: its shaker (vbo sandpaper) and open hat sit ~30 dB under loud percs; helyx-6
put 60% of them in the perc stem, 17% in snare and only 4% in hats, so the hat one-shots were kick bleed.
Its snare-tom (93% stacked on percs) goes about half to the kick stem in both models.

For the next run:
- `quiethats_prob: 0.25` (in the config): hats 12-30 dB under the other drums.
- Shakers count as hats, always (the producer: "shakers should double as a hihat"). Roles already do this.
- Data: beats with quiet hats/shakers under busy percussion; snare-toms stacked on percs.
- Kicks keep their top layer (Nov 24): the producer hears kicks missing their click/snap (providence:
  0.6-6 kHz attack 3-13 dB under the real kick; TABLA's old-school click layer above 500 Hz missing).
  The layer isn't recoverable from the other stems after the fact (at kick hits it's < 1/6 coherent in
  snare+perc), so the model must keep it: add layered kicks to the practice mixes (a clean kick plus a
  short bright click/old-school layer, both in the kick target), and real beats with layered kicks.
  Also: hasta luego's kick stem carries a ~55 Hz tone ringing past the real kick (done by 100 ms).
- A/B by ear with `eval/ab_kits.py` (same sound numbered the same, the song re-played with each kit).

## Stage 1 (Demucs) percussion
Percs staying in the music is Demucs' call, before Helyx sees anything. Options to test (cheap, no
training): run Helyx on drums + a little of "other" for perc-heavy songs, or a light separation
of "other" for transients. Measure with the perc routing on stunyun 137.
