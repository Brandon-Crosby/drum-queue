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
- Real one-shots in the synth clips (Nov 24): 3,500 one-shots picked from the producer's La Uzi library
  (900 kicks, 650 snares, 300 claps, 500 hats, 200 open hats, 300 shakers, 550 percs, 100 cymbals;
  418 MB, on the T7 at DrumSplits-train/oneshots). render.load_bank() uses them 60% of the time when
  /vol/oneshots exists: `modal volume put drumsplit-train "/Volumes/T7/DrumSplits-train/oneshots" /oneshots`
  then re-render the synth clips (prep --synth-only).
- A/B by ear with `eval/ab_kits.py` (same sound numbered the same, the song re-played with each kit).

## Stage 1 (Demucs) percussion
Percs staying in the music is Demucs' call, before Helyx sees anything. Options to test (cheap, no
training): run Helyx on drums + a little of "other" for perc-heavy songs, or a light separation
of "other" for transients. Measure with the perc routing on stunyun 137.

## helyx-8 goal: clean stems, no hiss (Nov 24, 2026)
The producer's goal for Helyx: drum stems, one-shots and loops as clean as possible, no hissing.
helyx-7 at epoch 2 (eval 2026-11-24): kick knock back to helyx-4 level (helyx-6 lost up to 2.4 dB),
snare SDR back down to 1.4 (more snare goes to perc), hat stem the noisiest (leftover junk under hits
median about -7 dB vs about -18 for kick). KUIKI's hats loop audibly hisses.

Where hiss comes from and what to do, in order:
1. Data first: upload the clean sources that are already prepared, then train on them.
   - `modal volume put drumsplit-train "/Volumes/T7/DrumSplits-train/oneshots" /oneshots` (3,500 La Uzi
     one-shots; synth clips use real one-shots 60% of the time)
   - `modal volume put drumsplit-train "/Volumes/T7/DrumSplits-train/flp-renders" /trainset-flp`
     (24 BSR beats rebuilt from their FLPs, 57.5 min, perfectly clean per-role targets)
2. Noise practice (mixaug `noisebed`, new): add hiss, tape/vinyl noise, room tone and a little
   lo-fi crunch to the INPUT mix only, targets stay clean (the noise is "other"). Same pattern as
   quiethats/layerkick. Start p 0.3, -45 to -25 dB under the mix.
3. Silence penalty: MSST's built-in `bleedless_penalty_loss` (penalizes anything in a stem the true stem
   doesn't have, in mel dB) on top of the main loss: `--bleedless 5e-6` on `go`. MSST suggests 2e-5, but
   on test tensors that term outweighed masked_loss ~1000x, so start lower and watch the `fullness`
   metric (now logged with `bleedless` next to SDR): if fullness drops, the stems are getting thin.
4. Safety net in the kit maker (not the model): gentle gate between hits on loops; the junk filter
   keeps covering one-shots. Stems are never post-processed.

How to judge it:
- `eval/hiss.py` (new): level of our stem where the true stem is silent (between hits), in dB under the
  hits; lower is cleaner. It counts hiss and bleed alike. Baseline, median on the test beats:
      stem    helyx-4  helyx-6  helyx-7(ep2)  truth's own floor
      kick     -37.4    -39.0    -42.8          -78
      snare    -30.8    -27.4    -29.4          -76
      perc     -25.2    -30.0    -28.7          -71
      hh       -34.6    -29.4    -35.6          -82
  About 40 dB of room on every stem. helyx-8 should beat helyx-7 here without losing SDR.
- Ears: KUIKI hats loop, plus the A/B kits on the usual beats. Ship only if the ears agree.

Launch (after helyx-7 finishes; data uploaded to /oneshots and /trainset-flp-bsr):
    cd ~/drum-queue-web; and git pull; and modal deploy train/modal_train.py
    modal run --detach train/modal_train.py::go --run helyx-8 --init runs/helyx-7/<best>.ckpt --hours 4 --synth 80 --bleedless 5e-6
`--synth 80` re-renders the synthesized clips so they use the real La Uzi one-shots.

## helyx-8 result (Nov 25, 2026): failed, NaN
Validation SDR 0.0000 after epoch 0, and the log shows loss=nan on every step. Cause, reproduced:
MSST's bleedless penalty converts to dB with AmplitudeToDB (floor 1e-10); under fp16 autocast that floor
underflows to 0, so a silent reference stem (any chunk with no perc or no hats, very common) gives -inf dB
and an infinite loss, which broke the weights. Even in fp32 a silent reference makes the penalty ~150
against a main loss of ~1e-3. Noise-bed practice tested clean (300 random runs, all finite).
Changes: `--bleedless` now refuses to run; training stops itself after 50 NaN/inf steps in a row (NaN
tripwire in msst_patch.py). Next run: helyx-9 = helyx-7 best + noise bed + the ~54k one-shot bank,
first as a 0.5 h sanity run (epoch 0 SDR should land near helyx-7's 2.6).
