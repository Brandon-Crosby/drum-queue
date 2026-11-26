const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
require("../web/site/browser/dsp.js");
const reference = { ...DrumDSP };
require("../web/site/browser/dsp-native.js");
let seed = 7;
const random = () => {
  seed = (Math.imul(seed, 1664525) + 1013904223) | 0;
  return (seed >>> 0) / 2 ** 32 - 0.5;
};
function compare(actual, expected) {
  assert.equal(actual.length, expected.length);
  let maximum = 0;
  for (let i = 0; i < actual.length; i++) {
    assert.ok(Number.isFinite(actual[i]));
    maximum = Math.max(maximum, Math.abs(actual[i] - expected[i]));
  }
  assert.ok(maximum <= 1e-7, "maximum transform error " + maximum);
}
const ready = WebAssembly.instantiate(
  fs.readFileSync(process.env.DS_DSP_MODULE || "web/site/browser/dsp.wasm"),
  {},
).then(({ instance }) => DrumDSPNative.install(instance.exports));
for (const nfft of [2048, 4096])
  test(
    "WASM analysis/reconstruction matches float64 JS reference " + nfft,
    async () => {
      await ready;
      const helyx = nfft === 2048,
        length = helyx ? 130560 : 343980,
        options = {
          nfft,
          hop: nfft / 4,
          frames: helyx ? 256 : 336,
          bins: nfft / 2,
          offset: helyx ? 0 : 512,
          normalized: !helyx,
          extraFrames: helyx ? 0 : 2,
          stems: 1,
          length,
        };
      const input = Array.from({ length: 2 }, () =>
        Float32Array.from({ length }, random),
      );
      const expected = reference.spectrum(input, options),
        actual = DrumDSP.spectrum(input, options);
      compare(actual, expected);
      const oldWaves = reference.waveform(expected, options),
        waves = DrumDSP.waveform(actual, options);
      for (let c = 0; c < 2; c++) compare(waves[0][c], oldWaves[0][c]);
      if (helyx) {
        const full = new Float32Array(actual.length * 4);
        for (let s = 0; s < 4; s++) full.set(actual, s * actual.length);
        const many = DrumDSP.waveform(full, { ...options, stems: 4 });
        assert.equal(many.length, 4);
        for (const stem of many)
          for (let c = 0; c < 2; c++) compare(stem[c], oldWaves[0][c]);
      }
    },
  );
test("WASM rejects shapes that exceed its fixed workspace", async () => {
  await ready;
  assert.throws(() =>
    DrumDSP.spectrum([new Float32Array(0)], {
      nfft: 2048,
      hop: 512,
      bins: 1024,
      frames: 256,
    }),
  );
  assert.throws(() =>
    DrumDSP.waveform(new Float32Array(1), {
      nfft: 2048,
      hop: 512,
      bins: 1024,
      frames: 256,
      stems: 4,
      length: 130560,
    }),
  );
});
