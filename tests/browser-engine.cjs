const { test } = require("node:test");
const assert = require("node:assert/strict");
require("../web/site/browser/engine.js");
const engine = globalThis.DrumBrowserEngine;
test("reject invalid, long and nonfinite audio before loading models", () => {
  for (const waves of [
    [],
    [new Float32Array(1)],
    [new Float32Array(1), new Float32Array(2)],
    [new Float32Array(44100 * 30 + 1), new Float32Array(44100 * 30 + 1)],
    [Float32Array.of(NaN), Float32Array.of(0)],
  ])
    assert.throws(() => engine.check(waves));
});
test("silence returns finite full-length stems without allocating a model", async () => {
  const out = await engine.separate(
    [new Float32Array(44100), new Float32Array(44100)],
    {
      load() {
        throw Error("unnecessary model allocation");
      },
    },
    () => {},
  );
  assert.equal(out.length, 4);
  for (const stem of out)
    for (const channel of stem) {
      assert.equal(channel.length, 44100);
      assert.ok(channel.every((v) => v === 0));
    }
});
test("PCM24 WAV preserves stereo layout, rate, clipping and frame count", () => {
  const b = engine.wav([Float32Array.of(-2, 0.5), Float32Array.of(2, -0.5)]),
    v = new DataView(b);
  assert.equal(b.byteLength, 56);
  assert.equal(v.getUint32(24, true), 44100);
  assert.equal(v.getUint16(34, true), 24);
  assert.equal(v.getUint16(22, true), 2);
  const pcm = (at) => {
    let n =
      v.getUint8(at) | (v.getUint8(at + 1) << 8) | (v.getUint8(at + 2) << 16);
    return n & 0x800000 ? n - 0x1000000 : n;
  };
  assert.equal(pcm(44), -8388608);
  assert.equal(pcm(47), 8388607);
  assert.equal(pcm(50), 4194304);
  assert.equal(pcm(53), -4194303);
});
test("overlap plans cover exact and partial boundaries", () => {
  for (const length of [1, 32640, 130560, 195841, 343980, 44100 * 30])
    for (const [size, step] of [
      [343980, 257985],
      [130560, 32640],
    ]) {
      const coverage = new Uint8Array(length);
      for (const chunk of engine.plans(length, size, step)) {
        assert.ok(chunk.length > 0);
        coverage.fill(1, chunk.start, chunk.start + chunk.length);
      }
      assert.ok(coverage.every((v) => v === 1));
    }
});
