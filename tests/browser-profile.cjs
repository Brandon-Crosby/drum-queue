const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync("tools/browser/profile-worker.js", "utf8");
test("diagnostic timings preserve worker outputs and record the GPU and DSP phases", async () => {
  const posts = [],
    marker = new ArrayBuffer(4);
  const context = vm.createContext({
    performance,
    postMessage: (...args) => posts.push(args),
    importScripts() {},
    DrumDSP: {},
    DrumDSPNative: {
      initialize: async () => {
        context.DrumDSP.spectrum = (x) => x;
        context.DrumDSP.waveform = (x) => x;
      },
    },
    DrumBrowserEngine: {
      separate: async (waves, runner) => {
        await runner.load("demucs");
        await runner.run("demucs");
        context.DrumDSP.spectrum(waves);
        context.DrumDSP.waveform(waves);
        await runner.release();
        return waves;
      },
    },
    ort: { InferenceSession: { create: async () => ({ tag: "session" }) } },
  });
  vm.runInContext(source, context);
  context.importScripts();
  await context.DrumDSPNative.initialize();
  const runner = {
    load: () => context.ort.InferenceSession.create(),
    run: async () => marker,
    release: async () => {},
  };
  assert.equal(
    await context.DrumBrowserEngine.separate(marker, runner),
    marker,
  );
  const message = { type: "done", files: [{ bytes: marker }] },
    transfer = [marker];
  context.postMessage(message, transfer);
  assert.equal(posts[1][0], message);
  assert.equal(posts[1][1], transfer);
  const stats = posts[0][0].stats;
  for (const name of [
    "model-load:demucs",
    "session-create:demucs",
    "inference:demucs",
    "release:demucs",
    "dsp:spectrum",
    "dsp:waveform",
  ]) {
    assert.equal(stats[name].calls, 1);
    assert.ok(stats[name].ms >= 0);
  }
});
test("diagnostics preserve inference failures and still report the measured phase", async () => {
  const posts = [],
    failure = Error("GPU lost");
  const context = vm.createContext({
    performance,
    postMessage: (x) => posts.push(x),
    importScripts() {},
    DrumBrowserEngine: {
      separate: async (_, runner) => {
        await runner.load("helyx");
        return runner.run("helyx");
      },
    },
  });
  vm.runInContext(source, context);
  context.importScripts();
  await assert.rejects(
    context.DrumBrowserEngine.separate([], {
      load: async () => {},
      run: async () => {
        throw failure;
      },
      release: async () => {},
    }),
    (error) => error === failure,
  );
  context.postMessage({ type: "error", message: failure.message });
  assert.equal(posts[0].stats["inference:helyx"].calls, 1);
  assert.equal(posts[1].message, "GPU lost");
});
