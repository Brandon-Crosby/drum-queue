const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

function fixture(options = {}) {
  const events = [];
  let releaseModel;
  const samples = {
    prepare: async () => {
      events.push("prepare");
      if (options.prepareFails) throw Error("runtime missing");
    },
    process: async (files) => {
      events.push("samples");
      return { artifacts: [], result: { shots: [] } };
    },
    dispose: () => events.push("dispose"),
  };
  const sandbox = {
    AbortController,
    DrumBrowserPostprocess: { create: () => samples },
    DrumBrowserProcessor: {
      process: async (file, { signal }) => {
        events.push("models");
        if (options.wait)
          await new Promise((resolve, reject) => {
            releaseModel = resolve;
            signal.addEventListener(
              "abort",
              () => reject(new DOMException("Cancelled", "AbortError")),
              { once: true },
            );
          });
        if (options.modelFails) throw Error("model failed");
        return { duration: 9, files: [] };
      },
    },
  };
  vm.runInNewContext(
    fs.readFileSync("web/site/browser/pipeline.js", "utf8"),
    sandbox,
  );
  return {
    pipeline: sandbox.DrumBrowserPipeline.create({
      allowKitOnly: true,
      ...options,
    }),
    events,
    release: () => releaseModel(),
  };
}

test("four-stem processing requires an explicit opt-in before any setup starts", () => {
  assert.throws(
    () => fixture({ allowKitOnly: false }),
    /explicitly for testing/,
  );
});

test("overlap prepares samples before model processing completes; serial retains the control path", async () => {
  for (const overlap of [false, true]) {
    const f = fixture({ overlap });
    const output = await f.pipeline.process({ name: "clip.wav" });
    assert.equal(output.duration, 9);
    assert.equal(f.events[0], overlap ? "prepare" : "models");
    assert.equal(f.events.at(-1), "samples");
    f.pipeline.dispose();
  }
});

test("runtime failure cancels pending models and preserves the underlying error", async () => {
  const f = fixture({ prepareFails: true, wait: true });
  await assert.rejects(
    f.pipeline.process({ name: "clip.wav" }),
    /runtime missing/,
  );
  assert.equal(f.events.at(-1), "dispose");
});

test("external cancellation, model failure and explicit disposal release preparation", async () => {
  const f = fixture({ wait: true }),
    controller = new AbortController();
  const job = f.pipeline.process(
    { name: "clip.wav" },
    { signal: controller.signal },
  );
  await assert.rejects(
    f.pipeline.process({ name: "other.wav" }),
    /already running/,
  );
  controller.abort();
  await assert.rejects(job, { name: "AbortError" });
  assert.equal(f.events.at(-1), "dispose");
  const broken = fixture({ modelFails: true });
  await assert.rejects(
    broken.pipeline.process({ name: "clip.wav" }),
    /model failed/,
  );
  assert.equal(broken.events.at(-1), "dispose");
  const fresh = fixture({ reuse: false });
  await fresh.pipeline.process({ name: "clip.wav" });
  assert.equal(fresh.events.at(-1), "dispose");
});
