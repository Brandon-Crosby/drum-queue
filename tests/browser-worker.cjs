const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const { webcrypto } = require("node:crypto");
const source = fs.readFileSync("web/site/browser/worker.js", "utf8");
async function fixture({
  corrupt = false,
  cached = false,
  runFails = false,
  gpu = true,
  configured = true,
  releaseFails = false,
  queueFails = false,
  queueStalls = false,
} = {}) {
  const bytes = Uint8Array.of(1, 2, 3),
    hash = Buffer.from(
      await webcrypto.subtle.digest("SHA-256", bytes),
    ).toString("hex");
  const events = [];
  const posts = [],
    requests = [],
    deleted = [],
    disposed = [];
  let loads = 0,
    releases = 0,
    puts = 0;
  const sandbox = {
    Float32Array,
    Uint8Array,
    ArrayBuffer,
    Blob,
    Response,
    ReadableStream,
    performance,
    setTimeout,
    clearTimeout,
    crypto: webcrypto,
    navigator: { gpu: gpu ? { requestAdapter: async () => ({}) } : undefined },
    postMessage: (message) => {
      events.push(message.type);
      posts.push(message);
    },
    fetch: async (url) => {
      requests.push(url);
      if (url.endsWith("manifest.json"))
        return {
          ok: configured,
          json: async () => ({
            schema: 1,
            runtime: "onnxruntime-web@1.23.2",
            models: { demucs: { sha256: hash, bytes: 3 } },
          }),
        };
      return new Response(bytes);
    },
    caches: {
      open: async () => ({
        match: async () =>
          cached
            ? new Response(corrupt ? Uint8Array.of(0, 0, 0) : bytes)
            : undefined,
        delete: async (url) => deleted.push(url),
        put: async () => puts++,
      }),
    },
    importScripts() {},
    ort: {
      env: {
        wasm: {},
        webgpu: {
          device: Promise.resolve({
            queue: {
              onSubmittedWorkDone: async () => {
                events.push("drain");
                if (queueStalls) return new Promise(() => {});
                if (queueFails) throw Error("Device lost");
              },
            },
            destroy: () => events.push("destroy"),
          }),
        },
      },
      Tensor: class {
        dispose() {
          disposed.push("input");
        }
      },
      InferenceSession: {
        create: async () => {
          loads++;
          return {
            run: async () => {
              if (runFails) throw Error("GPU lost");
              return {
                output: {
                  dispose() {
                    disposed.push("output");
                  },
                },
              };
            },
            release: async () => {
              releases++;
              events.push("release");
              if (releaseFails) throw Error("Release failed");
            },
          };
        },
      },
    },
    DrumDSPNative: { initialize: async () => {} },
    DrumBrowserEngine: {
      check() {},
      async separate(waves, runner) {
        await runner.load("demucs");
        try {
          const result = await runner.run("demucs", {
            wave: { data: bytes, dims: [3] },
          });
          result.dispose();
        } finally {
          await runner.release();
        }
        return [[Float32Array.of(0), Float32Array.of(0)]];
      },
      wav: () => new ArrayBuffer(44),
    },
  };
  vm.createContext(sandbox);
  vm.runInContext(
    fs.readFileSync("web/site/browser/model-loader.js", "utf8"),
    sandbox,
  );
  vm.runInContext(source, sandbox);
  await sandbox.onmessage({
    data: { type: "run", channels: [new ArrayBuffer(4), new ArrayBuffer(4)] },
  });
  return { posts, requests, deleted, disposed, loads, releases, puts, events };
}
test("verified download is cached and tensors/session are released", async () => {
  const t = await fixture();
  assert.equal(t.posts.at(-1).type, "done");
  assert.equal(t.puts, 1);
  assert.equal(t.loads, 1);
  assert.equal(t.releases, 1);
  assert.deepEqual(t.disposed, ["output", "input"]);
  assert.ok(t.requests.every((url) => url.startsWith("/browser-assets/v1/")));
});
test("verified cache avoids downloading model again", async () => {
  const t = await fixture({ cached: true });
  assert.equal(t.posts.at(-1).type, "done");
  assert.equal(t.requests.length, 1);
  assert.equal(t.puts, 0);
});
test("corrupt cache is deleted before GPU session creation", async () => {
  const t = await fixture({ cached: true, corrupt: true });
  assert.equal(t.posts.at(-1).type, "error");
  assert.match(t.posts.at(-1).message, /verification/);
  assert.equal(t.deleted.length, 1);
  assert.equal(t.loads, 0);
});
test("GPU run failure disposes input and session", async () => {
  const t = await fixture({ runFails: true });
  assert.equal(t.posts.at(-1).type, "error");
  assert.deepEqual(t.disposed, ["input"]);
  assert.equal(t.releases, 1);
});
for (const opts of [{ gpu: false }, { configured: false }])
  test(
    "unavailable processing never falls back to inference service " +
      JSON.stringify(opts),
    async () => {
      const t = await fixture(opts);
      assert.equal(t.posts.at(-1).type, "error");
      assert.equal(t.loads, 0);
      assert.ok(t.requests.every((url) => url.startsWith("/browser-assets/")));
    },
  );

test("success destroys the owned GPU device before reporting done", async () => {
  const t = await fixture();
  assert.deepEqual(
    t.events.filter((x) => ["release", "drain", "destroy", "done"].includes(x)),
    ["release", "drain", "destroy", "done"],
  );
});
test("inference failure destroys the GPU device before reporting error", async () => {
  const t = await fixture({ runFails: true });
  assert.ok(t.events.indexOf("destroy") < t.events.indexOf("error"));
  assert.equal(t.events.filter((x) => x === "destroy").length, 1);
});
test("release failure still destroys the device and returns an error", async () => {
  const t = await fixture({ releaseFails: true });
  assert.equal(t.releases, 1);
  assert.equal(t.posts.at(-1).type, "error");
  assert.match(t.posts.at(-1).message, /Release failed/);
  assert.ok(t.events.indexOf("destroy") < t.events.indexOf("error"));
});
test("a lost queue does not discard completed CPU audio", async () => {
  const t = await fixture({ queueFails: true });
  assert.equal(t.posts.at(-1).type, "done");
  assert.ok(t.events.indexOf("destroy") < t.events.indexOf("done"));
});

test(
  "a stalled GPU fence still destroys the device and completes",
  { timeout: 5000 },
  async () => {
    const t = await fixture({ queueStalls: true });
    assert.equal(t.posts.at(-1).type, "done");
    assert.ok(t.events.indexOf("destroy") < t.events.indexOf("done"));
  },
);
