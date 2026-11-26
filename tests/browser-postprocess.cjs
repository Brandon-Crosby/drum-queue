const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const { webcrypto } = require("node:crypto");

function clientFixture(t, { failPost = false, fakeTimers = false } = {}) {
  const workers = [];
  const timers = new Map();
  let timerId = 0;
  class Worker {
    constructor() {
      this.messages = [];
      workers.push(this);
    }
    postMessage(message) {
      if (failPost) {
        failPost = false;
        throw Error("Could not clone job");
      }
      this.messages.push(message);
    }
    terminate() {
      this.stopped = true;
    }
    reply(type, extra = {}) {
      this.onmessage({ data: { id: this.messages.at(-1).id, type, ...extra } });
    }
  }
  const sandbox = { Worker, DOMException, setTimeout, clearTimeout };
  if (fakeTimers) {
    sandbox.setTimeout = (callback) => {
      timers.set(++timerId, callback);
      return timerId;
    };
    sandbox.clearTimeout = (id) => timers.delete(id);
  }
  vm.runInNewContext(
    fs.readFileSync("web/site/browser/postprocess.js", "utf8"),
    sandbox,
  );
  const client = sandbox.DrumBrowserPostprocess.create();
  t.after(() => client.dispose());
  return {
    client,
    workers,
    expire: () => {
      const callbacks = [...timers.values()];
      timers.clear();
      callbacks.forEach((f) => f());
    },
  };
}

test("a synchronous post failure does not poison the next preparation", async (t) => {
  const { client, workers } = clientFixture(t, { failPost: true });
  await assert.rejects(client.prepare(), /clone job/);
  assert.equal(workers[0].stopped, true);
  const retry = client.prepare();
  workers[1].reply("ready");
  await retry;
});

test("an idle runtime is released and prepared again when needed", async (t) => {
  const { client, workers, expire } = clientFixture(t, { fakeTimers: true });
  let ready = client.prepare();
  workers[0].reply("ready");
  await ready;
  expire();
  assert.equal(workers[0].stopped, true);
  ready = client.prepare();
  workers[1].reply("ready");
  await ready;
  assert.equal(workers.length, 2);
});

test("preparation is shared and two jobs keep one initialized worker", async (t) => {
  const { client, workers } = clientFixture(t);
  const one = client.prepare(),
    two = client.prepare();
  assert.equal(one, two);
  workers[0].reply("ready");
  await one;
  for (let index = 0; index < 2; index++) {
    const job = client.process([], { name: "clip.wav" });
    await Promise.resolve();
    workers[0].reply("done", { artifacts: [] });
    await job;
  }
  assert.equal(workers.length, 1);
  assert.deepEqual(
    workers[0].messages.map((x) => x.type),
    ["prepare", "run", "run"],
  );
});

test("cancelling preparation terminates it and a retry gets a new worker", async (t) => {
  const { client, workers } = clientFixture(t),
    controller = new AbortController();
  const first = client.prepare({ signal: controller.signal });
  controller.abort();
  await assert.rejects(first, { name: "AbortError" });
  assert.equal(workers[0].stopped, true);
  const retry = client.prepare();
  workers[1].reply("ready");
  await retry;
  workers[0].onerror(); // A late error from a terminated worker must not destroy its replacement.
  assert.equal(workers[1].stopped, undefined);
});

test("a job can cancel immediately while joining an existing preparation", async (t) => {
  const { client, workers } = clientFixture(t);
  const preparation = client.prepare();
  const controller = new AbortController();
  const job = client.process([], { signal: controller.signal });
  const stoppedPreparation = assert.rejects(preparation, {
    name: "AbortError",
  });
  const stoppedJob = assert.rejects(job, { name: "AbortError" });
  controller.abort();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(
    workers[0].stopped,
    true,
    "cancellation must not wait for runtime initialization",
  );
  await Promise.all([stoppedPreparation, stoppedJob]);
});

test("cancelling a job and progress callback errors release runtime state", async (t) => {
  const { client, workers } = clientFixture(t);
  let ready = client.prepare();
  workers[0].reply("ready");
  await ready;
  const controller = new AbortController();
  const job = client.process([], { signal: controller.signal });
  await Promise.resolve();
  await assert.rejects(client.process([]), /already running/);
  controller.abort();
  await assert.rejects(job, { name: "AbortError" });
  assert.equal(workers[0].stopped, true);
  ready = client.prepare({
    onProgress: () => {
      throw Error("callback failed");
    },
  });
  workers[1].reply("progress");
  await assert.rejects(ready, /callback failed/);
  assert.equal(workers[1].stopped, true);
});

async function workerFixture({ corrupt = false } = {}) {
  const bytes = Uint8Array.of(1, 2, 3),
    hash = Buffer.from(
      await webcrypto.subtle.digest("SHA-256", bytes),
    ).toString("hex");
  const manifest = {
    schema: 1,
    runtime: "pyodide@0.29.0",
    python: { bytes: 3, sha256: hash },
    reference: { bytes: 3, sha256: hash },
  };
  const posts = [],
    events = [],
    written = new Map();
  let loads = 0;
  const py = {
    version: "0.29.0",
    loadPackage: async () => {},
    globals: { set() {} },
    FS: {
      mkdirTree() {},
      writeFile: (p, b) => written.set(p, b),
      unlink: (p) => written.delete(p),
      readFile: () => Uint8Array.of(7, 8),
    },
    runPythonAsync: async (code) => {
      if (code.includes("shutil.rmtree('/job'")) {
        events.push("cleanup");
        for (const p of written.keys())
          if (p.startsWith("/job/")) written.delete(p);
      }
      if (code.includes("result = finish("))
        return JSON.stringify({
          shots: [],
          artifacts: [
            { kind: "stem", name: "kick.wav", path: "/job/input/kick.wav" },
          ],
        });
    },
  };
  const sandbox = {
    Uint8Array,
    ArrayBuffer,
    Set,
    crypto: webcrypto,
    importScripts() {},
    loadPyodide: async () => {
      loads++;
      return py;
    },
    fetch: async (url) =>
      url.endsWith("manifest.json")
        ? { ok: true, json: async () => manifest }
        : {
            ok: true,
            arrayBuffer: async () =>
              (corrupt ? Uint8Array.of(9, 9, 9) : bytes).buffer,
          },
    postMessage: (value) => {
      events.push(value.type);
      posts.push(value);
    },
  };
  vm.runInNewContext(
    fs.readFileSync("web/site/browser/postprocess-worker.js", "utf8"),
    sandbox,
  );
  return {
    send: (data) => sandbox.onmessage({ data }),
    posts,
    events,
    written,
    loads: () => loads,
  };
}

test("worker cleans audio before completion and reuses runtime for the next job", async () => {
  const f = await workerFixture();
  await f.send({ type: "prepare", id: 1 });
  for (let id = 2; id < 4; id++) {
    const files = ["kick.wav", "snare.wav", "percussion.wav", "hi-hat.wav"].map(
      (name) => ({ name, bytes: new ArrayBuffer(44) }),
    );
    await f.send({
      type: "run",
      id,
      files,
      name: "clip.wav",
      allowKitOnly: true,
    });
    assert.equal(f.posts.at(-1).type, "done");
    assert.deepEqual(f.events.slice(-2), ["cleanup", "done"]);
    assert.equal(
      [...f.written.keys()].some((x) => x.startsWith("/job/")),
      false,
    );
  }
  assert.equal(f.loads(), 1);
});

test("worker requires full mix by default and rejects duplicate or unsafe filenames", async () => {
  for (const names of [
    ["kick.wav", "snare.wav", "perc.wav", "hh.wav"],
    ["kick.wav", "kick.wav"],
    ["../reference.npz"],
  ]) {
    const f = await workerFixture();
    await f.send({
      type: "run",
      id: 1,
      files: names.map((name) => ({ name, bytes: new ArrayBuffer(44) })),
      name: "clip.wav",
    });
    assert.equal(f.posts.at(-1).type, "error");
    assert.equal(
      [...f.written.keys()].some((x) => x.startsWith("/job/")),
      false,
    );
  }
});

test("worker verifies source assets before installing Python code", async () => {
  const f = await workerFixture({ corrupt: true });
  await f.send({ type: "prepare", id: 1 });
  assert.equal(f.posts.at(-1).type, "error");
  assert.match(f.posts.at(-1).message, /verification/);
  assert.equal(f.written.has("/python.zip"), false);
});
