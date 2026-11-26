const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const { JSDOM } = require("jsdom");
const script = fs.readFileSync("web/site/browser/processor.js", "utf8");
const tick = () => new Promise((resolve) => setImmediate(resolve));
function setup({ seconds = 9, gpu = true, decode } = {}) {
  const dom = new JSDOM(
      "<!doctype html><body><main>Existing application</main></body>",
      { runScripts: "outside-only" },
    ),
    w = dom.window,
    workers = [],
    revoked = [];
  Object.defineProperty(w.navigator, "gpu", { value: gpu ? {} : undefined });
  w.URL.createObjectURL = () => "blob:metadata";
  w.URL.revokeObjectURL = (url) => revoked.push(url);
  w.HTMLMediaElement.prototype.load = function () {};
  Object.defineProperty(w.HTMLMediaElement.prototype, "src", {
    set(value) {
      this.setAttribute("src", value);
      setImmediate(() => {
        if (this.onloadedmetadata) {
          Object.defineProperty(this, "duration", { value: seconds });
          this.onloadedmetadata();
        }
      });
    },
  });
  w.AudioContext = class {
    async close() {}
    async decodeAudioData() {
      return decode
        ? decode()
        : {
            sampleRate: 44100,
            length: 9 * 44100,
            numberOfChannels: 1,
            getChannelData: () => new w.Float32Array(9 * 44100),
          };
    }
  };
  w.Worker = class {
    constructor(url) {
      assert.equal(url, "/browser/worker.js");
      workers.push(this);
    }
    terminate() {
      this.terminated = true;
    }
    postMessage(data, transfer) {
      this.input = data;
      this.transfer = transfer;
    }
  };
  w.fetch = () => {
    throw Error("processor must not call cloud APIs");
  };
  w.XMLHttpRequest = class {
    constructor() {
      throw Error("processor must not upload audio");
    }
  };
  w.eval(script);
  return {
    dom,
    w,
    workers,
    revoked,
    api: w.DrumBrowserProcessor,
    file: { size: 100, arrayBuffer: async () => new ArrayBuffer(1) },
  };
}
test("returns PCM WAV buffers and progress without rendering a new interface", async () => {
  const t = setup(),
    before = t.w.document.body.innerHTML,
    progress = [];
  try {
    const done = t.api.process(t.file, { onProgress: (p) => progress.push(p) });
    await tick();
    await tick();
    const worker = t.workers[0];
    assert.equal(worker.input.channels.length, 2);
    assert.equal(worker.transfer.length, 2);
    worker.onmessage({
      data: { type: "progress", stage: "kit", message: "Splitting kit" },
    });
    const files = ["kick", "snare", "percussion", "hi-hat"].map((name) => ({
      name: name + ".wav",
      bytes: new ArrayBuffer(44),
    }));
    worker.onmessage({ data: { type: "done", files } });
    const result = await done;
    assert.equal(result.duration, 9);
    assert.deepEqual(result.files, files);
    assert.equal(progress.at(-1).stage, "kit");
    assert.equal(worker.terminated, true);
    assert.deepEqual(t.revoked, ["blob:metadata"]);
    assert.equal(t.w.document.body.innerHTML, before);
  } finally {
    t.w.close();
  }
});
for (const [label, opts] of [
  ["unsupported GPU", { gpu: false }],
  ["long clip", { seconds: 31 }],
  ["invalid duration", { seconds: Infinity }],
])
  test("rejects " + label + " before creating a worker", async () => {
    const t = setup(opts);
    try {
      await assert.rejects(t.api.process(t.file));
      assert.equal(t.workers.length, 0);
    } finally {
      t.w.close();
    }
  });
test("rejects oversized files before decoding", async () => {
  const t = setup();
  try {
    await assert.rejects(t.api.process({ ...t.file, size: 31 * 1024 * 1024 }));
    assert.equal(t.workers.length, 0);
  } finally {
    t.w.close();
  }
});
test("cancel terminates inference, ignores late replies, and permits retry", async () => {
  const t = setup();
  try {
    const done = t.api.process(t.file);
    const rejected = assert.rejects(done, { name: "AbortError" });
    await tick();
    await tick();
    const old = t.workers[0];
    t.api.cancel();
    await rejected;
    assert.equal(old.terminated, true);
    old.onmessage({ data: { type: "done", files: [] } });
    const retry = t.api.process(t.file);
    await tick();
    await tick();
    assert.equal(t.workers.length, 2);
    t.workers[1].onmessage({ data: { type: "done", files: [] } });
    await retry;
  } finally {
    t.w.close();
  }
});
test("external cancellation interrupts decode without starting inference", async () => {
  let finishDecode;
  const t = setup({
    decode: () =>
      new Promise((resolve) => {
        finishDecode = resolve;
      }),
  });
  try {
    const controller = new t.w.AbortController(),
      done = t.api.process(t.file, { signal: controller.signal }),
      rejected = assert.rejects(done, { name: "AbortError" });
    await tick();
    await tick();
    controller.abort();
    await rejected;
    assert.equal(t.workers.length, 0);
    finishDecode({});
    await tick();
  } finally {
    t.w.close();
  }
});
test("worker error rejects and frees the worker without cloud fallback", async () => {
  const t = setup();
  try {
    const done = t.api.process(t.file),
      rejected = assert.rejects(done, /Model verification failed/);
    await tick();
    await tick();
    t.workers[0].onmessage({
      data: { type: "error", message: "Model verification failed" },
    });
    await rejected;
    assert.equal(t.workers[0].terminated, true);
  } finally {
    t.w.close();
  }
});
test("concurrent processing is rejected while the first split runs", async () => {
  const t = setup();
  try {
    const done = t.api.process(t.file),
      rejected = assert.rejects(done, { name: "AbortError" });
    await assert.rejects(t.api.process(t.file), /already running/);
    t.api.cancel();
    await rejected;
  } finally {
    t.w.close();
  }
});

function pcmFile(seconds, { bits = 24, format = 1, junk = false } = {}) {
  const channels = 2,
    rate = 44100,
    align = (channels * bits) / 8;
  const dataBytes = seconds * rate * align,
    offset = junk ? 10 : 0;
  const header = new ArrayBuffer(44 + offset),
    view = new DataView(header);
  const tag = (at, value) =>
    [...value].forEach((c, i) => view.setUint8(at + i, c.charCodeAt(0)));
  tag(0, "RIFF");
  view.setUint32(4, 36 + offset + dataBytes, true);
  tag(8, "WAVE");
  if (junk) {
    tag(12, "JUNK");
    view.setUint32(16, 1, true);
  }
  tag(12 + offset, "fmt ");
  view.setUint32(16 + offset, 16, true);
  view.setUint16(20 + offset, format, true);
  view.setUint16(22 + offset, channels, true);
  view.setUint32(24 + offset, rate, true);
  view.setUint32(28 + offset, rate * align, true);
  view.setUint16(32 + offset, align, true);
  view.setUint16(34 + offset, bits, true);
  tag(36 + offset, "data");
  view.setUint32(40 + offset, dataBytes, true);
  return {
    size: header.byteLength + dataBytes,
    slice: (start, end) => {
      assert.equal(start, 0);
      assert.equal(end, 65536);
      return { arrayBuffer: async () => header };
    },
    arrayBuffer: async () => new ArrayBuffer(1),
  };
}
for (const options of [{}, { bits: 32, format: 3 }, { junk: true }])
  test(
    "repeated WAV processing avoids the media-element metadata loader " +
      JSON.stringify(options),
    async () => {
      const t = setup(),
        file = pcmFile(9, options);
      t.w.URL.createObjectURL = () => {
        throw Error("WAV must not depend on media-element loading");
      };
      try {
        for (let i = 0; i < 2; i++) {
          const done = t.api.process(file);
          await tick();
          await tick();
          t.workers[i].onmessage({ data: { type: "done", files: [] } });
          assert.equal((await done).duration, 9);
          assert.equal(t.workers[i].terminated, true);
        }
      } finally {
        t.w.close();
      }
    },
  );
test("WAV duration guard rejects a long clip before decoding", async () => {
  const t = setup({
    decode: () => {
      throw Error("must reject before decoding");
    },
  });
  try {
    await assert.rejects(t.api.process(pcmFile(31)), /up to 30 seconds/);
    assert.equal(t.workers.length, 0);
  } finally {
    t.w.close();
  }
});
test("unknown WAV encodings retain the browser metadata path", async () => {
  const t = setup();
  try {
    const done = t.api.process(pcmFile(9, { format: 17 }));
    await tick();
    await tick();
    t.workers[0].onmessage({ data: { type: "done", files: [] } });
    await done;
    assert.deepEqual(t.revoked, ["blob:metadata"]);
  } finally {
    t.w.close();
  }
});
test("cancellation interrupts the WAV header read and permits a retry", async () => {
  const t = setup();
  try {
    const file = {
      ...t.file,
      slice: () => ({ arrayBuffer: () => new Promise(() => {}) }),
    };
    const done = t.api.process(file),
      rejected = assert.rejects(done, { name: "AbortError" });
    t.api.cancel();
    await rejected;
    assert.equal(t.workers.length, 0);
    const retry = t.api.process(t.file);
    await tick();
    await tick();
    t.workers[0].onmessage({ data: { type: "done", files: [] } });
    await retry;
  } finally {
    t.w.close();
  }
});

test("WAV layouts with trailing chunks use browser metadata rather than a partial duration", async () => {
  const t = setup(),
    file = pcmFile(9),
    slice = file.slice;
  file.size += 8;
  file.slice = (start, end) => ({
    arrayBuffer: async () => {
      const header = await slice(start, end).arrayBuffer(),
        view = new DataView(header);
      view.setUint32(4, view.getUint32(4, true) + 8, true);
      return header;
    },
  });
  try {
    const done = t.api.process(file);
    await tick();
    await tick();
    t.workers[0].onmessage({ data: { type: "done", files: [] } });
    await done;
    assert.deepEqual(t.revoked, ["blob:metadata"]);
  } finally {
    t.w.close();
  }
});
