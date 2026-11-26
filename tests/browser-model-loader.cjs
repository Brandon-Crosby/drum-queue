const { test } = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const vm = require("node:vm");
const { webcrypto } = require("node:crypto");
const source = readFileSync("web/site/browser/model-loader.js", "utf8");
async function fixture({ cache, response, cacheApi } = {}) {
  const data = Uint8Array.of(1, 2, 3);
  const entry = {
    bytes: 3,
    sha256: Buffer.from(
      await webcrypto.subtle.digest("SHA-256", data),
    ).toString("hex"),
  };
  let downloads = 0;
  const context = vm.createContext({
    Uint8Array,
    Response,
    ReadableStream,
    performance,
    setTimeout,
    clearTimeout,
    crypto: webcrypto,
    caches: cacheApi || { open: async () => cache },
    fetch: async () => {
      downloads++;
      return response ? response() : new Response(data);
    },
  });
  vm.runInContext(source, context);
  return {
    loader: context.DrumBrowserModelLoader.create({ cacheTimeoutMs: 20 }),
    entry,
    downloads: () => downloads,
  };
}
test("a stalled cache write cannot block verified model loading or completion", async () => {
  let puts = 0;
  const f = await fixture({
    cache: {
      match: async () => undefined,
      put: () => {
        puts++;
        return new Promise(() => {});
      },
    },
  });
  assert.deepEqual(
    [...new Uint8Array(await f.loader.load(f.entry, "/model"))],
    [1, 2, 3],
  );
  await f.loader.flush();
  assert.equal(puts, 1);
});
test("cache access and quota errors remain optional", async () => {
  for (const cacheApi of [
    {
      open() {
        throw Error("Denied");
      },
    },
    {
      open: async () => ({
        match() {
          throw Error("Denied");
        },
        put() {
          throw Error("Quota");
        },
      }),
    },
  ]) {
    const f = await fixture({ cacheApi });
    assert.equal((await f.loader.load(f.entry, "/model")).byteLength, 3);
    await f.loader.flush();
  }
});
test("a stalled cache stream is canceled and replaced with a verified download", async () => {
  let canceled = false,
    saved;
  const f = await fixture({
    cache: {
      match: async () =>
        new Response(
          new ReadableStream({
            pull() {
              return new Promise(() => {});
            },
            cancel() {
              canceled = true;
            },
          }),
        ),
      put: async (_, response) => {
        saved = await response.arrayBuffer();
      },
    },
  });
  await f.loader.load(f.entry, "/model");
  await f.loader.flush();
  assert.equal(canceled, true);
  assert.equal(f.downloads(), 1);
  assert.deepEqual([...new Uint8Array(saved)], [1, 2, 3]);
});
test("cache read deadline applies to idle time rather than total transfer time", async () => {
  let offset = 0;
  const f = await fixture({
    cache: {
      match: async () =>
        new Response(
          new ReadableStream({
            async pull(controller) {
              await new Promise((resolve) => setTimeout(resolve, 10));
              if (offset === 3) controller.close();
              else controller.enqueue(Uint8Array.of(++offset));
            },
          }),
        ),
    },
  });
  assert.equal((await f.loader.load(f.entry, "/model")).byteLength, 3);
  assert.equal(f.downloads(), 0);
});
test("oversized and truncated downloads fail without persisting data", async () => {
  for (const bytes of [Uint8Array.of(1, 2), Uint8Array.of(1, 2, 3, 4)]) {
    let puts = 0;
    const f = await fixture({
      response: () => new Response(bytes),
      cache: { match: async () => undefined, put: async () => puts++ },
    });
    await assert.rejects(
      f.loader.load(f.entry, "/model"),
      /Unexpected model size/,
    );
    assert.equal(puts, 0);
  }
});
test("invalid hashes and manifest entries are rejected before caching", async () => {
  let puts = 0;
  const f = await fixture({
    response: () => new Response(Uint8Array.of(9, 9, 9)),
    cache: {
      match: async () => undefined,
      delete: async () => {},
      put: async () => puts++,
    },
  });
  await assert.rejects(f.loader.load(f.entry, "/model"), /verification failed/);
  for (const entry of [
    undefined,
    { bytes: -1, sha256: f.entry.sha256 },
    { ...f.entry, sha256: "../model" },
  ])
    await assert.rejects(
      f.loader.load(entry, "/model"),
      /Invalid model manifest/,
    );
  assert.equal(puts, 0);
});
