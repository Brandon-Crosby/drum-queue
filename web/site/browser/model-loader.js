/* Verified model loading with bounded buffers and best-effort browser caching. */
(() => {
  function create({ cacheTimeoutMs = 1500 } = {}) {
    const pending = new Set();
    async function optional(operation) {
      let timer;
      try {
        return await Promise.race([
          Promise.resolve()
            .then(operation)
            .catch(() => undefined),
          new Promise((resolve) => {
            timer = setTimeout(() => resolve(undefined), cacheTimeoutMs);
          }),
        ]);
      } finally {
        clearTimeout(timer);
      }
    }
    function validate(entry) {
      if (
        !entry ||
        !Number.isSafeInteger(entry.bytes) ||
        entry.bytes < 1 ||
        entry.bytes > 512 * 1024 * 1024 ||
        !/^[a-f0-9]{64}$/.test(entry.sha256)
      )
        throw Error("Invalid model manifest");
    }
    async function read(response, entry, onProgress, idleTimeoutMs = 0) {
      const bytes = new Uint8Array(entry.bytes),
        reader = response.body.getReader();
      let offset = 0,
        last = -Infinity;
      try {
        for (;;) {
          let timer;
          let chunk;
          try {
            chunk = idleTimeoutMs
              ? await Promise.race([
                  reader.read(),
                  new Promise((_, reject) => {
                    timer = setTimeout(
                      () => reject(Error("Model cache read stalled")),
                      idleTimeoutMs,
                    );
                  }),
                ])
              : await reader.read();
          } finally {
            clearTimeout(timer);
          }
          const { done, value } = chunk;
          if (done) break;
          if (offset + value.byteLength > bytes.length)
            throw Error("Unexpected model size");
          bytes.set(value, offset);
          offset += value.byteLength;
          const now = performance.now();
          if (now - last >= 200 || offset === bytes.length) {
            onProgress(Math.round((offset / bytes.length) * 100));
            last = now;
          }
        }
        if (offset !== bytes.length) throw Error("Unexpected model size");
        return bytes.buffer;
      } catch (error) {
        void reader.cancel().catch(() => {});
        throw error;
      } finally {
        reader.releaseLock();
      }
    }
    function persist(cache, url, bytes) {
      // A stream avoids Response(ArrayBuffer)'s eager full-buffer copy. Writes
      // overlap inference, and a slow/quota-limited cache cannot block output.
      const data = new Uint8Array(bytes);
      let offset = 0;
      const stream = new ReadableStream({
        pull(controller) {
          if (offset === data.length) {
            controller.close();
            return;
          }
          const end = Math.min(data.length, offset + 1024 * 1024);
          controller.enqueue(data.subarray(offset, end));
          offset = end;
        },
      });
      let task;
      try {
        task = Promise.resolve(
          cache.put(
            url,
            new Response(stream, {
              headers: { "Content-Length": String(data.length) },
            }),
          ),
        ).catch(() => {});
      } catch (_) {
        return;
      }
      pending.add(task);
      task.finally(() => pending.delete(task));
    }
    async function load(entry, url, onProgress = () => {}) {
      validate(entry);
      const cache =
        typeof caches === "undefined"
          ? undefined
          : await optional(() => caches.open("drumsplitter-browser-v1"));
      const cached = cache ? await optional(() => cache.match(url)) : undefined;
      let bytes,
        cacheHit = false;
      if (cached) {
        onProgress({ stage: "download", message: "Reading cached model" });
        // Limit idle time, rather than total time, so a large healthy cache
        // read can finish. Cancel a stalled stream before downloading again.
        try {
          bytes = await read(cached, entry, () => {}, cacheTimeoutMs);
          cacheHit = true;
        } catch (_) {
          /* Fall back to a verified model download. */
        }
      }
      if (!bytes) {
        const response = await fetch(url);
        if (!response.ok) throw Error("Model download failed");
        bytes = await read(response, entry, (percent) =>
          onProgress({
            stage: "download",
            message: "Downloading model · " + percent + "%",
          }),
        );
      }
      onProgress({ stage: "verify", message: "Verifying model" });
      const digest = await crypto.subtle.digest("SHA-256", bytes);
      const hash = [...new Uint8Array(digest)]
        .map((value) => value.toString(16).padStart(2, "0"))
        .join("");
      if (bytes.byteLength !== entry.bytes || hash !== entry.sha256) {
        if (cache) await optional(() => cache.delete(url));
        throw Error("Model verification failed");
      }
      if (cache && !cacheHit) persist(cache, url, bytes);
      return bytes;
    }
    async function flush() {
      if (pending.size) await optional(() => Promise.all([...pending]));
    }
    return { load, flush };
  }
  globalThis.DrumBrowserModelLoader = { create };
})();
