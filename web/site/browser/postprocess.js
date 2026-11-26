/* Reusable CPU worker. Prepared runtime state survives jobs; audio does not. */
(() => {
  const cancelled = () => new DOMException("Cancelled.", "AbortError");
  function create({
    workerURL = "/browser/postprocess-worker.js",
    idleMs = 300000,
  } = {}) {
    let worker,
      preparation,
      busy = false,
      sequence = 0,
      idle;
    const pending = new Map();
    function dispose(error = cancelled()) {
      clearTimeout(idle);
      worker?.terminate();
      worker = preparation = null;
      for (const request of pending.values()) request.finish(error);
      pending.clear();
      clearTimeout(idle);
    }
    async function request(type, data, { signal, onProgress = () => {} } = {}) {
      clearTimeout(idle);
      if (signal?.aborted) return Promise.reject(cancelled());
      if (!worker) {
        worker = new Worker(workerURL);
        const currentWorker = worker;
        worker.onerror = () => {
          if (worker === currentWorker)
            dispose(Error("Local sample processing stopped."));
        };
        worker.onmessage = ({ data }) => {
          if (worker !== currentWorker) return;
          const current = pending.get(data.id);
          if (!current) return;
          if (data.type === "progress") {
            try {
              current.onProgress(data);
            } catch (error) {
              dispose(error);
            }
          } else if (data.type === "error") dispose(Error(data.message));
          else if (data.type === "ready" || data.type === "done")
            current.finish(null, data);
        };
      }
      const id = ++sequence;
      return new Promise((resolve, reject) => {
        const abort = () => dispose(cancelled());
        const finish = (error, result) => {
          if (!pending.delete(id)) return;
          signal?.removeEventListener("abort", abort);
          if (!pending.size) idle = setTimeout(() => dispose(), idleMs);
          error ? reject(error) : resolve(result);
        };
        pending.set(id, { finish, onProgress });
        signal?.addEventListener("abort", abort, { once: true });
        try {
          // Do not detach the caller's WAVs: they remain available if this job is cancelled.
          worker.postMessage({ type, id, ...data });
        } catch (error) {
          dispose(error);
        }
      });
    }
    function prepare(options = {}) {
      if (options.signal?.aborted) return Promise.reject(cancelled());
      if (!preparation) {
        const current = request("prepare", {}, options);
        preparation = current;
        current.catch(() => {
          if (preparation === current) preparation = null;
        });
        return current;
      }
      if (!options.signal) return preparation;
      const current = preparation,
        signal = options.signal;
      return new Promise((resolve, reject) => {
        let settled = false;
        const finish = (error, result) => {
          if (settled) return;
          settled = true;
          signal.removeEventListener("abort", abort);
          error ? reject(error) : resolve(result);
        };
        const abort = () => {
          const error = cancelled();
          dispose(error);
          finish(error);
        };
        signal.addEventListener("abort", abort, { once: true });
        current.then(
          (result) => finish(null, result),
          (error) => finish(error),
        );
        if (signal.aborted) abort();
      });
    }
    async function process(
      files,
      { name = "audio.wav", allowKitOnly = false, ...options } = {},
    ) {
      if (busy) throw Error("Sample processing is already running.");
      busy = true;
      try {
        await prepare(options);
        if (options.signal?.aborted) throw cancelled();
        return await request("run", { files, name, allowKitOnly }, options);
      } finally {
        busy = false;
      }
    }
    return { prepare, process, dispose };
  }
  globalThis.DrumBrowserPostprocess = { create };
})();
