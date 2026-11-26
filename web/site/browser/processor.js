/* UI-independent entry point for the existing Drum Splitter flow.
 * Returns audio buffers and progress events; it never renders controls or uploads audio.
 */
(() => {
  const MAX_SECONDS = 30,
    MAX_BYTES = 30 * 1024 * 1024,
    RATE = 44100;
  let active = null;
  const cancelled = () => new DOMException("Cancelled.", "AbortError");

  function abortable(promise, signal) {
    return new Promise((resolve, reject) => {
      const abort = () => reject(cancelled());
      signal.addEventListener("abort", abort, { once: true });
      Promise.resolve(promise)
        .then(resolve, reject)
        .finally(() => signal.removeEventListener("abort", abort));
      if (signal.aborted) abort();
    });
  }

  async function wavDuration(file, signal) {
    if (!file.slice) return null;
    // Common PCM/float WAVs carry their duration in a bounded header read.
    // Other containers keep the browser metadata path and its duration guard.
    const header = await abortable(file.slice(0, 65536).arrayBuffer(), signal);
    if (header.byteLength < 12) return null;
    const view = new DataView(header);
    const tag = (at) => String.fromCharCode(...new Uint8Array(header, at, 4));
    if (tag(0) !== "RIFF" || tag(8) !== "WAVE") return null;
    const end = view.getUint32(4, true) + 8;
    if (end > file.size || end < 12) return null;
    let byteRate;
    let align;
    for (let at = 12; at + 8 <= Math.min(end, header.byteLength); ) {
      const size = view.getUint32(at + 4, true),
        start = at + 8;
      if (start + size > end) return null;
      if (tag(at) === "fmt ") {
        if (size < 16 || start + 16 > header.byteLength) return null;
        const format = view.getUint16(start, true),
          channels = view.getUint16(start + 2, true),
          rate = view.getUint32(start + 4, true),
          bits = view.getUint16(start + 14, true);
        byteRate = view.getUint32(start + 8, true);
        align = view.getUint16(start + 12, true);
        if (
          ![1, 3].includes(format) ||
          ![1, 2].includes(channels) ||
          !(format === 1 ? [8, 16, 24, 32] : [32, 64]).includes(bits) ||
          !rate ||
          align !== (channels * bits) / 8 ||
          byteRate !== rate * align
        )
          return null;
      } else if (tag(at) === "data") {
        // Unusual layouts (including multiple data chunks) stay with the
        // browser rather than estimating from only part of the audio.
        if (
          !byteRate ||
          !size ||
          size % align ||
          start + size + (size % 2) !== end
        )
          return null;
        return size / byteRate;
      }
      at = start + size + (size % 2);
    }
    return null;
  }

  async function duration(file, signal) {
    const seconds = await wavDuration(file, signal);
    if (seconds !== null) return seconds;
    return new Promise((resolve, reject) => {
      const audio = document.createElement("audio"),
        url = URL.createObjectURL(file);
      let settled = false;
      const finish = (error) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        signal.removeEventListener("abort", abort);
        const seconds = audio.duration;
        audio.onloadedmetadata = audio.onerror = null;
        audio.removeAttribute("src");
        audio.load();
        URL.revokeObjectURL(url);
        error ? reject(error) : resolve(seconds);
      };
      const abort = () => finish(cancelled());
      const timeout = setTimeout(
        () => finish(Error("Could not read this audio file.")),
        15000,
      );
      signal.addEventListener("abort", abort, { once: true });
      audio.preload = "metadata";
      audio.onloadedmetadata = () => finish();
      audio.onerror = () =>
        finish(Error("This browser cannot read that audio file."));
      if (signal.aborted) abort();
      else audio.src = url;
    });
  }

  async function process(file, { signal, onProgress = () => {} } = {}) {
    if (active) throw Error("A browser split is already running.");
    if (!file) throw Error("Choose an audio file first.");
    if (file.size > MAX_BYTES) throw Error("Choose a file smaller than 30 MB.");
    if (!navigator.gpu || !globalThis.Worker)
      throw Error("WebGPU is unavailable. Cloud processing was not started.");
    const controller = new AbortController(),
      job = { controller };
    active = job;
    const abort = () => controller.abort();
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) abort();
    let context, worker;
    try {
      onProgress({ stage: "read", message: "Reading audio…" });
      const seconds = await duration(file, controller.signal);
      if (!Number.isFinite(seconds) || seconds <= 0 || seconds > MAX_SECONDS)
        throw Error("Browser processing supports clips up to 30 seconds.");
      context = new AudioContext({ sampleRate: RATE });
      const bytes = await abortable(file.arrayBuffer(), controller.signal);
      const decoded = await abortable(
        context.decodeAudioData(bytes),
        controller.signal,
      );
      await context.close();
      context = null;
      if (controller.signal.aborted) throw cancelled();
      if (
        decoded.sampleRate !== RATE ||
        !decoded.length ||
        decoded.length > MAX_SECONDS * RATE
      )
        throw Error("Unsupported audio length or sample rate.");
      if (decoded.numberOfChannels < 1 || decoded.numberOfChannels > 2)
        throw Error("Choose mono or stereo audio.");
      const channels = [0, Math.min(1, decoded.numberOfChannels - 1)].map(
        (c) => new Float32Array(decoded.getChannelData(c)).buffer,
      );
      worker = new Worker("/browser/worker.js");
      const files = await new Promise((resolve, reject) => {
        const abortWorker = () => reject(cancelled());
        controller.signal.addEventListener("abort", abortWorker, {
          once: true,
        });
        const finish = (error, files) => {
          controller.signal.removeEventListener("abort", abortWorker);
          error ? reject(error) : resolve(files);
        };
        worker.onmessage = ({ data }) => {
          if (controller.signal.aborted) return;
          if (data.type === "progress") {
            try {
              onProgress(data);
            } catch (error) {
              finish(error);
            }
          } else if (data.type === "error") finish(Error(data.message));
          else if (data.type === "done") finish(null, data.files);
        };
        worker.onerror = () =>
          finish(
            Error(
              "Browser processing stopped. Cloud processing was not started.",
            ),
          );
        worker.postMessage({ type: "run", channels }, channels);
      });
      return { duration: decoded.length / RATE, files };
    } finally {
      worker?.terminate();
      if (context) await context.close().catch(() => {});
      signal?.removeEventListener("abort", abort);
      if (active === job) active = null;
    }
  }
  globalThis.DrumBrowserProcessor = {
    process,
    cancel: () => active?.controller.abort(),
  };
})();
