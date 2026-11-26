/* Experimental backend orchestration; the existing website does not select it yet. */
(() => {
  function create({ overlap = true, reuse = true, allowKitOnly = false } = {}) {
    if (allowKitOnly !== true)
      throw Error(
        "Full-feature browser processing is not ready. Enable the four-stem prototype explicitly for testing.",
      );
    const samples = DrumBrowserPostprocess.create();
    let active;
    async function process(file, { signal, onProgress = () => {} } = {}) {
      if (active) throw Error("A browser split is already running.");
      const controller = new AbortController();
      active = controller;
      const abort = () => controller.abort();
      signal?.addEventListener("abort", abort, { once: true });
      if (signal?.aborted) abort();
      let preparation, preparationError;
      const options = { signal: controller.signal, onProgress };
      try {
        if (overlap) {
          preparation = samples.prepare(options).catch((error) => {
            preparationError = error;
            controller.abort();
            throw error;
          });
          preparation.catch(() => {});
        }
        const split = await DrumBrowserProcessor.process(file, options);
        if (preparation) await preparation;
        const output = await samples.process(split.files, {
          ...options,
          name: file.name,
          allowKitOnly,
        });
        return { duration: split.duration, files: split.files, ...output };
      } catch (error) {
        samples.dispose();
        throw preparationError || error;
      } finally {
        if (!reuse) samples.dispose();
        signal?.removeEventListener("abort", abort);
        if (active === controller) active = null;
      }
    }
    return {
      process,
      cancel: () => active?.abort(),
      dispose: () => {
        active?.abort();
        samples.dispose();
      },
    };
  }
  globalThis.DrumBrowserPipeline = { create };
})();
