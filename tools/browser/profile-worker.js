/* Injected only by the local benchmark server; never shipped with the site. */
(() => {
  const stats = {},
    wrapped = new WeakSet();
  function record(name, start) {
    const entry = stats[name] || (stats[name] = { calls: 0, ms: 0 });
    entry.calls++;
    entry.ms += performance.now() - start;
  }
  function wrap(object, key, name, asynchronous = false) {
    const original = object[key];
    if (!original || wrapped.has(original)) return;
    const replacement = asynchronous
      ? async function (...args) {
          const start = performance.now();
          try {
            return await original.apply(this, args);
          } finally {
            record(typeof name === "function" ? name(args) : name, start);
          }
        }
      : function (...args) {
          const start = performance.now();
          try {
            return original.apply(this, args);
          } finally {
            record(name, start);
          }
        };
    wrapped.add(replacement);
    object[key] = replacement;
  }
  let stage = "";
  const originalImport = importScripts.bind(globalThis);
  globalThis.importScripts = (...urls) => {
    originalImport(...urls);
    if (globalThis.DrumDSPNative && !wrapped.has(DrumDSPNative.initialize)) {
      const initialize = DrumDSPNative.initialize;
      const replacement = async (...args) => {
        await initialize(...args);
        wrap(DrumDSP, "spectrum", "dsp:spectrum");
        wrap(DrumDSP, "waveform", "dsp:waveform");
      };
      wrapped.add(replacement);
      DrumDSPNative.initialize = replacement;
    }
    if (
      globalThis.DrumBrowserEngine &&
      !wrapped.has(DrumBrowserEngine.separate)
    ) {
      const separate = DrumBrowserEngine.separate;
      const replacement = async (waves, runner, progress) => {
        const load = runner.load;
        runner.load = async function (name) {
          stage = name;
          return load.call(this, name);
        };
        wrap(runner, "load", (args) => "model-load:" + args[0], true);
        wrap(runner, "run", (args) => "inference:" + args[0], true);
        wrap(runner, "release", () => "release:" + stage, true);
        return separate(waves, runner, progress);
      };
      wrapped.add(replacement);
      DrumBrowserEngine.separate = replacement;
    }
    if (globalThis.ort)
      wrap(
        ort.InferenceSession,
        "create",
        () => "session-create:" + stage,
        true,
      );
  };
  const send = postMessage.bind(globalThis);
  globalThis.postMessage = (message, transfer) => {
    if (message.type === "done" || message.type === "error")
      send({ type: "profile", stats });
    send(message, transfer);
  };
})();
