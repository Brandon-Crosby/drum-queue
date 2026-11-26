importScripts(
  "/browser/dsp.js",
  "/browser/dsp-native.js",
  "/browser/engine.js",
  "/browser/model-loader.js",
);
const post = (type, data = {}) => postMessage({ type, ...data });
let running = false;
onmessage = async ({ data }) => {
  if (running || data.type !== "run") return;
  running = true;
  let session, device, files, failure;
  try {
    await DrumDSPNative.initialize();
    const waves = data.channels.map((b) => new Float32Array(b));
    DrumBrowserEngine.check(waves);
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter)
      throw Error(
        "WebGPU is unavailable. Try a supported desktop browser; cloud processing was not started.",
      );
    const base = "/browser-assets/v1/",
      response = await fetch(base + "manifest.json");
    if (!response.ok)
      throw Error(
        "Browser models are not configured on this site yet. Cloud processing was not started.",
      );
    const manifest = await response.json();
    if (manifest.schema !== 1 || manifest.runtime !== "onnxruntime-web@1.23.2")
      throw Error("Unsupported browser model manifest");
    importScripts(base + "ort-1.23.2/ort.webgpu.min.js");
    ort.env.wasm.wasmPaths = base + "ort-1.23.2/";
    ort.env.wasm.numThreads = 1;
    const loader = DrumBrowserModelLoader.create();
    const runner = {
      async load(stage) {
        const entry = manifest.models?.[stage];
        if (!entry) throw Error("Invalid model manifest");
        const url = base + stage + "-" + entry.sha256 + ".onnx";
        const bytes = await loader.load(entry, url, (event) =>
          post("progress", {
            ...event,
            message: stage + ": " + event.message,
          }),
        );
        post("progress", {
          stage: "load",
          message: "Preparing " + stage + " on this device",
        });
        session = await ort.InferenceSession.create(bytes, {
          executionProviders: ["webgpu"],
        });
        device = await ort.env.webgpu?.device;
      },
      async run(stage, feeds) {
        const input = Object.fromEntries(
          Object.entries(feeds).map(([name, x]) => [
            name,
            new ort.Tensor("float32", x.data, x.dims),
          ]),
        );
        try {
          const output = await session.run(input);
          return {
            ...output,
            dispose() {
              for (const tensor of Object.values(output)) tensor.dispose();
              for (const tensor of Object.values(input)) tensor.dispose();
            },
          };
        } catch (error) {
          for (const tensor of Object.values(input)) tensor.dispose();
          throw error;
        }
      },
      async release() {
        const current = session;
        session = null;
        if (current) await current.release();
      },
    };
    const stems = await DrumBrowserEngine.separate(
      waves,
      runner,
      (stage, index, total) =>
        post("progress", {
          stage,
          message:
            (stage === "mix" ? "Separating drums" : "Splitting kit") +
            " · chunk " +
            (index + 1) +
            " of " +
            total,
        }),
    );
    files = stems.map((channels, i) => ({
      name: ["kick", "snare", "percussion", "hi-hat"][i] + ".wav",
      bytes: DrumBrowserEngine.wav(channels),
    }));
    await loader.flush();
  } catch (error) {
    failure = error;
  } finally {
    try {
      if (session) await session.release();
    } catch (error) {
      failure ||= error;
    }
    if (device) {
      let timer;
      try {
        // This worker owns its ORT device. Finish queued work and free it before
        // the main thread receives a terminal message and terminates the worker.
        await Promise.race([
          device.queue.onSubmittedWorkDone().catch(() => {}),
          new Promise((resolve) => {
            timer = setTimeout(resolve, 1000);
          }),
        ]);
      } catch (_) {
        // A lost device can reject the fence; exported CPU audio remains valid.
      } finally {
        clearTimeout(timer);
        try {
          device.destroy();
        } catch (error) {
          failure ||= error;
        }
      }
    }
    running = false;
  }
  if (failure) post("error", { message: failure.message || String(failure) });
  else
    postMessage(
      { type: "done", files },
      files.map((file) => file.bytes),
    );
};
