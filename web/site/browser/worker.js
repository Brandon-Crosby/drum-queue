importScripts("/browser/dsp.js", "/browser/engine.js");
const post = (type, data = {}) => postMessage({ type, ...data });
let running = false;
onmessage = async ({ data }) => {
  if (running || data.type !== "run") return;
  running = true;
  let session;
  try {
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
    const runner = {
      async load(stage) {
        const entry = manifest.models[stage];
        if (
          !entry ||
          !Number.isSafeInteger(entry.bytes) ||
          entry.bytes < 1 ||
          entry.bytes > 512 * 1024 * 1024 ||
          !/^[a-f0-9]{64}$/.test(entry.sha256)
        )
          throw Error("Invalid model manifest");
        const url = base + stage + "-" + entry.sha256 + ".onnx",
          cache = await caches.open("drumsplitter-browser-v1");
        let cached = await cache.match(url),
          bytes;
        if (cached) {
          bytes = await cached.arrayBuffer();
          post("progress", {
            stage: "download",
            message: "Reading cached " + stage + " model",
          });
        } else {
          post("progress", {
            stage: "download",
            message: "Downloading " + stage + " model · first use only",
          });
          const r = await fetch(url);
          if (!r.ok) throw Error("Model download failed");
          const reader = r.body.getReader(),
            parts = [];
          let total = 0;
          for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            total += value.length;
            if (total > entry.bytes) throw Error("Unexpected model size");
            parts.push(value);
            post("progress", {
              stage: "download",
              message:
                "Downloading " +
                stage +
                " · " +
                Math.round((total / entry.bytes) * 100) +
                "%",
            });
          }
          bytes = await new Blob(parts).arrayBuffer();
        }
        const hash = [
          ...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
        ]
          .map((v) => v.toString(16).padStart(2, "0"))
          .join("");
        if (bytes.byteLength !== entry.bytes || hash !== entry.sha256) {
          await cache.delete(url);
          throw Error("Model verification failed");
        }
        if (!cached) await cache.put(url, new Response(bytes));
        post("progress", {
          stage: "load",
          message: "Preparing " + stage + " on this device",
        });
        session = await ort.InferenceSession.create(bytes, {
          executionProviders: ["webgpu"],
        });
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
        if (session) {
          await session.release();
          session = null;
        }
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
    const files = stems.map((channels, i) => ({
      name: ["kick", "snare", "percussion", "hi-hat"][i] + ".wav",
      bytes: DrumBrowserEngine.wav(channels),
    }));
    postMessage(
      { type: "done", files },
      files.map((f) => f.bytes),
    );
  } catch (error) {
    post("error", { message: error.message || String(error) });
  } finally {
    if (session) await session.release();
    running = false;
  }
};
