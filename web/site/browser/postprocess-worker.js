/* CPU-only preparation can overlap model loading; each job has a fresh filesystem. */
(() => {
  const base = "/post-assets/v1/";
  let prepared,
    running = false;
  const progress = (id, message) =>
    postMessage({ type: "progress", id, stage: "samples", message });
  async function verified(entry, name, limit) {
    if (
      !entry ||
      !/^[a-f0-9]{64}$/.test(entry.sha256) ||
      !Number.isInteger(entry.bytes) ||
      entry.bytes < 1 ||
      entry.bytes > limit
    )
      throw Error("Invalid local sample asset manifest");
    const response = await fetch(base + name);
    if (!response.ok) throw Error("Local sample assets are not configured.");
    const bytes = new Uint8Array(await response.arrayBuffer());
    const hash = [
      ...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
    ]
      .map((x) => x.toString(16).padStart(2, "0"))
      .join("");
    if (bytes.byteLength !== entry.bytes || hash !== entry.sha256)
      throw Error("Local sample asset verification failed");
    return bytes;
  }
  async function prepare(id) {
    if (!prepared)
      prepared = (async () => {
        progress(id, "Preparing sample tools…");
        const response = await fetch(base + "manifest.json");
        if (!response.ok)
          throw Error("Local sample assets are not configured.");
        const manifest = await response.json();
        if (manifest.schema !== 1 || manifest.runtime !== "pyodide@0.29.0")
          throw Error("Unsupported sample runtime");
        importScripts(base + "runtime/pyodide.js");
        const py = await loadPyodide({ indexURL: base + "runtime/" });
        if (py.version !== "0.29.0") throw Error("Unsupported sample runtime");
        await py.loadPackage(["numpy", "scipy", "soundfile"]);
        if (manifest.bytecode) {
          py.FS.writeFile(
            "/bytecode.zip",
            await verified(manifest.bytecode, "bytecode.zip", 32 * 1024 * 1024),
          );
          await py.runPythonAsync(
            "import zipfile\nwith zipfile.ZipFile('/bytecode.zip') as archive:\n    archive.extractall('/')\n",
          );
          py.FS.unlink("/bytecode.zip");
        }
        py.FS.mkdirTree("/python");
        py.FS.writeFile(
          "/python.zip",
          await verified(manifest.python, "python.zip", 2 * 1024 * 1024),
        );
        py.FS.writeFile(
          "/reference.npz",
          await verified(manifest.reference, "reference.npz", 32 * 1024 * 1024),
        );
        await py.runPythonAsync(`
import sys, zipfile, shutil, json
with zipfile.ZipFile('/python.zip') as source:
    source.extractall('/python')
sys.path.insert(0, '/python')
from postprocess_job import finish
from kitmatch import clean, groove
import scipy.signal, scipy.ndimage, scipy.cluster.hierarchy
`);
        py.FS.unlink("/python.zip");
        return py;
      })().catch((error) => {
        prepared = null;
        throw error;
      });
    return prepared;
  }
  onmessage = async ({ data }) => {
    const { type, id } = data;
    if (!["prepare", "run"].includes(type)) return;
    if (running) {
      postMessage({ type: "error", id, message: "Sample worker is busy." });
      return;
    }
    running = true;
    let py, completed, failure;
    try {
      py = await prepare(id);
      if (type === "prepare") {
        progress(id, "Sample tools ready.");
        postMessage({ type: "ready", id });
        return;
      }
      const aliases = { "percussion.wav": "perc.wav", "hi-hat.wav": "hh.wav" };
      const allowed = [
        "kick.wav",
        "snare.wav",
        "perc.wav",
        "hh.wav",
        "drums.wav",
        "bass.wav",
        "vocals.wav",
        "other.wav",
      ];
      if (
        !Array.isArray(data.files) ||
        data.files.length > allowed.length ||
        typeof data.name !== "string" ||
        data.name.length > 200
      )
        throw Error("Invalid sample job");
      const seen = new Set();
      py.FS.mkdirTree("/job/input");
      for (const file of data.files) {
        const name = aliases[file.name] || file.name;
        if (
          !allowed.includes(name) ||
          seen.has(name) ||
          !(file.bytes instanceof ArrayBuffer) ||
          !file.bytes.byteLength ||
          file.bytes.byteLength > 30 * 44100 * 8 + 4096
        )
          throw Error("Invalid sample stem");
        seen.add(name);
        py.FS.writeFile("/job/input/" + name, new Uint8Array(file.bytes));
      }
      if (
        !allowed
          .slice(0, data.allowKitOnly === true ? 4 : 8)
          .every((name) => seen.has(name))
      )
        throw Error("The required stem set is incomplete");
      py.globals.set("job_name", data.name);
      py.globals.set("full_mix", data.allowKitOnly !== true);
      progress(id, "Preparing samples, loops and MIDI…");
      const encoded = await py.runPythonAsync(`
result = finish(job_name, '/job/input', '/job/output', '/reference.npz', require_full_mix=full_mix)
json.dumps(result)
`);
      const result = JSON.parse(encoded),
        artifacts = [];
      for (const artifact of result.artifacts) {
        if (!artifact.path.startsWith("/job/"))
          throw Error("Unexpected sample artifact path");
        const bytes = py.FS.readFile(artifact.path);
        artifacts.push({
          kind: artifact.kind,
          name: artifact.name,
          bytes: bytes.buffer.slice(
            bytes.byteOffset,
            bytes.byteOffset + bytes.byteLength,
          ),
        });
      }
      delete result.artifacts;
      completed = { type: "done", id, result, artifacts };
    } catch (error) {
      failure = error;
    } finally {
      if (py && type === "run") {
        try {
          await py.runPythonAsync(
            "shutil.rmtree('/job', ignore_errors=True)\n",
          );
        } catch (error) {
          prepared = null;
          failure ||= error;
        }
      }
      running = false;
    }
    if (failure)
      postMessage({
        type: "error",
        id,
        message: String(failure.message || failure),
      });
    else if (completed)
      postMessage(
        completed,
        completed.artifacts.map((x) => x.bytes),
      );
  };
})();
