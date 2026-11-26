(() => {
  "use strict";
  const $ = (id) => document.getElementById(id),
    input = $("clip"),
    run = $("run"),
    cancel = $("cancel"),
    status = $("status"),
    progress = $("progress"),
    results = $("results"),
    stems = $("stems");
  let current = null,
    urls = [];
  function clear() {
    for (const a of stems.querySelectorAll("audio")) {
      a.pause();
      a.removeAttribute("src");
      a.load();
    }
    stems.replaceChildren();
    urls.forEach((url) => URL.revokeObjectURL(url));
    urls = [];
    results.hidden = true;
  }
  function finish(job, message) {
    if (current !== job) return;
    job.worker?.terminate();
    current = null;
    run.disabled = false;
    input.disabled = false;
    cancel.hidden = true;
    progress.hidden = true;
    status.textContent = message;
  }
  function metadata(file, signal) {
    return new Promise((resolve, reject) => {
      const audio = document.createElement("audio"),
        url = URL.createObjectURL(file);
      let settled = false;
      const end = (error) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        signal.removeEventListener("abort", abort);
        const duration = audio.duration;
        audio.onloadedmetadata = audio.onerror = null;
        audio.removeAttribute("src");
        audio.load();
        URL.revokeObjectURL(url);
        error ? reject(error) : resolve(duration);
      };
      const abort = () => end(new Error("Cancelled."));
      const timeout = setTimeout(
        () =>
          end(new Error("Could not read this clip. Try another audio file.")),
        15000,
      );
      signal.addEventListener("abort", abort, { once: true });
      audio.preload = "metadata";
      audio.onloadedmetadata = () => end();
      audio.onerror = () =>
        end(new Error("This browser cannot read that audio file."));
      audio.src = url;
    });
  }
  async function start() {
    if (current) return;
    const file = input.files[0];
    if (!file) {
      status.textContent = "Choose an audio clip first.";
      return;
    }
    if (file.size > 30 * 1024 * 1024) {
      status.textContent = "Choose a file smaller than 30 MB.";
      return;
    }
    if (!navigator.gpu || !window.Worker) {
      status.textContent =
        "WebGPU is unavailable in this browser. Cloud processing was not started.";
      return;
    }
    const job = { abort: new AbortController() };
    current = job;
    run.disabled = input.disabled = true;
    cancel.hidden = false;
    progress.hidden = false;
    clear();
    status.textContent = "Reading your clip…";
    try {
      const duration = await metadata(file, job.abort.signal);
      if (current !== job) return;
      if (!Number.isFinite(duration) || duration <= 0 || duration > 30)
        throw Error("Choose a clip up to 30 seconds.");
      const context = new AudioContext({ sampleRate: 44100 });
      job.context = context;
      let decoded;
      try {
        decoded = await context.decodeAudioData(await file.arrayBuffer());
      } finally {
        await context.close();
        job.context = null;
      }
      if (current !== job) return;
      if (
        decoded.sampleRate !== 44100 ||
        decoded.length > 30 * 44100 ||
        decoded.length === 0
      )
        throw Error(
          "Choose a clip up to 30 seconds at a supported sample rate.",
        );
      if (decoded.numberOfChannels > 2)
        throw Error("Choose mono or stereo audio.");
      const channels = [0, Math.min(1, decoded.numberOfChannels - 1)].map(
        (c) => new Float32Array(decoded.getChannelData(c)).buffer,
      );
      job.worker = new Worker("/browser/worker.js");
      job.worker.onmessage = ({ data }) => {
        if (current !== job) return;
        if (data.type === "progress") status.textContent = data.message;
        else if (data.type === "error") finish(job, data.message);
        else if (data.type === "done") {
          for (const file of data.files) {
            const url = URL.createObjectURL(
              new Blob([file.bytes], { type: "audio/wav" }),
            );
            urls.push(url);
            const row = document.createElement("article"),
              heading = document.createElement("h3"),
              audio = document.createElement("audio"),
              link = document.createElement("a");
            heading.textContent = file.name.replace(".wav", "");
            audio.controls = true;
            audio.preload = "metadata";
            audio.src = url;
            audio.addEventListener("play", () => {
              for (const other of stems.querySelectorAll("audio"))
                if (other !== audio) other.pause();
            });
            link.href = url;
            link.download = file.name;
            link.textContent = "Download " + file.name;
            row.append(heading, audio, link);
            stems.append(row);
          }
          results.hidden = false;
          finish(job, "Done. Listen to or download your stems.");
        }
      };
      job.worker.onerror = () =>
        finish(
          job,
          "Browser processing stopped unexpectedly. Cloud processing was not started.",
        );
      job.worker.postMessage({ type: "run", channels }, channels);
    } catch (error) {
      finish(job, error.message || "Could not process this clip.");
    }
  }
  run.addEventListener("click", start);
  cancel.addEventListener("click", () => {
    const job = current;
    if (!job) return;
    job.abort.abort();
    finish(job, "Cancelled. Cloud processing was not started.");
  });
  $("stop").addEventListener("click", () => {
    for (const audio of stems.querySelectorAll("audio")) {
      audio.pause();
      audio.currentTime = 0;
    }
  });
  window.addEventListener("pagehide", () => {
    const job = current;
    if (job) {
      job.abort.abort();
      finish(job, "Cancelled. Choose a clip to start again.");
    }
    clear();
  });
})();
