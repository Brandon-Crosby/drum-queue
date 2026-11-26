/* Fixed-shape float32 models, bounded clip buffers, sequential GPU calls. */
(() => {
  const RATE = 44100,
    MAX_SECONDS = 30,
    D = 343980,
    D_STEP = Math.floor(D * 0.75),
    H = 130560,
    H_STEP = H / 4;
  function reflect(i, n) {
    if (n < 2) return 0;
    while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
    return i;
  }
  function check(waves) {
    if (
      waves.length !== 2 ||
      !waves.every(
        (w) => w instanceof Float32Array && w.length === waves[0].length,
      )
    )
      throw Error("Expected two equal float32 audio channels");
    if (!waves[0].length || waves[0].length > RATE * MAX_SECONDS)
      throw Error("Browser processing supports clips up to 30 seconds");
    for (const w of waves)
      for (const v of w)
        if (!Number.isFinite(v)) throw Error("Audio contains invalid samples");
  }
  function plans(length, size, step) {
    const out = [];
    for (let start = 0; start < length; start += step)
      out.push({ start, length: Math.min(size, length - start) });
    return out;
  }
  async function separate(waves, runner, progress) {
    check(waves);
    const length = waves[0].length,
      mean =
        waves[0].reduce((s, v, i) => s + (v + waves[1][i]) / 2, 0) / length;
    let variance = 0;
    for (let i = 0; i < length; i++)
      variance += ((waves[0][i] + waves[1][i]) / 2 - mean) ** 2;
    const std = Math.sqrt(variance / Math.max(1, length - 1));
    if (std < 1e-8)
      return Array.from({ length: 4 }, () => [
        new Float32Array(length),
        new Float32Array(length),
      ]);
    const normalized = waves.map((w) =>
        Float32Array.from(w, (v) => (v - mean) / std),
      ),
      drums = [new Float32Array(length), new Float32Array(length)],
      count = new Float32Array(length);
    const ds = plans(length, D, D_STEP),
      dOpts = {
        nfft: 4096,
        hop: 1024,
        frames: 336,
        bins: 2048,
        offset: 512,
        normalized: true,
      };
    await runner.load("demucs");
    try {
      for (let index = 0; index < ds.length; index++) {
        const { start, length: n } = ds[index],
          trim = Math.floor((D - n) / 2),
          chunk = normalized.map((w) =>
            Float32Array.from(
              { length: D },
              (_, i) => w[start - trim + i] || 0,
            ),
          ),
          flat = new Float32Array(D * 2);
        flat.set(chunk[0]);
        flat.set(chunk[1], D);
        const spectrum = DrumDSP.spectrum(chunk, dOpts);
        progress("mix", index, ds.length);
        const out = await runner.run("demucs", {
          wave: { data: flat, dims: [1, 2, D] },
          spectrum: { data: spectrum, dims: [1, 4, 2048, 336] },
        });
        try {
          const audio = DrumDSP.waveform(
            out.spectral.data.subarray(0, 4 * 2048 * 336),
            { ...dOpts, extraFrames: 2, stems: 1, length: D },
          )[0];
          for (let i = 0; i < n; i++) {
            const weight = Math.min(i + 1, D - i) / (D / 2);
            count[start + i] += weight;
            for (let c = 0; c < 2; c++)
              drums[c][start + i] +=
                (audio[c][trim + i] + out.temporal.data[c * D + trim + i]) *
                weight;
          }
        } finally {
          out.dispose();
        }
      }
    } finally {
      await runner.release();
    }
    for (let i = 0; i < length; i++)
      for (let c = 0; c < 2; c++)
        drums[c][i] = (drums[c][i] / count[i]) * std + mean;
    const border = H - H_STEP,
      padded = length > 2 * border,
      workLength = length + (padded ? 2 * border : 0),
      hs = plans(workLength, H, H_STEP),
      result = Array.from({ length: 4 }, () => [
        new Float32Array(length),
        new Float32Array(length),
      ]),
      weights = new Float32Array(length),
      fade = Math.floor(H / 10),
      hOpts = { nfft: 2048, hop: 512, frames: 256, bins: 1024 };
    await runner.load("helyx");
    try {
      for (let index = 0; index < hs.length; index++) {
        const { start, length: n } = hs[index],
          chunk = drums.map((w) =>
            Float32Array.from({ length: H }, (_, i) => {
              if (i >= n && n <= H / 2) return 0;
              const j = i < n ? i : reflect(i, n),
                pos = start + j - (padded ? border : 0);
              return w[padded ? reflect(pos, length) : pos] || 0;
            }),
          );
        progress("kit", index, hs.length);
        const spectrum = DrumDSP.spectrum(chunk, hOpts),
          out = await runner.run("helyx", {
            spectrum: { data: spectrum, dims: [1, 4, 1024, 256] },
          });
        try {
          const audio = DrumDSP.waveform(
            out.stems.data.subarray(0, 4 * 4 * 1024 * 256),
            { ...hOpts, stems: 4, length: H },
          );
          for (let i = 0; i < n; i++) {
            const pos = start + i - (padded ? border : 0);
            if (pos < 0 || pos >= length) continue;
            let weight = 1;
            if (i < fade && start !== 0) weight = i / (fade - 1);
            if (i >= H - fade && start + H_STEP < workLength)
              weight = (H - 1 - i) / (fade - 1);
            weights[pos] += weight;
            for (let s = 0; s < 4; s++)
              for (let c = 0; c < 2; c++)
                result[s][c][pos] += audio[s][c][i] * weight;
          }
        } finally {
          out.dispose();
        }
      }
    } finally {
      await runner.release();
    }
    for (let i = 0; i < length; i++) {
      if (!(weights[i] > 0)) throw Error("Incomplete audio coverage");
      for (let s = 0; s < 4; s++)
        for (let c = 0; c < 2; c++) {
          result[s][c][i] /= weights[i];
          if (!Number.isFinite(result[s][c][i]))
            throw Error("Model produced invalid audio");
        }
    }
    return result;
  }
  function wav(waves) {
    const n = waves[0].length,
      c = waves.length,
      b = new ArrayBuffer(44 + n * c * 3),
      v = new DataView(b);
    const text = (o, s) => {
      for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
    };
    text(0, "RIFF");
    v.setUint32(4, b.byteLength - 8, true);
    text(8, "WAVEfmt ");
    v.setUint32(16, 16, true);
    v.setUint16(20, 1, true);
    v.setUint16(22, c, true);
    v.setUint32(24, RATE, true);
    v.setUint32(28, RATE * c * 3, true);
    v.setUint16(32, c * 3, true);
    v.setUint16(34, 24, true);
    text(36, "data");
    v.setUint32(40, b.byteLength - 44, true);
    for (let i = 0; i < n; i++)
      for (let ch = 0; ch < c; ch++) {
        const sample = Math.max(
            -8388608,
            Math.min(8388607, Math.round(waves[ch][i] * 8388607)),
          ),
          at = 44 + (i * c + ch) * 3;
        v.setUint8(at, sample & 255);
        v.setUint8(at + 1, (sample >> 8) & 255);
        v.setUint8(at + 2, (sample >> 16) & 255);
      }
    return b;
  }
  globalThis.DrumBrowserEngine = {
    RATE,
    MAX_SECONDS,
    check,
    plans,
    separate,
    wav,
  };
})();
