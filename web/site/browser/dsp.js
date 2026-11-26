/* CPU audio transforms around unchanged exported neural networks. */
(() => {
  const plans = new Map();
  function plan(n) {
    if (plans.has(n)) return plans.get(n);
    const bits = Math.log2(n),
      rev = new Uint32Array(n),
      cos = new Float64Array(n / 2),
      sin = new Float64Array(n / 2);
    for (let i = 0; i < n; i++) {
      let a = i,
        b = 0;
      for (let k = 0; k < bits; k++) {
        b = (b << 1) | (a & 1);
        a >>= 1;
      }
      rev[i] = b;
    }
    for (let i = 0; i < n / 2; i++) {
      cos[i] = Math.cos((2 * Math.PI * i) / n);
      sin[i] = Math.sin((2 * Math.PI * i) / n);
    }
    const p = {
      rev,
      cos,
      sin,
      window: Float64Array.from(
        { length: n },
        (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n),
      ),
    };
    plans.set(n, p);
    return p;
  }
  function fft(re, im, inverse = false) {
    const n = re.length,
      p = plan(n);
    for (let i = 0; i < n; i++) {
      const j = p.rev[i];
      if (j > i) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    for (let len = 2; len <= n; len *= 2) {
      const half = len / 2,
        stride = n / len;
      for (let base = 0; base < n; base += len)
        for (let j = 0; j < half; j++) {
          const c = p.cos[j * stride],
            s = p.sin[j * stride] * (inverse ? 1 : -1),
            k = base + j,
            l = k + half,
            tr = re[l] * c - im[l] * s,
            ti = re[l] * s + im[l] * c;
          re[l] = re[k] - tr;
          im[l] = im[k] - ti;
          re[k] += tr;
          im[k] += ti;
        }
    }
    if (inverse)
      for (let i = 0; i < n; i++) {
        re[i] /= n;
        im[i] /= n;
      }
  }
  function reflect(i, n) {
    while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
    return i;
  }
  function spectrum(
    waves,
    { nfft, hop, frames, bins, offset = 0, normalized = false },
  ) {
    const out = new Float32Array(waves.length * 2 * bins * frames),
      p = plan(nfft);
    for (let c = 0; c < waves.length; c++)
      for (let t = 0; t < frames; t++) {
        const re = new Float64Array(nfft),
          im = new Float64Array(nfft);
        for (let k = 0; k < nfft; k++)
          re[k] =
            waves[c][
              reflect(t * hop + k - nfft / 2 + offset, waves[c].length)
            ] * p.window[k];
        fft(re, im);
        const scale = normalized ? 1 / Math.sqrt(nfft) : 1;
        for (let f = 0; f < bins; f++) {
          out[c * 2 * bins * frames + f * frames + t] = re[f] * scale;
          out[(c * 2 + 1) * bins * frames + f * frames + t] = im[f] * scale;
        }
      }
    return out;
  }
  function waveform(
    data,
    {
      nfft,
      hop,
      frames,
      bins,
      offset = 0,
      normalized = false,
      extraFrames = 0,
      stems,
      channels = 2,
      length,
    },
  ) {
    const outputs = [],
      p = plan(nfft),
      denom = new Float64Array(length);
    for (let t = -extraFrames; t < frames + extraFrames; t++)
      for (let k = 0; k < nfft; k++) {
        const pos = t * hop + k - nfft / 2 + offset;
        if (pos >= 0 && pos < length) denom[pos] += p.window[k] ** 2;
      }
    for (let s = 0; s < stems; s++) {
      const waves = [];
      for (let c = 0; c < channels; c++) {
        const wave = new Float64Array(length);
        for (let t = 0; t < frames; t++) {
          const re = new Float64Array(nfft),
            im = new Float64Array(nfft);
          for (let f = 0; f < bins; f++) {
            re[f] =
              data[(s * channels + c) * 2 * bins * frames + f * frames + t];
            im[f] =
              data[
                ((s * channels + c) * 2 + 1) * bins * frames + f * frames + t
              ];
            if (f > 0) {
              re[nfft - f] = re[f];
              im[nfft - f] = -im[f];
            }
          }
          fft(re, im, true);
          const scale = normalized ? Math.sqrt(nfft) : 1;
          for (let k = 0; k < nfft; k++) {
            const pos = t * hop + k - nfft / 2 + offset;
            if (pos >= 0 && pos < length)
              wave[pos] += re[k] * p.window[k] * scale;
          }
        }
        waves.push(
          Float32Array.from(wave, (v, i) => v / Math.max(denom[i], 1e-15)),
        );
      }
      outputs.push(waves);
    }
    return outputs;
  }
  globalThis.DrumDSP = { spectrum, waveform };
})();
