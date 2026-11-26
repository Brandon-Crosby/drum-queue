/* Same float64 transforms as dsp.js, compiled to WebAssembly for worker speed. */
(() => {
  const MAX_SPECTRUM = 16 * 1024 * 256,
    MAX_LENGTH = 343980;
  let initialization;
  function install(native) {
    if (native.version() !== 1)
      throw Error("Unsupported audio transform module");
    const memory = native.memory.buffer;
    const spectral = new Float32Array(
      memory,
      native.spectral_ptr(),
      MAX_SPECTRUM,
    );
    const waves = new Float32Array(memory, native.waves_ptr(), MAX_LENGTH * 8);
    let planned = 0;
    function plan(n) {
      if (planned === n) return;
      const window = new Float64Array(memory, native.window_ptr(), n);
      const cosine = new Float64Array(memory, native.cosine_ptr(), n / 2);
      const sine = new Float64Array(memory, native.sine_ptr(), n / 2);
      const reverse = new Uint32Array(memory, native.reverse_ptr(), n);
      const bits = Math.log2(n);
      for (let i = 0; i < n; i++) {
        let a = i,
          b = 0;
        for (let k = 0; k < bits; k++) {
          b = (b << 1) | (a & 1);
          a >>= 1;
        }
        reverse[i] = b;
        window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
      }
      for (let i = 0; i < n / 2; i++) {
        cosine[i] = Math.cos((2 * Math.PI * i) / n);
        sine[i] = Math.sin((2 * Math.PI * i) / n);
      }
      planned = n;
    }
    function validate(o, channels, stems, length) {
      if (
        ![2048, 4096].includes(o.nfft) ||
        o.hop !== o.nfft / 4 ||
        o.bins !== o.nfft / 2 ||
        !Number.isInteger(o.frames) ||
        o.frames < 1 ||
        o.frames > 336 ||
        !Number.isInteger(length) ||
        length < 2 ||
        length > MAX_LENGTH ||
        !Number.isInteger(channels) ||
        channels < 1 ||
        channels > 2 ||
        !Number.isInteger(stems) ||
        stems < 1 ||
        stems > 4 ||
        ![0, 512].includes(o.offset || 0) ||
        ![0, 2].includes(o.extraFrames || 0) ||
        stems * channels * 2 * o.bins * o.frames > MAX_SPECTRUM
      )
        throw Error("Unsupported audio transform shape");
    }
    DrumDSP.spectrum = (input, o) => {
      const length = input[0]?.length;
      validate(o, input.length, 1, length);
      if (
        !input.every(
          (channel) =>
            channel instanceof Float32Array && channel.length === length,
        )
      )
        throw Error("Expected equal float32 channels");
      input.forEach((channel, c) => waves.set(channel, c * length));
      plan(o.nfft);
      native.spectrum(
        o.nfft,
        o.hop,
        o.frames,
        o.bins,
        o.offset || 0,
        !!o.normalized,
        input.length,
        length,
      );
      return spectral.slice(0, input.length * 2 * o.bins * o.frames);
    };
    DrumDSP.waveform = (input, o) => {
      const channels = o.channels || 2;
      validate(o, channels, o.stems, o.length);
      const count = o.stems * channels * 2 * o.bins * o.frames;
      if (!(input instanceof Float32Array) || input.length !== count)
        throw Error("Unexpected spectral buffer length");
      spectral.set(input);
      plan(o.nfft);
      native.waveform(
        o.nfft,
        o.hop,
        o.frames,
        o.bins,
        o.offset || 0,
        !!o.normalized,
        o.extraFrames || 0,
        o.stems,
        channels,
        o.length,
      );
      return Array.from({ length: o.stems }, (_, s) =>
        Array.from({ length: channels }, (_, c) => {
          const start = (s * channels + c) * o.length;
          return waves.slice(start, start + o.length);
        }),
      );
    };
  }
  async function initialize() {
    if (!initialization)
      initialization = (async () => {
        const response = await fetch("/browser/dsp.wasm");
        if (!response.ok)
          throw Error("Could not load audio transforms. Please retry.");
        const { instance } = await WebAssembly.instantiate(
          await response.arrayBuffer(),
          {},
        );
        install(instance.exports);
      })().catch((error) => {
        initialization = null;
        throw error;
      });
    return initialization;
  }
  globalThis.DrumDSPNative = { install, initialize };
})();
