/* Float64 STFT/iSTFT for the fixed float32 network interfaces.
 * Keep arithmetic order in sync with web/site/browser/dsp.js; no fast-math.
 * Tables are initialized by JS Math.cos/sin, matching the reference implementation.
 */
#define MAX_FFT 4096
#define MAX_LENGTH 343980
#define MAX_SPECTRUM (16 * 1024 * 256)
#define MAX_WAVES (8 * MAX_LENGTH)
#define EXPORT(name) __attribute__((export_name(name)))

static float spectral[MAX_SPECTRUM], waves[MAX_WAVES];
static double window[MAX_FFT], cosine[MAX_FFT / 2], sine[MAX_FFT / 2];
static unsigned reverse[MAX_FFT];
static double real[MAX_FFT], imaginary[MAX_FFT];
static double denominator[MAX_LENGTH], accumulation[MAX_LENGTH];

EXPORT("version") int version(void) { return 1; }
EXPORT("spectral_ptr") float *spectral_ptr(void) { return spectral; }
EXPORT("waves_ptr") float *waves_ptr(void) { return waves; }
EXPORT("window_ptr") double *window_ptr(void) { return window; }
EXPORT("cosine_ptr") double *cosine_ptr(void) { return cosine; }
EXPORT("sine_ptr") double *sine_ptr(void) { return sine; }
EXPORT("reverse_ptr") unsigned *reverse_ptr(void) { return reverse; }

static int reflect(int i, int length) {
    while (i < 0 || i >= length) i = i < 0 ? -i : 2 * length - 2 - i;
    return i;
}

static void fft(int n, int inverse) {
    for (int i = 0; i < n; i++) {
        int j = reverse[i];
        if (j > i) {
            double temp = real[i]; real[i] = real[j]; real[j] = temp;
            temp = imaginary[i]; imaginary[i] = imaginary[j]; imaginary[j] = temp;
        }
    }
    for (int size = 2; size <= n; size *= 2) {
        int half = size / 2, stride = n / size;
        for (int base = 0; base < n; base += size) {
            for (int j = 0; j < half; j++) {
                double c = cosine[j * stride], s = sine[j * stride] * (inverse ? 1 : -1);
                int k = base + j, l = k + half;
                double tr = real[l] * c - imaginary[l] * s;
                double ti = real[l] * s + imaginary[l] * c;
                real[l] = real[k] - tr; imaginary[l] = imaginary[k] - ti;
                real[k] += tr; imaginary[k] += ti;
            }
        }
    }
    if (inverse) for (int i = 0; i < n; i++) { real[i] /= n; imaginary[i] /= n; }
}

EXPORT("spectrum") void spectrum(int n, int hop, int frames, int bins, int offset,
                                  int normalized, int channels, int length) {
    double scale = normalized ? 1 / __builtin_sqrt((double)n) : 1;
    for (int c = 0; c < channels; c++) for (int t = 0; t < frames; t++) {
        for (int k = 0; k < n; k++) {
            int index = reflect(t * hop + k - n / 2 + offset, length);
            real[k] = waves[c * length + index] * window[k]; imaginary[k] = 0;
        }
        fft(n, 0);
        for (int f = 0; f < bins; f++) {
            spectral[(c * 2) * bins * frames + f * frames + t] = real[f] * scale;
            spectral[(c * 2 + 1) * bins * frames + f * frames + t] = imaginary[f] * scale;
        }
    }
}

EXPORT("waveform") void waveform(int n, int hop, int frames, int bins, int offset,
                                  int normalized, int extra_frames, int stems,
                                  int channels, int length) {
    for (int i = 0; i < length; i++) denominator[i] = 0;
    for (int t = -extra_frames; t < frames + extra_frames; t++) {
        for (int k = 0; k < n; k++) {
            int pos = t * hop + k - n / 2 + offset;
            if (pos >= 0 && pos < length) denominator[pos] += window[k] * window[k];
        }
    }
    double scale = normalized ? __builtin_sqrt((double)n) : 1;
    for (int s = 0; s < stems; s++) for (int c = 0; c < channels; c++) {
        for (int i = 0; i < length; i++) accumulation[i] = 0;
        for (int t = 0; t < frames; t++) {
            for (int k = 0; k < n; k++) { real[k] = 0; imaginary[k] = 0; }
            for (int f = 0; f < bins; f++) {
                real[f] = spectral[((s * channels + c) * 2) * bins * frames + f * frames + t];
                imaginary[f] = spectral[((s * channels + c) * 2 + 1) * bins * frames + f * frames + t];
                if (f > 0) { real[n - f] = real[f]; imaginary[n - f] = -imaginary[f]; }
            }
            fft(n, 1);
            for (int k = 0; k < n; k++) {
                int pos = t * hop + k - n / 2 + offset;
                if (pos >= 0 && pos < length) accumulation[pos] += real[k] * window[k] * scale;
            }
        }
        for (int i = 0; i < length; i++) {
            double divisor = denominator[i] > 1e-15 ? denominator[i] : 1e-15;
            waves[(s * channels + c) * length + i] = accumulation[i] / divisor;
        }
    }
}
