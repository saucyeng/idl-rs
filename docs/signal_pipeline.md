# Signal Pipeline

Mathematical reference for the IDL0 Rust processing layer (`rust/core/src/`).
Each section documents what an operation does physically, which sci-rs / nalgebra /
rustfft call backs it and why, and the input/output units. Tests in the
corresponding `#[cfg(test)]` modules verify the wiring against known inputs.

---

## Welch Spectral Estimation (FFT chart)

The FFT chart estimates the one-sided spectrum with Welch's method, implemented in
[`rust/core/src/fft.rs`](../rust/core/src/fft.rs) `welch()`. sci-rs 0.4 exposes no
spectral-density API, so `welch()` is composed directly on `rustfft`, reusing the
`window_weights` helper that also backs the legacy `fft()` function.

**Why Welch.** A single periodogram is not a consistent estimator: adding samples
buys finer bins, each still ~100 % standard deviation, producing a noisy spectrum.
Welch trades frequency resolution for variance by splitting the record into
overlapping segments, transforming each, and averaging — variance falls roughly as
`1/S` for `S` segments.

### Per-segment pipeline

For each segment `x` of length `L = nperseg`, advanced by `step = L − noverlap`:

1. **Detrend.**
   - `None`: unchanged (bin 0 reflects the segment mean).
   - `Mean`: `x ← x − mean(x)`. Removes the DC offset (gravity + sensor bias) that
     would otherwise dominate bin 0.
   - `Linear`: subtract the least-squares fit `a·i + b`, removing mean **and** slow
     drift. (`a`, `b` from the normal equations; degenerate `L = 1` falls back to
     mean removal.)
2. **Window.** `xw[i] = x[i] · w[i]`, with `w = window_weights(window, L)`
   (Hann / Hamming / rectangular). Suppresses spectral leakage from segment-edge
   discontinuities.
3. **Transform + power.** `P[k] = |rustfft.forward(xw)[k]|²` for one-sided bins
   `k = 0 … L/2`.

### Cross-segment combine

Combine the `S` per-segment powers per bin:

- `Mean`: `P̄[k] = (1/S) Σ_s P_s[k]`.
- `Median`: `P̄[k] = median_s P_s[k]` — robust to transient spikes (impacts, chain
  slap) that would inflate the mean.

### Scaling → output values

- **Magnitude (RMS):** `value[k] = √P̄[k]`. A single full-record segment
  (`S = 1`) with rectangular window and `Detrend::None` gives
  `√(|X[k]|²) = |X[k]|`, i.e. bit-for-bit identical to `fft()`. Intentionally **not**
  window-gain normalised, to preserve that parity.
- **Density (PSD):** `value[k] = P̄[k] / (fs · Σ_i w[i]²)`, with interior bins
  (`0 < k < L/2`) multiplied by 2 for the one-sided fold. Units: input-units² / Hz.
  This is the calibrated, cross-segment-length-comparable scaling.

Bin frequencies: `freqs_hz[k] = k · fs / L`.

### Segmentation, clamping, edge cases

- `nperseg = 0` or `nperseg ≥ len` ⇒ one full-record segment (`L = len`) — the
  legacy single-periodogram case.
- `noverlap` clamped to `[0, L − 1]` so `step ≥ 1`.
- A trailing remainder shorter than `L` is dropped (matches `scipy.signal.welch`).
- Empty input ⇒ empty result (the chart filters empty channels upstream anyway).

### Auto segment length (app-side default)

`ChartSlot.autoFftSegmentLength(n)` (Dart) resolves the segment length when the
user leaves it blank: the largest power of two `≤ n/8`, clamped to `[256, 8192]`,
never exceeding `n`. Power-of-two for FFT speed (rustfft handles any length; pow2 is
just faster). This targets ≥ ~8 averaged segments at 50 % overlap — enough smoothing
to suppress periodogram variance while keeping usable low-frequency resolution for
suspension and tire content.

---

## Short-Time Fourier Transform (STFT)

`welch()` and `spectrogram()` share one `stft()` primitive in
[`rust/core/src/fft.rs`](../rust/core/src/fft.rs). The primitive:

1. **Segment.** Slice the input into overlapping frames of length `L = nperseg`,
   advancing by `step = L − noverlap` each time.
2. **Window.** Multiply each frame by `w = window_weights(window, L)` (Hann /
   Hamming / rectangular).
3. **Detrend.** Remove DC or linear trend per frame (see Welch section above).
4. **Real FFT.** Apply `realfft` (real-to-complex) to each windowed frame.
   `realfft` uses the conjugate-symmetry of a real input to compute only the
   `n/2 + 1` non-redundant bins, dropping the always-zero imaginary input and the
   redundant negative-frequency half — approximately 2× faster and half the memory
   of a full complex FFT.

`stft()` returns the complex frames (`S × (L/2 + 1)`). Callers then differ:

- **`welch()`** squares each bin, averages (mean or median) across the `S` frames,
  and scales → one `L/2 + 1` power spectrum.
- **`spectrogram()`** squares each bin but keeps all `S` frames → a `S × (L/2 + 1)`
  time×frequency power matrix.

Because both paths share the same `stft()` core, peaks in the spectrogram read
identically to the corresponding Welch spectrum — there is no numerical drift
between the FFT chart and the spectrogram chart.

Bin frequencies: `freqs_hz[k] = k · fs / L`, `k = 0 … L/2`.
Frame times: `times_secs[s] = (s · step + L/2) / fs` (frame centre).
