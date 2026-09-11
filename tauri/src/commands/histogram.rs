//! Value-distribution histogram command (C3 §3.6, ruling R215 item 2):
//! `fetch_histogram` over L3's [`idl_rs::histogram`] binner.
//!
//! **Why this is a command and not sandbox code.** The histogram chart is one
//! channel's value distribution over a selected window (C1 §6.1). Binning it
//! means an O(n) pass over every sample in that window — the whole point of
//! keeping samples in the engine (CLAUDE.md §2: physics of the bike → `core`;
//! design §4's "heavy arrays never cross IPC"). The result is small (one
//! `f64` edge per bin plus one count per bin, a few hundred numbers), so
//! unlike `fetch_tile`/`fetch_fft` this one returns **JSON**, not
//! `tauri::ipc::Response` bytes: there is no array here big enough for a
//! binary encoder to earn its own decoder on the app side.
//!
//! Same idiom as `commands/tiles.rs`/`commands/rasters.rs`: the
//! `#[tauri::command]` is a one-line wrapper over a `_via`-suffixed plain
//! function taking `data_dir: &Path` — this module's own tests exercise
//! `fetch_histogram_via` directly (`tauri::State` cannot be constructed
//! outside a running app).

use std::path::Path;

use crate::error::{IpcError, IpcErrorKind};
use crate::session_cache::SessionCache;
use crate::session_source::{resolve_window, session_dir, WindowDto};
use crate::state::DataDir;

/// Largest `bins` a caller may request, and the cap `bin_mode: "width"`
/// resolves against (C3 §3.6, ruling R215 item 2). A histogram is drawn one
/// bar per bin into a chart a few hundred CSS pixels wide; 4096 is already
/// far past the point where a bar is sub-pixel, and it bounds the response
/// at ~64 kB of JSON. Shares its value with `commands/tiles.rs`'
/// `MAX_COLUMN_COUNT` by coincidence of reasoning, not by dependency — the
/// two caps answer different questions and are free to diverge.
pub const MAX_HISTOGRAM_BINS: u32 = 4096;

/// `HistogramParams.bin_mode` (C3 §3.6): how `bin_value` is to be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinModeToken {
    /// `bin_value` is the number of equal-width bins, an integer.
    Count,
    /// `bin_value` is the width of one bin, in the channel's own unit; the
    /// engine derives the count that covers the resolved range
    /// ([`idl_rs::histogram::bins_for_width`]).
    Width,
}

/// `HistogramParams.normalise` (C3 §3.6): what `values` carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormaliseToken {
    /// `values[i] == counts[i]`, the raw finite-sample count in bin `i`.
    Counts,
    /// `values[i] == counts[i] / total`, in `0..=1` — the share of the
    /// window's finite samples that fell in bin `i`. All-zero when `total`
    /// is zero (no finite sample), never `NaN`.
    Fraction,
}

/// `fetch_histogram`'s `params` argument (C3 §3.6). Every field is required:
/// a histogram's picture is fully determined by these four values plus the
/// window, and a default living in the engine would be a parameter of the
/// picture the document does not state (CLAUDE.md §3, the same rule C2
/// §5.3's `fft_params` follows).
#[derive(Debug, Clone, Copy, serde::Deserialize)]
pub struct HistogramParams {
    pub bin_mode: BinModeToken,
    /// A bin count (`bin_mode: "count"`) or a bin width in the channel's own
    /// unit (`bin_mode: "width"`). Carried as `f64` for both so one wire
    /// field serves both modes; the `count` mode rejects a non-integer or
    /// out-of-range value rather than rounding it.
    pub bin_value: f64,
    /// Widen the auto range to `[-m, m]` so zero sits on a bin boundary —
    /// the natural frame for a signed suspension-velocity distribution
    /// (compression vs rebound).
    pub symmetric: bool,
    pub normalise: NormaliseToken,
}

/// `fetch_histogram`'s JSON result (C3 §3.6).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct HistogramResponse {
    /// Bin boundaries, ascending, length `counts.len() + 1`. Bin `i` spans
    /// `bin_edges[i]..bin_edges[i + 1]`; the last bin is closed on the right
    /// so the window's maximum sample lands in it. Empty for a degenerate
    /// result (see [`fetch_histogram_via`]).
    pub bin_edges: Vec<f64>,
    /// Finite-sample count per bin. Sums to `total`.
    pub counts: Vec<u32>,
    /// What the chart plots, per `params.normalise` — `counts[i]` as `f64`
    /// under `"counts"`, `counts[i] / total` under `"fraction"`. Computed
    /// here rather than app-side so no number the picture depends on is
    /// derived in JavaScript (CLAUDE.md §2).
    pub values: Vec<f64>,
    /// Total finite samples binned over the window. `0` for a degenerate
    /// result.
    pub total: u32,
    /// The bin count actually used. Equals `params.bin_value` under
    /// `bin_mode: "count"`; under `"width"` it is what the engine derived,
    /// which the caller cannot compute for itself without the samples — so
    /// it is reported rather than left implicit in `counts.len()`.
    pub bins: u32,
}

impl HistogramResponse {
    /// The degenerate result: no bins, no samples. Returned for a window
    /// containing no finite sample, a constant channel (zero-width range),
    /// or a `bin_mode: "width"` value wider than any resolvable range — the
    /// chart shows an empty state rather than an error, exactly as
    /// [`idl_rs::histogram::HistogramResult::empty`] intends.
    fn empty() -> Self {
        HistogramResponse { bin_edges: Vec::new(), counts: Vec::new(), values: Vec::new(), total: 0, bins: 0 }
    }
}

/// Validates `params.bin_value` for `bin_mode: "count"` and returns it as a
/// bin count. A non-integer, non-finite, zero, negative, or over-cap value is
/// `invalid_argument` with `detail: { "bin_value": v, "max_bins": n }` — never
/// rounded or clamped, since a silently-changed bin count changes the picture
/// without changing the document that states it.
fn bin_count_from_value(bin_value: f64) -> Result<usize, IpcError> {
    let invalid = || {
        IpcError::with_detail(
            IpcErrorKind::InvalidArgument,
            format!("bin_mode \"count\" requires an integer bin_value in 1..={MAX_HISTOGRAM_BINS}, got {bin_value}"),
            serde_json::json!({ "bin_value": bin_value, "max_bins": MAX_HISTOGRAM_BINS }),
        )
    };
    if !bin_value.is_finite() || bin_value.fract() != 0.0 {
        return Err(invalid());
    }
    if bin_value < 1.0 || bin_value > f64::from(MAX_HISTOGRAM_BINS) {
        return Err(invalid());
    }
    Ok(bin_value as usize)
}

/// Transport-agnostic core of `fetch_histogram` (C3 §3.6, ruling R215 item 2).
///
/// Resolution order mirrors [`crate::commands::rasters::fetch_fft_v2_via`]
/// exactly, and for the same reason (ruling R85): resolve the window first
/// — surfacing an unknown `lap` or a degenerate `range` before any sample is
/// read — then slice the channel to it, then bin the *sliced* window, never
/// the whole channel. A histogram consumes the time axis, so it is an
/// aggregation over the window (ruling R123's reasoning for the FFT applies
/// unchanged here).
///
/// The channel comes through the app's byte-budgeted [`SessionCache`] — one
/// column, not the whole `data.parquet` (ruling R203.1/R211) — and is decoded
/// at most once while the cache holds it.
///
/// `not_found`: unknown `session_id` or `channel`. `invalid_argument`: an
/// unknown `lap` number, a `range` failing R119/R120, or a `bin_value` that
/// fails [`bin_count_from_value`] under `bin_mode: "count"`. A `bin_mode:
/// "width"` value that resolves to no bins is **not** an error — it is the
/// degenerate result ([`HistogramResponse::empty`]), the same answer a
/// constant channel gets, because the width itself is legal and it is the
/// data that has nothing to bin.
pub fn fetch_histogram_via(
    cache: &SessionCache,
    data_dir: &Path,
    window: &WindowDto,
    channel: &str,
    params: &HistogramParams,
) -> Result<HistogramResponse, IpcError> {
    let ch = cache.channel(&session_dir(data_dir, &window.session_id), &window.session_id, channel)?;
    let (t0_secs, t1_secs) = resolve_window(data_dir, window)?;
    let samples = ch.slice_by_time(t0_secs, t1_secs);

    let bins = match params.bin_mode {
        BinModeToken::Count => bin_count_from_value(params.bin_value)?,
        BinModeToken::Width => idl_rs::histogram::bins_for_width(
            &samples,
            params.bin_value,
            params.symmetric,
            MAX_HISTOGRAM_BINS as usize,
        ),
    };
    if bins == 0 {
        return Ok(HistogramResponse::empty());
    }

    let h = idl_rs::histogram::histogram(&samples, bins, params.symmetric, None);
    if h.counts.is_empty() {
        return Ok(HistogramResponse::empty());
    }

    // `total == 0` cannot happen alongside a non-empty `counts` today (a
    // range is only resolvable from at least one finite sample), but the
    // guard is here rather than a `counts[i] as f64 / total as f64` that
    // would produce `NaN` if it ever did: a chart must never be handed a
    // `NaN` it would draw as a gap in real data.
    let values: Vec<f64> = match params.normalise {
        NormaliseToken::Counts => h.counts.iter().map(|&c| f64::from(c)).collect(),
        NormaliseToken::Fraction if h.total == 0 => vec![0.0; h.counts.len()],
        NormaliseToken::Fraction => h.counts.iter().map(|&c| f64::from(c) / f64::from(h.total)).collect(),
    };

    let bins_used = h.counts.len() as u32;
    Ok(HistogramResponse { bin_edges: h.bin_edges, counts: h.counts, values, total: h.total, bins: bins_used })
}

/// Bins one channel's values over `window` into a value-distribution
/// histogram (C3 §3.6, ruling R215 item 2). Small JSON, not binary bytes —
/// see this module's own doc comment. Settle-bound only (C3 §4): never a
/// hover/pan/zoom handler.
#[tauri::command(async)]
pub fn fetch_histogram(
    window: WindowDto,
    channel: String,
    params: HistogramParams,
    data_dir: tauri::State<'_, DataDir>,
    cache: tauri::State<'_, SessionCache>,
) -> Result<HistogramResponse, IpcError> {
    fetch_histogram_via(&cache, &data_dir.0, &window, &channel, &params)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::session_source::SpanDto;
    use idl_rs::session::{Channel, RawColumn, Session, SourceFormat, TimestampSource};
    use idl_rs::store::parquet::write_session_parquet;
    use uuid::Uuid;

    fn temp_root() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-tauri-histogram-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Seeds session `"s1"` with a 1 kHz channel `"Velocity"` whose sample
    /// `i` is `i as f64` over 1000 samples (values 0..999, `t_us` `i * 1000`).
    fn seed_session(root: &Path) {
        let n = 1000usize;
        let session = Session {
            session_id: "s1".to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            timestamp_source: TimestampSource::SourceFile,
            config_checksum: None,
            source_format: SourceFormat::Fit,
            blob_sha256: "a".repeat(64),
            channels: vec![Channel {
                channel_id: "Velocity".to_string(),
                t_us: (0..n).map(|i| i as i64 * 1000).collect(),
                t_recorded_us: None,
                nominal_rate_hz: 1000.0,
                column: RawColumn::F64((0..n).map(|i| i as f64).collect()),
                source_kind: "imu".to_string(),
                unit: "m/s".to_string(),
                gaps: Vec::new(),
            }],
        };
        write_session_parquet(root, &session, "0.1.0").unwrap();
    }

    fn session_window() -> WindowDto {
        WindowDto { session_id: "s1".to_string(), span: SpanDto::Session, colour: "--chart-1".to_string() }
    }

    fn count_params(bins: f64) -> HistogramParams {
        HistogramParams { bin_mode: BinModeToken::Count, bin_value: bins, symmetric: false, normalise: NormaliseToken::Counts }
    }

    #[test]
    fn fetch_histogram_via_count_mode_conserves_every_finite_sample_in_the_window() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let h = fetch_histogram_via(&SessionCache::new(), &root, &session_window(), "Velocity", &count_params(10.0)).unwrap();

        // Assert — 10 bins, 11 edges, every one of the 1000 samples binned.
        assert_eq!(h.bins, 10);
        assert_eq!(h.counts.len(), 10);
        assert_eq!(h.bin_edges.len(), 11);
        assert_eq!(h.total, 1000);
        assert_eq!(h.counts.iter().sum::<u32>(), 1000);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_histogram_via_normalise_fraction_values_sum_to_one_and_counts_are_still_raw() {
        // Arrange
        let root = temp_root();
        seed_session(&root);
        let params = HistogramParams { normalise: NormaliseToken::Fraction, ..count_params(10.0) };

        // Act
        let h = fetch_histogram_via(&SessionCache::new(), &root, &session_window(), "Velocity", &params).unwrap();

        // Assert
        let sum: f64 = h.values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "fractions sum to {sum}, not 1");
        assert_eq!(h.counts.iter().sum::<u32>(), 1000, "counts stay raw regardless of normalise");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_histogram_via_normalise_counts_values_equal_counts() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let h = fetch_histogram_via(&SessionCache::new(), &root, &session_window(), "Velocity", &count_params(4.0)).unwrap();

        // Assert
        let as_f64: Vec<f64> = h.counts.iter().map(|&c| f64::from(c)).collect();
        assert_eq!(h.values, as_f64);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_histogram_via_width_mode_derives_the_bin_count_and_reports_it() {
        // Arrange — values 0..999, so the auto range spans 999.
        let root = temp_root();
        seed_session(&root);
        let params = HistogramParams { bin_mode: BinModeToken::Width, bin_value: 100.0, ..count_params(0.0) };

        // Act
        let h = fetch_histogram_via(&SessionCache::new(), &root, &session_window(), "Velocity", &params).unwrap();

        // Assert — ceil(999 / 100) == 10.
        assert_eq!(h.bins, 10);
        assert_eq!(h.counts.len(), 10);
        assert_eq!(h.total, 1000);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_histogram_via_range_window_bins_only_that_window_not_the_whole_channel() {
        // Arrange — the first 100 ms of a 1 kHz channel is 100 samples.
        let root = temp_root();
        seed_session(&root);
        let window = WindowDto {
            session_id: "s1".to_string(),
            span: SpanDto::Range { t0_us: 0, t1_us: 100_000 },
            colour: "--chart-1".to_string(),
        };

        // Act
        let whole = fetch_histogram_via(&SessionCache::new(), &root, &session_window(), "Velocity", &count_params(10.0)).unwrap();
        let sliced = fetch_histogram_via(&SessionCache::new(), &root, &window, "Velocity", &count_params(10.0)).unwrap();

        // Assert
        assert_eq!(whole.total, 1000);
        assert!(sliced.total < whole.total, "a range window binned {} of {} samples", sliced.total, whole.total);
        assert!(*sliced.bin_edges.last().unwrap() < *whole.bin_edges.last().unwrap());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_histogram_via_symmetric_puts_zero_on_a_bin_edge() {
        // Arrange — this channel is 0..999, so a symmetric range is
        // [-999, 999] and an even bin count puts zero on an interior edge.
        let root = temp_root();
        seed_session(&root);
        let params = HistogramParams { symmetric: true, ..count_params(10.0) };

        // Act
        let h = fetch_histogram_via(&SessionCache::new(), &root, &session_window(), "Velocity", &params).unwrap();

        // Assert
        assert!(h.bin_edges[0] < 0.0);
        assert!(h.bin_edges.iter().any(|&e| e.abs() < 1e-9), "zero is a bin edge: {:?}", h.bin_edges);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_histogram_via_count_mode_non_integer_bin_value_invalid_argument() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let err = fetch_histogram_via(&SessionCache::new(), &root, &session_window(), "Velocity", &count_params(10.5)).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(err.detail.unwrap()["bin_value"], 10.5);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_histogram_via_count_mode_over_the_cap_invalid_argument_naming_the_cap() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let err = fetch_histogram_via(
            &SessionCache::new(),
            &root,
            &session_window(),
            "Velocity",
            &count_params(f64::from(MAX_HISTOGRAM_BINS) + 1.0),
        )
        .unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);
        assert_eq!(err.detail.unwrap()["max_bins"], MAX_HISTOGRAM_BINS);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_histogram_via_count_mode_zero_bins_invalid_argument() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let err = fetch_histogram_via(&SessionCache::new(), &root, &session_window(), "Velocity", &count_params(0.0)).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_histogram_via_width_wider_than_the_range_is_one_bin_not_an_error() {
        // Arrange
        let root = temp_root();
        seed_session(&root);
        let params = HistogramParams { bin_mode: BinModeToken::Width, bin_value: 1e9, ..count_params(0.0) };

        // Act
        let h = fetch_histogram_via(&SessionCache::new(), &root, &session_window(), "Velocity", &params).unwrap();

        // Assert
        assert_eq!(h.bins, 1);
        assert_eq!(h.total, 1000);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_histogram_via_unknown_channel_not_found() {
        // Arrange
        let root = temp_root();
        seed_session(&root);

        // Act
        let err = fetch_histogram_via(&SessionCache::new(), &root, &session_window(), "NopeChannel", &count_params(10.0)).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_histogram_via_unknown_session_not_found() {
        // Arrange
        let root = temp_root();
        let window = WindowDto { session_id: "nope".to_string(), span: SpanDto::Session, colour: "--chart-1".to_string() };

        // Act
        let err = fetch_histogram_via(&SessionCache::new(), &root, &window, "Velocity", &count_params(10.0)).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_histogram_via_range_wholly_outside_the_session_invalid_argument() {
        // Arrange
        let root = temp_root();
        seed_session(&root);
        let window = WindowDto {
            session_id: "s1".to_string(),
            span: SpanDto::Range { t0_us: 10_000_000, t1_us: 11_000_000 },
            colour: "--chart-1".to_string(),
        };

        // Act
        let err = fetch_histogram_via(&SessionCache::new(), &root, &window, "Velocity", &count_params(10.0)).unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::InvalidArgument);

        let _ = std::fs::remove_dir_all(&root);
    }
}
