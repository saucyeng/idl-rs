//! One channel's decoded samples, read on its own (ruling R203.1).
//!
//! `data.parquet` is columnar: a request for one channel reads that column
//! and the `t` column, never the whole file. [`ChannelSamples`] is what
//! `crate::store::parquet::read_channel` returns — the same fields a
//! [`Channel`] carries, with the two long axes behind [`Arc`] so a cache can
//! hand the same decode to many callers without copying it (ruling R203.2).

use std::sync::Arc;

use crate::session::{Channel, GapSpan, RawColumn};

/// One channel of a session, decoded from `data.parquet` on its own.
///
/// Field-for-field a [`Channel`], except that `t_us`/`t_recorded_us` are
/// `Arc<[i64]>` rather than `Vec<i64>`: a session cache holds
/// `Arc<ChannelSamples>` and every consumer that needs a time axis
/// (`crate::math::eval::LookupChannel`, the tile encoder, the cursor) can
/// clone the `Arc` instead of the axis. Values stay in the compact
/// [`RawColumn`] form — widening to f64 is the caller's explicit
/// [`Self::materialize`] call, exactly as on [`Channel`].
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelSamples {
    /// Registry name, e.g. `IMU0_AccelZ`. Matches the parquet column name.
    pub channel_id: String,
    /// Per-sample time, µs since the session's first sample (C1 §3.1).
    pub t_us: Arc<[i64]>,
    /// Verbatim recorded time before burst-seam correction (C1 §3.2), or
    /// `None` when no correction was ever applied to this channel.
    pub t_recorded_us: Option<Arc<[i64]>>,
    /// Declared sample rate, Hz. `0.0` for event-driven channels.
    pub nominal_rate_hz: f64,
    /// Compact sample storage; `physical = raw × scale + offset`.
    pub column: RawColumn,
    /// C1 §4.2 `source_kind` (`imu0`, `gps`, `wheel`, `synthesized`, …).
    pub source_kind: String,
    /// Unit label, e.g. `m/s`. Empty means "no unit recorded" (C1 §4.1).
    pub unit: String,
    /// Synthesized-sample runs (IDL0_SPEC §15.2), grid-slot coordinates.
    pub gaps: Vec<GapSpan>,
}

impl ChannelSamples {
    /// Wraps an owned [`Channel`], moving its two time axes into `Arc`s.
    /// The one constructor — every read path funnels through here so the
    /// `Channel`/`ChannelSamples` field mapping lives in exactly one place.
    pub fn from_channel(c: Channel) -> Self {
        ChannelSamples {
            channel_id: c.channel_id,
            t_us: Arc::from(c.t_us),
            t_recorded_us: c.t_recorded_us.map(Arc::from),
            nominal_rate_hz: c.nominal_rate_hz,
            column: c.column,
            source_kind: c.source_kind,
            unit: c.unit,
            gaps: c.gaps,
        }
    }

    /// Copies back into an owned [`Channel`], for the code paths that still
    /// speak `Channel` (`SessionHandle`, the export writers). Copies both
    /// time axes — callers that only need to read them should use the
    /// `Arc`s directly.
    pub fn to_channel(&self) -> Channel {
        Channel {
            channel_id: self.channel_id.clone(),
            t_us: self.t_us.to_vec(),
            t_recorded_us: self.t_recorded_us.as_ref().map(|r| r.to_vec()),
            nominal_rate_hz: self.nominal_rate_hz,
            column: self.column.clone(),
            source_kind: self.source_kind.clone(),
            unit: self.unit.clone(),
            gaps: self.gaps.clone(),
        }
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.column.len()
    }

    /// `true` when the channel holds no samples.
    pub fn is_empty(&self) -> bool {
        self.column.is_empty()
    }

    /// Physical values, widened to f64 (`raw × scale + offset`). Allocates
    /// `8 × len` bytes on every call — the caller owns the result, this
    /// type never caches it.
    pub fn materialize(&self) -> Vec<f64> {
        self.column.materialize()
    }

    /// Physical values over the half-open sample range `[start, end)`,
    /// clamped to the channel's length.
    pub fn materialize_range(&self, start: usize, end: usize) -> Vec<f64> {
        self.column.materialize_range(start, end)
    }

    /// Physical values within the inclusive recording-time window
    /// `[t0_secs, t1_secs]`.
    ///
    /// The same window rule `SessionHandle::slice_by_time` applies — one
    /// half-open index range from
    /// [`crate::session::handle::time_window_index_range`] over this
    /// channel's own `t_us`, then a range materialize — so a lap or window
    /// slice taken from a single-channel read matches one taken from a
    /// whole-session handle sample for sample. Empty when the channel is
    /// empty or the window covers no sample.
    pub fn slice_by_time(&self, t0_secs: f64, t1_secs: f64) -> Vec<f64> {
        if self.is_empty() {
            return Vec::new();
        }
        let (lo, hi) = crate::session::handle::time_window_index_range(&self.t_us, t0_secs, t1_secs);
        if lo >= hi {
            return Vec::new();
        }
        self.column.materialize_range(lo, hi)
    }

    /// This channel's `t_us` restricted to the inclusive recording-time
    /// window `[t0_secs, t1_secs]` — the timestamps matching
    /// [`Self::slice_by_time`]'s values, one for one.
    pub fn slice_t_us_by_time(&self, t0_secs: f64, t1_secs: f64) -> Vec<i64> {
        let (lo, hi) = crate::session::handle::time_window_index_range(&self.t_us, t0_secs, t1_secs);
        self.t_us[lo..hi].to_vec()
    }

    /// Resident heap bytes: both time axes plus the compact sample storage.
    ///
    /// The number a byte-budgeted cache evicts on (ruling R203.2). It counts
    /// the sample and timestamp buffers, not the small `String`/`Vec<GapSpan>`
    /// metadata, which is bounded per channel and irrelevant at session
    /// scale; it is an accounting figure, not an allocator measurement.
    pub fn resident_bytes(&self) -> usize {
        let t = self.t_us.len() * std::mem::size_of::<i64>();
        let r = self.t_recorded_us.as_ref().map_or(0, |v| v.len() * std::mem::size_of::<i64>());
        t + r + column_bytes(&self.column)
    }
}

/// Resident heap bytes of one [`RawColumn`] — per-sample width × length for
/// the compact variants, the `base` array for `Interp`, zero for `Ramp`
/// (closed-form, no storage).
fn column_bytes(col: &RawColumn) -> usize {
    match col {
        RawColumn::I16 { data, .. } => data.len() * std::mem::size_of::<i16>(),
        RawColumn::I32 { data, .. } => data.len() * std::mem::size_of::<i32>(),
        RawColumn::F32 { data, .. } => data.len() * std::mem::size_of::<f32>(),
        RawColumn::F64(data) => data.len() * std::mem::size_of::<f64>(),
        RawColumn::Ramp { .. } => 0,
        RawColumn::Interp { base, .. } => base.len() * std::mem::size_of::<f64>(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn channel(column: RawColumn, t: Vec<i64>) -> Channel {
        Channel {
            channel_id: "C".to_string(),
            t_us: t,
            t_recorded_us: None,
            nominal_rate_hz: 100.0,
            column,
            source_kind: "imu0".to_string(),
            unit: "g".to_string(),
            gaps: Vec::new(),
        }
    }

    #[test]
    fn channel_samples_round_trip_an_i16_channel_with_a_recorded_axis_restores_every_field() {
        // Arrange
        let mut c = channel(RawColumn::I16 { data: vec![1, 2, 3], scale: 0.5, offset: 1.0 }, vec![0, 10, 20]);
        c.t_recorded_us = Some(vec![0, 11, 19]);
        c.gaps = vec![GapSpan { start: 1, len: 1 }];

        // Act
        let back = ChannelSamples::from_channel(c.clone()).to_channel();

        // Assert
        assert_eq!(back, c);
    }

    #[test]
    fn resident_bytes_i16_samples_plus_one_time_axis_counts_two_bytes_per_sample_and_eight_per_timestamp() {
        // Arrange
        let c = channel(RawColumn::I16 { data: vec![0; 100], scale: 1.0, offset: 0.0 }, vec![0; 100]);

        // Act
        let bytes = ChannelSamples::from_channel(c).resident_bytes();

        // Assert
        assert_eq!(bytes, 100 * 2 + 100 * 8);
    }

    #[test]
    fn resident_bytes_a_recorded_axis_is_present_counts_it_as_a_second_time_axis() {
        // Arrange
        let mut c = channel(RawColumn::F64(vec![0.0; 10]), vec![0; 10]);
        c.t_recorded_us = Some(vec![0; 10]);

        // Act
        let bytes = ChannelSamples::from_channel(c).resident_bytes();

        // Assert
        assert_eq!(bytes, 10 * 8 + 10 * 8 + 10 * 8);
    }

    #[test]
    fn resident_bytes_a_closed_form_ramp_column_counts_no_sample_storage() {
        // Arrange
        let c = channel(RawColumn::Ramp { len: 1_000_000, rate: 1000.0 }, Vec::new());

        // Act
        let bytes = ChannelSamples::from_channel(c).resident_bytes();

        // Assert
        assert_eq!(bytes, 0);
    }

    #[test]
    fn materialize_an_i16_channel_with_scale_and_offset_widens_raw_values_the_same_way_channel_does() {
        // Arrange
        let c = channel(RawColumn::I16 { data: vec![2, 4], scale: 0.5, offset: 1.0 }, vec![0, 10]);

        // Act
        let samples = ChannelSamples::from_channel(c).materialize();

        // Assert
        assert_eq!(samples, vec![2.0, 3.0]);
    }
}
