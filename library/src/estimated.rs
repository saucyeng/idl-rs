//! The physics solver's outputs as channels of a ride.
//!
//! `session estimate-physics` (and the app's Solve) writes its outputs to
//! `sessions/<id>/derived/<hash>.parquet` with `derived_kind` `"physics"`
//! (C1 §5), one file per solve. [`apply_estimated_channels`] reads every such
//! file for a session and hands each output to the session's handle under its
//! plain name ([`estimated_channel_name`]: `Physics front travel (mm)` on disk
//! reads as `Front travel (mm)`), so a workbook names it like any other
//! channel and the `wheel_travel` / `attitude("steer")` builtins read it.
//!
//! Solves can cover different stretches of a ride (one per run). Where they
//! overlap the newest (`computed_at_utc_ms`) wins; where none ran the channel
//! has no samples, and one `NaN` sample each side of a gap between two solved
//! stretches keeps a nearest-sample read from carrying a value across it. A
//! re-import deletes `derived/` (C3 §3.3), so a file here was solved from the
//! ride's current data.
//!
//! The one call an opener makes, beside
//! [`crate::bikelog::apply_session_calibration`], so the CLI and the app
//! cannot read different channels.

use std::path::{Path, PathBuf};

use arrow::array::{Float64Array, Int64Array};
use arrow::record_batch::RecordBatch;
use idl_rs::session::handle::SessionHandle;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// The `derived_kind` the physics solver writes (`crate::physics`).
pub const PHYSICS_DERIVED_KIND: &str = "physics";

/// The prefix the solver's outputs carry on disk, dropped on read.
const STORED_PREFIX: &str = "Physics ";

/// The name a stored solver output reads as: the `Physics ` prefix dropped
/// and the first letter capitalised (`Physics front travel (mm)` →
/// `Front travel (mm)`). A name without the prefix is returned unchanged.
pub fn estimated_channel_name(stored: &str) -> String {
    let Some(rest) = stored.strip_prefix(STORED_PREFIX) else {
        return stored.to_string();
    };
    let mut chars = rest.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => stored.to_string(),
    }
}

/// One solve's outputs, read from one derived file.
#[derive(Debug, Clone, PartialEq)]
pub struct SolvedSegment {
    /// When the solve ran, ms since the Unix epoch; newer wins an overlap.
    pub computed_at_utc_ms: i64,
    /// The solve's time grid, µs since session start, ascending.
    pub t_us: Vec<i64>,
    /// The solver's nominal output rate, Hz.
    pub rate_hz: f64,
    /// `(stored name, values)`, each the length of `t_us`.
    pub channels: Vec<(String, Vec<f64>)>,
}

/// One channel merged across every solve of a ride.
#[derive(Debug, Clone, PartialEq)]
pub struct MergedChannel {
    /// The plain name ([`estimated_channel_name`]).
    pub name: String,
    /// The newest contributing solve's rate, Hz.
    pub rate_hz: f64,
    /// Times, µs since session start, strictly ascending.
    pub t_us: Vec<i64>,
    /// Values, `NaN` at the gap markers.
    pub values: Vec<f64>,
}

/// Merges `segments` channel by channel: newest first, each solve
/// contributing only the samples outside the time spans of newer solves of
/// that channel, then everything in time order with a `NaN` sample one
/// sample period inside each side of a gap between two stretches. Pure.
pub fn merge_segments(segments: &[SolvedSegment]) -> Vec<MergedChannel> {
    let mut order: Vec<&SolvedSegment> = segments.iter().filter(|s| !s.t_us.is_empty()).collect();
    order.sort_by(|a, b| b.computed_at_utc_ms.cmp(&a.computed_at_utc_ms));

    let mut names: Vec<String> = Vec::new();
    for s in &order {
        for (name, _) in &s.channels {
            if !names.contains(name) {
                names.push(name.clone());
            }
        }
    }

    let mut out = Vec::new();
    for name in names {
        // (start, end, rate) of each kept run, and its samples.
        let mut taken: Vec<(i64, i64)> = Vec::new();
        let mut runs: Vec<(Vec<i64>, Vec<f64>, f64)> = Vec::new();
        let mut rate_hz = 0.0;
        for s in &order {
            let Some((_, values)) = s.channels.iter().find(|(n, _)| *n == name) else { continue };
            if rate_hz == 0.0 {
                rate_hz = s.rate_hz;
            }
            let n = values.len().min(s.t_us.len());
            let mut run_t = Vec::new();
            let mut run_v = Vec::new();
            for i in 0..n {
                let t = s.t_us[i];
                let covered = taken.iter().any(|&(a, b)| t >= a && t <= b);
                if covered {
                    if !run_t.is_empty() {
                        runs.push((std::mem::take(&mut run_t), std::mem::take(&mut run_v), s.rate_hz));
                    }
                } else {
                    run_t.push(t);
                    run_v.push(values[i]);
                }
            }
            if !run_t.is_empty() {
                runs.push((run_t, run_v, s.rate_hz));
            }
            taken.push((s.t_us[0], s.t_us[n - 1]));
        }
        runs.sort_by_key(|(t, _, _)| t[0]);

        let mut t_us: Vec<i64> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        for (k, (rt, rv, rate)) in runs.iter().enumerate() {
            let period_us = if *rate > 0.0 { (1e6 / rate).round() as i64 } else { 1 };
            if k > 0 {
                let before = rt[0] - period_us.max(1);
                if t_us.last().map_or(true, |&last| before > last) {
                    t_us.push(before);
                    vals.push(f64::NAN);
                }
            }
            for (&t, &v) in rt.iter().zip(rv) {
                if t_us.last().map_or(true, |&last| t > last) {
                    t_us.push(t);
                    vals.push(v);
                }
            }
            if k + 1 < runs.len() {
                let after = rt[rt.len() - 1] + period_us.max(1);
                if after < runs[k + 1].0[0] {
                    t_us.push(after);
                    vals.push(f64::NAN);
                }
            }
        }
        out.push(MergedChannel { name: estimated_channel_name(&name), rate_hz, t_us, values: vals });
    }
    out
}

/// What [`apply_estimated_channels`] did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EstimatedLoad {
    /// The plain names handed to the session, in solve-output order.
    pub channels: Vec<String>,
    /// How many physics solves were read.
    pub solves: usize,
    /// `(file, reason)` for each derived file that could not be read; a bad
    /// file is skipped, never fatal (the ride still opens).
    pub skipped: Vec<(PathBuf, String)>,
}

/// Reads every physics solve of `session_id` and stores each output on
/// `handle` under its plain name ([`merge_segments`]). A session never
/// solved, or with no `derived/` folder, loads nothing.
pub fn apply_estimated_channels(data_root: &Path, session_id: &str, handle: &SessionHandle) -> EstimatedLoad {
    let mut load = EstimatedLoad::default();
    let dir = data_root.join("sessions").join(session_id).join("derived");
    let Ok(entries) = std::fs::read_dir(&dir) else { return load };
    let mut segments = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
            continue;
        }
        match read_solved_segment(&path) {
            Ok(Some(segment)) => segments.push(segment),
            Ok(None) => {}
            Err(reason) => load.skipped.push((path, reason)),
        }
    }
    load.solves = segments.len();
    for merged in merge_segments(&segments) {
        load.channels.push(merged.name.clone());
        handle.store_math_with_times(&merged.name, merged.rate_hz, merged.values, merged.t_us);
    }
    load
}

/// The part of an evaluation cache key that changes when `session_id`'s
/// solves do: `solves:` and the sorted names of its derived files, or empty
/// when it has none. Names only (content-addressed, C1 §5), no file is opened.
pub fn solves_key_part(data_root: &Path, session_id: &str) -> String {
    let dir = data_root.join("sessions").join(session_id).join("derived");
    let Ok(entries) = std::fs::read_dir(&dir) else { return String::new() };
    let mut names: Vec<String> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("parquet"))
        .filter_map(|p| p.file_stem().and_then(|s| s.to_str()).map(str::to_string))
        .collect();
    if names.is_empty() {
        return String::new();
    }
    names.sort();
    format!("solves:{}", names.join(","))
}

/// One derived file as a [`SolvedSegment`], or `None` when it is not a
/// physics solve. `Err` carries why a physics file could not be read.
fn read_solved_segment(path: &Path) -> Result<Option<SolvedSegment>, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| e.to_string())?;
    let kv = builder.metadata().file_metadata().key_value_metadata().cloned().unwrap_or_default();
    let meta = |key: &str| kv.iter().find(|k| k.key == key).and_then(|k| k.value.clone());
    if meta("derived_kind").as_deref() != Some(PHYSICS_DERIVED_KIND) {
        return Ok(None);
    }
    let computed_at_utc_ms = meta("computed_at_utc_ms").and_then(|v| v.parse().ok()).unwrap_or(0);
    let schema = builder.schema().clone();
    let reader = builder.build().map_err(|e| e.to_string())?;
    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())?;
    let batch = arrow::compute::concat_batches(&schema, &batches).map_err(|e| e.to_string())?;
    let t_us = batch
        .column_by_name("t")
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>().map(|a| a.values().to_vec()))
        .ok_or_else(|| "missing or non-Int64 t column".to_string())?;
    let mut rate_hz = 0.0;
    let mut channels = Vec::new();
    for field in schema.fields() {
        if field.name() == "t" {
            continue;
        }
        let Some(col) = batch.column_by_name(field.name()).and_then(|c| c.as_any().downcast_ref::<Float64Array>()) else {
            return Err(format!("column {} is not Float64", field.name()));
        };
        if rate_hz == 0.0 {
            rate_hz = field.metadata().get("nominal_rate_hz").and_then(|v| v.parse().ok()).unwrap_or(0.0);
        }
        channels.push((field.name().clone(), col.values().to_vec()));
    }
    Ok(Some(SolvedSegment { computed_at_utc_ms, t_us, rate_hz, channels }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(at: i64, t0: i64, n: usize, name: &str, value: f64) -> SolvedSegment {
        SolvedSegment {
            computed_at_utc_ms: at,
            t_us: (0..n as i64).map(|i| t0 + i * 1_000).collect(),
            rate_hz: 1_000.0,
            channels: vec![(name.to_string(), vec![value; n])],
        }
    }

    #[test]
    fn estimated_channel_name_physics_prefix_dropped_and_capitalised() {
        // Arrange
        let stored = "Physics front travel (mm)";

        // Act
        let name = estimated_channel_name(stored);

        // Assert
        assert_eq!(name, "Front travel (mm)");
        assert_eq!(estimated_channel_name("Physics airborne"), "Airborne");
        assert_eq!(estimated_channel_name("IMU0_AccelX"), "IMU0_AccelX");
    }

    #[test]
    fn merge_segments_one_solve_its_samples_unchanged_no_markers() {
        // Arrange
        let segments = [seg(1, 10_000, 5, "Physics front travel (mm)", 40.0)];

        // Act
        let merged = merge_segments(&segments);

        // Assert
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].name, "Front travel (mm)");
        assert_eq!(merged[0].t_us, vec![10_000, 11_000, 12_000, 13_000, 14_000]);
        assert!(merged[0].values.iter().all(|v| *v == 40.0));
    }

    #[test]
    fn merge_segments_two_apart_a_nan_each_side_of_the_gap() {
        // Arrange
        let segments = [seg(1, 0, 3, "Physics steer (deg)", 1.0), seg(2, 100_000, 3, "Physics steer (deg)", 2.0)];

        // Act
        let merged = merge_segments(&segments);

        // Assert
        let ch = &merged[0];
        assert_eq!(ch.t_us, vec![0, 1_000, 2_000, 3_000, 99_000, 100_000, 101_000, 102_000]);
        assert!(ch.values[3].is_nan() && ch.values[4].is_nan());
        assert_eq!(ch.values[2], 1.0);
        assert_eq!(ch.values[5], 2.0);
    }

    #[test]
    fn merge_segments_overlapping_solves_newest_wins_where_they_overlap() {
        // Arrange — old covers 0..9 ms, new covers 5..9 ms.
        let old = seg(1, 0, 10, "Physics rear travel (mm)", 1.0);
        let new = seg(2, 5_000, 5, "Physics rear travel (mm)", 2.0);

        // Act
        let merged = merge_segments(&[old, new]);

        // Assert — old's 0..4 ms, then new's 5..9 ms, ascending, no duplicates.
        let ch = &merged[0];
        assert_eq!(ch.t_us, (0..10).map(|i| i * 1_000).collect::<Vec<i64>>());
        assert_eq!(&ch.values[..5], &[1.0; 5]);
        assert_eq!(&ch.values[5..], &[2.0; 5]);
    }

    #[test]
    fn merge_segments_no_solves_nothing() {
        // Arrange + Act
        let merged = merge_segments(&[]);

        // Assert
        assert!(merged.is_empty());
    }

    #[test]
    fn solves_key_part_no_derived_folder_empty() {
        // Arrange
        let root = std::env::temp_dir().join(format!("idl-solves-key-{}", std::process::id()));

        // Act
        let part = solves_key_part(&root, "nope");

        // Assert
        assert_eq!(part, "");
    }

    #[test]
    fn apply_estimated_channels_no_derived_folder_loads_nothing() {
        // Arrange
        let root = std::env::temp_dir().join(format!("idl-estimated-{}", std::process::id()));
        let meta = idl_rs::session::handle::SessionMetaInput { session_id: String::new(), device_id: None, timestamp_utc_ms: 0, config_checksum: None };
        let handle = SessionHandle::from_channels(meta, Vec::new());

        // Act
        let load = apply_estimated_channels(&root, "nope", &handle);

        // Assert
        assert_eq!(load, EstimatedLoad::default());
    }
}
