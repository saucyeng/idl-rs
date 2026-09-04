//! `data.parquet` Arrow schema, writer, and reader (contract C1 §4). One
//! wide file per session, row-indexed by the union time axis `t`.

use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Encoding;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::schema::types::ColumnPath;

use crate::session::{Channel, GapSpan, RawColumn, Session, SourceFormat};
use crate::store::atomic::write_atomic;

/// `data.parquet`'s row-group target (C1 §4.4).
const ROW_GROUP_SIZE: usize = 1_000_000;

/// Discriminant for [`ParquetStoreError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParquetStoreErrorKind {
    /// A filesystem or Arrow/Parquet I/O operation failed.
    Io,
    /// The file (or a column's metadata) didn't parse as a valid C1 §4
    /// `data.parquet` — malformed schema, missing required metadata key, or
    /// a value that didn't round-trip through its string encoding.
    Schema,
}

/// Error from the Parquet store. Never `Err(String)` (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParquetStoreError {
    pub kind: ParquetStoreErrorKind,
    pub message: String,
}

impl ParquetStoreError {
    fn new(kind: ParquetStoreErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}

impl fmt::Display for ParquetStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for ParquetStoreError {}

/// Sorted, deduplicated union of every channel's `t_us` (C1 §3.5 invariant
/// 2) — the file's row axis, in microseconds.
fn union_t_axis(session: &Session) -> Vec<i64> {
    let mut set: BTreeSet<i64> = BTreeSet::new();
    for c in &session.channels {
        set.extend(c.t_us.iter().copied());
    }
    set.into_iter().collect()
}

/// For channel `c`, the row index in `t` (sorted ascending) that each of
/// `c`'s own samples belongs to — a lookup miss is a bug, not user data
/// (`t` is the union of every channel's `t_us`), so it is surfaced as a
/// hard [`ParquetStoreError`] rather than silently dropping a sample).
fn row_indices_for(c: &Channel, t: &[i64]) -> Result<Vec<usize>, ParquetStoreError> {
    c.t_us
        .iter()
        .map(|&ts| {
            t.binary_search(&ts).map_err(|_| {
                ParquetStoreError::new(
                    ParquetStoreErrorKind::Schema,
                    format!("channel {} has t_us={ts} not present in the union axis (internal bug)", c.channel_id),
                )
            })
        })
        .collect()
}

/// Scatters `c`'s raw samples into a nullable Arrow array of `t.len()`
/// rows, null everywhere `c` did not sample. One arm per [`RawColumn`]
/// variant that round-trips (C1 §2's table) — `Ramp`/`Interp` are never
/// columns (never called with those variants; synthesized channels are
/// excluded from `data.parquet` entirely, see [`write_session_parquet`]).
fn channel_array(c: &Channel, rows: &[usize], n_rows: usize) -> Result<ArrayRef, ParquetStoreError> {
    match &c.column {
        RawColumn::I16 { data, .. } => {
            let mut vals: Vec<Option<i16>> = vec![None; n_rows];
            for (&row, &v) in rows.iter().zip(data.iter()) {
                vals[row] = Some(v);
            }
            Ok(Arc::new(Int16Array::from(vals)))
        }
        RawColumn::I32 { data, .. } => {
            let mut vals: Vec<Option<i32>> = vec![None; n_rows];
            for (&row, &v) in rows.iter().zip(data.iter()) {
                vals[row] = Some(v);
            }
            Ok(Arc::new(Int32Array::from(vals)))
        }
        RawColumn::F32 { data, .. } => {
            let mut vals: Vec<Option<f32>> = vec![None; n_rows];
            for (&row, &v) in rows.iter().zip(data.iter()) {
                vals[row] = Some(v);
            }
            Ok(Arc::new(Float32Array::from(vals)))
        }
        RawColumn::F64(data) => {
            let mut vals: Vec<Option<f64>> = vec![None; n_rows];
            for (&row, &v) in rows.iter().zip(data.iter()) {
                vals[row] = Some(v);
            }
            Ok(Arc::new(Float64Array::from(vals)))
        }
        RawColumn::Ramp { .. } | RawColumn::Interp { .. } => Err(ParquetStoreError::new(
            ParquetStoreErrorKind::Schema,
            format!("{} is a synthesized (Ramp/Interp) column — never written to data.parquet (C1 §2)", c.channel_id),
        )),
    }
}

/// Scatters a `<source>_t_recorded_us` column: the verbatim recorded time
/// at every row this source actually sampled (its own `t_us`, since that's
/// where the row lives), null elsewhere. One call per distinct
/// `source_kind` present, using any one of that source's channels (they
/// all share the same `t_us`/`t_recorded_us` by construction — one FIFO
/// read per source, C1 §3.2).
fn recorded_us_array(c: &Channel, rows: &[usize], n_rows: usize) -> ArrayRef {
    let recorded = c.t_recorded_us_or_t_us();
    let mut vals: Vec<Option<i64>> = vec![None; n_rows];
    for (&row, &v) in rows.iter().zip(recorded.iter()) {
        vals[row] = Some(v);
    }
    Arc::new(Int64Array::from(vals))
}

/// Column metadata for one channel column (C1 §4.2). `scale`/`offset` only
/// on `Int16`/`Int32`/`Float32`; every other key always present.
fn column_metadata(c: &Channel) -> Vec<(String, String)> {
    let mut kv = Vec::new();
    match &c.column {
        RawColumn::I16 { scale, offset, .. }
        | RawColumn::I32 { scale, offset, .. }
        | RawColumn::F32 { scale, offset, .. } => {
            kv.push(("scale".to_string(), scale.to_string()));
            kv.push(("offset".to_string(), offset.to_string()));
        }
        _ => {}
    }
    kv.push(("nominal_rate_hz".to_string(), c.nominal_rate_hz.to_string()));
    kv.push(("unit".to_string(), c.unit.clone()));
    kv.push(("source_kind".to_string(), c.source_kind.clone()));
    kv.push(("channel_kind".to_string(), if c.nominal_rate_hz == 0.0 { "event" } else { "fixed-rate" }.to_string()));
    if !c.gaps.is_empty() {
        let gaps_json = gaps_to_json(&c.gaps);
        kv.push(("gaps".to_string(), gaps_json));
    }
    kv
}

/// `[{"start":N,"len":N}, ...]`, hand-rolled (no serde dependency needed for
/// this one small, fixed shape — keeps `store/parquet.rs` free of a
/// `serde_json` requirement beyond what `session_json.rs` needs elsewhere).
fn gaps_to_json(gaps: &[GapSpan]) -> String {
    let parts: Vec<String> = gaps.iter().map(|g| format!(r#"{{"start":{},"len":{}}}"#, g.start, g.len)).collect();
    format!("[{}]", parts.join(","))
}

/// Parses `gaps_to_json`'s output back. `Schema`-kind error on malformed
/// JSON (never panics on untrusted file content, CLAUDE.md §5).
fn gaps_from_json(s: &str) -> Result<Vec<GapSpan>, ParquetStoreError> {
    let inner = s.trim().trim_start_matches('[').trim_end_matches(']');
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split("},")
        .map(|chunk| {
            let chunk = chunk.trim_start_matches('{').trim_end_matches('}');
            let mut start = None;
            let mut len = None;
            for field in chunk.split(',') {
                let (k, v) = field.split_once(':').ok_or_else(|| {
                    ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("malformed gaps JSON: {s}"))
                })?;
                let v: usize = v.trim().parse().map_err(|_| {
                    ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("malformed gaps JSON: {s}"))
                })?;
                match k.trim().trim_matches('"') {
                    "start" => start = Some(v),
                    "len" => len = Some(v),
                    _ => {}
                }
            }
            match (start, len) {
                (Some(start), Some(len)) => Ok(GapSpan { start, len }),
                _ => Err(ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("malformed gaps JSON: {s}"))),
            }
        })
        .collect()
}

/// File-level key-value metadata (C1 §4.3).
fn file_metadata(session: &Session, importer_version: &str) -> Vec<(String, String)> {
    let mut kv = vec![
        ("session_id".to_string(), session.session_id.clone()),
        ("timestamp_utc_ms".to_string(), session.timestamp_utc_ms.to_string()),
        ("blob_sha256".to_string(), session.blob_sha256.clone()),
        ("source_format".to_string(), session.source_format.as_str().to_string()),
        ("importer_version".to_string(), importer_version.to_string()),
        ("engine_version".to_string(), crate::VERSION.to_string()),
        ("seam_correction_version".to_string(), crate::session::seam_correction::SEAM_CORRECTION_VERSION.to_string()),
    ];
    if let Some(d) = &session.device_id {
        kv.push(("device_id".to_string(), d.clone()));
    }
    if let Some(c) = &session.config_checksum {
        kv.push(("config_checksum".to_string(), c.clone()));
    }
    kv
}

/// Writes `session` to `<data_root>/sessions/<session_id>/data.parquet`
/// (path per contract C4 §2) via the atomic-write primitive. Excludes
/// engine-synthesized channels (`Time`, `Distance` — never columns per C1
/// §2's table), identified by `source_kind == "synthesized"`, **not** by
/// `RawColumn` variant — Task 4 changes `Time`'s in-memory representation
/// to `RawColumn::F64` (C1 §3.5 invariant 4), so a `RawColumn`-variant
/// filter would silently stop excluding `Time`, violating C1 §4.1.
/// `source_kind: "synthesized"` is set on both `Time` and `Distance` by
/// `session::synthesis` and is stable regardless of either channel's
/// underlying `RawColumn` representation.
pub fn write_session_parquet(
    data_root: &Path,
    session: &Session,
    importer_version: &str,
) -> Result<PathBuf, ParquetStoreError> {
    let t = union_t_axis(session);
    let n_rows = t.len();

    let mut fields = vec![Field::new("t", DataType::Int64, false)];
    let mut arrays: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(t.clone()))];

    // <source>_t_recorded_us columns — one per distinct source_kind among
    // real (non-synthesized) channels.
    let mut seen_sources: Vec<&str> = Vec::new();
    for c in &session.channels {
        if c.source_kind == "synthesized" {
            continue; // Time/Distance — excluded (C1 §2), by source_kind not RawColumn variant
        }
        if seen_sources.contains(&c.source_kind.as_str()) {
            continue;
        }
        seen_sources.push(&c.source_kind);
        let rows = row_indices_for(c, &t)?;
        let field_name = format!("{}_t_recorded_us", c.source_kind);
        fields.push(
            Field::new(field_name.as_str(), DataType::Int64, true)
                .with_metadata([("source_kind".to_string(), c.source_kind.clone())].into_iter().collect()),
        );
        arrays.push(recorded_us_array(c, &rows, n_rows));
    }

    // Channel value columns.
    for c in &session.channels {
        if c.source_kind == "synthesized" {
            continue;
        }
        let rows = row_indices_for(c, &t)?;
        let arrow_type = match &c.column {
            RawColumn::I16 { .. } => DataType::Int16,
            RawColumn::I32 { .. } => DataType::Int32,
            RawColumn::F32 { .. } => DataType::Float32,
            RawColumn::F64(_) => DataType::Float64,
            RawColumn::Ramp { .. } | RawColumn::Interp { .. } => {
                return Err(ParquetStoreError::new(
                    ParquetStoreErrorKind::Schema,
                    format!(
                        "{} is a synthesized (Ramp/Interp) column with source_kind != \"synthesized\" (internal bug)",
                        c.channel_id
                    ),
                ))
            }
        };
        let metadata: std::collections::HashMap<String, String> = column_metadata(c).into_iter().collect();
        fields.push(Field::new(c.channel_id.as_str(), arrow_type, true).with_metadata(metadata));
        arrays.push(channel_array(c, &rows, n_rows)?);
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("RecordBatch::try_new: {e}")))?;

    // `WriterPropertiesBuilder::set_key_value_metadata` *replaces* rather
    // than accumulates (verified against the pinned 59.3.0 source,
    // `parquet::file::properties::WriterPropertiesBuilder::set_key_value_metadata`
    // — `self.key_value_metadata = value`), so every file-metadata pair is
    // collected into one `Vec<KeyValue>` and set in a single call.
    let all_kv: Vec<KeyValue> = file_metadata(session, importer_version)
        .into_iter()
        .map(|(k, v)| KeyValue::new(k, v))
        .collect();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(ROW_GROUP_SIZE))
        .set_column_encoding(ColumnPath::from("t"), Encoding::DELTA_BINARY_PACKED)
        .set_column_statistics_enabled(ColumnPath::from("t"), EnabledStatistics::Chunk)
        .set_key_value_metadata(Some(all_kv))
        .build();

    let mut buf: Vec<u8> = Vec::new();
    {
        let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(props))
            .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("ArrowWriter::try_new: {e}")))?;
        writer.write(&batch).map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("write: {e}")))?;
        writer.close().map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("close: {e}")))?;
    }

    let target = data_root.join("sessions").join(&session.session_id).join("data.parquet");
    write_atomic(data_root, &target, &buf, None)
        .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, e.to_string()))?;
    Ok(target)
}

/// `data.parquet`'s file-level key-value metadata (C1 §4.3) — the nine keys
/// every `data.parquet` file carries. Returned by [`read_session_metadata`],
/// which reads only the file footer, no row-group/column data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionParquetMetadata {
    pub session_id: String,
    pub timestamp_utc_ms: i64,
    pub device_id: Option<String>,
    pub config_checksum: Option<String>,
    pub blob_sha256: String,
    /// One of `"idl0"`/`"fit"`/`"gpx"`/`"csv"` (the raw wire token, not the
    /// parsed [`SourceFormat`] — [`read_session_parquet`] parses it further).
    pub source_format: String,
    pub importer_version: String,
    pub engine_version: String,
    pub seam_correction_version: String,
}

/// Reads `data.parquet`'s file-level key-value metadata (C1 §4.3) only — no
/// row-group or column materialization, just the Parquet footer. The single
/// parser for these nine keys; [`read_session_parquet`] calls this rather
/// than duplicating the parsing logic.
pub fn read_session_metadata(path: &Path) -> Result<SessionParquetMetadata, ParquetStoreError> {
    let file = std::fs::File::open(path)
        .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("open {}: {e}", path.display())))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("{}: {e}", path.display())))?;

    let file_kv: std::collections::HashMap<String, String> = builder
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|kv| kv.value.map(|v| (kv.key, v)))
        .collect();

    let get = |k: &str| -> Result<String, ParquetStoreError> {
        file_kv.get(k).cloned().ok_or_else(|| {
            ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("missing required file metadata key {k}"))
        })
    };
    let timestamp_utc_ms: i64 = get("timestamp_utc_ms")?.parse().map_err(|_| {
        ParquetStoreError::new(ParquetStoreErrorKind::Schema, "timestamp_utc_ms did not parse as i64".to_string())
    })?;
    Ok(SessionParquetMetadata {
        session_id: get("session_id")?,
        timestamp_utc_ms,
        device_id: file_kv.get("device_id").cloned(),
        config_checksum: file_kv.get("config_checksum").cloned(),
        blob_sha256: get("blob_sha256")?,
        source_format: get("source_format")?,
        importer_version: get("importer_version")?,
        engine_version: get("engine_version")?,
        seam_correction_version: get("seam_correction_version")?,
    })
}

/// Reads `data.parquet` back into a [`Session`] (contract C1 §4.5's read
/// rule: for each channel column, filter to non-null rows, take `t` at
/// those rows as `t_us`, the values as the compact `RawColumn`).
/// Synthesized `Time`/`Distance` are **not** reconstructed here — they are
/// re-derived by `crate::session::synthesis::synthesize_base_channels`
/// after this function returns, exactly as it already runs after parsing.
pub fn read_session_parquet(path: &Path) -> Result<Session, ParquetStoreError> {
    let meta = read_session_metadata(path)?;

    let file = std::fs::File::open(path)
        .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("open {}: {e}", path.display())))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("{}: {e}", path.display())))?;
    let schema = builder.schema().clone();

    let reader = builder
        .build()
        .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("build reader: {e}")))?;
    let batches: Vec<RecordBatch> = reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("read batches: {e}")))?;
    // A `data.parquet` file is a single logical table; concatenate row
    // groups into one batch for the (session-scale, not season-scale) read
    // this function does. A streaming, per-row-group reconstruction is a
    // future optimisation, not required by C1's round-trip guarantees.
    let batch = arrow::compute::concat_batches(&schema, &batches)
        .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("concat_batches: {e}")))?;

    let session_id = meta.session_id;
    let timestamp_utc_ms = meta.timestamp_utc_ms;
    let blob_sha256 = meta.blob_sha256;
    let source_format = match meta.source_format.as_str() {
        "idl0" => SourceFormat::Idl0,
        "fit" => SourceFormat::Fit,
        "gpx" => SourceFormat::Gpx,
        "csv" => SourceFormat::Csv,
        other => {
            return Err(ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("unknown source_format {other}")))
        }
    };
    let device_id = meta.device_id;
    let config_checksum = meta.config_checksum;

    let t_col = batch
        .column_by_name("t")
        .ok_or_else(|| ParquetStoreError::new(ParquetStoreErrorKind::Schema, "missing t column".to_string()))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| ParquetStoreError::new(ParquetStoreErrorKind::Schema, "t column is not Int64".to_string()))?;

    let mut channels = Vec::new();
    for field in schema.fields() {
        let name = field.name();
        if name == "t" || name.ends_with("_t_recorded_us") {
            continue;
        }
        let meta = field.metadata();
        let get_meta = |k: &str| -> Result<String, ParquetStoreError> {
            meta.get(k).cloned().ok_or_else(|| {
                ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("column {name} missing metadata key {k}"))
            })
        };
        let nominal_rate_hz: f64 = get_meta("nominal_rate_hz")?.parse().map_err(|_| {
            ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("column {name}: bad nominal_rate_hz"))
        })?;
        let unit = get_meta("unit")?;
        let source_kind = get_meta("source_kind")?;
        let gaps = match meta.get("gaps") {
            Some(g) => gaps_from_json(g)?,
            None => Vec::new(),
        };

        let col = batch
            .column_by_name(name)
            .ok_or_else(|| ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("missing column {name}")))?;

        let recorded_col_name = format!("{source_kind}_t_recorded_us");
        let recorded_col = batch
            .column_by_name(&recorded_col_name)
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());

        let (t_us, t_recorded_us, column) = read_column(col, t_col, recorded_col, meta)?;

        channels.push(Channel {
            channel_id: name.clone(),
            t_us,
            t_recorded_us,
            nominal_rate_hz,
            column,
            source_kind,
            unit,
            gaps,
        });
    }

    Ok(Session { session_id, device_id, timestamp_utc_ms, config_checksum, source_format, blob_sha256, channels })
}

/// Filters `col`'s non-null rows, pairing each with `t`'s value at that row
/// (C1 §4.5's read rule) and, when present, `recorded_col`'s value at that
/// row (`None` overall when `recorded_col` is absent — same non-IMU-source
/// case `Channel.t_recorded_us` documents as `None`). When `recorded_col` is
/// present but element-wise identical to the gathered `t_us` (every
/// non-burst-corrected source, C1 §3.3 — no correction ever applied), the
/// reconstructed `t_recorded_us` collapses back to `None` rather than a
/// duplicate `Some`, matching C1 §2's signed-field convention and what was
/// originally written. Reconstructs the typed [`RawColumn`] from
/// `scale`/`offset` metadata for `Int16`/`Int32`/`Float32`; `Float64` is
/// read verbatim (bit-exact, including `-0.0`/`NaN` — no `× 1.0 + 0.0`,
/// C1 §2's `F64` round-trip guarantee).
fn read_column(
    col: &ArrayRef,
    t_col: &Int64Array,
    recorded_col: Option<&Int64Array>,
    meta: &std::collections::HashMap<String, String>,
) -> Result<(Vec<i64>, Option<Vec<i64>>, RawColumn), ParquetStoreError> {
    let parse_f64 = |k: &str| -> Result<f64, ParquetStoreError> {
        meta.get(k)
            .ok_or_else(|| ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("missing {k}")))?
            .parse()
            .map_err(|_| ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("bad {k}")))
    };

    macro_rules! gather {
        ($arr_ty:ty, $wrap:expr) => {{
            let arr = col.as_any().downcast_ref::<$arr_ty>().ok_or_else(|| {
                ParquetStoreError::new(ParquetStoreErrorKind::Schema, "column type mismatch".to_string())
            })?;
            let mut t_us = Vec::new();
            let mut t_recorded_us = Vec::new();
            let mut values = Vec::new();
            for i in 0..arr.len() {
                if arr.is_valid(i) {
                    t_us.push(t_col.value(i));
                    if let Some(rc) = recorded_col {
                        if rc.is_valid(i) {
                            t_recorded_us.push(rc.value(i));
                        }
                    }
                    values.push(arr.value(i));
                }
            }
            let t_recorded_us =
                if recorded_col.is_some() && t_recorded_us.len() == t_us.len() && t_recorded_us != t_us {
                    Some(t_recorded_us)
                } else {
                    None
                };
            (t_us, t_recorded_us, $wrap(values))
        }};
    }

    let (t_us, t_recorded_us, column) = match col.data_type() {
        DataType::Int16 => {
            let scale = parse_f64("scale")?;
            let offset = parse_f64("offset")?;
            gather!(Int16Array, |data| RawColumn::I16 { data, scale, offset })
        }
        DataType::Int32 => {
            let scale = parse_f64("scale")?;
            let offset = parse_f64("offset")?;
            gather!(Int32Array, |data| RawColumn::I32 { data, scale, offset })
        }
        DataType::Float32 => {
            let scale = parse_f64("scale")?;
            let offset = parse_f64("offset")?;
            gather!(Float32Array, |data| RawColumn::F32 { data, scale, offset })
        }
        DataType::Float64 => gather!(Float64Array, RawColumn::F64),
        other => {
            return Err(ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("unsupported column type {other:?}")))
        }
    };
    Ok((t_us, t_recorded_us, column))
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A small, realistic synthetic session: one IMU-shaped I16 channel
    /// (scale/offset like a real accel axis), one GPS-shaped F64 channel
    /// carrying -0.0/NaN, distinct source_kinds so both `<source>_t_recorded_us`
    /// columns exist, and one gap.
    fn sample_session() -> Session {
        let imu = Channel {
            channel_id: "IMU0_AccelX".to_string(),
            t_us: vec![0, 1250, 2500, 5000], // a gap between index 1 and 2's neighbour
            t_recorded_us: Some(vec![0, 1250, 2500, 5000]),
            nominal_rate_hz: 800.0,
            column: RawColumn::I16 { data: vec![100, -200, 300, 400], scale: 32.0 / 32768.0, offset: 0.0 },
            source_kind: "imu0".to_string(),
            unit: "g".to_string(),
            gaps: vec![GapSpan { start: 2, len: 1 }],
        };
        let gps = Channel {
            channel_id: "GPS_EpochMs".to_string(),
            t_us: vec![0, 2500],
            t_recorded_us: None,
            nominal_rate_hz: 10.0,
            column: RawColumn::F64(vec![-0.0, f64::NAN]),
            source_kind: "gps".to_string(),
            unit: "ms_raw".to_string(),
            gaps: Vec::new(),
        };
        Session {
            session_id: "0102030405060708090a0b0c0d0e0f10".to_string(),
            device_id: Some("b0b1b2b3b4b5".to_string()),
            timestamp_utc_ms: 1_756_857_600_000,
            config_checksum: Some("cafebabe".to_string()),
            source_format: SourceFormat::Idl0,
            blob_sha256: "0".repeat(64),
            channels: vec![imu, gps],
        }
    }

    #[test]
    fn round_trip_recorded_stamps_bit_exact() {
        // Arrange
        let root = temp_root();
        let session = sample_session();
        let path = write_session_parquet(&root, &session, "0.1.0").unwrap();

        // Act
        let back = read_session_parquet(&path).unwrap();

        // Assert — C1 §7 #1.
        let imu = back.channels.iter().find(|c| c.channel_id == "IMU0_AccelX").unwrap();
        assert_eq!(imu.t_recorded_us_or_t_us(), &[0, 1250, 2500, 5000][..]);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn round_trip_recorded_stamps_none_stays_none() {
        // Arrange — `sample_session()`'s GPS channel is written with
        // `t_recorded_us: None` (the documented common case, C1 §2/§3.3: no
        // burst correction ever applies to GPS, so the `gps_t_recorded_us`
        // column is present per §4.1 but element-wise identical to `t`).
        let root = temp_root();
        let session = sample_session();
        assert_eq!(session.channels.iter().find(|c| c.channel_id == "GPS_EpochMs").unwrap().t_recorded_us, None);
        let path = write_session_parquet(&root, &session, "0.1.0").unwrap();

        // Act
        let back = read_session_parquet(&path).unwrap();

        // Assert — C1 §2's signed-field contract: a channel whose recorded
        // stamps never diverged from `t` round-trips to `None`, not
        // `Some(<duplicate of t_us>)`.
        let gps = back.channels.iter().find(|c| c.channel_id == "GPS_EpochMs").unwrap();
        assert_eq!(gps.t_recorded_us, None);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn round_trip_raw_i16_counts_bit_exact() {
        // Arrange
        let root = temp_root();
        let session = sample_session();
        let path = write_session_parquet(&root, &session, "0.1.0").unwrap();

        // Act
        let back = read_session_parquet(&path).unwrap();

        // Assert — C1 §7 #2.
        let imu = back.channels.iter().find(|c| c.channel_id == "IMU0_AccelX").unwrap();
        match &imu.column {
            RawColumn::I16 { data, .. } => assert_eq!(data, &vec![100i16, -200, 300, 400]),
            other => panic!("expected I16, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn round_trip_scale_offset_metadata_bit_exact() {
        // Arrange
        let root = temp_root();
        let session = sample_session();
        let path = write_session_parquet(&root, &session, "0.1.0").unwrap();

        // Act
        let back = read_session_parquet(&path).unwrap();

        // Assert — C1 §7 #3: parsed-back bit pattern, not just string equality.
        let imu = back.channels.iter().find(|c| c.channel_id == "IMU0_AccelX").unwrap();
        match &imu.column {
            RawColumn::I16 { scale, offset, .. } => {
                assert_eq!(scale.to_bits(), (32.0f64 / 32768.0).to_bits());
                assert_eq!(offset.to_bits(), 0.0f64.to_bits());
            }
            other => panic!("expected I16, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn round_trip_nominal_rate_hz_preserved() {
        // Arrange
        let root = temp_root();
        let session = sample_session();
        let path = write_session_parquet(&root, &session, "0.1.0").unwrap();

        // Act
        let back = read_session_parquet(&path).unwrap();

        // Assert — C1 §7 #4.
        let imu = back.channels.iter().find(|c| c.channel_id == "IMU0_AccelX").unwrap();
        assert_eq!(imu.nominal_rate_hz.to_bits(), 800.0f64.to_bits());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn round_trip_gaps_preserved() {
        // Arrange
        let root = temp_root();
        let session = sample_session();
        let path = write_session_parquet(&root, &session, "0.1.0").unwrap();

        // Act
        let back = read_session_parquet(&path).unwrap();

        // Assert — C1 §7 #5.
        let imu = back.channels.iter().find(|c| c.channel_id == "IMU0_AccelX").unwrap();
        assert_eq!(imu.gaps, vec![GapSpan { start: 2, len: 1 }]);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn round_trip_union_axis_sorted_unique_and_channels_reproduce_their_own_t_us() {
        // Arrange
        let root = temp_root();
        let session = sample_session();
        let path = write_session_parquet(&root, &session, "0.1.0").unwrap();

        // Act
        let back = read_session_parquet(&path).unwrap();

        // Assert — C1 §7 #6.
        let imu = back.channels.iter().find(|c| c.channel_id == "IMU0_AccelX").unwrap();
        assert_eq!(imu.t_us, vec![0, 1250, 2500, 5000]);
        let gps = back.channels.iter().find(|c| c.channel_id == "GPS_EpochMs").unwrap();
        assert_eq!(gps.t_us, vec![0, 2500]);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn round_trip_f64_negative_zero_and_nan_preserved_bit_exact() {
        // Arrange
        let root = temp_root();
        let session = sample_session();
        let path = write_session_parquet(&root, &session, "0.1.0").unwrap();

        // Act
        let back = read_session_parquet(&path).unwrap();

        // Assert — C1 §7 #7.
        let gps = back.channels.iter().find(|c| c.channel_id == "GPS_EpochMs").unwrap();
        match &gps.column {
            RawColumn::F64(data) => {
                assert!(data[0].is_sign_negative() && data[0] == 0.0);
                assert!(data[1].is_nan());
            }
            other => panic!("expected F64, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn file_metadata_round_trips() {
        // Arrange
        let root = temp_root();
        let session = sample_session();
        let path = write_session_parquet(&root, &session, "0.1.0").unwrap();

        // Act
        let back = read_session_parquet(&path).unwrap();

        // Assert
        assert_eq!(back.session_id, session.session_id);
        assert_eq!(back.device_id, session.device_id);
        assert_eq!(back.config_checksum, session.config_checksum);
        assert_eq!(back.blob_sha256, session.blob_sha256);
        assert_eq!(back.timestamp_utc_ms, session.timestamp_utc_ms);
        assert!(matches!(back.source_format, SourceFormat::Idl0));

        let _ = std::fs::remove_dir_all(&root);
    }
}
