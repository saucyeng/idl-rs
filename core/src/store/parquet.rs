//! `data.parquet` Arrow schema, writer, and reader (contract C1 §4). One
//! wide file per session, row-indexed by the union time axis `t`.

use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanBufferBuilder, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, PrimitiveArray,
    RecordBatchReader,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{ArrowPrimitiveType, DataType, Field, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::{compute_leaves, ArrowColumnWriter};
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::basic::Encoding;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::schema::types::ColumnPath;

use crate::session::handle::{ChannelSource, LazyChannelInfo, SessionHandle};
use crate::session::{Channel, ChannelSamples, GapSpan, RawColumn, Session, SourceFormat};
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
    /// The file is a valid `data.parquet`, but it holds no channel by the
    /// requested name (and the name is not one this store can synthesize).
    /// Raised only by the single-channel read path ([`read_channel`]) —
    /// the whole-file readers never ask for a name.
    NotFound,
    /// The allocator refused a buffer this read needs (ruling R211.4). The
    /// file is fine and the request is legitimate — this machine cannot
    /// hold the result right now. Never a panic and never an abort: the
    /// caller maps it to C3 §1's `resource_exhausted` and the app shows a
    /// toast.
    ResourceExhausted,
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

/// Calls `f(row, value)` for each of a channel's samples, `row` being that
/// sample's index on the union axis `t`.
///
/// Both `t` (sorted and deduplicated) and a well-formed `c_t_us` (C1 §3.5
/// invariant 1) ascend, so this is one merge walk rather than a binary
/// search per sample, and it never materialises the per-sample row-index
/// vector the previous version built — 8 B/sample, which at session scale
/// is a bigger allocation than the samples themselves (ruling R203.3
/// counts it).
///
/// Samples are visited in the channel's own order and a row may be visited
/// more than once (a channel with repeated timestamps); what that means is
/// `f`'s decision, not this function's.
fn for_each_row<T>(
    channel_id: &str,
    c_t_us: &[i64],
    t: &[i64],
    n_rows: usize,
    values: impl Iterator<Item = T>,
    mut f: impl FnMut(usize, T),
) -> Result<(), ParquetStoreError> {
    let mut row = 0usize;
    for (i, v) in values.enumerate() {
        let Some(&ts) = c_t_us.get(i) else { break };
        while row < n_rows && t[row] < ts {
            row += 1;
        }
        // The forward walk covers a strictly increasing `t_us` (C1 §3.5
        // invariant 1) in one pass. Not every real source honours that
        // invariant, though — an event-driven channel like `HR_RR` can
        // report two intervals at one timestamp, or report them slightly
        // out of order — so a walk miss falls back to a search over the
        // whole axis rather than failing. Only then is a miss a genuine
        // internal bug: `t` is the union of every channel's own `t_us`, so
        // a timestamp the search cannot find was never in the union.
        let hit = if row < n_rows && t[row] == ts {
            row
        } else {
            t.binary_search(&ts).map_err(|_| {
                ParquetStoreError::new(
                    ParquetStoreErrorKind::Schema,
                    format!("channel {channel_id} has t_us={ts} not present in the union axis (internal bug)"),
                )
            })?
        };
        row = hit;
        f(hit, v);
    }
    Ok(())
}

/// Finishes a [`scatter_into`] buffer pair into a nullable Arrow array,
/// dropping the validity mask when every row is set.
///
/// Dropping it matters for byte parity: `PrimitiveArray::from(Vec<Option<T>>)`
/// — what this function replaced — yields `nulls: None` for a fully dense
/// column, and the writer's def-level encoding follows the null buffer's
/// presence.
fn finish_column<T: ArrowPrimitiveType>(values: Vec<T::Native>, mut valid: BooleanBufferBuilder) -> ArrayRef {
    let n_rows = values.len();
    let mask = valid.finish();
    let nulls = if mask.count_set_bits() == n_rows { None } else { Some(NullBuffer::new(mask)) };
    Arc::new(PrimitiveArray::<T>::new(values.into(), nulls))
}

/// Scatters `c`'s raw samples into a nullable Arrow array of `t.len()`
/// rows, null everywhere `c` did not sample. One arm per [`RawColumn`]
/// variant that round-trips (C1 §2's table) — `Ramp`/`Interp` are never
/// columns (never called with those variants; synthesized channels are
/// excluded from `data.parquet` entirely, see [`write_session_parquet`]).
fn channel_array(c: &Channel, t: &[i64], n_rows: usize) -> Result<ArrayRef, ParquetStoreError> {
    // A row a channel samples twice keeps the later value, which is what
    // the `vals[row] = Some(v)` scatter this replaced did.
    macro_rules! scatter {
        ($ty:ty, $arrow:ty, $data:expr) => {{
            let mut vals: Vec<$ty> = vec![<$ty>::default(); n_rows];
            let mut valid = new_validity(n_rows);
            for_each_row(&c.channel_id, &c.t_us, t, n_rows, $data.iter().copied(), |row, v| {
                vals[row] = v;
                valid.set_bit(row, true);
            })?;
            Ok(finish_column::<$arrow>(vals, valid))
        }};
    }
    match &c.column {
        RawColumn::I16 { data, .. } => scatter!(i16, Int16Type, data),
        RawColumn::I32 { data, .. } => scatter!(i32, Int32Type, data),
        RawColumn::F32 { data, .. } => scatter!(f32, Float32Type, data),
        RawColumn::F64(data) => scatter!(f64, Float64Type, data),
        RawColumn::Ramp { .. } | RawColumn::Interp { .. } => Err(ParquetStoreError::new(
            ParquetStoreErrorKind::Schema,
            format!("{} is a synthesized (Ramp/Interp) column — never written to data.parquet (C1 §2)", c.channel_id),
        )),
    }
}

/// Encodes one whole column into every row group's writer for that column,
/// slicing it at the same `ROW_GROUP_SIZE` boundaries `ArrowWriter` would.
///
/// Slicing an Arrow array is a zero-copy view, so a column is built once
/// and handed to each row group in turn; the caller drops it before
/// building the next one, which is what keeps peak residency at one column
/// (ruling R203.3). `field_index` addresses the leaf writer: `data.parquet`'s
/// schema is flat, so field index and leaf index coincide (the caller
/// checks that invariant once, when the writers are created).
fn feed_column(
    row_groups: &mut [Vec<ArrowColumnWriter>],
    field: &Field,
    field_index: usize,
    array: &ArrayRef,
    n_rows: usize,
) -> Result<(), ParquetStoreError> {
    for (i, writers) in row_groups.iter_mut().enumerate() {
        let offset = i * ROW_GROUP_SIZE;
        let len = (n_rows - offset).min(ROW_GROUP_SIZE);
        let slice = array.slice(offset, len);
        let leaves = compute_leaves(field, &slice)
            .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("compute_leaves: {e}")))?;
        for leaf in &leaves {
            writers[field_index].write(leaf).map_err(|e| {
                ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("write column {}: {e}", field.name()))
            })?;
        }
    }
    Ok(())
}

/// An all-false validity mask of `n_rows` bits, the starting state for
/// every [`scatter_into`] destination.
fn new_validity(n_rows: usize) -> BooleanBufferBuilder {
    let mut valid = BooleanBufferBuilder::new(n_rows);
    valid.append_n(n_rows, false);
    valid
}

/// Builds one `<source>_t_recorded_us` column: the verbatim recorded time
/// (`t_recorded_us_or_t_us()`) at every row any channel of `source_kind`
/// sampled, null elsewhere.
///
/// Merged across every channel of the source, not just the first (L2-R10):
/// FIT/GPX/CSV channels of one `source_kind` do not necessarily share a
/// `t_us` the way idl0's single-FIFO-per-source channels do, so a row only
/// a later channel covers must not read back null. First channel to fill a
/// row wins (C1 §3.2); a later one never overwrites it.
fn recorded_us_array(
    session: &Session,
    source_kind: &str,
    t: &[i64],
    n_rows: usize,
) -> Result<ArrayRef, ParquetStoreError> {
    let mut vals: Vec<i64> = vec![0; n_rows];
    let mut valid = new_validity(n_rows);
    for c in session.channels.iter().filter(|c| c.source_kind == source_kind) {
        // `own` marks the rows *this* channel filled, so a repeated
        // timestamp within one channel keeps its later value while a row an
        // earlier channel already filled is left alone — exactly what the
        // per-channel scatter plus first-fill-wins merge this replaced did.
        // A bitmask, so it costs one bit per row, not eight bytes.
        let mut own = new_validity(n_rows);
        let recorded = c.t_recorded_us_or_t_us();
        for_each_row(&c.channel_id, &c.t_us, t, n_rows, recorded.iter().copied(), |row, v| {
            if !valid.get_bit(row) || own.get_bit(row) {
                vals[row] = v;
                valid.set_bit(row, true);
                own.set_bit(row, true);
            }
        })?;
    }
    Ok(finish_column::<Int64Type>(vals, valid))
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
    write_session_parquet_replacing(data_root, session, importer_version, None)
}

/// [`write_session_parquet`]'s general form: `data.parquet` is write-once
/// for a fresh import (that function always passes `based_on_hash: None`,
/// since nothing exists yet to conflict with), but a rebuild (C3 §3.3
/// `reimport_sessions`) must replace an *existing* file in one rename so a
/// failed rebuild leaves the old file intact (C4 §4's optimistic-concurrency
/// primitive, `write_atomic`) — this is that replace path. `based_on_hash`
/// is the sha256 hex of the `data.parquet` bytes the caller read before
/// deciding to rebuild; a mismatch at rename time (another writer landed in
/// between) surfaces as `AtomicWriteErrorKind::RenameConflict` via
/// [`write_atomic`], leaving the on-disk file untouched.
pub fn write_session_parquet_replacing(
    data_root: &Path,
    session: &Session,
    importer_version: &str,
    based_on_hash: Option<&str>,
) -> Result<PathBuf, ParquetStoreError> {
    let t = union_t_axis(session);
    let n_rows = t.len();

    // Schema first, arrays later: the columns are built and encoded one at
    // a time below (ruling R203.3), so nothing here may hold sample data.
    let mut fields = vec![Field::new("t", DataType::Int64, false)];

    // <source>_t_recorded_us columns — one per distinct source_kind among
    // real (non-synthesized) channels. L2-R10: a source's channels do not
    // all share one `t_us` (a FIT import's per-field channels each only
    // sample the records that carried that field, C1 §4.1), so the column
    // is built from the union of every channel of that source_kind, not
    // just the first one encountered — otherwise a row only a later
    // channel covers would wrongly read back null on `<source>_t_recorded_us`
    // despite `t` having a real row there.
    let mut seen_sources: Vec<&str> = Vec::new();
    for c in &session.channels {
        if c.source_kind == "synthesized" {
            continue; // Time/Distance — excluded (C1 §2), by source_kind not RawColumn variant
        }
        if seen_sources.contains(&c.source_kind.as_str()) {
            continue;
        }
        seen_sources.push(&c.source_kind);
        let field_name = format!("{}_t_recorded_us", c.source_kind);
        fields.push(
            Field::new(field_name.as_str(), DataType::Int64, true)
                .with_metadata([("source_kind".to_string(), c.source_kind.clone())].into_iter().collect()),
        );
    }

    // Channel value columns.
    let value_channels: Vec<&Channel> = session.channels.iter().filter(|c| c.source_kind != "synthesized").collect();
    for c in &value_channels {
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
    }

    let schema = Arc::new(Schema::new(fields));

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

    // One column at a time (ruling R203.3). The high-level `ArrowWriter`
    // takes a whole `RecordBatch`, which means every channel's Arrow array
    // resident at once — a third full-session-sized structure alongside the
    // parsed `Session` and the output buffer, and the shape of the
    // out-of-memory incident this ruling answers. Instead the file is
    // opened through `ArrowWriter` (so the footer, the Arrow schema
    // metadata and every writer property stay exactly what the high-level
    // path would have written) and immediately lowered to its serialized
    // writer, whose per-column writers accept one column at a time. Each
    // channel's Arrow array is built, encoded into every row group's column
    // writer, and dropped before the next channel is built, so peak Arrow
    // residency is one channel rather than all of them.
    //
    // Row groups are cut at exactly `ROW_GROUP_SIZE` rows with a remainder
    // last, which is what `ArrowWriter::write` itself does when
    // `max_row_group_row_count` is set and `max_row_group_bytes` is not
    // (the default) — this loop reproduces that split rather than choosing
    // its own, so the bytes are identical to the previous implementation's.
    let mut buf: Vec<u8> = Vec::new();
    {
        let writer = ArrowWriter::try_new(&mut buf, schema.clone(), Some(props))
            .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("ArrowWriter::try_new: {e}")))?;
        let (mut file_writer, row_group_factory) = writer
            .into_serialized_writer()
            .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("into_serialized_writer: {e}")))?;

        let n_row_groups = n_rows.div_ceil(ROW_GROUP_SIZE);
        let mut row_groups: Vec<Vec<ArrowColumnWriter>> = Vec::with_capacity(n_row_groups);
        for i in 0..n_row_groups {
            let writers = row_group_factory
                .create_column_writers(i)
                .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("create_column_writers: {e}")))?;
            if writers.len() != schema.fields().len() {
                return Err(ParquetStoreError::new(
                    ParquetStoreErrorKind::Schema,
                    format!(
                        "{} leaf column writers for {} schema fields — data.parquet's schema is flat by construction (C1 §4), so they must match",
                        writers.len(),
                        schema.fields().len()
                    ),
                ));
            }
            row_groups.push(writers);
        }

        // Value columns first, then the recorded-time columns, then `t`
        // last so the union axis can be consumed rather than cloned. Column
        // writers are addressed by field index, so feeding order is free.
        for (i, c) in value_channels.iter().enumerate() {
            let field_index = 1 + seen_sources.len() + i;
            let array = channel_array(c, &t, n_rows)?;
            feed_column(&mut row_groups, schema.field(field_index), field_index, &array, n_rows)?;
        }
        for (i, source_kind) in seen_sources.iter().enumerate() {
            let field_index = 1 + i;
            let array = recorded_us_array(session, source_kind, &t, n_rows)?;
            feed_column(&mut row_groups, schema.field(field_index), field_index, &array, n_rows)?;
        }
        let t_array: ArrayRef = Arc::new(Int64Array::from(t));
        feed_column(&mut row_groups, schema.field(0), 0, &t_array, n_rows)?;
        drop(t_array);

        for writers in row_groups {
            let mut row_group = file_writer
                .next_row_group()
                .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("next_row_group: {e}")))?;
            for w in writers {
                w.close()
                    .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("column close: {e}")))?
                    .append_to_row_group(&mut row_group)
                    .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("append_to_row_group: {e}")))?;
            }
            row_group
                .close()
                .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("row_group close: {e}")))?;
        }
        file_writer.close().map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("close: {e}")))?;
    }

    let target = data_root.join("sessions").join(&session.session_id).join("data.parquet");
    write_atomic(data_root, &target, &buf, based_on_hash)
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

/// Extracts the nine C1 §4.3 file-level key-value metadata keys from an
/// already-open [`ParquetRecordBatchReaderBuilder`]'s footer — no additional
/// I/O. The single parser for these keys; both [`read_session_metadata`]
/// (which opens the file itself, for callers that only want metadata) and
/// [`read_session_parquet`] (which needs the builder afterward too, to
/// avoid parsing the footer twice) call this.
fn metadata_from_builder(
    builder: &ParquetRecordBatchReaderBuilder<std::fs::File>,
) -> Result<SessionParquetMetadata, ParquetStoreError> {
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

/// One of C1 §4.3's four `source_format` tokens, parsed, or a
/// [`ParquetStoreErrorKind::Schema`] error naming the token that is not one
/// of them.
///
/// Public because the footer-only readers need the same validation the
/// whole-file read has always done: a caller that reports `source_format`
/// out of [`SessionParquetMetadata`] must reject a corrupt or foreign token
/// rather than pass it through (C1 treats stored metadata as data to
/// validate, not trust).
pub fn parse_source_format(token: &str) -> Result<SourceFormat, ParquetStoreError> {
    match token {
        "idl0" => Ok(SourceFormat::Idl0),
        "fit" => Ok(SourceFormat::Fit),
        "gpx" => Ok(SourceFormat::Gpx),
        "csv" => Ok(SourceFormat::Csv),
        other => Err(ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("unknown source_format {other}"))),
    }
}

/// Reads `data.parquet`'s file-level key-value metadata (C1 §4.3) only — no
/// row-group or column materialization, just the Parquet footer.
pub fn read_session_metadata(path: &Path) -> Result<SessionParquetMetadata, ParquetStoreError> {
    let file = std::fs::File::open(path)
        .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("open {}: {e}", path.display())))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("{}: {e}", path.display())))?;
    metadata_from_builder(&builder)
}

/// Reads `data.parquet` back into a [`Session`] (contract C1 §4.5's read
/// rule: for each channel column, filter to non-null rows, take `t` at
/// those rows as `t_us`, the values as the compact `RawColumn`).
/// Synthesized `Time`/`Distance` are **not** reconstructed here — they are
/// re-derived by `crate::session::synthesis::synthesize_base_channels`
/// after this function returns, exactly as it already runs after parsing.
pub fn read_session_parquet(path: &Path) -> Result<Session, ParquetStoreError> {
    let file = std::fs::File::open(path)
        .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("open {}: {e}", path.display())))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("{}: {e}", path.display())))?;
    let meta = metadata_from_builder(&builder)?;
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

    let mut session = session_shell(meta)?;

    let t_col = batch
        .column_by_name("t")
        .ok_or_else(|| ParquetStoreError::new(ParquetStoreErrorKind::Schema, "missing t column".to_string()))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| ParquetStoreError::new(ParquetStoreErrorKind::Schema, "t column is not Int64".to_string()))?;

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

        session.channels.push(Channel {
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

    Ok(session)
}

/// The [`Session`] `data.parquet`'s file-level C1 §4.3 metadata describes,
/// with **no** channels: identity, provenance and format, nothing decoded.
///
/// Shared by [`read_session_parquet`], which then fills `channels` in, and
/// by [`open_session_lazy`], which leaves them empty and reaches each one
/// through [`read_channel`] on demand (ruling R211.1) — so the two paths
/// cannot drift on how a stored `source_format` token becomes a
/// [`SourceFormat`].
fn session_shell(meta: SessionParquetMetadata) -> Result<Session, ParquetStoreError> {
    let source_format = parse_source_format(&meta.source_format)?;
    // `timestamp_source` is deliberately not among `data.parquet`'s C1 §4.3
    // metadata keys (it lives only in `session.json`, R194) — this read-back
    // path has no recorded provenance to recover, so it approximates from
    // `source_format` the same way importer-absent test fixtures do
    // (`Idl0` -> `Header`, everything else -> `SourceFile`). Nothing today
    // reads this field off a parquet-round-tripped `Session`.
    let timestamp_source = match source_format {
        SourceFormat::Idl0 => crate::session::TimestampSource::Header,
        _ => crate::session::TimestampSource::SourceFile,
    };
    Ok(Session {
        session_id: meta.session_id,
        device_id: meta.device_id,
        timestamp_utc_ms: meta.timestamp_utc_ms,
        timestamp_source,
        config_checksum: meta.config_checksum,
        source_format,
        blob_sha256: meta.blob_sha256,
        channels: Vec::new(),
    })
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
/// The [`ParquetStoreErrorKind::ResourceExhausted`] error for a buffer the
/// allocator refused (ruling R211.4), naming which buffer and how many
/// samples it was for.
fn out_of_memory(what: &str, rows: usize) -> ParquetStoreError {
    ParquetStoreError::new(
        ParquetStoreErrorKind::ResourceExhausted,
        format!("could not allocate {what} for {rows} samples"),
    )
}

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
            // Reserved up front and fallibly (ruling R211.4): these three
            // buffers are the session-scaled allocations of a channel read,
            // and growing them by doubling would both copy more and abort
            // on refusal instead of reporting one.
            let rows = arr.len() - arr.null_count();
            let mut t_us = Vec::new();
            let mut t_recorded_us = Vec::new();
            let mut values = Vec::new();
            t_us.try_reserve_exact(rows).map_err(|_| out_of_memory("t_us", rows))?;
            values.try_reserve_exact(rows).map_err(|_| out_of_memory("values", rows))?;
            if recorded_col.is_some() {
                t_recorded_us.try_reserve_exact(rows).map_err(|_| out_of_memory("t_recorded_us", rows))?;
            }
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

// ---------------------------------------------------------------------------
// Single-channel reads (ruling R203.1)
//
// `data.parquet` is columnar, so a request for one channel reads that
// channel's column plus the `t` axis (and, when the channel has one, its
// `<source>_t_recorded_us` companion) and nothing else. The whole-file
// [`read_session_parquet`] above survives for import verification, export
// and rebuild — the paths that genuinely want every channel.
// ---------------------------------------------------------------------------

/// The synthesized time channel's id (C1 §2). Never a stored column. One
/// definition, in `session::handle`, so the lazy channel index and this
/// reader cannot disagree about the name.
use crate::session::handle::TIME_CHANNEL_ID;
/// The synthesized cumulative-distance channel's id (C1 §2). Never a stored
/// column.
pub(crate) const DISTANCE_CHANNEL_ID: &str = "Distance";
/// The channel `Distance` integrates. Absent → no `Distance` exists.
const GPS_SPEED_CHANNEL_ID: &str = "GPS_SpeedKmh";

/// One stored channel column's identity, read from the file footer alone.
///
/// Everything here comes from `data.parquet`'s schema and row count — no
/// row group is decoded — so building the whole index of a 400 MB session
/// costs one footer read. [`read_channel`] uses it to pick the synthesized
/// `Time` channel's source without decoding every column, and the memory
/// failsafe (ruling R203.4) uses it to size a decode before attempting it.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelColumnInfo {
    /// Column name = `Channel::channel_id`.
    pub channel_id: String,
    /// C1 §4.2 `nominal_rate_hz` metadata. `0.0` for event-driven columns.
    pub nominal_rate_hz: f64,
    /// C1 §4.2 `unit` metadata. Empty means "no unit recorded".
    pub unit: String,
    /// C1 §4.2 `source_kind` metadata (`imu0`, `gps`, `wheel`, …).
    pub source_kind: String,
    /// Bytes one sample occupies once decoded into its compact
    /// [`RawColumn`] variant: 2 (`Int16`), 4 (`Int32`/`Float32`), 8
    /// (`Float64`).
    pub sample_bytes: usize,
    /// Rows in the file. An **upper bound** on this channel's own sample
    /// count — the exact count is the column's non-null rows, which needs a
    /// data read (C1 §4.5).
    pub file_rows: usize,
}

/// Opens `<session_dir>/data.parquet` for reading, or a
/// [`ParquetStoreErrorKind::NotFound`] naming the path when it is absent.
fn open_data_parquet(session_dir: &Path) -> Result<ParquetRecordBatchReaderBuilder<std::fs::File>, ParquetStoreError> {
    let path = session_dir.join("data.parquet");
    let file = std::fs::File::open(&path).map_err(|e| {
        let kind = if e.kind() == std::io::ErrorKind::NotFound {
            ParquetStoreErrorKind::NotFound
        } else {
            ParquetStoreErrorKind::Io
        };
        ParquetStoreError::new(kind, format!("open {}: {e}", path.display()))
    })?;
    ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("{}: {e}", path.display())))
}

/// `true` for the column names that are not channels: the shared `t` axis
/// and every `<source>_t_recorded_us` companion (C1 §3.2).
fn is_axis_column(name: &str) -> bool {
    name == "t" || name.ends_with("_t_recorded_us")
}

/// Bytes per sample of the compact [`RawColumn`] an Arrow column decodes
/// to, or `None` for a type C1 §2 defines no variant for.
fn sample_bytes_of(dt: &DataType) -> Option<usize> {
    match dt {
        DataType::Int16 => Some(2),
        DataType::Int32 => Some(4),
        DataType::Float32 => Some(4),
        DataType::Float64 => Some(8),
        _ => None,
    }
}

/// Every stored channel column in `<session_dir>/data.parquet`, in schema
/// order (the order [`read_session_parquet`] returns them in), read from
/// the footer only. Synthesized `Time`/`Distance` are not included — they
/// are never columns (C1 §2).
pub fn read_channel_index(session_dir: &Path) -> Result<Vec<ChannelColumnInfo>, ParquetStoreError> {
    let builder = open_data_parquet(session_dir)?;
    let file_rows = builder.metadata().file_metadata().num_rows().max(0) as usize;
    let schema = builder.schema().clone();

    let mut out = Vec::new();
    for field in schema.fields() {
        if is_axis_column(field.name()) {
            continue;
        }
        let meta = field.metadata();
        let get = |k: &str| -> Result<String, ParquetStoreError> {
            meta.get(k).cloned().ok_or_else(|| {
                ParquetStoreError::new(
                    ParquetStoreErrorKind::Schema,
                    format!("column {} missing metadata key {k}", field.name()),
                )
            })
        };
        let nominal_rate_hz: f64 = get("nominal_rate_hz")?.parse().map_err(|_| {
            ParquetStoreError::new(
                ParquetStoreErrorKind::Schema,
                format!("column {}: bad nominal_rate_hz", field.name()),
            )
        })?;
        let sample_bytes = sample_bytes_of(field.data_type()).ok_or_else(|| {
            ParquetStoreError::new(
                ParquetStoreErrorKind::Schema,
                format!("unsupported column type {:?} on {}", field.data_type(), field.name()),
            )
        })?;
        out.push(ChannelColumnInfo {
            channel_id: field.name().clone(),
            nominal_rate_hz,
            unit: get("unit")?,
            source_kind: get("source_kind")?,
            sample_bytes,
            file_rows,
        });
    }
    Ok(out)
}

/// Reads only the named columns of `<session_dir>/data.parquet` into one
/// concatenated [`RecordBatch`].
///
/// `wanted` is matched against top-level schema fields by name; a name the
/// file does not carry is skipped silently (callers check existence first).
/// The projection is what makes a single-channel read cheap — parquet
/// decodes only the requested column chunks, so peak bytes scale with the
/// requested channels, not with the session.
///
/// **`t` is not implied.** A caller that gathers samples needs it and names
/// it; a caller decoding one axis column on its own (ruling R232.1's shared
/// axis) must not pay for it, and an unconditional `t` would have made the
/// whole shared-axis read pointless.
fn read_projected_batch(session_dir: &Path, wanted: &[&str]) -> Result<RecordBatch, ParquetStoreError> {
    read_projected_batch_with_progress(session_dir, wanted, &mut |_, _| {})
}

/// [`read_projected_batch`], reporting `(rows decoded so far, rows in the
/// file)` to `on_progress` after every [`RecordBatch`] the reader yields
/// (ruling R221 item 1).
///
/// The callback is the only difference: the read itself, its projection and
/// its errors are identical, so the progress-reporting and the silent path
/// cannot decode differently. Reporting happens **during** the streaming
/// read only — the `concat_batches` that follows, and the gather
/// [`channel_samples_from_batch`] does after it, are single steps with no
/// intermediate observation, so a caller sees the count reach the file's row
/// count slightly before the decode returns.
fn read_projected_batch_with_progress(
    session_dir: &Path,
    wanted: &[&str],
    on_progress: &mut dyn FnMut(usize, usize),
) -> Result<RecordBatch, ParquetStoreError> {
    let builder = open_data_parquet(session_dir)?;
    let total_rows = builder.metadata().file_metadata().num_rows().max(0) as usize;
    let schema = builder.schema().clone();
    let roots: Vec<usize> = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, f)| wanted.contains(&f.name().as_str()))
        .map(|(i, _)| i)
        .collect();
    let mask = ProjectionMask::roots(builder.parquet_schema(), roots);
    let reader = builder
        .with_projection(mask)
        .build()
        .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("build reader: {e}")))?;
    let projected_schema = reader.schema();
    let mut batches: Vec<RecordBatch> = Vec::new();
    let mut done_rows = 0usize;
    for batch in reader {
        let batch = batch.map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("read batches: {e}")))?;
        done_rows += batch.num_rows();
        batches.push(batch);
        on_progress(done_rows.min(total_rows), total_rows);
    }
    arrow::compute::concat_batches(&projected_schema, &batches)
        .map_err(|e| ParquetStoreError::new(ParquetStoreErrorKind::Io, format!("concat_batches: {e}")))
}

/// `batch`'s `t` column, or a [`ParquetStoreErrorKind::Schema`] error.
fn t_column_of(batch: &RecordBatch) -> Result<&Int64Array, ParquetStoreError> {
    batch
        .column_by_name("t")
        .ok_or_else(|| ParquetStoreError::new(ParquetStoreErrorKind::Schema, "missing t column".to_string()))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| ParquetStoreError::new(ParquetStoreErrorKind::Schema, "t column is not Int64".to_string()))
}

/// Decodes one stored column of `batch` into a [`ChannelSamples`], applying
/// C1 §4.5's read rule via the same [`read_column`] the whole-file reader
/// uses — the two paths share that gather so they cannot drift.
fn channel_samples_from_batch(batch: &RecordBatch, name: &str) -> Result<ChannelSamples, ParquetStoreError> {
    let schema = batch.schema();
    let (idx, field) = schema
        .fields()
        .iter()
        .enumerate()
        .find(|(_, f)| f.name() == name)
        .ok_or_else(|| ParquetStoreError::new(ParquetStoreErrorKind::NotFound, format!("column {name} not projected")))?;
    let meta = field.metadata();
    let get = |k: &str| -> Result<String, ParquetStoreError> {
        meta.get(k).cloned().ok_or_else(|| {
            ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("column {name} missing metadata key {k}"))
        })
    };
    let nominal_rate_hz: f64 = get("nominal_rate_hz")?
        .parse()
        .map_err(|_| ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("column {name}: bad nominal_rate_hz")))?;
    let unit = get("unit")?;
    let source_kind = get("source_kind")?;
    let gaps = match meta.get("gaps") {
        Some(g) => gaps_from_json(g)?,
        None => Vec::new(),
    };

    let t_col = t_column_of(batch)?;
    let recorded_col = batch
        .column_by_name(&recorded_axis_column(&source_kind))
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
    let (t_us, t_recorded_us, column) = read_column(batch.column(idx), t_col, recorded_col, meta)?;

    Ok(ChannelSamples::from_channel(Channel {
        channel_id: name.to_string(),
        t_us,
        t_recorded_us,
        nominal_rate_hz,
        column,
        source_kind,
        unit,
        gaps,
    }))
}

/// Non-null row count of `batch`'s column `name` — the channel's own sample
/// count under C1 §4.5's read rule, without gathering its values.
fn non_null_len(batch: &RecordBatch, name: &str) -> usize {
    match batch.column_by_name(name) {
        Some(col) => col.len() - col.null_count(),
        None => 0,
    }
}

/// Picks the channel `session::synthesis::synthesize_base_channels` would
/// build `Time` from, without decoding the whole file.
///
/// Reproduces that function's rule exactly: the highest positive
/// `nominal_rate_hz` wins; among channels at that rate the longest wins,
/// earliest in schema order breaking ties; when no channel declares a
/// positive rate, the longest channel of all wins (ledger R23 Q2, rate
/// `0.0`). Returns `(source channel id, rate, length)`, or `None` when the
/// file has no channel with samples — the case where synthesis appends no
/// `Time` at all.
///
/// "Longest" is measured the way the reader measures it: a channel's own
/// sample count is its **non-null rows** (C1 §4.5's read rule), which is
/// what `Channel::len()` reports on a `Session` that came back out of
/// `read_session_parquet`. A channel whose recorded timestamps repeat
/// therefore counts once per distinct timestamp on both sides of the
/// comparison — see the duplicate-timestamp election test below, which
/// pins this against `read_session_parquet` + synthesis.
fn time_source(
    session_dir: &Path,
    index: &[ChannelColumnInfo],
) -> Result<Option<(String, f64, usize)>, ParquetStoreError> {
    let max_rate = index.iter().map(|c| c.nominal_rate_hz).filter(|r| *r > 0.0).fold(0.0_f64, f64::max);
    let candidates: Vec<&ChannelColumnInfo> = if max_rate > 0.0 {
        index.iter().filter(|c| c.nominal_rate_hz == max_rate).collect()
    } else {
        index.iter().collect()
    };
    if candidates.is_empty() {
        return Ok(None);
    }
    // Only the candidates' own columns: this read counts non-null rows and
    // never gathers a sample, so it does not need `t` (which
    // `read_projected_batch` no longer implies).
    let names: Vec<&str> = candidates.iter().map(|c| c.channel_id.as_str()).collect();
    let batch = read_projected_batch(session_dir, &names)?;

    let mut best: Option<(String, usize)> = None;
    for c in &candidates {
        let len = non_null_len(&batch, &c.channel_id);
        if best.as_ref().map_or(true, |(_, blen)| len > *blen) {
            best = Some((c.channel_id.clone(), len));
        }
    }
    match best {
        None | Some((_, 0)) => Ok(None),
        Some((id, len)) => Ok(Some((id, max_rate, len))),
    }
}

/// One decoded timestamp column of `data.parquet` — the union `t` axis, or
/// a `<source>_t_recorded_us` companion — held apart from any one channel so
/// every channel that reads through it decompresses it once (ruling R232.1).
///
/// Before this type each channel decode projected its own copy of `t` and of
/// its source's recorded axis, so six `imu0` channels decompressed the same
/// two i64 columns six times over a 516 MB file. The column is opaque on
/// purpose: callers (the app's `SessionCache`) hold it, size it and hand it
/// back to [`read_channel_sharing_axes`], but never read a timestamp out of it,
/// so no Arrow type crosses out of this crate.
///
/// **Row-indexed, not sample-indexed**: the array has one entry per row of
/// the file, nulls included, exactly as the reader yields it. That is what
/// makes it shareable — a channel selects its own samples out of it with its
/// own validity mask.
#[derive(Debug, Clone)]
pub struct AxisColumn {
    /// The column as read, one entry per file row.
    values: Int64Array,
}

impl AxisColumn {
    /// Rows in the column — the file's row count, not any channel's sample
    /// count.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// `true` when the file has no rows.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Heap bytes this column occupies, for a byte-budgeted cache's
    /// accounting. Arrow's own figure for the backing buffers, so a shared
    /// or sliced buffer is counted as the allocation it really is.
    pub fn resident_bytes(&self) -> usize {
        self.values.get_array_memory_size()
    }
}

/// The already-decoded timestamp columns a channel decode may borrow
/// instead of reading again (ruling R232.1).
///
/// `t` is the union axis, one per session; `recorded` is the channel's
/// `<source>_t_recorded_us` companion, one per source kind. `None` means
/// "not held -- read it", and whatever is read comes back in
/// [`DecodedChannel`] for the caller to keep.
#[derive(Debug, Clone, Copy, Default)]
pub struct BorrowedAxes<'a> {
    /// The shared union time axis (column `t`, C1 3.2), when already held.
    pub t: Option<&'a AxisColumn>,
    /// The channel's source's recorded-time companion, when already held.
    pub recorded: Option<&'a AxisColumn>,
}

/// One channel's samples, plus whichever timestamp axes had to be read to
/// produce them (ruling R232.1).
///
/// The axes are handed back rather than thrown away because they were
/// decompressed in the same pass as the channel: a caching caller keeps
/// them, and every later channel of that session and source borrows them
/// instead of paying again.
#[derive(Debug)]
pub struct DecodedChannel {
    /// The channel, field-for-field what [`read_channel`] returns.
    pub samples: ChannelSamples,
    /// The union `t` axis, when this read is the one that decoded it.
    pub t: Option<AxisColumn>,
    /// The source's recorded-time companion, when this read decoded it.
    pub recorded: Option<AxisColumn>,
}

/// The `<source>_t_recorded_us` column name for `source_kind` (C1 3.2) --
/// the one place that spelling is built, so a cache's axis keys and the
/// reader's projection cannot drift apart.
pub fn recorded_axis_column(source_kind: &str) -> String {
    format!("{source_kind}_t_recorded_us")
}

/// [`read_channel_with_progress`], borrowing the timestamp axes the caller
/// already holds and handing back the ones it had to read (ruling R232.1).
///
/// **One pass over the file, whatever is borrowed.** The projection is the
/// channel's own column plus only the axes `have` does not supply, so this
/// never costs more than [`read_channel_with_progress`] and costs
/// substantially less once a session's axes are held: six `imu0` channels
/// of a 516 MB file used to decompress the union `t` and the `imu0`
/// recorded companion six times each, which is the cost ruling R232 exists
/// to remove.
///
/// A borrowed axis must be the same file's, and is row-indexed over the
/// whole file; one of a different length is
/// [`ParquetStoreErrorKind::Schema`] rather than a silently wrong answer.
///
/// A synthesized `Time`/`Distance` has no stored column to project and
/// borrows nothing: it falls through to [`read_channel_with_progress`],
/// which owns the answer (and the error message) for both, as it does for a
/// name this file does not carry at all.
pub fn read_channel_sharing_axes(
    session_dir: &Path,
    channel: &str,
    have: BorrowedAxes<'_>,
    on_progress: &mut dyn FnMut(usize, usize),
) -> Result<DecodedChannel, ParquetStoreError> {
    let index = read_channel_index(session_dir)?;
    let Some(info) = index.iter().find(|c| c.channel_id == channel) else {
        let samples = read_channel_with_progress(session_dir, channel, on_progress)?;
        return Ok(DecodedChannel { samples, t: None, recorded: None });
    };
    let recorded_name = recorded_axis_column(&info.source_kind);

    let mut wanted: Vec<&str> = vec![channel];
    if have.t.is_none() {
        wanted.push("t");
    }
    if have.recorded.is_none() {
        wanted.push(recorded_name.as_str());
    }
    let batch = read_projected_batch_with_progress(session_dir, &wanted, on_progress)?;

    let column = batch
        .column_by_name(channel)
        .ok_or_else(|| ParquetStoreError::new(ParquetStoreErrorKind::NotFound, format!("column {channel} not projected")))?
        .clone();
    let rows = column.len();

    // Each axis is either the caller's -- checked against this file's row
    // count, since a borrowed axis is only usable on the file it came from
    // -- or this batch's, which is correct by construction.
    let read_t = match have.t {
        Some(_) => None,
        None => Some(axis_from_batch(&batch, "t")?),
    };
    let read_recorded = match have.recorded {
        Some(_) => None,
        None => match batch.column_by_name(&recorded_name) {
            Some(_) => Some(axis_from_batch(&batch, &recorded_name)?),
            None => None,
        },
    };
    let t = match (have.t, read_t.as_ref()) {
        (Some(held), _) => check_axis_rows(held, "t", rows)?,
        (None, Some(fresh)) => fresh,
        (None, None) => {
            return Err(ParquetStoreError::new(ParquetStoreErrorKind::Schema, "missing t column".to_string()))
        }
    };
    let recorded = match (have.recorded, read_recorded.as_ref()) {
        (Some(held), _) => Some(check_axis_rows(held, &recorded_name, rows)?),
        (None, Some(fresh)) => Some(fresh),
        (None, None) => None,
    };

    let schema = batch.schema();
    let field = schema
        .fields()
        .iter()
        .find(|f| f.name() == channel)
        .ok_or_else(|| ParquetStoreError::new(ParquetStoreErrorKind::NotFound, format!("column {channel} not projected")))?
        .clone();
    let meta = field.metadata();
    let get = |k: &str| -> Result<String, ParquetStoreError> {
        meta.get(k).cloned().ok_or_else(|| {
            ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("column {channel} missing metadata key {k}"))
        })
    };
    let nominal_rate_hz: f64 = get("nominal_rate_hz")?
        .parse()
        .map_err(|_| ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("column {channel}: bad nominal_rate_hz")))?;
    let unit = get("unit")?;
    let source_kind = get("source_kind")?;
    let gaps = match meta.get("gaps") {
        Some(g) => gaps_from_json(g)?,
        None => Vec::new(),
    };

    let (t_us, t_recorded_us, raw) = read_column(&column, &t.values, recorded.map(|a| &a.values), meta)?;

    Ok(DecodedChannel {
        samples: ChannelSamples::from_channel(Channel {
            channel_id: channel.to_string(),
            t_us,
            t_recorded_us,
            nominal_rate_hz,
            column: raw,
            source_kind,
            unit,
            gaps,
        }),
        t: read_t,
        recorded: read_recorded,
    })
}

/// One Int64 column of `batch` as an [`AxisColumn`].
fn axis_from_batch(batch: &RecordBatch, name: &str) -> Result<AxisColumn, ParquetStoreError> {
    let values = batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| {
            ParquetStoreError::new(ParquetStoreErrorKind::Schema, format!("axis column {name} is missing or not Int64"))
        })?
        .clone();
    Ok(AxisColumn { values })
}

/// Every axis is row-indexed over the whole file, so a borrowed one of a
/// different length came from a different file -- a caller error, reported
/// rather than gathered against.
fn check_axis_rows<'a>(axis: &'a AxisColumn, name: &str, rows: usize) -> Result<&'a AxisColumn, ParquetStoreError> {
    if axis.len() == rows {
        Ok(axis)
    } else {
        Err(ParquetStoreError::new(
            ParquetStoreErrorKind::Schema,
            format!("borrowed {name} has {} rows, this file has {rows}", axis.len()),
        ))
    }
}

/// Reads one channel of `<session_dir>/data.parquet` (ruling R203.1).
///
/// Decodes that channel's column, the shared `t` axis and — when the
/// channel has one — its `<source>_t_recorded_us` companion, and nothing
/// else: peak bytes scale with the one channel, not the session. The result
/// is field-for-field what [`read_session_parquet`] followed by
/// `session::synthesis::synthesize_base_channels` produces for the same
/// name, including the two synthesized channels:
///
/// - `Time` is rebuilt from its source channel's own recorded `t_us` (C1
///   §3.5 invariant 4 forbids `i / rate`), reading only the columns that
///   could win the source election.
/// - `Distance` integrates `GPS_SpeedKmh`, reading that column and the
///   `Time` source's.
///
/// A name that is neither a stored column nor one of those two is
/// [`ParquetStoreErrorKind::NotFound`], as is a `Time`/`Distance` request
/// against a session synthesis would not have given one to (no samples, or
/// no usable `GPS_SpeedKmh`).
pub fn read_channel(session_dir: &Path, channel: &str) -> Result<ChannelSamples, ParquetStoreError> {
    read_channel_with_progress(session_dir, channel, &mut |_, _| {})
}

/// [`read_channel`], reporting `(rows decoded so far, rows this decode has
/// to get through)` to `on_progress` as it streams (ruling R221 item 1).
///
/// `read_channel` is this function with a callback that does nothing, so the
/// two can never decode differently.
///
/// **The count is monotonic over the whole decode, not per column read.** A
/// stored channel is one pass over the file, so the total is the file's row
/// count. `Time` is also one pass (its source channel's). `Distance` is
/// two — its `Time` source and `GPS_SpeedKmh` — so its total is twice the
/// file's row count and its second pass continues where the first stopped,
/// rather than sending a ring back to zero halfway through.
///
/// Nothing is reported for the integration and interpolation `Distance` does
/// after its reads, nor for the per-sample gather every channel ends with:
/// they are single steps with no intermediate observation. A caller that
/// draws the fraction therefore sees it reach 1 shortly before the decode
/// returns, never after.
pub fn read_channel_with_progress(
    session_dir: &Path,
    channel: &str,
    on_progress: &mut dyn FnMut(usize, usize),
) -> Result<ChannelSamples, ParquetStoreError> {
    let index = read_channel_index(session_dir)?;
    let file_rows = index.first().map_or(0, |c| c.file_rows);

    if let Some(info) = index.iter().find(|c| c.channel_id == channel) {
        let recorded = recorded_axis_column(&info.source_kind);
        let batch = read_projected_batch_with_progress(session_dir, &["t", channel, recorded.as_str()], on_progress)?;
        return channel_samples_from_batch(&batch, channel);
    }

    if channel != TIME_CHANNEL_ID && channel != DISTANCE_CHANNEL_ID {
        return Err(ParquetStoreError::new(
            ParquetStoreErrorKind::NotFound,
            format!("channel '{channel}' is not a column of {}", session_dir.join("data.parquet").display()),
        ));
    }

    let no_time = || {
        ParquetStoreError::new(
            ParquetStoreErrorKind::NotFound,
            format!("session at {} has no channel with samples, so no '{channel}' is synthesized", session_dir.display()),
        )
    };
    let (source_id, max_rate, max_rate_len) = time_source(session_dir, &index)?.ok_or_else(no_time)?;

    // How many passes over the file this decode makes, so the reported
    // fraction is monotonic across them (see this function's doc comment).
    let passes = if channel == TIME_CHANNEL_ID { 1 } else { 2 };

    // The source channel's own recorded time axis — `Time`'s values are it,
    // in seconds, and `Distance` is presented on it (they share one axis by
    // construction, exactly as synthesis builds them).
    let source = {
        let mut first = |done: usize, total: usize| on_progress(done, total * passes);
        read_channel_with_progress(session_dir, &source_id, &mut first)?
    };
    let time_t_us = source.t_us.clone();
    drop(source);

    if channel == TIME_CHANNEL_ID {
        let values: Vec<f64> = time_t_us.iter().map(|&t| t as f64 / 1_000_000.0).collect();
        return Ok(ChannelSamples {
            channel_id: TIME_CHANNEL_ID.to_string(),
            t_us: time_t_us,
            t_recorded_us: None,
            nominal_rate_hz: max_rate,
            column: RawColumn::F64(values),
            source_kind: "synthesized".to_string(),
            unit: "s".to_string(),
            gaps: Vec::new(),
        });
    }

    // `Distance`: cumulative metres from `GPS_SpeedKmh`, held at the GPS
    // rate and interpolated onto the `Time` grid on demand — the same
    // `RawColumn::Interp` synthesis builds, so values are bit-identical.
    let speed_absent = || {
        ParquetStoreError::new(
            ParquetStoreErrorKind::NotFound,
            format!("session at {} has no usable {GPS_SPEED_CHANNEL_ID}, so no 'Distance' is synthesized", session_dir.display()),
        )
    };
    let base_rate = index
        .iter()
        .find(|c| c.channel_id == GPS_SPEED_CHANNEL_ID && c.nominal_rate_hz > 0.0)
        .ok_or_else(speed_absent)?
        .nominal_rate_hz;
    let speed = {
        let mut second = |done: usize, total: usize| on_progress(file_rows + done, total * passes);
        read_channel_with_progress(session_dir, GPS_SPEED_CHANNEL_ID, &mut second)?
    };
    if speed.is_empty() {
        return Err(speed_absent());
    }
    let ms: Vec<f64> = speed.materialize().into_iter().map(|s| s / 3.6).collect();
    let base = crate::integration::integrate(&ms, base_rate);
    Ok(ChannelSamples {
        channel_id: DISTANCE_CHANNEL_ID.to_string(),
        t_us: time_t_us,
        t_recorded_us: None,
        nominal_rate_hz: max_rate,
        column: RawColumn::Interp { base, base_rate, out_rate: max_rate, len: max_rate_len },
        source_kind: "synthesized".to_string(),
        unit: "m".to_string(),
        gaps: Vec::new(),
    })
}

/// Peak heap bytes [`read_channel`] needs for `channel`, from the footer
/// alone (ruling R203.4's estimate).
///
/// Counts, per row of the file: the channel's Arrow column and the `Vec`
/// gathered out of it (`sample_bytes` each), the gathered `t_us` (8 B), the
/// `t` axis (8 B) and an assumed `<source>_t_recorded_us` companion (8 B).
/// It is a **ceiling** — it charges every row to a channel that may only
/// sample some of them, and charges a recorded axis that may collapse to
/// `None` — because a failsafe that under-estimates aborts the process it
/// exists to protect. Synthesized `Time`/`Distance` are sized as the widest
/// stored column (their source is one of them) plus an f64 output.
pub fn estimate_channel_bytes(session_dir: &Path, channel: &str) -> Result<u64, ParquetStoreError> {
    let index = read_channel_index(session_dir)?;
    let rows = index.first().map_or(0, |c| c.file_rows) as u64;
    let per_row = match index.iter().find(|c| c.channel_id == channel) {
        Some(info) => info.sample_bytes as u64 * 2 + 24,
        None if channel == TIME_CHANNEL_ID || channel == DISTANCE_CHANNEL_ID => {
            index.iter().map(|c| c.sample_bytes as u64).max().unwrap_or(8) * 2 + 32
        }
        None => {
            return Err(ParquetStoreError::new(
                ParquetStoreErrorKind::NotFound,
                format!("channel '{channel}' is not a column of {}", session_dir.join("data.parquet").display()),
            ))
        }
    };
    Ok(rows.saturating_mul(per_row))
}

/// Peak heap bytes [`read_session_parquet`] needs for the whole file, from
/// the footer alone (ruling R203.4's estimate).
///
/// The same per-row ceiling as [`estimate_channel_bytes`], summed over
/// every stored channel, plus the shared `t` axis once. Whole-session loads
/// are what the incident this ruling answers ran out of memory on, so the
/// estimate is deliberately generous rather than tight.
pub fn estimate_session_bytes(session_dir: &Path) -> Result<u64, ParquetStoreError> {
    let index = read_channel_index(session_dir)?;
    let rows = index.first().map_or(0, |c| c.file_rows) as u64;
    let per_row: u64 = index.iter().map(|c| c.sample_bytes as u64 * 2 + 16).sum::<u64>() + 8;
    Ok(rows.saturating_mul(per_row))
}

/// The session's recorded time span, microseconds, as
/// `Some((first_us, last_us))` — the smallest and largest value of the
/// shared `t` axis — or `None` when the file has no rows (ruling R211.1).
///
/// `t` is the **union** axis: every channel's timestamps appear in it, so
/// its minimum is the earliest first sample of any channel and its maximum
/// the latest last sample of any channel. That is exactly what a
/// whole-session span is, which is why it can be read without decoding a
/// single channel — this replaces the whole-session decode the app used to
/// do on every window resolution.
///
/// Read from `t`'s column-chunk statistics, which `write_session_parquet`
/// enables explicitly for that column; a file whose chunks carry none falls
/// back to reading the `t` column alone (8 B/row, no channel data).
pub fn read_time_axis_bounds(session_dir: &Path) -> Result<Option<(i64, i64)>, ParquetStoreError> {
    let builder = open_data_parquet(session_dir)?;
    let metadata = builder.metadata().clone();
    drop(builder);

    let mut bounds: Option<(i64, i64)> = None;
    let mut have_stats = true;
    for group in metadata.row_groups() {
        if group.num_rows() == 0 {
            continue;
        }
        let Some(chunk) = group.columns().iter().find(|c| c.column_path().string() == "t") else {
            have_stats = false;
            break;
        };
        let stats = match chunk.statistics() {
            Some(parquet::file::statistics::Statistics::Int64(s)) => s.clone(),
            _ => {
                have_stats = false;
                break;
            }
        };
        let (Some(&min), Some(&max)) = (stats.min_opt(), stats.max_opt()) else {
            have_stats = false;
            break;
        };
        bounds = Some(match bounds {
            Some((lo, hi)) => (lo.min(min), hi.max(max)),
            None => (min, max),
        });
    }
    if have_stats {
        return Ok(bounds);
    }

    let batch = read_projected_batch(session_dir, &["t"])?;
    let t = t_column_of(&batch)?;
    let mut out: Option<(i64, i64)> = None;
    for i in 0..t.len() {
        if t.is_null(i) {
            continue;
        }
        let v = t.value(i);
        out = Some(match out {
            Some((lo, hi)) => (lo.min(v), hi.max(v)),
            None => (v, v),
        });
    }
    Ok(out)
}

/// Every stored channel's **exact** sample count — its non-null rows, C1
/// §4.5's read rule — without decoding any samples (ruling R211.1).
///
/// Read from the row groups' column-chunk statistics in the footer, which
/// `write_session_parquet` leaves at the writer's default (per-page
/// statistics, which include the chunk null count). A column whose chunks
/// carry no null count falls back to decoding that column and counting —
/// correct on any file, whatever wrote it, at the cost of one column read;
/// the fallback is per column, so one statistics-less column never widens
/// into a whole-session decode.
///
/// Returned in schema order, matching [`read_channel_index`]. Synthesized
/// `Time`/`Distance` are not included — they are never columns (C1 §2).
pub fn read_channel_sample_counts(session_dir: &Path) -> Result<Vec<(String, usize)>, ParquetStoreError> {
    let builder = open_data_parquet(session_dir)?;
    let metadata = builder.metadata().clone();
    let schema = builder.schema().clone();
    drop(builder);

    let mut out = Vec::new();
    let mut needs_decode: Vec<String> = Vec::new();
    for field in schema.fields() {
        if is_axis_column(field.name()) {
            continue;
        }
        let mut rows = 0usize;
        let mut nulls = 0usize;
        let mut have_stats = true;
        for group in metadata.row_groups() {
            let Some(chunk) = group.columns().iter().find(|c| c.column_path().string() == *field.name()) else {
                have_stats = false;
                break;
            };
            let Some(null_count) = chunk.statistics().and_then(|s| s.null_count_opt()) else {
                have_stats = false;
                break;
            };
            rows += group.num_rows().max(0) as usize;
            nulls += null_count as usize;
        }
        if have_stats {
            out.push((field.name().clone(), rows.saturating_sub(nulls)));
        } else {
            needs_decode.push(field.name().clone());
            out.push((field.name().clone(), 0));
        }
    }

    for name in needs_decode {
        let batch = read_projected_batch(session_dir, &[name.as_str()])?;
        let exact = non_null_len(&batch, &name);
        if let Some(slot) = out.iter_mut().find(|(n, _)| *n == name) {
            slot.1 = exact;
        }
    }
    Ok(out)
}

/// A [`ChannelSource`] over one session directory: every lookup is a
/// [`read_channel`] of that one column (ruling R211.1).
///
/// Nothing is cached here — a lazy [`SessionHandle`] remembers what it has
/// already asked for, and the app's `SessionCache` (`idl-rs-tauri`) is the
/// cross-request, byte-budgeted cache. This type is the plain, unbudgeted
/// source the CLI and the import pipeline use.
#[derive(Debug, Clone)]
pub struct ParquetChannelSource {
    session_dir: PathBuf,
}

impl ParquetChannelSource {
    /// A source over `<data>/sessions/<session_id>` (C4 §2).
    pub fn new(session_dir: impl Into<PathBuf>) -> Self {
        ParquetChannelSource { session_dir: session_dir.into() }
    }
}

impl ChannelSource for ParquetChannelSource {
    /// One channel, or `None` when the file has no such channel **or** the
    /// read failed. [`ChannelSource`] has no error channel by design (see
    /// its docs): a failed decode degrades that one channel to absent
    /// rather than failing a whole import or evaluation.
    fn channel(&self, channel_id: &str) -> Option<Channel> {
        read_channel(&self.session_dir, channel_id).ok().map(ChannelSamples::into_channel)
    }
}

/// Every channel a lazy [`SessionHandle`] over `session_dir` can serve,
/// from the file footer alone — the stored columns plus the synthesized
/// `Time`/`Distance` (ruling R211.1).
///
/// The two synthesized entries are listed on the same conditions
/// `session::synthesis::synthesize_base_channels` appends them, as far as
/// the footer can tell: `Time` whenever the file has a channel column and
/// at least one row, `Distance` additionally when `GPS_SpeedKmh` is stored
/// at a positive rate. The footer cannot see that every row of a column is
/// null, so a listed entry can still resolve to `None` at decode time —
/// [`SessionHandle::with_channel`] treats that exactly as an absent
/// channel, which is what synthesis would have produced anyway.
///
/// [`LazyChannelInfo::max_length`] is the file's row count for every entry:
/// a per-channel sample count is its non-null rows, which the footer does
/// not carry (C1 §4.5).
pub fn read_lazy_channel_index(session_dir: &Path) -> Result<Vec<LazyChannelInfo>, ParquetStoreError> {
    let index = read_channel_index(session_dir)?;
    let rows = index.first().map_or(0, |c| c.file_rows);
    let max_rate = index.iter().map(|c| c.nominal_rate_hz).filter(|r| *r > 0.0).fold(0.0_f64, f64::max);

    let mut out: Vec<LazyChannelInfo> = index
        .iter()
        .map(|c| LazyChannelInfo {
            channel_id: c.channel_id.clone(),
            nominal_rate_hz: c.nominal_rate_hz,
            unit: c.unit.clone(),
            max_length: rows,
            synthesized: false,
        })
        .collect();

    if !index.is_empty() && rows > 0 {
        // `Time`/`Distance` carry the elected source channel's rate, exactly
        // as synthesis gives them (`0.0` when no channel declares a positive
        // rate — ledger R23 Q2 — never a fabricated one).
        out.push(LazyChannelInfo {
            channel_id: TIME_CHANNEL_ID.to_string(),
            nominal_rate_hz: max_rate,
            unit: "s".to_string(),
            max_length: rows,
            synthesized: true,
        });
        if index.iter().any(|c| c.channel_id == GPS_SPEED_CHANNEL_ID && c.nominal_rate_hz > 0.0) {
            out.push(LazyChannelInfo {
                channel_id: DISTANCE_CHANNEL_ID.to_string(),
                nominal_rate_hz: max_rate,
                unit: "m".to_string(),
                max_length: rows,
                synthesized: true,
            });
        }
    }
    Ok(out)
}

/// A [`SessionHandle`] over `<session_dir>/data.parquet` that decodes
/// **nothing** until a channel is asked for, then decodes that channel
/// alone (ruling R211.1).
///
/// The replacement for `read_session_parquet` + `SessionHandle::
/// from_session` everywhere a caller wants a handle rather than a whole
/// `Session`: lap indexing after an import, a rescan, and the app's
/// workbook evaluator (which supplies its own byte-budgeted
/// [`ChannelSource`] instead of [`ParquetChannelSource`]). Costs two footer
/// reads and no row-group decode.
pub fn open_session_lazy(session_dir: &Path) -> Result<SessionHandle, ParquetStoreError> {
    open_session_lazy_with(session_dir, Arc::new(ParquetChannelSource::new(session_dir)))
}

/// [`open_session_lazy`] with a caller-supplied [`ChannelSource`] in place
/// of the plain [`ParquetChannelSource`].
///
/// The app passes its byte-budgeted `SessionCache` here (ruling R211.2), so
/// the evaluator's decodes queue behind the same budget as every other
/// command's rather than each passing a check none of them can honour
/// together. `source` is trusted to serve the same channels the index
/// lists, out of the same file.
pub fn open_session_lazy_with(
    session_dir: &Path,
    source: Arc<dyn ChannelSource>,
) -> Result<SessionHandle, ParquetStoreError> {
    let meta = read_session_metadata(&session_dir.join("data.parquet"))?;
    let index = read_lazy_channel_index(session_dir)?;
    let session = session_shell(meta)?;
    Ok(SessionHandle::lazy(session, index, source))
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
    pub(super) fn sample_session() -> Session {
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
            timestamp_source: crate::session::TimestampSource::Header,
            config_checksum: Some("cafebabe".to_string()),
            source_format: SourceFormat::Idl0,
            blob_sha256: "0".repeat(64),
            channels: vec![imu, gps],
        }
    }

    /// The axes of `sample_session`'s `imu0` source, decoded the way a
    /// caching caller decodes them once and then shares them: one ordinary
    /// read that hands its axes back (ruling R232.1).
    fn imu0_axes(dir: &Path) -> (AxisColumn, Option<AxisColumn>) {
        let first = read_channel_sharing_axes(dir, "IMU0_AccelX", BorrowedAxes::default(), &mut |_, _| {}).unwrap();
        (first.t.expect("the first read decodes the union axis"), first.recorded)
    }

    #[test]
    fn read_channel_sharing_axes_the_first_read_hands_back_one_axis_entry_per_file_row() {
        // Arrange
        let root = temp_root();
        let path = write_session_parquet(&root, &sample_session(), "0.1.0").unwrap();
        let dir = path.parent().unwrap();

        // Act
        let first = read_channel_sharing_axes(dir, "IMU0_AccelX", BorrowedAxes::default(), &mut |_, _| {}).unwrap();

        // Assert -- row-indexed over the union axis (0, 1250, 2500, 5000),
        // not over either channel's own sample count.
        let t = first.t.unwrap();
        assert_eq!(t.len(), 4);
        assert!(t.resident_bytes() > 0);
        assert_eq!(first.recorded.unwrap().len(), 4);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_channel_sharing_axes_a_read_that_borrows_both_axes_reads_neither_again() {
        // Arrange
        let root = temp_root();
        let path = write_session_parquet(&root, &sample_session(), "0.1.0").unwrap();
        let dir = path.parent().unwrap();
        let (t, recorded) = imu0_axes(dir);

        // Act
        let second =
            read_channel_sharing_axes(dir, "IMU0_AccelX", BorrowedAxes { t: Some(&t), recorded: recorded.as_ref() }, &mut |_, _| {})
                .unwrap();

        // Assert -- nothing handed back means nothing was decoded twice,
        // which is exactly what ruling R232.1 asks for.
        assert!(second.t.is_none());
        assert!(second.recorded.is_none());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_channel_sharing_axes_a_borrowed_axis_produces_the_same_channel_as_reading_it_alone() {
        // Arrange
        let root = temp_root();
        let path = write_session_parquet(&root, &sample_session(), "0.1.0").unwrap();
        let dir = path.parent().unwrap();
        let (t, recorded) = imu0_axes(dir);

        // Act
        let shared = read_channel_sharing_axes(
            dir,
            "IMU0_AccelX",
            BorrowedAxes { t: Some(&t), recorded: recorded.as_ref() },
            &mut |_, _| {},
        )
        .unwrap()
        .samples;
        let alone = read_channel(dir, "IMU0_AccelX").unwrap();

        // Assert -- borrowing the axis must not change one field of the answer.
        assert_eq!(shared.t_us, alone.t_us);
        assert_eq!(shared.t_recorded_us, alone.t_recorded_us);
        assert_eq!(shared.materialize(), alone.materialize());
        assert_eq!(shared.unit, alone.unit);
        assert_eq!(shared.source_kind, alone.source_kind);
        assert_eq!(shared.nominal_rate_hz, alone.nominal_rate_hz);
        assert_eq!(shared.gaps, alone.gaps);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_channel_sharing_axes_six_channels_of_one_source_agree_with_six_independent_reads() {
        // Arrange -- the shape ruling R232.1 exists for: several channels of
        // one source, whose axes are decoded by the first and borrowed by
        // the rest.
        let root = temp_root();
        let mut session = sample_session();
        let base = session.channels[0].clone();
        session.channels = (0..6).map(|i| Channel { channel_id: format!("IMU0_C{i}"), ..base.clone() }).collect();
        let path = write_session_parquet(&root, &session, "0.1.0").unwrap();
        let dir = path.parent().unwrap();
        let first = read_channel_sharing_axes(dir, "IMU0_C0", BorrowedAxes::default(), &mut |_, _| {}).unwrap();
        let t = first.t.unwrap();
        let recorded = first.recorded;

        // Act
        let shared: Vec<ChannelSamples> = (0..6)
            .map(|i| {
                read_channel_sharing_axes(
                    dir,
                    &format!("IMU0_C{i}"),
                    BorrowedAxes { t: Some(&t), recorded: recorded.as_ref() },
                    &mut |_, _| {},
                )
                .unwrap()
                .samples
            })
            .collect();

        // Assert
        for (i, got) in shared.iter().enumerate() {
            let alone = read_channel(dir, &format!("IMU0_C{i}")).unwrap();
            assert_eq!(got.t_us, alone.t_us);
            assert_eq!(got.t_recorded_us, alone.t_recorded_us);
            assert_eq!(got.materialize(), alone.materialize());
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_channel_sharing_axes_an_axis_from_a_different_file_is_a_schema_error_not_a_wrong_answer() {
        // Arrange -- a second, shorter session's `t` handed to the first's
        // channel: the mix-up a cache keyed wrongly would produce.
        let root = temp_root();
        let path = write_session_parquet(&root, &sample_session(), "0.1.0").unwrap();
        let dir = path.parent().unwrap();
        let other_root = temp_root();
        let mut short = sample_session();
        short.session_id = "1".repeat(32);
        short.channels[0].t_us = vec![0];
        short.channels[0].t_recorded_us = Some(vec![0]);
        short.channels[0].column = RawColumn::I16 { data: vec![1], scale: 1.0, offset: 0.0 };
        short.channels[0].gaps = Vec::new();
        short.channels.truncate(1);
        let other_path = write_session_parquet(&other_root, &short, "0.1.0").unwrap();
        let other_t =
            read_channel_sharing_axes(other_path.parent().unwrap(), "IMU0_AccelX", BorrowedAxes::default(), &mut |_, _| {})
                .unwrap()
                .t
                .unwrap();

        // Act
        let err =
            read_channel_sharing_axes(dir, "IMU0_AccelX", BorrowedAxes { t: Some(&other_t), recorded: None }, &mut |_, _| {})
                .unwrap_err();

        // Assert
        assert_eq!(err.kind, ParquetStoreErrorKind::Schema);

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&other_root);
    }

    #[test]
    fn read_channel_sharing_axes_a_synthesized_channel_falls_through_to_the_ordinary_read() {
        // Arrange
        let root = temp_root();
        let path = write_session_parquet(&root, &sample_session(), "0.1.0").unwrap();
        let dir = path.parent().unwrap();
        let (t, recorded) = imu0_axes(dir);

        // Act -- `Time` has no stored column to project.
        let shared = read_channel_sharing_axes(
            dir,
            "Time",
            BorrowedAxes { t: Some(&t), recorded: recorded.as_ref() },
            &mut |_, _| {},
        )
        .unwrap();
        let alone = read_channel(dir, "Time").unwrap();

        // Assert -- it borrows nothing and hands nothing back.
        assert_eq!(shared.samples.t_us, alone.t_us);
        assert_eq!(shared.samples.materialize(), alone.materialize());
        assert!(shared.t.is_none());

        let _ = std::fs::remove_dir_all(&root);
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

    /// thing — condition — result: two `fit`-source channels whose `t_us`
    /// sets are disjoint on some rows — `fit_t_recorded_us` is non-null on
    /// every row `t` has, not just the rows the first-registered channel
    /// happened to cover (L2-R10).
    #[test]
    fn fit_t_recorded_us_is_non_null_on_the_union_of_its_channels_rows() {
        // Arrange — GPS_Latitude only samples rows 0 and 2 (t_us
        // 0/2_000_000); HR_BPM samples all three rows (0/1_000_000/
        // 2_000_000). GPS_Latitude is registered first, so the pre-L2-R10
        // bug would leave row 1 (t = 1_000_000) null in `fit_t_recorded_us`.
        let root = temp_root();
        let lat = Channel {
            channel_id: "GPS_Latitude".to_string(),
            t_us: vec![0, 2_000_000],
            t_recorded_us: None,
            nominal_rate_hz: 0.0,
            column: RawColumn::F64(vec![45.0, 45.1]),
            source_kind: "fit".to_string(),
            unit: "deg".to_string(),
            gaps: Vec::new(),
        };
        let hr = Channel {
            channel_id: "HR_BPM".to_string(),
            t_us: vec![0, 1_000_000, 2_000_000],
            t_recorded_us: None,
            nominal_rate_hz: 0.0,
            column: RawColumn::F64(vec![140.0, 141.0, 142.0]),
            source_kind: "fit".to_string(),
            unit: "bpm".to_string(),
            gaps: Vec::new(),
        };
        let session = Session {
            session_id: "1112131415161718191a1b1c1d1e1f20".to_string(),
            device_id: None,
            timestamp_utc_ms: 0,
            timestamp_source: crate::session::TimestampSource::SourceFile,
            config_checksum: None,
            source_format: SourceFormat::Fit,
            blob_sha256: "1".repeat(64),
            channels: vec![lat, hr],
        };
        let path = write_session_parquet(&root, &session, "0.1.0").unwrap();

        // Act — read the raw column directly; `read_session_parquet` drops
        // `<source>_t_recorded_us` columns from its `Channel` output
        // (they're not a channel), so this test opens the file itself.
        let file = std::fs::File::open(&path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let schema = builder.schema().clone();
        let reader = builder.build().unwrap();
        let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        let batch = arrow::compute::concat_batches(&schema, &batches).unwrap();
        let col = batch
            .column_by_name("fit_t_recorded_us")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        // Assert — 3 rows total (union of both channels' t_us), none null.
        assert_eq!(col.len(), 3);
        assert_eq!(col.null_count(), 0);
        assert_eq!(col.values(), &[0, 1_000_000, 2_000_000]);

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

/// Byte-for-byte parity between the one-column-at-a-time writer (ruling
/// R203.3) and the whole-`RecordBatch` writer it replaced.
///
/// The replaced implementation is kept here verbatim as a reference: the
/// guarantee under test is that changing *how* `data.parquet` is encoded
/// changed nothing about *what* is encoded, and a golden hash constant
/// would only say the bytes stopped changing, not that they still match
/// what the old code produced.
#[cfg(test)]
mod streaming_write_parity_tests {
    use super::tests::sample_session;
    use super::*;
    use uuid::Uuid;

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-parity-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The pre-R203 writer: every column's Arrow array built up front, one
    /// `RecordBatch`, one `ArrowWriter::write`. Returns the raw file bytes.
    fn reference_bytes(session: &Session, importer_version: &str) -> Vec<u8> {
        let t = union_t_axis(session);
        let n_rows = t.len();

        let mut fields = vec![Field::new("t", DataType::Int64, false)];
        let mut arrays: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(t.clone()))];

        let rows_for = |c: &Channel| -> Vec<usize> { c.t_us.iter().map(|ts| t.binary_search(ts).unwrap()).collect() };

        let mut seen_sources: Vec<&str> = Vec::new();
        for c in &session.channels {
            if c.source_kind == "synthesized" || seen_sources.contains(&c.source_kind.as_str()) {
                continue;
            }
            seen_sources.push(&c.source_kind);
            fields.push(
                Field::new(format!("{}_t_recorded_us", c.source_kind).as_str(), DataType::Int64, true)
                    .with_metadata([("source_kind".to_string(), c.source_kind.clone())].into_iter().collect()),
            );
            let mut merged: Vec<Option<i64>> = vec![None; n_rows];
            for c2 in session.channels.iter().filter(|c2| c2.source_kind == c.source_kind) {
                let rows = rows_for(c2);
                for (&row, &v) in rows.iter().zip(c2.t_recorded_us_or_t_us().iter()) {
                    if merged[row].is_none() {
                        merged[row] = Some(v);
                    }
                }
            }
            arrays.push(Arc::new(Int64Array::from(merged)));
        }

        for c in &session.channels {
            if c.source_kind == "synthesized" {
                continue;
            }
            let rows = rows_for(c);
            let (arrow_type, array): (DataType, ArrayRef) = match &c.column {
                RawColumn::I16 { data, .. } => {
                    let mut vals: Vec<Option<i16>> = vec![None; n_rows];
                    for (&row, &v) in rows.iter().zip(data.iter()) {
                        vals[row] = Some(v);
                    }
                    (DataType::Int16, Arc::new(Int16Array::from(vals)))
                }
                RawColumn::I32 { data, .. } => {
                    let mut vals: Vec<Option<i32>> = vec![None; n_rows];
                    for (&row, &v) in rows.iter().zip(data.iter()) {
                        vals[row] = Some(v);
                    }
                    (DataType::Int32, Arc::new(Int32Array::from(vals)))
                }
                RawColumn::F32 { data, .. } => {
                    let mut vals: Vec<Option<f32>> = vec![None; n_rows];
                    for (&row, &v) in rows.iter().zip(data.iter()) {
                        vals[row] = Some(v);
                    }
                    (DataType::Float32, Arc::new(Float32Array::from(vals)))
                }
                RawColumn::F64(data) => {
                    let mut vals: Vec<Option<f64>> = vec![None; n_rows];
                    for (&row, &v) in rows.iter().zip(data.iter()) {
                        vals[row] = Some(v);
                    }
                    (DataType::Float64, Arc::new(Float64Array::from(vals)))
                }
                RawColumn::Ramp { .. } | RawColumn::Interp { .. } => unreachable!("never a column"),
            };
            let metadata: std::collections::HashMap<String, String> = column_metadata(c).into_iter().collect();
            fields.push(Field::new(c.channel_id.as_str(), arrow_type, true).with_metadata(metadata));
            arrays.push(array);
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();
        let all_kv: Vec<KeyValue> =
            file_metadata(session, importer_version).into_iter().map(|(k, v)| KeyValue::new(k, v)).collect();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(ROW_GROUP_SIZE))
            .set_column_encoding(ColumnPath::from("t"), Encoding::DELTA_BINARY_PACKED)
            .set_column_statistics_enabled(ColumnPath::from("t"), EnabledStatistics::Chunk)
            .set_key_value_metadata(Some(all_kv))
            .build();

        let mut buf: Vec<u8> = Vec::new();
        {
            let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(props)).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }
        buf
    }

    /// Writes `session` with the shipping writer and returns the file bytes.
    fn written_bytes(session: &Session, importer_version: &str) -> Vec<u8> {
        let root = temp_root();
        let path = write_session_parquet(&root, session, importer_version).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_dir_all(&root);
        bytes
    }

    #[test]
    fn streaming_writer_the_mixed_imu_and_gps_fixture_writes_the_same_bytes_as_the_whole_batch_writer() {
        // Arrange
        let session = sample_session();

        // Act
        let streamed = written_bytes(&session, "0.1.0");

        // Assert
        assert_eq!(streamed, reference_bytes(&session, "0.1.0"));
    }

    #[test]
    fn streaming_writer_a_dense_single_channel_session_writes_the_same_bytes_as_the_whole_batch_writer() {
        // Arrange — every row sampled, so the value column carries no nulls
        // at all: the case where the null buffer must be dropped rather than
        // written as an all-true mask.
        let n = 1000usize;
        let mut session = sample_session();
        session.channels = vec![Channel {
            channel_id: "WheelFront".to_string(),
            t_us: (0..n).map(|i| i as i64 * 1000).collect(),
            t_recorded_us: None,
            nominal_rate_hz: 1000.0,
            column: RawColumn::I32 { data: (0..n).map(|i| i as i32).collect(), scale: 1.0, offset: 0.0 },
            source_kind: "wheel".to_string(),
            unit: "pulse".to_string(),
            gaps: Vec::new(),
        }];

        // Act
        let streamed = written_bytes(&session, "0.1.0");

        // Assert
        assert_eq!(streamed, reference_bytes(&session, "0.1.0"));
    }

    #[test]
    fn streaming_writer_a_session_longer_than_one_row_group_splits_at_the_same_boundaries() {
        // Arrange — one row past ROW_GROUP_SIZE, so the file has one full
        // row group and a one-row remainder, the split the whole-batch
        // writer makes internally.
        let n = ROW_GROUP_SIZE + 1;
        let mut session = sample_session();
        session.channels = vec![Channel {
            channel_id: "IMU0_AccelX".to_string(),
            t_us: (0..n).map(|i| i as i64 * 1250).collect(),
            t_recorded_us: None,
            nominal_rate_hz: 800.0,
            column: RawColumn::I16 { data: (0..n).map(|i| (i % 3000) as i16).collect(), scale: 0.001, offset: 0.0 },
            source_kind: "imu0".to_string(),
            unit: "g".to_string(),
            gaps: Vec::new(),
        }];

        // Act
        let streamed = written_bytes(&session, "0.1.0");

        // Assert
        assert_eq!(streamed, reference_bytes(&session, "0.1.0"));
    }

    /// A channel whose `t_us` is neither strictly increasing nor unique.
    ///
    /// Found by importing a real 377 MB log: `HR_RR` reports heart-rate
    /// intervals whose recorded timestamps repeat and step backwards, so
    /// C1 §3.5 invariant 1 does not hold for every source in practice. The
    /// column writer must place those samples exactly where the previous
    /// binary-search scatter did — later sample wins a repeated row — and
    /// must not reject the session.
    fn session_with_unordered_timestamps() -> Session {
        let mut session = sample_session();
        session.channels = vec![Channel {
            channel_id: "HR_RR".to_string(),
            t_us: vec![0, 3000, 1000, 1000, 2000, 3000],
            t_recorded_us: None,
            nominal_rate_hz: 0.0,
            column: RawColumn::F64(vec![10.0, 40.0, 20.0, 21.0, 30.0, 41.0]),
            source_kind: "hr".to_string(),
            unit: "ms".to_string(),
            gaps: Vec::new(),
        }];
        session
    }

    #[test]
    fn streaming_writer_a_channel_whose_timestamps_repeat_and_step_backwards_writes_the_same_bytes() {
        // Arrange
        let session = session_with_unordered_timestamps();

        // Act
        let streamed = written_bytes(&session, "0.1.0");

        // Assert
        assert_eq!(streamed, reference_bytes(&session, "0.1.0"));
    }

    #[test]
    fn streaming_writer_a_channel_whose_timestamps_repeat_and_step_backwards_round_trips_the_later_value() {
        // Arrange
        let root = temp_root();
        let session = session_with_unordered_timestamps();

        // Act
        let path = write_session_parquet(&root, &session, "0.1.0").unwrap();
        let back = read_session_parquet(&path).unwrap();

        // Assert — four distinct timestamps; the repeated ones keep the
        // later of the two samples written to them.
        let ch = back.channels.iter().find(|c| c.channel_id == "HR_RR").unwrap();
        assert_eq!(ch.t_us, vec![0, 1000, 2000, 3000]);
        let RawColumn::F64(values) = &ch.column else { panic!("not F64") };
        assert_eq!(values, &vec![10.0, 21.0, 30.0, 41.0]);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn streaming_writer_a_session_with_no_rows_writes_the_same_empty_file_as_the_whole_batch_writer() {
        // Arrange
        let mut session = sample_session();
        session.channels = Vec::new();

        // Act
        let streamed = written_bytes(&session, "0.1.0");

        // Assert
        assert_eq!(streamed, reference_bytes(&session, "0.1.0"));
    }
}

/// [`read_channel`] against [`read_session_parquet`] + synthesis: the
/// columnar read must return exactly what the whole-file read would have,
/// for stored and synthesized channels alike (ruling R203.1).
#[cfg(test)]
mod channel_read_tests {
    use super::tests::sample_session;
    use super::*;
    use crate::math::eval::ChannelLookup;
    use crate::session::synthesis::synthesize_base_channels;
    use uuid::Uuid;

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-readchan-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Writes `session` and returns `(data_root, session_dir)`.
    fn seed(session: &Session) -> (PathBuf, PathBuf) {
        let root = temp_root();
        write_session_parquet(&root, session, "0.1.0").unwrap();
        let dir = root.join("sessions").join(&session.session_id);
        (root, dir)
    }

    /// The whole-file read of `session_dir`, synthesis included — the
    /// reference every `read_channel` result is compared against.
    fn whole_file(session_dir: &Path) -> Session {
        let mut s = read_session_parquet(&session_dir.join("data.parquet")).unwrap();
        synthesize_base_channels(&mut s);
        s
    }

    /// A GPS-carrying session, so `Time` and `Distance` both synthesize.
    fn session_with_gps_speed() -> Session {
        let mut session = sample_session();
        session.channels.push(Channel {
            channel_id: "GPS_SpeedKmh".to_string(),
            t_us: vec![0, 2500],
            t_recorded_us: None,
            nominal_rate_hz: 10.0,
            column: RawColumn::F64(vec![18.0, 36.0]),
            source_kind: "gps".to_string(),
            unit: "km/h".to_string(),
            gaps: Vec::new(),
        });
        session
    }

    /// Every `(done, total)` `read_channel_with_progress` reported, in order.
    fn progress_of(session_dir: &Path, channel: &str) -> (ChannelSamples, Vec<(usize, usize)>) {
        let mut seen: Vec<(usize, usize)> = Vec::new();
        let samples = {
            let mut record = |done: usize, total: usize| seen.push((done, total));
            read_channel_with_progress(session_dir, channel, &mut record).unwrap()
        };
        (samples, seen)
    }

    #[test]
    fn read_channel_with_progress_a_stored_channel_reports_up_to_the_files_row_count_and_decodes_the_same_samples() {
        // Arrange
        let session = sample_session();
        let (root, dir) = seed(&session);
        let file_rows = read_channel_index(&dir).unwrap().first().unwrap().file_rows;

        // Act
        let (got, seen) = progress_of(&dir, "IMU0_AccelX");

        // Assert — one pass over the file, ending exactly at its row count,
        // and the samples are `read_channel`'s own.
        assert!(!seen.is_empty());
        assert!(seen.iter().all(|&(_, total)| total == file_rows));
        assert_eq!(seen.last().unwrap().0, file_rows);
        assert_eq!(got.to_channel(), read_channel(&dir, "IMU0_AccelX").unwrap().to_channel());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_channel_with_progress_the_synthesized_distance_channel_counts_its_two_passes_monotonically() {
        // Arrange — `Distance` reads its `Time` source and `GPS_SpeedKmh`.
        let session = session_with_gps_speed();
        let (root, dir) = seed(&session);
        let file_rows = read_channel_index(&dir).unwrap().first().unwrap().file_rows;

        // Act
        let (_, seen) = progress_of(&dir, "Distance");

        // Assert — one total covering both passes, never sent back to zero.
        assert!(seen.iter().all(|&(_, total)| total == file_rows * 2));
        assert!(seen.windows(2).all(|w| w[0].0 <= w[1].0));
        assert_eq!(seen.last().unwrap().0, file_rows * 2);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_channel_a_stored_i16_channel_matches_the_whole_file_read_field_for_field() {
        // Arrange
        let session = sample_session();
        let (root, dir) = seed(&session);
        let expected = whole_file(&dir);

        // Act
        let got = read_channel(&dir, "IMU0_AccelX").unwrap();

        // Assert
        let want = expected.channels.iter().find(|c| c.channel_id == "IMU0_AccelX").unwrap();
        assert_eq!(&got.to_channel(), want);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_channel_a_stored_f64_channel_with_negative_zero_and_nan_matches_the_whole_file_read() {
        // Arrange
        let session = sample_session();
        let (root, dir) = seed(&session);
        let expected = whole_file(&dir);

        // Act
        let got = read_channel(&dir, "GPS_EpochMs").unwrap();

        // Assert — bit-exact, including -0.0 and NaN (C1 §2's F64 rule).
        let want = expected.channels.iter().find(|c| c.channel_id == "GPS_EpochMs").unwrap();
        let (RawColumn::F64(a), RawColumn::F64(b)) = (&got.column, &want.column) else { panic!("not F64") };
        assert_eq!(a[0].to_bits(), b[0].to_bits());
        assert!(a[1].is_nan() && b[1].is_nan());
        assert_eq!(got.t_us.as_ref(), want.t_us.as_slice());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_channel_the_synthesized_time_channel_matches_what_synthesis_appends() {
        // Arrange
        let session = session_with_gps_speed();
        let (root, dir) = seed(&session);
        let expected = whole_file(&dir);

        // Act
        let got = read_channel(&dir, "Time").unwrap();

        // Assert
        let want = expected.channels.iter().find(|c| c.channel_id == "Time").unwrap();
        assert_eq!(&got.to_channel(), want);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_channel_the_synthesized_distance_channel_matches_what_synthesis_appends() {
        // Arrange
        let session = session_with_gps_speed();
        let (root, dir) = seed(&session);
        let expected = whole_file(&dir);

        // Act
        let got = read_channel(&dir, "Distance").unwrap();

        // Assert
        let want = expected.channels.iter().find(|c| c.channel_id == "Distance").unwrap();
        assert_eq!(&got.to_channel(), want);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_channel_a_session_without_gps_speed_reports_distance_as_not_found() {
        // Arrange — `sample_session()` has no GPS_SpeedKmh, so synthesis
        // appends `Time` but never `Distance`.
        let session = sample_session();
        let (root, dir) = seed(&session);
        assert!(whole_file(&dir).channels.iter().all(|c| c.channel_id != "Distance"));

        // Act
        let err = read_channel(&dir, "Distance").unwrap_err();

        // Assert
        assert_eq!(err.kind, ParquetStoreErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_channel_an_unknown_name_is_not_found() {
        // Arrange
        let session = sample_session();
        let (root, dir) = seed(&session);

        // Act
        let err = read_channel(&dir, "NopeChannel").unwrap_err();

        // Assert
        assert_eq!(err.kind, ParquetStoreErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_channel_a_duplicate_timestamp_candidate_elects_the_same_time_source_as_the_whole_file_read() {
        // Arrange — two rate-0 channels, the election's fallback branch
        // (longest wins). `noisy` has more raw samples than `clean`, but
        // its timestamps repeat, so it has fewer distinct rows. Whichever
        // channel wins, both readers must pick the same one: the risk is
        // that a single-column read counts raw samples while the
        // whole-file read counts the rows a duplicate collapsed to.
        let mut session = sample_session();
        session.channels = vec![
            Channel {
                channel_id: "noisy".to_string(),
                t_us: vec![0, 1000, 1000, 2000, 2000, 3000],
                t_recorded_us: None,
                nominal_rate_hz: 0.0,
                column: RawColumn::F64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                source_kind: "hr".to_string(),
                unit: "ms".to_string(),
                gaps: Vec::new(),
            },
            Channel {
                channel_id: "clean".to_string(),
                t_us: vec![0, 1000, 2000, 3000, 4000],
                t_recorded_us: None,
                nominal_rate_hz: 0.0,
                column: RawColumn::F64(vec![10.0, 20.0, 30.0, 40.0, 50.0]),
                source_kind: "gps".to_string(),
                unit: "m".to_string(),
                gaps: Vec::new(),
            },
        ];
        let (root, dir) = seed(&session);
        let expected = whole_file(&dir);

        // Act
        let got = read_channel(&dir, "Time").unwrap();

        // Assert
        let want = expected.channels.iter().find(|c| c.channel_id == "Time").unwrap();
        assert_eq!(&got.to_channel(), want);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_channel_index_lists_every_stored_column_with_its_metadata_and_never_the_synthesized_ones() {
        // Arrange
        let session = session_with_gps_speed();
        let (root, dir) = seed(&session);

        // Act
        let index = read_channel_index(&dir).unwrap();

        // Assert
        let ids: Vec<&str> = index.iter().map(|c| c.channel_id.as_str()).collect();
        assert_eq!(ids, vec!["IMU0_AccelX", "GPS_EpochMs", "GPS_SpeedKmh"]);
        assert_eq!(index[0].sample_bytes, 2);
        assert_eq!(index[0].nominal_rate_hz, 800.0);
        assert_eq!(index[0].source_kind, "imu0");
        assert_eq!(index[1].sample_bytes, 8);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_lazy_channel_index_lists_every_stored_column_plus_the_two_synthesized_ones() {
        // Arrange
        let session = session_with_gps_speed();
        let (root, dir) = seed(&session);
        let expected = whole_file(&dir);

        // Act
        let index = read_lazy_channel_index(&dir).unwrap();

        // Assert — the same set of names a whole-session read plus synthesis
        // produces, which is what makes the lazy handle interchangeable.
        let mut got: Vec<&str> = index.iter().map(|i| i.channel_id.as_str()).collect();
        let mut want: Vec<&str> = expected.channels.iter().map(|c| c.channel_id.as_str()).collect();
        got.sort();
        want.sort();
        assert_eq!(got, want);
        assert!(index.iter().find(|i| i.channel_id == "Time").unwrap().synthesized);
        assert!(index.iter().find(|i| i.channel_id == "Distance").unwrap().synthesized);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn open_session_lazy_serves_the_same_samples_units_and_rates_as_the_whole_file_read() {
        // Arrange
        let session = session_with_gps_speed();
        let (root, dir) = seed(&session);
        let expected = whole_file(&dir);

        // Act
        let handle = open_session_lazy(&dir).unwrap();

        // Assert — every channel, including both synthesized ones, resolves
        // through the lazy path to what synthesis would have built.
        for want in &expected.channels {
            // Bitwise: these fixtures carry `-0.0` and `NaN` deliberately,
            // and `NaN != NaN` would pass a wrong answer as a failure and a
            // sign flip as a match.
            let bits = |v: Vec<f64>| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
            assert_eq!(
                bits(handle.channel_samples(&want.channel_id)),
                bits(want.materialize()),
                "channel {} differs", want.channel_id
            );
            assert_eq!(handle.channel_dims(&want.channel_id), Some((want.len(), want.nominal_rate_hz)));
        }
        assert_eq!(handle.metadata().session_id, expected.session_id);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn open_session_lazy_decodes_nothing_until_a_channel_is_asked_for() {
        // Arrange
        let session = session_with_gps_speed();
        let (root, dir) = seed(&session);

        // Act
        let handle = open_session_lazy(&dir).unwrap();
        let before = handle.resident_bytes();
        let _ = handle.channel_samples("IMU0_AccelX");

        // Assert
        assert_eq!(before, 0);
        assert!(handle.resident_bytes() > 0);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn open_session_lazy_a_name_the_session_does_not_have_reads_as_absent_rather_than_failing() {
        // Arrange
        let session = session_with_gps_speed();
        let (root, dir) = seed(&session);

        // Act
        let handle = open_session_lazy(&dir).unwrap();

        // Assert
        assert!(handle.channel_samples("NopeChannel").is_empty());
        assert_eq!(handle.channel_dims("NopeChannel"), None);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_channel_sample_counts_match_the_whole_file_reads_per_channel_lengths() {
        // Arrange — channels of different lengths, so a per-file row count
        // could not stand in for any of them.
        let session = session_with_gps_speed();
        let (root, dir) = seed(&session);
        let expected = read_session_parquet(&dir.join("data.parquet")).unwrap();

        // Act
        let counts = read_channel_sample_counts(&dir).unwrap();

        // Assert
        for c in &expected.channels {
            let got = counts.iter().find(|(n, _)| *n == c.channel_id).map(|(_, n)| *n);
            assert_eq!(got, Some(c.len()), "channel {} count differs", c.channel_id);
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_time_axis_bounds_match_the_earliest_and_latest_sample_of_the_whole_file_read() {
        // Arrange
        let session = session_with_gps_speed();
        let (root, dir) = seed(&session);
        let expected = read_session_parquet(&dir.join("data.parquet")).unwrap();
        let first = expected.channels.iter().filter_map(|c| c.t_us.first().copied()).min().unwrap();
        let last = expected.channels.iter().filter_map(|c| c.t_us.last().copied()).max().unwrap();

        // Act
        let got = read_time_axis_bounds(&dir).unwrap();

        // Assert
        assert_eq!(got, Some((first, last)));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn estimate_channel_bytes_one_i16_channel_is_far_below_the_whole_session_estimate() {
        // Arrange
        let session = session_with_gps_speed();
        let (root, dir) = seed(&session);

        // Act
        let one = estimate_channel_bytes(&dir, "IMU0_AccelX").unwrap();
        let all = estimate_session_bytes(&dir).unwrap();

        // Assert
        assert!(one > 0);
        assert!(one < all, "one channel ({one} B) must estimate below the whole session ({all} B)");

        let _ = std::fs::remove_dir_all(&root);
    }
}
