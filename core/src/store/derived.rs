//! `derived/<hash>.parquet` — materialised estimator outputs (contract C1
//! §5). Content-addressed by `sha256(input column hashes ‖ config json ‖
//! engine version)`; the file already existing at that path is
//! authoritative (design doc §5 — "sync keeps whichever it has").

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use sha2::{Digest, Sha256};

use crate::store::atomic::write_atomic;

/// Discriminant for [`DerivedStoreError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivedStoreErrorKind {
    /// A filesystem or Arrow/Parquet I/O operation failed.
    Io,
    /// Config serialisation failed, or the outputs given to
    /// [`write_derived_parquet`] violate a schema invariant (e.g. mismatched
    /// `t` axes).
    Schema,
}

/// Error from the derived-file store. Never `Err(String)` (CLAUDE.md §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedStoreError {
    pub kind: DerivedStoreErrorKind,
    pub message: String,
}

impl DerivedStoreError {
    fn new(kind: DerivedStoreErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}

impl fmt::Display for DerivedStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for DerivedStoreError {}

/// One input channel's `column_hash` (C1 §5, step 1): `SHA256(channel_id
/// UTF-8 bytes ++ 0x00 ++ for each sample in t_us order: t_us LE i64 bytes
/// ++ physical value LE f64 bytes)`. `values[i]` must be the *physical*
/// value (`materialize()[i]`, C1 §5's own wording) — raw wire
/// representation is deliberately not hashed, so a future storage-format
/// change doesn't change existing derived-file identities.
pub fn derived_input_hash(channel_id: &str, t_us: &[i64], values: &[f64]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(channel_id.as_bytes());
    hasher.update([0u8]);
    for (&t, &v) in t_us.iter().zip(values.iter()) {
        hasher.update(t.to_le_bytes());
        hasher.update(v.to_le_bytes());
    }
    hasher.finalize().into()
}

/// The canonical config JSON bytes C1 §5 step 2 requires: recursively
/// key-sorted, compact, no trailing newline. `serde_json::Value`'s default
/// `Map` (this crate's `serde_json` dependency does **not** enable the
/// `preserve_order` feature — confirmed in `core/Cargo.toml`) is
/// `BTreeMap`-backed, so every object — nested included — is *already*
/// serialized in ascending-key order by plain `serde_json::to_vec`; no
/// hand-rolled recursive sort is needed.
pub fn canonical_config_json(config: &serde_json::Value) -> Result<Vec<u8>, DerivedStoreError> {
    serde_json::to_vec(config)
        .map_err(|e| DerivedStoreError::new(DerivedStoreErrorKind::Schema, format!("config serialisation: {e}")))
}

/// The overall derived-file hash (C1 §5, step 3): `SHA256(for each input,
/// sorted by channel_id ascending UTF-8: its column_hash (32 bytes) ++ 0x00
/// ++ config_json_bytes ++ 0x00 ++ engine_version UTF-8 bytes)`. Returns the
/// 32 raw digest bytes; `filename_hex` (the `derived/<hash>.parquet` name)
/// is `lowercase_hex(hash_bytes)`.
pub fn derived_file_hash(
    inputs: &[(String, [u8; 32])],
    config_json_bytes: &[u8],
    engine_version: &str,
) -> [u8; 32] {
    let mut sorted = inputs.to_vec();
    sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));

    let mut hasher = Sha256::new();
    for (_, column_hash) in &sorted {
        hasher.update(column_hash);
    }
    hasher.update([0u8]);
    hasher.update(config_json_bytes);
    hasher.update([0u8]);
    hasher.update(engine_version.as_bytes());
    hasher.finalize().into()
}

/// Lowercase hex of a 32-byte digest — the `derived/<hex>.parquet` filename.
pub fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// One output channel of a materialised estimator (contract C1 §5's file
/// content — `Float64`, no `scale`/`offset`).
pub struct DerivedOutput {
    /// The output channel's identifier, e.g. `"Roll (deg)"`.
    pub channel_id: String,
    /// Sample times, microseconds, ascending — shared across every output of
    /// one derived file (C1 §5).
    pub t_us: Vec<i64>,
    /// Physical values, one per `t_us` entry.
    pub values: Vec<f64>,
    /// Nominal sample rate, Hz. `0.0` marks an event (non-fixed-rate) channel.
    pub nominal_rate_hz: f64,
    /// Physical unit string, e.g. `"deg"`.
    pub unit: String,
}

/// Writes `derived/<hash>.parquet` for one session. `inputs` is
/// `(channel_id, column_hash)` for every input channel, already computed by
/// the caller via [`derived_input_hash`] (the caller — the estimator glue,
/// not this store module, since only it knows which channels were actually
/// consumed). If the target path already exists, this is a no-op and its
/// existing bytes are authoritative (C1 §5 — content-addressed, same rule
/// as [`crate::store::blob::write_blob`]).
#[allow(clippy::too_many_arguments)]
pub fn write_derived_parquet(
    data_root: &Path,
    session_id: &str,
    derived_kind: &str,
    inputs: &[(String, [u8; 32])],
    config: &serde_json::Value,
    outputs: &[DerivedOutput],
    computed_at_utc_ms: i64,
) -> Result<PathBuf, DerivedStoreError> {
    let config_json_bytes = canonical_config_json(config)?;
    let hash_bytes = derived_file_hash(inputs, &config_json_bytes, crate::VERSION);
    let filename_hex = hex(&hash_bytes);
    let target = data_root
        .join("sessions")
        .join(session_id)
        .join("derived")
        .join(format!("{filename_hex}.parquet"));

    if target.is_file() {
        return Ok(target); // content-addressed no-op, C1 §5.
    }

    // Every output shares this derived file's own `t` axis: per C1 §5,
    // "typically inherited from its primary input's post-correction t" —
    // this function takes the caller's already-decided `t_us` per output
    // rather than deriving one itself (estimator-specific knowledge, not
    // this store module's).
    let t: Vec<i64> = outputs.first().map(|o| o.t_us.clone()).unwrap_or_default();
    if outputs.iter().any(|o| o.t_us != t) {
        return Err(DerivedStoreError::new(
            DerivedStoreErrorKind::Schema,
            "all outputs of one derived file must share the same t axis (C1 §5)".to_string(),
        ));
    }

    let mut fields = vec![Field::new("t", DataType::Int64, false)];
    let mut arrays: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(t.clone()))];
    for o in outputs {
        let metadata: std::collections::HashMap<String, String> = [
            ("nominal_rate_hz".to_string(), o.nominal_rate_hz.to_string()),
            ("unit".to_string(), o.unit.clone()),
            ("channel_kind".to_string(), if o.nominal_rate_hz == 0.0 { "event" } else { "fixed-rate" }.to_string()),
        ]
        .into_iter()
        .collect();
        fields.push(Field::new(&o.channel_id, DataType::Float64, false).with_metadata(metadata));
        arrays.push(Arc::new(Float64Array::from(o.values.clone())));
    }

    // C1 §5: the `inputs` metadata value must be "in the same canonical
    // order used to compute the file hash" — channel_id ascending, the same
    // sort `derived_file_hash` applies to its own internal copy — so a
    // consumer can verify/re-derive without re-hashing every input in
    // whatever order the caller happened to pass them.
    let inputs_json = {
        let mut sorted_inputs = inputs.to_vec();
        sorted_inputs.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        let entries: Vec<String> = sorted_inputs
            .iter()
            .map(|(id, h)| format!(r#"{{"channel_id":{},"column_hash":"{}"}}"#, serde_json::to_string(id).unwrap(), hex(h)))
            .collect();
        format!("[{}]", entries.join(","))
    };

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|e| DerivedStoreError::new(DerivedStoreErrorKind::Schema, format!("RecordBatch::try_new: {e}")))?;

    // `WriterPropertiesBuilder::set_key_value_metadata` *replaces* rather
    // than accumulates (verified against the pinned 59.3.0 source,
    // `parquet::file::properties::WriterPropertiesBuilder::set_key_value_metadata`
    // — `self.key_value_metadata = value`; same fact already documented at
    // `core/src/store/parquet.rs`'s file-metadata write site), so every
    // file-metadata pair is collected into one `Vec<KeyValue>` and set in a
    // single call.
    let all_kv: Vec<parquet::file::metadata::KeyValue> = vec![
        parquet::file::metadata::KeyValue::new("derived_kind".to_string(), derived_kind.to_string()),
        parquet::file::metadata::KeyValue::new("inputs".to_string(), inputs_json),
        parquet::file::metadata::KeyValue::new(
            "config_json".to_string(),
            String::from_utf8_lossy(&config_json_bytes).into_owned(),
        ),
        parquet::file::metadata::KeyValue::new("engine_version".to_string(), crate::VERSION.to_string()),
        parquet::file::metadata::KeyValue::new("computed_at_utc_ms".to_string(), computed_at_utc_ms.to_string()),
    ];
    let props = WriterProperties::builder().set_key_value_metadata(Some(all_kv)).build();

    let mut buf: Vec<u8> = Vec::new();
    {
        let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(props))
            .map_err(|e| DerivedStoreError::new(DerivedStoreErrorKind::Io, e.to_string()))?;
        writer.write(&batch).map_err(|e| DerivedStoreError::new(DerivedStoreErrorKind::Io, e.to_string()))?;
        writer.close().map_err(|e| DerivedStoreError::new(DerivedStoreErrorKind::Io, e.to_string()))?;
    }

    write_atomic(data_root, &target, &buf, None)
        .map_err(|e| DerivedStoreError::new(DerivedStoreErrorKind::Io, e.to_string()))?;
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use uuid::Uuid;

    /// Reads back a written derived parquet file's file-level key-value
    /// metadata (not its columns) as a plain map, for asserting against C1
    /// §5's file-metadata table in tests.
    fn read_file_metadata(path: &Path) -> std::collections::HashMap<String, String> {
        let file = std::fs::File::open(path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        builder
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|kv| kv.value.map(|v| (kv.key, v)))
            .collect()
    }

    fn temp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("idl-rs-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn derived_input_hash_is_deterministic_and_sensitive_to_every_component() {
        // Arrange
        let t = vec![0i64, 1000, 2000];
        let v = vec![1.0, 2.0, 3.0];

        // Act
        let h1 = derived_input_hash("Fork travel (mm)", &t, &v);
        let h2 = derived_input_hash("Fork travel (mm)", &t, &v);
        let h3 = derived_input_hash("Fork velocity (mm/s)", &t, &v); // different name
        let h4 = derived_input_hash("Fork travel (mm)", &t, &[1.0, 2.0, 3.5]); // different value

        // Assert
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        assert_ne!(h1, h4);
    }

    #[test]
    fn derived_file_hash_is_order_independent_in_input_list_but_sorted_internally() {
        // Arrange — two input orderings that should hash identically because
        // step 3 sorts by channel_id before hashing.
        let ha = derived_input_hash("A", &[0], &[1.0]);
        let hb = derived_input_hash("B", &[0], &[2.0]);
        let config = b"{}";

        // Act
        let h1 = derived_file_hash(&[("A".to_string(), ha), ("B".to_string(), hb)], config, "0.1.0");
        let h2 = derived_file_hash(&[("B".to_string(), hb), ("A".to_string(), ha)], config, "0.1.0");

        // Assert
        assert_eq!(h1, h2);
    }

    #[test]
    fn canonical_config_json_sorts_nested_object_keys() {
        // Arrange
        let config: serde_json::Value = serde_json::from_str(r#"{"z":1,"a":{"y":2,"b":3}}"#).unwrap();

        // Act
        let bytes = canonical_config_json(&config).unwrap();

        // Assert
        assert_eq!(String::from_utf8(bytes).unwrap(), r#"{"a":{"b":3,"y":2},"z":1}"#);
    }

    #[test]
    fn write_derived_parquet_same_hash_content_second_write_is_a_no_op() {
        // Arrange
        let root = temp_root();
        let outputs = vec![DerivedOutput {
            channel_id: "Roll (deg)".to_string(),
            t_us: vec![0, 1000],
            values: vec![0.5, 0.6],
            nominal_rate_hz: 800.0,
            unit: "deg".to_string(),
        }];
        let config = serde_json::json!({});
        let ih = derived_input_hash("IMU0_AccelX", &[0, 1000], &[1.0, 2.0]);
        let inputs = vec![("IMU0_AccelX".to_string(), ih)];

        // Act
        let p1 = write_derived_parquet(&root, "s1", "iekf_suspension_attitude", &inputs, &config, &outputs, 0).unwrap();
        let mtime1 = std::fs::metadata(&p1).unwrap().modified().unwrap();
        let p2 = write_derived_parquet(&root, "s1", "iekf_suspension_attitude", &inputs, &config, &outputs, 999).unwrap();
        let mtime2 = std::fs::metadata(&p2).unwrap().modified().unwrap();

        // Assert — same path (content-addressed), file untouched by the
        // second call (mtime unchanged) even though computed_at_utc_ms differs.
        assert_eq!(p1, p2);
        assert_eq!(mtime1, mtime2);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn write_derived_parquet_writes_all_five_file_metadata_keys() {
        // Arrange
        let root = temp_root();
        let outputs = vec![DerivedOutput {
            channel_id: "Roll (deg)".to_string(),
            t_us: vec![0, 1000],
            values: vec![0.5, 0.6],
            nominal_rate_hz: 800.0,
            unit: "deg".to_string(),
        }];
        let config = serde_json::json!({"gain": 1.5});
        let ih = derived_input_hash("IMU0_AccelX", &[0, 1000], &[1.0, 2.0]);
        let inputs = vec![("IMU0_AccelX".to_string(), ih)];

        // Act
        let path =
            write_derived_parquet(&root, "s1", "iekf_suspension_attitude", &inputs, &config, &outputs, 1234).unwrap();
        let kv = read_file_metadata(&path);

        // Assert — all five keys C1 §5 requires "Always" survive the write,
        // not just the last one set (the bug: a per-key
        // `set_key_value_metadata` loop replaces rather than merges).
        assert_eq!(kv.get("derived_kind").map(String::as_str), Some("iekf_suspension_attitude"));
        let expected_inputs = format!(r#"[{{"channel_id":"IMU0_AccelX","column_hash":"{}"}}]"#, hex(&ih));
        assert_eq!(kv.get("inputs"), Some(&expected_inputs));
        assert_eq!(kv.get("config_json").map(String::as_str), Some(r#"{"gain":1.5}"#));
        assert_eq!(kv.get("engine_version").map(String::as_str), Some(crate::VERSION));
        assert_eq!(kv.get("computed_at_utc_ms").map(String::as_str), Some("1234"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn write_derived_parquet_inputs_metadata_is_sorted_by_channel_id_regardless_of_call_order() {
        // Arrange — pass inputs deliberately out of channel_id order.
        let root = temp_root();
        let outputs = vec![DerivedOutput {
            channel_id: "Roll (deg)".to_string(),
            t_us: vec![0],
            values: vec![0.5],
            nominal_rate_hz: 800.0,
            unit: "deg".to_string(),
        }];
        let config = serde_json::json!({});
        let hb = derived_input_hash("IMU0_GyroY", &[0], &[2.0]);
        let ha = derived_input_hash("IMU0_AccelX", &[0], &[1.0]);
        // Out-of-order: "IMU0_GyroY" > "IMU0_AccelX", but passed first.
        let inputs = vec![("IMU0_GyroY".to_string(), hb), ("IMU0_AccelX".to_string(), ha)];

        // Act
        let path = write_derived_parquet(&root, "s2", "iekf_suspension_attitude", &inputs, &config, &outputs, 0).unwrap();
        let kv = read_file_metadata(&path);
        let written_inputs = kv.get("inputs").cloned().unwrap();

        // Assert — the written `inputs` field lists channel_id-ascending
        // order (matching derived_file_hash's own internal sort), not the
        // caller's original argument order.
        let expected = format!(
            r#"[{{"channel_id":"IMU0_AccelX","column_hash":"{}"}},{{"channel_id":"IMU0_GyroY","column_hash":"{}"}}]"#,
            hex(&ha),
            hex(&hb)
        );
        assert_eq!(written_inputs, expected);

        let _ = std::fs::remove_dir_all(&root);
    }
}
