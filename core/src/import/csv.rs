//! Trivial CSV import (design doc D4 — low priority; §15 "CSV beyond a
//! trivial importer" is explicitly deferred). The input shape here is
//! `docs/IDL0_SPEC.md` §15a.4's minimal definition: a `t_seconds` column
//! plus one column per channel, comma-separated, no quoting/escaping.

use crate::session::{Channel, Session, SourceFormat};

use super::{ImportedSession, Importer, ImporterError, ImporterWarning};

/// This importer implementation's own version string (C1 §4.3).
pub const CSV_IMPORTER_VERSION: &str = "0.1.0";

/// Imports the trivial CSV shape documented in `docs/IDL0_SPEC.md` §15a.4:
/// header `t_seconds,<channel>,...`, comma-separated, no quoting/escaping,
/// an empty cell meaning "no sample for this channel at this row."
pub struct CsvImporter;

impl Importer for CsvImporter {
    fn source_format(&self) -> SourceFormat {
        SourceFormat::Csv
    }

    fn import(&self, bytes: &[u8], blob_sha256: &str) -> Result<ImportedSession, ImporterError> {
        let text = std::str::from_utf8(bytes).map_err(|e| ImporterError::NotUtf8(e.to_string()))?;
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());

        let header = lines
            .next()
            .ok_or_else(|| ImporterError::CsvMalformed("empty file".to_string()))?;
        let columns: Vec<&str> = header.split(',').map(|c| c.trim()).collect();
        if columns.len() < 2 || columns[0] != "t_seconds" {
            return Err(ImporterError::CsvMalformed(
                "header must be \"t_seconds,<channel>,...\" with at least one channel column"
                    .to_string(),
            ));
        }
        // L2-R11: reject a header naming a column `t`/`Time`/`Distance`
        // (collides with the Parquet union-axis column or a synthesized
        // base channel — G5.4), and reject any duplicated column name
        // (including a duplicated `t_seconds`) — two Arrow fields sharing a
        // name.
        for &name in columns[1..].iter() {
            if name == "t" || name == "Time" || name == "Distance" {
                return Err(ImporterError::CsvMalformed(format!(
                    "header column \"{name}\" collides with a reserved axis/synthesized name"
                )));
            }
        }
        let mut seen = std::collections::HashSet::with_capacity(columns.len());
        for &name in columns.iter() {
            if !seen.insert(name) {
                return Err(ImporterError::CsvMalformed(format!(
                    "header column \"{name}\" is duplicated"
                )));
            }
        }
        let channel_names = &columns[1..];

        let mut per_channel: Vec<Vec<(f64, f64)>> = vec![Vec::new(); channel_names.len()];
        let mut min_t_seconds: Option<f64> = None;

        // L2-R7(a): a CSV file is not guaranteed sorted by `t_seconds`
        // (§15a.4: "monotonic non-decreasing is not required" for the file
        // as a whole) — collect every row's `(t_seconds, cells)` first so
        // `t0` can be the minimum across the whole file, not whichever row
        // is read first.
        let mut rows: Vec<(usize, f64, Vec<&str>)> = Vec::new();
        for (row_idx, line) in lines.enumerate() {
            let cells: Vec<&str> = line.split(',').collect();
            if cells.len() != columns.len() {
                return Err(ImporterError::CsvMalformed(format!(
                    "row {row_idx} has {} cells, expected {}",
                    cells.len(),
                    columns.len()
                )));
            }
            let t_seconds: f64 = cells[0].trim().parse().map_err(|_| {
                ImporterError::CsvMalformed(format!(
                    "row {row_idx}: unparseable t_seconds \"{}\"",
                    cells[0]
                ))
            })?;
            min_t_seconds = Some(match min_t_seconds {
                Some(m) => m.min(t_seconds),
                None => t_seconds,
            });
            rows.push((row_idx, t_seconds, cells));
        }

        let first_t_seconds = match min_t_seconds {
            Some(m) => m,
            None => return Err(ImporterError::CsvMalformed("no data rows".to_string())),
        };

        // The per-channel duplicate/non-monotonic check below compares each
        // row to the previous one it kept — meaningful only in ascending
        // `t_seconds` order (L2-R7(a): the file itself is not guaranteed
        // sorted). Stable sort preserves document order among equal
        // `t_seconds` values, so the golden fixture's already-sorted input
        // is unaffected.
        rows.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut warnings = Vec::new();
        for (row_idx, t_seconds, cells) in rows {
            for (ci, cell) in cells[1..].iter().enumerate() {
                let cell = cell.trim();
                if cell.is_empty() {
                    continue;
                }
                let value: f64 = cell.parse().map_err(|_| {
                    ImporterError::CsvMalformed(format!(
                        "row {row_idx}, column \"{}\": unparseable value \"{cell}\"",
                        channel_names[ci]
                    ))
                })?;
                let series = &mut per_channel[ci];
                if let Some(&(last_t_seconds, _)) = series.last() {
                    if t_seconds <= last_t_seconds {
                        warnings.push(ImporterWarning::new(format!(
                            "dropped CSV row {row_idx}, column \"{}\": duplicate/non-monotonic t_seconds",
                            channel_names[ci]
                        )));
                        continue;
                    }
                }
                series.push((t_seconds, value));
            }
        }

        let mut channels = Vec::new();
        for (name, series) in channel_names.iter().zip(per_channel.into_iter()) {
            if series.is_empty() {
                continue;
            }
            let (t_us, values): (Vec<i64>, Vec<f64>) = series
                .into_iter()
                .map(|(t_seconds, value)| {
                    let t_us = ((t_seconds - first_t_seconds) * 1_000_000.0).round() as i64;
                    (t_us, value)
                })
                .unzip();
            channels.push(Channel::from_f64_with_times(
                name.to_string(),
                0.0,
                values,
                t_us,
                "csv",
            ));
        }
        if channels.is_empty() {
            return Err(ImporterError::CsvMalformed(
                "no channel had any parseable sample".to_string(),
            ));
        }

        let session = Session {
            session_id: super::session_id_from_blob_hash(blob_sha256),
            device_id: None,
            timestamp_utc_ms: 0,
            config_checksum: None,
            source_format: SourceFormat::Csv,
            blob_sha256: blob_sha256.to_string(),
            channels,
        };

        Ok(ImportedSession { session, warnings })
    }

    fn importer_version(&self) -> &'static str {
        CSV_IMPORTER_VERSION
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// thing — condition — result: golden fixture with a duplicate
    /// `t_seconds` row and a sparse cell — maps columns to channels, drops
    /// the duplicate, and leaves `unit` empty.
    #[test]
    fn csv_importer_golden_fixture_maps_columns_and_drops_duplicate_row() {
        // Arrange — column "b" is empty (no sample) on the last row.
        let csv = "t_seconds,a,b\n0.0,10.0,1.0\n1.0,11.0,2.0\n1.0,12.0,3.0\n2.5,13.0,\n";

        // Act
        let outcome = CsvImporter.import(csv.as_bytes(), &"cc".repeat(32)).unwrap();

        // Assert — two warnings: column "a" and column "b" both hit the
        // duplicate t_seconds=1.0 row independently.
        assert_eq!(outcome.warnings.len(), 2);

        let a = outcome
            .session
            .channels
            .iter()
            .find(|c| c.channel_id == "a")
            .unwrap();
        assert_eq!(a.t_us, vec![0, 1_000_000, 2_500_000]);
        assert_eq!(a.materialize(), vec![10.0, 11.0, 13.0]);
        assert_eq!(a.unit, "");

        let b = outcome
            .session
            .channels
            .iter()
            .find(|c| c.channel_id == "b")
            .unwrap();
        assert_eq!(b.t_us, vec![0, 1_000_000]);
        assert_eq!(b.materialize(), vec![1.0, 2.0]);
        assert_eq!(b.unit, "");

        assert_eq!(outcome.session.timestamp_utc_ms, 0);
        assert_eq!(outcome.session.source_format, SourceFormat::Csv);
        assert_eq!(outcome.session.device_id, None);
    }

    /// thing — condition — result: header not shaped `t_seconds,<channel>,...`
    /// — typed error.
    #[test]
    fn csv_importer_missing_header_returns_typed_error() {
        // Arrange
        let csv = "not,a,valid,header\n1,2,3,4\n";

        // Act
        let result = CsvImporter.import(csv.as_bytes(), &"dd".repeat(32));

        // Assert
        assert!(matches!(result, Err(ImporterError::CsvMalformed(_))));
    }

    /// thing — condition — result: header only, no data rows — typed error.
    #[test]
    fn csv_importer_no_data_rows_returns_typed_error() {
        // Arrange
        let csv = "t_seconds,a\n";

        // Act
        let result = CsvImporter.import(csv.as_bytes(), &"ee".repeat(32));

        // Assert
        assert!(matches!(result, Err(ImporterError::CsvMalformed(_))));
    }

    /// thing — condition — result: rows not sorted by `t_seconds` — `t0`
    /// anchors to the file-wide minimum, not the first row read (L2-R7(a)).
    #[test]
    fn csv_importer_out_of_order_rows_anchors_t0_to_file_wide_minimum() {
        // Arrange — row 0 has t_seconds=1.0, row 1 has t_seconds=0.0 (the
        // true minimum), reversed from document order.
        let csv = "t_seconds,a\n1.0,10.0\n0.0,20.0\n";

        // Act
        let outcome = CsvImporter.import(csv.as_bytes(), &"11".repeat(32)).unwrap();

        // Assert — the minimum `t_seconds` (row 1's 0.0) anchors `t=0`
        // regardless of row order: row 0's value (10.0, at t_seconds=1.0)
        // gets `t_us == 1_000_000`, not `0`, and row 1's value (20.0, at
        // the true minimum) gets `t_us == 0`, not a negative number. C1
        // §3.5 invariant 1 (`t_us` strictly increasing) means the channel's
        // sample order is by ascending time, not document row order, so
        // row 1's sample (t_us=0) comes first.
        let a = outcome
            .session
            .channels
            .iter()
            .find(|c| c.channel_id == "a")
            .unwrap();
        assert_eq!(a.t_us, vec![0, 1_000_000]);
        assert_eq!(a.materialize(), vec![20.0, 10.0]);
    }

    /// thing — condition — result: a header column literally named `t`,
    /// `Time`, or `Distance` — typed error (L2-R11).
    #[test]
    fn csv_importer_reserved_column_name_returns_typed_error() {
        for reserved in ["t", "Time", "Distance"] {
            // Arrange
            let csv = format!("t_seconds,a,{reserved}\n0.0,1.0,2.0\n");

            // Act
            let result = CsvImporter.import(csv.as_bytes(), &"22".repeat(32));

            // Assert
            assert!(
                matches!(result, Err(ImporterError::CsvMalformed(_))),
                "expected CsvMalformed for reserved column name \"{reserved}\""
            );
        }
    }

    /// thing — condition — result: a header with a duplicated column name
    /// — typed error (L2-R11).
    #[test]
    fn csv_importer_duplicated_column_name_returns_typed_error() {
        // Arrange
        let csv = "t_seconds,a,a\n0.0,1.0,2.0\n";

        // Act
        let result = CsvImporter.import(csv.as_bytes(), &"33".repeat(32));

        // Assert
        assert!(matches!(result, Err(ImporterError::CsvMalformed(_))));
    }
}
