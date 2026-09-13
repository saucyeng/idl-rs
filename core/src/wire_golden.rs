//! Cross-language wire golden fixtures (ruling R236): one small deterministic
//! fixture per binary format a Tauri command emits over IPC (`IDLH`, `IDLT`,
//! `IDLS`, `IDLG`, `IDLR`), each paired with the JSON a TypeScript decoder
//! test deep-equals against.
//!
//! Every fixture is built from fixed, hand-written inputs — no RNG, no
//! timestamps — through the real wire encoders (`workbook::v3::
//! encode_host_channel_idlh`, `tile::build_tile_bytes`, `scatter_wire::
//! encode_scatter_idls`, `gps_wire::encode_gps_trace_idlg`, `raster::
//! build_histogram2d_raster_bytes`), so the `.bin` bytes are exactly what
//! the engine would send an app over IPC. The JSON expectation is built
//! from the same inputs the encoder saw (or, for the tile's bucket/column
//! regions, the same pure helper functions the encoder itself calls), never
//! by re-implementing the byte layout — this module has no decoder.
//!
//! Pure: no filesystem access. `idl-rs docs wire` (the CLI's `docs wire`
//! verb) is the only writer.

use serde_json::{json, Value};

use crate::chart_decimation::{column_stats, column_times_us, decimate_channel};
use crate::gps_wire::encode_gps_trace_idlg;
use crate::math::units::UnitLabel;
use crate::raster::build_histogram2d_raster_bytes;
use crate::scatter_wire::encode_scatter_idls;
use crate::tile::build_tile_bytes;
use crate::workbook::v3::{encode_host_channel_idlh, AxisKind, HostChannel};

/// One golden fixture: a file stem (written as `<name>.bin` / `<name>.json`
/// by the caller), the encoded bytes, and the decoded expectation.
pub struct WireFixture {
    /// File stem, e.g. `"idlh-v2-time"`.
    pub name: &'static str,
    pub bin: Vec<u8>,
    pub json: Value,
}

/// Builds every wire golden fixture (ruling R236). Same output on every run
/// and every platform: every input is a fixed literal, every encoder writes
/// little-endian regardless of host, and nothing here reads the clock.
pub fn build_wire_fixtures() -> Vec<WireFixture> {
    vec![idlh_time(), idlh_lap(), idlt_v2(), idls_v1(), idlg_v1(), idlr_v1()]
}

/// `IDLH` v2, a recorded time axis (`axis_kind == Time`) — one of the two
/// `axis_kind` values this fixture set exercises (the other is
/// [`idlh_lap`]); `axis_kind == None` is already implied by `hasT == false`,
/// which every non-`IDLH` fixture's absent-`t` fields already cover.
fn idlh_time() -> WireFixture {
    let hc = HostChannel {
        length: 4,
        t: vec![0.0, 1.0, 2.0, 3.0],
        v: vec![1.0, 2.5, -3.25, 4.0],
        unit: UnitLabel::Dimensionless,
        axis: AxisKind::Time,
    };
    let bin = encode_host_channel_idlh(&hc, 10);
    let json = json!({
        "format": "IDLH",
        "version": 2,
        "hasT": true,
        "axisKind": AxisKind::Time as u16,
        "t": hc.t,
        "v": hc.v,
    });
    WireFixture { name: "idlh-v2-time", bin, json }
}

/// `IDLH` v2, an ordinal lap axis (`axis_kind == Lap`, C2 §3.6.1) — a
/// `[lap]`-shaped `math`-cell definition's own axis kind.
fn idlh_lap() -> WireFixture {
    let hc = HostChannel {
        length: 3,
        t: vec![1.0, 2.0, 3.0],
        v: vec![10.0, 20.0, 30.0],
        unit: UnitLabel::Dimensionless,
        axis: AxisKind::Lap,
    };
    let bin = encode_host_channel_idlh(&hc, 10);
    let json = json!({
        "format": "IDLH",
        "version": 2,
        "hasT": true,
        "axisKind": AxisKind::Lap as u16,
        "t": hc.t,
        "v": hc.v,
    });
    WireFixture { name: "idlh-v2-lap", bin, json }
}

/// `IDLT` v2 (chart tile). Eight raw samples at tier 0 (bucket size 1) fit
/// entirely inside pixel column 0 of a 4-column tile — column 0 carries the
/// real min/max/mean, columns 1–3 land past the sample range and exercise
/// the all-NaN / `i64::MIN` sentinel column path (C3 §3.5).
fn idlt_v2() -> WireFixture {
    let samples: Vec<f64> = (1..=8).map(|i| i as f64).collect();
    let t_us: Vec<i64> = (0..8).map(|i| i * 1000).collect();
    let (tier, tile_index, column_count) = (0u32, 0u32, 4u32);

    let bin = build_tile_bytes(&samples, &t_us, tier, tile_index, column_count);

    // Same pure helpers `build_tile_bytes` itself calls — the numbers the
    // encoder wrote, not a re-derivation of the byte layout.
    let sample_pairs = decimate_channel(&samples, tier, tile_index);
    let columns = column_stats(&samples, tier, tile_index, column_count);
    let column_t_us = column_times_us(&t_us, tier, tile_index, column_count);

    let narrow = |v: f64| v as f32;
    let first_sample_pairs: Vec<f32> = sample_pairs.iter().take(4).copied().map(narrow).collect();
    let last_sample_pairs: Vec<f32> = sample_pairs.iter().rev().take(4).rev().copied().map(narrow).collect();

    let json = json!({
        "format": "IDLT",
        "version": 2,
        "tier": tier,
        "tileIndex": tile_index,
        "sampleCount": (sample_pairs.len() / 2) as u32,
        "columnCount": column_count,
        "firstSamplePairs": first_sample_pairs,
        "lastSamplePairs": last_sample_pairs,
        "columnMin": columns.iter().map(|c| c.0).collect::<Vec<f32>>(),
        "columnMax": columns.iter().map(|c| c.1).collect::<Vec<f32>>(),
        "columnMean": columns.iter().map(|c| c.2).collect::<Vec<f32>>(),
        "columnTUs": column_t_us,
    });
    WireFixture { name: "idlt-v2", bin, json }
}

/// `IDLS` v1 (scatter cloud). Four points; bounds are the exact min/max of
/// `xs`/`ys` so the header's pre-decimation extent matches the array values
/// verbatim (no thinning at this point count).
fn idls_v1() -> WireFixture {
    let xs = vec![-1.0, 0.0, 2.5, 3.0];
    let ys = vec![3.0, -0.5, 1.25, -2.0];
    let (x_min, x_max, y_min, y_max) = (-1.0, 3.0, -2.0, 3.0);
    let bin = encode_scatter_idls(&xs, &ys, x_min, x_max, y_min, y_max);
    let json = json!({
        "format": "IDLS",
        "version": 1,
        "pointCount": xs.len() as u32,
        "xMin": x_min, "xMax": x_max, "yMin": y_min, "yMax": y_max,
        "xs": xs, "ys": ys,
    });
    WireFixture { name: "idls-v1", bin, json }
}

/// `IDLG` v1 (GPS trace), with a colour-by channel present (`flags` bit 0
/// set) — the header's other state, no `c` block, is already covered by
/// `ipc/gps.test.ts`'s own hand-built fixtures, so this golden exercises the
/// less-trivial `has_c` layout.
fn idlg_v1() -> WireFixture {
    let xs = vec![0.0, 10.0, 20.0];
    let ys = vec![0.0, 5.0, -5.0];
    let ts = vec![0.0, 1.0, 2.0];
    let cs = vec![1.0, 2.0, 3.0];
    let bin = encode_gps_trace_idlg(&xs, &ys, &ts, Some(&cs));
    let json = json!({
        "format": "IDLG",
        "version": 1,
        "hasC": true,
        "pointCount": xs.len() as u32,
        "xs": xs, "ys": ys, "ts": ts, "cs": cs,
    });
    WireFixture { name: "idlg-v1", bin, json }
}

/// `IDLR` v1 (raster), the `histogram2d` kind: a 4×3 pixel grid over eight
/// points, with an explicit range on both axes so the bin edges (and so
/// every pixel) are fixed regardless of the point set's own extent.
fn idlr_v1() -> WireFixture {
    let xs = vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0];
    let ys = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0];
    let (width, height) = (4u16, 3u16);
    let bin = build_histogram2d_raster_bytes(&xs, &ys, width, height, Some((0.0, 4.0)), Some((0.0, 3.0)));

    const HEADER_LEN: usize = 16;
    let first_pixels: Vec<u8> = bin[HEADER_LEN..HEADER_LEN + 8].to_vec();
    let last_pixels: Vec<u8> = bin[bin.len() - 8..].to_vec();

    let json = json!({
        "format": "IDLR",
        "version": 1,
        "width": width,
        "height": height,
        "formatCode": 0,
        "firstPixels": first_pixels,
        "lastPixels": last_pixels,
    });
    WireFixture { name: "idlr-v1", bin, json }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_wire_fixtures_is_deterministic_across_calls() {
        // Arrange / Act
        let first: Vec<u8> = build_wire_fixtures().into_iter().flat_map(|f| f.bin).collect();
        let second: Vec<u8> = build_wire_fixtures().into_iter().flat_map(|f| f.bin).collect();

        // Assert
        assert_eq!(first, second);
    }

    #[test]
    fn every_fixture_bin_starts_with_its_own_magic() {
        // Arrange
        let fixtures = build_wire_fixtures();

        // Act / Assert
        for f in &fixtures {
            let magic = &f.bin[0..4];
            let expected = f.json["format"].as_str().unwrap().as_bytes();
            assert_eq!(magic, expected, "{} magic mismatch", f.name);
        }
    }

    #[test]
    fn fixture_names_are_unique() {
        // Arrange
        let fixtures = build_wire_fixtures();

        // Act
        let mut names: Vec<&str> = fixtures.iter().map(|f| f.name).collect();
        names.sort_unstable();
        let mut deduped = names.clone();
        deduped.dedup();

        // Assert
        assert_eq!(names, deduped);
    }
}
