//! A maker's geometry table, read from the words an OCR engine found on a
//! screenshot of it (whole-bike sim spec §9, stride 2). Deterministic: the
//! same words give the same table. The OCR itself is the caller's (the app
//! uses the platform's engine); this takes each word with its box.
//!
//! The table is found as a header line of size names (S, M, L, SM, MD, LG,
//! XL, S1..S6, 15.5", ...) with one row per line below it: the row's label
//! is its leading words, its values the numbers, each put in the size
//! column whose header is nearest across the page. Labels map to the
//! parameter vocabulary's `geometry.*` keys ([`super::params`]) by keyword;
//! inches and degrees are recognised and lengths come out in millimetres.

use serde::{Deserialize, Serialize};

/// One word the OCR engine found, with its box in image pixels (y down).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OcrWord {
    /// The word's text.
    pub text: String,
    /// Left edge, px.
    pub x: f64,
    /// Top edge, px.
    pub y: f64,
    /// Width, px.
    pub w: f64,
    /// Height, px.
    pub h: f64,
}

/// One row of the table.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GeometryRow {
    /// The row's label as printed.
    pub label: String,
    /// The vocabulary key it maps to (`geometry.reach_mm`), if any.
    pub key: Option<&'static str>,
    /// Per size, in the header's order: the values printed in that cell, in
    /// the key's unit (mm or degrees). More than one where the cell gives a
    /// value per flip-chip or adjustable setting ("435 / 445").
    pub values: Vec<Vec<f64>>,
}

/// A geometry table.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GeometryTable {
    /// The size names, left to right.
    pub sizes: Vec<String>,
    /// The rows, top to bottom.
    pub rows: Vec<GeometryRow>,
}

impl GeometryTable {
    /// The values for size `size` (index into [`GeometryTable::sizes`]) by
    /// key: each mapped row's first value, or its `choice`-th when the cell
    /// has several and `choice` is in range.
    pub fn for_size(&self, size: usize, choice: usize) -> Vec<(&'static str, f64)> {
        self.rows
            .iter()
            .filter_map(|r| {
                let cell = r.values.get(size)?;
                Some((r.key?, *cell.get(choice).or(cell.first())?))
            })
            .collect()
    }
}

/// Size names a header can hold, upper-cased.
const SIZE_NAMES: &[&str] = &["XXS", "XS", "S", "SM", "M", "MD", "MED", "L", "LG", "XL", "XXL", "2XL", "S1", "S2", "S3", "S4", "S5", "S6", "P1", "P2", "P3", "P4", "P5", "1", "2", "3", "4", "5", "6"];

/// Whether `t` names a size: a listed name, or a frame size in inches or cm
/// (13–24, with or without the inch mark).
fn is_size(t: &str) -> bool {
    let u = t.trim_matches(|c: char| c == '"' || c == '\'' || c == '”' || c == '″').to_ascii_uppercase();
    if SIZE_NAMES.contains(&u.as_str()) {
        return true;
    }
    (t.ends_with('"') || t.ends_with('”') || t.ends_with('″')) && u.parse::<f64>().is_ok_and(|v| (13.0..=24.0).contains(&v))
}

/// Keyword rules, first match wins: every word in `all` must appear in the
/// label, and none in `none`.
const RULES: &[(&[&str], &[&str], &str)] = &[
    (&["head", "angle"], &[], "geometry.head_angle_deg"),
    (&["hta"], &[], "geometry.head_angle_deg"),
    (&["seat", "angle"], &[], "geometry.seat_angle_deg"),
    (&["sta"], &[], "geometry.seat_angle_deg"),
    (&["stack"], &[], "geometry.stack_mm"),
    (&["reach"], &[], "geometry.reach_mm"),
    (&["chainstay"], &[], "geometry.chainstay_mm"),
    (&["chain", "stay"], &[], "geometry.chainstay_mm"),
    (&["rear", "centre"], &[], "geometry.chainstay_mm"),
    (&["rear", "center"], &[], "geometry.chainstay_mm"),
    (&["wheelbase"], &[], "geometry.wheelbase_mm"),
    (&["wheel", "base"], &[], "geometry.wheelbase_mm"),
    (&["bb", "drop"], &[], "geometry.bb_drop_mm"),
    (&["bracket", "drop"], &[], "geometry.bb_drop_mm"),
    (&["bb", "height"], &[], "geometry.bb_height_mm"),
    (&["bracket", "height"], &[], "geometry.bb_height_mm"),
    (&["head", "tube"], &["angle"], "geometry.head_tube_mm"),
    (&["seat", "tube"], &["angle"], "geometry.seat_tube_mm"),
    (&["top", "tube"], &["actual"], "geometry.top_tube_mm"),
    (&["offset"], &[], "geometry.fork_offset_mm"),
    (&["rake"], &[], "geometry.fork_offset_mm"),
    (&["trail"], &[], "geometry.trail_mm"),
    (&["standover"], &[], "geometry.standover_mm"),
    (&["stand", "over"], &[], "geometry.standover_mm"),
    (&["axle", "crown"], &[], "geometry.fork_length_mm"),
    (&["fork", "length"], &[], "geometry.fork_length_mm"),
    (&["travel", "front"], &[], "fork.travel_mm"),
    (&["fork", "travel"], &[], "fork.travel_mm"),
    (&["travel", "rear"], &[], "shock.wheel_travel_mm"),
];

/// The vocabulary key for a row label, if one matches.
pub fn key_for(label: &str) -> Option<&'static str> {
    let lower = label.to_lowercase();
    let words: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    let has = |w: &str| words.iter().any(|x| *x == w || (w.len() > 4 && x.starts_with(w)));
    RULES.iter().find(|(all, none, _)| all.iter().all(|w| has(w)) && !none.iter().any(|w| has(w))).map(|(_, _, k)| *k)
}

/// Where a key's values are plausible, in its unit (mm or degrees).
fn plausible(key: &str) -> (f64, f64) {
    match key {
        "geometry.head_angle_deg" => (55.0, 80.0),
        "geometry.seat_angle_deg" => (60.0, 85.0),
        "geometry.bb_drop_mm" => (-20.0, 90.0),
        "geometry.fork_offset_mm" => (20.0, 70.0),
        "geometry.trail_mm" => (40.0, 180.0),
        "geometry.head_tube_mm" => (60.0, 250.0),
        "geometry.bb_height_mm" => (250.0, 400.0),
        "geometry.chainstay_mm" => (380.0, 500.0),
        "fork.travel_mm" | "shock.wheel_travel_mm" => (80.0, 250.0),
        _ => (250.0, 1500.0),
    }
}

/// The numbers in one OCR token ("63.5°", "445", "435/445", "1,280"),
/// with whether it carried an inch mark.
fn numbers_in(t: &str) -> (Vec<f64>, bool) {
    let inch = t.contains('"') || t.contains('”') || t.contains('″');
    let cleaned: String = t.chars().map(|c| if c == ',' { '.' } else { c }).collect();
    let nums = cleaned
        .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
        .filter(|s| !s.is_empty() && s.chars().any(|c| c.is_ascii_digit()))
        .filter_map(|s| s.trim_matches('.').parse::<f64>().ok())
        .collect();
    (nums, inch)
}

/// A number token's value in the key's unit. Degrees are kept when
/// plausible. A length is kept as millimetres when plausible and the table
/// isn't in inches; scaled by 25.4 when it is, or when the token has an
/// inch mark, or when it is plausible only as inches. Otherwise dropped.
fn in_unit(key: Option<&str>, v: f64, inch: bool) -> Option<f64> {
    let Some(key) = key else { return Some(v) };
    let (lo, hi) = plausible(key);
    let ok = |x: f64| (lo..=hi).contains(&x);
    if key.ends_with("_deg") {
        // a degree sign read as a trailing 0 ("64°" as "640")
        return if ok(v) { Some(v) } else { ok(v / 10.0).then_some(v / 10.0) };
    }
    if !inch && ok(v) {
        Some(v)
    } else if ok(v * 25.4) {
        Some((v * 25.4 * 10.0).round() / 10.0)
    } else {
        None
    }
}

/// Whether a length value reads only as inches: implausible as millimetres
/// and plausible times 25.4.
fn only_inches(key: &str, v: f64) -> bool {
    let (lo, hi) = plausible(key);
    !key.ends_with("_deg") && !(lo..=hi).contains(&v) && (lo..=hi).contains(&(v * 25.4))
}

/// A line of words: its centre height and the words left to right.
struct Line {
    y: f64,
    words: Vec<OcrWord>,
}

fn lines_of(words: &[OcrWord]) -> Vec<Line> {
    let mut heights: Vec<f64> = words.iter().map(|w| w.h).filter(|h| *h > 0.0).collect();
    heights.sort_by(f64::total_cmp);
    let tol = 0.6 * heights.get(heights.len() / 2).copied().unwrap_or(10.0);
    let mut sorted: Vec<&OcrWord> = words.iter().collect();
    sorted.sort_by(|a, b| (a.y + a.h / 2.0).total_cmp(&(b.y + b.h / 2.0)));
    let mut lines: Vec<Line> = Vec::new();
    for w in sorted {
        let cy = w.y + w.h / 2.0;
        match lines.last_mut() {
            Some(l) if (cy - l.y).abs() <= tol => {
                let n = l.words.len() as f64;
                l.y = (l.y * n + cy) / (n + 1.0);
                l.words.push(w.clone());
            }
            _ => lines.push(Line { y: cy, words: vec![w.clone()] }),
        }
    }
    for l in &mut lines {
        l.words.sort_by(|a, b| a.x.total_cmp(&b.x));
    }
    lines
}

/// Size names in the orders makers use, for naming a column the OCR found
/// numbers in but whose header word it missed.
const SEQUENCES: &[&[&str]] = &[
    &["XXS", "XS", "S", "M", "L", "XL", "XXL"],
    &["XS", "SM", "MD", "LG", "XL", "XXL"],
    &["S1", "S2", "S3", "S4", "S5", "S6"],
    &["P1", "P2", "P3", "P4", "P5"],
    &["1", "2", "3", "4", "5", "6"],
];

/// The table's columns, left to right: their size names and centres. The
/// centres come from where the rows' numbers line up (single-linkage on
/// their x centres, split at a gap of 2.5 text heights, kept with two or
/// more rows), each header name goes to its nearest column, and a column no
/// header word reached is named from the maker's sequence between its
/// named neighbours, or "?". Without numbers to line up, the header's own.
fn columns(header: &[(String, f64)], rows: &[Line]) -> (Vec<String>, Vec<f64>) {
    let mut cx: Vec<(f64, f64)> = rows
        .iter()
        .flat_map(|l| l.words.iter())
        .filter(|w| w.text.chars().any(|c| c.is_ascii_digit()))
        .map(|w| (w.x + w.w / 2.0, w.h))
        .collect();
    let header_x0 = header.iter().map(|h| h.1).fold(f64::INFINITY, f64::min);
    let header_pitch = if header.len() > 1 { (header.iter().map(|h| h.1).fold(f64::NEG_INFINITY, f64::max) - header_x0) / (header.len() - 1) as f64 } else { f64::INFINITY };
    // numbers well left of the first header word are in labels
    cx.retain(|(x, _)| *x > header_x0 - 0.75 * header_pitch.min(4.0 * header_x0.abs().max(1.0)));
    if cx.is_empty() {
        return (header.iter().map(|h| h.0.clone()).collect(), header.iter().map(|h| h.1).collect());
    }
    let mut hs: Vec<f64> = cx.iter().map(|c| c.1).collect();
    hs.sort_by(f64::total_cmp);
    let gap = 2.5 * hs[hs.len() / 2].max(1.0);
    cx.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut clusters: Vec<Vec<f64>> = vec![vec![cx[0].0]];
    for w in cx.windows(2) {
        if w[1].0 - w[0].0 > gap {
            clusters.push(Vec::new());
        }
        clusters.last_mut().map(|c| c.push(w[1].0));
    }
    let centres: Vec<f64> = clusters.iter().filter(|c| c.len() >= 2).map(|c| c.iter().sum::<f64>() / c.len() as f64).collect();
    if centres.len() < header.len() {
        return (header.iter().map(|h| h.0.clone()).collect(), header.iter().map(|h| h.1).collect());
    }
    let mut names: Vec<Option<String>> = vec![None; centres.len()];
    for (name, x) in header {
        let i = centres.iter().enumerate().min_by(|a, b| (a.1 - x).abs().total_cmp(&(b.1 - x).abs())).map(|(i, _)| i).unwrap_or(0);
        names[i].get_or_insert_with(|| name.clone());
    }
    // name the gaps from a sequence holding every found name in order
    let found: Vec<(usize, String)> = names.iter().enumerate().filter_map(|(i, n)| n.clone().map(|n| (i, n.to_ascii_uppercase()))).collect();
    let seq = SEQUENCES.iter().find(|seq| {
        let pos: Vec<Option<usize>> = found.iter().map(|(_, n)| seq.iter().position(|s| s == n)).collect();
        pos.iter().all(Option::is_some) && found.windows(2).zip(pos.windows(2)).all(|(f, p)| p[1].unwrap() as i64 - p[0].unwrap() as i64 == f[1].0 as i64 - f[0].0 as i64)
    });
    let sizes = names
        .iter()
        .enumerate()
        .map(|(i, n)| {
            n.clone().unwrap_or_else(|| {
                let guess = seq.and_then(|seq| {
                    let (fi, fname) = found.first()?;
                    let at = seq.iter().position(|s| s == fname)? as i64 + i as i64 - *fi as i64;
                    usize::try_from(at).ok().and_then(|a| seq.get(a)).map(|s| s.to_string())
                });
                guess.unwrap_or_else(|| "?".to_string())
            })
        })
        .collect();
    (sizes, centres)
}

/// Reads a geometry table from OCR words. `None` when no header of at least
/// two size names is found.
pub fn parse(words: &[OcrWord]) -> Option<GeometryTable> {
    let lines = lines_of(words);
    // the header: the line with the most size names (two or more), its sizes' centres
    let (hi, header) = lines
        .iter()
        .enumerate()
        .map(|(i, l)| (i, l.words.iter().filter(|w| is_size(&w.text)).collect::<Vec<_>>()))
        .filter(|(_, s)| s.len() >= 2)
        .max_by_key(|(i, s)| (s.len(), std::cmp::Reverse(*i)))?;
    let header_names: Vec<(String, f64)> = header.iter().map(|w| (w.text.trim_matches(|c: char| c == '"' || c == '”' || c == '″').to_string(), w.x + w.w / 2.0)).collect();
    let (sizes, centres) = columns(&header_names, &lines[hi + 1..]);
    let pitch = if centres.len() > 1 { (centres[centres.len() - 1] - centres[0]).abs() / (centres.len() - 1) as f64 } else { f64::INFINITY };
    let first_col = centres.iter().copied().fold(f64::INFINITY, f64::min);

    // each row's label, key and numbers by column, before units
    let edge = first_col - 0.5 * pitch;
    let mut raw: Vec<(String, Option<&'static str>, Vec<Vec<(f64, bool)>>)> = Vec::new();
    for line in &lines[hi + 1..] {
        // the label: words left of the first size column (less half a column)
        let label: Vec<&str> = line.words.iter().filter(|w| w.x + w.w / 2.0 < edge).map(|w| w.text.as_str()).collect();
        let label = label.join(" ");
        if label.trim().is_empty() {
            continue;
        }
        let mut cells: Vec<Vec<(f64, bool)>> = vec![Vec::new(); sizes.len()];
        for w in line.words.iter().filter(|w| w.x + w.w / 2.0 >= edge) {
            let cx = w.x + w.w / 2.0;
            let col = centres.iter().enumerate().min_by(|a, b| (a.1 - cx).abs().total_cmp(&(b.1 - cx).abs())).map(|(i, _)| i).unwrap_or(0);
            if (centres[col] - cx).abs() > 0.6 * pitch {
                continue;
            }
            let (nums, inch) = numbers_in(&w.text);
            cells[col].extend(nums.into_iter().map(|v| (v, inch)));
        }
        raw.push((label.clone(), key_for(&label), cells));
    }
    // a table in inches: a label says so, or most length rows read only as inches
    let said = lines.iter().any(|l| l.words.iter().any(|w| {
        let t = w.text.to_lowercase();
        t == "(in)" || t == "in." || t == "inches" || t == "(inches)"
    }));
    let (mut inch_rows, mut length_rows) = (0, 0);
    for (_, key, cells) in &raw {
        let Some(k) = key.filter(|k| !k.ends_with("_deg")) else { continue };
        let vals: Vec<f64> = cells.iter().flatten().map(|(v, _)| *v).collect();
        if vals.is_empty() {
            continue;
        }
        length_rows += 1;
        if vals.iter().filter(|v| only_inches(k, **v)).count() * 2 > vals.len() {
            inch_rows += 1;
        }
    }
    let inch_table = said || (length_rows > 0 && inch_rows * 2 > length_rows);
    let rows = raw
        .into_iter()
        .filter_map(|(label, key, cells)| {
            let values: Vec<Vec<f64>> = cells.into_iter().map(|c| c.into_iter().filter_map(|(v, inch)| in_unit(key, v, inch || inch_table)).collect()).collect();
            values.iter().any(|v| !v.is_empty()).then_some(GeometryRow { label, key, values })
        })
        .collect();
    Some(GeometryTable { sizes, rows })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Words laid out as a table: a label column then one column per size,
    /// 30 px rows, 120 px columns centred from x = 340, 8 px a character.
    fn table(header: &[&str], rows: &[(&str, &[&str])]) -> Vec<OcrWord> {
        let word = |text: &str, x: f64, y: f64| OcrWord { text: text.into(), x, y, w: 8.0 * text.chars().count() as f64, h: 14.0 };
        // a cell's words, centred on `cx`
        let cell = |text: &str, cx: f64, y: f64| {
            let parts: Vec<&str> = text.split(' ').collect();
            let width = 8.0 * text.chars().count() as f64;
            let mut x = cx - width / 2.0;
            parts
                .into_iter()
                .map(|p| {
                    let w = word(p, x, y);
                    x += 8.0 * (p.chars().count() + 1) as f64;
                    w
                })
                .collect::<Vec<_>>()
        };
        let mut out = vec![word("Geometry", 20.0, 0.0)];
        for (i, s) in header.iter().enumerate() {
            out.extend(cell(s, 340.0 + 120.0 * i as f64, 40.0));
        }
        for (r, (label, cells)) in rows.iter().enumerate() {
            let y = 70.0 + 30.0 * r as f64;
            out.extend(cell(label, 20.0 + 4.0 * label.chars().count() as f64, y + 1.0));
            for (i, c) in cells.iter().enumerate() {
                out.extend(cell(c, 340.0 + 120.0 * i as f64, y - 1.0));
            }
        }
        out
    }

    fn gt_force() -> Vec<OcrWord> {
        table(
            &["SM", "MD", "LG", "XL"],
            &[
                ("Head Tube Angle", &["63.5°", "63.5°", "63.5°", "63.5°"]),
                ("Seat Tube Angle (effective)", &["77.5°", "77.5°", "77.5°", "77.5°"]),
                ("Stack", &["618", "627", "645", "663"]),
                ("Reach", &["430", "455", "480", "505"]),
                ("Chainstay Length", &["435 / 445", "435 / 445", "435 / 445", "435 / 445"]),
                ("Wheelbase", &["1188", "1217", "1257", "1290"]),
                ("BB Drop", &["30", "30", "30", "30"]),
                ("Fork Rake", &["44", "44", "44", "44"]),
            ],
        )
    }

    #[test]
    fn parse_gt_force_finds_the_sizes_and_maps_every_row() {
        // Arrange
        let words = gt_force();

        // Act
        let t = parse(&words).unwrap();

        // Assert
        assert_eq!(t.sizes, ["SM", "MD", "LG", "XL"]);
        assert_eq!(t.rows.len(), 8);
        assert!(t.rows.iter().all(|r| r.key.is_some()), "{:?}", t.rows.iter().map(|r| (&r.label, r.key)).collect::<Vec<_>>());
    }

    #[test]
    fn for_size_large_long_chainstay_takes_the_second_setting() {
        // Arrange
        let t = parse(&gt_force()).unwrap();

        // Act
        let lg: std::collections::BTreeMap<_, _> = t.for_size(2, 1).into_iter().collect();

        // Assert
        assert_eq!(lg["geometry.head_angle_deg"], 63.5);
        assert_eq!(lg["geometry.stack_mm"], 645.0);
        assert_eq!(lg["geometry.reach_mm"], 480.0);
        assert_eq!(lg["geometry.chainstay_mm"], 445.0);
        assert_eq!(lg["geometry.bb_drop_mm"], 30.0);
        assert_eq!(lg["geometry.fork_offset_mm"], 44.0);
        // a single value per cell is the same whichever setting is picked
        assert_eq!(lg["geometry.wheelbase_mm"], 1257.0);
    }

    #[test]
    fn parse_inch_table_comes_out_in_millimetres() {
        // Arrange
        let words = table(&["S", "M", "L"], &[("Reach (in)", &["16.9", "17.7", "18.5"]), ("BB Drop", &["1.2", "1.2", "1.2"])]);

        // Act
        let t = parse(&words).unwrap();

        // Assert
        assert_eq!(t.rows[0].values[1], vec![449.6]);
        assert_eq!(t.rows[1].values[0], vec![30.5]);
    }

    #[test]
    fn key_for_tube_lengths_are_not_angles() {
        // Arrange / Act / Assert
        assert_eq!(key_for("Head Tube Length"), Some("geometry.head_tube_mm"));
        assert_eq!(key_for("Head tube angle"), Some("geometry.head_angle_deg"));
        assert_eq!(key_for("Seat Tube Length"), Some("geometry.seat_tube_mm"));
        assert_eq!(key_for("Effective Top Tube"), Some("geometry.top_tube_mm"));
        assert_eq!(key_for("Travel rear"), Some("shock.wheel_travel_mm"));
        assert_eq!(key_for("Top tube actual"), None);
        assert_eq!(key_for("Front center"), None);
    }

    #[test]
    fn parse_without_a_header_finds_nothing() {
        // Arrange
        let words = vec![OcrWord { text: "Reach".into(), x: 0.0, y: 0.0, w: 40.0, h: 14.0 }];

        // Act / Assert
        assert!(parse(&words).is_none());
    }

    #[test]
    fn parse_header_words_the_ocr_missed_are_named_from_the_sequence() {
        // Arrange: the header lost MD and LG, the numbers are all there
        let mut words = gt_force();
        words.retain(|w| !(w.text == "MD" || w.text == "LG"));

        // Act
        let t = parse(&words).unwrap();

        // Assert
        assert_eq!(t.sizes, ["SM", "MD", "LG", "XL"]);
        let lg: std::collections::BTreeMap<_, _> = t.for_size(2, 1).into_iter().collect();
        assert_eq!(lg["geometry.reach_mm"], 480.0);
    }

    #[test]
    fn parse_degree_sign_read_as_a_zero() {
        // Arrange
        let words = table(&["S", "M"], &[("Head angle", &["640", "64.50"]), ("Reach", &["450", "475"])]);

        // Act
        let t = parse(&words).unwrap();

        // Assert
        assert_eq!(t.rows[0].values, vec![vec![64.0], vec![64.5]]);
    }

    #[test]
    fn parse_gt_site_table_cells_as_whole_words() {
        // Arrange: GT's own layout, one OCR word per cell (labels whole), a spaced degree sign
        let mut words = Vec::new();
        let cell = |t: &str, cx: f64, y: f64| OcrWord { text: t.into(), x: cx - 4.0 * t.len() as f64, y, w: 8.0 * t.len() as f64, h: 9.0 };
        words.push(cell("Sizes (mm)", 60.0, 20.0));
        for (i, z) in ["SM", "MD", "LG", "XL"].iter().enumerate() {
            words.push(cell(z, 210.0 + 92.0 * i as f64, 20.0));
        }
        let rows: [(&str, [&str; 4]); 6] = [
            ("Head tube angle", ["63.5 °"; 4]),
            ("Top tube horizontal", ["563", "590", "617", "654"]),
            ("Top tube actual", ["544", "565", "586", "620"]),
            ("Wheelbase", ["1211/1221", "1240/1250", "1270/1280", "1309/1319"]),
            ("Bottom bracket drop", ["30"; 4]),
            ("Travel rear", ["160"; 4]),
        ];
        for (r, (label, cells)) in rows.iter().enumerate() {
            let y = 60.0 + 38.5 * r as f64;
            words.push(OcrWord { text: (*label).into(), x: 23.0, y, w: 8.0 * label.len() as f64 * 0.6, h: 9.0 });
            for (i, c) in cells.iter().enumerate() {
                words.push(cell(c, 210.0 + 92.0 * i as f64, y));
            }
        }

        // Act
        let t = parse(&words).unwrap();
        let lg: std::collections::BTreeMap<_, _> = t.for_size(2, 1).into_iter().collect();

        // Assert
        assert_eq!(t.sizes, ["SM", "MD", "LG", "XL"]);
        assert_eq!(lg["geometry.head_angle_deg"], 63.5);
        assert_eq!(lg["geometry.top_tube_mm"], 617.0);
        assert_eq!(lg["geometry.wheelbase_mm"], 1280.0);
        assert_eq!(lg["geometry.bb_drop_mm"], 30.0);
        assert_eq!(lg["shock.wheel_travel_mm"], 160.0);
    }
}
