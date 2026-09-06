//! Front-matter parsing for workbook v3 (C2 §1): the YAML block bounded by
//! `---` lines at the top of a `.idl1wb` document.

use std::collections::HashMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::error::{self, WorkbookError};

/// Workbook-level unit-system *preference* (C2 §1). Consumed only by the
/// editor UI (L6) for axis-label/number-format suggestions — has no effect
/// on parsing or evaluation. Defaults to [`Self::Si`] when the `units` key
/// is absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum UnitsPref {
    /// Metric (m, m/s, kg, …) — the default.
    Si,
    /// Imperial/US customary (ft, mph, lb, …).
    Imperial,
}

impl Default for UnitsPref {
    fn default() -> Self {
        UnitsPref::Si
    }
}

/// A front-matter constant's raw value (C2 §1, §3.1): a bare number, or a
/// number with a display-only unit suffix that is never dimensionally
/// checked or converted (matches idl0's existing `constants` semantics).
#[derive(Debug, Clone, PartialEq)]
pub enum ConstantRaw {
    /// A bare YAML number, unitless.
    Number(f64),
    /// Parsed from a `"<number> <unit>"` YAML string (C2 §3.1's unit-suffix
    /// grammar). `unit_display` is metadata only.
    WithUnit { value: f64, unit_display: String },
}

/// The two YAML shapes a constant's value may take on the wire — resolved
/// into [`ConstantRaw`] by that type's own `Deserialize` impl below.
#[derive(Deserialize)]
#[serde(untagged)]
enum ConstantYamlValue {
    Number(f64),
    Text(String),
}

impl<'de> Deserialize<'de> for ConstantRaw {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match ConstantYamlValue::deserialize(deserializer)? {
            ConstantYamlValue::Number(value) => Ok(ConstantRaw::Number(value)),
            ConstantYamlValue::Text(text) => match parse_unit_suffix(&text) {
                Some((value, unit_display)) => Ok(ConstantRaw::WithUnit { value, unit_display }),
                None => Err(serde::de::Error::custom(format!(
                    "constant value '{text}' is neither a bare number nor a '<number> <unit>' string"
                ))),
            },
        }
    }
}

/// Serialises the inverse of [`ConstantRaw`]'s custom `Deserialize` impl: a
/// bare number stays a YAML number, `WithUnit` becomes the `"<value>
/// <unit>"` string [`parse_unit_suffix`] accepts back. `value`'s `Display`
/// formatting (Rust's default `f64` formatter) never emits a value
/// `parse_unit_suffix`'s grammar rejects, so this round-trips through
/// [`parse_front_matter`] for every value this type can hold.
impl Serialize for ConstantRaw {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            ConstantRaw::Number(value) => serializer.serialize_f64(*value),
            ConstantRaw::WithUnit { value, unit_display } => {
                serializer.serialize_str(&format!("{value} {unit_display}"))
            }
        }
    }
}

/// Parses a front-matter constant string against C2 §3.1's unit-suffix
/// grammar: `/^\s*(-?\d+(\.\d+)?([eE][+-]?\d+)?)\s+(\S.*)\s*$/`. Hand-written
/// rather than via the `regex` crate (not one of Task 1's two new
/// dependencies, see the plan's compute rules). Returns `(value,
/// unit_display)` — `unit_display` has its own leading/trailing whitespace
/// trimmed — or `None` if `raw` doesn't match the grammar.
fn parse_unit_suffix(raw: &str) -> Option<(f64, String)> {
    let s = raw.trim_start();
    let bytes = s.as_bytes();
    let mut i = 0usize;

    if i < bytes.len() && bytes[i] == b'-' {
        i += 1;
    }
    let digits_start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == digits_start {
        return None; // no leading integer digits — `-?\d+` didn't match
    }

    if i < bytes.len() && bytes[i] == b'.' {
        let mut j = i + 1;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
        }
        if j > i + 1 {
            i = j; // `(\.\d+)?` matched; otherwise leave `i` before the dot
        }
    }

    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        let mut j = i + 1;
        if j < bytes.len() && (bytes[j] == b'+' || bytes[j] == b'-') {
            j += 1;
        }
        let exp_digits_start = j;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
        }
        if j > exp_digits_start {
            i = j; // `([eE][+-]?\d+)?` matched; otherwise leave `i` before the 'e'
        }
    }

    let number_str = &s[..i];
    let rest = &s[i..];
    let after_ws = rest.trim_start();
    if after_ws.len() == rest.len() {
        return None; // `\s+` requires at least one separating whitespace char
    }
    let unit = after_ws.trim_end();
    if unit.is_empty() {
        return None; // `\S.*` requires at least one non-space unit char
    }

    let value: f64 = number_str.parse().ok()?;
    Some((value, unit.to_string()))
}

fn default_version() -> u32 {
    3
}

/// The `.idl1wb` front-matter block (C2 §1). `version` defaults to `3` only
/// when the YAML key is absent — an explicit `version: 2` survives here
/// unmodified; [`super::parse_workbook`] is where a non-`3` value becomes
/// [`WorkbookErrorKind::UnsupportedWorkbookVersion`].
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FrontMatter {
    /// Stable workbook identity (C2 §1) — required to be a UUIDv4 string;
    /// checked by [`parse_front_matter`], not by this struct's own
    /// deserialization (which only requires it be present and a string).
    pub id: String,
    /// Display name.
    pub name: String,
    /// Named scalars for math-cell literal substitution (C2 §1, §3.1); empty
    /// when the `constants` key is absent. Omitted entirely by
    /// [`render_front_matter`] when empty, rather than written as `{}`, to
    /// keep a freshly created workbook's front matter minimal (C2 §1 lists
    /// `constants` as optional with a `{}` default either way, so an absent
    /// key and an explicit empty map parse identically).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub constants: HashMap<String, ConstantRaw>,
    /// Editor unit-system preference; defaults to SI.
    #[serde(default)]
    pub units: UnitsPref,
    /// Schema version; defaults to `3` only when the key is absent.
    #[serde(default = "default_version")]
    pub version: u32,
}

/// Splits `markdown` into its front-matter YAML and the remaining document
/// body (C2 §1's `document ::= front_matter "\n" body` grammar), parses the
/// YAML, and validates that `id` is a well-formed UUIDv4.
///
/// Every failure mode here — no `---`-delimited block, YAML that doesn't
/// deserialize into [`FrontMatter`] at all (a syntax error, a required field
/// missing, an unrecognised `units` value, a constant string matching
/// neither `ConstantRaw` shape, …), or a present-but-malformed/missing `id`
/// — collapses to a single [`WorkbookErrorKind::MissingFrontMatterId`]: the
/// plan's Step 4 lists "YAML that doesn't parse at all" as its own
/// front-matter-fatal case alongside `MissingFrontMatterId`, but C2 §3.5.A
/// defines no separate error kind for it, and its own message ("front matter
/// is missing a valid 'id'") already covers "we couldn't even parse the
/// front matter to find one" — so this task does not invent a fourth kind
/// for it. `version` validation is deliberately *not* done here — see
/// [`super::parse_workbook`].
pub fn parse_front_matter(markdown: &str) -> Result<(FrontMatter, &str), WorkbookError> {
    let missing_id = error::missing_front_matter_id;

    let rest = markdown.strip_prefix("---\n").ok_or_else(missing_id)?;
    let (yaml_block, body) = rest.split_once("\n---\n").ok_or_else(missing_id)?;

    let front_matter: FrontMatter = serde_yaml_ng::from_str(yaml_block).map_err(|_| missing_id())?;

    let is_uuid_v4 = uuid::Uuid::parse_str(&front_matter.id)
        .map(|u| u.get_version() == Some(uuid::Version::Random))
        .unwrap_or(false);
    if !is_uuid_v4 {
        return Err(missing_id());
    }

    Ok((front_matter, body))
}

/// Renders `fm` as a `.idl1wb` front-matter block (C2 §1's `front_matter ::=
/// "---\n" yaml_block "---\n"`) — the inverse of [`parse_front_matter`] (R75:
/// front matter is serialised by core, never hand-built in `tauri`).
/// `serde_yaml_ng` quotes/escapes any YAML-significant byte in `fm.name`
/// (a colon-space, a leading `#`/`-`/`&`/`*`/`!`/`%`/`@`, embedded quotes, a
/// newline, leading/trailing spaces, …) itself, so the result always
/// round-trips through [`parse_front_matter`] back to the same `FrontMatter`
/// — that round-trip, not the exact bytes emitted, is this function's
/// contract; `serde_yaml_ng` is free to choose block, quoted, or literal
/// scalar style for any given string.
pub fn render_front_matter(fm: &FrontMatter) -> String {
    let yaml_block = serde_yaml_ng::to_string(fm).expect("FrontMatter has no non-serialisable field");
    format!("---\n{yaml_block}---\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::error::WorkbookErrorKind;

    const VALID_ID: &str = "9f3c1e2d-4b6a-4f1c-9c3d-2a7e8f9b0c1d";

    fn doc(front_matter_yaml: &str) -> String {
        format!("---\n{front_matter_yaml}\n---\nbody\n")
    }

    #[test]
    fn front_matter_version_key_absent_defaults_to_3() {
        // Arrange
        let text = doc(&format!("id: {VALID_ID}\nname: Fork tuning"));

        // Act
        let (fm, _body) = parse_front_matter(&text).unwrap();

        // Assert
        assert_eq!(fm.version, 3);
    }

    #[test]
    fn front_matter_version_2_explicit_is_not_silently_upgraded() {
        // Arrange
        let text = doc(&format!("id: {VALID_ID}\nname: Fork tuning\nversion: 2"));

        // Act
        let (fm, _body) = parse_front_matter(&text).unwrap();

        // Assert
        assert_eq!(fm.version, 2);
    }

    #[test]
    fn front_matter_missing_id_is_missing_front_matter_id() {
        // Arrange
        let text = doc("name: Fork tuning");

        // Act
        let err = parse_front_matter(&text).unwrap_err();

        // Assert
        assert_eq!(err.kind, WorkbookErrorKind::MissingFrontMatterId);
        assert_eq!(err.cell_id, "front-matter");
    }

    #[test]
    fn front_matter_id_not_a_uuidv4_is_missing_front_matter_id() {
        // Arrange
        let text = doc("id: not-a-uuid\nname: Fork tuning");

        // Act
        let err = parse_front_matter(&text).unwrap_err();

        // Assert
        assert_eq!(err.kind, WorkbookErrorKind::MissingFrontMatterId);
    }

    #[test]
    fn constant_unit_suffix_82_kg_parses_to_82_0_some_kg() {
        // Arrange
        let text = doc(&format!("id: {VALID_ID}\nname: Fork tuning\nconstants:\n  rider_mass_kg: \"82 kg\""));

        // Act
        let (fm, _body) = parse_front_matter(&text).unwrap();

        // Assert
        assert_eq!(
            fm.constants.get("rider_mass_kg"),
            Some(&ConstantRaw::WithUnit { value: 82.0, unit_display: "kg".to_string() })
        );
    }

    #[test]
    fn constant_bare_number_9_80665_parses_to_9_80665_none() {
        // Arrange
        let text = doc(&format!("id: {VALID_ID}\nname: Fork tuning\nconstants:\n  g: 9.80665"));

        // Act
        let (fm, _body) = parse_front_matter(&text).unwrap();

        // Assert
        assert_eq!(fm.constants.get("g"), Some(&ConstantRaw::Number(9.80665)));
    }

    /// A minimal, freshly-minted [`FrontMatter`] with the given `name` —
    /// mirrors `create_workbook_via`'s call shape (`constants` empty, `units`
    /// defaulted, `version: 3`).
    fn fresh(name: &str) -> FrontMatter {
        FrontMatter {
            id: VALID_ID.to_string(),
            name: name.to_string(),
            constants: HashMap::new(),
            units: UnitsPref::Si,
            version: 3,
        }
    }

    /// Renders `fm`, re-parses the result, and returns the round-tripped
    /// [`FrontMatter`] — the shared shape of every `render_front_matter`
    /// round-trip test below.
    fn round_trip(fm: &FrontMatter) -> FrontMatter {
        let markdown = render_front_matter(fm);
        parse_front_matter(&markdown).unwrap().0
    }

    #[test]
    fn render_front_matter_a_name_with_a_colon_space_round_trips_byte_for_byte() {
        // Arrange
        let fm = fresh("Wheel: front");

        // Act
        let round_tripped = round_trip(&fm);

        // Assert
        assert_eq!(round_tripped.name, "Wheel: front");
    }

    #[test]
    fn render_front_matter_a_name_with_a_space_then_hash_round_trips_byte_for_byte() {
        // Arrange
        let fm = fresh("Test #1");

        // Act
        let round_tripped = round_trip(&fm);

        // Assert
        assert_eq!(round_tripped.name, "Test #1");
    }

    #[test]
    fn render_front_matter_a_name_with_double_quotes_round_trips_byte_for_byte() {
        // Arrange
        let fm = fresh("Lap \"the good one\"");

        // Act
        let round_tripped = round_trip(&fm);

        // Assert
        assert_eq!(round_tripped.name, "Lap \"the good one\"");
    }

    #[test]
    fn render_front_matter_a_name_with_single_quotes_round_trips_byte_for_byte() {
        // Arrange
        let fm = fresh("Rider's setup");

        // Act
        let round_tripped = round_trip(&fm);

        // Assert
        assert_eq!(round_tripped.name, "Rider's setup");
    }

    #[test]
    fn render_front_matter_a_name_with_an_embedded_newline_round_trips_byte_for_byte() {
        // Arrange
        let fm = fresh("Fork tuning\nsecond line");

        // Act
        let round_tripped = round_trip(&fm);

        // Assert
        assert_eq!(round_tripped.name, "Fork tuning\nsecond line");
    }

    #[test]
    fn render_front_matter_a_name_with_leading_and_trailing_spaces_round_trips_byte_for_byte() {
        // Arrange
        let fm = fresh("  Fork tuning  ");

        // Act
        let round_tripped = round_trip(&fm);

        // Assert
        assert_eq!(round_tripped.name, "  Fork tuning  ");
    }

    #[test]
    fn render_front_matter_version_3_and_id_round_trip() {
        // Arrange
        let fm = fresh("Fork tuning");

        // Act
        let round_tripped = round_trip(&fm);

        // Assert
        assert_eq!(round_tripped.version, 3);
        assert_eq!(round_tripped.id, VALID_ID);
    }
}
