//! On-disk filename derivation for `session.json`-adjacent human-facing
//! names (port of `session_filename.dart`). C4 §2 makes `session_id` the
//! *directory* name now (not this module's concern) — this module is for
//! the still-timestamp-derived names the design carries forward:
//! `workbooks/<file_name>.idl1wb` (C4 §2) and any future human-facing
//! export filename.

/// Formats a local-time instant, already decomposed into its calendar
/// fields, as `YYYY-MM-DD_HH-MM-SS` — every component zero-padded, no
/// colons (valid on every target filesystem). Mirrors
/// `formatSessionFileBase` exactly; this crate has no date/time library
/// dependency (none pinned in the ecosystem report, none in `core`'s
/// `Cargo.toml`), so the caller does the UTC→local conversion and calendar
/// decomposition before calling this.
pub fn format_session_file_base(year: i32, month: u32, day: u32, hour: u32, minute: u32, second: u32) -> String {
    format!("{year:04}-{month:02}-{day:02}_{hour:02}-{minute:02}-{second:02}")
}

/// Returns the first of `base`, `<base>-2`, `<base>-3`, … for which
/// `is_taken` reports `false`.
pub fn unique_file_base(base: &str, mut is_taken: impl FnMut(&str) -> bool) -> String {
    if !is_taken(base) {
        return base.to_string();
    }
    let mut n = 2u32;
    loop {
        let candidate = format!("{base}-{n}");
        if !is_taken(&candidate) {
            return candidate;
        }
        n += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_session_file_base_zero_pads_every_component() {
        // Arrange / Act
        let s = format_session_file_base(2026, 9, 3, 8, 5, 0);

        // Assert
        assert_eq!(s, "2026-09-03_08-05-00");
    }

    #[test]
    fn unique_file_base_returns_the_base_when_untaken() {
        // Act
        let s = unique_file_base("fork-tuning", |_| false);

        // Assert
        assert_eq!(s, "fork-tuning");
    }

    #[test]
    fn unique_file_base_appends_suffix_until_untaken() {
        // Arrange
        let taken = ["a", "a-2", "a-3"];

        // Act
        let s = unique_file_base("a", |c| taken.contains(&c));

        // Assert
        assert_eq!(s, "a-4");
    }
}
