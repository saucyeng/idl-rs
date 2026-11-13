//! Wall-clock time, `std`-only (no date/time crate is pinned in this workspace).

/// Milliseconds since the Unix epoch (UTC) now; 0 if the system clock is before the epoch.
pub fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}
