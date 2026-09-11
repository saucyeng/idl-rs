//! The app's memory budget and the pre-allocation failsafe (ruling R203.4).
//!
//! Every allocation that scales with a session — an import, a channel
//! decode, a raster — is sized before it is attempted and refused, as the
//! typed `resource_exhausted` error (C3 §1), when it would not fit. The app
//! shows that as a toast and the notebook keeps running. Nothing here ever
//! aborts, panics or `unwrap`s on an allocation: an out-of-memory abort is
//! precisely the failure this module exists to replace.

use std::sync::OnceLock;

use crate::error::{IpcError, IpcErrorKind};

/// Hard ceiling on the budget, regardless of how much RAM the machine has:
/// 2 GiB. A workstation with 128 GB of RAM should not let one notebook
/// session cache 32 GB of decoded samples.
pub const MAX_BUDGET_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Fraction of physical RAM the budget may claim, as `(numerator,
/// denominator)` — 25 % (ruling R203.2). The app shares the machine with
/// the OS, a browser-engine webview and whatever else is running; a
/// quarter is what it may treat as its own.
pub const BUDGET_FRACTION: (u64, u64) = (1, 4);

/// Floor on the budget: 256 MiB. Reached only when physical memory reads
/// implausibly small (a container limit, or a platform where the query
/// fails and returns 0). Without it the failsafe would refuse every
/// session on such a machine rather than merely the large ones.
pub const MIN_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

/// Safety margin applied to every estimate before it is compared with the
/// budget, as `(numerator, denominator)` — the estimate is multiplied by
/// 5/4. The estimates are already ceilings, but a decode's transient peak
/// is not perfectly predictable (allocator slack, an Arrow buffer's growth
/// doubling), and being refused one session early is a far better failure
/// than an abort mid-pan.
pub const ESTIMATE_MARGIN: (u64, u64) = (5, 4);

/// Physical RAM, bytes, read once per process.
static PHYSICAL_MEMORY_BYTES: OnceLock<u64> = OnceLock::new();

/// Total physical memory in bytes, read once at first use and cached.
///
/// Uses `sysinfo`'s total-memory query only — no other part of that crate
/// is used, and it is refreshed once rather than polled, because the
/// budget is a startup constant, not a live signal. The standard library
/// exposes no physical-memory API on any supported platform, which is why
/// there is a dependency here at all. `0` when the query fails; callers go
/// through [`budget_bytes`], which floors it.
pub fn physical_memory_bytes() -> u64 {
    *PHYSICAL_MEMORY_BYTES.get_or_init(|| {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        sys.total_memory()
    })
}

/// The app's memory budget in bytes: `min(2 GiB, 25 % of physical RAM)`,
/// floored at [`MIN_BUDGET_BYTES`] (ruling R203.2).
///
/// One number serves two jobs: it caps the session cache's residency, and
/// it is the ceiling every [`ensure_fits`] check compares against. They are
/// deliberately the same number — a decode the cache could not hold is a
/// decode the app should not attempt.
pub fn budget_bytes() -> u64 {
    let (num, den) = BUDGET_FRACTION;
    let share = physical_memory_bytes() / den * num;
    share.min(MAX_BUDGET_BYTES).max(MIN_BUDGET_BYTES)
}

/// The C3 §1 `resource_exhausted` error for an allocation of
/// `needed_bytes` against `budget_bytes`, with a human-readable `hint`
/// naming what was being attempted.
///
/// `detail` is exactly C3 §1's shape: `{ needed_bytes, budget_bytes, hint }`.
/// The message carries both figures in GB so a toast is readable without
/// the app doing arithmetic of its own.
pub fn resource_exhausted(needed_bytes: u64, budget: u64, hint: &str) -> IpcError {
    IpcError::with_detail(
        IpcErrorKind::ResourceExhausted,
        format!(
            "Session too large for available memory: {:.1} GB needed, {:.1} GB budget",
            needed_bytes as f64 / 1e9,
            budget as f64 / 1e9
        ),
        serde_json::json!({
            "needed_bytes": needed_bytes,
            "budget_bytes": budget,
            "hint": hint,
        }),
    )
}

/// `needed_bytes` with [`ESTIMATE_MARGIN`] applied — the figure every
/// budget decision is actually made against.
///
/// Its own function because the margin is now applied in two places that
/// must agree: [`ensure_fits`]'s one-shot check, and the byte-counting
/// reservation `session_cache::SessionCache::reserve` holds for a decode's
/// duration (ruling R211.2). Saturating, so an absurd estimate clamps
/// rather than wrapping to a small number that would pass.
pub fn with_estimate_margin(needed_bytes: u64) -> u64 {
    let (num, den) = ESTIMATE_MARGIN;
    needed_bytes.saturating_mul(num) / den
}

/// Refuses `needed_bytes` when it would not fit the budget with
/// [`ESTIMATE_MARGIN`] applied, as [`IpcErrorKind::ResourceExhausted`].
///
/// `hint` names the work being sized ("decode channel IMU0_AccelX",
/// "import 2026-09-07_09-43-52.idl0") and reaches the UI verbatim in
/// `detail.hint`. `Ok(())` means the caller may proceed; it is not a
/// reservation, and nothing here allocates.
pub fn ensure_fits(needed_bytes: u64, hint: &str) -> Result<(), IpcError> {
    let budget = budget_bytes();
    let with_margin = with_estimate_margin(needed_bytes);
    if with_margin > budget {
        return Err(resource_exhausted(with_margin, budget, hint));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_bytes_on_this_machine_is_within_its_documented_floor_and_ceiling() {
        // Arrange + Act
        let budget = budget_bytes();

        // Assert
        assert!(budget >= MIN_BUDGET_BYTES);
        assert!(budget <= MAX_BUDGET_BYTES);
    }

    #[test]
    fn physical_memory_bytes_reports_a_plausible_nonzero_total_for_the_host() {
        // Arrange + Act
        let total = physical_memory_bytes();

        // Assert — any machine that can build this crate has at least 1 GB.
        assert!(total >= 1_000_000_000, "physical memory read back as {total} bytes");
    }

    #[test]
    fn ensure_fits_an_allocation_far_under_the_budget_is_allowed() {
        // Arrange
        let needed = MIN_BUDGET_BYTES / 8;

        // Act
        let got = ensure_fits(needed, "decode channel Speed");

        // Assert
        assert!(got.is_ok());
    }

    #[test]
    fn ensure_fits_an_allocation_past_the_ceiling_is_refused_as_resource_exhausted() {
        // Arrange
        let needed = MAX_BUDGET_BYTES * 4;

        // Act
        let err = ensure_fits(needed, "decode session s1").unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::ResourceExhausted);
        let detail = err.detail.unwrap();
        assert_eq!(detail["hint"], "decode session s1");
        assert!(detail["needed_bytes"].as_u64().unwrap() > detail["budget_bytes"].as_u64().unwrap());
    }

    #[test]
    fn ensure_fits_applies_the_margin_so_an_estimate_just_under_the_budget_is_still_refused() {
        // Arrange — 90 % of the budget, which the 5/4 margin lifts past it.
        let budget = budget_bytes();
        let needed = budget / 10 * 9;

        // Act
        let err = ensure_fits(needed, "raster").unwrap_err();

        // Assert
        assert_eq!(err.kind, IpcErrorKind::ResourceExhausted);
    }

    #[test]
    fn resource_exhausted_detail_carries_the_three_c3_keys_and_the_message_names_both_figures() {
        // Arrange + Act
        let err = resource_exhausted(1_900_000_000, 1_200_000_000, "decode session s1");

        // Assert
        assert_eq!(err.kind, IpcErrorKind::ResourceExhausted);
        assert_eq!(err.message, "Session too large for available memory: 1.9 GB needed, 1.2 GB budget");
        let detail = err.detail.unwrap();
        assert_eq!(detail["needed_bytes"], 1_900_000_000u64);
        assert_eq!(detail["budget_bytes"], 1_200_000_000u64);
        assert_eq!(detail["hint"], "decode session s1");
    }
}
