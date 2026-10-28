//! The app's memory budget and the pre-allocation failsafe (ruling R203.4),
//! sized from the machine (evaluation-memory spec §4.2.G, Phase 3).
//!
//! Every allocation that scales with a session — an import, a channel
//! decode, a raster — is sized before it is attempted and refused, as the
//! typed `resource_exhausted` error (C3 §1), when it would not fit. The app
//! shows that as a toast and the notebook keeps running. Nothing here ever
//! aborts, panics or `unwrap`s on an allocation: an out-of-memory abort is
//! precisely the failure this module exists to replace.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::error::{IpcError, IpcErrorKind};

/// Fraction of physical RAM the budget may claim, as `(numerator,
/// denominator)` — 50 % (evaluation-memory spec §4.2.G, Isaac's Q3 answer,
/// 2026-10-27, replacing ruling R203.2's 25 % capped at 2 GiB). There is no
/// absolute cap: a larger machine gets a larger budget. Free physical memory
/// is deliberately not the limit — the OS pages other programs out to make
/// room — the commit guard ([`MachineBudget::budget_now`]) is.
pub const BUDGET_FRACTION: (u64, u64) = (1, 2);

/// Floor on the budget: 256 MiB. Reached only when physical memory reads
/// implausibly small (a container limit, or a platform where the query
/// fails and returns 0). Without it the failsafe would refuse every
/// session on such a machine rather than merely the large ones.
pub const MIN_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

/// Floor on the commit headroom an admission must leave: 512 MiB
/// (spec §4.2.G: `headroom = max(512 MB, 5 % of RAM)`).
pub const MIN_HEADROOM_BYTES: u64 = 512 * 1024 * 1024;

/// The headroom as a fraction of physical RAM, `(numerator, denominator)`:
/// 5 %.
pub const HEADROOM_FRACTION: (u64, u64) = (1, 20);

/// How long a commit reading is reused before the machine is asked again:
/// 250 ms (spec §4.2.G). Admission decisions run far more often than that.
pub const PROBE_REFRESH: Duration = Duration::from_millis(250);

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
/// `0` when the query fails; callers go through [`budget_bytes`], which
/// floors it.
pub fn physical_memory_bytes() -> u64 {
    *PHYSICAL_MEMORY_BYTES.get_or_init(|| {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        sys.total_memory()
    })
}

/// The budget's ceiling for a machine with `physical_bytes` of RAM:
/// [`BUDGET_FRACTION`] of it, floored at [`MIN_BUDGET_BYTES`], never capped.
pub fn ceiling_for(physical_bytes: u64) -> u64 {
    let (num, den) = BUDGET_FRACTION;
    (physical_bytes / den * num).max(MIN_BUDGET_BYTES)
}

/// The commit headroom an admission must leave on a machine with
/// `physical_bytes` of RAM: `max(512 MiB, 5 % of RAM)`.
pub fn headroom_for(physical_bytes: u64) -> u64 {
    let (num, den) = HEADROOM_FRACTION;
    (physical_bytes / den * num).max(MIN_HEADROOM_BYTES)
}

/// This machine's budget ceiling `B_max`, bytes: [`ceiling_for`] its
/// physical RAM. The one-shot checks ([`ensure_fits`]) compare against it;
/// the session cache's ledger admits against [`MachineBudget::budget_now`],
/// which is never more.
pub fn budget_bytes() -> u64 {
    ceiling_for(physical_memory_bytes())
}

/// One reading of the machine's commit state, bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitSample {
    /// Commit this process holds right now, as far as it knows (the Rust
    /// heap, from the counting allocator when the app installed it); `0`
    /// when unknown, which only makes admission stricter.
    pub own_commit_bytes: u64,
    /// Commit the machine can still grant: the commit limit less the commit
    /// in use, **plus** what the page file may still grow by when its size
    /// may grow (Isaac's Q3 answer: count page-file growth).
    pub available_commit_bytes: u64,
}

/// Where the budget reads the machine from: the real system in the app, a
/// fake in tests.
pub trait MemoryProbe: Send + Sync {
    /// Physical RAM, bytes.
    fn physical_bytes(&self) -> u64;
    /// The commit state now. Called at most once per [`PROBE_REFRESH`].
    fn commit(&self) -> CommitSample;
}

/// The real machine: `sysinfo` for RAM and commit (on Windows its swap
/// figures are the commit limit and the commit total less physical RAM, so
/// `free_swap` is exactly the commit still available), the registry for how
/// far the page file may grow, and the counting allocator for this process's
/// own heap.
#[derive(Debug, Default)]
pub struct SystemProbe;

impl MemoryProbe for SystemProbe {
    fn physical_bytes(&self) -> u64 {
        physical_memory_bytes()
    }

    fn commit(&self) -> CommitSample {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        let available = if cfg!(windows) {
            sys.free_swap()
        } else {
            sys.available_memory().saturating_add(sys.free_swap())
        };
        let growth = if cfg!(windows) { page_file_growth_bytes(sys.total_swap(), sys.total_memory()) } else { 0 };
        CommitSample {
            own_commit_bytes: crate::crash::live_heap_bytes(),
            available_commit_bytes: available.saturating_add(growth),
        }
    }
}

/// How many bytes the page file can still grow by, given its current size
/// `current_bytes` (the commit limit less physical RAM) on a machine with
/// `physical_bytes` of RAM. `0` when the setting cannot be read.
///
/// Read from `HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Memory
/// Management`'s `PagingFiles`: an entry `path min max` (MB) grows to `max`;
/// `path 0 0` and `?:\pagefile.sys` are system-managed, which grow to
/// `max(3 × RAM, 4 GB)` (Microsoft's documented system-managed maximum).
// TODO(idl0): the growth is bounded neither by the volume's free space nor
// by the system-managed cap of 1/8 of the volume; a full disk makes it
// optimistic.
fn page_file_growth_bytes(current_bytes: u64, physical_bytes: u64) -> u64 {
    let Some(entries) = paging_files() else { return 0 };
    let max_bytes: u64 = entries.iter().map(|e| page_file_max_bytes(e, physical_bytes)).sum();
    max_bytes.saturating_sub(current_bytes)
}

/// The most one `PagingFiles` entry lets its page file reach, bytes.
fn page_file_max_bytes(entry: &str, physical_bytes: u64) -> u64 {
    let system_managed = physical_bytes.saturating_mul(3).max(4 << 30);
    let parts: Vec<&str> = entry.split_whitespace().collect();
    if parts.first().is_some_and(|p| p.starts_with('?')) {
        return system_managed;
    }
    let min_mb = parts.get(1).and_then(|v| v.parse::<u64>().ok());
    let max_mb = parts.get(2).and_then(|v| v.parse::<u64>().ok());
    match (min_mb, max_mb) {
        (Some(0), Some(0)) | (None, None) => system_managed,
        (_, Some(max_mb)) => max_mb << 20,
        _ => 0,
    }
}

#[cfg(windows)]
fn paging_files() -> Option<Vec<String>> {
    let key = winreg::RegKey::predef(winreg::enums::HKEY_LOCAL_MACHINE)
        .open_subkey("SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Memory Management")
        .ok()?;
    key.get_value::<Vec<String>, _>("PagingFiles").ok()
}

#[cfg(not(windows))]
fn paging_files() -> Option<Vec<String>> {
    None
}

/// The machine-aware budget (evaluation-memory spec §4.2.G):
///
/// ```text
/// B_max  = physical RAM × 50 %                     (no absolute cap)
/// B_live = min(own commit, accounted) + available commit − headroom
/// B      = min(B_max, B_live)
/// ```
///
/// `accounted` is what the caller's ledger already holds. Only that much of
/// the process's own commit is credited back, so memory the ledger never
/// counted (the app's baseline) is not mistaken for budget, and a
/// reservation granted but not yet allocated is not granted twice.
pub struct MachineBudget {
    probe: Box<dyn MemoryProbe>,
    last: Mutex<Option<(Instant, CommitSample)>>,
}

impl std::fmt::Debug for MachineBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MachineBudget").field("ceiling_bytes", &self.ceiling_bytes()).finish_non_exhaustive()
    }
}

impl MachineBudget {
    /// A budget reading `probe`.
    pub fn new(probe: Box<dyn MemoryProbe>) -> Self {
        MachineBudget { probe, last: Mutex::new(None) }
    }

    /// `B_max`, bytes: the most this budget ever grants.
    pub fn ceiling_bytes(&self) -> u64 {
        ceiling_for(self.probe.physical_bytes())
    }

    /// The commit headroom admission leaves, bytes.
    pub fn headroom_bytes(&self) -> u64 {
        headroom_for(self.probe.physical_bytes())
    }

    /// `B` now, bytes, for a ledger already holding `accounted_bytes`. The
    /// commit reading is at most [`PROBE_REFRESH`] old.
    pub fn budget_now(&self, accounted_bytes: u64) -> u64 {
        let sample = {
            let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
            match *last {
                Some((at, sample)) if at.elapsed() < PROBE_REFRESH => sample,
                _ => {
                    let sample = self.probe.commit();
                    *last = Some((Instant::now(), sample));
                    sample
                }
            }
        };
        let live = sample
            .own_commit_bytes
            .min(accounted_bytes)
            .saturating_add(sample.available_commit_bytes)
            .saturating_sub(self.headroom_bytes());
        self.ceiling_bytes().min(live)
    }
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

    /// A machine with `physical` bytes of RAM and `available` bytes of
    /// commit to spare, counting how often it is asked.
    struct FakeProbe {
        physical: u64,
        available: u64,
        asked: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl MemoryProbe for FakeProbe {
        fn physical_bytes(&self) -> u64 {
            self.physical
        }

        fn commit(&self) -> CommitSample {
            self.asked.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            CommitSample { own_commit_bytes: GB, available_commit_bytes: self.available }
        }
    }

    fn fake(physical: u64, available: u64) -> MachineBudget {
        MachineBudget::new(Box::new(FakeProbe { physical, available, asked: Default::default() }))
    }

    const GB: u64 = 1 << 30;

    #[test]
    fn budget_fake_probe_with_64_gb_physical_b_max_scales_no_2_gib_ceiling() {
        // Arrange — plenty of commit to spare.
        let budget = fake(64 * GB, 200 * GB);

        // Act
        let ceiling = budget.ceiling_bytes();
        let now = budget.budget_now(0);

        // Assert
        assert_eq!(ceiling, 32 * GB);
        assert_eq!(now, 32 * GB);
    }

    #[test]
    fn budget_now_short_commit_limits_b_below_b_max_and_leaves_the_headroom() {
        // Arrange — 16 GB machine, 3 GB commit free, a ledger holding 2 GB of
        // which the process has allocated 1 GB.
        let budget = fake(16 * GB, 3 * GB);

        // Act
        let now = budget.budget_now(2 * GB);

        // Assert — own commit is credited only up to what the ledger holds.
        assert_eq!(budget.headroom_bytes(), 16 * GB / 20);
        assert_eq!(now, GB + 3 * GB - 16 * GB / 20);
    }

    #[test]
    fn budget_now_asked_twice_inside_the_refresh_window_reads_the_machine_once() {
        // Arrange
        let asked = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let budget =
            MachineBudget::new(Box::new(FakeProbe { physical: 16 * GB, available: 4 * GB, asked: asked.clone() }));

        // Act
        budget.budget_now(0);
        budget.budget_now(0);

        // Assert
        assert_eq!(asked.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn page_file_max_bytes_custom_system_managed_and_automatic_entries_their_maximum() {
        // Arrange + Act + Assert
        assert_eq!(page_file_max_bytes("c:\\pagefile.sys 16384 24576", 16 * GB), 24 * GB);
        assert_eq!(page_file_max_bytes("c:\\pagefile.sys 0 0", 16 * GB), 48 * GB);
        assert_eq!(page_file_max_bytes("?:\\pagefile.sys", GB), 4 * GB);
    }

    #[test]
    fn budget_bytes_on_this_machine_is_half_its_physical_memory() {
        // Arrange + Act
        let budget = budget_bytes();

        // Assert
        assert!(budget >= MIN_BUDGET_BYTES);
        assert_eq!(budget, ceiling_for(physical_memory_bytes()));
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
        let needed = budget_bytes() * 4;

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
