//! Shared, single-flight workbook evaluations (evaluation-memory spec §4.2.A,
//! Phase 1).
//!
//! Before this, every `fetch_host_channel_v2` re-parsed the workbook and re-ran
//! `eval_cells` over every definition to return one of them, so a notebook with
//! five charts ran six evaluations at once, each holding gigabytes. Now an
//! evaluation is identified by its inputs ([`EvalKey`]); the first caller for a
//! key runs it, every concurrent caller for the same key waits for that one
//! result, and a finished result stays in a byte-accounted LRU so a later
//! caller reads it instead of recomputing. A change in any input is a new key,
//! so the next call evaluates again.
//!
//! Results that are errors are never kept: a refused decode is a statement
//! about the machine's state right now, and the next caller deserves a fresh
//! attempt. The callers already waiting on the failed flight do receive that
//! same error, once.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Condvar, Mutex};

use idl_rs::workbook::v3::CellEvalResult;
use sha2::{Digest, Sha256};

use crate::error::{IpcError, IpcErrorKind};

/// What identifies one evaluation: a SHA-256 over every input that can change
/// its answer. Two keys are equal exactly when their inputs were byte-equal.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EvalKey(String);

impl EvalKey {
    /// Hashes `parts` in order. Each part is length-prefixed, so
    /// `["ab", "c"]` and `["a", "bc"]` are different keys.
    pub fn from_parts(parts: &[&[u8]]) -> Self {
        let mut hasher = Sha256::new();
        for part in parts {
            hasher.update((part.len() as u64).to_le_bytes());
            hasher.update(part);
        }
        EvalKey(hex::encode(hasher.finalize()))
    }

    /// The key as 64 lowercase hex digits.
    pub fn as_hex(&self) -> &str {
        &self.0
    }
}

/// One finished evaluation: every cell's result, in document order. Its
/// definitions hold the evaluator's own `Arc` buffers (shared with the
/// cross-reference overlay while it ran), so keeping it costs 8 B per sample
/// per retained value plus one copy of each distinct time axis.
#[derive(Debug)]
pub struct Evaluation {
    cells: Vec<CellEvalResult>,
    bytes: u64,
}

impl Evaluation {
    /// Wraps `cells`, measuring what they hold ([`Self::bytes`]).
    pub fn new(cells: Vec<CellEvalResult>) -> Self {
        let bytes = held_bytes(&cells);
        Evaluation { cells, bytes }
    }

    /// One result per workbook cell, in document order.
    pub fn cells(&self) -> &[CellEvalResult] {
        &self.cells
    }

    /// Bytes of sample and time-axis buffers this evaluation keeps alive,
    /// counting a buffer shared by several definitions once.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The first definition called `name`, in document order.
    pub fn def(&self, name: &str) -> Option<&idl_rs::workbook::v3::CellDefResult> {
        self.cells.iter().flat_map(|c| c.defs.iter()).find(|d| d.name == name)
    }
}

// TODO(idl0): the registry cap (`memory::eval_cache_cap_bytes`) and the bytes
// `held_bytes` counts are a second budget beside `SessionCache`'s, so the
// process can hold both in full. Phase 3 of the evaluation-memory spec
// (2026-10-27-idl1-evaluation-memory-DRAFT.md §4.2.G, one ledger for cache,
// decodes and evaluations) replaces both with one admission ledger.
/// Distinct buffer bytes behind `cells`' values: an `Arc` that appears in
/// several definitions (a shared time axis) is counted once.
fn held_bytes(cells: &[CellEvalResult]) -> u64 {
    let mut seen: HashSet<usize> = HashSet::new();
    let mut total: u64 = 0;
    for def in cells.iter().flat_map(|c| c.defs.iter()) {
        let Some(view) = &def.value else { continue };
        if seen.insert(Arc::as_ptr(&view.v) as *const u8 as usize) {
            total += view.v.len() as u64 * 8;
        }
        if seen.insert(Arc::as_ptr(&view.t_us) as *const u8 as usize) {
            total += view.t_us.len() as u64 * 8;
        }
    }
    total
}

/// One evaluation in progress: its result, once the leader has one.
struct Flight {
    result: Mutex<Option<Result<Arc<Evaluation>, IpcError>>>,
    done: Condvar,
}

struct Inner {
    /// Evaluations running now; a second caller for the key waits on these.
    flights: HashMap<EvalKey, Arc<Flight>>,
    /// Finished evaluations kept for later callers.
    done: HashMap<EvalKey, Arc<Evaluation>>,
    /// `done`'s keys, least recently used first.
    order: Vec<EvalKey>,
    /// Sum of `done`'s [`Evaluation::bytes`].
    resident_bytes: u64,
    /// Evaluations the closure of a first caller has actually run.
    runs: u64,
}

/// The process's evaluations, single-flight per [`EvalKey`], LRU by bytes.
///
/// Lives inside the [`crate::session_cache::SessionCache`] the app already
/// manages, so it needs no state of its own in the app crate.
pub struct EvalRegistry {
    cap_bytes: u64,
    inner: Mutex<Inner>,
}

impl std::fmt::Debug for EvalRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvalRegistry").field("cap_bytes", &self.cap_bytes).finish_non_exhaustive()
    }
}

/// Completes a flight even if its closure panics, so waiters are never left
/// asleep on a result that cannot come.
struct Leader<'a> {
    registry: &'a EvalRegistry,
    key: EvalKey,
    flight: Arc<Flight>,
    finished: bool,
}

impl Leader<'_> {
    fn finish(&mut self, result: Result<Arc<Evaluation>, IpcError>) {
        self.finished = true;
        {
            let mut inner = self.registry.lock();
            inner.flights.remove(&self.key);
            if let Ok(evaluation) = &result {
                self.registry.admit(&mut inner, &self.key, evaluation);
            }
        }
        *self.flight.result.lock().unwrap_or_else(|e| e.into_inner()) = Some(result);
        self.flight.done.notify_all();
    }
}

impl Drop for Leader<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.finish(Err(IpcError::new(IpcErrorKind::Internal, "the workbook evaluation panicked")));
        }
    }
}

impl EvalRegistry {
    /// A registry keeping at most `cap_bytes` of finished evaluations. See
    /// [`crate::memory::eval_cache_cap_bytes`] for how the app sizes it.
    pub fn with_cap(cap_bytes: u64) -> Self {
        EvalRegistry {
            cap_bytes,
            inner: Mutex::new(Inner {
                flights: HashMap::new(),
                done: HashMap::new(),
                order: Vec::new(),
                resident_bytes: 0,
                runs: 0,
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Keeps `evaluation` for later callers, evicting the least recently used
    /// to stay under the cap. One larger than the whole cap is not kept (it is
    /// still returned to everyone waiting on it).
    fn admit(&self, inner: &mut Inner, key: &EvalKey, evaluation: &Arc<Evaluation>) {
        if evaluation.bytes() > self.cap_bytes {
            return;
        }
        while inner.resident_bytes + evaluation.bytes() > self.cap_bytes && !inner.order.is_empty() {
            let oldest = inner.order.remove(0);
            if let Some(gone) = inner.done.remove(&oldest) {
                inner.resident_bytes -= gone.bytes();
            }
        }
        inner.resident_bytes += evaluation.bytes();
        inner.done.insert(key.clone(), Arc::clone(evaluation));
        inner.order.push(key.clone());
    }

    /// The evaluation for `key`: a kept one, the result of the flight already
    /// running for it, or — for the first caller — whatever `evaluate` makes.
    ///
    /// `evaluate` runs on the calling thread, with no lock held, and at most
    /// once per key at a time. An `Err` from it reaches every caller waiting on
    /// that flight and is not kept.
    pub fn get_or_evaluate(
        &self,
        key: &EvalKey,
        evaluate: impl FnOnce() -> Result<Vec<CellEvalResult>, IpcError>,
    ) -> Result<Arc<Evaluation>, IpcError> {
        let flight = {
            let mut inner = self.lock();
            if let Some(hit) = inner.done.get(key).cloned() {
                if let Some(at) = inner.order.iter().position(|k| k == key) {
                    let k = inner.order.remove(at);
                    inner.order.push(k);
                }
                return Ok(hit);
            }
            if let Some(running) = inner.flights.get(key) {
                let running = Arc::clone(running);
                drop(inner);
                return Self::wait(&running);
            }
            let flight = Arc::new(Flight { result: Mutex::new(None), done: Condvar::new() });
            inner.flights.insert(key.clone(), Arc::clone(&flight));
            inner.runs += 1;
            flight
        };

        let mut leader = Leader { registry: self, key: key.clone(), flight, finished: false };
        let result = evaluate().map(|cells| Arc::new(Evaluation::new(cells)));
        leader.finish(result.clone());
        result
    }

    fn wait(flight: &Flight) -> Result<Arc<Evaluation>, IpcError> {
        let mut slot = flight.result.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(result) = slot.as_ref() {
                return result.clone();
            }
            slot = flight.done.wait(slot).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// How many times a first caller's `evaluate` has run since the registry
    /// was made. Concurrent callers on one key add 1, not one each.
    pub fn runs(&self) -> u64 {
        self.lock().runs
    }

    /// Bytes of finished evaluations currently kept.
    pub fn resident_bytes(&self) -> u64 {
        self.lock().resident_bytes
    }

    /// Number of finished evaluations currently kept.
    pub fn len(&self) -> usize {
        self.lock().done.len()
    }

    /// `true` when no finished evaluation is kept.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use idl_rs::workbook::v3::{AxisKind, CellDefResult, CellKindToken, HostChannelView};

    fn cell_with_values(n: usize) -> Vec<CellEvalResult> {
        let view = HostChannelView {
            length: n,
            v: vec![0.0; n].into(),
            t_us: vec![0i64; n].into(),
            unit: idl_rs::math::units::UnitLabel::Dimensionless,
            axis: AxisKind::Time,
        };
        vec![CellEvalResult {
            cell_id: "aaaaaaaa".to_string(),
            kind: CellKindToken::Math,
            defs: vec![CellDefResult {
                name: "x".to_string(),
                label: None,
                value: Some(view),
                sample_rate_hz: Some(100.0),
                unit: idl_rs::math::units::UnitLabel::Dimensionless,
                unit_notes: Vec::new(),
                error: None,
            }],
            errors: Vec::new(),
        }]
    }

    fn key(name: &str) -> EvalKey {
        EvalKey::from_parts(&[name.as_bytes()])
    }

    #[test]
    fn eval_key_parts_that_concatenate_alike_are_different_keys() {
        // Arrange + Act
        let a = EvalKey::from_parts(&[b"ab", b"c"]);
        let b = EvalKey::from_parts(&[b"a", b"bc"]);

        // Assert
        assert_ne!(a, b);
        assert_eq!(a.as_hex().len(), 64);
    }

    #[test]
    fn get_or_evaluate_a_key_seen_before_returns_the_kept_evaluation_without_running_again() {
        // Arrange
        let registry = EvalRegistry::with_cap(1 << 20);
        let first = registry.get_or_evaluate(&key("k"), || Ok(cell_with_values(4))).unwrap();

        // Act
        let second = registry.get_or_evaluate(&key("k"), || panic!("must not run")).unwrap();

        // Assert
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(registry.runs(), 1);
    }

    #[test]
    fn get_or_evaluate_concurrent_callers_on_one_key_wait_on_one_run() {
        // Arrange
        let registry = Arc::new(EvalRegistry::with_cap(1 << 20));
        let started = Arc::new(std::sync::Barrier::new(7));

        // Act
        let handles: Vec<_> = (0..7)
            .map(|_| {
                let registry = Arc::clone(&registry);
                let started = Arc::clone(&started);
                std::thread::spawn(move || {
                    started.wait();
                    registry
                        .get_or_evaluate(&key("k"), || {
                            std::thread::sleep(std::time::Duration::from_millis(100));
                            Ok(cell_with_values(4))
                        })
                        .unwrap()
                })
            })
            .collect();
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // Assert
        assert_eq!(registry.runs(), 1);
        assert!(results.iter().all(|r| Arc::ptr_eq(r, &results[0])));
    }

    #[test]
    fn get_or_evaluate_an_error_reaches_the_waiters_and_is_not_kept() {
        // Arrange
        let registry = EvalRegistry::with_cap(1 << 20);
        let refused = IpcError::new(IpcErrorKind::ResourceExhausted, "busy");

        // Act
        let first = registry.get_or_evaluate(&key("k"), || Err(refused.clone()));
        let second = registry.get_or_evaluate(&key("k"), || Ok(cell_with_values(4)));

        // Assert
        assert_eq!(first.unwrap_err().kind, IpcErrorKind::ResourceExhausted);
        assert!(second.is_ok());
        assert_eq!(registry.runs(), 2);
    }

    #[test]
    fn get_or_evaluate_past_the_cap_the_least_recently_used_evaluation_is_evicted() {
        // Arrange — each evaluation holds 2 buffers x 100 x 8 B = 1600 B.
        let registry = EvalRegistry::with_cap(3500);
        registry.get_or_evaluate(&key("a"), || Ok(cell_with_values(100))).unwrap();
        registry.get_or_evaluate(&key("b"), || Ok(cell_with_values(100))).unwrap();
        registry.get_or_evaluate(&key("a"), || panic!("a is kept")).unwrap();

        // Act — a third pushes the oldest-used ("b") out.
        registry.get_or_evaluate(&key("c"), || Ok(cell_with_values(100))).unwrap();

        // Assert
        assert_eq!(registry.len(), 2);
        assert_eq!(registry.resident_bytes(), 3200);
        registry.get_or_evaluate(&key("a"), || panic!("a is still kept")).unwrap();
        registry.get_or_evaluate(&key("b"), || Ok(cell_with_values(100))).unwrap();
        assert_eq!(registry.runs(), 4);
    }

    #[test]
    fn get_or_evaluate_one_evaluation_larger_than_the_cap_is_returned_but_not_kept() {
        // Arrange
        let registry = EvalRegistry::with_cap(100);

        // Act
        let got = registry.get_or_evaluate(&key("big"), || Ok(cell_with_values(1000))).unwrap();

        // Assert
        assert_eq!(got.bytes(), 16_000);
        assert!(registry.is_empty());
    }
}
