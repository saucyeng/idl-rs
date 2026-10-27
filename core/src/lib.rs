//! idl-rs — pure data-acquisition engine for IDL0 (.idl0) files.
//!
//! Parsing, data model, DSP, and analysis. No Tauri, no async runtime,
//! no clap, no I/O beyond std::fs. Consumed by the idl1 app (via
//! idl-rs-tauri) and the idl-rs CLI, and future Python/WASM bindings.

pub mod calibration;
pub mod chart_decimation;
pub mod clip_reconstruct;
pub mod colormap;
/// The one command table (ruling R230) and the core operations behind the
/// rows R229 added. Pure data plus pure functions; the CLI builds its clap
/// tree from it and `docs cli` renders it.
pub mod commands;
pub mod config;
pub mod cursor;
/// Renders `docs/WORKBOOK-REFERENCE.md` from the engine's own catalogs
/// (ruling R222 item 1). Pure: no I/O, no filesystem.
pub mod docs;
pub mod estimate;
pub mod export;
pub mod fft;
pub mod fft_wire;
pub mod filters;
pub mod gps;
pub mod gps_projection;
pub mod gps_wire;
pub mod histogram;
pub mod histogram2d;
pub mod import;
pub mod integration;
pub mod laps;
pub mod math;
pub mod parse;
pub mod raster;
pub mod rotation;
pub mod scatter;
pub mod scatter_wire;
pub mod session;
pub mod spectrogram;
pub mod statistics;
pub mod store;
pub mod synth;
pub mod table;
pub mod tile;
pub mod track_artifact;
pub mod track_projection;
pub mod tracks;
pub mod variance;
pub mod video;
pub mod wire_golden;
pub mod workbook;

/// The engine's crate version, stamped into derived-file hashes and reported
/// over IPC so every device can prove it computes with the same engine.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Per-thread allocation counters for memory-shape tests (evaluation memory
/// model, Phase 1). The counters are thread-local so the inline tests, which
/// the harness runs in parallel, each measure only their own allocations; the
/// evaluator is single-threaded per evaluation.
#[cfg(test)]
pub(crate) mod counting_alloc {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        // `const` initialisers with no destructor: safe to touch from inside
        // the allocator itself.
        static LIVE: Cell<isize> = const { Cell::new(0) };
        static PEAK: Cell<isize> = const { Cell::new(0) };
    }

    /// The test build's global allocator: `System`, plus this thread's live
    /// and peak byte counts.
    pub(crate) struct Counting;

    // SAFETY: every call forwards to `System` unchanged; the counters only
    // observe sizes.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let p = System.alloc(layout);
            if !p.is_null() {
                bump(layout.size() as isize);
            }
            p
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let p = System.alloc_zeroed(layout);
            if !p.is_null() {
                bump(layout.size() as isize);
            }
            p
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            System.dealloc(ptr, layout);
            bump(-(layout.size() as isize));
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let p = System.realloc(ptr, layout, new_size);
            if !p.is_null() {
                bump(new_size as isize - layout.size() as isize);
            }
            p
        }
    }

    fn bump(delta: isize) {
        let live = LIVE.with(|l| {
            l.set(l.get() + delta);
            l.get()
        });
        PEAK.with(|p| {
            if live > p.get() {
                p.set(live);
            }
        });
    }

    /// Runs `f` and returns its result with the most bytes this thread held
    /// above where it started, at any instant during `f`. Bytes `f` returns
    /// (still live at the end) count; bytes allocated before `f` do not.
    pub(crate) fn peak_of<R>(f: impl FnOnce() -> R) -> (R, usize) {
        let base = LIVE.with(|l| l.get());
        PEAK.with(|p| p.set(base));
        let out = f();
        let peak = PEAK.with(|p| p.get());
        (out, (peak - base).max(0) as usize)
    }
}

#[cfg(test)]
#[global_allocator]
static COUNTING_ALLOCATOR: counting_alloc::Counting = counting_alloc::Counting;
