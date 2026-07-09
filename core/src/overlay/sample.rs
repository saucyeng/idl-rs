//! Frame sampling for overlay rendering: prepare once, sample per frame.
//! All `t` are session recording-time seconds. See docs/IDL0_SPEC.md §33.2.
//!
//! Lands with `SampleContext` in the next task.
