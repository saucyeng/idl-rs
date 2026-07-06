//! Track matching: assign session GPS to known tracks and coalesce contiguous
//! on-track runs into visit windows. Pure: reads GPS from the session handle
//! (via `crate::gps`), takes reference polylines as input, returns windows.
//! Ported from the Dart `TrackMatcher`.

pub mod detect;
pub mod geometry;

pub use detect::{detect_visits, TrackRef, VisitParams, VisitWindow};
