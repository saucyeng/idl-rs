//! Portable Track artifact (`.idl0t`): read a GUI-authored track config from a
//! file into the domain [`Track`], for the CLI's `laps`/`visits`. Read-only on
//! the format — authoring stays in the app. See
//! `docs/superpowers/specs/2026-06-02-idl-rs-phase-5b-track-artifact-design.md`.

pub mod model;
pub mod read;

pub use model::{Track, SUPPORTED_TRACK_ARTIFACT_VERSION};
pub use read::{parse_track, read_track};
