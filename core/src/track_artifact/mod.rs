//! Portable Track artifact (`.idl0t`): read a GUI-authored track config from a
//! file into the domain [`Track`] for the CLI's `laps`/`visits`, and write one
//! back out (authoring itself stays in the app; this is the persistence
//! primitive the app calls, C4 §2).

pub mod model;
pub mod read;
pub mod write;

pub use model::{Track, SUPPORTED_TRACK_ARTIFACT_VERSION};
pub use read::{parse_track, read_track};
pub use write::write_track;
