//! Canvas-agnostic overlay system: positioned, channel-bound elements sampled
//! at a time. The video compositor (`video::render`) is the first consumer;
//! chart-canvas overlays are future work. See docs/IDL0_SPEC.md §33.

pub mod model;
pub mod sample;
