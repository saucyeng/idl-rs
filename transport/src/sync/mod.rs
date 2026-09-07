//! LAN peer sync: wire DTOs and pairing (Task 7); the `axum` HTTP server
//! and hand-parsed `Range` support (Task 8); mDNS advertise/browse (Task 9);
//! the pull/push client (this task, PLAN §2/§4) — the driver that fetches a
//! peer's manifest, asks core's `plan_sync` what moves, and moves it.

pub mod client;
pub mod discovery;
#[cfg(test)]
mod loopback_tests;
pub mod pairing;
pub mod range;
pub mod server;
pub mod wire;

pub use client::{pair_with_peer, sync_with_peer, SyncProgress, SyncRunResult};
pub use discovery::{advertise, browse, build_txt, parse_txt, Advertisement, DiscoveredPeer, SERVICE_TYPE};
pub use pairing::{check_protocol_version, load_peers, save_peers, PairingState, MAX_ATTEMPTS, PAIRING_TTL_MS};
pub use range::{parse_range, RangeError, RangeErrorKind};
pub use server::{SyncServer, SyncServerConfig};
pub use wire::{PairRequest, PairResponse, PairingOffer, Peer, PROTOCOL_VERSION};
