//! LAN peer sync: wire DTOs and pairing (Task 7); the `axum` HTTP server
//! and hand-parsed `Range` support (this task). mDNS discovery and the
//! pull/push client follow in later L11 tasks (PLAN §2/§4).

pub mod pairing;
pub mod range;
pub mod server;
pub mod wire;

pub use pairing::{check_protocol_version, load_peers, save_peers, PairingState, MAX_ATTEMPTS, PAIRING_TTL_MS};
pub use range::{parse_range, RangeError, RangeErrorKind};
pub use server::{SyncServer, SyncServerConfig};
pub use wire::{PairRequest, PairResponse, PairingOffer, Peer, PROTOCOL_VERSION};
