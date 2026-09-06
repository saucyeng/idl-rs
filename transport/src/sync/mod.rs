//! LAN peer sync: wire DTOs and pairing (this task); the HTTP server, mDNS
//! discovery and pull/push client follow in later L11 tasks (PLAN §2/§4).

pub mod pairing;
pub mod wire;

pub use pairing::{check_protocol_version, load_peers, save_peers, PairingState, MAX_ATTEMPTS, PAIRING_TTL_MS};
pub use wire::{PairRequest, PairResponse, PairingOffer, Peer, PROTOCOL_VERSION};
