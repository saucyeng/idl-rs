//! Wire DTOs for LAN peer sync (C3 §3.9, PLAN §3). Plain data; no I/O, no
//! protocol logic — that lives in `pairing.rs`.

use serde::{Deserialize, Serialize};

/// The protocol version this build speaks. Bumped only by a contract change
/// (PLAN §3: a peer advertising a different `v` is listed incompatible).
pub const PROTOCOL_VERSION: u32 = 1;

/// A peer we have paired with. Serialised to the peer file (PLAN §8 Q7);
/// that file lives outside `<data>` so `token` never syncs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Peer {
    /// Stable identifier for the peer app instance.
    pub peer_id: String,
    /// Display name shown in the Settings peer list.
    pub name: String,
    /// Bearer token this instance presents to the peer on every request but
    /// `POST /pair`. Never logged, never displayed.
    pub token: String,
    /// Protocol version the peer spoke at pairing time.
    pub protocol_version: u32,
    /// Wall-clock time pairing completed, epoch milliseconds.
    pub paired_at_ms: i64,
}

/// A pairing code offered by this instance (C3 §3.9's `PairingCode`).
/// Single-use, short-lived (PLAN §8 Q3): burnt on success, on expiry, or
/// after `pairing::MAX_ATTEMPTS` failed redemptions.
#[derive(Debug, Clone, PartialEq)]
pub struct PairingOffer {
    /// Exactly six decimal digits; leading zeros preserved (never parsed as
    /// an integer).
    pub code: String,
    /// Epoch milliseconds after which the code no longer redeems.
    pub expires_at_ms: i64,
}

/// `POST /idl1/v1/pair`'s request body (PLAN §3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairRequest {
    /// The code shown on the offering instance's screen.
    pub code: String,
    /// Stable identifier of the instance requesting to pair.
    pub peer_id: String,
    /// Display name of the instance requesting to pair.
    pub name: String,
    /// Protocol version the requester speaks.
    pub protocol_version: u32,
}

/// `POST /idl1/v1/pair`'s response body (PLAN §3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairResponse {
    /// Stable identifier of the instance that offered the code.
    pub peer_id: String,
    /// Display name of the instance that offered the code.
    pub name: String,
    /// Bearer token minted for the requester, returned once.
    pub token: String,
    /// Protocol version the offering instance speaks.
    pub protocol_version: u32,
}
