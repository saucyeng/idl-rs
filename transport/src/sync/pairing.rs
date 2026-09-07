//! Pairing: the 6-digit code, its lifetime, and the per-peer token store
//! (C3 §3.9, PLAN §3, §8 Q2/Q3/Q7).
//!
//! The peer file lives outside `<data>` (PLAN §8 Q7), so this module takes
//! its path as a parameter rather than resolving one itself — this crate
//! never depends on Tauri for a path (CLAUDE.md §2). Its own atomic write
//! (tmp sibling -> fsync -> rename) is a simpler recipe than `idl-rs`'s
//! `store::atomic::write_atomic`: that primitive's optimistic-concurrency
//! check is built for content-addressed files under `<data>` and requires
//! the caller to track the target's last-read hash across calls, which does
//! not fit a small local peer list with no such caller-tracked state. This
//! avoids taking a dependency on `idl-rs` for a mismatched fit.

use std::io::Write;
use std::path::Path;

use uuid::Uuid;

use crate::error::{TransportError, TransportErrorKind};

use super::wire::{PairRequest, PairingOffer, Peer, PROTOCOL_VERSION};

/// How long a minted pairing code stays redeemable, milliseconds
/// (PLAN §8 Q3).
pub const PAIRING_TTL_MS: i64 = 120_000;

/// Failed redemption attempts against one offer before it is burnt
/// (PLAN §8 Q3).
pub const MAX_ATTEMPTS: u32 = 5;

/// The live pairing state of one app instance. Holds at most one open offer
/// at a time; minting a new one discards the last.
#[derive(Debug, Default)]
pub struct PairingState {
    offer: Option<PairingOffer>,
    failed_attempts: u32,
}

impl PairingState {
    /// Mints a fresh offer, replacing any outstanding one (its code
    /// immediately stops redeeming) and resetting the failed-attempt
    /// counter.
    pub fn offer(&mut self, now_ms: i64) -> PairingOffer {
        let offer = PairingOffer { code: six_digit_code(), expires_at_ms: now_ms + PAIRING_TTL_MS };
        self.offer = Some(offer.clone());
        self.failed_attempts = 0;
        offer
    }

    /// Checks `code` against the outstanding offer. Consumes the offer on a
    /// correct code inside its TTL (single use). A wrong code counts toward
    /// `MAX_ATTEMPTS`; the fifth failure burns the offer so a new one must
    /// be minted. An expired offer is burnt and refused without counting
    /// toward the attempt limit. The comparison is constant-time over the
    /// fixed six-byte code.
    pub fn redeem(&mut self, code: &str, now_ms: i64) -> Result<(), TransportError> {
        let Some(current) = self.offer.clone() else {
            return Err(TransportError::new(TransportErrorKind::Sync, "no outstanding pairing offer"));
        };

        if now_ms > current.expires_at_ms {
            self.offer = None;
            return Err(TransportError::new(TransportErrorKind::Sync, "pairing code expired"));
        }

        if constant_time_eq(code, &current.code) {
            self.offer = None;
            self.failed_attempts = 0;
            return Ok(());
        }

        self.failed_attempts += 1;
        if self.failed_attempts >= MAX_ATTEMPTS {
            self.offer = None;
            return Err(TransportError::new(
                TransportErrorKind::Sync,
                "wrong pairing code; too many attempts, offer burnt",
            ));
        }
        Err(TransportError::new(TransportErrorKind::Sync, "wrong pairing code"))
    }
}

/// Refuses a `PairRequest` whose `protocol_version` this build does not
/// speak, naming both versions (PLAN §3: never a partial pair).
pub fn check_protocol_version(request: &PairRequest) -> Result<(), TransportError> {
    if request.protocol_version != PROTOCOL_VERSION {
        return Err(TransportError::new(
            TransportErrorKind::Sync,
            format!(
                "unsupported protocol version: peer speaks {}, this build speaks {PROTOCOL_VERSION}",
                request.protocol_version
            ),
        ));
    }
    Ok(())
}

/// Six decimal digits derived from a `uuid::Uuid::new_v4()` (PLAN §8 Q2: no
/// `rand` dependency). Leading zeros are preserved by formatting into a
/// fixed-width string, never by round-tripping through an integer.
fn six_digit_code() -> String {
    let bytes = Uuid::new_v4().into_bytes();
    let n = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    six_digit_code_from_u32(n)
}

/// Formats `n` as a six-digit decimal string, zero-padded. Split out from
/// [`six_digit_code`] so the zero-padding behaviour is deterministically
/// testable independent of `Uuid::new_v4()`'s randomness.
fn six_digit_code_from_u32(n: u32) -> String {
    format!("{:06}", n % 1_000_000)
}

/// Compares `a` and `b` without short-circuiting on the first differing
/// byte, so a wrong pairing code's length-of-common-prefix is not
/// observable via timing.
fn constant_time_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// Loads the peer list from `peers_path`. An absent file is an empty list,
/// not an error; a malformed file is a `Sync` error naming the path, never
/// silently truncated.
pub fn load_peers(peers_path: &Path) -> Result<Vec<Peer>, TransportError> {
    let bytes = match std::fs::read(peers_path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => {
            return Err(TransportError::new(
                TransportErrorKind::Sync,
                format!("reading peer file {}: {e}", peers_path.display()),
            ))
        }
    };

    serde_json::from_slice(&bytes).map_err(|e| {
        TransportError::new(TransportErrorKind::Sync, format!("malformed peer file {}: {e}", peers_path.display()))
    })
}

/// Writes `peers` to `peers_path` atomically: a `tmp` sibling is written and
/// fsynced, then renamed into place. `peers_path` is passed in by the
/// caller (never resolved from a Tauri app handle here) and lives outside
/// `<data>` so its contents never sync (PLAN §8 Q7).
pub fn save_peers(peers_path: &Path, peers: &[Peer]) -> Result<(), TransportError> {
    let bytes = serde_json::to_vec_pretty(peers)
        .map_err(|e| TransportError::new(TransportErrorKind::Sync, format!("encoding peer file: {e}")))?;

    let parent = peers_path.parent().ok_or_else(|| {
        TransportError::new(
            TransportErrorKind::Sync,
            format!("peer file path has no parent directory: {}", peers_path.display()),
        )
    })?;
    std::fs::create_dir_all(parent)
        .map_err(|e| TransportError::new(TransportErrorKind::Sync, format!("creating {}: {e}", parent.display())))?;

    let tmp_name = match peers_path.file_name() {
        Some(name) => format!("{}.tmp-{}", name.to_string_lossy(), Uuid::new_v4()),
        None => format!("peers.tmp-{}", Uuid::new_v4()),
    };
    let tmp_path = parent.join(tmp_name);

    write_and_fsync(&tmp_path, &bytes)?;

    if let Err(e) = std::fs::rename(&tmp_path, peers_path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(TransportError::new(
            TransportErrorKind::Sync,
            format!("renaming {} -> {}: {e}", tmp_path.display(), peers_path.display()),
        ));
    }

    fsync_parent_dir(peers_path);
    Ok(())
}

fn write_and_fsync(path: &Path, bytes: &[u8]) -> Result<(), TransportError> {
    let mut f = std::fs::File::create(path)
        .map_err(|e| TransportError::new(TransportErrorKind::Sync, format!("creating {}: {e}", path.display())))?;
    f.write_all(bytes)
        .map_err(|e| TransportError::new(TransportErrorKind::Sync, format!("writing {}: {e}", path.display())))?;
    f.sync_all()
        .map_err(|e| TransportError::new(TransportErrorKind::Sync, format!("fsyncing {}: {e}", path.display())))?;
    Ok(())
}

#[cfg(unix)]
fn fsync_parent_dir(target: &Path) {
    if let Some(parent) = target.parent() {
        if let Ok(dir) = std::fs::File::open(parent) {
            let _ = dir.sync_all();
        }
    }
}

#[cfg(not(unix))]
fn fsync_parent_dir(_target: &Path) {
    // NTFS commits the rename as a single MFT transaction; no directory
    // fsync exists or is needed on Windows.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::wire::PairResponse;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("idl-transport-test-{}-{}", name, Uuid::new_v4()))
    }

    #[test]
    fn offer_code_is_six_digits_and_may_start_with_zero() {
        // Arrange
        let mut state = PairingState::default();

        // Act
        let offer = state.offer(0);

        // Assert
        assert_eq!(offer.code.len(), 6);
        assert!(offer.code.chars().all(|c| c.is_ascii_digit()));
        // The formatting itself preserves a leading zero: a small input
        // number zero-pads rather than shortening.
        assert_eq!(six_digit_code_from_u32(42), "000042");
    }

    #[test]
    fn redeem_correct_code_inside_ttl_then_second_redeem_fails() {
        // Arrange
        let mut state = PairingState::default();
        let offer = state.offer(1_000);

        // Act
        let first = state.redeem(&offer.code, 1_000 + PAIRING_TTL_MS);
        let second = state.redeem(&offer.code, 1_000 + PAIRING_TTL_MS);

        // Assert
        assert!(first.is_ok());
        assert_eq!(second.unwrap_err().kind, TransportErrorKind::Sync);
    }

    #[test]
    fn redeem_correct_code_after_ttl_is_a_sync_error() {
        // Arrange
        let mut state = PairingState::default();
        let offer = state.offer(1_000);

        // Act
        let result = state.redeem(&offer.code, 1_000 + PAIRING_TTL_MS + 1);

        // Assert
        assert_eq!(result.unwrap_err().kind, TransportErrorKind::Sync);
    }

    #[test]
    fn redeem_five_wrong_codes_then_the_right_one_is_refused_offer_burnt() {
        // Arrange
        let mut state = PairingState::default();
        let offer = state.offer(0);
        let wrong = if offer.code == "000000" { "111111" } else { "000000" };

        // Act
        for _ in 0..MAX_ATTEMPTS {
            let _ = state.redeem(wrong, 0);
        }
        let result = state.redeem(&offer.code, 0);

        // Assert
        assert_eq!(result.unwrap_err().kind, TransportErrorKind::Sync);
    }

    #[test]
    fn redeem_with_no_outstanding_offer_is_a_sync_error() {
        // Arrange
        let mut state = PairingState::default();

        // Act
        let result = state.redeem("123456", 0);

        // Assert
        assert_eq!(result.unwrap_err().kind, TransportErrorKind::Sync);
    }

    #[test]
    fn offer_called_twice_the_first_code_no_longer_redeems() {
        // Arrange
        let mut state = PairingState::default();
        let first = state.offer(0);

        // Act
        let _second = state.offer(0);
        let result = state.redeem(&first.code, 0);

        // Assert
        assert_eq!(result.unwrap_err().kind, TransportErrorKind::Sync);
    }

    #[test]
    fn check_protocol_version_refuses_unknown_version_naming_both() {
        // Arrange
        let request = PairRequest {
            code: "123456".to_string(),
            peer_id: "peer-1".to_string(),
            name: "Pit Laptop".to_string(),
            protocol_version: PROTOCOL_VERSION + 1,
        };

        // Act
        let result = check_protocol_version(&request);

        // Assert
        let err = result.unwrap_err();
        assert_eq!(err.kind, TransportErrorKind::Sync);
        assert!(err.message.contains(&(PROTOCOL_VERSION + 1).to_string()));
        assert!(err.message.contains(&PROTOCOL_VERSION.to_string()));
    }

    #[test]
    fn load_peers_absent_file_is_an_empty_list() {
        // Arrange
        let path = tmp_path("absent");

        // Act
        let peers = load_peers(&path).unwrap();

        // Assert
        assert!(peers.is_empty());
    }

    #[test]
    fn load_peers_malformed_file_is_a_sync_error_naming_the_path() {
        // Arrange
        let path = tmp_path("malformed");
        std::fs::write(&path, b"not json").unwrap();

        // Act
        let result = load_peers(&path);

        // Assert
        let err = result.unwrap_err();
        assert_eq!(err.kind, TransportErrorKind::Sync);
        assert!(err.message.contains(&path.display().to_string()));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn save_peers_then_load_peers_round_trips_tokens_intact() {
        // Arrange
        let dir = tmp_path("peers-dir");
        let path = dir.join("peers.json");
        let peers = vec![Peer {
            peer_id: "peer-1".to_string(),
            name: "Pit Laptop".to_string(),
            token: "secret-token-value".to_string(),
            protocol_version: PROTOCOL_VERSION,
            paired_at_ms: 1_700_000_000_000,
        }];

        // Act
        save_peers(&path, &peers).unwrap();
        let loaded = load_peers(&path).unwrap();

        // Assert
        assert_eq!(loaded, peers);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wire_dtos_round_trip_through_serde_json_with_c3_field_names() {
        // Arrange
        let request = PairRequest {
            code: "004200".to_string(),
            peer_id: "peer-1".to_string(),
            name: "Pit Laptop".to_string(),
            protocol_version: PROTOCOL_VERSION,
        };
        let response = PairResponse {
            peer_id: "peer-2".to_string(),
            name: "Pit Tablet".to_string(),
            token: "tok".to_string(),
            protocol_version: PROTOCOL_VERSION,
        };
        let peer = Peer {
            peer_id: "peer-2".to_string(),
            name: "Pit Tablet".to_string(),
            token: "tok".to_string(),
            protocol_version: PROTOCOL_VERSION,
            paired_at_ms: 42,
        };

        // Act
        let request_json = serde_json::to_string(&request).unwrap();
        let response_json = serde_json::to_string(&response).unwrap();
        let peer_json = serde_json::to_string(&peer).unwrap();

        // Assert
        assert!(request_json.contains("\"peer_id\":\"peer-1\""));
        assert!(request_json.contains("\"protocol_version\":1"));
        assert_eq!(serde_json::from_str::<PairRequest>(&request_json).unwrap(), request);
        assert!(response_json.contains("\"token\":\"tok\""));
        assert_eq!(serde_json::from_str::<PairResponse>(&response_json).unwrap(), response);
        assert!(peer_json.contains("\"paired_at_ms\":42"));
        assert_eq!(serde_json::from_str::<Peer>(&peer_json).unwrap(), peer);
    }
}
