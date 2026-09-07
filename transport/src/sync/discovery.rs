//! mDNS advertise and browse for `_idl1._tcp` (PLAN §3, this task's brief).
//! TXT-record encode/decode is kept pure (`build_txt`/`parse_txt`) so the
//! peer-compatibility decision is unit-tested without a network; the
//! `mdns-sd` socket work around it is kept as thin as possible and does not
//! run in the test filter (only the `#[ignore]`d loopback test touches it).
//!
//! This crate never creates a tokio runtime (see `ble_transport::scan`'s
//! doc comment) — `browse`'s channel is fed by a task spawned onto whichever
//! runtime is already driving the call.

use std::net::{SocketAddr, SocketAddrV6};

use mdns_sd::{ScopedIp, ServiceDaemon, ServiceEvent, ServiceInfo};
use tokio::sync::mpsc;

use super::wire::PROTOCOL_VERSION;
use crate::error::{TransportError, TransportErrorKind};

/// The service type this app advertises and browses.
pub const SERVICE_TYPE: &str = "_idl1._tcp.local.";

/// A peer seen on the LAN. Says nothing about whether we have paired with
/// it — that decision belongs to later tasks (Task 10/12), not here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredPeer {
    /// Stable identifier for the peer app instance (matches `Peer::peer_id`
    /// once paired).
    pub peer_id: String,
    /// Display name advertised by the peer. A label, not an identity — may
    /// be empty.
    pub name: String,
    /// Protocol version the peer speaks, from its TXT record's `v` key.
    /// May differ from [`PROTOCOL_VERSION`]; surfaced as-is so the caller
    /// can decide what "incompatible" means.
    pub protocol_version: u32,
    /// Address and port to reach the peer's sync server on.
    pub addr: SocketAddr,
}

/// Builds the TXT-record key/value pairs this instance advertises: exactly
/// `pid` (peer id), `name`, and `v` (this build's [`PROTOCOL_VERSION`] as a
/// decimal string). Pure — no network, so it is directly testable.
pub fn build_txt(peer_id: &str, name: &str) -> Vec<(String, String)> {
    vec![
        ("pid".to_string(), peer_id.to_string()),
        ("name".to_string(), name.to_string()),
        ("v".to_string(), PROTOCOL_VERSION.to_string()),
    ]
}

/// Parses a browsed peer's TXT records into a [`DiscoveredPeer`]. Pure — no
/// network, so it is directly testable.
///
/// Returns `None` for a record missing `pid` or `v`, or carrying a `v` that
/// does not parse as a decimal `u32` — an unparseable neighbour is ignored,
/// never a hard failure. `name` defaults to an empty string when absent (a
/// name is a label, not an identity). A `v` that parses but differs from
/// [`PROTOCOL_VERSION`] is still `Some`, carrying the peer's real version —
/// deciding what to do about a version mismatch is not this function's job.
pub fn parse_txt(txt: &[(String, String)], addr: SocketAddr) -> Option<DiscoveredPeer> {
    let peer_id = txt.iter().find(|(k, _)| k == "pid").map(|(_, v)| v.clone())?;
    let protocol_version =
        txt.iter().find(|(k, _)| k == "v").map(|(_, v)| v.clone())?.parse::<u32>().ok()?;
    let name = txt.iter().find(|(k, _)| k == "name").map(|(_, v)| v.clone()).unwrap_or_default();
    Some(DiscoveredPeer { peer_id, name, protocol_version, addr })
}

fn sync_error(message: impl Into<String>) -> TransportError {
    TransportError::new(TransportErrorKind::Sync, message.into())
}

/// Converts a `mdns-sd` [`ScopedIp`] into a [`SocketAddr`], preserving the
/// IPv6 zone/scope id (review-task9 Minor). `ScopedIp::to_ip_addr` drops it,
/// which makes a link-local IPv6 peer (`fe80::1`) ambiguous on a
/// multi-interface host — a subsequent connect can pick the wrong NIC or
/// fail outright. IPv4 addresses have no scope concept and pass through
/// unchanged.
fn scoped_addr_to_socket_addr(addr: &ScopedIp, port: u16) -> SocketAddr {
    match addr {
        ScopedIp::V4(v4) => SocketAddr::new(std::net::IpAddr::V4(*v4.addr()), port),
        ScopedIp::V6(v6) => {
            let scope_id = v6.scope_id().index;
            SocketAddr::V6(SocketAddrV6::new(*v6.addr(), port, 0, scope_id))
        }
        // `ScopedIp` is `#[non_exhaustive]` (future mdns-sd variants); fall
        // back to the scope-losing conversion rather than fail to compile.
        other => SocketAddr::new(other.to_ip_addr(), port),
    }
}

/// Handle to this instance's mDNS advertisement. The service is withdrawn
/// when this value is dropped (best-effort — `mdns-sd` unregisters
/// asynchronously and drop cannot wait on it).
pub struct Advertisement {
    daemon: ServiceDaemon,
    fullname: String,
}

impl Drop for Advertisement {
    fn drop(&mut self) {
        let _ = self.daemon.unregister(&self.fullname);
    }
}

/// Advertises this instance as `_idl1._tcp` until the returned
/// [`Advertisement`] is dropped. `peer_id`/`name` become the TXT record
/// (`build_txt`); `port` is this instance's sync server port (PLAN §3).
///
/// Errors typed [`TransportErrorKind::Sync`] — no multicast on this network
/// or another local failure to start the daemon or register the service —
/// never a panic.
pub fn advertise(peer_id: &str, name: &str, port: u16) -> Result<Advertisement, TransportError> {
    let daemon =
        ServiceDaemon::new().map_err(|e| sync_error(format!("starting mDNS daemon: {e}")))?;

    let host_name = format!("{peer_id}.local.");
    let txt = build_txt(peer_id, name);
    let properties: Vec<(&str, &str)> =
        txt.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();

    let service = ServiceInfo::new(SERVICE_TYPE, peer_id, &host_name, "", port, &properties[..])
        .map_err(|e| sync_error(format!("building mDNS service info: {e}")))?
        .enable_addr_auto();

    let fullname = service.get_fullname().to_string();

    daemon
        .register(service)
        .map_err(|e| sync_error(format!("registering mDNS service: {e}")))?;

    Ok(Advertisement { daemon, fullname })
}

/// Browses for `_idl1._tcp` peers. The receiver is fed from a task spawned
/// on the caller's own runtime — this crate never creates one (see
/// `ble_transport::scan`'s doc comment). A record this build cannot parse
/// (`parse_txt` returning `None`) is skipped, not surfaced as an error.
///
/// Errors typed [`TransportErrorKind::Sync`] — no multicast on this
/// network, a firewall, or another local failure to start the daemon or
/// begin the browse — never a panic.
pub fn browse() -> Result<mpsc::Receiver<DiscoveredPeer>, TransportError> {
    let (rx, _handle) = browse_with_handle()?;
    Ok(rx)
}

/// Same as [`browse`], but also returns the spawned task's
/// [`tokio::task::JoinHandle`] so a test can await its completion. Not part
/// of the public API surface a caller needs — `browse` is — kept
/// `pub(crate)` purely so `mod tests` below can prove the task exits
/// promptly once the receiver is dropped (review-task9 Important), which
/// isn't observable from `browse`'s signature alone.
pub(crate) fn browse_with_handle()
-> Result<(mpsc::Receiver<DiscoveredPeer>, tokio::task::JoinHandle<()>), TransportError> {
    let daemon =
        ServiceDaemon::new().map_err(|e| sync_error(format!("starting mDNS daemon: {e}")))?;
    let events = daemon
        .browse(SERVICE_TYPE)
        .map_err(|e| sync_error(format!("browsing for {SERVICE_TYPE}: {e}")))?;

    let (tx, rx) = mpsc::channel(32);
    let handle = tokio::spawn(async move {
        // `daemon` is moved into this task so it (and the underlying
        // browse) stays alive for as long as anyone holds the receiver;
        // `drain_events` returning (either the channel closing or the
        // receiver dropping) ends this task and lets `_daemon` drop,
        // tearing the mDNS daemon down.
        let _daemon = daemon;
        drain_events(events, tx).await;
    });
    Ok((rx, handle))
}

/// Turns raw `mdns-sd` events into [`DiscoveredPeer`]s on `tx` until either
/// side closes: `events` closing (the daemon shut down), or `tx` closing
/// (the caller dropped its receiver). `tokio::select!`'s `biased` ordering
/// checks `tx.closed()` first each iteration, so a dropped receiver ends
/// this loop on its own without waiting on the next mDNS event — the fix
/// for review-task9's Important finding, where the old single-armed
/// `while let Ok(event) = events.recv_async().await` only noticed a
/// dropped receiver from inside a successful `tx.send`, never on its own.
async fn drain_events(events: mdns_sd::Receiver<ServiceEvent>, tx: mpsc::Sender<DiscoveredPeer>) {
    loop {
        tokio::select! {
            biased;
            _ = tx.closed() => break,
            event = events.recv_async() => {
                let Ok(event) = event else { break };
                let ServiceEvent::ServiceResolved(info) = event else { continue };
                let Some(addr) = info.get_addresses().iter().next() else { continue };
                let socket_addr = scoped_addr_to_socket_addr(addr, info.get_port());
                let txt: Vec<(String, String)> = info
                    .get_properties()
                    .iter()
                    .map(|p| (p.key().to_string(), p.val_str().to_string()))
                    .collect();
                let Some(peer) = parse_txt(&txt, socket_addr) else { continue };
                if tx.send(peer).await.is_err() {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_txt_then_parse_txt_round_trips_peer_id_name_and_version() {
        // Arrange
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();
        let txt = build_txt("peer-abc", "Isaac's laptop");

        // Act
        let parsed = parse_txt(&txt, addr).expect("valid record parses");

        // Assert
        assert_eq!(parsed.peer_id, "peer-abc");
        assert_eq!(parsed.name, "Isaac's laptop");
        assert_eq!(parsed.protocol_version, PROTOCOL_VERSION);
        assert_eq!(parsed.addr, addr);
    }

    #[test]
    fn parse_txt_a_record_with_no_pid_none() {
        // Arrange
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();
        let txt = vec![("name".to_string(), "peer".to_string()), ("v".to_string(), "1".to_string())];

        // Act
        let parsed = parse_txt(&txt, addr);

        // Assert
        assert_eq!(parsed, None);
    }

    #[test]
    fn parse_txt_a_v_that_is_not_a_number_none() {
        // Arrange
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();
        let txt = vec![
            ("pid".to_string(), "peer-abc".to_string()),
            ("name".to_string(), "peer".to_string()),
            ("v".to_string(), "not-a-number".to_string()),
        ];

        // Act
        let parsed = parse_txt(&txt, addr);

        // Assert
        assert_eq!(parsed, None);
    }

    #[test]
    fn parse_txt_a_v_of_2_against_protocol_version_1_some_carrying_2() {
        // Arrange
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();
        let txt = vec![
            ("pid".to_string(), "peer-abc".to_string()),
            ("name".to_string(), "peer".to_string()),
            ("v".to_string(), "2".to_string()),
        ];

        // Act
        let parsed = parse_txt(&txt, addr).expect("v=2 still parses");

        // Assert
        assert_eq!(parsed.protocol_version, 2);
        assert_ne!(parsed.protocol_version, PROTOCOL_VERSION);
    }

    #[test]
    fn parse_txt_an_empty_name_some_with_an_empty_name() {
        // Arrange
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();
        let txt = vec![
            ("pid".to_string(), "peer-abc".to_string()),
            ("name".to_string(), "".to_string()),
            ("v".to_string(), "1".to_string()),
        ];

        // Act
        let parsed = parse_txt(&txt, addr).expect("empty name still parses");

        // Assert
        assert_eq!(parsed.name, "");
    }

    #[test]
    fn build_txt_the_keys_are_exactly_pid_name_v() {
        // Arrange / Act
        let txt = build_txt("peer-abc", "peer");
        let keys: Vec<&str> = txt.iter().map(|(k, _)| k.as_str()).collect();

        // Assert
        assert_eq!(keys, vec!["pid", "name", "v"]);
    }

    /// review-task9 fix: proves `drain_events` ends promptly when the
    /// caller drops the `DiscoveredPeer` receiver, even though no mDNS
    /// event ever arrives on the (synthetic, no-daemon) event channel — the
    /// exact "quiet LAN" scenario the Important finding described. Uses a
    /// bare `flume` channel rather than a real `ServiceDaemon` so it needs
    /// no multicast and runs in the gate.
    #[tokio::test]
    async fn drain_events_receiver_dropped_no_event_arrives_task_ends() {
        // Arrange
        let (_events_tx, events_rx) = flume::unbounded::<ServiceEvent>();
        let (tx, rx) = mpsc::channel::<DiscoveredPeer>(1);
        let handle = tokio::spawn(drain_events(events_rx, tx));

        // Act
        drop(rx);
        let result =
            tokio::time::timeout(std::time::Duration::from_secs(2), handle).await;

        // Assert
        result.expect("drain_events ends without waiting on an event").expect("task did not panic");
    }

    /// Manual-only: advertises and browses on the real loopback interface.
    /// Not run in the gate — mDNS multicast is not guaranteed to work in a
    /// CI/sandbox network namespace. Run with
    /// `cargo test -p idl-transport sync::discovery -- --ignored`.
    #[tokio::test]
    #[ignore]
    async fn advertise_and_browse_round_trip_on_loopback() {
        // Arrange
        let _advertisement = advertise("peer-loopback", "Loopback peer", 4242)
            .expect("advertise starts the mDNS daemon");
        let mut peers = browse().expect("browse starts the mDNS daemon");

        // Act
        let found = tokio::time::timeout(std::time::Duration::from_secs(5), peers.recv())
            .await
            .ok()
            .flatten();

        // Assert
        let found = found.expect("saw at least one peer within 5s");
        assert_eq!(found.peer_id, "peer-loopback");
    }
}
