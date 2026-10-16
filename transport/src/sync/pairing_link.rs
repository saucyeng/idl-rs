//! The QR pairing link (2026-10-16, Isaac: "click the sync button on the
//! computer, snap the QR code"). The offering device shows
//! `idl1://pair?v=1&peer=<id>&name=<name>&code=<6 digits>&addr=<ip:port>…`
//! as a QR code; the phone's camera opens the app with it, and the app pairs
//! through the same `POST /pair` as a typed code. Carrying the addresses
//! means pairing does not depend on the phone seeing the computer over
//! mDNS, which Android networks often block.

use std::net::{IpAddr, SocketAddr};

use crate::error::{TransportError, TransportErrorKind};

/// The link's scheme.
pub const LINK_SCHEME: &str = "idl1";

/// What a pairing link carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairingLink {
    /// The offering device's stable id.
    pub peer_id: String,
    /// Its display name.
    pub name: String,
    /// The six-digit pairing code it is offering.
    pub code: String,
    /// Where its sync server listens, most likely first.
    pub addrs: Vec<SocketAddr>,
}

fn link_error(message: impl Into<String>) -> TransportError {
    TransportError::new(TransportErrorKind::Sync, message.into())
}

/// The link's text, every value percent-encoded.
pub fn build_link(link: &PairingLink) -> String {
    let mut url = reqwest::Url::parse(&format!("{LINK_SCHEME}://pair")).expect("static URL");
    {
        let mut q = url.query_pairs_mut();
        q.append_pair("v", "1");
        q.append_pair("peer", &link.peer_id);
        q.append_pair("name", &link.name);
        q.append_pair("code", &link.code);
        for addr in &link.addrs {
            q.append_pair("addr", &addr.to_string());
        }
    }
    url.to_string()
}

/// Reads a link made by [`build_link`]. Refuses another scheme or route, a
/// version other than 1, a missing peer id, a code that is not six digits,
/// or a link with no usable address.
pub fn parse_link(text: &str) -> Result<PairingLink, TransportError> {
    let url = reqwest::Url::parse(text.trim()).map_err(|e| link_error(format!("not a pairing link: {e}")))?;
    if url.scheme() != LINK_SCHEME || url.host_str() != Some("pair") {
        return Err(link_error(format!("not a pairing link: {text}")));
    }
    let (mut version, mut peer_id, mut name, mut code, mut addrs) = (None, None, String::new(), None, Vec::new());
    for (key, value) in url.query_pairs() {
        match key.as_ref() {
            "v" => version = Some(value.into_owned()),
            "peer" => peer_id = Some(value.into_owned()),
            "name" => name = value.into_owned(),
            "code" => code = Some(value.into_owned()),
            "addr" => {
                if let Ok(addr) = value.parse::<SocketAddr>() {
                    addrs.push(addr);
                }
            }
            _ => {}
        }
    }
    if version.as_deref() != Some("1") {
        return Err(link_error("this pairing link is from a newer app version: update the app"));
    }
    let peer_id = peer_id.filter(|p| !p.is_empty()).ok_or_else(|| link_error("pairing link has no device id"))?;
    let code = code.filter(|c| c.len() == 6 && c.chars().all(|ch| ch.is_ascii_digit())).ok_or_else(|| link_error("pairing link has no six-digit code"))?;
    if addrs.is_empty() {
        return Err(link_error("pairing link has no address to reach the other device on"));
    }
    Ok(PairingLink { peer_id, name, code, addrs })
}

/// Whether `ip` is a private LAN address a phone on the same network could
/// reach: 10/8, 172.16/12, 192.168/16, or a link-local 169.254/16.
fn is_lan(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_private() || v4.is_link_local(),
        IpAddr::V6(_) => false,
    }
}

/// This machine's LAN addresses with `port`, for the link: private IPv4
/// interfaces, 192.168/16 first (home networks), loopback and public
/// addresses left out.
pub fn lan_addrs(port: u16) -> Vec<SocketAddr> {
    let mut ips: Vec<IpAddr> = if_addrs::get_if_addrs()
        .map(|list| list.into_iter().filter(|i| !i.is_loopback()).map(|i| i.ip()).filter(is_lan).collect())
        .unwrap_or_default();
    ips.sort_by_key(|ip| match ip {
        IpAddr::V4(v4) if v4.octets()[0] == 192 => 0,
        IpAddr::V4(v4) if v4.octets()[0] == 10 => 1,
        _ => 2,
    });
    ips.dedup();
    ips.into_iter().map(|ip| SocketAddr::new(ip, port)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> PairingLink {
        PairingLink {
            peer_id: "a1b2c3".to_string(),
            name: "Isaac's PC & bench".to_string(),
            code: "042137".to_string(),
            addrs: vec!["192.168.1.20:47801".parse().unwrap(), "10.0.0.5:47801".parse().unwrap()],
        }
    }

    #[test]
    fn build_link_then_parse_link_round_trips_with_awkward_characters() {
        // Arrange
        let link = sample();

        // Act
        let text = build_link(&link);
        let back = parse_link(&text).unwrap();

        // Assert
        assert!(text.starts_with("idl1://pair?v=1&peer=a1b2c3&"), "{text}");
        assert_eq!(back, link);
    }

    #[test]
    fn parse_link_refuses_wrong_scheme_bad_code_newer_version_and_no_address() {
        // Arrange
        let cases = [
            "https://pair?v=1&peer=a&code=123456&addr=192.168.1.2:1",
            "idl1://pair?v=1&peer=a&code=12345&addr=192.168.1.2:1",
            "idl1://pair?v=2&peer=a&code=123456&addr=192.168.1.2:1",
            "idl1://pair?v=1&peer=a&code=123456",
            "idl1://pair?v=1&code=123456&addr=192.168.1.2:1",
        ];

        // Act
        let results: Vec<bool> = cases.iter().map(|c| parse_link(c).is_err()).collect();

        // Assert
        assert_eq!(results, vec![true; 5]);
    }

    #[test]
    fn is_lan_keeps_private_and_link_local_ipv4_only() {
        // Arrange
        let ips: Vec<IpAddr> = ["192.168.1.2", "10.1.2.3", "172.20.0.1", "169.254.3.4", "8.8.8.8", "::1"].iter().map(|s| s.parse().unwrap()).collect();

        // Act
        let kept: Vec<bool> = ips.iter().map(is_lan).collect();

        // Assert
        assert_eq!(kept, vec![true, true, true, true, false, false]);
    }
}
