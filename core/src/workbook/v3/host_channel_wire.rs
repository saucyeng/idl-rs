//! `IDLH` v1 binary wire encoder (C3 §3.4, ruling R59 Q3(a)'s 24-byte
//! padded header) for a [`super::HostChannel`] crossing Tauri IPC as raw
//! bytes — the byte-path half of the host-channel story; a `HostChannelRef`-
//! shaped JSON marker (built in `idl-rs-tauri`, not here) stays the light
//! "does a value exist" half. Kept as its own module, not folded into
//! `host.rs`, so this pure byte-layout concern stays apart from `host.rs`'s
//! channel lookup/conversion logic.

use super::HostChannel;

/// Magic bytes identifying an `IDLH` v1 payload (C3 §3.4).
const MAGIC: &[u8; 4] = b"IDLH";

/// The `IDLH` wire format's version number (C3 §3.4).
const VERSION: u16 = 1;

/// `flags` bit 0: the payload carries a non-empty `t` (recorded axis).
const FLAG_HAS_T: u16 = 1;

/// Encodes `hc`, decimated to at most `budget` points, as `IDLH` v1 bytes
/// (C3 §3.4, ruling R59 Q3(a)'s 24-byte padded header). Little-endian
/// throughout: `magic` (4 bytes, ASCII "IDLH") at offset 0, `version`
/// (`u16`, `1`) at offset 4, `flags` (`u16`, bit 0 = `has_t`) at offset 6,
/// `length` (`u32`, number of `f64` in `v`) at offset 8, `t_length` (`u32`,
/// number of `f64` in `t`) at offset 12, an 8-byte zero-filled `reserved` at
/// offset 16 padding the header to 24 bytes so both payload arrays start on
/// an 8-byte boundary. Then `t` as `t_length` times `f64` (seconds) at
/// offset 24, then `v` as `length` times `f64` at offset `24 + t_length*8`.
///
/// Decimation is a plain fixed-stride sample (`step = ceil(hc.v.len() /
/// budget)`, keep every `step`-th index), **not**
/// `crate::chart_decimation::decimate_channel`'s bucket min/max: that
/// function doubles point count with a `(min, max)` pair per bucket for a
/// tile's rendering, which does not fit a host channel bound to a JS
/// variable and plotted as a single-value-per-point line series. `v` and
/// (when non-empty) `t` are decimated together at the same indices so a
/// sample's value and its recorded time stay paired. `hc.t` is already in
/// seconds by the time it reaches this function (`super::to_host_channel`
/// does the microsecond-to-second conversion) — this encoder does not
/// repeat it.
///
/// `budget` is assumed to be at least 1 — validating the caller's `budget`
/// argument (C3 §3.4's `1..=65536` range) is the calling command's job,
/// before this function is ever reached; this function does not exceed
/// `budget` but does not itself validate it.
pub fn encode_host_channel_idlh(hc: &HostChannel, budget: u32) -> Vec<u8> {
    let budget = budget.max(1) as usize;
    let step = hc.v.len().div_ceil(budget).max(1);

    let v: Vec<f64> = hc.v.iter().step_by(step).copied().collect();
    let t: Vec<f64> = if hc.t.is_empty() { Vec::new() } else { hc.t.iter().step_by(step).copied().collect() };

    let has_t = !t.is_empty();
    let length = v.len() as u32;
    let t_length = t.len() as u32;

    let mut out = Vec::with_capacity(24 + t.len() * 8 + v.len() * 8);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(if has_t { FLAG_HAS_T } else { 0u16 }).to_le_bytes());
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(&t_length.to_le_bytes());
    out.extend_from_slice(&[0u8; 8]);

    for x in &t {
        out.extend_from_slice(&x.to_le_bytes());
    }
    for x in &v {
        out.extend_from_slice(&x.to_le_bytes());
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_u16(bytes: &[u8], offset: usize) -> u16 {
        u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
    }

    fn read_u32(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

    fn read_f64(bytes: &[u8], offset: usize) -> f64 {
        f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
    }

    #[test]
    fn encode_host_channel_idlh_a_channel_with_a_recorded_axis_round_trips_every_value_and_flag() {
        // Arrange
        let hc = HostChannel { length: 3, t: vec![0.0, 0.1, 0.2], v: vec![1.0, 2.0, 3.0], unit: crate::math::units::UnitLabel::Dimensionless };

        // Act
        let bytes = encode_host_channel_idlh(&hc, 65536);

        // Assert — header
        assert_eq!(&bytes[0..4], b"IDLH");
        assert_eq!(read_u16(&bytes, 4), 1);
        assert_eq!(read_u16(&bytes, 6), 1, "flags bit 0 (has_t) must be set");
        assert_eq!(read_u32(&bytes, 8), 3, "length");
        assert_eq!(read_u32(&bytes, 12), 3, "t_length");
        assert_eq!(&bytes[16..24], &[0u8; 8], "reserved is zero-filled");

        // Assert — t region starts at offset 24
        assert_eq!(read_f64(&bytes, 24), 0.0);
        assert_eq!(read_f64(&bytes, 32), 0.1);
        assert_eq!(read_f64(&bytes, 40), 0.2);

        // Assert — v region starts at offset 24 + t_length*8 = 48
        assert_eq!(read_f64(&bytes, 48), 1.0);
        assert_eq!(read_f64(&bytes, 56), 2.0);
        assert_eq!(read_f64(&bytes, 64), 3.0);

        assert_eq!(bytes.len(), 24 + 3 * 8 + 3 * 8);
    }

    #[test]
    fn encode_host_channel_idlh_an_empty_t_channel_encodes_t_length_zero_flag_clear_and_v_starts_at_24() {
        // Arrange — a scalar/table-column result: no recorded axis.
        let hc = HostChannel { length: 1, t: Vec::new(), v: vec![5.0], unit: crate::math::units::UnitLabel::Dimensionless };

        // Act
        let bytes = encode_host_channel_idlh(&hc, 65536);

        // Assert
        assert_eq!(read_u16(&bytes, 6), 0, "flags bit 0 (has_t) must be clear");
        assert_eq!(read_u32(&bytes, 8), 1, "length");
        assert_eq!(read_u32(&bytes, 12), 0, "t_length");
        assert_eq!(read_f64(&bytes, 24), 5.0, "v starts immediately at offset 24, no gap for an absent t");
        assert_eq!(bytes.len(), 24 + 1 * 8);
    }

    #[test]
    fn encode_host_channel_idlh_source_exceeds_budget_decimates_to_at_most_budget_points() {
        // Arrange — 10 samples, budget 3: step = ceil(10/3) = 4, indices 0,4,8 -> 3 points.
        let v: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
        let hc = HostChannel { length: 10, t, v, unit: crate::math::units::UnitLabel::Dimensionless };

        // Act
        let bytes = encode_host_channel_idlh(&hc, 3);

        // Assert
        let length = read_u32(&bytes, 8);
        let t_length = read_u32(&bytes, 12);
        assert!(length <= 3, "length must not exceed budget");
        assert_eq!(length, t_length, "v and t decimated together, same point count");
        assert_eq!(length, 3, "fixed stride 4 over 10 samples lands exactly on budget here");

        // paired: value at index i must match the value at the same source index as its time
        assert_eq!(read_f64(&bytes, 24), 0.0); // t[0]
        assert_eq!(read_f64(&bytes, 24 + 3 * 8), 0.0); // v[0]
        assert_eq!(read_f64(&bytes, 24 + 8), 0.4); // t[4]
        assert_eq!(read_f64(&bytes, 24 + 3 * 8 + 8), 4.0); // v[4]
    }

    #[test]
    fn encode_host_channel_idlh_source_at_or_under_budget_is_a_no_op() {
        // Arrange
        let hc = HostChannel { length: 3, t: vec![0.0, 0.1, 0.2], v: vec![1.0, 2.0, 3.0], unit: crate::math::units::UnitLabel::Dimensionless };

        // Act
        let bytes = encode_host_channel_idlh(&hc, 100);

        // Assert
        assert_eq!(read_u32(&bytes, 8), 3);
        assert_eq!(read_u32(&bytes, 12), 3);
    }

    #[test]
    fn encode_host_channel_idlh_header_is_exactly_24_bytes_in_every_case() {
        // Arrange
        let with_t = HostChannel { length: 2, t: vec![0.0, 1.0], v: vec![1.0, 2.0], unit: crate::math::units::UnitLabel::Dimensionless };
        let without_t = HostChannel { length: 2, t: Vec::new(), v: vec![1.0, 2.0], unit: crate::math::units::UnitLabel::Dimensionless };

        // Act
        let a = encode_host_channel_idlh(&with_t, 65536);
        let b = encode_host_channel_idlh(&without_t, 65536);

        // Assert — the first `t` byte always sits at offset 24 (the header's
        // own fixed size); the first `v` byte sits at `24 + t_length*8`,
        // computed from the encoder's own header field, not assumed.
        let t_len_a = read_u32(&a, 12) as usize;
        assert_eq!(t_len_a, 2);
        assert_eq!(24 + t_len_a * 8, 40, "with_t's first v byte is at 24 + t_length*8");

        let t_len_b = read_u32(&b, 12) as usize;
        assert_eq!(t_len_b, 0);
        assert_eq!(24 + t_len_b * 8, 24, "without_t's first v byte is at 24, no gap for an absent t");
    }
}
