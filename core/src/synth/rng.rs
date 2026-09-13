//! Deterministic pseudo-randomness for the synthetic generator (C1 §9.7
//! rule 2), plus the CRC-32 the `.idl0` header needs.
//!
//! PCG32 (O'Neill's `pcg_setseq_64_xsh_rr_32`), chosen because its whole
//! state is one `u64` of integer arithmetic — no floating point, no platform
//! entropy, no `std::collections` hashing — so a given seed produces the same
//! stream on every target. `rand` is not a dependency of this crate and this
//! is not a reason to add one: a fixture generator wants a *frozen* algorithm
//! more than it wants a good one.
//!
//! Gaussian noise is Irwin–Hall: the sum of twelve uniforms on `[0, 1)` minus
//! six, which has mean 0 and variance exactly 1 and needs neither a logarithm
//! nor a trigonometric function (Box–Muller needs both, and `ln` is libm's,
//! whose last ULP is unspecified — see [`super::dtrig`]). Its support is
//! bounded at ±6σ and its kurtosis is slightly low; at the σ this generator
//! uses, nothing downstream can tell the difference.

/// PCG32's multiplier, from the reference implementation. Not a tunable.
const PCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
/// The default stream increment (must be odd).
const PCG_INCREMENT: u64 = 1_442_695_040_888_963_407;

/// A seeded PCG32 stream.
#[derive(Debug, Clone)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    /// Seeds a stream. The seeding sequence is the reference implementation's
    /// `pcg32_srandom_r`: set the state to zero, step once, add the seed,
    /// step again.
    pub fn new(seed: u64) -> Pcg32 {
        let mut rng = Pcg32 { state: 0, inc: PCG_INCREMENT };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// The next 32 bits.
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(PCG_MULTIPLIER).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// A uniform `f64` on `[0, 1)`. The division is by 2³², exact in `f64`,
    /// so every value is representable and the mapping is exactly uniform
    /// over the 2³² outputs.
    pub fn next_unit(&mut self) -> f64 {
        f64::from(self.next_u32()) / 4_294_967_296.0
    }

    /// One draw from an approximately standard normal distribution
    /// (Irwin–Hall with n = 12: mean 0, variance 1, support ±6).
    pub fn next_gaussian(&mut self) -> f64 {
        let mut sum = 0.0;
        for _ in 0..12 {
            sum += self.next_unit();
        }
        sum - 6.0
    }
}

/// CRC-32/ISO-HDLC (zlib/PKZIP/gzip), the variant `docs/IDL0_SPEC.md` §5.1
/// specifies for the header's config checksum: reflected polynomial
/// `0xEDB88320`, init and final XOR `0xFFFFFFFF`.
///
/// Computed bitwise rather than from a lookup table — this runs once per
/// generated file, over a few hundred bytes.
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_the_same_stream() {
        // Arrange
        let mut a = Pcg32::new(42);
        let mut b = Pcg32::new(42);

        // Act
        let first: Vec<u32> = (0..64).map(|_| a.next_u32()).collect();
        let second: Vec<u32> = (0..64).map(|_| b.next_u32()).collect();

        // Assert
        assert_eq!(first, second);
    }

    #[test]
    fn different_seeds_give_different_streams() {
        // Arrange
        let mut a = Pcg32::new(1);
        let mut b = Pcg32::new(2);

        // Act
        let first: Vec<u32> = (0..64).map(|_| a.next_u32()).collect();
        let second: Vec<u32> = (0..64).map(|_| b.next_u32()).collect();

        // Assert
        assert_ne!(first, second);
    }

    #[test]
    fn unit_draws_stay_inside_the_half_open_interval() {
        // Arrange
        let mut rng = Pcg32::new(7);

        // Act / Assert
        for _ in 0..100_000 {
            let u = rng.next_unit();
            assert!((0.0..1.0).contains(&u), "{u}");
        }
    }

    #[test]
    fn gaussian_draws_have_unit_variance_and_zero_mean() {
        // Arrange
        let mut rng = Pcg32::new(9);
        let n = 200_000;

        // Act
        let xs: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;

        // Assert — the standard error of the mean at n = 200000 is ~0.0022.
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "variance {var}");
    }

    #[test]
    fn gaussian_draws_are_bounded_at_six_sigma() {
        // Arrange
        let mut rng = Pcg32::new(11);

        // Act / Assert
        for _ in 0..200_000 {
            let x = rng.next_gaussian();
            assert!(x.abs() <= 6.0, "{x}");
        }
    }

    #[test]
    fn crc32_matches_the_published_check_value() {
        // Arrange — the CRC-32/ISO-HDLC catalogue check: "123456789".
        let input = b"123456789";

        // Act
        let got = crc32(input);

        // Assert
        assert_eq!(got, 0xCBF4_3926);
    }

    #[test]
    fn crc32_of_the_empty_input_is_zero() {
        // Arrange / Act
        let got = crc32(&[]);

        // Assert
        assert_eq!(got, 0);
    }
}
