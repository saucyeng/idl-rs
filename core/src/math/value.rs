//! Math-expression value types. Mirrors the Dart `_Value` hierarchy
//! (`app/lib/data/math_channel_evaluator.dart` — `_Channel`/`_Scalar`/`_StringVal`).

use std::sync::Arc;

/// A time-series channel value produced or consumed during evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelValue {
    /// Sample values, in the channel's physical units. An `Arc<[f64]>` so a
    /// `[Name]` reference and every operand share the one widened buffer instead
    /// of cloning; only operations that produce genuinely new data allocate
    /// (wrapping their output `Vec` via `Arc::from`). The per-pass `MemoLookup`
    /// widens a referenced channel once.
    pub samples: Arc<[f64]>,
    /// Sample rate in Hz. `0.0` denotes a scalar-as-channel (rate-0, one sample).
    pub sample_rate_hz: f64,
    /// Source registry name when this is the direct resolution of a
    /// `[ChannelName]` reference; `None` for derived (function/arithmetic)
    /// results. Used by `variance_*` to find the matching overlay channel.
    pub channel_id: Option<String>,
}

/// A 3-vector intermediate value: three component values sharing the
/// broadcasting rules of the rest of the evaluator. Each component is itself a
/// [`Value::Scalar`] (one value broadcast across every sample) or a
/// [`Value::Channel`] (one value per sample); the vector/rotation functions
/// produce and consume `Vec3` element-wise over the component buffers.
///
/// A `Vec3` is an **intermediate** value: charts plot scalars, so the top-level
/// result of an expression must reduce to a scalar channel. A vector expression
/// is reduced via `vx`/`vy`/`vz` (component) or `norm` (magnitude) — a top-level
/// `Vec3` is rejected by [`evaluate`](crate::math::evaluate). See SPEC §19.
#[derive(Debug, Clone, PartialEq)]
pub struct Vec3Value {
    /// X component — a scalar (broadcast) or a per-sample channel.
    pub x: Value,
    /// Y component — a scalar (broadcast) or a per-sample channel.
    pub y: Value,
    /// Z component — a scalar (broadcast) or a per-sample channel.
    pub z: Value,
}

/// A runtime value: a channel, a dimensionless scalar, a string literal
/// (strings appear only as function arguments, e.g. `fft(ch, "hann")`), or an
/// intermediate 3-vector (see [`Vec3Value`]).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Channel(ChannelValue),
    Scalar(f64),
    Str(String),
    /// A 3-vector built by `vec(...)` and the vector/rotation functions. Boxed
    /// because [`Vec3Value`] contains `Value` components (recursive type).
    Vec3(Box<Vec3Value>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_channel_holds_samples_rate_and_optional_id() {
        // Arrange / Act
        let v = Value::Channel(ChannelValue {
            samples: vec![1.0, 2.0, 3.0].into(),
            sample_rate_hz: 100.0,
            channel_id: Some("IMU0_AccelZ".to_string()),
        });

        // Assert
        match v {
            Value::Channel(c) => {
                assert_eq!(c.samples, vec![1.0, 2.0, 3.0].into());
                assert_eq!(c.sample_rate_hz, 100.0);
                assert_eq!(c.channel_id.as_deref(), Some("IMU0_AccelZ"));
            }
            _ => panic!("expected channel"),
        }
    }

    #[test]
    fn value_scalar_and_string_construct() {
        // Arrange / Act / Assert
        assert!(matches!(Value::Scalar(2.5), Value::Scalar(x) if x == 2.5));
        assert!(matches!(Value::Str("hann".to_string()), Value::Str(s) if s == "hann"));
    }

    #[test]
    fn value_vec3_holds_scalar_and_channel_components() {
        // Arrange / Act — a vector with two scalar components and one channel.
        let v = Value::Vec3(Box::new(Vec3Value {
            x: Value::Scalar(1.0),
            y: Value::Scalar(2.0),
            z: Value::Channel(ChannelValue {
                samples: vec![3.0, 4.0].into(),
                sample_rate_hz: 10.0,
                channel_id: None,
            }),
        }));

        // Assert
        match v {
            Value::Vec3(b) => {
                assert!(matches!(b.x, Value::Scalar(x) if x == 1.0));
                assert!(matches!(b.y, Value::Scalar(y) if y == 2.0));
                assert!(matches!(b.z, Value::Channel(c) if c.samples == vec![3.0, 4.0].into()));
            }
            _ => panic!("expected vec3"),
        }
    }
}
