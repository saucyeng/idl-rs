//! Math-expression value types. Mirrors the Dart `_Value` hierarchy
//! (`app/lib/data/math_channel_evaluator.dart` — `_Channel`/`_Scalar`/`_StringVal`).

use std::sync::Arc;

/// Which axis a rank-1 math value's entries run along — C2 §3.6.1's `Axis
/// kind`, cut down to the two kinds the evaluator can actually produce today
/// (ruling R233, the "minimal shapes" scope). `[t]` is every session channel
/// and every elementwise result over one; `[lap]` is one entry per lap of the
/// selected window, produced only by the lap-shaped builtins (`lap_number()`,
/// `lap_time()`, `sector_time(i)`).
///
/// §3.6.1's `freq`, `window`, `component` and `index` kinds are deliberately
/// **not** here: no evaluator path produces them, and a variant nothing can
/// construct is a claim the code does not back. Adding one is the work of the
/// lane that makes such a value, not a placeholder to be filled in later.
///
/// This is the value-side vocabulary; [`crate::workbook::v3::AxisKind`] is the
/// wire-side one (C3 §3.4's `axis_kind`), and `From<ValueAxis>` is the single
/// mapping between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValueAxis {
    /// Seconds since the session's first sample. The default: every value
    /// that existed before shapes did is on this axis, so nothing landed
    /// changes meaning.
    #[default]
    Time,
    /// Ordinal lap number, 1-based (C1 `laps[]`). One entry per lap of the
    /// selected window.
    Lap,
}

impl ValueAxis {
    /// The axis written the way C2 §3.6.1 writes it (`t`, `lap`) — used in
    /// error messages so a shape mismatch names the two shapes in the same
    /// vocabulary the spec and the graph card's port labels use.
    pub fn symbol(self) -> &'static str {
        match self {
            ValueAxis::Time => "t",
            ValueAxis::Lap => "lap",
        }
    }
}

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
    /// Per-sample recording time, **microseconds since the session's first
    /// sample** — the C1 §8 item 5 resolution: this is the same clock as
    /// [`crate::session::Channel::t_us`], carried through evaluation so a
    /// derived channel stays aligned to its source's *real* recorded time
    /// rather than a fabricated `i / rate` ramp. `samples.len() == t_us.len()`
    /// whenever `t_us` is non-empty. Empty is the explicit "no established
    /// time axis" marker — used for scalar-as-channel results, rate-0 table
    /// columns, and any value with no single source channel to inherit from
    /// (e.g. an FFT's frequency bins); never filled with a synthetic ramp.
    /// An `Arc<[i64]>` for the same reason `samples` is: shared, not
    /// re-copied, across every reference to one widened channel.
    ///
    /// **On a [`ValueAxis::Lap`] value this field carries the 1-based lap
    /// number, not microseconds** — it is the axis's coordinate vector (C2
    /// §3.6.1's `coords`), and `axis` says what the numbers mean. Every
    /// consumer that converts it to seconds must branch on `axis` first;
    /// [`crate::workbook::v3::to_host_channel`] is that one conversion site.
    pub t_us: Arc<[i64]>,
    /// What [`Self::t_us`] measures — C2 §3.6.1's axis kind, minimal subset.
    /// [`ValueAxis::Time`] for every channel read from a session and every
    /// elementwise result over one.
    pub axis: ValueAxis,
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
            t_us: Arc::from(&[] as &[i64]),
            axis: ValueAxis::Time,
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
    fn value_axis_defaults_to_time_and_writes_itself_the_way_the_spec_does() {
        // Arrange / Act / Assert — the default keeps every pre-shapes value
        // on the time axis; the symbols are C2 §3.6.1's written forms.
        assert_eq!(ValueAxis::default(), ValueAxis::Time);
        assert_eq!(ValueAxis::Time.symbol(), "t");
        assert_eq!(ValueAxis::Lap.symbol(), "lap");
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
                t_us: Arc::from(&[] as &[i64]),
                axis: ValueAxis::Time,
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
