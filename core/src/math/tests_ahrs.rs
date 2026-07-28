//! Attitude-from-expressions gate: proves the workbook-level AHRS recovers true
//! lean, pitch, and gravity-removed body acceleration from IMU0 + GPS-derived
//! speed, using only functions the math evaluator already exposes (no engine
//! change, no estimator run).
//!
//! **Why this exists.** Neither accelerometer axis can see attitude on its own.
//! In a coordinated turn the specific-force resultant runs down the bike's own
//! vertical axis, so `asin(a_y)` reads ~0° no matter how far the bike is leaned
//! — the same reason a turn-and-bank ball stays centred. Under braking, `a_x`
//! reads as nose-down pitch that isn't there. Both are fixed by removing the
//! inertial term first: `v·ψ̇` for lean, `dv/dt` for pitch. The two
//! `*_is_blind_*` tests pin those blind spots so the compensation cannot be
//! quietly dropped.
//!
//! **Frames.** IMU0 is mounted X-rear/Y-right (a 180° yaw from the ISO 8855
//! chassis frame X-forward/Y-left/Z-up), so chassis = sensor with X and Y
//! negated — see `estimate::geometry::BikeGeometry::reference_bike`. Roll is
//! positive leaning right; pitch is positive nose-up, matching
//! `estimate::attitude::roll_pitch_deg`.
//!
//! **Shape.** [`AHRS_CHANNELS`] holds the shipped channel definitions verbatim;
//! `resolved_ride` evaluates them in dependency order against a synthetic ride
//! the same way the engine's resolver does against a real session. So these
//! tests exercise the exact expression text the app ships, not a paraphrase.

use crate::math::eval::{evaluate, ChannelLookup, LookupChannel, MathLapContext};

/// Standard gravity, m/s². Matches `estimate::attitude::G_MPS2`.
const G: f64 = 9.80665;
const D2R: f64 = std::f64::consts::PI / 180.0;
const R2D: f64 = 180.0 / std::f64::consts::PI;

const FS: f64 = 800.0;
const CRUISE_MPS: f64 = 8.0;
/// Steady lean during the modelled corner, degrees (negative ⇒ leaning left).
const STEADY_ROLL_DEG: f64 = -20.0;
/// Longitudinal acceleration during the modelled brake, m/s² (negative ⇒
/// slowing). Large enough that an uncompensated accelerometer reads ~5° of
/// phantom nose-down pitch.
const BRAKE_MPS2: f64 = -0.8;
/// Constant gyro bias baked into IMU0_GyroX, dps. Integrated naively this is a
/// 30°-over-60 s ramp — the complementary highpass must reject it.
const GYRO_BIAS_DPS: f64 = 0.5;

// ---- The shipped channel graph ----

/// Chassis longitudinal specific force, g (positive ⇒ forward).
const AX: &str = "(-[IMU0_AccelX])";
/// Chassis lateral specific force, g (positive ⇒ toward the rider's left).
const AY: &str = "(-[IMU0_AccelY])";

/// The AHRS channel definitions, in dependency order.
///
/// **These strings are load-bearing.** They are mirrored verbatim by
/// `kBuiltinMathChannels` in `app/lib/data/math_channel.dart`; this array is
/// what proves they are physically correct. Change one, change both.
///
/// Named intermediates rather than one inlined expression: the resolver stores
/// each by name, so `butter` over the speed channel runs once instead of once
/// per reference, and every stage is independently chartable when a result
/// looks wrong.
///
/// `pi` and `g` are the language's universal constants (SPEC §19) — bare
/// identifiers resolved to literals at parse time, so no magic numbers appear
/// in the expressions.
const AHRS_CHANNELS: &[(&str, &str)] = &[
    // Ground speed, m/s, smoothed at 0.5 Hz. `Distance` is synthesized at the
    // IMU rate by `session::synthesis` (a lazy lerp of the 1 Hz GPS integral),
    // so this lands already rate-matched to the gyro — which is the whole
    // reason the chain is expressible without a resample(). Its derivative is a
    // staircase and its second derivative is impulsive, so it is smoothed once
    // here and reused; both compensation terms need only the slow component.
    ("Speed (m/s)", "butter(2, 0.5, \"low\", differentiate([Distance]))"),
    // Chassis roll rate, dps (positive ⇒ rolling right). IMU0 is mounted
    // X-rear/Y-right, a 180° yaw from the ISO chassis frame, so X and Y negate.
    ("Roll rate (deg/s)", "-[IMU0_GyroX]"),
    // Turn-compensated lean, degrees. Lateral specific force is
    // `f_y = v·ψ̇·cos φ + g·sin φ` — the centripetal term is horizontal, so it
    // projects onto the tilted body-y axis through cos φ. The *body* yaw rate
    // carries the same factor (`g_z = ψ̇·cos φ`), so substituting it cancels the
    // projection exactly: `sin φ = a_y − v·g_z/g`. No small-angle assumption
    // and no iteration — using the nav-frame turn rate here would be wrong, not
    // more accurate.
    //
    // The lowpass sits INSIDE the asin, not after it. On real trail data the
    // raw argument exceeds ±1 g about 10% of the time; clamping those to ±90°
    // and then averaging biases the level. Filtering in the measurement domain,
    // before the nonlinearity, drops saturation to zero. `declip` repairs
    // samples where the accelerometer railed on a hard hit, matching what the
    // estimator's own input adapter does.
    (
        "Roll reference (deg)",
        "asin(clamp(butter(2, 0.2, \"low\", declip(-[IMU0_AccelY]) - [Speed (m/s)] * [IMU0_GyroZ] * pi / 180 / g), -1, 1)) * 180 / pi",
    ),
    // The complementary blend: the (already band-limited) reference carries the
    // level, high-passed integrated gyro carries the dynamics. `butter` is
    // sosfiltfilt (zero-phase, forward-backward), so the crossover adds no
    // phase distortion — that is what makes this usable offline, where a causal
    // filter lags corner entry. The highpass branch also strips gyro bias,
    // which would otherwise integrate into unbounded phantom lean. 0.2 Hz sits
    // below cornering dynamics (~0.5–2 Hz) and well above drift.
    (
        "Roll (deg)",
        "[Roll reference (deg)] + integrate([Roll rate (deg/s)]) - butter(2, 0.2, \"low\", integrate([Roll rate (deg/s)]))",
    ),
    // Nose-up pitch rate, dps. **Not** simply the body Y rate: the Euler rate
    // is `θ̇ = cos φ·q − sin φ·r`, and our convention is nose-up-positive
    // (`-θ`), giving `sin φ·r − cos φ·q`. With q = −[IMU0_GyroY] that is the
    // form below. Skipping this coupling injects a large false pitch rate
    // through a sustained corner — at 20° of lean the sin φ·r term is ~8 dps,
    // which is the entire signal.
    (
        "Pitch rate (deg/s)",
        "sin([Roll (deg)] * pi / 180) * [IMU0_GyroZ] + cos([Roll (deg)] * pi / 180) * [IMU0_GyroY]",
    ),
    // Pitch reference, degrees, with the braking/accelerating term removed:
    // `f_x = a_x + g·sin θ`, so `sin θ = (f_x − dv/dt)/g`. Without this a bare
    // accelerometer reads every brake as nose-down pitch. The lowpass sits
    // inside the asin for the same reason as the roll reference above.
    (
        "Pitch reference (deg)",
        "asin(clamp(butter(2, 0.2, \"low\", declip(-[IMU0_AccelX]) - differentiate([Speed (m/s)]) / g), -1, 1)) * 180 / pi",
    ),
    (
        "Pitch (deg)",
        "[Pitch reference (deg)] + integrate([Pitch rate (deg/s)]) - butter(2, 0.2, \"low\", integrate([Pitch rate (deg/s)]))",
    ),
    // Gravity-removed body acceleration, g. Attitude is what tells us how much
    // of each accelerometer axis is gravity; subtracting it leaves the real
    // acceleration. Longitudinal is positive forward, lateral positive right
    // (`a_lat = sin φ − a_y`, and a_y = −[IMU0_AccelY] folds the sign in).
    ("Longitudinal accel (g)", "declip(-[IMU0_AccelX]) - sin([Pitch (deg)] * pi / 180)"),
    ("Lateral accel (g)", "sin([Roll (deg)] * pi / 180) - declip(-[IMU0_AccelY])"),
];

/// Looks up one shipped definition by name.
fn ahrs_expr(name: &str) -> &'static str {
    AHRS_CHANNELS
        .iter()
        .find(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("no AHRS channel named {name}"))
        .1
}

// ---- Synthetic ride ----

/// Ground-truth roll at time `t`, degrees: upright, a 5 s roll-in to
/// [`STEADY_ROLL_DEG`], a 20 s steady corner, a 5 s roll-out, upright.
fn truth_roll_deg(t: f64) -> f64 {
    match t {
        t if t < 20.0 => 0.0,
        t if t < 25.0 => STEADY_ROLL_DEG * (t - 20.0) / 5.0,
        t if t < 45.0 => STEADY_ROLL_DEG,
        t if t < 50.0 => STEADY_ROLL_DEG * (50.0 - t) / 5.0,
        _ => 0.0,
    }
}

/// Ground-truth roll rate, dps (analytic derivative of [`truth_roll_deg`]).
fn truth_roll_rate_dps(t: f64) -> f64 {
    match t {
        t if (20.0..25.0).contains(&t) => STEADY_ROLL_DEG / 5.0,
        t if (45.0..50.0).contains(&t) => -STEADY_ROLL_DEG / 5.0,
        _ => 0.0,
    }
}

/// Ground-truth speed, m/s: cruise, a 5 s brake to half speed, a 5 s
/// re-acceleration, then cruise for the whole cornering section.
fn truth_speed_mps(t: f64) -> f64 {
    match t {
        t if t < 10.0 => CRUISE_MPS,
        t if t < 15.0 => CRUISE_MPS + BRAKE_MPS2 * (t - 10.0),
        t if t < 20.0 => CRUISE_MPS + BRAKE_MPS2 * (20.0 - t),
        _ => CRUISE_MPS,
    }
}

/// Ground-truth longitudinal acceleration, m/s² (derivative of
/// [`truth_speed_mps`]).
fn truth_accel_mps2(t: f64) -> f64 {
    match t {
        t if (10.0..15.0).contains(&t) => BRAKE_MPS2,
        t if (15.0..20.0).contains(&t) => -BRAKE_MPS2,
        _ => 0.0,
    }
}

/// Deterministic broadband trail chatter, g. A sum of sines well above [`FC`],
/// standing in for the bumps a real accelerometer rides on.
fn chatter(t: f64) -> f64 {
    0.08 * (2.0 * std::f64::consts::PI * 3.1 * t).sin()
        + 0.06 * (2.0 * std::f64::consts::PI * 7.3 * t + 1.0).sin()
        + 0.05 * (2.0 * std::f64::consts::PI * 13.7 * t + 2.0).sin()
}

/// Builds the six IMU0 sensor channels plus `Distance` for a ride that brakes,
/// re-accelerates, then holds a coordinated corner.
///
/// Coordinated means the specific-force resultant stays in the bike's plane of
/// symmetry: chassis `a_y = 0`, and `a_z = 1/cos φ` (the load factor). The
/// required nav-frame turn rate follows from `sin φ = −v·ψ̇/g`. True pitch is
/// zero throughout, so any pitch the chain reports during braking or cornering
/// is an artefact.
fn ride_ctx() -> Ctx {
    let n = (60.0 * FS) as usize;
    let mut ax = Vec::with_capacity(n);
    let mut ay = Vec::with_capacity(n);
    let mut az = Vec::with_capacity(n);
    let mut gx = Vec::with_capacity(n);
    let mut gy = Vec::with_capacity(n);
    let mut gz = Vec::with_capacity(n);
    let mut distance = Vec::with_capacity(n);
    // Cumulative trapezoid of `truth_speed_mps` — exact for a piecewise-linear
    // speed profile, and consistent with it by construction, so
    // `differentiate([Distance])` recovers the speed the fixture intended.
    let mut travelled = 0.0;

    for i in 0..n {
        let t = i as f64 / FS;
        if i > 0 {
            let prev = truth_speed_mps((i - 1) as f64 / FS);
            travelled += 0.5 * (prev + truth_speed_mps(t)) / FS;
        }
        let phi = truth_roll_deg(t) * D2R;
        let v = truth_speed_mps(t);
        // Coordinated turn: the resultant of gravity and the (horizontal)
        // centripetal acceleration lies in the bike's plane of symmetry, so
        // tan φ = −v·ψ̇/g  ⇒  ψ̇ = −g·tan φ / v (nav frame).
        let psi_dot = -G * phi.tan() / v;

        // Chassis-frame specific force, g. Longitudinal carries the brake;
        // lateral is zero by the coordinated condition; vertical carries the
        // load factor. True pitch is zero, so no gravity leaks into X.
        let ax_chassis = truth_accel_mps2(t) / G;
        let ay_chassis = chatter(t);
        let az_chassis = 1.0 / phi.cos() + chatter(t + 0.5);

        // Chassis-frame angular rate, dps. The nav-frame yaw axis projects onto
        // body Y and Z as (sin φ, cos φ) — the Y term is the cross-coupling the
        // pitch rate transformation has to undo.
        let gx_chassis = truth_roll_rate_dps(t) + GYRO_BIAS_DPS;
        let gy_chassis = psi_dot * phi.sin() * R2D;
        let gz_chassis = psi_dot * phi.cos() * R2D;

        // Sensor frame = chassis rotated 180° about Z ⇒ negate X and Y.
        ax.push(-ax_chassis);
        ay.push(-ay_chassis);
        az.push(az_chassis);
        gx.push(-gx_chassis);
        gy.push(-gy_chassis);
        gz.push(gz_chassis);
        distance.push(travelled);
    }

    ctx(&[
        ("IMU0_AccelX", ax, FS),
        ("IMU0_AccelY", ay, FS),
        ("IMU0_AccelZ", az, FS),
        ("IMU0_GyroX", gx, FS),
        ("IMU0_GyroY", gy, FS),
        ("IMU0_GyroZ", gz, FS),
        ("Distance", distance, FS),
    ])
}

// ---- Harness ----

struct Ctx(Vec<(String, Vec<f64>, f64)>);
impl ChannelLookup for Ctx {
    fn lookup(&self, name: &str) -> Option<LookupChannel> {
        self.0
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, s, r)| LookupChannel { samples: s.clone().into(), sample_rate_hz: *r })
    }
    fn best_time_base_dims(&self) -> Option<(usize, f64)> {
        self.0
            .iter()
            .filter(|(_, _, r)| *r > 0.0)
            .map(|(_, s, r)| (s.len(), *r))
            .fold(None, |best, (len, rate)| match best {
                Some((bl, _)) if bl >= len => best,
                _ => Some((len, rate)),
            })
    }
}

fn ctx(pairs: &[(&str, Vec<f64>, f64)]) -> Ctx {
    Ctx(pairs.iter().cloned().map(|(n, s, r)| (n.to_string(), s, r)).collect())
}

/// Evaluates [`AHRS_CHANNELS`] in order, appending each result to the context
/// under its own name — the same store-by-name the engine's
/// `math::resolve::resolve_dependencies` performs against a real
/// `SessionHandle`, without needing to build one. Because the array is already
/// in dependency order, one pass suffices.
fn resolved_ride() -> Ctx {
    let mut c = ride_ctx();
    for (name, expression) in AHRS_CHANNELS {
        let out = evaluate(expression, &c, &MathLapContext::default())
            .unwrap_or_else(|e| panic!("resolving `{name}`: {e}"));
        c.0.push((name.to_string(), out.samples, out.sample_rate_hz));
    }
    c
}

/// The resolved samples of one AHRS channel.
fn resolved(c: &Ctx, name: &str) -> Vec<f64> {
    c.0.iter()
        .find(|(n, _, _)| n == name)
        .unwrap_or_else(|| panic!("channel {name} was not resolved"))
        .1
        .clone()
}

fn eval(expr: &str, c: &Ctx) -> Vec<f64> {
    evaluate(expr, c, &MathLapContext::default())
        .unwrap_or_else(|e| panic!("evaluating `{}...`: {e}", &expr[..expr.len().min(160)]))
        .samples
}

/// Samples over `[from_s, to_s)`.
fn window(samples: &[f64], from_s: f64, to_s: f64) -> &[f64] {
    &samples[(from_s * FS) as usize..(to_s * FS) as usize]
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn std_dev(xs: &[f64]) -> f64 {
    let m = mean(xs);
    (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64).sqrt()
}

// ---- Roll ----

#[test]
fn naive_accel_roll_is_blind_to_lean_in_a_turn() {
    // Arrange — a steady 20° coordinated turn.
    let c = resolved_ride();

    // Act — level off the accelerometer with no turn compensation at all.
    let naive = eval(&format!("asin(clamp({AY}, -1, 1)) * {R2D}"), &c);

    // Assert — physically: in a coordinated turn the resultant runs down the
    // bike's own vertical axis, so the accelerometer reports ~level despite 20°
    // of real lean. This is the failure the turn compensation exists to fix.
    let steady = mean(window(&naive, 30.0, 40.0));
    assert!(
        steady.abs() < 2.0,
        "naive accel roll should read ~0° in a coordinated turn, got {steady:.2}°"
    );
}

#[test]
fn turn_compensated_reference_recovers_true_lean() {
    // Arrange
    let c = resolved_ride();

    // Act
    let reference = resolved(&c, "Roll reference (deg)");

    // Assert — removing v·ψ̇ recovers the real lean the accelerometer could not
    // see. The single refinement pass leaves well under a degree of bias.
    let steady = mean(window(&reference, 30.0, 40.0));
    assert!(
        (steady - STEADY_ROLL_DEG).abs() < 0.75,
        "expected ~{STEADY_ROLL_DEG}°, got {steady:.2}°"
    );
}

#[test]
fn roll_blend_keeps_the_level_and_rejects_accelerometer_noise() {
    // Arrange
    let c = resolved_ride();

    // Act
    let reference = resolved(&c, "Roll reference (deg)");
    let blended = resolved(&c, "Roll (deg)");

    // Assert — same steady level, far less chatter. The reference is memoryless
    // but rides on the raw accelerometer; the blend takes its level and the
    // gyro's much quieter dynamics.
    let steady = mean(window(&blended, 30.0, 40.0));
    assert!(
        (steady - STEADY_ROLL_DEG).abs() < 0.75,
        "blended steady lean expected ~{STEADY_ROLL_DEG}°, got {steady:.2}°"
    );

    let noise_reference = std_dev(window(&reference, 30.0, 40.0));
    let noise_blended = std_dev(window(&blended, 30.0, 40.0));
    assert!(
        noise_blended < noise_reference / 3.0,
        "blend should cut chatter ≥3×: reference σ={noise_reference:.2}°, blended σ={noise_blended:.2}°"
    );
}

#[test]
fn roll_blend_rejects_a_constant_gyro_bias() {
    // Arrange — IMU0_GyroX carries a 0.5 dps bias for the whole 60 s ride.
    let c = resolved_ride();

    // Act
    let gyro_only = eval(&format!("integrate({})", ahrs_expr("Roll rate (deg/s)")), &c);
    let blended = resolved(&c, "Roll (deg)");

    // Assert — raw integration walks away (0.5 dps × ~55 s ≈ 27° of phantom
    // lean); the blend's highpass branch removes the ramp, so upright reads
    // upright at the end of the ride.
    let drift = mean(window(&gyro_only, 50.0, 58.0));
    assert!(drift > 20.0, "test fixture should drift; got {drift:.2}°");

    let settled = mean(window(&blended, 52.0, 58.0));
    assert!(settled.abs() < 1.5, "blended roll should return to ~0°, got {settled:.2}°");
}

#[test]
fn roll_blend_tracks_the_roll_in_without_lag() {
    // Arrange
    let c = resolved_ride();

    // Act
    let blended = resolved(&c, "Roll (deg)");

    // Assert — halfway through the 5 s roll-in the truth is exactly half the
    // steady lean. A causal filter would still be catching up here; the
    // zero-phase blend should sit on the ramp.
    let at_mid = mean(window(&blended, 22.4, 22.6));
    let expected = truth_roll_deg(22.5);
    assert!(
        (at_mid - expected).abs() < 2.0,
        "mid roll-in expected ~{expected:.1}°, got {at_mid:.2}°"
    );
}

// ---- Pitch ----

#[test]
fn naive_accel_pitch_reads_braking_as_phantom_nose_down() {
    // Arrange — a 0.8 m/s² brake with the bike perfectly level.
    let c = resolved_ride();

    // Act — level off the longitudinal axis with no compensation.
    let naive = eval(&format!("asin(clamp({AX}, -1, 1)) * {R2D}"), &c);

    // Assert — physically: an accelerometer cannot tell braking from pitching
    // nose-down. 0.8 m/s² ≈ 0.082 g ≈ 4.7° of pitch that never happened.
    let braking = mean(window(&naive, 11.0, 14.0));
    assert!(
        braking < -3.0,
        "naive accel pitch should show phantom nose-down under braking, got {braking:.2}°"
    );
}

#[test]
fn pitch_reference_removes_the_braking_term() {
    // Arrange
    let c = resolved_ride();

    // Act
    let reference = resolved(&c, "Pitch reference (deg)");

    // Assert — subtracting dv/dt leaves the true (zero) pitch through the brake.
    let braking = mean(window(&reference, 11.0, 14.0));
    assert!(
        braking.abs() < 1.0,
        "compensated pitch should read ~0° under braking, got {braking:.2}°"
    );
}

#[test]
fn pitch_rate_transformation_cancels_the_cornering_cross_coupling() {
    // Arrange — in a 20° corner the nav yaw axis projects ~8 dps onto body Y,
    // even though the bike is not pitching at all.
    let c = resolved_ride();

    // Act
    let raw_gy = eval("-[IMU0_GyroY]", &c);
    let transformed = resolved(&c, "Pitch rate (deg/s)");

    // Assert — the raw body rate is large and wrong; the Euler transformation
    // sin φ·r − cos φ·q cancels it against the yaw term, leaving ~0 dps.
    let raw = mean(window(&raw_gy, 30.0, 40.0));
    assert!(raw.abs() > 5.0, "fixture should have cross-coupling; got {raw:.2} dps");

    let corrected = mean(window(&transformed, 30.0, 40.0));
    assert!(
        corrected.abs() < 1.0,
        "transformed pitch rate should be ~0 dps in a steady corner, got {corrected:.2} dps"
    );
}

#[test]
fn pitch_stays_level_through_braking_and_cornering() {
    // Arrange — true pitch is zero for the entire ride.
    let c = resolved_ride();

    // Act
    let pitch = resolved(&c, "Pitch (deg)");

    // Assert — neither the brake nor the sustained corner should tilt it.
    let braking = mean(window(&pitch, 11.0, 14.0));
    assert!(braking.abs() < 2.0, "pitch under braking expected ~0°, got {braking:.2}°");

    let cornering = mean(window(&pitch, 30.0, 40.0));
    assert!(cornering.abs() < 2.0, "pitch in a corner expected ~0°, got {cornering:.2}°");
}

// ---- Gravity-removed body acceleration ----

#[test]
fn body_accel_long_reports_braking_not_gravity() {
    // Arrange
    let c = resolved_ride();

    // Act
    let along = resolved(&c, "Longitudinal accel (g)");

    // Assert — the brake is real acceleration and must survive gravity removal:
    // −0.8 m/s² ÷ g ≈ −0.082 g.
    let braking = mean(window(&along, 11.0, 14.0));
    let expected = BRAKE_MPS2 / G;
    assert!(
        (braking - expected).abs() < 0.02,
        "expected ~{expected:.3} g under braking, got {braking:.3} g"
    );

    // ...and cruising straight is zero.
    let cruise = mean(window(&along, 30.0, 40.0));
    assert!(cruise.abs() < 0.02, "expected ~0 g cruising, got {cruise:.3} g");
}

#[test]
fn body_accel_lat_reports_cornering_the_bare_accelerometer_cannot_see() {
    // Arrange
    let c = resolved_ride();

    // Act
    let lat = resolved(&c, "Lateral accel (g)");

    // Assert — chassis a_y is ~0 through the whole corner (that is the
    // coordinated condition), yet the bike is pulling real lateral g. This is
    // body-frame acceleration (matching `estimate::attitude::body_accel_g`), so
    // the centripetal term projects through cos φ and it reads sin φ. Negative:
    // a left lean is a left turn, and "acceleration to the right" is then
    // negative. (The *horizontal* lateral g is the larger tan φ.)
    let expected = (STEADY_ROLL_DEG * D2R).sin();
    let cornering = mean(window(&lat, 30.0, 40.0));
    assert!(
        (cornering - expected).abs() < 0.05,
        "expected ~{expected:.3} g in the corner, got {cornering:.3} g"
    );

    // ...and riding straight is zero.
    let straight = mean(window(&lat, 3.0, 8.0));
    assert!(straight.abs() < 0.05, "expected ~0 g riding straight, got {straight:.3} g");
}