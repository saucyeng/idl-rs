//! The observability ledger (design §3): per-state-component **confidence** (the
//! marginal standard deviation from the final covariance) plus a **DC-source**
//! classification — whether the run actually pinned a component's absolute level or
//! only observed its relative/AC content.
//!
//! The DC-source is inferred unit-agnostically by comparing each component's final
//! marginal σ to its initial prior σ: a meaningful reduction means the factors
//! injected absolute information (Pinned); no reduction (or growth) means only
//! relative information was available (RelativeOnly) — e.g. travel with no stationary
//! window to anchor its DC. Frozen (un-estimated) components are reported as such.
//! Attributing *which* factor pins each component is a richer M3 concern.

use crate::estimate::iekf::FilterState;
use crate::estimate::schema::StateSchema;

/// Where a component's absolute (DC) level came from this run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DcSource {
    /// A factor injected absolute information (final σ ≪ prior σ).
    Pinned,
    /// Only relative/AC content observed; the DC level is unanchored.
    RelativeOnly,
    /// Component not estimated this run (held frozen at its initial value).
    Frozen,
}

/// Per-component observability summary.
#[derive(Debug, Clone)]
pub struct ComponentObservability {
    /// Component symbol (matches the schema).
    pub symbol: &'static str,
    /// Marginal standard deviation (RMS over the component's error block).
    pub confidence_sd: f64,
    /// DC-source classification.
    pub dc_source: DcSource,
}

/// The full per-component ledger for one run.
#[derive(Debug, Clone)]
pub struct ObservabilityLedger {
    /// One entry per state component, in schema order.
    pub components: Vec<ComponentObservability>,
}

impl ObservabilityLedger {
    /// Builds the ledger from the initial and final filter beliefs and the schema. A
    /// component is `Pinned` when its final marginal σ dropped below
    /// `pin_ratio × initial σ` (default callers use ≈0.5), `RelativeOnly` otherwise,
    /// and `Frozen` when the schema marks it inactive.
    pub fn build(
        initial: &FilterState,
        final_belief: &FilterState,
        schema: &StateSchema,
        pin_ratio: f64,
    ) -> ObservabilityLedger {
        let components = schema
            .components
            .iter()
            .map(|c| {
                let sd = block_sd(&final_belief.p, c.error_index, c.dim);
                let dc_source = if !c.active {
                    DcSource::Frozen
                } else {
                    let init_sd = block_sd(&initial.p, c.error_index, c.dim);
                    if sd < pin_ratio * init_sd {
                        DcSource::Pinned
                    } else {
                        DcSource::RelativeOnly
                    }
                };
                ComponentObservability { symbol: c.symbol, confidence_sd: sd, dc_source }
            })
            .collect();
        ObservabilityLedger { components }
    }

    /// Looks up a component's summary by symbol.
    pub fn get(&self, symbol: &str) -> Option<&ComponentObservability> {
        self.components.iter().find(|c| c.symbol == symbol)
    }
}

/// RMS of the covariance diagonal over a component's error block `[start, start+dim)`.
fn block_sd(p: &nalgebra::DMatrix<f64>, start: usize, dim: usize) -> f64 {
    let sum: f64 = (0..dim).map(|k| p[(start + k, start + k)]).sum();
    (sum / dim as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::geometry::BikeGeometry;
    use crate::estimate::iekf::{FilterState, Iekf, InitStd};
    use crate::estimate::measurements::gravity::GravityLeveling;
    use crate::estimate::measurements::zupt::{GyroBias, ZeroAngularRate, ZeroVelocity};
    use crate::estimate::model::{ImuInput, SampleContext};
    use crate::estimate::process::{MtbProcess, GRAVITY};
    use crate::estimate::schema::StateSchema;
    use crate::estimate::state::MtbState;
    use nalgebra::{UnitQuaternion, Vector3};

    fn schema() -> StateSchema {
        StateSchema::from_geometry(&BikeGeometry::reference_bike(), false)
    }

    fn rest_state() -> MtbState {
        MtbState {
            r_chassis: UnitQuaternion::identity(),
            v_chassis: Vector3::zeros(),
            b_g0: Vector3::zeros(),
            b_a0: Vector3::zeros(),
            b_g1: Vector3::zeros(),
            b_g2: Vector3::zeros(),
            d_f: 0.0,
            dd_f: 0.0,
            s_r: 0.0,
            ds_r: 0.0,
            psi: 0.0,
            dpsi: 0.0,
        }
    }

    #[test]
    fn frozen_components_are_reported_frozen() {
        // Arrange — M2a freezes steering; initial == final (no run needed for this).
        let init = FilterState::initial(rest_state(), &schema(), &InitStd::default());

        // Act
        let ledger = ObservabilityLedger::build(&init, &init, &schema(), 0.5);

        // Assert
        assert_eq!(ledger.get("psi").unwrap().dc_source, DcSource::Frozen);
        assert_eq!(ledger.get("dpsi").unwrap().dc_source, DcSource::Frozen);
    }

    #[test]
    fn static_run_pins_velocity_and_bias_but_yaw_keeps_attitude_relative() {
        // Arrange — a stationary run with ZUPT + ZARU + gravity drives down the
        // marginal σ of velocity and b_g0 (absolute information injected). Attitude
        // is only partly observed: gravity pins tilt but yaw is a free gauge at rest
        // (no motion, no GPS course), so the 3-axis attitude block stays RelativeOnly.
        let f = Iekf::new(MtbProcess::new(schema()));
        let init = FilterState::initial(rest_state(), &schema(), &InitStd::default());
        let mut fs = init.clone();
        let dt = 0.01;
        let st = SampleContext { stationary: true };
        let u = ImuInput { gyro0: Vector3::zeros(), accel0: Vector3::new(0.0, 0.0, GRAVITY), wheel_accel_front: 0.0, wheel_accel_rear: 0.0 };
        for _ in 0..500 {
            fs = f.predict(&fs, &u, dt);
            fs = f.update(&fs, &ZeroVelocity { sigma: 0.01 }, &st);
            fs = f.update(&fs, &ZeroAngularRate { target: GyroBias::Imu0, measured: Vector3::zeros(), sigma: 1e-3 }, &st);
            fs = f.update(&fs, &GravityLeveling { accel0: Vector3::new(0.0, 0.0, GRAVITY), a_kin_nav: Vector3::zeros(), sigma: 0.02 }, &st);
        }

        // Act
        let ledger = ObservabilityLedger::build(&init, &fs, &schema(), 0.5);

        // Assert — velocity and gyro bias became confidently observed; attitude
        // stays relative-only because yaw is unobservable at rest (the ledger
        // correctly flags the rest-yaw gauge freedom).
        assert_eq!(ledger.get("v_chassis").unwrap().dc_source, DcSource::Pinned);
        assert_eq!(ledger.get("b_g0").unwrap().dc_source, DcSource::Pinned);
        assert_eq!(ledger.get("R_chassis").unwrap().dc_source, DcSource::RelativeOnly);
        // Confidence is a real, positive σ.
        assert!(ledger.get("v_chassis").unwrap().confidence_sd > 0.0);
    }

    #[test]
    fn unanchored_front_travel_is_relative_only() {
        // Arrange — predict-only (no sag/ZUPT on travel): the wheel double-integrator
        // covariance inflates, so its DC stays unpinned.
        let f = Iekf::new(MtbProcess::new(schema()));
        let init = FilterState::initial(rest_state(), &schema(), &InitStd::default());
        let mut fs = init.clone();
        let dt = 0.01;
        let u = ImuInput { gyro0: Vector3::zeros(), accel0: Vector3::new(0.0, 0.0, GRAVITY), wheel_accel_front: 0.0, wheel_accel_rear: 0.0 };
        for _ in 0..500 {
            fs = f.predict(&fs, &u, dt);
        }

        // Act
        let ledger = ObservabilityLedger::build(&init, &fs, &schema(), 0.5);

        // Assert — travel DC unobserved without an anchor.
        assert_eq!(ledger.get("w_f").unwrap().dc_source, DcSource::RelativeOnly);
    }
}
