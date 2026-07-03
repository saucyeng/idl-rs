//! Soft travel priors (design §5): the **sag prior** (mean wheel travel ≈ sag —
//! anchors travel DC "semi-confidently") and the **bounded-travel barrier** (a
//! soft one-sided quadratic keeping travel within `[0, travel_max]`). Both are soft
//! factors on a single wheel-travel scalar.

use crate::estimate::measurements::{I_WF, I_WR};
use crate::estimate::model::{MeasurementModel, SampleContext};
use crate::estimate::state::MtbState;
use nalgebra::{DMatrix, DVector};

/// Which wheel's travel a scalar factor acts on (selects the error-state column).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wheel {
    /// Front wheel travel, column `I_WF`.
    Front,
    /// Rear wheel travel, column `I_WR`.
    Rear,
}

impl Wheel {
    fn col(self) -> usize {
        match self {
            Wheel::Front => I_WF,
            Wheel::Rear => I_WR,
        }
    }

    fn travel_of(self, x: &MtbState) -> f64 {
        match self {
            Wheel::Front => x.d_f,
            Wheel::Rear => x.s_r,
        }
    }
}

/// Sag prior: soft pull of wheel travel toward the static sag value. Residual
/// `r = sag − w`. Loosely weighted (the runner widens `sigma` away from
/// near-static windows, where mean-while-descending biases above static sag).
#[derive(Debug, Clone, Copy)]
pub struct SagPrior {
    /// Target wheel.
    pub wheel: Wheel,
    /// Static sag travel, m.
    pub sag: f64,
    /// Prior std-dev, m.
    pub sigma: f64,
}

impl MeasurementModel<MtbState> for SagPrior {
    fn dim(&self) -> usize {
        1
    }

    fn residual(&self, x: &MtbState) -> DVector<f64> {
        DVector::from_row_slice(&[self.sag - self.wheel.travel_of(x)])
    }

    fn jacobian(&self, _x: &MtbState) -> DMatrix<f64> {
        let mut h = DMatrix::zeros(1, MtbState::DOF);
        h[(0, self.wheel.col())] = -1.0;
        h
    }

    fn noise(&self, _x: &MtbState) -> DMatrix<f64> {
        DMatrix::from_element(1, 1, self.sigma.powi(2))
    }

    fn active(&self, _x: &MtbState, _ctx: &SampleContext) -> bool {
        true
    }
}

/// Bounded-travel barrier: a soft one-sided quadratic keeping travel within
/// `[0, travel_max]`. The 2-row residual `[relu(−w), relu(w − travel_max)]` is zero
/// inside the band and grows linearly outside, so its squared cost is a one-sided
/// quadratic wall. Jacobian is piecewise-constant (one-sided).
#[derive(Debug, Clone, Copy)]
pub struct TravelBarrier {
    /// Target wheel.
    pub wheel: Wheel,
    /// Maximum travel, m.
    pub travel_max: f64,
    /// Barrier std-dev (smaller ⇒ stiffer wall), m.
    pub sigma: f64,
}

impl MeasurementModel<MtbState> for TravelBarrier {
    fn dim(&self) -> usize {
        2
    }

    fn residual(&self, x: &MtbState) -> DVector<f64> {
        let w = self.wheel.travel_of(x);
        let below = (-w).max(0.0); // > 0 when w < 0
        let above = (w - self.travel_max).max(0.0); // > 0 when w > travel_max
        DVector::from_row_slice(&[below, above])
    }

    fn jacobian(&self, x: &MtbState) -> DMatrix<f64> {
        let w = self.wheel.travel_of(x);
        let c = self.wheel.col();
        let mut h = DMatrix::zeros(2, MtbState::DOF);
        if w < 0.0 {
            h[(0, c)] = -1.0; // d(relu(−w))/dw = −1
        }
        if w > self.travel_max {
            h[(1, c)] = 1.0; // d(relu(w−max))/dw = +1
        }
        h
    }

    fn noise(&self, _x: &MtbState) -> DMatrix<f64> {
        DMatrix::identity(2, 2) * self.sigma.powi(2)
    }

    fn active(&self, _x: &MtbState, _ctx: &SampleContext) -> bool {
        true
    }
}

/// Topout reference (design §5): on a detected topout event — free fall / fully
/// unweighted (a jump), where the wheel extends to full droop — travel is 0. A
/// tight soft pin `r = 0 − w = −w`. This is the **load-bearing travel-DC anchor**:
/// each airborne moment re-zeros the double-integrator, so drift is bounded by the
/// (short) inter-event interval rather than accumulating over the whole ride. The
/// runner gates it to detected airborne samples.
#[derive(Debug, Clone, Copy)]
pub struct TopoutReference {
    /// Target wheel.
    pub wheel: Wheel,
    /// Reference std-dev (smaller ⇒ harder zero), m.
    pub sigma: f64,
}

impl MeasurementModel<MtbState> for TopoutReference {
    fn dim(&self) -> usize {
        1
    }

    fn residual(&self, x: &MtbState) -> DVector<f64> {
        DVector::from_row_slice(&[-self.wheel.travel_of(x)])
    }

    fn jacobian(&self, _x: &MtbState) -> DMatrix<f64> {
        let mut h = DMatrix::zeros(1, MtbState::DOF);
        h[(0, self.wheel.col())] = -1.0;
        h
    }

    fn noise(&self, _x: &MtbState) -> DMatrix<f64> {
        DMatrix::from_element(1, 1, self.sigma.powi(2))
    }

    fn active(&self, _x: &MtbState, _ctx: &SampleContext) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::measurements::finite_difference_jacobian;
    use approx::assert_relative_eq;
    use nalgebra::{UnitQuaternion, Vector3};

    fn state_with_travel(d_f: f64, s_r: f64) -> MtbState {
        MtbState {
            r_chassis: UnitQuaternion::identity(),
            v_chassis: Vector3::zeros(),
            b_g0: Vector3::zeros(),
            b_a0: Vector3::zeros(),
            b_g1: Vector3::zeros(),
            b_g2: Vector3::zeros(),
            d_f,
            dd_f: 0.0,
            s_r,
            ds_r: 0.0,
            psi: 0.0,
            dpsi: 0.0,
        }
    }

    #[test]
    fn sag_residual_pulls_travel_toward_sag() {
        // Arrange — front sag 46 mm (27% of 170), current travel 60 mm.
        let p = SagPrior { wheel: Wheel::Front, sag: 0.046, sigma: 0.02 };
        let x = state_with_travel(0.060, 0.0);

        // Act
        let r = p.residual(&x);

        // Assert — r = sag − w = −0.014.
        assert_relative_eq!(r[0], -0.014, epsilon = 1e-12);
    }

    #[test]
    fn sag_jacobian_matches_finite_difference() {
        // Arrange
        let p = SagPrior { wheel: Wheel::Rear, sag: 0.043, sigma: 0.02 };
        let x = state_with_travel(0.05, 0.07);

        // Act + Assert
        assert_relative_eq!(p.jacobian(&x), finite_difference_jacobian(&p, &x), epsilon = 1e-7);
    }

    #[test]
    fn barrier_is_zero_inside_band() {
        // Arrange — travel comfortably within [0, 0.170].
        let b = TravelBarrier { wheel: Wheel::Front, travel_max: 0.170, sigma: 0.005 };
        let x = state_with_travel(0.08, 0.0);

        // Act
        let r = b.residual(&x);

        // Assert
        assert_relative_eq!(r[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(r[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn barrier_penalizes_below_zero_and_above_max() {
        // Arrange
        let b = TravelBarrier { wheel: Wheel::Front, travel_max: 0.170, sigma: 0.005 };

        // Act — 10 mm below bottom; 20 mm past full travel.
        let below = b.residual(&state_with_travel(-0.010, 0.0));
        let above = b.residual(&state_with_travel(0.190, 0.0));

        // Assert
        assert_relative_eq!(below[0], 0.010, epsilon = 1e-12);
        assert_relative_eq!(above[1], 0.020, epsilon = 1e-12);
    }

    #[test]
    fn topout_reference_pins_travel_to_zero_and_matches_finite_difference() {
        // Arrange — front travel currently 80 mm but a topout (airborne) event says 0.
        let t = TopoutReference { wheel: Wheel::Front, sigma: 0.01 };
        let x = state_with_travel(0.080, 0.0);

        // Act + Assert — residual = −w pulls travel to 0; Jacobian matches FD.
        assert_relative_eq!(t.residual(&x)[0], -0.080, epsilon = 1e-12);
        assert_relative_eq!(t.jacobian(&x), finite_difference_jacobian(&t, &x), epsilon = 1e-7);
    }

    #[test]
    fn barrier_jacobian_matches_finite_difference_outside_band() {
        // Arrange — sample below the lower bound (away from the kink at w=0).
        let b = TravelBarrier { wheel: Wheel::Front, travel_max: 0.170, sigma: 0.005 };
        let x = state_with_travel(-0.030, 0.0);

        // Act + Assert
        assert_relative_eq!(b.jacobian(&x), finite_difference_jacobian(&b, &x), epsilon = 1e-7);
    }
}
