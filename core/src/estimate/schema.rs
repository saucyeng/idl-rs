//! The geometry-derived, inspectable state-vector schema (design §3). A named,
//! ordered, typed declaration of every error-state component and whether this run
//! **estimates** it (`active`) or **holds it frozen at its initial value** (a tight
//! prior, no process noise). The error-state index layout is fixed — it mirrors
//! [`crate::estimate::state::MtbState`]'s 24-DOF ordering exactly — so the same
//! covariance index space is used whether a component is active or frozen. What
//! changes per bike/stage is which components are active:
//!
//! - **Hardtail** drops the rear-unsprung gyro bias and both rear wheel states.
//! - **Wheels-first (M2a)** holds the steering states frozen (steering is M2b).
//!
//! See docs/superpowers/specs/2026-06-23-suspension-estimator-design.md §3, §4.

use crate::estimate::geometry::{BikeGeometry, Topology};

/// The manifold a state component lives on (drives boxplus/Jacobian dimension).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Manifold {
    /// Rotation (3-DOF tangent, right-perturbation `R·exp([δθ]×)`).
    SO3,
    /// Euclidean 3-vector.
    R3,
    /// Euclidean scalar.
    R1,
}

/// One named component of the estimator state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateComponent {
    /// Stable symbol (e.g. `"R_chassis"`, `"w_f"`).
    pub symbol: &'static str,
    /// Human-readable meaning.
    pub meaning: &'static str,
    /// Manifold the component lives on.
    pub manifold: Manifold,
    /// Physical unit of the component's value.
    pub unit: &'static str,
    /// Start index of this component in the error-state vector.
    pub error_index: usize,
    /// Error-state dimension (3 for SO3/R3, 1 for R1).
    pub dim: usize,
    /// Whether this run estimates the component (`true`) or holds it frozen at its
    /// initial value via a tight prior (`false`).
    pub active: bool,
}

/// The full ordered state schema for one run (always covers all 24 error indices;
/// `active` flags select the estimated subset).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateSchema {
    /// Components in error-state index order.
    pub components: Vec<StateComponent>,
}

impl StateSchema {
    /// Derives the schema from bike geometry and whether steering is estimated this
    /// run (M2a passes `false` — wheels first; M2b passes `true`). A hardtail
    /// freezes the rear-unsprung bias and rear wheel states.
    pub fn from_geometry(geometry: &BikeGeometry, estimate_steering: bool) -> StateSchema {
        let full_sus = geometry.topology == Topology::FullSuspension;
        // Fixed error-state ordering, matching MtbState (state.rs): δθ, δv, δb_g0,
        // δb_a0, δb_g1, δb_g2, δw_f, δẇ_f, δw_r, δẇ_r, δψ, δψ̇.
        let defs: [(&str, &str, Manifold, &str, bool); 12] = [
            ("R_chassis", "Chassis attitude (sensor/vehicle → nav)", Manifold::SO3, "rad", true),
            ("v_chassis", "Chassis velocity, nav frame", Manifold::R3, "m/s", true),
            ("b_g0", "IMU0 gyro bias", Manifold::R3, "rad/s", true),
            ("b_a0", "IMU0 accel bias", Manifold::R3, "m/s^2", true),
            ("b_g1", "Front-unsprung (IMU1) gyro bias", Manifold::R3, "rad/s", true),
            ("b_g2", "Rear-unsprung (IMU2) gyro bias", Manifold::R3, "rad/s", full_sus),
            ("w_f", "Front wheel travel", Manifold::R1, "m", true),
            ("dw_f", "Front wheel velocity", Manifold::R1, "m/s", true),
            ("w_r", "Rear wheel travel", Manifold::R1, "m", full_sus),
            ("dw_r", "Rear wheel velocity", Manifold::R1, "m/s", full_sus),
            ("psi", "Steer angle about the steer axis", Manifold::R1, "rad", estimate_steering),
            ("dpsi", "Steer rate", Manifold::R1, "rad/s", estimate_steering),
        ];

        let mut components = Vec::with_capacity(defs.len());
        let mut idx = 0;
        for (symbol, meaning, manifold, unit, active) in defs {
            let dim = match manifold {
                Manifold::SO3 | Manifold::R3 => 3,
                Manifold::R1 => 1,
            };
            components.push(StateComponent {
                symbol,
                meaning,
                manifold,
                unit,
                error_index: idx,
                dim,
                active,
            });
            idx += dim;
        }
        StateSchema { components }
    }

    /// Total error-state dimension of the **active** components.
    pub fn active_error_dim(&self) -> usize {
        self.components.iter().filter(|c| c.active).map(|c| c.dim).sum()
    }

    /// Looks up a component by symbol.
    pub fn component(&self, symbol: &str) -> Option<&StateComponent> {
        self.components.iter().find(|c| c.symbol == symbol)
    }

    /// Whether the named component is estimated this run.
    pub fn is_active(&self, symbol: &str) -> bool {
        self.component(symbol).map(|c| c.active).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::state::MtbState;

    #[test]
    fn full_suspension_wheels_first_active_set() {
        // Arrange — reference full-sus bike, steering off (M2a).
        let g = BikeGeometry::reference_bike();

        // Act
        let s = StateSchema::from_geometry(&g, false);

        // Assert — chassis, velocity, all four biases, and all four wheel states are
        // estimated; the two steering states are frozen.
        for sym in ["R_chassis", "v_chassis", "b_g0", "b_a0", "b_g1", "b_g2", "w_f", "dw_f", "w_r", "dw_r"] {
            assert!(s.is_active(sym), "{sym} should be active");
        }
        assert!(!s.is_active("psi"), "steering frozen in M2a");
        assert!(!s.is_active("dpsi"), "steering rate frozen in M2a");
        // 3+3+3+3+3+3 + 1+1+1+1 = 22 active DOF.
        assert_eq!(s.active_error_dim(), 22);
    }

    #[test]
    fn schema_spans_the_full_fixed_error_layout_contiguously() {
        // Arrange
        let g = BikeGeometry::reference_bike();
        let s = StateSchema::from_geometry(&g, true);

        // Act + Assert — components are contiguous from 0 and total MtbState::DOF,
        // matching the fixed error-state ordering regardless of active flags.
        let mut next = 0;
        for c in &s.components {
            assert_eq!(c.error_index, next, "component {} is not contiguous", c.symbol);
            next += c.dim;
        }
        assert_eq!(next, MtbState::DOF);
    }

    #[test]
    fn hardtail_freezes_rear_bias_and_rear_wheel_states() {
        // Arrange — a hardtail: no rear path, no IMU2.
        let mut g = BikeGeometry::reference_bike();
        g.topology = Topology::Hardtail;
        g.rear_path = None;
        g.imu2 = None;

        // Act
        let s = StateSchema::from_geometry(&g, false);

        // Assert — rear-unsprung bias and rear wheel states drop out.
        assert!(!s.is_active("b_g2"));
        assert!(!s.is_active("w_r"));
        assert!(!s.is_active("dw_r"));
        // Front states still estimated.
        assert!(s.is_active("w_f"));
        assert!(s.is_active("b_g1"));
        // 22 − (3 + 1 + 1) = 17 active DOF.
        assert_eq!(s.active_error_dim(), 17);
    }

    #[test]
    fn steering_active_when_requested_on_full_suspension() {
        // Arrange
        let g = BikeGeometry::reference_bike();

        // Act
        let s = StateSchema::from_geometry(&g, true);

        // Assert
        assert!(s.is_active("psi"));
        assert!(s.is_active("dpsi"));
        assert_eq!(s.active_error_dim(), 24);
    }
}
