//! Measurement / pseudo-measurement factors (design §5). Each implements
//! [`crate::estimate::model::MeasurementModel`] for [`crate::estimate::state::MtbState`].
//!
//! **Frozen conventions (load-bearing, shared by the IEKF and the future batch
//! smoother):**
//! - `residual(x) = z ⊟ h(x)` in the measurement's own (Euclidean, for M2a) space.
//! - `jacobian(x) = ∂residual/∂δ` at `δ = 0`, shape `dim × MtbState::DOF`, under the
//!   pinned right-perturbation boxplus. **Analytic, never autodiff.**
//! - The Gauss-Newton / EKF step (`iekf.rs`) consumes these as
//!   `δ* = −(P⁻¹ + ΣHᵀR⁻¹H)⁻¹ ΣHᵀR⁻¹ r` — the same factors a batch solve stacks.
//!
//! Every factor is **soft** (finite `R`); there are no hard equality constraints.
//!
//! M2a ships: [`zupt`] (ZUPT + ZARU), [`gravity`] (accel-compensated leveling),
//! [`gps`] (velocity anchor), [`prior`] (sag + travel barrier). The off-axis
//! diff-accel pin and the unsprung-link gyro-rate factor are noted in the design
//! §5 and land alongside the IEKF tuning (off-axis carries ~no information in the
//! static first-light case and is sequenced with the dynamic synthetic harness).

pub mod gps;
pub mod gravity;
pub mod prior;
pub mod zupt;

/// Error-state column offsets — the fixed [`crate::estimate::state::MtbState`]
/// 24-DOF ordering, shared by every factor's Jacobian.
pub(crate) const I_THETA: usize = 0;
pub(crate) const I_V: usize = 3;
pub(crate) const I_BG0: usize = 6;
pub(crate) const I_BA0: usize = 9;
pub(crate) const I_BG1: usize = 12;
pub(crate) const I_BG2: usize = 15;
pub(crate) const I_WF: usize = 18;
pub(crate) const I_DWF: usize = 19;
pub(crate) const I_WR: usize = 20;
pub(crate) const I_DWR: usize = 21;
// Forward-looking entries used by the deferred steering factors (M2b); kept here so
// the column map is complete and authoritative in one place.
#[allow(dead_code)]
pub(crate) const I_PSI: usize = 22;
#[allow(dead_code)]
pub(crate) const I_DPSI: usize = 23;

/// Central-difference Jacobian of a factor's residual w.r.t. the error state, used
/// by every factor's test to confirm the analytic `jacobian` against the pinned
/// boxplus. Shape `dim × MtbState::DOF`.
#[cfg(test)]
pub(crate) fn finite_difference_jacobian<M>(m: &M, x: &crate::estimate::state::MtbState) -> nalgebra::DMatrix<f64>
where
    M: crate::estimate::model::MeasurementModel<crate::estimate::state::MtbState>,
{
    use crate::estimate::model::ErrorState;
    use crate::estimate::state::MtbState;
    use nalgebra::{DMatrix, DVector};

    let n = MtbState::DOF;
    let dim = m.dim();
    let eps = 1e-6;
    let mut j = DMatrix::zeros(dim, n);
    for col in 0..n {
        let mut ep = DVector::zeros(n);
        ep[col] = eps;
        let mut em = DVector::zeros(n);
        em[col] = -eps;
        let d = (m.residual(&x.oplus(&ep)) - m.residual(&x.oplus(&em))) / (2.0 * eps);
        for row in 0..dim {
            j[(row, col)] = d[row];
        }
    }
    j
}
