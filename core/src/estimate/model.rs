//! Estimator-agnostic model traits — the navlie/factor blueprint that the IEKF
//! (M2) and the future batch smoother (M5) both consume **unchanged**.
//!
//! Pinned convention (load-bearing): error-state, **right (local) perturbation**,
//! `x_true = x_nom ⊞ δ`, with the SO(3) block `R_nom · exp([δθ]_×)`. Every
//! analytic Jacobian is `∂r/∂δ` at `δ = 0` under this one boxplus.

use nalgebra::{DMatrix, DVector, Vector3};

/// Driving input for one propagation step (SI units). Carries the IMU0 strapdown
/// input plus the two precomputed wheel-drive controls (the projected differential
/// specific forces, design §5) — the runner builds these per sample from the
/// unsprung IMUs + geometry so the process model stays a clean kinematic
/// integrator. `gyro0`/`accel0` are in the **chassis frame** (IMU0 mount applied).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuInput {
    /// IMU0 angular rate (chassis frame), rad/s.
    pub gyro0: Vector3<f64>,
    /// IMU0 specific force (chassis frame), m/s².
    pub accel0: Vector3<f64>,
    /// Front-wheel drive acceleration ẅ_f — projected differential specific force
    /// along the fork tangent (design §5), m/s². Control input to `{w_f, ẇ_f}`.
    pub wheel_accel_front: f64,
    /// Rear-wheel drive acceleration ẅ_r — projected along the rear axle-path
    /// tangent, m/s². Control input to `{w_r, ẇ_r}`.
    pub wheel_accel_rear: f64,
}

/// Per-sample context a measurement's validity gate may consult (e.g. ZUPT/ZARU
/// fire only on stationary-flagged samples — see [`crate::statistics::zupt_flags`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleContext {
    /// Whether this sample is flagged (quasi-)stationary.
    pub stationary: bool,
}

/// On-manifold state with boxplus/boxminus retraction. The tangent (error-state)
/// dimension is [`dof`](ErrorState::dof). SO(3) blocks use the pinned
/// right-perturbation convention `R · exp([δθ]_×)`; Euclidean blocks add.
pub trait ErrorState: Clone {
    /// Tangent-space dimension == error-state size.
    fn dof(&self) -> usize;
    /// Retraction `x ⊞ δ` (`delta` has length [`dof`](ErrorState::dof)).
    fn oplus(&self, delta: &DVector<f64>) -> Self;
    /// Local coordinates `x ⊟ other` (returns a length-`dof` tangent vector).
    fn ominus(&self, other: &Self) -> DVector<f64>;
}

/// Motion/dynamics model = the process factor. Reused verbatim by the IEKF time
/// update and (M5) the batch smoother's between-state factor.
pub trait ProcessModel<S: ErrorState> {
    /// Propagate `x` over `dt` (seconds) under the IMU0 input `u`.
    fn predict(&self, x: &S, u: &ImuInput, dt: f64) -> S;
    /// `(F, Q)` in error-state coordinates — the transition Jacobian and the
    /// discrete process-noise covariance for the step.
    fn jacobian_noise(&self, x: &S, u: &ImuInput, dt: f64) -> (DMatrix<f64>, DMatrix<f64>);
}

/// One measurement / pseudo-measurement = a data factor. The IEKF stacks
/// `(residual, jacobian, noise)` into the Kalman update; the batch smoother wraps
/// the same `residual` + `jacobian` as a factor — analytic Jacobians (never
/// autodiff) so they port between the two verbatim.
pub trait MeasurementModel<S: ErrorState> {
    /// Residual dimension.
    fn dim(&self) -> usize;
    /// `r = z ⊟ h(x)` (manifold-aware where the measurement is a rotation/direction).
    fn residual(&self, x: &S) -> DVector<f64>;
    /// Analytic `H = ∂r/∂δ` at `δ = 0` under the pinned right-perturbation boxplus.
    fn jacobian(&self, x: &S) -> DMatrix<f64>;
    /// Measurement covariance `R` (information = `R⁻¹`); may be state-gated.
    fn noise(&self, x: &S) -> DMatrix<f64>;
    /// Validity gate — defaults to always active.
    fn active(&self, _x: &S, _ctx: &SampleContext) -> bool {
        true
    }
}
