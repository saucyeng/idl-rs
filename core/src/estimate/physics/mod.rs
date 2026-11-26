//! Physics-model estimator (port in progress of `ml/`, the PyTorch reference).
//!
//! A differentiable model of the bike predicts what the fork and rear pods
//! should read, given the frame pod's measurement and latent states (travel,
//! steer, fork bending, rear and frame-pod flex); a Gauss-Newton solve picks
//! the latents that make the prediction match the recording. Travel is seen
//! through the lever arm (pod position moves with travel), so it is never
//! double-integrated. See `ml/README.md` for the method, its results and how
//! far to trust it.
//!
//! Status: the forward model and its per-sample Jacobian ([`readings`]), the
//! objective ([`objective`]) and the window solver ([`solve`]), each held
//! equal to the PyTorch reference by a golden test. Not ported yet: the bike
//! fit (geometry from several windows), the session pipeline (cache,
//! resampling, clip and airtime masks) and multi-threading.
//!
//! Everything is generic over [`la::Real`]: `f64` for values, num-dual's
//! hyper-dual numbers for exact time derivatives, and dual numbers on top for
//! Jacobians. The physics is written once; a solver and (later) training a
//! network both use the same residuals-and-Jacobian function.

pub mod bike;
#[cfg(test)]
mod fixture;
pub mod forces;
pub mod fourbar;
pub mod la;
pub mod nav;
pub mod objective;
pub mod pipeline;
pub mod readings;
pub mod run;
pub mod setup;
pub mod so3spline;
pub mod solve;
