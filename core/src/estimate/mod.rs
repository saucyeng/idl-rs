//! Offline geometry-constrained suspension-kinematics estimator (M1: foundations).
//!
//! Pure core — data in, data out, no flutter_rust_bridge, no I/O. Separate from
//! the stateless `math` expression evaluator: the IEKF (M2) and a future batch
//! smoother (M5) are recursive/stateful/joint and compose the manifold primitives
//! (`crate::rotation`) plus the model traits and state defined here. User-facing
//! outputs surface as derived channels (virtual sensors).

pub mod detect;
pub mod geometry;
pub mod iekf;
pub mod ledger;
pub mod measurements;
pub mod model;
pub mod noise;
pub mod orient;
pub mod process;
pub mod run;
pub mod schema;
pub mod smooth;
pub mod state;
