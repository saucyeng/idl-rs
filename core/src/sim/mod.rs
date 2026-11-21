//! Drivetrain simulator: chain growth through travel (kinematics) and the
//! chain loop's dynamics over bumps (chain slap, clutch slip, kickback).
//! Spec: idl1-app `docs/superpowers/specs/2026-11-19-idl1-drivetrain-sim-DRAFT.md`.
//!
//! Everything is in the bike side plane seen from the drive side (origin
//! rear axle, x toward the front axle, y up, metres, angles CCW in radians),
//! in the front triangle's frame. The rear end moves through the Horst 4-bar
//! of [`crate::calibration::linkage`]; the drivetrain is
//! [`crate::calibration::drivetrain::Drivetrain`].
//!
//! - [`bike`]: the linkage and drivetrain put together; where every wheel is
//!   at a travel, gear and cage angle.
//! - [`path`]: the chain's path around those wheels (tangents and wraps),
//!   which rollers it touches, and the cage angle that closes the loop.
//! - [`kinematics`]: growth, kickback, cage angle and clearance through travel.
//! - [`excite`]: what drives a dynamic run (synthetic bumps).
//! - [`suspension`]: shock stroke and leverage, air springs, static loads, sag, and the springs fitted to measured sag.
//! - [`replay`]: a recorded ride (IMU0 in the bike's side plane, the physics estimator's rear travel, GPS speed) as the excitation.
//! - [`ride`]: the rear end riding over terrain (springs, damper, tyre), giving the travel the chain's dynamics run on.
//! - [`dynamics`]: the chain as one point mass per link (XPBD), with the
//!   cassette, jockeys, rollers and cage as rotating bodies, the freehub
//!   ratchet, the clutch, and contact with the stays.

pub mod bike;
pub mod dynamics;
pub mod excite;
pub mod kinematics;
pub mod path;
pub mod replay;
pub mod ride;
pub mod suspension;

/// What went wrong in the simulator.
#[derive(Debug, Clone, PartialEq)]
pub enum SimErrorKind {
    /// The bike has no linkage, or it cannot be used (it locks before full travel).
    Linkage,
    /// The drivetrain's numbers don't describe a drivetrain (no cassette, a gear out of range).
    Drivetrain,
    /// The chain cannot go round the wheels in this gear and travel: two
    /// wheels overlap, or no cage angle closes the loop (chain too short or long).
    NoPath,
    /// The excitation is empty or not on increasing time.
    Excitation,
}

/// A simulator failure, with a sentence for the user.
#[derive(Debug, Clone, PartialEq)]
pub struct SimError {
    /// What kind of failure.
    pub kind: SimErrorKind,
    /// What happened, in words.
    pub message: String,
}

impl SimError {
    /// A failure of `kind` saying `message`.
    pub fn new(kind: SimErrorKind, message: impl Into<String>) -> SimError {
        SimError { kind, message: message.into() }
    }
}

impl std::fmt::Display for SimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SimError {}
