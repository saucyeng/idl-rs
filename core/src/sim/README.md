# idl-rs sim: drivetrain simulation (experimental)

A 2-D model of a bike's chain, cassette, chainring, derailleur and idlers,
moving with the rear suspension. Two halves:

- **Kinematics** (`kinematics`, `path`): at each rear travel and gear, the
  chain's exact path round its wheels and rollers, so chain growth, pedal
  kickback, the cage angle that closes the loop, how much chain the cage has
  to take up or pay out, and the chain's clearance to the stays. Instant.
- **Dynamics** (`dynamics`): the whole loop link by link through a train of
  hits, with the cranks held or turning, the cassette's inertia and freehub
  ratchet, the cage's spring, clutch and mass, and the chain hitting the
  stays. What chain slap, clutch slip and kickback look like in time.
- **Suspension** (`suspension`): the shock's stroke and leverage ratio
  through travel, air springs with self-equalising negative chambers
  (isothermal for sag, adiabatic over a bump), the static load on each wheel
  from the centre of mass, the sag, and `fit`, which sizes an air spring
  (piston area, then positive volume) from measured sag.
- **Ride** (`ride`): the rear end riding over terrain, a two-mass model (the
  rear's sprung share on the shock through the leverage, the unsprung mass on
  the tyre), giving the travel and chassis motion the dynamics run on.

`calibration::geometry_table` reads a maker's geometry table from OCR words
with their boxes (sizes across, rows down; inches and settings like
"435 / 445" handled) into the `geometry.*` parameters.

The physics layer is `core` (bike physics, no app). The app's Simulate tab
(idl1-app, C3 §3.15) draws both over the bike's photo; a dynamic run reaches
it as a sim result (`IDLM` v1, encoded in idl1-bridge).

Specs (idl1-app): `docs/superpowers/specs/2026-11-19-idl1-drivetrain-sim-DRAFT.md`
(this simulator; §13-15 record why the contact model is what it is) and
`2026-11-19-idl1-whole-bike-sim-loop-DRAFT.md` (where it is going: the chain
as one component of a whole-bike simulator).

## Quick start

There are no CLI verbs yet (`idl-rs sim kinematics` / `sim run` are planned).
From Rust:

```rust
use idl_rs::sim::{bike::SimBike, kinematics::gear_curves, dynamics::{simulate, DynConfig}, excite::BumpTrain};

let bike = SimBike::new(&linkage, &drivetrain)?;           // the bike file's two sections
let curves = gear_curves(&bike, 5, 17, &|_| true)?;        // gear 6, 17 travels, every roller fitted
let run = simulate(&bike, 5, &|_| true, &BumpTrain::default().excitation(1000.0),
                   &DynConfig::default(), &mut |_progress| true)?;
```

`core/examples/lower_idler_search.rs` is a worked use of the kinematics: a
grid search for where a lower idler keeps the clutch from working through
travel (`cargo run -p idl-rs --example lower_idler_search -- out.csv 10`).

Inputs are the bike file's `linkage` (the Horst 4-bar, clicked on the photo)
and `drivetrain` (BB, chainring, cassette, chain, derailleur points and
parameters, rollers, contact surfaces) sections. `Drivetrain::typical` is a
placeholder 12-speed 1x until the real points are measured.

## What comes out

- `GearCurves` per gear, one value per travel sample: growth (mm), kickback
  (deg of crank), cage angle (deg forward), loop change (mm the cage must
  take up), top span length, clearance to each surface, whether each roller
  touches the chain. `chain_fit`: the chain lengths that close in every gear.
- `DynResult`: every link's position per frame, the moving points (linkage,
  cage pivot, wheels, surfaces), each wheel's angle, and 15 channels (rear
  travel, cage angle, clutch slipping and total slip, top and lower span
  tension, pedal force, span deflections, links on surfaces, impact speed and
  count, links on the chainring, freehub engaged, link error), plus totals.

## How far to trust it

In order of how much each matters:

1. **The numbers are placeholders until the bike is measured.** The
   derailleur's points, spring curve, clutch torque (1.2 N m) and inertias,
   the BB position and the chain's mass come from `Drivetrain::typical`.
   Shapes and comparisons (with and without an idler, gear to gear) mean
   something now; absolute tensions and slip angles do not.
2. **The suspension drives the chain; the chain does not push back.**
   Travel and frame motion are prescribed (`Excitation`). Chain tension,
   squat and the clutch's force on the swingarm are computed but never act
   on the suspension, so "how much does the clutch stiffen the rear" is not
   answerable yet. Fixing this is D8 of the whole-bike spec.
3. **The input is a synthetic bump train**, not a recorded ride. Replay of a
   ride's IMU0 motion, estimated rear travel and GPS speed is planned (spec §5).
   The train defines rear travel over time, not a ground shape;
   `BumpTrain::ground_height` gives the ground that alone would produce it
   (rigid tyre, level frame), which the app draws scrolling under the wheel.
4. **The clutch is a plain one-way brake after its free play.** It holds the
   cage forward with a constant torque once the play
   (`clutch_free_play_deg`) is taken up, and lets it return freely; real
   clutches vary with speed and wear, and the spring curve is two points.
5. **A single run's peaks are chaotic.** The chain is a chaotic system: a
   1 % change in a bump moves the worst impact a lot. Compare setups by
   statistics over many hits (counts, distributions, totals), never by one
   run's maximum (spec §14).
6. **2-D.** No chain line, no lateral flex, no cross-chaining.

## How it works

- **Kinematics.** Wheels are circles at their pitch radius (toothed) or
  roller radius (smooth). The chain's path is tangent lines and wraps round
  them, in order; a roller is in the path when the chain would otherwise
  cross it (a guide) or always (a threaded idler). The cage angle that
  closes a loop of the chain's length is found by scanning -40 to +100 deg
  from the drawn angle and bisecting the root nearest it.
- **Dynamics.** One point mass per link, the links held at the pitch by
  XPBD distance constraints (stiffness 1e7 N/m, 2 iterations per step), a
  50 us step (`DynConfig::default`; about 5.5x real time in release), frames
  recorded at 500 Hz. Everything is in the front triangle's frame, so the
  frame's measured acceleration and rotation enter as inertial forces.
- **Chain on wheels is a force, not a seating rule.** A link whose pin is
  inside a wheel's pitch circle feels a stiff damped normal force
  (2e5 N/m, 20 N s/m); a toothed wheel holds the link where it touched down
  with no slip until it lifts out; the reaction turns the wheel, and through
  a jockey's axle, the cage. So links leave at the tangent on their own.
- **Each wheel's spin is stepped implicitly in the teeth holding it.** A
  light wheel on a large radius held by several stiff teeth (a 16T idler at
  2e-6 kg m^2) overshoots an explicit step and spins up from rest; the step
  is linearised backward Euler in the held teeth's rotational stiffness and
  damping (sum of k d^2, c d^2).
- **The stays** are polylines on a body, with the same contact and rubber's
  friction (0.5); only links on the surface's side and within 6 mm feel them.

## Key decisions

- Contact as forces (spec §13): the first model seated links by rule and
  carried the chain past the tangent; the force model passes a bench test
  where a taut, driven chain holds the exact tangent path's wrap within one
  link on every wheel.
- Teeth hold without slip (spec §15): a capped grip let a slack chain creep
  73 deg round held cranks.
- Rollers are threaded idlers or guides (spec §14), so a main-pivot idler is
  always in the path and an ISCG roller only where the chain reaches it.

## Tests

`cargo test -p idl-rs --lib sim::`. Among them: a still bike keeps its chain
on at the pitch; the lower span's tension balances the cage spring; a bump
train swings the cage; a driven taut chain leaves every wheel at the
tangent; held cranks let a link move round the ring only by lifting over the
teeth; with a top idler the chain starts outside every wheel, no link is ever
inside a wheel in any gear, and the still idler stays still; a threaded idler
on the main pivot keeps growth smooth through travel.

## Next steps

- Clutch free play, and the measured drivetrain (spec §11).
- Replay a recorded ride; `idl-rs sim` CLI verbs writing a sim result.
- Two-way coupling with a suspension model (whole-bike spec phase 1), so the
  clutch's effect on the rear and an idler's benefit can be predicted.
