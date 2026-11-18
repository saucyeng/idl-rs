# The bike-state estimator

This folder turns a ride's raw sensor streams into what the bike was doing:
how far it leaned, where it pointed, where it was going, which way the bars
were turned, and how fast each wheel's suspension moved. Workbooks read the
results through `attitude(...)`, `body_accel(...)`, `wheel_travel(...)` and
`wheel_velocity(...)`.

This file says what each output is, how far you can trust it, what a bike
calibration is and how to make one, and what is still missing. The design
notes live in the code and in the commit messages on the
`estimator-align-chassis` branch; the long message on `b199bc7` is the summary.

## What goes in

- **IMU0** on the frame (required), **IMU1** on the fork lower, **IMU2** on the
  swingarm (absent on a hardtail).
- **GPS** speed, course and altitude (optional, but without it there is no
  heading, course, slide angle or understeer angle).
- The bike's **calibration**, if it has one (see below).

The three IMUs run on their own clocks, so every channel is resampled onto
IMU0's gyro timeline by its recorded time before anything else happens. The
outputs share that timeline. **It starts at IMU0's first sample, which is not
zero on every ride**, so line outputs up with other channels by time, not by
sample index.

GPS fixes arrive late: the receiver reports where the bike was a moment ago.
The estimator measures how late on each ride by lining the GPS course changes
up with the gyro's turning (`fit_gps_latency` in `run.rs`). It found 0.39 s on
the GT Force's 3 Hz receiver and 0.61 s on the moto's 1 Hz one. If a ride has
too little turning to measure it, 0.2 s is assumed.

## How it works, in one paragraph

An error-state Kalman filter (the type is called `Iekf`, for *iterated*; it
runs one iteration by default, which makes it a plain EKF) integrates IMU0's
gyro and accelerometer to track the frame's attitude and velocity, and
corrects them with GPS velocity, with "the bike is stopped" updates when it is
stationary, and with gravity when GPS is not available. The heading is set
from the GPS course **once**, the first time the bike moves faster than
3 m/s, and after that it is the gyro's: it is never pulled back onto the GPS
course, so a slide (frame pointing one way, bike going another) stays visible.
The fork and swingarm IMUs, compared with the frame's, drive the suspension
and the steering. A stationary gyro-bias correction is skipped when it would
need the bias to jump further than the filter thinks it can (`gated_zaru`);
without that, a slow turn while nearly stopped was learned as gyro drift and
ran the heading away.

## The outputs and how far to trust them

The numbers below were measured on two rides only, against GPS, and are what
"good" looked like there. They are not a specification.

- **Beacon, ride `173461a2`**: GT Force, calibrated, 3 Hz GPS.
- **Walden, ride `4f3f3032`**: moto, no calibration, 1 Hz GPS.

| Output | Workbook | What it is | Trust |
|---|---|---|---|
| Roll | `attitude("roll")` | Lean about the bike's own forward axis, degrees, positive = leaning right. | **Good.** Against the lean a steady turn implies from GPS speed and turn rate: slope 0.89, correlation 0.93 (Beacon); slope 0.97, correlation 0.97 (Walden). |
| Pitch | `attitude("pitch")` | Nose up, degrees. | Same machinery as roll. Not separately checked. |
| Heading | `attitude("heading")`, `attitude("heading_sigma")` | Where the frame points, degrees clockwise from north. Blank until the bike first moves. | **Usable at 3 Hz GPS.** Heading minus GPS course above 4 m/s: median −0.4°, RMS 5.8° (Beacon). **Rough at 1 Hz and uncalibrated**: median −1.8°, RMS 13.2° (Walden). Some of that RMS is real slide, not error. |
| Course | `attitude("course")` | Where the rear wheel is travelling, degrees clockwise from north, from the estimator's own velocity. Blank below 3 m/s. | Follows the GPS between fixes; as good as the GPS rate allows. |
| Slide angle | `attitude("slip")`, `attitude("slip_sigma")` | Heading minus course at the rear wheel, degrees, positive = nose to the right of travel. Blank below 3 m/s. | **Use with care.** It is the difference of two angles that are each good to a few degrees, so a slide of a few degrees is at the noise level. `slip_sigma` is reported **far too small**: do not use it as an error bar. **Unusable on 1 Hz GPS.** |
| Steer | `attitude("steer")`, `attitude("steer_sigma")` | Bar angle about the steering axis, degrees, positive = bars left. **Needs a calibration with a steering axis**; blank otherwise (so blank on the moto today). | **Good for shape and timing.** Against the bar angle a non-sliding bike needs: correlation 0.86, slope 0.74, about 3° scatter (Beacon). The zero is set from one ride's straight-line riding. |
| Steer needed | `attitude("steer_needed")` | The bar angle a bike that does not slide needs for its speed, turn rate and lean, degrees. **Needs the wheelbase too.** | **Relative only.** The formula is approximate (see Known gaps). |
| Understeer | `attitude("understeer")` | Steer minus steer needed, on the ground and toward the turn, degrees: positive = understeer (front pushing), negative = oversteer (rear stepping out). Blank going straight. | **Relative only.** Median −1.6° on Beacon, but the zero point is not calibrated: compare corners, laps and setups with each other, not with zero. |
| Body acceleration | `body_accel("long")`, `body_accel("lat")` | Acceleration with gravity taken out, in the bike's own axes, g. Long positive = speeding up; lat positive = to the right. | **Good for braking and cornering g.** Depends on roll and pitch being right: one degree of attitude error leaks about 0.017 g of gravity in. |
| Wheel travel | `wheel_travel("front"\|"rear")` | Suspension travel, mm. | **Not meaningful as a position.** Do not read sag, bottom-outs or travel used from it. |
| Wheel velocity | `wheel_velocity("front"\|"rear")` | Suspension velocity, mm/s, positive = compressing. | **Plausible.** Use it for velocity histograms and spectra (FFT), and for comparing setups. |

The `_sigma` outputs are the filter's own opinion of its uncertainty. For
heading and steer they are a reasonable guide; for slide angle they are not.

## Bike calibration

### What it is

A calibration says where the IMUs sit on a particular bike and which way that
bike points. Without one, the estimator uses a built-in reference bike and
fits IMU0's mount from each ride's straight-line riding, which costs a couple
of degrees and gives no steering.

A calibration holds, all in IMU0's own axes:

- **up**: which way is up in IMU0 with the bike still and level;
- each pod's **rotation and lever arm** relative to IMU0 (fork pod and
  swingarm pod);
- the **steering axis**;
- **forward**: from the steering axis (it leans back in the bike's centre
  plane), or, without one, from the line to the fork pod;
- optionally the **steering zero** and a **forward trim**, both set from a ride;
- optionally the **wheelbase**, typed in.

It does **not** hold suspension travel, sag or the rear axle path: those stay
the reference bike's.

It is stored in the bike's file (`bikes/<bike_id>.json` in the library), with
the sessions and windows it was fitted from. A ride uses the newest
calibration captured at or before it, or, if there is none, the oldest one
captured after it (a calibration recorded at the end of a day still describes
that day's bike). `idl-rs bike show <bike_id>` lists them.

### What a calibration changes

| The bike has... | IMU0 mount | Forward | Steering |
|---|---|---|---|
| No calibration | Fitted from the ride | Trimmed per ride so straight riding reads zero slide | Blank |
| A still-only calibration (IMU0 alone) | Calibration's tilt, ride's forward | Trimmed per ride | Blank |
| A full calibration | Calibration's | Calibration's, the same on every ride (the per-ride trim is still measured and reported as a check) | Estimated |

Using the same forward on every ride matters for comparing rides: what then
differs between two rides is the riding, not the frame they were read in.

### How to make one

Record a calibration session with the logger on the bike, then name the
windows in it. Nothing is segmented automatically: you give the times
(milliseconds from the first sample every IMU has).

1. **Still and level**: the bike upright on level ground, nobody touching it.
   A few seconds is enough. This gives up and the gyro offsets. A hand on the
   bars is enough movement to fail it.
2. **Rigid**: turn the bike about by hand as one body (tip it, pitch it, yaw
   it), with the bars held still and the suspension not moving. This gives the
   swingarm pod's rotation and lever arm, and the fork pod's too.
3. **Fork** (optional): a part of the rigid window where the bars certainly
   did not turn. The fork pod is fitted over it; by default it uses the whole
   rigid window.
4. **Steer**: the frame held still while the bars are swung lock to lock a few
   times. This gives the steering axis, and from it forward.

Then fit and save:

```
idl-rs bike calibrate gt-force --still <session> --still-from-ms 1000 --still-to-ms 6000 \
    --rigid-from-ms <ms> --rigid-to-ms <ms> \
    --fork-from-ms <ms> --fork-to-ms <ms> \
    --steer-from-ms <ms> --steer-to-ms <ms>
```

The still window and the hand-protocol windows can come from different
recordings: `--protocol <session>` names the second one. `--dry-run` shows
what would be saved; `--note` keeps a remark with it; `--delete <id>` removes
one.

Two more steps finish it, each changing the newest calibration rather than
fitting a new one:

- **`--forward-from <ride>`**: runs the estimator over a normal ride and turns
  the calibration's forward and steering zero so that the ride's straight-line
  riding reads zero slide and zero steer. Use a ride with plenty of straight
  riding above 15 km/h; with less than 30 s of it the command refuses. Each
  run turns the calibration further by what that ride still reads, so
  running it again on the same ride should report a turn near zero. The hand protocol
  cannot know whether you held the bars dead straight in the rigid window or
  exactly where the bike's centre plane is; a ride does.
- **`--wheelbase-mm <mm>`**: the wheelbase with the suspension topped out,
  measured with a tape. The understeer outputs need it; no recording measures
  it.

The GT Force has a full calibration with forward, steering zero and wheelbase.
The moto has none yet.

## What the outputs are good for

- **Lean angle** through corners, and comparing it between laps, lines and
  riders.
- **Braking and cornering g** (`body_accel`), and how they combine.
- **Steering**: when and how much the rider steers, countersteer at corner
  entry, how quickly the bars settle, comparing it between laps or setups.
- **Understeer and oversteer as changes**: did a setup change make the front
  push more in the same corners? Compare the same corners across rides, never
  against zero.
- **Slides** on a 3 Hz GPS ride: a slide big enough to stand well clear of a
  few degrees.
- **Suspension velocity**: histograms, spectra and setup comparisons from
  `wheel_velocity`.

Not supported: absolute suspension travel or sag, an absolute understeer
figure, slide angle on 1 Hz GPS, anything on the moto that needs steering.

## Known gaps

These are known and not being worked on yet.

- **Wheel travel as a position is not meaningful.** Travel is integrated from
  the difference between the frame's and the wheel's accelerations, which
  drifts; the only anchors are the physical end stops and "topped out while
  airborne". Velocity comes out plausible, position does not. The geometry
  (travel, sag, rear axle path) is also the reference bike's, not measured.
- **Slide-angle uncertainty is reported far too small**, and slide angle is
  unusable on 1 Hz GPS.
- **The understeer zero point is not calibrated.** The formula in
  `kinematic_steer` (`run.rs`) for the bar angle a non-sliding bike needs was
  written from memory and leaves out trail, tyre width, and the wheelbase
  getting shorter as the fork compresses. Differences between rides and
  corners mean something; the absolute value does not.
- **The LSM6DSV320X high-g channel can be off by up to ±1.5 g** (the
  datasheet's typical zero-g offset for the high-g range, Table 3 `LA_TyOff`),
  and nothing corrects it. That channel is merged into IMU1's acceleration on
  newer hardware.
- **Moto steering and swingarm angle** wait on a calibration recording that
  has not been made yet.
- The filter's `_sigma` outputs are its own bookkeeping, not checked
  uncertainties, except where the table above says so.

## Checking a change for regressions

`core/examples/estimator_regression.rs` runs the estimator over one library
session, the way a workbook would, and prints the numbers in the trust table
above. Run it before and after a change to the estimator:

```
cargo run -q --release -p idl-rs --example estimator_regression -- \
    <library>/sessions/173461a2b9cddbf2d35a0c020099547f --bike <library>/bikes/gt-force.json
cargo run -q --release -p idl-rs --example estimator_regression -- \
    <library>/sessions/4f3f3032c3fb672444108c5212063f00
```

Beacon takes about two minutes, Walden about nine. What each number means,
and exactly how it is measured, is at the top of the example. The reference
values:

| | Beacon `173461a2` (GT Force, calibrated) | Walden `4f3f3032` (moto, uncalibrated) |
|---|---|---|
| GPS lateness | 0.39 s | 0.61 s |
| Forward trim | 0.0° (measured, not applied) | 12.2° (applied) |
| Roll vs GPS lean | slope 0.89, corr 0.93 | slope 0.97, corr 0.97 |
| Heading − course | median −0.4°, RMS 5.8° | median −1.8°, RMS 13.2° |
| Steer vs needed | slope 0.74, corr 0.86, scatter 3.1° | — |
| Understeer median | −1.6° | — |

## The physics estimator (`physics/`)

A second estimator, ported from the PyTorch research code in `ml/` (whose
README has the method, the results and how far to trust them). Instead of
integrating accelerations forward in time, it fits a model of the bike to a
whole window at once: given the frame pod's measurement and guessed latents
(front and rear travel, steer, fork bending, rear and frame-pod flex), the
model predicts what the fork and rear pods should read, and a Gauss-Newton
solve picks the latents that make the prediction match. Travel is seen
through the pods' lever arms (a pod's position moves with travel, and the
frame's rotation turns that into acceleration), so it does not drift and
stays inside its end stops. Fast motion (hits, bending, flex) is good; slow
front travel (sag) is the weak part until the pods' accelerometers are
calibrated against each other.

It needs a calibrated bike with both pods, the bike's travel, and for a
4-bar rear its linkage from the photo tool (`ml/tools/linkage.html`):

```
idl-rs bike set gt-force --linkage gt-force-linkage.json --shock-mm 230 --front-travel-mm 170
idl-rs session estimate-physics <session id>
```

The first run fits the geometry a calibration cannot give (steering-axis
point, fork bending length, the linkage photo's scale) from four windows of
that session and saves it in the calibration (`physics`); later runs reuse
it (`--refit` to fit again). Every stretch of riding (GPS faster than 3 m/s)
is then solved in 12 s windows overlapping by 2 s, and the `Physics ...`
channels are written to `sessions/<id>/derived/<hash>.parquet` (C1 §5,
`derived_kind` `physics`). The app does not read them yet.

| File | What |
|---|---|
| `physics/bike.rs` | The pose functions: steer, travel, bending, 4-bar or generic rear, frame-pod flex. |
| `physics/readings.rs` | What each pod reads, and its exact per-sample Jacobian (hyper-dual numbers). |
| `physics/objective.rs` | The residuals: pods, end stops, flex and bending priors, airtime anchor. |
| `physics/solve.rs` | Banded Gauss-Newton over one window or several sharing globals. |
| `physics/pipeline.rs` | A session into windows: native-rate clip masks, airborne detection, still biases. |
| `physics/run.rs` | The bike fit, the whole-ride solve, the output channels. |
| `physics/setup.rs`, `physics/fourbar.rs` | The bike from a calibration, its linkage and its physics fit. |

Golden tests hold the forward model, its Jacobian and a real window's solve
equal to the PyTorch reference (`ml/experiments/export_golden.py`,
`export_window.py` write the fixtures); regenerate them after a model change
there.

## Where things are

| File | What |
|---|---|
| `run.rs` | The entry point (`run`), the input resampling, GPS lateness, the forward trim, the understeer formula, and the main loop. |
| `iekf.rs` | The Kalman filter's predict and update steps. |
| `process.rs` | How the state moves between samples. |
| `measurements/` | Each correction: GPS, gravity, stopped, suspension limits, steering. |
| `orient.rs` | IMU0's mount from riding, and the starting attitude. |
| `smooth.rs` | The backward smoothing pass over the wheel travel. |
| `geometry.rs` | The reference bike. |
| `../calibration/bike.rs` | The bike calibration and how it is applied. |
| `../commands/calibration_ops.rs` | Fitting a calibration from the recordings. |
