# idl-ml: physics-model bike-state estimation (experimental)

A self-contained PyTorch project that estimates suspension travel, steer,
fork bending and frame/rear flex from the three IMUs, by fitting a
differentiable model of the bike to what the pods measured. It reads the
idl1 library's `data.parquet` files directly and touches nothing in the
Rust crates. Deleting this folder removes it completely.

**Status (2026-11-19): a working research tool, not a product.** It runs on
one bike (GT Force, calibrated) and has been checked on one ride (Beacon,
session `173461a2`). Travel does not drift (it is never double-integrated),
stays inside its end stops, and reads ~1 mm in the air with the airtime
anchor. It is good at fast motion (hits, bending, flex) and only fair at
slow travel (sag, slow pitch): see "How far to trust it".

This folder is now the **reference**; the engine runs a Rust port of it
(`core/src/estimate/physics/`, documented in `core/src/estimate/README.md`
§"The physics estimator"), held equal to it by golden tests. The port runs
whole sessions from the CLI (`idl-rs session estimate-physics`) and stores
its channels in the library; the app does not show them yet. Networks and
the dynamics layer (loads, tyres, rider) are not started: the dynamics layer
is planned in idl1-app's whole-bike sim loop spec (§5.2).

### The structure, in plain words

Despite the folder name, **nothing here is trained yet**: there is no neural
network. What runs today is *fitting*, closer to a least-squares curve fit
than to machine learning. PyTorch is used because it computes exact
derivatives of any formula automatically and runs on the GPU. The pieces:

1. **A physics model of the bike** (`idlml/bike/`). Give it a guess of the
   unknowns over time (front and rear travel, steer, fork bending, flex), plus
   what the frame pod measured, and it predicts what the fork and rear pods
   *should* have measured. It's pure geometry: where each pod sits for a given
   travel and steer, and how it must then move.
2. **A score** (`idlml/factors/objective.py`). It compares those predictions
   with what the pods actually recorded, and adds penalties for implausible
   guesses: travel past the end stops, travel that isn't near 0 in the air,
   bending that is too large. A lower score means a better explanation of the
   data. In ML terms this is the "loss".
3. **A solver** (`idlml/solve/gn.py`). It adjusts the guess to lower the
   score until it can't improve. It uses Gauss-Newton: at each step it
   approximates the score as a bowl and jumps to the bowl's bottom, so it
   needs only a handful of steps. Each step solves one big but sparse linear
   system, because each moment of the ride only touches the unknowns near that
   moment. That sparsity is what keeps it fast enough to run.
4. **Two stages per bike.** First, the bike's own unknowns that the
   calibration can't give (steer-axis point, fork bending length, linkage
   scale) are fitted once from a few busy windows of a ride, and saved
   (`bikes/gt-force-fit.json`). Then each stretch of riding is solved with
   that bike held fixed.

Where ML comes in later: a network would be trained to produce the solver's
answer directly from the raw IMU data, in milliseconds instead of minutes,
so it could run on a phone. It would be trained with the same score, so the
physics is written once.

Contents: [Quick start](#quick-start) · [What comes out](#what-comes-out) ·
[How far to trust it](#how-far-to-trust-it) · [Results so far](#results-so-far-gt-force-beacon-173461a2) ·
[How it works](#how-it-works) · [Key decisions](#key-decisions) ·
[Layout](#layout) · [Next steps](#next-steps-and-the-road-to-idl1-app)

## Quick start

Environment: Python 3.14, an NVIDIA GPU with CUDA (developed on a 4 GB GTX
1650; float64 solves, so a consumer card's slow fp64 sets the speed), and

```
pip install -r requirements.txt     # torch with CUDA: install from pytorch.org for your CUDA first
```

The library location comes from `IDL_LIBRARY` (default
`C:\Users\isaac\Documents\idl1-library\data`). All commands run from `ml/`.

```
python -m pytest -q tests                          # 11 tests, ~1 min: kinematics vs brute force, solver vs reference
python experiments/recover.py                      # synthetic ride with known answers: what the estimator recovers
python experiments/ride_fit.py --air               # real ride: estimate the target window with the saved bike fit
python experiments/ride_fit.py --air --refit       # ... and first re-learn the bike geometry (~15 min)
python experiments/travel_uncertainty.py --air     # how well each travel timescale is determined
```

`ride_fit.py` options: `--session` (id prefix, default `173461a2`),
`--start/--end` (session seconds, default Lower Broken Bomber 5784.3-5834.4),
`--air` (airtime anchor: **use it**), `--refit` (re-learn the bike),
`--cal` (with `--refit`: also fit pod mount rotations and accelerometer
scales; **does not work, do not use**: see limit 2), `--fit-file` (which bike fit to read/write),
`--flex-scale` (frame-flex prior, diagnostic), `--tag` (output name).

What happens on a first run:
1. **Cache.** The session's parquet is streamed once into `cache/<id>_400/`
   (each IMU anti-alias filtered on its own recorded clock, then all three
   read on one 400 Hz grid; memory-mapped `.npy` files). A 100-minute ride
   takes a few minutes and ~200 MB.
2. **Bike fit** (only with `--refit` or when no fit file exists): the
   geometry the calibration cannot give (steer point, fork bending length,
   4-bar photo scale) is solved jointly over four 30 s windows (two with the
   most airtime, two with the most frame rotation; the target is held out)
   and saved to `bikes/gt-force-fit.json`. ~14 min, 20 iterations.
3. **Target solve**: geometry frozen, ~80 s for 50 s of ride (7 iterations).

Timing on the GTX 1650: about 1.6 s of compute per second of ride. Here a
whole ride is not wired up (windows are solved one at a time by hand); for a
whole session, or a stretch of one, use the Rust port:
`idl-rs session estimate-physics <id> [--from-ms N --to-ms N] [--refit]`.

## What comes out

Each `ride_fit.py` run prints a report to stdout and saves
`runs/real/ride_fit_<tag>.npz`:

| key | shape | what |
|---|---|---|
| `t` | [T] | session seconds (400 Hz) |
| `q` | [T, 13] | latents, order `idlml.bike.model.LATENTS`: `d` front travel (m, along the steer axis), `psi` steer (rad), `b_fa`/`b_lat` fork bending fore-aft/lateral (rad), `s` rear wheel travel (m), `r_roll`/`r_yaw` rear-member flex (rad), `f_rx..f_rz` frame-pod flex rotation (rad), `f_tx..f_tz` frame-pod flex translation (m) |
| `pod.fork`, `pod.rear` | [T, 6] | whitened residuals (predicted - measured) gyro xyz, accel xyz; multiply by `gyro_sigma` 0.05 rad/s and accel sqrt(2) m/s^2 for sensor units |
| `mask_fork`, `mask_rear` | [T] | samples used (False = clipped) |
| `air_mask` | [T] | samples anchored as airborne |
| `terms` | | loss by term |

The printed report: loss by term; travel percentiles; front and rear travel
at take-off, lowest point and landing of every airborne event; frame-pod
flex size (RMS, the part below 5 Hz, peak frequency); fitted biases; and
residual RMS per pod channel.

`travel_uncertainty.py` prints, for travel averaged over 0.05 s to the whole
window, the solver's own posterior std and how correlated the residuals are
at that timescale (see below for how to read it).

## How far to trust it

In order of how much each matters, as measured on the Beacon ride:

1. **Slow travel (sag, below ~1 Hz) is the weak part, mostly at the front.**
   At low frequency the pods see travel only through its second derivative,
   so a slowly varying pod-vs-frame acceleration error a appears as about
   a / (2 pi f)^2 of travel (0.1 m/s^2 at 0.5 Hz is ~10 mm). The rear is
   ~2.5x better determined, because the 4-bar turns the seatstay with
   travel and the gyro sees that directly; the fork slides without turning.
   Rough honest error, from the solver's std scaled by the residual
   correlation: **front ~12 mm, rear ~5 mm at 1 s timescales; ~7 / 3 mm
   averaged over 5 s.** Without the airtime anchor, airborne travel read
   20-26 mm where it should be ~0, more than that estimate allows, so there
   is systematic error beyond it. Compare: the old approach high-passed
   travel at 0.5 Hz, i.e. had no slow travel at all.
2. **Pod-vs-frame accelerometer calibration.** At rest on 11/4 the IMUs read
   |f| = 9.670 (frame), 9.830 (fork), 9.680 (rear) m/s^2: the fork pod and
   frame pod disagree by ~1.6% along gravity (datasheet: +-0.5% sensitivity
   and +-20 mg offset per chip, LSM6DSO32 DS rev 1, PDF p.25). On 1-3 g of
   load that is 0.15-0.5 m/s^2 of fake relative acceleration, the size that
   biases slow front travel by tens of mm. The slow fork residual (0.1-0.3
   m/s^2) is about half explained by a fixed ~1 deg / ~1% pod calibration
   error. **Fitting the calibration from riding does not work:** `--cal`
   (per-axis accel scales, prior 1%, plus mount rotations) returned scale
   errors of 10-14% (10 sigma, physically impossible) for a 3.7% lower
   loss, and moved median front travel 67 -> 111 mm and unanchored airborne
   travel 21 -> 40 mm. Accel scale trades almost freely against the
   lever-arm / travel terms in riding data. The fix is to **measure** it: a
   **six-position static test per pod** (each face down, ~10 s still) gives
   per-axis scale and offset to ~0.1%; then fix them in the sensor model
   (`window_objective(..., sensor_cal=...)` already takes them). The
   rejected fit is kept as `runs/real/gt-force-fit-cal_REJECTED.json`.
3. **Clipping.** The ±16 g pods clip on hits. The clip mask is taken at
   the native rate on the raw int16 counts and widened 10 ms each side
   (cache v2), because the anti-alias filter smears a clipped plateau into
   its neighbours. On Lower Broken Bomber it masks 3.8% (fork) and 6.0%
   (rear) of samples; the old value test on the filtered data caught only
   1.0% / 1.5%. Masked samples carry no information, and they are the big
   hits, so travel through a hard hit is interpolated by the model. Riding
   at ±32 g (moto pods) avoids most of it.
4. **Airtime detection** (`data/real.py: airborne`): frame |f| (10 Hz
   low-pass) under 0.6 g AND both unsprung pods quiet (30-150 Hz accel RMS
   under 1.5 m/s^2) for >= 0.12 s, first 80 ms and last 30 ms of each event
   not anchored. In the air the pods read a median 0.1-0.3 m/s^2 against 7-8
   rolling; the pod check rejects unweighting over a rise while the tires
   still roll. A quiet pod alone is not enough (10-25% of riding at speed
   is that quiet on smooth dirt), so one-wheel-off (manual, wheelie)
   anchoring is not done. The Beacon ride has only ~3 s of anchorable air
   in 100 minutes.
5. **Fast motion is good.** Residuals after the fit: fork gyro 2-3.5 deg/s,
   accel 0.3-0.55 m/s^2; rear gyro 1.2-2.3 deg/s, accel 0.4-0.5 m/s^2 RMS.
   On synthetic rides with exact geometry, travel is recovered to ~1-1.7 mm
   RMS and steer to 0.04 deg; a 0.5 ms pod clock error costs ~3.4 mm, 3x the
   noise ~3 mm.
6. **Known model gaps**: the fork is more compliant near full extension: the
   bushing overlap is shortest there, so the bushing loads and the tubes'
   moment at the bushings are largest, and the crown-to-axle arm is longest
   (Isaac, 2026-11-18). The bending lever now shortens with travel
   (`bend_len` is the crown-to-pod length at top-out). A bending *prior*
   that loosens toward extension was tried and is off: a prior whose sigma
   depends on travel makes less travel cheaper (the Gaussian's log-sigma
   term that would cancel it cannot be a least-squares residual), and it
   moved front median travel 67 -> 25 mm. How bending varies with travel
   should be measured from the solved bending instead, which is the number
   to compare across forks. 60-200 Hz gyro vibration is
   not modelled (largest remaining gyro residual). Pod clocks are fixed (the
   parser's alignment is good to <0.5-1 ms). Frame flex is "frame pod vs a
   rigid chassis", and its ~20 Hz peak coincides with the fork's fore-aft
   mode, so it may be a whole-bike mode or the logger mount, not frame
   stiffness alone.
7. **The bike fit is not converged** at its 20-iteration cap (loss still
   falling ~0.2% per iteration); the geometry itself has stopped moving.
8. **Gauges** (things the data cannot tell apart, held by priors): the
   frame-rate latent vs all gyro biases (anchored to the still recording's
   at-rest biases); a fork mount rotation about the steer axis vs the steer
   zero (removed from the model); linkage scale vs travel scale (prior from
   the photo and the 230 mm shock).

## Results so far (GT Force, Beacon `173461a2`)

Lower Broken Bomber (lap 6, 5784.3-5834.4 s), geometry from the bike fit,
airtime anchor on, native-rate clip mask, bending lever shortening with
travel (2026-11-18):

- Travel: front 1% 0 / median 42 / 99% 105 / max 182 mm (stop 170); rear
  1 / 51 / 135 / 160 mm (stop 160); airborne median 0 / 1 mm.
- Without the airtime anchor (the held-out check of slow travel), airborne
  travel reads front 15 / rear 14 mm median (should be ~0; before the clip
  and bending changes 21 / 20). Front events: 6, 42, 1, 12, 1 mm; the 42 is
  the event at 5793.1 s, where the frame only reached 0.3-0.6 g.
- Front median travel has moved with each model change (67 mm with a fixed
  bending lever, 42 mm now): its absolute level is the least trustworthy
  output until the pods' accelerometers are calibrated (limit 2).
- Fitted geometry: steer point [-0.830, 0.023, -0.245] m (IMU0 axes),
  fork bending length at top-out 0.633 m, 4-bar scale 1.096 (photo + shock
  prior 1.066: the ride data agrees with the photo's linkage to 3%).
- Fork fore-aft bending ~0.26-0.33 deg RMS, mode ~20 Hz; lateral ~40 Hz.
  Rear-member flex 10-45 Hz.
- Frame-pod flex vs chassis: rotation 0.03-0.07 deg RMS, translation
  0.18-0.34 mm RMS, peaks 10 and 19-20 Hz, < 0.02 mm below 5 Hz.
- Modelling the frame-pod flex cut the loss 4.7x and the accel residuals
  3-6x, and moved the 4-bar scale from an implausible 0.87 to 1.08.
- Logs of each step: `runs/real/*.log`.

## How it works

A **differentiable physical model of the bike** predicts what the fork and
rear pods should measure, given the frame pod's measurement and a set of
**latent states** (travel, steer, bending, flex). The estimator chooses the
latents (and sensor biases, and optionally bike geometry) so the prediction
matches the real pods: a maximum a posteriori fit of whitened residuals.
The same objective can later train a network to do it fast (phone-able).

```
                 ┌──────────── latents q(t): travel, steer, bend, flex ─────────┐
  estimator ─────┤  Gauss-Newton batch solve (spline coefficients)  [later: Net] │
                 │                                                               ▼
  frame IMU ────────────────────────────────────────►  Bike forward model  ──► predicted pod IMUs
  (frame rate is a latent the frame gyro measures)      (4-bar, steer, bending,       │
                                                         flex, sensor model)          ▼
  pod IMUs (measured, clip-masked) ────────────────────────────────────────────►  Objective
                                       + end stops, airtime anchor, flex/bend priors, smoothness
```

The key idea: travel is seen through the **lever arm**. A pod's
acceleration relative to the frame pod includes the frame's angular
acceleration and rate squared times the pod's position, and that position
moves with travel, so travel is observed as a position, not integrated from
an acceleration. Gravity is common to all pods and cancels, so no attitude,
velocity or GPS is needed.

## Key decisions

- **The frame's rotation is a latent, not a raw input.** The lever-arm terms
  need the frame's angular acceleration; differencing the raw gyro amplifies
  its noise ~280-fold, which at the fork pod's 0.9 m lever arm swamped
  everything. So the frame rate is a full-rate spline that the frame gyro
  measures, and its exact derivative is the acceleration.
- **Joints are pose functions only.** A joint says where a pod is for given
  latents; exact velocities and accelerations come from nested forward-mode
  AD (`torch.func.jvp`) along the latent trajectory. A new joint or flex
  mode is one function (`idlml/bike/model.py`).
- **Latents are uniform cubic B-splines** at several knot spacings (2.5 ms
  for bending, 5 ms for travel and frame flex, 10 ms for steer), with exact
  derivatives.
- **The rear is a planar 4-bar** from `bikes/gt-force-linkage.json`
  (clicked on a side photo with `tools/linkage.html`); travel -> seatstay
  angle is a degree-9 polynomial of the closure (error < 1e-5 rad). Bikes
  without a linkage file use a generic rotate-and-translate rear.
- **Frame-pod flex**: the frame pod is not the rigid chassis; it flexes on
  it (6 latents, zero-mean priors 0.002 rad / 0.5 mm). The other pods are
  predicted from the chassis.
- **Objective terms are noise-weighted residuals**, so the objective is a
  negative log-likelihood and the batch solve is MAP. Residuals are
  time-local segments, which the solver exploits.
- **The solver is sparse Gauss-Newton / Levenberg-Marquardt**
  (`idlml/solve/gn.py`). Travel enters through its second derivative, so its
  slow modes are nearly flat and first-order optimisers stall (L-BFGS left
  front travel 23 mm low at 13x the true loss). Each sample's residuals
  depend on 45 local quantities and the globals; one batched forward-mode
  call gives every sample's small Jacobian, the spline basis chains it to
  the coefficients, and the normal matrix is a band (unknowns in time order)
  plus a border for the globals: blocked Cholesky on the GPU in float64 and
  a Schur complement. Damping starts near zero (1e-10): Marquardt damping
  brakes hardest on exactly the slow travel modes. Several windows can
  share their globals (`gauss_newton_multi`), which is how the bike is fit.
- **Gyro biases are anchored** to the still recording's at-rest means
  (sigma 0.002 rad/s): otherwise the frame-rate latent and all three biases
  trade a common offset freely.
- **torch.compile does not help yet**: it captures nothing of the
  forward-mode transforms (0 graphs). Fused kernels would need the time
  derivatives as explicit Taylor-mode arithmetic (not done; risky rewrite).
- **Precision**: physics in float32, normal equations and solve in float64.

## Layout

```
ml/
  bikes/        gt-force-linkage.json (4-bar points from the photo tool),
                gt-force-fit.json (fitted geometry; written by ride_fit --refit)
  cache/        per-session 400 Hz caches (gitignored, rebuildable)
  runs/         run outputs and logs (gitignored)
  idlml/
    data/       cache.py: parquet -> 400 Hz cache; real.py: the GT Force bike,
                window objectives, still-recording biases, airtime detection,
                fit-window choice, saving/loading bike fits
    geometry/   so3.py: rotations, batched over time
    bike/       model.py: latents, Bike (steer, travel, bending, 4-bar or
                generic rear, flex), pod_readings; fourbar.py: the 4-bar
    sensors/    imu.py: PodSensor (bias, scale/misalignment, clock delay), clip mask
    traj/       spline.py: cubic B-spline latents with exact derivatives
    factors/    objective.py: Weights, the residual terms, priors
    solve/      gn.py: Gauss-Newton (band + border, GPU); batch.py: latent
                layout (knot spacing, scales) and the reference Jacobian, L-BFGS
    sim/        synth.py: synthetic rides from real frame motion + invented latents
  experiments/  ride_fit.py, travel_uncertainty.py (real rides); recover.py,
                diagnose.py (synthetic); real_window.py, rigid_check.py,
                pod_lag.py (one-off real-data checks); profile_gn.py,
                bench_solve.py, band.py, try_compile.py, explain_compile.py (speed)
  explore/      throwaway analysis (moto swingarm hinge test)
  tools/        linkage.html: click a bike's pivots on a side photo, export the 4-bar
  tests/        kinematics vs brute-force world motion, solver vs reference
```

Adding a bike: a calibration in the library's bike file (mounts, lever
arms, up, forward, steer axis), a linkage JSON from `tools/linkage.html`
(optional), at-rest biases from a still recording (`data/real.py: STILL`),
the travel limits (`Weights.travel_max`), and the IMU full scales
(`ACC_FS`, `GYRO_FS`; today hard-coded for the GT's ±16 g / 2000 dps).
`gt_force()` is the template; it is not yet generic.

## Next steps and the road to idl1-app

Near term, in order:
1. Six-position static test of each pod, and use the measured scales and
   offsets (limit 2). Then re-check airborne travel *without* the anchor:
   that is the honest test of slow travel.
2. Measure the crown-to-fork-pod length at top-out and fix the bending
   length to it: fitted bending length and slow front travel trade almost
   freely (the Rust port's own fit moved front median travel 39 -> 67 mm;
   `core/src/estimate/README.md`).
3. A second bike (dual-crown MTB or the moto, which needs a calibration
   first) to compare fork bending and frame flex across bikes.

**Rust port** (`core/src/estimate/physics/`, 2026-11-18): done, apart from
speed work. The forward model, its per-sample Jacobian, the objective and
the window solver are held equal to this code by golden tests
(`ml/experiments/export_golden.py`, `export_window.py` write the fixtures;
regenerate them after a model change here). The port also has the session
pipeline this folder does by hand: resampled input and native-rate clip
masks, the 0.4x-grid-rate low-pass this folder's cache applies, airborne
detection, still-recording biases, window picking, the 4-bar from the photo
tool's points, the bike fit across 15 s windows (stored on the calibration
as its `physics` fit), and a threaded whole-ride solve in overlapping
windows. First end-to-end check on Beacon, with this folder's fitted
geometry: front travel 1 % / median / 99 % 0 / 39 / 103 mm and rear
0 / 51 / 133 mm, against 0 / 42 / 105 and 1 / 51 / 135 here.

Toward the app: the port's channels are written to the session's derived
store (`sessions/<id>/derived/<hash>.parquet`, contract C1 §5,
`derived_kind` `"physics"`), content-addressed by the IMU channels, the
configuration (including the bike fit) and the engine version, so a new
estimator version or a new fit gives a new file and an old one is never
mistaken for current. They are never written into
`data.parquet` (a pure function of the blob and the importer). **Nothing
reads them yet**: neither the app nor `workbook eval`. The missing piece is
a lazy channel source over the newest current physics file, beside
`data.parquet`'s, in `core/src/session/handle.rs`, so the channels plot
beside the IEKF's and workbooks can use them. Running the estimator on a
phone is the later network step (train a network on this objective, export
.tflite).
