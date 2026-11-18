"""Synthetic rides with known answers.

Real frame motion from a ride (the frame pod's rate and specific force),
invented latents shaped like riding (travel follows the frame's vertical and
braking load, steer wanders, the fork bends with braking and vibrates), and
pod readings from the same forward model the estimators use, with sensor
errors added. Every number the estimators recover can then be checked.

The latent shapes and sizes are guesses, chosen to be plausible, not
measured. Bending sizes especially are placeholders until the free 6-DOF fit
on real rides says what they are.
"""
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from ..bike.model import B_FA, B_LAT, D, FRAME_ROT, FRAME_TRANS, NL, PSI, R_ROLL, R_YAW, S, pod_readings
from ..sensors.imu import G, PodSensor
from ..traj.spline import evaluate

ACC_FS = 32 * G
GYRO_FS = 500 * np.pi / 180


@dataclass
class SimConfig:
    travel_max: tuple = (0.17, 0.16)  # front, rear, m
    sag: tuple = (0.045, 0.045)
    accel_noise: float = 0.3  # m/s^2 rms per axis, white (sensor + unmodelled vibration)
    gyro_noise: float = 0.01  # rad/s rms
    frame_accel_noise: float = 0.3
    frame_gyro_noise: float = 0.01
    pod_accel_bias: float = 0.2  # m/s^2 rms per axis
    pod_gyro_bias: float = 0.005
    frame_gyro_bias: float = 0.003
    clock: dict = field(default_factory=lambda: {"fork": 0.0, "rear": 0.0})
    bend_fa_brake: float = 0.0015  # rad per (m/s^2 of braking / g) -> ~0.1 deg at 1 g
    bend_vib: tuple = (0.0015, 0.0008)  # rad rms, fore-aft, lateral
    bend_band: tuple = (12.0, 40.0)  # Hz
    steer_rms: float = np.radians(6)
    rear_flex_vib: tuple = (0.002, 0.0015)  # rad rms, roll, yaw (real Beacon spectra: ~0.1 deg, 10-45 Hz)
    frame_flex: tuple = (0.0005, 0.0002)  # rms of the frame pod's flex vs the chassis: rad, m (10-45 Hz), ~Beacon residual size
    knots: dict = field(default_factory=lambda: {D: 2, PSI: 4, B_FA: 1, B_LAT: 1, S: 2, R_ROLL: 1, R_YAW: 1,
                                                 **{i: 2 for i in FRAME_ROT + FRAME_TRANS}})


def _band(x, lo, hi, fs):
    if lo is None:
        sos = butter(2, hi, fs=fs, output="sos")
    else:
        sos = butter(2, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def _softclip(x, lo, hi, k=0.01):
    """Smooth clamp into [lo, hi] (softplus walls of width k)."""
    sp = lambda v: k * np.logaddexp(0, v / k)
    return lo + sp(x - lo) - sp(x - hi)


def invent_latents(w0, f0, rate, up, fwd, cfg, rng):
    """Latent values [T, NL] shaped by the frame's real motion."""
    t = len(w0)
    fz = f0 @ up  # vertical specific force, m/s^2 (g when level and steady)
    brake = np.clip(-(f0 @ fwd), 0, None)  # forward deceleration felt, m/s^2
    lp = lambda x, f: _band(x, None, f, rate)
    noise = lambda lo, hi: _band(rng.standard_normal(t), lo, hi, rate)
    q = np.zeros((t, NL))
    for i, wheel_lag in ((D, 0), (S, int(0.05 * rate))):
        k = 0 if i == D else 1
        load = lp(fz - G, 4.0) * 0.004 + (lp(brake, 1.0) * 0.004 if i == D else 0)
        x = cfg.sag[k] + load + 0.012 * noise(0.3, 6.0) / max(noise(0.3, 6.0).std(), 1e-9)
        x = np.roll(x, wheel_lag)
        q[:, i] = _softclip(x, 0.003, cfg.travel_max[k] - 0.003)
    s = noise(0.05, 2.0)
    q[:, PSI] = cfg.steer_rms * s / s.std()
    vfa, vlat = noise(*cfg.bend_band), noise(*cfg.bend_band)
    q[:, B_FA] = cfg.bend_fa_brake * lp(brake, 3.0) / G + cfg.bend_vib[0] * vfa / vfa.std()
    q[:, B_LAT] = cfg.bend_vib[1] * vlat / vlat.std()
    for i in FRAME_ROT + FRAME_TRANS:
        x = noise(10.0, 45.0)
        q[:, i] = (cfg.frame_flex[0] if i in FRAME_ROT else cfg.frame_flex[1]) * x / x.std()
    for i, amp in ((R_ROLL, cfg.rear_flex_vib[0]), (R_YAW, cfg.rear_flex_vib[1])):
        x = noise(10.0, 45.0)
        q[:, i] = amp * x / x.std()
    return q


def synth(bike, w0, f0, rate, cfg=SimConfig(), seed=0, device="cpu"):
    """One synthetic window. w0, f0: real frame rate and specific force
    (numpy [T,3], IMU0 axes), taken as the truth. Returns tensors."""
    rng = np.random.default_rng(seed)
    up, fwd = bike.up.cpu().numpy(), bike.fwd.cpu().numpy()
    vals = invent_latents(w0, f0, rate, up, fwd, cfg, rng)
    t = len(w0)
    dt = 1.0 / rate
    # the truth is a spline whose coefficients are the invented values, so its
    # derivatives are exact
    q = torch.zeros(t, NL); qd = torch.zeros(t, NL); qdd = torch.zeros(t, NL)
    for i, m in cfg.knots.items():
        c = torch.tensor(vals[::m, i], dtype=torch.float32)
        k = int(np.ceil(t / m)) + 3
        c = torch.cat([c[:1].expand(1), c, c[-1:].expand(k - 1 - len(c))])[None]
        a, b, cc = evaluate(c, m, t, dt)
        q[:, i], qd[:, i], qdd[:, i] = a[:, 0], b[:, 0], cc[:, 0]
    # the true frame rate is a full-rate spline through the real gyro (the
    # estimator's own representation), so its derivative is exact
    g = torch.tensor(w0, dtype=torch.float32)
    wcoef = torch.cat([g[:1], g, g[-1:].expand(2, 3)]).T.contiguous()
    w, a0, _ = evaluate(wcoef, 1, t, dt)
    w0t = w
    f0t = torch.tensor(f0, dtype=torch.float32)
    f = f0t
    with torch.no_grad():
        true = pod_readings(bike, q, qd, qdd, w, a0, f)
    frame_gyro_true = true.pop("frame")[0]
    out = {"q": q, "qd": qd, "qdd": qdd, "rate": rate, "pods": {}, "sensors": {}, "frame_coef": wcoef}
    for name, (g, a) in true.items():
        sens = PodSensor()
        with torch.no_grad():
            sens.gyro_bias.copy_(torch.tensor(rng.normal(0, cfg.pod_gyro_bias, 3)))
            sens.accel_bias.copy_(torch.tensor(rng.normal(0, cfg.pod_accel_bias, 3)))
            sens.clock.fill_(cfg.clock.get(name, 0.0))
            gm, am = sens(g, a, rate)
        gm = gm + torch.tensor(rng.normal(0, cfg.gyro_noise, (t, 3)), dtype=torch.float32)
        am = am + torch.tensor(rng.normal(0, cfg.accel_noise, (t, 3)), dtype=torch.float32)
        mask = (am.abs() < 0.98 * ACC_FS).all(-1) & (gm.abs() < 0.98 * GYRO_FS).all(-1)
        out["pods"][name] = {"gyro": gm.clamp(-GYRO_FS, GYRO_FS), "accel": am.clamp(-ACC_FS, ACC_FS), "mask": mask}
        out["sensors"][name] = sens
    fb = rng.normal(0, cfg.frame_gyro_bias, 3)
    out["frame"] = {
        "gyro": frame_gyro_true + torch.tensor(fb + rng.normal(0, cfg.frame_gyro_noise, (t, 3)), dtype=torch.float32),
        "accel": f0t + torch.tensor(rng.normal(0, cfg.frame_accel_noise, (t, 3)), dtype=torch.float32),
        "gyro_bias": torch.tensor(fb, dtype=torch.float32),
    }
    return out
