"""Real rides into the estimator: a window of the cache as an Objective, and
the GT Force bike with its calibration and (if present) its 4-bar."""
import os

import numpy as np
import torch

from ..bike.model import Bike, load_calibration
from ..factors.objective import LatentSpec, Objective
from ..sensors.imu import G, PodSensor
from ..solve.batch import INIT, LEVELS, SCALE

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# full scales of this hardware; read per session from the parquet metadata later
ACC_FS = 16 * G  # 0.000488 g/LSB
GYRO_FS = np.radians(2000)


def gt_force(use_linkage=True):
    """The GT Force: calibration from the library, 4-bar from ml/bikes if present.
    The linkage prior is scaled by the shock: the photo read the 230 mm
    eye-to-eye as 215.8 mm."""
    link_path = os.path.join(ROOT, "bikes", "gt-force-linkage.json")
    linkage = None
    if use_linkage and os.path.exists(link_path):
        from ..bike.fourbar import FourBar, load_linkage
        linkage = FourBar(load_linkage(link_path)[0], 0.16)
    return Bike(load_calibration("gt-force"), linkage=linkage, link_scale=230 / 215.83 if linkage else 1.0)


def default_fit(bike):
    return ["steer_point", "bend_len"] + (["link_scale"] if bike.linkage is not None else ["rear_rot", "rear_dir", "rear_pivot"])


STILL = {"gt-force": "25544763"}  # still recordings for at-rest gyro biases


def still_biases(session):
    """At-rest gyro biases (rad/s, each IMU's own axes) from a still recording,
    keyed by objective global name."""
    from .cache import load
    d = load(session)
    skip = int(1.0 * float(d["rate"]))
    names = ["frame_bias", "fork.gyro_bias", "rear.gyro_bias"]
    return {n: torch.tensor(d[f"gyr{k}"][skip:].mean(0), dtype=torch.float32) for k, n in enumerate(names)}


FIT_FILE = os.path.join(ROOT, "bikes", "gt-force-fit.json")


def save_fit(values, meta, path=FIT_FILE):
    """Fitted bike geometry {FITTABLE name: tensor or list} plus how it was got."""
    import json
    vals = {k: np.asarray(v.detach().cpu() if torch.is_tensor(v) else v, dtype=float).reshape(-1).tolist() for k, v in values.items()}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"values": vals, **meta}, fh, indent=1)


def load_fit(path=FIT_FILE):
    import json
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def apply_fit(bike, values):
    """Copy fitted geometry into the bike's parameters, in place."""
    params = dict(bike.named_parameters())
    with torch.no_grad():
        for k, v in values.items():
            if k.startswith("sensor."):  # pod sensor calibration: see sensor_cal()
                continue
            params[k].copy_(torch.as_tensor(v, dtype=params[k].dtype).reshape(params[k].shape))
    return bike


def sensor_cal(values):
    """Pod accelerometer scale errors from a saved fit: {pod: [3]}."""
    return {k.split(".")[1]: v for k, v in values.items() if k.startswith("sensor.") and k.endswith(".accel_scale")}


def airborne(d, sl, min_s=0.12, settle_s=0.08, trim_s=0.03, g_frac=0.6, quiet=1.5):
    """Free-fall samples of window `sl`. Airborne = the frame's specific force
    (10 Hz low-pass) under g_frac * g AND both unsprung pods quiet (30-150 Hz
    accel rms, ~20 Hz envelope, under `quiet` m/s^2: no tire contact noise)
    for at least min_s. On the Beacon ride: in free fall the pods read a
    median 0.1-0.3 m/s^2 against 7-8 rolling; frame g alone misses jumps the
    rider pushes through (0.3-0.6 g), and pod quiet alone also matches smooth
    ground. Each event's first settle_s is dropped (the fork may still be
    rebounding toward top-out) and its last trim_s (filters smear the
    landing). Returns (mask [T], events [(i0, i1)])."""
    from scipy.signal import butter, sosfiltfilt
    rate = float(d["rate"])
    f = np.linalg.norm(np.asarray(d["acc0"][sl]), axis=1)
    ok = sosfiltfilt(butter(2, 10, fs=rate, output="sos"), f) < g_frac * G
    bp = butter(4, [30, min(150, 0.45 * rate)], btype="band", fs=rate, output="sos")
    env = butter(2, 20, fs=rate, output="sos")
    for k in (1, 2):
        hf = sosfiltfilt(bp, np.asarray(d[f"acc{k}"][sl]), axis=0)
        ok &= np.sqrt(np.clip(sosfiltfilt(env, (hf**2).sum(1)), 0, None)) < quiet
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(np.int8), 0]))
    mask = np.zeros(len(f), bool)
    events = []
    for i0, i1 in zip(edges[::2], edges[1::2]):
        if (i1 - i0) / rate >= min_s:
            events.append((int(i0), int(i1)))
            mask[i0 + int(settle_s * rate):i1 - int(trim_s * rate)] = True
    return mask, events


def window_objective(d, sl, bike, weights, fit=(), dev="cuda", bias_prior="auto", air=False, accel_scale=False,
                     sensor_cal=None):
    """An Objective over samples `sl` of a cached ride. bias_prior "auto":
    the GT Force still recording's at-rest gyro biases. air: anchor travel
    near 0 in free fall (see airborne). accel_scale: estimate each pod
    accelerometer's per-axis scale; sensor_cal {pod: [3]}: fixed known scales instead."""
    if bias_prior == "auto":
        bias_prior = still_biases(STILL["gt-force"])
    tt = lambda k: torch.tensor(d[k][sl], dtype=torch.float32, device=dev)
    frame = {"gyro": tt("gyr0"), "accel": tt("acc0")}
    # clipping from the cache's native-rate masks (cache v2); the value test below stays as a backstop
    nat = lambda k: torch.tensor(~np.asarray(d[f"clip{k}"][sl]), device=dev) if f"clip{k}" in d else True
    frame_ok = (frame["accel"].abs() < 0.98 * ACC_FS).all(-1) & nat(0)
    pods = {}
    for name, k in (("fork", 1), ("rear", 2)):
        g, acc = tt(f"gyr{k}"), tt(f"acc{k}")
        mask = (acc.abs() < 0.98 * ACC_FS).all(-1) & (g.abs() < 0.98 * GYRO_FS).all(-1) & frame_ok & nat(k)
        pods[name] = {"gyro": g, "accel": acc, "mask": mask}
    sensors = {k: PodSensor().to(dev) for k in pods}
    for name, s in sensors.items():
        s.clock.requires_grad_(False)
        if sensor_cal and name in sensor_cal:
            with torch.no_grad():
                s.accel_mis.copy_(torch.diag(torch.as_tensor(sensor_cal[name], dtype=torch.float32, device=dev)))
    n = sl.stop - sl.start
    rate = float(d["rate"])
    return Objective(bike.to(dev), sensors, frame, pods, rate, LatentSpec(n, 1 / rate, LEVELS, INIT, SCALE), weights, fit=fit,
                     bias_prior=bias_prior,
                     air=torch.tensor(airborne(d, sl)[0], device=dev) if air else None, accel_scale=accel_scale)


def _pick(score, n, count, exclude=None):
    """Greedy: the `count` non-overlapping n-sample windows with the highest
    mean of per-sample `score`, skipping any that touch `exclude` (a slice or a bool mask)."""
    cs = np.concatenate([[0.0], np.cumsum(score, dtype=np.float64)])
    c = (cs[n:] - cs[:-n]) / n  # moving mean (np.convolve here is O(N n): minutes on a whole ride)
    taken = np.zeros(len(score), bool)
    if exclude is not None:
        excl = exclude if isinstance(exclude, (list, tuple)) else [exclude]
        for e in excl:
            taken[e] = True
    out = []
    for i in np.argsort(c)[::-1]:
        if taken[i:i + n].any() or c[i] <= 0:
            continue
        out.append(slice(int(i), int(i) + n))
        taken[max(0, i - n):i + 2 * n] = True
        if len(out) == count:
            break
    return sorted(out, key=lambda s: s.start)


def busy_windows(d, seconds, count, exclude=None):
    """The `count` non-overlapping windows with the most frame rotation (the
    most informative for geometry), skipping samples in `exclude`."""
    rate = float(d["rate"])
    w = np.empty(len(d["gyr0"]), np.float32)
    for i in range(0, len(w), 1_000_000):  # in chunks: the cache is memory-mapped
        w[i:i + 1_000_000] = np.linalg.norm(d["gyr0"][i:i + 1_000_000], axis=1)
    return _pick(w, int(seconds * rate), count, exclude)


def ride_airborne(d, chunk_s=600.0):
    """airborne() over the whole ride, in overlapping chunks (memory-mapped cache)."""
    rate = float(d["rate"])
    n, ch, pad = len(d["t"]), int(chunk_s * rate), int(2 * rate)
    mask = np.zeros(n, bool)
    for i in range(0, n, ch):
        s = slice(max(0, i - pad), min(n, i + ch + pad))
        m, _ = airborne(d, s)
        k0 = i - s.start
        mask[i:min(n, i + ch)] = m[k0:k0 + min(ch, n - i)]
    return mask


def airtime_windows(d, seconds, count, exclude=None, air=None):
    """The `count` non-overlapping windows with the most airtime (where the
    airtime anchor pins travel's slow part), skipping `exclude`."""
    air = ride_airborne(d) if air is None else air
    return _pick(air.astype(np.float32), int(seconds * float(d["rate"])), count, exclude)
