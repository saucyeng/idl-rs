"""Load one library session into a compact, common-timebase numpy cache.

Streams the session's data.parquet one row group at a time (the files are
~350 MB and RAM is tight), keeps each IMU on its own recorded clock, anti-alias
filters it at its native ~800 Hz, then reads every IMU at the same grid times
(default 400 Hz) from IMU0's first to the last time all three IMUs cover.

Output (cache/<session>_<rate>.npz), float32 SI units in each sensor's own axes:
  t        grid times, s from the start of the grid
  t0_us    the grid's first time, recorded-clock microseconds (int64)
  acc{k}   (n,3) specific force, m/s^2      gyr{k}  (n,3) angular rate, rad/s
  clip{k}  (n,) bool: a native sample within CLIP_SMEAR_S of this grid time
           was at full scale on any int16 axis (taken before filtering: the
           anti-alias filter smears a clipped plateau into its neighbours)
  gps_t, gps_speed (m/s), gps_course (deg), gps_lat, gps_lon, gps_alt
"""
import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.signal import butter, sosfiltfilt

G = 9.80665
LIBRARY = os.environ.get("IDL_LIBRARY", r"C:\Users\isaac\Documents\idl1-library\data")
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(HERE, "cache")
AXES = ("X", "Y", "Z")
CLIP_FRAC = 0.98  # of the int16 full scale (32767 counts)
CLIP_SMEAR_S = 0.010  # s each side: the 4th-order filtfilt at 160 Hz rings ~this long
CACHE_VERSION = 2  # 2: clip masks


def session_dir(session):
    d = os.path.join(LIBRARY, "sessions")
    hits = [s for s in os.listdir(d) if s.startswith(session)]
    if len(hits) != 1:
        sys.exit(f"session prefix {session!r} matches {len(hits)} sessions")
    return os.path.join(d, hits[0])


def read_streams(path):
    f = pq.ParquetFile(path)
    names = set(f.schema_arrow.names)
    meta = {fld.name: {k.decode(): v.decode() for k, v in (fld.metadata or {}).items()} for fld in f.schema_arrow}
    imus = [k for k in range(3) if f"IMU{k}_AccelX" in names]
    gps_cols = ["GPS_SpeedKmh", "GPS_Heading", "GPS_Latitude", "GPS_Longitude", "GPS_Altitude", "GPS_FixQuality"]
    acc = {k: {"t": [], "v": [], "clip": []} for k in imus}
    gps = {"t": [], "v": []}
    for rg in range(f.num_row_groups):
        for k in imus:
            cols = [f"imu{k}_t_recorded_us"] + [f"IMU{k}_{q}{a}" for q in ("Accel", "Gyro") for a in AXES]
            tb = f.read_row_group(rg, columns=cols)
            tt = tb.column(0)
            mask = tt.is_valid()
            tt = tt.filter(mask).to_numpy()
            raw = [tb.column(c).filter(mask) for c in cols[1:]]
            v = np.stack([r.to_numpy().astype(np.float32) for r in raw], 1)
            # clipping, on the raw integer counts (float columns such as IMU0's high-g accel have no clip level)
            ints = [r.to_numpy() for r in raw if r.type == "int16"]
            clip = (np.abs(np.stack(ints, 1).astype(np.int32)) >= CLIP_FRAC * 32767).any(1) if ints else np.zeros(len(tt), bool)
            acc[k]["t"].append(tt)
            acc[k]["v"].append(v)
            acc[k]["clip"].append(clip)
        tb = f.read_row_group(rg, columns=["gps_t_recorded_us"] + gps_cols)
        tt = tb.column(0)
        mask = tt.is_valid()
        gps["t"].append(tt.filter(mask).to_numpy())
        gps["v"].append(np.stack([tb.column(c).filter(mask).to_numpy().astype(np.float64) for c in gps_cols], 1))
    out = {}
    for k in imus:
        t = np.concatenate(acc[k]["t"])
        v = np.concatenate(acc[k]["v"])
        clip = np.concatenate(acc[k]["clip"])
        o = np.argsort(t, kind="stable")
        t, v, clip = t[o], v[o], clip[o]
        sa = float(meta[f"IMU{k}_AccelX"]["scale"]) * G
        sg = float(meta[f"IMU{k}_GyroX"]["scale"]) * np.pi / 180
        v[:, :3] *= sa
        v[:, 3:] *= sg
        out[k] = (t, v, float(meta[f"IMU{k}_AccelX"]["nominal_rate_hz"]), clip)
    gt = np.concatenate(gps["t"])
    gv = np.concatenate(gps["v"])
    gv[:, 0] *= 0.01 / 3.6  # raw int scale 0.01 km/h -> m/s
    return out, (gt, gv)


def build(session, rate=400.0, cutoff=None):
    sdir = session_dir(session)
    sid = os.path.basename(sdir)
    imus, (gt, gv) = read_streams(os.path.join(sdir, "data.parquet"))
    cutoff = cutoff or 0.4 * rate  # leaves margin below the new Nyquist
    t0 = imus[0][0][0]
    t1 = min(v[0][-1] for v in imus.values())
    smear_us = CLIP_SMEAR_S * 1e6
    grid = np.arange(t0, t1, 1e6 / rate)
    res = {"t": ((grid - t0) / 1e6).astype(np.float64), "t0_us": np.int64(t0), "rate": rate,
           "version": np.int64(CACHE_VERSION)}
    for k, (t, v, fs, clip) in imus.items():
        # a grid time is clipped if any clipped native sample lies within smear_us of it
        ct = t[clip].astype(np.float64)
        lo = np.searchsorted(ct, grid - smear_us)
        hi = np.searchsorted(ct, grid + smear_us, side="right")
        res[f"clip{k}"] = hi > lo
        sos = butter(4, cutoff, fs=fs, output="sos")
        vf = sosfiltfilt(sos, v, axis=0)
        r = np.empty((len(grid), 6), np.float32)
        for c in range(6):
            r[:, c] = np.interp(grid, t, vf[:, c])
        res[f"acc{k}"] = r[:, :3]
        res[f"gyr{k}"] = r[:, 3:]
        del vf
    keep = (gt >= t0) & (gt <= t1)
    res["gps_t"] = (gt[keep] - t0) / 1e6
    for i, name in enumerate(["gps_speed", "gps_course", "gps_lat", "gps_lon", "gps_alt", "gps_fix"]):
        res[name] = gv[keep, i]
    with open(os.path.join(sdir, "session.json"), encoding="utf-8") as fh:
        sj = json.load(fh)
    laps = [(l["lap_number"], l.get("kind", "normal"), l["start_time_secs"], l["end_time_secs"]) for l in sj.get("laps", [])]
    res["laps"] = np.array([(n, s, e) for n, k, s, e in laps], dtype=np.float64).reshape(-1, 3)
    res["lap_start_offset_s"] = 0.0  # lap times are session seconds; see notes in README
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, f"{sid[:8]}_{int(rate)}_v{CACHE_VERSION}.npz")
    np.savez(out, **res)
    return out


def load(session, rate=400.0):
    """The cached session as a dict of arrays, memory-mapped: each array is
    its own .npy file in cache/<id>_<rate>/, so slicing a window reads only
    that window from disk. Built from the .npz (or the parquet) the first time."""
    base = os.path.join(CACHE, f"{session[:8]}_{int(rate)}_v{CACHE_VERSION}")
    if not os.path.isdir(base):
        p = base + ".npz"
        if not os.path.exists(p):
            p = build(session, rate)
        os.makedirs(base + ".tmp", exist_ok=True)
        with np.load(p) as z:
            for k in z.files:
                np.save(os.path.join(base + ".tmp", k + ".npy"), z[k])
        os.replace(base + ".tmp", base)
    return {f[:-4]: np.load(os.path.join(base, f), mmap_mode="r") for f in os.listdir(base) if f.endswith(".npy")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--rate", type=float, default=400.0)
    a = ap.parse_args()
    p = build(a.session, a.rate)
    d = np.load(p)
    print(p, {k: d[k].shape for k in d.files})
