"""Is a pod on a hinge relative to the frame, and can its angle be read absolutely?

For IMU k against IMU0: find the fixed rotation R (k -> frame) by Kabsch on
the gyro vectors, then the axis of the relative rate w_k - R^T w_0 (PCA). A
true hinge puts nearly all relative-rate variance on one axis. Then the hinge
angle two ways:
  gyro : integral of the relative rate about the axis (drifts with bias)
  accel: angle about the axis between f_k and R^T f_0, both low-passed
         (gravity + bike acceleration seen from the two bodies; no drift)
and a complementary blend of the two.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from idlml.data.cache import load

HERE = os.path.dirname(os.path.abspath(__file__))


def kabsch(a, b):
    """R minimising |R b - a| over rows (maps b's frame into a's)."""
    h = b.T @ a
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1, 1, d]) @ u.T


def lowpass(x, fc, fs, order=2):
    return sosfiltfilt(butter(order, fc, fs=fs, output="sos"), x, axis=0)


def hinge(d, k, fc=0.3):
    fs = float(d["rate"])
    w0, wk = d["gyr0"].astype(np.float64), d[f"gyr{k}"].astype(np.float64)
    f0, fk = d["acc0"].astype(np.float64), d[f"acc{k}"].astype(np.float64)
    sel = np.arange(0, len(w0), 8)
    R = kabsch(w0[sel], wk[sel])  # R @ wk ~ w0
    rel = wk - w0 @ R  # w_k - R^T w_0, in IMU k axes
    rel_c = rel[sel] - rel[sel].mean(0)
    _, s, vt = np.linalg.svd(rel_c, full_matrices=False)
    axis = vt[0]
    frac = s**2 / (s**2).sum()
    rate = rel @ axis
    th_gyro = np.cumsum(rate - np.median(rate)) / fs
    # accel angle: both low-passed, in IMU k axes, projected on the plane normal to the axis
    a = lowpass(fk, fc, fs)
    b = lowpass(f0 @ R, fc, fs)

    def proj(v):
        return v - np.outer(v @ axis, axis)

    pa, pb = proj(a), proj(b)
    th_acc = np.arctan2(np.cross(pb, pa) @ axis, (pa * pb).sum(1))
    # complementary: gyro above fc, accel below
    hp = th_gyro - lowpass(th_gyro, fc, fs)
    th = hp + lowpass(th_acc, fc, fs)
    return dict(R=R, axis=axis, frac=frac, th_gyro=th_gyro, th_acc=th_acc, th=th, rate=rate,
                axis_in_frame=R @ axis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--imu", type=int, default=2)
    ap.add_argument("--lap", type=int, default=3)
    a = ap.parse_args()
    d = load(a.session)
    h = hinge(d, a.imu)
    deg = np.degrees
    print(f"IMU{a.imu}: relative-rate variance by axis {np.round(h['frac'], 4)}")
    print(f"hinge axis in IMU{a.imu} axes {np.round(h['axis'], 3)}, in IMU0 axes {np.round(h['axis_in_frame'], 3)}")
    t = d["t"]
    th = deg(h["th"])
    lo, hi = np.percentile(th, [0.1, 99.9])
    print(f"blended angle: 0.1%..99.9% = {lo:.1f}..{hi:.1f} deg, range {hi - lo:.1f}; median {np.median(th):.1f}")
    drift = deg(h["th_gyro"][-1] - h["th_gyro"][0])
    print(f"gyro-only angle drift over the ride: {drift:.0f} deg in {t[-1] / 60:.0f} min")
    laps = d["laps"]
    row = laps[laps[:, 0] == a.lap][0] if len(laps) else (0, t[0], t[0] + 120)
    # lap times are session seconds; the grid starts at IMU0's first sample - near enough for a look
    m = (t >= row[1]) & (t < row[2])
    fig, ax = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    ax[0].plot(t[m], deg(h["th_acc"][m]) - np.median(deg(h["th_acc"][m])), lw=0.6, label="accel (gravity) angle")
    g = deg(h["th_gyro"][m])
    ax[0].plot(t[m], g - np.median(g), lw=0.6, label="gyro-integrated angle")
    ax[0].legend(); ax[0].set_ylabel("deg (median removed)")
    ax[1].plot(t[m], th[m], lw=0.6, color="k"); ax[1].set_ylabel("blended hinge angle, deg")
    ax[1].axhline(lo, ls=":"); ax[1].axhline(hi, ls=":")
    ax[2].plot(t[m], deg(h["rate"][m]), lw=0.4); ax[2].set_ylabel("hinge rate, deg/s"); ax[2].set_xlabel("s")
    fig.suptitle(f"{a.session[:8]} IMU{a.imu} vs IMU0 hinge, lap {a.lap}")
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    p = os.path.join(HERE, "out", f"hinge_{a.session[:8]}_imu{a.imu}_lap{a.lap}.png")
    fig.tight_layout(); fig.savefig(p, dpi=110)
    print(p)


if __name__ == "__main__":
    main()
