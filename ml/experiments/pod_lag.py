"""Timing between the frame pod and each other pod, from the data.

The frame and a pod share the bike's rotation, so their rates (in the same
axes, via the calibrated mount) should be in phase. A clock offset tau shows
as a cross-spectrum phase of -2 pi f tau. Fit tau over the frequencies where
the two are coherent, per axis, on a window of the ride.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
from scipy.signal import coherence, csd  # noqa: E402

from idlml.bike.model import Bike, load_calibration  # noqa: E402
from idlml.data.cache import load  # noqa: E402


def lag(x, y, fs, fmax=60.0, coh_min=0.8):
    """Delay of y relative to x (s), from the cross-spectrum phase slope."""
    f, pxy = csd(x, y, fs=fs, nperseg=1024)
    _, c = coherence(x, y, fs=fs, nperseg=1024)
    use = (f > 1.0) & (f < fmax) & (c > coh_min)
    if use.sum() < 5:
        return float("nan"), 0, 0.0
    ph = np.unwrap(np.angle(pxy[use]))
    wts = c[use]
    # phase = -2 pi f tau (+ 0): weighted least squares through the origin
    tau = -np.sum(wts * ph * f[use]) / (2 * np.pi * np.sum(wts * f[use] ** 2))
    return tau, int(use.sum()), float(c[use].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="173461a2")
    ap.add_argument("--start", type=float, default=5784.3)
    ap.add_argument("--end", type=float, default=5834.4)
    a = ap.parse_args()
    d = load(a.session)
    fs = float(d["rate"])
    t = d["t"]
    sl = slice(int(np.searchsorted(t, a.start)), int(np.searchsorted(t, a.end)))
    bike = Bike(load_calibration("gt-force"))
    w0 = d["gyr0"][sl].astype(np.float64)
    print(f"window {a.start:.1f}-{a.end:.1f} s; delay of each pod behind the frame pod, per frame axis")
    for name, k, r in (("fork", 1, bike.fork_mount.numpy()), ("rear", 2, bike.rear_mount.numpy())):
        wk = d[f"gyr{k}"][sl].astype(np.float64) @ r.T
        for ax in range(3):
            tau, nf, c = lag(w0[:, ax], wk[:, ax], fs)
            print(f"  {name} axis {'xyz'[ax]}: {tau * 1e3:+7.2f} ms   ({nf} coherent bins, mean coherence {c:.2f}, rms {np.degrees(w0[:, ax].std()):.0f} deg/s)")


if __name__ == "__main__":
    main()
