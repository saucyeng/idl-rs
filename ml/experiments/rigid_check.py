"""Model-free check of how the three gyros relate when the bike moves as one body.

Rigid bodies share one angular velocity, so each pod's gyro should read the
frame gyro's rate rotated by a fixed mount: w_k = M w_0 + b with M a pure
rotation (singular values 1). Fits M (full 3x3) and b by least squares and
reports:
  - singular values of M (scale mismatch between chips shows as != 1),
  - M's nearest rotation against the calibration's mount (angle between them),
  - the offset b against the true at-rest biases from a still recording,
  - the time lag (cross-spectrum phase), and the fit residual.

Data: the 11/4 hand protocol (rigid window), the still recording, and the
ride (only the axes no joint turns, below 8 Hz) for comparison.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402

from experiments.pod_lag import lag  # noqa: E402
from idlml.bike.model import load_calibration  # noqa: E402
from idlml.data.cache import load  # noqa: E402
from idlml.geometry.so3 import quat_to_mat  # noqa: E402
import torch  # noqa: E402

CAL = load_calibration("gt-force")
MOUNT = {1: quat_to_mat(torch.tensor(CAL["fork_mount"], dtype=torch.float64)).numpy(),
         2: quat_to_mat(torch.tensor(CAL["rear_mount"], dtype=torch.float64)).numpy()}
NAMES = {1: "fork", 2: "rear"}


def fit_affine(w0, wk):
    """wk ~ M w0 + b, least squares."""
    X = np.column_stack([w0, np.ones(len(w0))])
    sol, *_ = np.linalg.lstsq(X, wk, rcond=None)
    M, b = sol[:3].T, sol[3]
    res = wk - X @ sol
    return M, b, res


def nearest_rotation(M):
    u, s, vt = np.linalg.svd(M)
    r = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
    return r, s


def angle_between(r1, r2):
    c = (np.trace(r1.T @ r2) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def report(label, w0, d, k, sl, fs):
    wk = d[f"gyr{k}"][sl].astype(np.float64)
    M, b, res = fit_affine(w0, wk)
    r, s = nearest_rotation(M)
    cal = MOUNT[k].T  # calibration mount maps pod -> frame; M maps frame -> pod
    print(f"  {NAMES[k]} pod, {label}:")
    print(f"    singular values of M  {np.round(s, 4)}   (rigid, same scale: 1, 1, 1)")
    print(f"    fitted rotation vs calibration mount: {angle_between(r, cal):.2f} deg")
    print(f"    offset b (deg/s)      {np.round(np.degrees(b), 2)}")
    print(f"    residual RMS (deg/s)  {np.round(np.degrees(res.std(0)), 2)}   frame rate RMS {np.round(np.degrees(w0.std(0)), 1)}")
    taus = [lag(w0[:, i], (wk @ r)[:, i], fs, coh_min=0.9)[0] for i in range(3)]
    print(f"    lag per frame axis (ms) {np.round(np.array(taus) * 1e3, 2)}")
    return M, b


def main():
    still = load("25544763")
    print("at-rest gyro biases, still recording (deg/s):")
    for k in range(3):
        g = still[f"gyr{k}"][int(1 * still["rate"]):]
        print(f"  IMU{k}  mean {np.round(np.degrees(g.mean(0)), 3)}   noise RMS {np.round(np.degrees(g.std(0)), 3)}")

    p = load("e75f5f92")
    fs = float(p["rate"])
    t = p["t"]
    sl = slice(int(np.searchsorted(t, 16.0)), int(np.searchsorted(t, 34.0)))
    w0 = p["gyr0"][sl].astype(np.float64)
    print(f"\nhand protocol, rigid window 16-34 s (frame rate RMS {np.round(np.degrees(w0.std(0)), 1)} deg/s):")
    for k in (1, 2):
        report("rigid window", w0, p, k, sl, fs)

    ride = load("173461a2")
    fs = float(ride["rate"])
    lp = butter(2, 8.0, fs=fs, output="sos")
    sl = slice(int(np.searchsorted(ride["t"], 600.0)), int(np.searchsorted(ride["t"], 5900.0)))
    w0 = sosfiltfilt(lp, ride["gyr0"][sl].astype(np.float64), axis=0)
    print("\nride 600-5900 s, below 8 Hz (joints and flex included, so only indicative):")
    for k in (1, 2):
        d2 = {f"gyr{k}": sosfiltfilt(lp, ride[f"gyr{k}"][sl].astype(np.float64), axis=0)}
        report("ride, < 8 Hz", w0, d2, k, slice(None), fs)


if __name__ == "__main__":
    main()
