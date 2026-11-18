"""A real window: which flex modes are there, and what does the solver make of it?

1. Free relative rotation. Each pod's angular rate relative to the frame
   (calibrated mounts), steer removed for the fork pod, split into named axes,
   with spectra. A rigid joint puts everything on its joint axis; flex shows
   up on the others.
2. The batch solver on the real pods, with the GT calibration and default
   guesses for what the calibration lacks (steering-axis point, bend length,
   rear axle path). Reports residuals per channel against the noise the
   objective assumes, travel against the stops, and travel when airborne.

Default window: Lower Broken Bomber, Beacon 2026-11-04 (173461a2), lap 6,
5784.3-5834.4 s.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.signal import butter, sosfiltfilt, welch  # noqa: E402

from idlml.bike.model import B_FA, B_LAT, D, FITTABLE, LATENTS, PSI, R_ROLL, R_YAW, S, Bike, load_calibration  # noqa: E402
from idlml.data.cache import load  # noqa: E402
from idlml.factors.objective import LatentSpec, Objective, Weights  # noqa: E402
from idlml.geometry import so3  # noqa: E402
from idlml.sensors.imu import G, PodSensor  # noqa: E402
from idlml.solve.batch import INIT, LEVELS, SCALE  # noqa: E402
from idlml.solve.gn import gauss_newton  # noqa: E402

ACC_FS = 16 * G  # this hardware: 0.000488 g/LSB
GYRO_FS = np.radians(2000)


def rel_rotation(d, sl, bike, rate):
    """Pod-vs-frame relative rate on named axes (rad/s), steer removed."""
    w0 = d["gyr0"][sl].astype(np.float64)
    a = bike.steer_axis.detach().numpy()
    a = a / np.linalg.norm(a)
    fwd = bike.fwd.numpy()
    fa = fwd - a * (a @ fwd)
    fa /= np.linalg.norm(fa)
    lat = np.cross(a, fa)
    out = {}
    r1 = bike.fork_mount.numpy()
    w1 = d["gyr1"][sl].astype(np.float64) @ r1.T  # pod rate in frame axes, at zero steer
    # steer angle: integrate the rate about the axis (high-passed against drift), then undo it
    sr = (w1 - w0) @ a
    psi = np.cumsum(sr - sr.mean()) / rate
    psi -= sosfiltfilt(butter(2, 0.05, fs=rate, btype="low", output="sos"), psi)
    rot = so3.rot_axis(torch.tensor(a), torch.tensor(psi)).numpy()
    w1s = np.einsum("tij,tj->ti", rot, w1)
    rel = w1s - w0
    out["fork"] = {"steer axis (steer + twist)": rel @ a, "lateral axis (fore-aft bend)": rel @ lat,
                   "fore-aft axis (lateral bend)": rel @ fa}
    r2 = bike.rear_mount.numpy()
    rel2 = d["gyr2"][sl].astype(np.float64) @ r2.T - w0
    up = bike.up.numpy()
    lat0 = np.cross(up, fwd)
    lat0 /= np.linalg.norm(lat0)
    out["rear"] = {"lateral (swing)": rel2 @ lat0, "fore-aft (roll flex)": rel2 @ fwd, "up (yaw flex)": rel2 @ up}
    return out, psi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="173461a2")
    ap.add_argument("--start", type=float, default=5784.3)
    ap.add_argument("--end", type=float, default=5834.4)
    ap.add_argument("--accel-sigma", type=float, default=1.0)
    ap.add_argument("--gyro-sigma", type=float, default=0.05)
    ap.add_argument("--fit", nargs="*", default=None, help="bike geometry to estimate (default: all that apply)")
    ap.add_argument("--no-linkage", action="store_true", help="use the generic rear path even if a 4-bar file exists")
    a = ap.parse_args()
    dev = "cuda"
    d = load(a.session)
    rate = float(d["rate"])
    t = d["t"]
    sl = slice(int(np.searchsorted(t, a.start)), int(np.searchsorted(t, a.end)))
    n = sl.stop - sl.start
    out_dir = os.path.join(ROOT, "runs", "real", f"{a.session}_{int(a.start)}")
    os.makedirs(out_dir, exist_ok=True)
    link_path = os.path.join(ROOT, "bikes", "gt-force-linkage.json")
    linkage = None
    if os.path.exists(link_path) and not a.no_linkage:
        from idlml.bike.fourbar import FourBar, load_linkage
        linkage = FourBar(load_linkage(link_path)[0], 0.16)
        print("rear: 4-bar from", link_path)
    # the photo read the 230 mm shock's eye-to-eye as 215.8 mm: scale the linkage prior by that
    bike = Bike(load_calibration("gt-force"), linkage=linkage, link_scale=230 / 215.83 if linkage else 1.0)
    if a.fit is None:
        a.fit = ["steer_point", "bend_len"] + (["link_scale"] if linkage else ["rear_rot", "rear_dir", "rear_pivot"])

    # --- 1. free relative rotation ---
    rel, _ = rel_rotation(d, sl, bike, rate)
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    print("relative rotation rate, RMS deg/s (total / above 5 Hz):")
    hp = butter(2, 5.0, fs=rate, btype="high", output="sos")
    for i, (pod, comps) in enumerate(rel.items()):
        for name, x in comps.items():
            f, p = welch(np.degrees(x), fs=rate, nperseg=2048)
            ax[i].loglog(f[1:], p[1:], lw=0.8, label=name)
            print(f"  {pod:5s} {name:30s} {np.degrees(x).std():7.1f} / {np.degrees(sosfiltfilt(hp, x)).std():6.1f}")
        ax[i].set_title(f"{pod} pod rate relative to frame"); ax[i].set_ylabel("(deg/s)^2/Hz"); ax[i].legend()
    ax[1].set_xlabel("Hz")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "relative_rotation_spectra.png"), dpi=100); plt.close(fig)

    # --- 2. the solver on real pods ---
    tt = lambda k: torch.tensor(d[k][sl], dtype=torch.float32, device=dev)
    frame = {"gyro": tt("gyr0"), "accel": tt("acc0")}
    frame_ok = (frame["accel"].abs() < 0.98 * ACC_FS).all(-1)
    pods = {}
    for name, k in (("fork", 1), ("rear", 2)):
        g, acc = tt(f"gyr{k}"), tt(f"acc{k}")
        mask = (acc.abs() < 0.98 * ACC_FS).all(-1) & (g.abs() < 0.98 * GYRO_FS).all(-1) & frame_ok
        pods[name] = {"gyro": g, "accel": acc, "mask": mask}
        print(f"{name} pod: {100 * (1 - mask.float().mean()):.1f}% of samples masked (clipped, it or the frame)")
    sensors = {k: PodSensor().to(dev) for k in pods}
    for s in sensors.values():
        s.clock.requires_grad_(False)
    w = Weights(gyro_sigma=a.gyro_sigma, accel_sigma=a.accel_sigma, frame_accel_sigma=a.accel_sigma)
    obj = Objective(bike.to(dev), sensors, frame, pods, rate, LatentSpec(n, 1 / rate, LEVELS, INIT, SCALE), w, fit=a.fit)
    vars, info = gauss_newton(obj, verbose="terms")
    print(f"solved {n / rate:.1f} s of data in {info['seconds']:.1f} s; final loss {info['loss'][-1]:.0f}")
    with torch.no_grad():
        q = obj.latents(vars)[0].cpu().numpy()
        segs = {s.name: s.r.cpu().numpy() for s in obj.residuals(vars) if s.r.dim() == 2}
    tw = t[sl]
    # residuals in sensor units, per channel: RMS over unmasked samples
    print("residual RMS (sensor units) on unmasked samples:")
    for name in ("fork", "rear"):
        r = segs[f"pod.{name}"]
        m = pods[name]["mask"].cpu().numpy()
        acc_sig = (w.accel_sigma**2 + w.frame_accel_sigma**2) ** 0.5
        gy = np.degrees(r[m, :3] * w.gyro_sigma).std(0)
        ac = (r[m, 3:] * acc_sig).std(0)
        print(f"  {name}: gyro {np.round(gy, 1)} deg/s   accel {np.round(ac, 2)} m/s^2")
    print("frame gyro residual RMS deg/s", np.round(np.degrees(segs["frame"] * w.gyro_sigma).std(0), 2))
    # travel diagnostics
    fz = np.linalg.norm(d["acc0"][sl], axis=1)
    air = sosfiltfilt(butter(2, 10, fs=rate, output="sos"), fz) < 0.3 * G
    for i, nm, top in ((D, "front", w.travel_max[0]), (S, "rear", w.travel_max[1])):
        x = q[:, i] * 1000
        print(f"{nm} travel mm: min {x.min():.0f}  1% {np.percentile(x, 1):.0f}  median {np.median(x):.0f}  99% {np.percentile(x, 99):.0f}  max {x.max():.0f}"
              f"  (stops 0..{top * 1000:.0f});  airborne ({air.sum() / rate:.2f} s): median {np.median(x[air]) if air.any() else float('nan'):.0f}")
    print(f"steer deg: 1%..99% {np.degrees(np.percentile(q[:, PSI], 1)):.1f}..{np.degrees(np.percentile(q[:, PSI], 99)):.1f}")
    print(f"bend deg RMS: fore-aft {np.degrees(q[:, B_FA].std()):.3f}  lateral {np.degrees(q[:, B_LAT].std()):.3f}")
    print(f"rear flex deg RMS: roll {np.degrees(q[:, R_ROLL].std()):.3f}  yaw {np.degrees(q[:, R_YAW].std()):.3f}")
    print("biases:", {k: np.round(v.cpu().numpy(), 3).tolist() for k, v in vars.items() if "bias" in k})
    for k in a.fit:
        v0 = obj.prior[f"bike.{k}"][0].detach().cpu().numpy().reshape(-1)
        v1 = vars[f"bike.{k}"].cpu().numpy().reshape(-1)
        print(f"  {k:12s} start {np.round(v0, 3).tolist()}  fitted {np.round(v1, 3).tolist()}  (prior sigma {FITTABLE[k]})")
    np.savez(os.path.join(out_dir, "solution.npz"), t=tw, q=q, air=air)
    units = {"d": (1000, "mm"), "psi": (57.3, "deg"), "b_fa": (57.3, "deg"), "b_lat": (57.3, "deg"), "s": (1000, "mm"),
             "r_roll": (57.3, "deg"), "r_yaw": (57.3, "deg")}
    fig, ax = plt.subplots(len(LATENTS) + 2, 1, figsize=(16, 21), sharex=True)
    for i, lat in enumerate(LATENTS):
        k, u = units[lat]
        ax[i].plot(tw, q[:, i] * k, lw=0.6, color="k")
        ax[i].set_ylabel(f"{lat} ({u})")
    for i, top in ((D, w.travel_max[0]), (S, w.travel_max[1])):
        ax[i].axhline(0, color="r", lw=0.5); ax[i].axhline(top * 1000, color="r", lw=0.5)
        ax[i].fill_between(tw, ax[i].get_ylim()[0], ax[i].get_ylim()[1], where=air, color="tab:blue", alpha=0.2, lw=0)
    for j, name in ((len(LATENTS), "fork"), (len(LATENTS) + 1, "rear")):
        r = segs[f"pod.{name}"]
        ax[j].plot(tw, np.abs(r[:, 3:]).max(1), lw=0.4, label="accel |r|/sigma (max axis)")
        ax[j].plot(tw, np.abs(r[:, :3]).max(1), lw=0.4, label="gyro |r|/sigma (max axis)", alpha=0.7)
        ax[j].set_yscale("log"); ax[j].set_ylabel(f"{name} residual"); ax[j].legend(loc="upper right")
    ax[-1].set_xlabel("session s")
    fig.suptitle(f"{a.session} {a.start:.1f}-{a.end:.1f} s: solver on real pods (blue = airborne)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "solution.png"), dpi=90); plt.close(fig)
    print(out_dir)


if __name__ == "__main__":
    main()
