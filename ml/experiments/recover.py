"""Synthetic recovery: can the latents be recovered from the pods, and how
much does a wrong calibration hurt?

Real frame motion from a GT Force ride, invented latents, simulated pods. The
estimator then gets the pods plus the frame and solves for the latents, in
scenarios that differ only in what it gets wrong:

  exact      geometry exact; pod biases and frame gyro bias estimated
  lever5mm   the fork pod's lever arm off by 5 mm (not estimated)
  mount0.5   the fork pod's mount off by 0.5 deg (not estimated)
  clock0.5ms the fork pod's clock 0.5 ms late (not modelled)
  noise3x    sensor noise three times the baseline

Writes runs/recover/<scenario>.png and prints an error table.
"""
import argparse
import copy
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from idlml.bike.model import LATENTS, Bike, load_calibration  # noqa: E402
from idlml.data.cache import load  # noqa: E402
from idlml.factors.objective import LatentSpec, Objective, Weights  # noqa: E402
from idlml.sensors.imu import PodSensor  # noqa: E402
from idlml.sim.synth import SimConfig, synth  # noqa: E402
from idlml.solve.batch import INIT, LEVELS, SCALE, lbfgs  # noqa: E402
from idlml.solve.gn import gauss_newton  # noqa: E402

UNITS = {"d": (1000, "mm"), "psi": (57.2958, "deg"), "b_fa": (57.2958, "deg"), "b_lat": (57.2958, "deg"), "s": (1000, "mm"),
         "r_roll": (57.2958, "deg"), "r_yaw": (57.2958, "deg"),
         **{k: (57.2958, "deg") for k in ("f_rx", "f_ry", "f_rz")}, **{k: (1000, "mm") for k in ("f_tx", "f_ty", "f_tz")}}


def busy_window(d, seconds, rate):
    """The window with the most frame rotation (landings, braking bumps)."""
    w = np.linalg.norm(d["gyr0"], axis=1)
    n = int(seconds * rate)
    c = np.convolve(w, np.ones(n) / n, mode="valid")
    i = int(np.argmax(c[: len(c) - n]))
    return slice(i, i + n)


def lowpass(x, fc, rate):
    return sosfiltfilt(butter(2, fc, fs=rate, output="sos"), x, axis=0)


VERBOSE = True


def run(name, sim, bike_true, rate, weights, perturb, device, solver="gn"):
    bike = copy.deepcopy(bike_true).to(device)
    for p in bike.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        perturb(bike)
    sensors = {}
    for pod in ("fork", "rear"):
        s = PodSensor().to(device)
        s.clock.requires_grad_(False)
        sensors[pod] = s
    pods = {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in sim["pods"].items()}
    frame = {k: v.to(device) for k, v in sim["frame"].items()}
    n = sim["q"].shape[0]
    spec = LatentSpec(n, 1.0 / rate, LEVELS, INIT, SCALE)
    obj = Objective(bike, sensors, frame, pods, rate, spec, weights)
    vars, info = (gauss_newton if solver == "gn" else lbfgs)(obj, verbose=VERBOSE)
    with torch.no_grad():
        q = obj.latents(vars)[0].cpu().numpy()
    return q, info, vars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="173461a2")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--solver", default="gn", choices=["gn", "lbfgs"])
    ap.add_argument("--terms", action="store_true", help="print each objective term per iteration")
    a = ap.parse_args()
    global VERBOSE
    VERBOSE = "terms" if a.terms else True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = load(a.session)
    rate = float(d["rate"])
    win = busy_window(d, a.seconds, rate)
    w0, f0 = d["gyr0"][win].astype(np.float64), d["acc0"][win].astype(np.float64)
    bike_true = Bike(load_calibration("gt-force"), rear_rot_per_m=0.05)
    base = SimConfig()
    out_dir = os.path.join(ROOT, "runs", "recover")
    os.makedirs(out_dir, exist_ok=True)

    def lever(b):
        b.fork_lever.add_(torch.tensor([0.003, 0.003, -0.0029], device=b.fork_lever.device))

    def mount(b):
        b.mount_err["fork"].add_(torch.tensor([0.0, np.radians(0.5), 0.0], device=b.fork_lever.device, dtype=torch.float32))

    noop = lambda b: None
    scenarios = {
        "exact": (base, Weights(), noop),
        "lever5mm": (base, Weights(), lever),
        "mount0.5": (base, Weights(), mount),
        "clock0.5ms": (SimConfig(clock={"fork": 0.0005, "rear": 0.0}), Weights(), noop),
        "noise3x": (SimConfig(accel_noise=0.9, gyro_noise=0.03, frame_accel_noise=0.9, frame_gyro_noise=0.03),
                    Weights(gyro_sigma=0.03, accel_sigma=0.9), noop),
    }
    rows = []
    for name, (cfg, wts, pert) in scenarios.items():
        if a.only and name not in a.only:
            continue
        sim = synth(bike_true, w0, f0, rate, cfg, seed=1)
        q, info, _ = run(name, sim, bike_true, rate, wts, pert, device, a.solver)
        qt = sim["q"].numpy()
        err = q - qt
        row = {"scenario": name, "seconds": info["seconds"]}
        for i, lat in enumerate(LATENTS):
            k, _ = UNITS[lat]
            row[lat] = np.sqrt(np.mean(err[:, i] ** 2)) * k
            row[lat + "_slow"] = np.sqrt(np.mean(lowpass(err[:, i], 1.0, rate) ** 2)) * k
            row[lat + "_truth"] = np.std(qt[:, i]) * k
        rows.append(row)
        t = np.arange(len(qt)) / rate
        fig, ax = plt.subplots(len(LATENTS), 1, figsize=(15, 16), sharex=True)
        for i, lat in enumerate(LATENTS):
            k, u = UNITS[lat]
            ax[i].plot(t, qt[:, i] * k, lw=0.7, label="truth")
            ax[i].plot(t, q[:, i] * k, lw=0.7, label="estimate", alpha=0.8)
            ax[i].set_ylabel(f"{lat} ({u})")
        ax[0].legend(loc="upper right")
        ax[-1].set_xlabel("s")
        fig.suptitle(f"synthetic recovery: {name} ({info['seconds']:.0f} s solve)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=100)
        plt.close(fig)
        print(f"{name}: solved in {info['seconds']:.0f} s", flush=True)
    hdr = f"{'scenario':12s}" + "".join(f"{lat + ' (' + UNITS[lat][1] + ')':>22s}" for lat in LATENTS)
    print("\nRMS error, all / below 1 Hz   [truth spread]")
    print(hdr)
    for r in rows:
        print(f"{r['scenario']:12s}" + "".join(f"{r[l]:>9.2f} /{r[l + '_slow']:>6.2f} [{r[l + '_truth']:4.1f}]" for l in LATENTS))
    print(out_dir)


if __name__ == "__main__":
    main()
