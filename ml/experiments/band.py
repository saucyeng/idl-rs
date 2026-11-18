"""Solver debugging: size, nonzeros and bandwidth of the Gauss-Newton normal
matrix on a short synthetic window."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from idlml.bike.model import Bike, load_calibration  # noqa: E402
from idlml.data.cache import load  # noqa: E402
from idlml.factors.objective import LatentSpec, Objective, Weights  # noqa: E402
from idlml.sensors.imu import PodSensor  # noqa: E402
from idlml.sim.synth import SimConfig, synth  # noqa: E402
from idlml.solve.batch import INIT, LEVELS, SCALE, _time_order, jacobian  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
dev = "cuda"
d = load("173461a2")
rate = 400.0
sl = slice(1000000, 1000000 + n)
bike = Bike(load_calibration("gt-force"), rear_rot_per_m=0.05)
sim = synth(bike, d["gyr0"][sl].astype(float), d["acc0"][sl].astype(float), rate, SimConfig(), seed=1)
sens = {k: PodSensor().to(dev) for k in ("fork", "rear")}
for s in sens.values():
    s.clock.requires_grad_(False)
pods = {k: {kk: vv.to(dev) for kk, vv in v.items()} for k, v in sim["pods"].items()}
frame = {k: v.to(dev) for k, v in sim["frame"].items()}
o = Objective(bike.to(dev), sens, frame, pods, rate, LatentSpec(n, 1 / rate, LEVELS, INIT, SCALE), Weights())
v = o.initial_vars()
free = list(v)
t0 = time.time()
J, r, cols = jacobian(o, v, free)
print(f"jacobian {time.time() - t0:.1f} s, J {J.shape}, nnz per row {J.nnz / J.shape[0]:.1f}", flush=True)
perm = _time_order(o, v, free, cols)
H = (J[:, perm].T @ J[:, perm]).tocoo()
ng = sum(v[k].numel() for k in free if not (k.startswith("lvl") or k == "frame_coef"))
m = (H.row < H.shape[0] - ng) & (H.col < H.shape[0] - ng)
print(f"H {H.shape}, nnz {H.nnz}, globals {ng}, bandwidth without border {np.abs(H.row[m] - H.col[m]).max()}")
