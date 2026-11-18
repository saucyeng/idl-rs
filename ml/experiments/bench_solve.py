"""GPU blocked Cholesky against the CPU (LAPACK) banded solve, on real-size
normal equations from a synthetic window. Usage: bench_solve.py [samples]"""
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
from idlml.solve import batch, gn  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
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
o = Objective(bike.to(dev), sens, frame, pods, rate, LatentSpec(n, 1 / rate, batch.LEVELS, batch.INIT, batch.SCALE), Weights())
v = o.initial_vars()
lay = gn.Layout(o, v)


def timed(label, fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    print(f"{label:26s} {time.perf_counter() - t:7.3f} s", flush=True)
    return out


ab, B, C, gl, gg, _ = timed("normal equations", lambda: gn.normal_equations(o, v, lay))
print(f"unknowns {lay.n}, band width {ab.shape[0] - 1}")
xc = timed("cpu solve (LAPACK)", lambda: gn.solve_border(ab, B, C, gl, gg, 1e-3))[0]
for blk in (96, 192, 384, 768):
    xg = timed(f"gpu solve, block {blk}", lambda: gn.solve_border_gpu(ab, B, C, gl, gg, 1e-3, block=blk))[0]
    print(f"   max rel diff vs cpu {float(np.abs(xg.cpu().numpy() - xc).max() / np.abs(xc).max()):.1e}")
