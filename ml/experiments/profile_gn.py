"""Where does a Gauss-Newton iteration spend its time? One 20 s synthetic window."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

from idlml.bike.model import Bike, load_calibration  # noqa: E402
from idlml.data.cache import load  # noqa: E402
from idlml.factors.objective import LatentSpec, Objective, Weights  # noqa: E402
from idlml.sensors.imu import PodSensor  # noqa: E402
from idlml.sim.synth import SimConfig, synth  # noqa: E402
from idlml.solve import batch  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
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
free = list(v)


def timed(label, fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    print(f"{label:28s} {time.perf_counter() - t:7.3f} s", flush=True)
    return out


with torch.no_grad():
    timed("loss (one forward)", lambda: o.loss(v))
J, r, cols = timed("jacobian (all passes + unpack)", lambda: batch.jacobian(o, v, free))
perm = timed("time order", lambda: batch._time_order(o, v, free, cols))
Jp = J[:, perm]
H = timed("J^T J", lambda: (Jp.T @ Jp).tocsr())
g = Jp.T @ r
ng = sum(v[k].numel() for k in free if batch._spacing(o, k) is None)
timed("banded solve", lambda: batch._solve_banded_border(H, H.diagonal() * 1e-3 + 1e-9, -g, ng))

# --- inside one forward pass ---
from idlml.bike.model import pod_readings, pod_motion  # noqa: E402
from idlml.traj.spline import evaluate  # noqa: E402
with torch.no_grad():
    q, qd, qdd = timed("  latents (spline eval)", lambda: o.spec.evaluate(v))
    w0, a0, _ = timed("  frame rate (spline eval)", lambda: evaluate(v["frame_coef"], 1, n, 1 / rate))
    timed("  front pose only", lambda: o.bike.front_pose(q))
    timed("  front pose + 2 nested jvps", lambda: pod_motion(o.bike.front_pose, q, qd, qdd))
    timed("  pod_readings (both pods)", lambda: pod_readings(o.bike, q, qd, qdd, w0, a0, frame["accel"]))
    timed("  all residual segments", lambda: o.residuals(v))
    print("  segments:", len(o.residuals(v)))

# --- the per-sample path ---
from idlml.solve import gn  # noqa: E402
lay = gn.Layout(o, v)
for rep in range(2):
    ab, B, C, gl, gg, _ = timed(f"fast normal equations ({rep})", lambda: gn.normal_equations(o, v, lay))
    timed(f"fast solve, CPU ({rep})", lambda: gn.solve_border(ab, B, C, gl, gg, 1e-3))
print("band width", ab.shape[0] - 1, "unknowns", lay.n)

# --- inside the fast path ---
from torch.func import jvp as _jvp, vmap as _vmap  # noqa: E402
with torch.no_grad():
    local = o.local_inputs(v)
    glob = tuple(v[k] for k in lay.globals)
    prim = local + glob
    for chunk in (12, 36):
        tans = [torch.randn((chunk,) + p.shape, device=dev) for p in prim]
        timed(f"  vmap jvp, {chunk} directions", lambda: _vmap(lambda *t: _jvp(o.sample_residuals, prim, t)[1])(*tans))
    timed("  one plain jvp", lambda: _jvp(o.sample_residuals, prim, tuple(torch.randn_like(p) for p in prim)))
    timed("  plain forward", lambda: o.sample_residuals(*prim))

# --- GPU blocked Cholesky vs LAPACK ---
for blk in (96, 192, 384, 768):
    timed(f"gpu solve, block {blk}", lambda: gn.solve_border_gpu(ab, B, C, gl, gg, 1e-3, block=blk))
timed("cpu solve (LAPACK)", lambda: gn.solve_border(ab, B, C, gl, gg, 1e-3))
xc = gn.solve_border(ab, B, C, gl, gg, 1e-3)[0]
xg_ = gn.solve_border_gpu(ab, B, C, gl, gg, 1e-3, block=384)[0].cpu().numpy()
print("max rel diff gpu vs cpu", float(np.abs(xg_ - xc).max() / np.abs(xc).max()))
