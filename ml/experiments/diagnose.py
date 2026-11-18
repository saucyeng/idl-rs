"""Is a recovery error the solver's or the physics'? Score the objective at
the truth (true latents, frame rate, biases) and at the solver's answer."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, torch
from idlml.bike.model import Bike, load_calibration
from idlml.data.cache import load
from idlml.factors.objective import Objective, Weights
from idlml.sim.synth import SimConfig, synth
from experiments.recover import busy_window, run

class Fixed:
    levels = []
    def __init__(self, v): self.v = v
    def __call__(self): return self.v

dev = "cuda"
d = load("173461a2"); rate = float(d["rate"]); win = busy_window(d, 20.0, rate)
w0, f0 = d["gyr0"][win].astype(np.float64), d["acc0"][win].astype(np.float64)
bike = Bike(load_calibration("gt-force"), rear_rot_per_m=0.05)
sim = synth(bike, w0, f0, rate, SimConfig(), seed=1)
pods = {k: {kk: vv.to(dev) for kk, vv in v.items()} for k, v in sim["pods"].items()}
frame = {k: v.to(dev) for k, v in sim["frame"].items()}
from idlml.factors.objective import LatentSpec
from idlml.solve.batch import INIT, LEVELS, SCALE
n = sim["q"].shape[0]
spec = LatentSpec(n, 1.0 / rate, LEVELS, INIT, SCALE)
o = Objective(bike.to(dev), {k: s.to(dev) for k, s in sim["sensors"].items()}, frame, pods, rate, spec, Weights())
truth = tuple(x.to(dev) for x in (sim["q"], sim["qd"], sim["qdd"]))
spec.evaluate = lambda vars: truth
v = o.initial_vars()
v["frame_coef"] = sim["frame_coef"].to(dev); v["frame_bias"] = frame["gyro_bias"]
for k, sens in sim["sensors"].items():
    v[f"{k}.gyro_bias"] = sens.gyro_bias.detach().to(dev); v[f"{k}.accel_bias"] = sens.accel_bias.detach().to(dev)
with torch.no_grad():
    t = o.terms(v)
print("noise floor ~ %d (pod) %d (frame)" % (n * 12 / 2, n * 3 / 2))
print("truth   :", {k: round(float(v)) for k, v in t.items()})
q_est, info, sensors = run("exact", sim, bike, rate, Weights(), lambda b: None, dev)
print("solver  : final loss %d after %d L-BFGS rounds, %.0f s" % (info["loss"][-1], len(info["loss"]), info["seconds"]))
err = q_est - sim["q"].numpy()
print("mean error d %.1f mm, s %.1f mm, psi %.2f deg, b_fa %.3f deg" % (err[:,0].mean()*1e3, err[:,4].mean()*1e3, np.degrees(err[:,1].mean()), np.degrees(err[:,2].mean())))
print("rms error  d %.1f mm, s %.1f mm, psi %.2f deg, b_fa %.3f deg" % (err[:,0].std()*1e3, err[:,4].std()*1e3, np.degrees(err[:,1].std()), np.degrees(err[:,2].std())))
