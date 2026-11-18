import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from idlml.bike.model import Bike  # noqa: E402
from idlml.factors.objective import LatentSpec, Objective, Weights  # noqa: E402
from idlml.sensors.imu import PodSensor  # noqa: E402
from idlml.sim.synth import SimConfig, synth  # noqa: E402
from idlml.solve import batch, gn  # noqa: E402

CAL = {"up": [0.2, 0, 0.98], "forward": [-0.98, 0, 0.2], "steer_axis": [0.6, -0.04, 0.79],
       "fork_mount": [0.63, 0.67, -0.28, 0.26], "fork_lever": [-0.83, -0.04, -0.25],
       "rear_mount": [0.66, 0.68, 0.21, -0.23], "rear_lever": [0.34, -0.09, -0.41]}


def problem(n=600, dev="cuda" if torch.cuda.is_available() else "cpu", fit=()):
    rng = np.random.default_rng(3)
    rate = 400.0
    w0 = np.cumsum(rng.normal(0, 0.05, (n, 3)), 0)
    f0 = np.array([2.0, 0.0, 9.8]) + rng.normal(0, 3.0, (n, 3))
    bike = Bike(CAL, rear_rot_per_m=0.05)
    sim = synth(bike, w0, f0, rate, SimConfig(), seed=1)
    sens = {k: PodSensor().to(dev) for k in ("fork", "rear")}
    for s in sens.values():
        s.clock.requires_grad_(False)
    pods = {k: {kk: vv.to(dev) for kk, vv in v.items()} for k, v in sim["pods"].items()}
    frame = {k: v.to(dev) for k, v in sim["frame"].items()}
    obj = Objective(bike.to(dev), sens, frame, pods, rate, LatentSpec(n, 1 / rate, batch.LEVELS, batch.INIT, batch.SCALE), Weights(), fit=fit)
    vars = obj.initial_vars()
    g = torch.Generator(device="cpu").manual_seed(0)
    for k in list(vars):  # move off the starting point so nonlinear terms are exercised
        vars[k] = vars[k] + 0.3 * torch.randn(vars[k].shape, generator=g).to(dev) * (0.05 if k == "frame_coef" else 1.0)
    return obj, vars


import pytest  # noqa: E402


@pytest.mark.parametrize("fit", [(), ("steer_point", "bend_len", "rear_rot", "rear_dir", "rear_pivot")])
def test_fast_normal_equations_match_the_coloured_jacobian(fit):
    # Arrange
    obj, vars = problem(fit=fit)
    lay = gn.Layout(obj, vars)
    free = list(vars)
    J, r, cols = batch.jacobian(obj, vars, free)
    # our column order -> the coloured Jacobian's column order
    perm = np.empty(lay.n, dtype=np.int64)
    for gi, (var, row, _, _, _) in enumerate(lay.groups):
        kn = vars[var].shape[1]
        perm[lay.pos[gi].cpu().numpy()] = cols[var] + row * kn + np.arange(kn)
    o = lay.n_loc
    for k in lay.globals:
        perm[o:o + vars[k].numel()] = cols[k] + np.arange(vars[k].numel())
        o += vars[k].numel()

    # Act
    ab, B, C, gl, gg, loss = gn.normal_equations(obj, vars, lay)

    # Assert: gradient and products with a random vector agree
    g_ref = (J.T @ r)[perm]
    g_fast = np.concatenate([gl.cpu().numpy(), gg.cpu().numpy()])
    assert np.allclose(g_fast, g_ref, rtol=1e-3, atol=1e-3 * np.abs(g_ref).max())
    x = np.random.default_rng(1).normal(size=lay.n)
    xc = np.empty_like(x)
    xc[perm] = x
    hx_ref = (J.T @ (J @ xc))[perm]
    abn, Bn, Cn = ab.cpu().numpy(), B.cpu().numpy(), C.cpu().numpy()
    xl, xg = x[:lay.n_loc], x[lay.n_loc:]
    hl = abn[0] * xl
    for d in range(1, abn.shape[0]):
        hl[d:] += abn[d, :-d] * xl[:-d]
        hl[:-d] += abn[d, :-d] * xl[d:]
    hl += Bn @ xg
    hg = Bn.T @ xl + Cn @ xg
    hx = np.concatenate([hl, hg])
    assert np.allclose(hx, hx_ref, rtol=1e-3, atol=1e-3 * np.abs(hx_ref).max())
    assert np.isclose(loss, float(obj.loss(vars)), rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_multi_window_solver_with_one_window_matches_the_single_window_solver():
    # Arrange
    obj, _ = problem(n=800, fit=("bend_len",))
    # Act
    v1, i1 = gn.gauss_newton(obj, verbose=False)
    vm, im = gn.gauss_newton_multi([obj], verbose=False)
    # Assert
    assert abs(i1["loss"][-1] - im["loss"][-1]) < 1e-3 * i1["loss"][-1]
    q1, qm = obj.latents(v1)[0], obj.latents(vm[0])[0]
    assert (q1 - qm).abs().max() < 1e-3
