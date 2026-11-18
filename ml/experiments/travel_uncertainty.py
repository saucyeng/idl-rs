"""How well the sensors pin down travel at each timescale.

Solve the target window (geometry frozen at bikes/gt-force-fit.json), then
from the Gauss-Newton normal matrix H at the solution take the posterior
std of block-averaged travel, g^T H^-1 g with g the block mean of travel(t)
as a linear function of the spline coefficients, for blocks from 0.05 s
(about pointwise) to the whole window (the offset). The globals (biases)
are marginalised through the Schur complement.

That std assumes the whitened residuals are white. Real ones carry model
error with structure, so for each block length the script also reports the
residuals' block-mean variance against white (N x var of block means / var);
its square root scales the formal std to an honest one, roughly.

  --air   include the airtime anchor (default: sensors only)
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from idlml.bike.model import D, S  # noqa: E402
from idlml.data.cache import load  # noqa: E402
from idlml.data.real import FIT_FILE, apply_fit, gt_force, load_fit, sensor_cal, window_objective  # noqa: E402
from idlml.factors.objective import Weights  # noqa: E402
from idlml.solve.gn import Layout, _band_solve_gpu, gauss_newton, normal_equations  # noqa: E402
from idlml.traj.spline import _basis  # noqa: E402

BLOCKS = (0.05, 0.25, 1.0, 2.0, 5.0, None)  # s; None = the whole window


def block_functionals(obj, lay, latent, blk_n, most=60):
    """Columns g [n_loc, n_blocks]: block-mean travel (m) of `latent`, linear in the coefficients."""
    n = obj.spec.n
    for li, (m, ix) in enumerate(obj.spec.levels):
        if latent in ix:
            gi = next(g for g, (var, row, *_) in enumerate(lay.groups) if var == f"lvl{li}" and row == ix.index(latent))
            break
    m = lay.groups[gi][2]
    s = obj.spec.scale[latent]
    b = _basis(m, torch.float64, "cpu")[0].numpy()  # [4, m]
    t = np.arange(n)
    w = b[:, t % m].T * s  # [T, 4]
    cols = lay.pos[gi].cpu().numpy()[(t // m)[:, None] + np.arange(4)]  # [T, 4]
    nb = n // blk_n
    pick = np.unique(np.linspace(0, nb - 1, min(nb, most)).round().astype(int))  # at most `most`, evenly spread
    g = np.zeros((lay.n_loc, len(pick)))
    for j, k in enumerate(pick):
        sl = slice(k * blk_n, (k + 1) * blk_n)
        np.add.at(g[:, j], cols[sl].ravel(), w[sl].ravel() / blk_n)
    return g


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="173461a2")
    ap.add_argument("--start", type=float, default=5784.3)
    ap.add_argument("--end", type=float, default=5834.4)
    ap.add_argument("--air", action="store_true")
    ap.add_argument("--fit-file", default=FIT_FILE)
    a = ap.parse_args()
    dev = "cuda"
    d = load(a.session)
    rate = float(d["rate"])
    t = d["t"]
    sl = slice(int(np.searchsorted(t, a.start)), int(np.searchsorted(t, a.end)))
    w = Weights(gyro_sigma=0.05, accel_sigma=1.0, frame_accel_sigma=1.0)
    vals = load_fit(a.fit_file)["values"]
    bike = apply_fit(gt_force(), vals)
    obj = window_objective(d, sl, bike, w, dev=dev, air=a.air, sensor_cal=sensor_cal(vals))
    v, _ = gauss_newton(obj, verbose=False)
    lay = Layout(obj, v)
    ab, B, C, gl, gg, loss = normal_equations(obj, v, lay)
    print(f"solved, loss {loss:.0f}; airtime anchor {'on' if a.air else 'off'}")

    # residual whiteness: pod accel channels, block-mean variance against white
    with torch.no_grad():
        segs = {sg.name: sg.r.cpu().numpy() for sg in obj.residuals(v) if sg.r.dim() == 2}
    res = []
    for name in ("fork", "rear"):
        r, m = segs[f"pod.{name}"], obj.pods[name]["mask"].cpu().numpy()
        res.append(np.where(m[:, None], r, 0.0))
    res = np.concatenate(res, 1)  # [T, 12]: gyro and accel, whitened

    n = obj.spec.n
    print("block      front std mm      rear std mm     residual block-variance vs white (gyro, accel; median channel)")
    for blk in BLOCKS:
        bn = n if blk is None else int(blk * rate)
        cols = [block_functionals(obj, lay, lat, bn) for lat in (D, S)]
        g = torch.tensor(np.concatenate(cols, 1), device=dev)
        z = _band_solve_gpu(ab, torch.cat([g, B], 1), 0.0)
        zg, zb = z[:, :g.shape[1]], z[:, g.shape[1]:]
        schur = C + torch.eye(C.shape[0], device=dev, dtype=C.dtype) * 1e-9 - B.T @ zb
        u = B.T @ zg
        var = (g * zg).sum(0) + (u * torch.linalg.solve(schur, u)).sum(0)
        std = var.clamp(min=0).sqrt().cpu().numpy() * 1000
        nf = cols[0].shape[1]
        kb = n // bn
        bm = res[:kb * bn].reshape(kb, bn, -1).mean(1)
        ratio = bn * (bm**2).mean(0) / np.maximum((res**2).mean(0), 1e-12) if kb > 1 else np.full(res.shape[1], np.nan)
        gy, ac = np.median(ratio[[0, 1, 2, 6, 7, 8]]), np.median(ratio[[3, 4, 5, 9, 10, 11]])
        lab = "window" if blk is None else f"{blk:5.2f} s"
        print(f"{lab:8s}   median {np.median(std[:nf]):6.2f}        median {np.median(std[nf:]):6.2f}"
              f"      {gy:7.1f}  {ac:7.1f}" + ("" if kb > 1 else "   (one block: no estimate)"))


if __name__ == "__main__":
    main()
