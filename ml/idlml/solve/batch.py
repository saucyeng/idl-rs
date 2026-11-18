"""Batch MAP solvers over an Objective's named variables.

`gauss_newton` is the main one: Levenberg-Marquardt with an exact sparse
Jacobian. Each residual depends only on spline coefficients near its own
time (plus a few global biases), so coefficients `C` knots apart never share
a residual. One forward-mode pass that perturbs every C-th coefficient of a
row at once therefore yields C-1 zeros and one exact column per residual,
and C passes per spline row recover the whole Jacobian (compressed Jacobian
by colouring). The normal equations are then solved directly, which fixes
the slow, nearly flat modes (travel enters through its second derivative)
in one step instead of thousands of first-order ones.

`lbfgs` is kept for comparison and for objectives without that locality.
"""
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch.func import jvp, vmap

from ..bike.model import B_FA, B_LAT, D, FRAME_ROT, FRAME_TRANS, PSI, R_ROLL, R_YAW, S

LEVELS = [(1, [B_FA, B_LAT, R_ROLL, R_YAW]), (2, [D, S] + FRAME_ROT + FRAME_TRANS), (4, [PSI])]
SCALE = (0.01, 0.1, 0.001, 0.001, 0.01, 0.001, 0.001) + (0.001,) * 6  # cm, ~6 deg, mrad.., cm, mrad.., frame flex mrad / mm
INIT = (0.045, 0.0, 0.0, 0.0, 0.045, 0.0, 0.0) + (0.0,) * 6
COLORS = 4  # a cubic B-spline coefficient touches 4 knot spans; priors span 3


def _spacing(obj, name):
    if name == "frame_coef":
        return 1
    if name.startswith("lvl"):
        return obj.spec.levels[int(name[3:])][0]
    return None  # a global


def jacobian(obj, vars, free, batch=8):
    """Sparse Jacobian of the flattened residual vector w.r.t. the free vars."""
    names = list(vars)
    primals = tuple(vars[k] for k in names)
    segs = obj.residuals(vars)
    layout, off = [], 0
    for s in segs:
        layout.append((off, s))
        off += s.r.numel()
    n_rows = off
    r0 = torch.cat([s.r.reshape(-1) for s in segs])

    def f(*vals):
        return torch.cat([s.r.reshape(-1) for s in obj.residuals(dict(zip(names, vals)))])

    cols, col_off = {}, 0
    for k in free:
        cols[k] = col_off
        col_off += vars[k].numel()
    R, Cc, V = [], [], []

    def push(rows, cs, vals):
        R.append(rows.cpu().numpy()); Cc.append(cs.cpu().numpy()); V.append(vals.detach().cpu().numpy())

    # every forward-mode pass needed: (var, row j, colour c) for spline rows,
    # (var, element e) for globals
    jobs = []
    for k in free:
        x = vars[k]
        if _spacing(obj, k) is None:
            jobs += [(k, None, e) for e in range(x.numel())]
        else:
            jobs += [(k, j, c) for j in range(x.shape[0]) for c in range(COLORS)]

    def tangent(job):
        k, j, c = job
        t = [torch.zeros_like(p) for p in primals]
        if j is None:
            t[names.index(k)].view(-1)[c] = 1.0
        else:
            t[names.index(k)][j, c::COLORS] = 1.0
        return t

    one = lambda *t: jvp(f, primals, t)[1]
    for b0 in range(0, len(jobs), batch):
        chunk = jobs[b0:b0 + batch]
        ts = [tangent(job) for job in chunk]
        stacked = tuple(torch.stack([t[i] for t in ts]) for i in range(len(primals)))
        jvs = vmap(one)(*stacked)
        for (k, j, c), jv in zip(chunk, jvs):
            if j is None:
                nz = torch.nonzero(jv).squeeze(1)
                push(nz, torch.full_like(nz, cols[k] + c), jv[nz])
                continue
            m, kn = _spacing(obj, k), vars[k].shape[1]
            for o, s in layout:
                v = jv[o:o + s.r.numel()]
                if s.r.dim() == 2:  # per sample: the colour-c coefficient among t//m .. t//m+3
                    tt, ch = s.r.shape
                    base = torch.arange(tt, device=v.device) // m
                    kk = base + (c - base) % COLORS
                    v = v.view(tt, ch)
                    rows = (torch.arange(tt, device=v.device)[:, None] * ch + torch.arange(ch, device=v.device)) + o
                    colk = (cols[k] + j * kn + kk)[:, None].expand(tt, ch)
                elif s.var == k and s.row == j:  # prior on this row: among i .. i+2
                    i = torch.arange(s.r.numel(), device=v.device)
                    kk = i + (c - i) % COLORS
                    rows, colk = i + o, cols[k] + j * kn + kk
                else:
                    continue
                nz = v != 0
                push(rows[nz], colk[nz], v[nz])
    J = sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(Cc))), shape=(n_rows, col_off))
    return J, r0.detach().cpu().numpy(), cols


def _time_order(obj, vars, free, cols):
    """Column order by the time each coefficient acts at, globals last: the
    normal matrix is then a narrow band plus a thin border, so a factorisation
    in this order (no fill-reducing reordering) stays sparse."""
    keys, idx = [], []
    for k in free:
        x = vars[k]
        m = _spacing(obj, k)
        if m is None:
            keys.append(np.full(x.numel(), np.inf))
        else:
            kn = x.shape[1]
            keys.append(np.tile((np.arange(kn) - 1.5) * m, x.shape[0]))
        idx.append(np.arange(cols[k], cols[k] + x.numel()))
    keys, idx = np.concatenate(keys), np.concatenate(idx)
    return idx[np.argsort(keys, kind="stable")]


def _solve_banded_border(H, damp, rhs, n_glob):
    """Solve (H + diag(damp)) x = rhs for symmetric positive definite H that is
    banded except for its last n_glob rows/columns: banded Cholesky on the
    band, Schur complement for the border."""
    from scipy.linalg import cho_factor, cho_solve, solveh_banded
    n = H.shape[0]
    nb = n - n_glob
    Hd = H + sp.diags(damp)
    A = Hd[:nb, :nb].tocoo()
    low = A.row >= A.col
    bw = int((A.row[low] - A.col[low]).max())
    ab = np.zeros((bw + 1, nb))
    ab[A.row[low] - A.col[low], A.col[low]] = A.data[low]
    B = Hd[:nb, nb:].toarray()
    C = Hd[nb:, nb:].toarray()
    sol = solveh_banded(ab, np.column_stack([rhs[:nb], B]), lower=True)
    ainv_r, ainv_b = sol[:, 0], sol[:, 1:]
    if n_glob:
        schur = C - B.T @ ainv_b
        x2 = cho_solve(cho_factor(schur), rhs[nb:] - B.T @ ainv_r)
        x1 = ainv_r - ainv_b @ x2
        return np.concatenate([x1, x2])
    return ainv_r


def _apply(vars, free, cols, delta):
    out = dict(vars)
    for k in free:
        x = vars[k]
        d = torch.as_tensor(delta[cols[k]:cols[k] + x.numel()], dtype=x.dtype, device=x.device).view_as(x)
        out[k] = x + d
    return out


def gauss_newton(obj, vars=None, free=None, iters=15, lam=1e-3, verbose=True):
    vars = vars or obj.initial_vars()
    free = free or list(vars)
    t0 = time.time()
    with torch.no_grad():
        loss = float(obj.loss(vars))
    hist = [loss]
    for it in range(iters):
        J, r, cols = jacobian(obj, vars, free)
        perm = _time_order(obj, vars, free, cols)
        n_glob = sum(vars[k].numel() for k in free if _spacing(obj, k) is None)
        Jp = J[:, perm]
        H = (Jp.T @ Jp).tocsr()
        g = Jp.T @ r
        dg = H.diagonal()
        while True:
            dp = _solve_banded_border(H, dg * lam + 1e-9, -g, n_glob)
            delta = np.empty_like(dp)
            delta[perm] = dp
            cand = _apply(vars, free, cols, delta)
            with torch.no_grad():
                new = float(obj.loss(cand))
            if new < loss:
                vars, lam = cand, max(lam / 3, 1e-7)
                break
            lam *= 4
            if lam > 1e6:
                break
        rel = (loss - new) / max(loss, 1e-12)
        loss = min(loss, new)
        hist.append(loss)
        if verbose:
            print(f"  GN {it:2d}  loss {loss:.6g}  lambda {lam:.1e}  nnz(J) {J.nnz/1e6:.1f}M  {time.time() - t0:.0f} s", flush=True)
        if rel < 1e-7 or lam > 1e6:
            break
    return vars, {"seconds": time.time() - t0, "loss": hist}


def lbfgs(obj, vars=None, free=None, iters=400, verbose=True):
    vars = {k: v.clone() for k, v in (vars or obj.initial_vars()).items()}
    free = free or list(vars)
    for k in free:
        vars[k].requires_grad_(True)
    opt = torch.optim.LBFGS([vars[k] for k in free], lr=1, max_iter=20, history_size=50, line_search_fn="strong_wolfe")
    t0, hist = time.time(), []

    def closure():
        opt.zero_grad()
        loss = obj.loss(vars)
        loss.backward()
        return loss

    for it in range(iters // 20):
        hist.append(float(opt.step(closure)))
        if verbose:
            print(f"  LBFGS {it * 20:4d}  loss {hist[-1]:.6g}", flush=True)
        if len(hist) > 2 and abs(hist[-2] - hist[-1]) < 1e-7 * abs(hist[-1]):
            break
    return {k: v.detach() for k, v in vars.items()}, {"seconds": time.time() - t0, "loss": hist}
