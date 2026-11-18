"""Gauss-Newton normal equations assembled per sample, on the GPU.

Every per-sample residual r_t depends on NZ local quantities z_t (latents q,
q', q'' and the frame's rate and angular acceleration at t) and on 15 global
biases. So:

1. One vmap'd forward-mode call gives every sample's small Jacobians
   dr_t/dz_t [R x NZ] and dr_t/dglobals at once.
2. z_t is linear in the spline coefficients through fixed basis weights, so
   the chain rule to the 4 coefficients per spline row that touch sample t is
   a few multiplies: A_t [R x 32] over 8 spline rows x 4 coefficients.
3. A_t^T A_t, A_t^T r_t and the border terms are scatter-added straight
   into a banded matrix in time order (globals as a border), plus the
   coefficient priors, which are linear and added in closed form.

This replaces ~55 full forward-mode passes with one batched call, and the
GPU-to-CPU traffic with one copy of the band.
"""
import time

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve, solveh_banded
from torch.func import jvp, vmap

from ..bike.model import NL
from ..traj.spline import _basis

NZ = 3 * NL + 6  # q, q', q'' (NL each), w0(3), a0(3)


class Layout:
    """Column order: every spline coefficient by the time it acts at, then
    the globals. Built once per problem shape."""

    def __init__(self, obj, vars):
        spec = obj.spec
        self.groups = []  # (var, row, m, scale, (z columns), weights kind)
        for li, (m, ix) in enumerate(spec.levels):
            for j, lat in enumerate(ix):
                self.groups.append((f"lvl{li}", j, m, spec.scale[lat], (lat, NL + lat, 2 * NL + lat)))
        for x in range(3):
            self.groups.append(("frame_coef", x, 1, 1.0, (3 * NL + x, 3 * NL + 3 + x)))
        keys, owners = [], []
        for gi, (var, row, m, _, _) in enumerate(self.groups):
            kn = vars[var].shape[1]
            keys.append((np.arange(kn) - 1.5) * m)
            owners.append(np.stack([np.full(kn, gi), np.arange(kn)], 1))
        keys, owners = np.concatenate(keys), np.concatenate(owners)
        order = np.argsort(keys, kind="stable")
        self.n_loc = len(order)
        dev = vars["frame_coef"].device
        self.pos = []
        cursor = 0
        inv = np.empty(len(order), dtype=np.int64)
        inv[order] = np.arange(len(order))
        for gi, (var, row, m, _, _) in enumerate(self.groups):
            kn = vars[var].shape[1]
            self.pos.append(torch.as_tensor(inv[cursor:cursor + kn], device=dev))
            cursor += kn
        self.globals = list(obj.globals)
        self.n_glob = sum(vars[k].numel() for k in self.globals)
        self.n = self.n_loc + self.n_glob

    def apply(self, vars, delta_loc, delta_glob):
        out = dict(vars)
        for gi, (var, row, _, _, _) in enumerate(self.groups):
            if out[var] is vars[var]:
                out[var] = vars[var].clone()
            out[var][row] += delta_loc[self.pos[gi]].to(out[var].dtype)
        o = 0
        for k in self.globals:
            n = vars[k].numel()
            out[k] = vars[k] + delta_glob[o:o + n].view_as(vars[k]).to(vars[k].dtype)
            o += n
        return out


@torch.no_grad()
def normal_equations(obj, vars, lay, chunk=36, global_prior=True):
    """Band (lower, [bw+1, n_loc]), border B [n_loc, n_glob], C [n_glob, n_glob],
    gradient (local, global) and the loss, all float64 on the GPU."""
    t = obj.spec.n
    dt = obj.spec.dt
    dev = vars["frame_coef"].device
    local = obj.local_inputs(vars)
    glob = tuple(vars[k] for k in lay.globals)
    primals = local + glob
    f = obj.sample_residuals
    r = f(*primals)  # [T, R]
    n_dir = NZ + lay.n_glob
    # tangent directions: one per local quantity (set at every sample), one per global element
    widths = [p.shape[-1] for p in local] + [p.numel() for p in glob]
    starts = np.cumsum([0] + widths[:-1])
    jac = []
    for d0 in range(0, n_dir, chunk):
        ds = range(d0, min(d0 + chunk, n_dir))
        tans = [torch.zeros((len(ds),) + p.shape, device=dev) for p in primals]
        for b, d in enumerate(ds):
            pi = int(np.searchsorted(starts, d, side="right") - 1)
            e = d - starts[pi]
            if pi < len(local):
                tans[pi][b, :, e] = 1.0
            else:
                tans[pi][b].view(-1)[e] = 1.0
        jac.append(vmap(lambda *tt: jvp(f, primals, tt)[1])(*tans))
    jac = torch.cat(jac).permute(1, 2, 0).double()  # [T, R, n_dir]
    jz, jg = jac[..., :NZ], jac[..., NZ:]
    r64 = r.double()
    # chain to spline coefficients
    tt = torch.arange(t, device=dev)
    blocks, cols = [], []
    for gi, (var, row, m, s, zc) in enumerate(lay.groups):
        b, db, ddb = (x.double() for x in _basis(m, torch.float64, dev))
        jj, base = tt % m, tt // m
        h = m * dt
        w = [b[:, jj].T * s, db[:, jj].T * (s / h), ddb[:, jj].T * (s / h**2)]  # each [T, 4]
        a = sum(jz[:, :, c, None] * wk[:, None, :] for c, wk in zip(zc, w))  # [T, R, 4]
        blocks.append(a)
        cols.append(lay.pos[gi][base[:, None] + torch.arange(4, device=dev)])
    A = torch.cat(blocks, 2)  # [T, R, 32]
    col = torch.cat(cols, 1)  # [T, 32]
    # priors: k * (c_i - 2 c_{i+1} + c_{i+2}) on each latent spline row
    k = obj.smooth_weight()
    pw = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64, device=dev) * k
    prior_cols, prior_r = [], []
    for gi, (var, row, m, s, zc) in enumerate(lay.groups):
        if var == "frame_coef":
            continue
        c = vars[var][row].double()
        prior_r.append(k * (c[:-2] - 2 * c[1:-1] + c[2:]))
        i = torch.arange(c.numel() - 2, device=dev)
        prior_cols.append(lay.pos[gi][i[:, None] + torch.arange(3, device=dev)])
    pc = torch.cat(prior_cols)  # [P, 3]
    pr = torch.cat(prior_r)  # [P]
    bw = int(max((col.max(1).values - col.min(1).values).max(), (pc.max(1).values - pc.min(1).values).max()))
    n = lay.n_loc
    ab = torch.zeros(bw + 1, n, dtype=torch.float64, device=dev)

    def scatter_band(ci, cj, v):
        keep = ci >= cj
        ab.view(-1).index_add_(0, ((ci - cj) * n + cj)[keep], v[keep])

    m_t = torch.einsum("tri,trj->tij", A, A)
    scatter_band(col[:, :, None].expand_as(m_t).reshape(-1), col[:, None, :].expand_as(m_t).reshape(-1), m_t.reshape(-1))
    outer = (pw[:, None] * pw[None, :]).expand(pc.shape[0], 3, 3)
    scatter_band(pc[:, :, None].expand(-1, 3, 3).reshape(-1), pc[:, None, :].expand(-1, 3, 3).reshape(-1), outer.reshape(-1))
    g_loc = torch.zeros(n, dtype=torch.float64, device=dev)
    g_loc.index_add_(0, col.reshape(-1), torch.einsum("tri,tr->ti", A, r64).reshape(-1))
    g_loc.index_add_(0, pc.reshape(-1), (pw[None, :] * pr[:, None]).reshape(-1))
    B = torch.zeros(n, lay.n_glob, dtype=torch.float64, device=dev)
    B.index_add_(0, col.reshape(-1), torch.einsum("tri,trg->tig", A, jg).reshape(-1, lay.n_glob))
    C = torch.einsum("trg,trh->gh", jg, jg)
    g_glob = torch.einsum("trg,tr->g", jg, r64)
    # priors on the globals: r = w (x - centre), so J = diag(w)
    loss = 0.5 * (r64**2).sum() + 0.5 * (pr**2).sum()
    if global_prior:
        gp, gw = (x.double() for x in obj.global_prior(vars))
        C = C + torch.diag(gw**2)
        g_glob = g_glob + gw * gp
        loss = loss + 0.5 * (gp**2).sum()
    return ab, B, C, g_loc, g_glob, float(loss)


def _quad(ab, B, C, xl, xg):
    """x^T H x for the banded-plus-border H."""
    hl = ab[0] * xl
    for d in range(1, ab.shape[0]):
        hl[d:] += ab[d, :-d] * xl[:-d]
        hl[:-d] += ab[d, :-d] * xl[d:]
    return xl @ hl + 2 * xl @ (B @ xg) + xg @ (C @ xg)


def solve_border(ab, B, C, g_loc, g_glob, lam):
    """(H + lam diag(H)) x = -g for the banded-plus-border H, on the CPU
    (LAPACK banded Cholesky + Schur complement)."""
    ab = ab.cpu().numpy().copy()
    d = ab[0].copy()
    ab[0] += lam * d + 1e-9
    B, C = B.cpu().numpy(), C.cpu().numpy().copy()
    C[np.diag_indices_from(C)] += lam * np.diag(C) + 1e-9
    gl, gg = -g_loc.cpu().numpy(), -g_glob.cpu().numpy()
    sol = solveh_banded(ab, np.column_stack([gl, B]), lower=True, check_finite=False)
    ainv_r, ainv_b = sol[:, 0], sol[:, 1:]
    x2 = cho_solve(cho_factor(C - B.T @ ainv_b), gg - B.T @ ainv_r)
    return ainv_r - ainv_b @ x2, x2


def solve_band_gpu(ab, rhs_cols, lam, block=192):
    """(A + lam diag(A)) X = rhs_cols for the banded A, blocked Cholesky on the GPU."""
    return _band_solve_gpu(ab, rhs_cols, lam, block)


def solve_border_gpu(ab, B, C, g_loc, g_glob, lam, block=192):
    sol = _band_solve_gpu(ab, torch.cat([-g_loc[:, None], B], 1), lam, block)
    ainv_r, ainv_b = sol[:, 0], sol[:, 1:]
    Cd = C.clone()
    Cd.diagonal().add_(lam * C.diagonal() + 1e-9)
    schur = Cd - B.T @ ainv_b
    x2 = torch.linalg.solve(schur, -g_glob - B.T @ ainv_r)
    return ainv_r - ainv_b @ x2, x2


def _band_solve_gpu(ab, rhs_cols, lam, block=192):
    """As solve_border, on the GPU in float64. The band (bandwidth bw) is cut
    into dense blocks of size >= bw, so only neighbouring blocks couple and a
    blocked Cholesky (one small factorisation, triangular solve and product
    per block) solves it. Memory: the diagonal and sub-diagonal blocks only,
    each overwritten in place by its factor. The globals' border is handled
    by the Schur complement, as on the CPU."""
    dev = ab.device
    bw = ab.shape[0] - 1
    n = ab.shape[1]
    b = max(block, bw)
    nb = -(-n // b)
    N = nb * b
    D = torch.zeros(nb, b, b, dtype=torch.float64, device=dev)  # becomes the Cholesky factors
    L = torch.zeros(max(nb - 1, 1), b, b, dtype=torch.float64, device=dev)  # becomes W_k = L_k C_k^-T
    j = torch.arange(n, device=dev)
    for d in range(bw + 1):  # one band diagonal at a time: A[j+d, j] = ab[d, j]
        jj = j[: n - d]
        i = jj + d
        v = ab[d, : n - d]
        bi, bj = i // b, jj // b
        same = bi == bj
        D[bi[same], i[same] % b, jj[same] % b] = v[same]
        if d:
            D[bj[same], jj[same] % b, i[same] % b] = v[same]
            nxt = ~same
            L[bj[nxt], i[nxt] % b, jj[nxt] % b] = v[nxt]
    diag = torch.diagonal(D, dim1=1, dim2=2)
    diag.mul_(1 + lam).add_(1e-9)
    if N > n:  # padding: identity
        D[-1].diagonal()[n - (nb - 1) * b:] = 1.0
    rhs = torch.zeros(N, rhs_cols.shape[1], dtype=torch.float64, device=dev)
    rhs[:n] = rhs_cols
    rhs = rhs.view(nb, b, -1)
    tri = torch.linalg.solve_triangular
    # factor: C_k = chol(D_k - W_{k-1} W_{k-1}^T), W_k = L_k C_k^{-T}
    for k in range(nb):
        if k:
            D[k] -= L[k - 1] @ L[k - 1].T
        D[k] = torch.linalg.cholesky_ex(D[k])[0]  # _ex: no per-block error check, which would sync the GPU every block
        if k < nb - 1:
            L[k] = tri(D[k], L[k].T, upper=False).T
    # forward: y_k = C_k^{-1} (r_k - W_{k-1} y_{k-1}), in place in rhs
    for k in range(nb):
        if k:
            rhs[k] -= L[k - 1] @ rhs[k - 1]
        rhs[k] = tri(D[k], rhs[k], upper=False)
    # back: x_k = C_k^{-T} (y_k - W_k^T x_{k+1})
    for k in range(nb - 1, -1, -1):
        if k < nb - 1:
            rhs[k] -= L[k].T @ rhs[k + 1]
        rhs[k] = tri(D[k].T, rhs[k], upper=True)
    if not torch.isfinite(D).all():  # one check for the whole factorisation
        raise torch.linalg.LinAlgError("blocked Cholesky failed: the system is not positive definite")
    return rhs.reshape(N, -1)[:n]


def gauss_newton(obj, vars=None, iters=20, lam=1e-10, verbose=True, tol=1.0):
    """Levenberg-Marquardt. Stops when an iteration lowers the loss by less
    than `tol`. The loss is a chi-square / 2, so changes below ~1 are not
    statistically meaningful. A step that fails at heavy damping also means
    converged.

    Damping starts near zero. The problem is close to linear (measured
    actual/predicted reduction 1.00), and Marquardt damping lam*diag(H) brakes
    hardest on exactly the slow travel modes, whose eigenvalues are tiny next
    to their diagonal entries. At lam=1e-3 every iteration only halved the
    loss; nearly undamped, the solve takes a few iterations."""
    vars = vars or obj.initial_vars()
    lay = Layout(obj, vars)
    t0 = time.time()
    hist = []
    with torch.no_grad():
        loss = float(obj.loss(vars))
    hist.append(loss)
    rho = float("nan")
    timing = {"normal equations": 0.0, "solve": 0.0, "loss evaluations": 0.0}

    def clock(key, fn, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timing[key] += time.perf_counter() - t
        return out

    for it in range(iters):
        ab, B, C, gl, gg, _ = clock("normal equations", normal_equations, obj, vars, lay)
        while True:
            xl, xg = clock("solve", solve_border_gpu if ab.is_cuda else solve_border, ab, B, C, gl, gg, lam)
            dev = gl.device
            cand = lay.apply(vars, torch.as_tensor(xl, device=dev), torch.as_tensor(xg, device=dev))
            with torch.no_grad():
                new = clock("loss evaluations", lambda: float(obj.loss(cand)))
            if new < loss:
                # predicted reduction of the linear model: -g.x - x.H.x/2 (H x from the band + border)
                xl_t = torch.as_tensor(xl, device=dev, dtype=torch.float64)
                xg_t = torch.as_tensor(xg, device=dev, dtype=torch.float64)
                pred = -float(gl @ xl_t + gg @ xg_t) - 0.5 * float(_quad(ab, B, C, xl_t, xg_t))
                rho = (loss - new) / max(pred, 1e-12)
                vars, lam = cand, max(lam / 10, 1e-12)
                break
            lam *= 10
            if lam > 1e2:
                break
        gain = loss - new
        loss = min(loss, new)
        hist.append(loss)
        if verbose:
            extra = ""
            if verbose == "terms":
                with torch.no_grad():
                    extra = "  " + "  ".join(f"{k} {float(v):.4g}" for k, v in obj.terms(vars).items())
            print(f"  GN {it:2d}  loss {loss:.6g}  rho {rho:.2f}  lambda {lam:.1e}  {time.time() - t0:.1f} s{extra}", flush=True)
        if gain < tol or lam > 1e2:
            break
    if verbose:
        print("  time: " + ", ".join(f"{k} {v:.1f} s" for k, v in timing.items()), flush=True)
    return vars, {"seconds": time.time() - t0, "loss": hist, "timing": timing, "iterations": it + 1}


def gauss_newton_multi(objs, vars_list=None, iters=20, lam=1e-10, verbose=True, tol=1.0):
    """Levenberg-Marquardt over several windows that share their globals
    (sensor biases and fitted bike geometry); latents are per window. Each
    window's band is factored on its own; the shared unknowns come from the
    summed Schur complements. Used to learn the bike from several windows at
    once, before per-window solves with the geometry frozen."""
    vars_list = vars_list or [o.initial_vars() for o in objs]
    lays = [Layout(o, v) for o, v in zip(objs, vars_list)]
    glob = objs[0].globals
    assert all(o.globals == glob for o in objs), "windows must share the same globals"

    def share(vl):  # every window sees window 0's globals
        return [{**v, **{k: vl[0][k] for k in glob}} for v in vl]

    def total_loss(vl):
        with torch.no_grad():
            gp, _ = objs[0].global_prior(vl[0])
            # each window's loss includes the global prior once; count it once overall
            return sum(float(o.loss(v)) for o, v in zip(objs, vl)) - (len(objs) - 1) * 0.5 * float((gp.double() ** 2).sum())

    vars_list = share(vars_list)
    t0 = time.time()
    loss = total_loss(vars_list)
    hist = [loss]
    rho = float("nan")
    for it in range(iters):
        nes = [normal_equations(o, v, l, global_prior=(w == 0)) for w, (o, v, l) in enumerate(zip(objs, vars_list, lays))]
        while True:
            dev = nes[0][0].device
            parts = []
            schur, rhs = None, None
            for ab, B, C, gl, gg, _ in nes:
                sol = _band_solve_gpu(ab, torch.cat([-gl[:, None], B], 1), lam)
                ar, abm = sol[:, 0], sol[:, 1:]
                Cd = C.clone()
                Cd.diagonal().add_(lam * C.diagonal())
                sc = Cd - B.T @ abm
                rh = -gg - B.T @ ar
                schur = sc if schur is None else schur + sc
                rhs = rh if rhs is None else rhs + rh
                parts.append((ar, abm))
            schur.diagonal().add_(1e-9)
            x2 = torch.linalg.solve(schur, rhs)
            cand = []
            for (ar, abm), v, l in zip(parts, vars_list, lays):
                cand.append(l.apply(v, ar - abm @ x2, x2))
            cand = share(cand)
            new = total_loss(cand)
            if new < loss:
                vars_list, lam = cand, max(lam / 10, 1e-12)
                break
            lam *= 10
            if lam > 1e2:
                break
        gain = loss - new
        loss = min(loss, new)
        hist.append(loss)
        if verbose:
            print(f"  GN* {it:2d}  loss {loss:.7g}  lambda {lam:.1e}  {time.time() - t0:.1f} s", flush=True)
        if gain < tol or lam > 1e2:
            break
    return vars_list, {"seconds": time.time() - t0, "loss": hist, "iterations": it + 1}
