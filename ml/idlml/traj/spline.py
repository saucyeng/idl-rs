"""Latent trajectories as uniform cubic B-splines with exact derivatives.

A `Spline` level holds coefficients for some latents at a knot spacing of
`m` samples; `MultiResSpline` sums levels (e.g. a coarse level for slow
travel and a per-sample level for bending), so slow and fast states each get
the resolution they need. Every trajectory returns (q, qd, qdd), each
[T, n_latent], which is the interface the forward model consumes.
"""
import math

import torch
from torch import nn


def _basis(m, dtype, device):
    u = torch.arange(m, dtype=dtype, device=device) / m
    b = torch.stack([(1 - u) ** 3, 3 * u**3 - 6 * u**2 + 4, -3 * u**3 + 3 * u**2 + 3 * u + 1, u**3]) / 6
    db = torch.stack([-3 * (1 - u) ** 2, 9 * u**2 - 12 * u, -9 * u**2 + 6 * u + 3, 3 * u**2]) / 6
    ddb = torch.stack([6 * (1 - u), 18 * u - 12, -18 * u + 6, 6 * u]) / 6
    return b, db, ddb


def n_knots(n_samples, m):
    return math.ceil(n_samples / m) + 3


def evaluate(coef, m, n_samples, dt):
    """coef [L, K] -> q, qd, qdd each [n_samples, L]."""
    b, db, ddb = _basis(m, coef.dtype, coef.device)
    win = coef.unfold(-1, 4, 1)  # [L, K-3, 4]
    h = m * dt
    out = []
    for basis, scale in ((b, 1.0), (db, 1 / h), (ddb, 1 / h**2)):
        v = torch.einsum("lki,im->lkm", win, basis).reshape(coef.shape[0], -1)[:, :n_samples]
        out.append((v * scale).T)
    return out


class Spline(nn.Module):
    def __init__(self, n_samples, n_latent, m, dt, init=None):
        super().__init__()
        self.n, self.m, self.dt = n_samples, m, dt
        k = n_knots(n_samples, m)
        c = torch.zeros(n_latent, k) if init is None else init.expand(n_latent, k).clone()
        self.coef = nn.Parameter(c)

    def forward(self):
        return evaluate(self.coef, self.m, self.n, self.dt)


class MultiResSpline(nn.Module):
    """Sum of spline levels. `levels` is a list of (m, latent_indices)."""

    def __init__(self, n_samples, n_latent, levels, dt, init=None, scale=None):
        super().__init__()
        self.n, self.n_latent, self.dt = n_samples, n_latent, dt
        self.idx = [torch.as_tensor(ix) for _, ix in levels]
        self.levels = nn.ModuleList([Spline(n_samples, len(ix), m, dt) for m, ix in levels])
        self.register_buffer("init", torch.zeros(n_latent) if init is None else torch.as_tensor(init, dtype=torch.float32))
        # coefficients are in units of `scale` per latent, so an optimiser sees
        # travel (cm) and bending (mrad) on similar footings
        self.register_buffer("scale", torch.ones(n_latent) if scale is None else torch.as_tensor(scale, dtype=torch.float32))

    def forward(self):
        dev = self.init.device
        q = self.init.expand(self.n, -1).clone()
        qd = torch.zeros(self.n, self.n_latent, device=dev)
        qdd = torch.zeros_like(qd)
        for ix, lvl in zip(self.idx, self.levels):
            sc = self.scale[ix.to(dev)]
            a, b, c = (v * sc for v in lvl())
            q = q.index_add(1, ix.to(dev), a)
            qd = qd.index_add(1, ix.to(dev), b)
            qdd = qdd.index_add(1, ix.to(dev), c)
        return q, qd, qdd


def fit_values(values, m):
    """Spline coefficients [L, K] whose curve approximates `values` [T, L]
    (least squares), for building known trajectories in tests and sims."""
    t, l = values.shape
    k = n_knots(t, m)
    eye = torch.eye(k, dtype=values.dtype)
    basis = evaluate(eye, m, t, 1.0)[0]  # [T, K]: column j is knot j's curve
    sol = torch.linalg.lstsq(basis, values).solution  # [K, L]
    return sol.T.contiguous()
