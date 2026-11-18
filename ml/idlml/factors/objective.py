"""The objective: whitened residuals whose half sum of squares is a negative
log-likelihood.

Everything the estimate depends on is a named tensor in `vars`:
  lvl<i>              spline coefficients of latent level i, [L_i, K_i]
  frame_coef          the frame's angular-rate spline, [3, n+3] (full rate)
  frame_bias          frame gyro bias, [3]
  <pod>.gyro_bias, <pod>.accel_bias      each pod's biases, [3]

`residuals(vars)` returns a list of `Segment`s, each a block of whitened
residuals that is either per-sample ([T, ch], row t depends only on
coefficients near sample t) or a prior on one variable's coefficients
([K-2], row i depends on coefficients i..i+2 of that variable). That
locality is what the sparse Gauss-Newton solver exploits; L-BFGS just sums
the squares.

Terms:
  frame    the frame gyro against the frame-rate latent
  pod      each pod's gyro and accelerometer against the forward model,
           clipped samples masked out
  bounds   one-sided walls keeping travel inside [0, travel_max]
  bend     a zero-mean prior on fork bending (it is a spring)
  smooth   a second-difference prior on each latent level's coefficients
  air      (if an airborne mask is given) travel near 0 while in free fall
"""
from dataclasses import dataclass

import torch
from torch.func import functional_call

from ..bike.model import B_FA, B_LAT, D, FITTABLE, FRAME_ROT, FRAME_TRANS, R_ROLL, R_YAW, S
from ..traj.spline import evaluate


@dataclass
class Weights:
    gyro_sigma: float = 0.01
    accel_sigma: float = 0.3
    frame_accel_sigma: float = 0.3  # the frame's specific force noise also lands on every pod
    travel_max: tuple = (0.17, 0.16)  # GT Force: 170 front, 160 rear
    wall_sigma: float = 0.002  # m past an end stop costs one sigma per 2 mm
    bend_sigma: float = 0.01  # rad, fork bending
    rear_flex_sigma: float = 0.01  # rad, rear member roll / yaw
    # frame pod vs chassis. Sized from the Beacon residuals: ~1.5 m/s^2 at 20 Hz is ~0.1 mm (a / w^2),
    # and the rotation-like part ~0.005 deg. Looser lets the flex's slow part trade with travel offsets.
    frame_flex_rot_sigma: float = 0.002  # rad
    frame_flex_trans_sigma: float = 0.0005  # m
    smooth: float = 1e-3
    # priors on the global unknowns, so they cannot soak up model error freely
    gyro_bias_sigma: float = 0.002  # rad/s, about the still recording's means (0.1 deg/s); looser lets the
    # frame-rate latent and all three biases trade a common offset (seen: 7.5 deg/s on Beacon)
    accel_bias_sigma: float = 0.5  # m/s^2 (covers the high-g channel's offset)
    # pod accelerometer scale per axis (with --cal). LSM6DSO32 datasheet: +-0.5% at component level
    # (DS rev 1, PDF p.25, Table 3); looser for solder stress and the frame pod's own error, since
    # only pod-vs-frame matters
    accel_scale_sigma: float = 0.01
    # airtime anchor: in free fall nothing loads the wheels, so both should sit near top-out
    air_sigma: float = 0.005  # m


@dataclass
class Segment:
    name: str
    r: torch.Tensor  # [T, ch] per-sample, or [K-2] prior
    var: str = None  # for a prior: the variable (and row) it acts on
    row: int = 0


@dataclass
class LatentSpec:
    """How latent trajectories are built from the lvl<i> coefficients."""
    n: int
    dt: float
    levels: list  # [(m, [latent indices])]
    init: tuple
    scale: tuple

    def evaluate(self, vars):
        dev = vars["lvl0"].device
        init = torch.tensor(self.init, device=dev, dtype=torch.float32)
        scale = torch.tensor(self.scale, device=dev, dtype=torch.float32)
        q = init.expand(self.n, -1)
        qd = torch.zeros(self.n, len(self.init), device=dev)
        qdd = torch.zeros_like(qd)
        for i, (m, ix) in enumerate(self.levels):
            ix_t = torch.tensor(ix, device=dev)
            a, b, c = evaluate(vars[f"lvl{i}"], m, self.n, self.dt)
            sc = scale[ix_t]
            q = q.index_add(1, ix_t, a * sc)
            qd = qd.index_add(1, ix_t, b * sc)
            qdd = qdd.index_add(1, ix_t, c * sc)
        return q, qd, qdd

    def zeros(self, device):
        from ..traj.spline import n_knots
        return {f"lvl{i}": torch.zeros(len(ix), n_knots(self.n, m), device=device) for i, (m, ix) in enumerate(self.levels)}


class Objective:
    """`fit` names the bike geometry to estimate along with the latents (keys
    of bike.model.FITTABLE); each gets a Gaussian prior around its current
    value. Sensor biases are always estimated, with priors from Weights."""

    def __init__(self, bike, sensors, frame, pods, rate, spec, w=Weights(), fit=(), bias_prior=None, air=None, accel_scale=False):
        """bias_prior: measured at-rest gyro biases {global name: tensor [3]}
        (e.g. from a still recording); they become the prior centres.
        air: bool [T], samples in free fall, where travel is anchored near 0.
        accel_scale: also estimate each pod accelerometer's per-axis scale error."""
        self.air = air
        self.bike, self.sensors, self.frame, self.pods = bike, sensors, frame, pods
        self.rate, self.spec, self.w = rate, spec, w
        self.fit = tuple(fit)
        self.globals = ["frame_bias"] + [f"{p}.{b}" for p in pods for b in ("gyro_bias", "accel_bias") + (("accel_scale",) if accel_scale else ())]             + [f"bike.{k}" for k in self.fit]
        # prior centre and sigma for each global
        self.prior = {"frame_bias": (0.0, w.gyro_bias_sigma)}
        for p in pods:
            self.prior[f"{p}.gyro_bias"] = (0.0, w.gyro_bias_sigma)
            self.prior[f"{p}.accel_bias"] = (0.0, w.accel_bias_sigma)
            self.prior[f"{p}.accel_scale"] = (0.0, w.accel_scale_sigma)
        for k in self.fit:
            self.prior[f"bike.{k}"] = (self._param(k).detach().clone(), FITTABLE[k])
        for k, v in (bias_prior or {}).items():
            self.prior[k] = (torch.as_tensor(v, dtype=torch.float32), self.prior[k][1])

    def _param(self, name):
        """A bike parameter by its dotted name (e.g. mount_err.fork)."""
        return dict(self.bike.named_parameters())[name]

    def initial_vars(self):
        dev = self.frame["gyro"].device
        g = self.frame["gyro"]
        v = self.spec.zeros(dev)
        v["frame_coef"] = torch.cat([g[:1], g, g[-1:].expand(2, 3)]).T.contiguous().clone()
        v["frame_bias"] = torch.zeros(3, device=dev)
        for pod in self.pods:
            v[f"{pod}.gyro_bias"] = torch.zeros(3, device=dev)
            v[f"{pod}.accel_bias"] = torch.zeros(3, device=dev)
            if f"{pod}.accel_scale" in self.globals:
                v[f"{pod}.accel_scale"] = torch.zeros(3, device=dev)
        for k, (c, _) in self.prior.items():  # start the biases at their prior centres
            if k in v and torch.is_tensor(c) and not k.startswith("bike."):
                v[k] = c.to(dev).clone()
        for k in self.fit:
            v[f"bike.{k}"] = self._param(k).detach().clone().reshape(-1)
        return v

    def latents(self, vars):
        return self.spec.evaluate(vars)

    def local_inputs(self, vars):
        """What each sample's residuals depend on, besides the globals:
        latents q, q', q'' [T,NL] and the frame's rate and angular acceleration [T,3]."""
        q, qd, qdd = self.spec.evaluate(vars)
        w0, a0, _ = evaluate(vars["frame_coef"], 1, self.spec.n, self.spec.dt)
        return q, qd, qdd, w0, a0

    def sample_segments(self, q, qd, qdd, w0, a0, glob):
        """Per-sample residual segments ([T, ch] each) from the local inputs and
        the globals (a dict keyed by self.globals)."""
        w = self.w
        f0 = self.frame["accel"]
        geom = {k: glob[f"bike.{k}"].reshape(self._param(k).shape) for k in self.fit}
        pred = functional_call(self.bike, geom, (q, qd, qdd, w0, a0, f0))
        frame_gyro = pred.pop("frame")[0]
        segs = [Segment("frame", (frame_gyro + glob["frame_bias"] - self.frame["gyro"]) / w.gyro_sigma)]
        acc_sigma = (w.accel_sigma**2 + w.frame_accel_sigma**2) ** 0.5
        for name, (g, a) in pred.items():
            sens = self.sensors[name]
            p = {"gyro_bias": glob[f"{name}.gyro_bias"], "accel_bias": glob[f"{name}.accel_bias"]}
            if f"{name}.accel_scale" in glob:
                p["accel_mis"] = torch.diag(glob[f"{name}.accel_scale"])
            gm, am = functional_call(sens, p, (g, a, self.rate))
            meas = self.pods[name]
            m = meas["mask"][:, None].float()
            segs.append(Segment(f"pod.{name}", torch.cat([(gm - meas["gyro"]) / w.gyro_sigma * m,
                                                          (am - meas["accel"]) / acc_sigma * m], 1)))
        walls = []
        for i, top in ((D, w.travel_max[0]), (S, w.travel_max[1])):
            walls += [torch.relu(-q[:, i]), torch.relu(q[:, i] - top)]
        segs.append(Segment("bounds", torch.stack(walls, 1) / w.wall_sigma))
        # softer near full extension (see Bike.bend_compliance)
        bsig = w.bend_sigma * self.bike.bend_compliance(q[:, D])[:, None]
        segs.append(Segment("bend", q[:, [B_FA, B_LAT]] / bsig))
        segs.append(Segment("rflex", q[:, [R_ROLL, R_YAW]] / w.rear_flex_sigma))
        segs.append(Segment("fflex", torch.cat([q[:, FRAME_ROT] / w.frame_flex_rot_sigma,
                                                q[:, FRAME_TRANS] / w.frame_flex_trans_sigma], 1)))
        if self.air is not None:
            segs.append(Segment("air", q[:, [D, S]] * self.air[:, None].float() / w.air_sigma))
        return segs

    def sample_residuals(self, q, qd, qdd, w0, a0, *glob):
        """All per-sample residuals as one [T, ch] tensor (globals positional, in self.globals order)."""
        segs = self.sample_segments(q, qd, qdd, w0, a0, dict(zip(self.globals, glob)))
        return torch.cat([s.r for s in segs], 1)

    def smooth_weight(self):
        return (self.w.smooth * 2) ** 0.5

    def prior_segments(self, vars):
        k = self.smooth_weight()
        segs = []
        for i in range(len(self.spec.levels)):
            c = vars[f"lvl{i}"]
            for j in range(c.shape[0]):
                segs.append(Segment("smooth", k * (c[j, 2:] - 2 * c[j, 1:-1] + c[j, :-2]), var=f"lvl{i}", row=j))
        return segs

    def global_prior(self, vars):
        """Whitened prior residuals of the globals, flat in self.globals order,
        and their per-element weights (1/sigma): r = (x - centre) / sigma."""
        rs, ws = [], []
        for k in self.globals:
            c, sig = self.prior[k]
            x = vars[k].reshape(-1)
            c = torch.as_tensor(c, dtype=x.dtype, device=x.device).reshape(-1) if torch.is_tensor(c) else torch.full_like(x, c)
            rs.append((x - c) / sig)
            ws.append(torch.full_like(x, 1.0 / sig))
        return torch.cat(rs), torch.cat(ws)

    def residuals(self, vars):
        glob = {k: vars[k] for k in self.globals}
        gp, _ = self.global_prior(vars)
        return (self.sample_segments(*self.local_inputs(vars), glob) + self.prior_segments(vars)
                + [Segment("gprior", gp, var="__globals__")])

    def terms(self, vars):
        out = {}
        for s in self.residuals(vars):
            out[s.name.split(".")[0]] = out.get(s.name.split(".")[0], 0.0) + 0.5 * (s.r**2).sum()
        return out

    def loss(self, vars):
        return sum(0.5 * (s.r**2).sum() for s in self.residuals(vars))
