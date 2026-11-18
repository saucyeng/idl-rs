"""A planar 4-bar rear end (e.g. a Horst link): chainstay on the main pivot,
rocker on the frame, seatstay joining them and carrying the axle and the rear
brake caliper (and its pod).

Points come from `ml/tools/linkage.html` in the bike's side plane (metres,
origin at the rear axle, x toward the front axle, y up). The rear travel
latent is the axle's vertical travel. The chainstay angle for a given travel
comes from a polynomial fitted to the closed-form linkage (exact to
micro-radians), and the seatstay pose from the closed-form circle
intersection, so the pose is smooth and its time derivatives exact under AD.

`scale` scales the whole linkage about IMU0, for a photo whose scale is off
(the pixel scale comes from the wheelbase, and perspective can bend it a few
percent). Uniform scaling leaves the linkage's angles as they are and
stretches travel, so theta(s) = theta_unit(s / scale).
"""
import json

import numpy as np
import torch

POINTS = ("main_pivot", "horst", "ss_rocker", "rocker_frame")


def load_linkage(path):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return {k: np.array(v, dtype=np.float64) for k, v in d["points_m"].items()}, d


def _rot2(v, t):
    c, s = np.cos(t), np.sin(t)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def closure_np(pts, theta, sign):
    """Numpy closed form: chainstay angle -> (Horst pivot, rocker pivot, seatstay angle, axle)."""
    m, h, s, r = (pts[k] for k in POINTS)
    h2 = m + _rot2(h - m, theta)
    s2 = _intersect_np(h2, np.linalg.norm(s - h), r, np.linalg.norm(s - r), sign)
    phi = np.arctan2(*(s2 - h2)[..., ::-1].T) - np.arctan2(*(s - h)[::-1])
    axle = h2 + _rot2(-h, phi)
    return h2, s2, phi, axle


def _intersect_np(c0, r0, c1, r1, sign):
    d = c1 - c0
    dn = np.linalg.norm(d, axis=-1)
    a = (r0**2 - r1**2 + dn**2) / (2 * dn)
    hh = np.sqrt(np.maximum(r0**2 - a**2, 0))
    u = d / dn[..., None]
    mid = c0 + a[..., None] * u
    perp = np.stack([-u[..., 1], u[..., 0]], -1)
    return mid + sign * hh[..., None] * perp


class FourBar:
    """Precomputes the travel -> chainstay angle polynomial and holds the
    linkage in side-plane coordinates relative to IMU0."""

    def __init__(self, pts, travel_max, degree=9):
        self.pts = {k: pts[k] for k in POINTS}
        self.imu0 = pts["imu0"]
        m, h, s, r = (self.pts[k] for k in POINTS)
        # the branch of the circle intersection that the drawn linkage is on
        cand = [_intersect_np(h, np.linalg.norm(s - h), r, np.linalg.norm(s - r), sg) for sg in (1.0, -1.0)]
        self.sign = 1.0 if np.linalg.norm(cand[0] - s) < np.linalg.norm(cand[1] - s) else -1.0
        # which way of turning the chainstay lifts the axle
        up = closure_np(self.pts, np.array([1e-3]), self.sign)[3][0, 1]
        self.dir = 1.0 if up > 0 else -1.0
        th = self.dir * np.linspace(0, 0.6, 6001)
        travel = closure_np(self.pts, th, self.sign)[3][:, 1]
        use = travel <= travel_max * 1.3
        if travel[use].max() < travel_max:
            raise ValueError(f"linkage locks at {travel[use].max() * 1000:.0f} mm, before {travel_max * 1000:.0f} mm")
        self.coef = np.polynomial.polynomial.polyfit(travel[use], th[use], degree)
        fit_err = np.abs(np.polynomial.polynomial.polyval(travel[use], self.coef) - th[use]).max()
        assert fit_err < 1e-5, f"travel -> angle fit error {fit_err:.2e} rad"
        self.travel_max = travel_max

    def side_torch(self, like):
        """Side-plane points relative to IMU0, as tensors."""
        t = lambda v: torch.as_tensor(v - self.imu0, dtype=like.dtype, device=like.device)
        return {k: t(v) for k, v in self.pts.items()} | {"axle": t(np.zeros(2))}

    def pose(self, s, scale, like):
        """Vertical travel s [T] -> (seatstay angle phi [T], Horst pivot [T,2],
        rest Horst pivot [2]); side plane, relative to IMU0, scaled."""
        p = self.side_torch(like)
        m, h, sr, r = (p[k] * scale for k in POINTS)
        c = torch.as_tensor(self.coef, dtype=like.dtype, device=like.device)
        u = s / scale
        th = sum(c[i] * u**i for i in range(len(c)))
        ct, st = torch.cos(th), torch.sin(th)
        hm = h - m
        h2 = m + torch.stack([ct * hm[0] - st * hm[1], st * hm[0] + ct * hm[1]], -1)
        r0, r1 = (sr - h).norm(), (sr - r).norm()
        d = r - h2
        dn = d.norm(dim=-1)
        a = (r0**2 - r1**2 + dn**2) / (2 * dn)
        hh = torch.sqrt(torch.clamp(r0**2 - a**2, min=1e-12))
        un = d / dn[:, None]
        s2 = h2 + a[:, None] * un + self.sign * hh[:, None] * torch.stack([-un[:, 1], un[:, 0]], -1)
        v, v0 = s2 - h2, sr - h
        phi = torch.atan2(v[:, 1], v[:, 0]) - torch.atan2(v0[1], v0[0])
        return phi, h2, h
