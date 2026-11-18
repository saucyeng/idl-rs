"""The bike as joints between the frame and each pod, and what each pod reads.

Everything is in IMU0's own axes (the frame pod), the same axes as the
calibrations in bikes/<id>.json. Latents, in this order (LATENTS):

  d      front travel along the steering axis, m (fork compression; 0 = the
         calibration pose)
  psi    steer angle about the steering axis, rad
  b_fa   fork fore-aft bend, rad (positive = axle moves rearward, as braking)
  b_lat  fork lateral bend, rad (positive = axle moves to IMU0's left of the fork plane)
  s      rear wheel travel, m
  r_roll rear member roll flex about the fore-aft axis, rad
  r_yaw  rear member yaw flex about the up axis, rad
  f_r*   frame-reference flex: rotation of the frame pod (IMU0) relative to
         the rigid chassis, rad, about IMU0's x, y, z
  f_t*   frame-reference flex: translation of IMU0 relative to the chassis, m

The chassis is the rigid body the pods' lever arms are measured on. The frame
pod sits on it through a small flexible pose (f_r*, f_t*): on the GT Force the
fork and rear residuals at 10-30 Hz were the same signal (coherence 0.85
laterally), so the frame pod does not read a rigid chassis there.

A pod's pose is a function of the latents only (`front_pose`, `rear_pose`).
Its velocity and acceleration relative to the frame come from nested
forward-mode AD along the latent trajectory, so a new joint or flex mode is
just a new pose function. The pod's readings then follow from rigid-body
transport with the chassis motion (w0, a0: the chassis rate latent and its
derivative; f0: the chassis specific force, recovered from the frame pod):

  w_pod  = R^T (w0 + w_rel)
  f_pod  = R^T (f0 + a0 x p + w0 x (w0 x p) + 2 w0 x p' + p'')

Gravity is common to f0 and f_pod, so no attitude is needed here.
"""
import json
import os

import torch
from torch import nn
from torch.func import jvp

from ..geometry import so3

LATENTS = ("d", "psi", "b_fa", "b_lat", "s", "r_roll", "r_yaw", "f_rx", "f_ry", "f_rz", "f_tx", "f_ty", "f_tz")
D, PSI, B_FA, B_LAT, S, R_ROLL, R_YAW, F_RX, F_RY, F_RZ, F_TX, F_TY, F_TZ = range(13)
FRAME_ROT, FRAME_TRANS = [F_RX, F_RY, F_RZ], [F_TX, F_TY, F_TZ]
NL = len(LATENTS)
# geometry the calibration does not hold: fitted, with a prior around its default (sigma in its units)
FITTABLE = {"steer_point": 0.05, "bend_len": 0.15, "rear_rot": 0.5, "rear_dir": 0.3, "rear_pivot": 0.1,
            "link_scale": 0.07,  # the 4-bar's photo scale (about +-6% uncertain)
            "mount_err.fork": 0.03, "mount_err.rear": 0.03}  # rad: pod mount corrections (calibration up-disagreement 1.2-1.5 deg);
# the fork's has no part about the steer axis (see front_pose)
LIBRARY = os.environ.get("IDL_LIBRARY", r"C:\Users\isaac\Documents\idl1-library\data")


def load_calibration(bike_id, library=LIBRARY):
    """The newest calibration in bikes/<bike_id>.json, as plain lists."""
    with open(os.path.join(library, "bikes", f"{bike_id}.json"), encoding="utf-8") as fh:
        cals = json.load(fh)["calibrations"]
    c = max(cals, key=lambda c: c["captured_at_ms"])
    pods = {p["imu_index"]: p for p in c["imus"]}
    return {
        "up": c["up"], "forward": c["forward"], "steer_axis": c.get("steer_axis"),
        "fork_mount": pods[1]["mount"], "fork_lever": pods[1]["lever_m"],
        "rear_mount": pods[2]["mount"], "rear_lever": pods[2]["lever_m"],
    }


class Bike(nn.Module):
    """Per-bike geometry. All learnable; the defaults that are not in a
    calibration (steering-axis point, bend length, rear path) are guesses to
    be fitted."""

    def __init__(self, cal, steer_offset=0.06, bend_len=0.45, rear_rot_per_m=0.25,
                 rear_pivot=None, rear_dir=None, linkage=None, link_scale=1.0):
        """`linkage`: a fourbar.FourBar for the rear; without one the rear
        uses the generic path (rear_rot, rear_dir, rear_pivot)."""
        super().__init__()
        self.linkage = linkage
        self.link_scale = nn.Parameter(torch.tensor(float(link_scale)))
        f = lambda v: torch.tensor(v, dtype=torch.float32)
        up = so3.unit(f(cal["up"]))
        fwd = so3.unit(f(cal["forward"]))
        axis = so3.unit(f(cal["steer_axis"])) if cal.get("steer_axis") else up
        self.register_buffer("up", up)
        self.register_buffer("fwd", fwd)
        self.steer_axis = nn.Parameter(axis)
        r1 = f(cal["fork_lever"])
        lat = so3.unit(torch.linalg.cross(axis, fwd))
        # a point on the steering axis, placed so the fork pod sits
        # `steer_offset` beside the axis (a guess until fitted)
        self.steer_point = nn.Parameter(r1 + lat * steer_offset)
        self.register_buffer("fork_mount", so3.quat_to_mat(f(cal["fork_mount"])))
        self.fork_lever = nn.Parameter(r1)
        self.bend_len = nn.Parameter(torch.tensor(bend_len))
        self.register_buffer("bend_k", torch.tensor(2.0 / 3.0))  # cantilever tip: deflection = 2/3 L angle
        r2 = f(cal["rear_lever"])
        self.register_buffer("rear_mount", so3.quat_to_mat(f(cal["rear_mount"])))
        self.rear_lever = nn.Parameter(r2)
        self.rear_axis = nn.Parameter(so3.unit(torch.linalg.cross(up, fwd)))  # lateral
        self.rear_pivot = nn.Parameter(f(rear_pivot) if rear_pivot is not None else 0.6 * r2)
        self.rear_rot = nn.Parameter(torch.tensor(rear_rot_per_m))
        self.rear_dir = nn.Parameter(f(rear_dir) if rear_dir is not None else up.clone())
        self.mount_err = nn.ParameterDict({  # small rotation-vector corrections to each pod's mount
            "fork": nn.Parameter(torch.zeros(3)), "rear": nn.Parameter(torch.zeros(3))})

    # --- poses: latents [T, NL] -> (R pod->frame [T,3,3], p in frame [T,3]) ---

    def front_pose(self, q):
        a = so3.unit(self.steer_axis)
        fa = so3.unit(self.fwd - a * (a @ self.fwd))  # fore-aft, normal to the axis
        lat = torch.linalg.cross(a, fa)  # left
        bfa, blat = q[:, B_FA], q[:, B_LAT]
        kl = self.bend_k * self.bend_len
        x = (self.fork_lever + q[:, D, None] * a
             - kl * bfa[:, None] * torch.linalg.cross(lat, a)
             - kl * blat[:, None] * torch.linalg.cross(fa, a))
        r_bend = so3.rot_axis(lat, bfa) @ so3.rot_axis(fa, blat)
        r_steer = so3.rot_axis(a, q[:, PSI])
        c = self.steer_point
        p = c + (r_steer @ (x - c)[..., None])[..., 0]
        # mount correction, less its part about the steer axis: that part is exactly a steer offset
        # (degenerate with the steer latent; freeing it let the mount turn 25 deg)
        ap = self.fork_mount.T @ a  # steer axis in pod axes
        e = self.mount_err["fork"]
        mount = self.fork_mount @ so3.exp(e - ap * (ap @ e))
        return r_steer @ r_bend @ mount, p

    def rear_pose(self, q):
        if self.linkage is not None:
            return self._rear_fourbar(q)
        return self._rear_generic(q)

    def _side_to_frame(self, v):
        """Side-plane vectors (x toward the front axle, y up) into IMU0 axes."""
        return v[..., :1] * self.fwd + v[..., 1:2] * self.up

    def _rear_fourbar(self, q):
        """Rear pod on the seatstay of a 4-bar: the seatstay turns by phi about
        the lateral axis while its Horst pivot moves, then roll/yaw flex about
        that pivot."""
        phi, h2, h0 = self.linkage.pose(q[:, S], self.link_scale, q)
        n = so3.unit(torch.linalg.cross(self.fwd, self.up))  # +phi in the side plane turns about fwd x up
        r = so3.rot_axis(n, phi)
        h2 = self._side_to_frame(h2)
        h0 = self._side_to_frame(h0)
        p = h2 + (r @ (self.rear_lever - h0)[..., None])[..., 0]
        r_flex = so3.rot_axis(self.fwd, q[:, R_ROLL]) @ so3.rot_axis(self.up, q[:, R_YAW])
        p = h2 + (r_flex @ (p - h2)[..., None])[..., 0]
        mount = self.rear_mount @ so3.exp(self.mount_err["rear"])
        return r_flex @ r @ mount, p

    def _rear_generic(self, q):
        """Generic rear path: rotation rear_rot (rad per m of travel) about a
        lateral axis through rear_pivot plus a translation along rear_dir, then
        roll and yaw flex of the rear member about the same pivot. A linkage
        model (4-bar) replaces the path part once its pivots are known."""
        s = q[:, S]
        r = so3.rot_axis(so3.unit(self.rear_axis), self.rear_rot * s)
        c = self.rear_pivot
        p = c + (r @ (self.rear_lever - c)[..., None])[..., 0] + s[:, None] * so3.unit(self.rear_dir)
        r_flex = so3.rot_axis(self.fwd, q[:, R_ROLL]) @ so3.rot_axis(self.up, q[:, R_YAW])
        p = c + (r_flex @ (p - c)[..., None])[..., 0]
        mount = self.rear_mount @ so3.exp(self.mount_err["rear"])
        return r_flex @ r @ mount, p

    def frame_pose(self, q):
        """The frame pod (IMU0) relative to the rigid chassis: (R IMU0 -> chassis, position)."""
        return so3.exp(q[:, FRAME_ROT]), q[:, FRAME_TRANS]

    def poses(self, q):
        return {"fork": self.front_pose(q), "rear": self.rear_pose(q)}

    def forward(self, q, qd, qdd, w0, a0, f_meas):
        """Readings of every pod (see pod_readings); a module call so that
        torch.func.functional_call can swap in fitted geometry."""
        return pod_readings(self, q, qd, qdd, w0, a0, f_meas)


def pod_motion(pose_fn, q, qd, qdd):
    """Pose and its exact first and second time derivatives along q(t)."""
    def flat(q):
        r, p = pose_fn(q)
        return torch.cat([r.reshape(-1, 9), p], 1)

    def vel(q, qd):
        return jvp(flat, (q,), (qd,))[1]

    x, xd = jvp(flat, (q,), (qd,))
    _, xdd = jvp(vel, (q, qd), (qd, qdd))
    sp = lambda v: (v[:, :9].reshape(-1, 3, 3), v[:, 9:])
    return sp(x), sp(xd), sp(xdd)


def pod_readings(bike, q, qd, qdd, w0, a0, f_meas):
    """True (error-free) readings, each in its pod's own axes.

    w0, a0 [T,3]: the rigid chassis's rate and angular acceleration (chassis
    axes = IMU0's nominal axes). f_meas [T,3]: the frame pod's measured
    specific force. Returns {"frame": (gyro, None), "fork": (gyro, accel),
    "rear": (gyro, accel)}: the frame pod's gyro as it should read given its
    flex, and the other pods' readings from the chassis motion."""
    cr = torch.linalg.cross
    (rf, tf), (rfd, tfd), (_, tfdd) = pod_motion(bike.frame_pose, q, qd, qdd)
    w_flex = so3.vee(rfd @ rf.transpose(-1, -2))
    rft = rf.transpose(-1, -2)
    out = {"frame": ((rft @ (w0 + w_flex)[..., None])[..., 0], None)}
    # the chassis specific force at its origin, from the frame pod's measurement
    f0 = (rf @ f_meas[..., None])[..., 0] - (cr(a0, tf) + cr(w0, cr(w0, tf)) + 2 * cr(w0, tfd) + tfdd)
    for name, fn in (("fork", bike.front_pose), ("rear", bike.rear_pose)):
        (r, p), (rd, pd), (_, pdd) = pod_motion(fn, q, qd, qdd)
        w_rel = so3.vee(rd @ r.transpose(-1, -2))
        f = f0 + cr(a0, p) + cr(w0, cr(w0, p)) + 2 * cr(w0, pd) + pdd
        rt = r.transpose(-1, -2)
        out[name] = ((rt @ (w0 + w_rel)[..., None])[..., 0], (rt @ f[..., None])[..., 0])
    return out
