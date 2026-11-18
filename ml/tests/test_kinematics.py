import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from idlml.bike.model import Bike, load_calibration, pod_readings  # noqa: E402
from idlml.geometry import so3  # noqa: E402
from idlml.sensors.imu import delay  # noqa: E402
from idlml.traj.spline import evaluate  # noqa: E402



@pytest.fixture(autouse=True)
def float64():
    """These tests compare against finite differences, so run in double
    precision, and restore the default for other test files."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)

GW = torch.tensor([0.0, 0.0, -9.80665], dtype=torch.float64)


def cal():
    try:
        return load_calibration("gt-force")
    except FileNotFoundError:
        return {"up": [0.2, 0, 0.98], "forward": [-0.98, 0, 0.2], "steer_axis": [0.6, -0.04, 0.79],
                "fork_mount": [0.63, 0.67, -0.28, 0.26], "fork_lever": [-0.83, -0.04, -0.25],
                "rear_mount": [0.66, 0.68, 0.21, -0.23], "rear_lever": [0.34, -0.09, -0.41]}


def latents(t):
    """Smooth latents and their exact first and second derivatives."""
    spec = [(0.05, 0.03, 5.0), (0.0, 0.2, 2.0), (0.0, 0.01, 7.0), (0.0, 0.005, 4.0), (0.1, 0.04, 3.0),
            (0.0, 0.004, 6.0), (0.0, 0.003, 8.0),
            (0.0, 0.003, 9.0), (0.0, 0.002, 11.0), (0.0, 0.004, 7.5),       # frame pod flex: rotation
            (0.0, 0.002, 10.0), (0.0, 0.0015, 12.0), (0.0, 0.001, 6.5)]     # and translation
    q = torch.stack([c + a * torch.sin(w * t) for c, a, w in spec], -1)
    qd = torch.stack([a * w * torch.cos(w * t) for c, a, w in spec], -1)
    qdd = torch.stack([-a * w * w * torch.sin(w * t) for c, a, w in spec], -1)
    return q, qd, qdd


def frame_pose(t):
    rv = torch.stack([0.3 * torch.sin(3 * t), 0.5 * t, 0.2 * torch.cos(t)], -1)
    x = 0.5 * torch.stack([torch.sin(t), t * t, torch.cos(2 * t)], -1)
    return so3.exp(rv), x


LINKAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bikes", "gt-force-linkage.json")


@pytest.mark.parametrize("rear", ["generic", "fourbar"])
def test_pod_readings_match_brute_force_world_motion(rear):
    # Arrange: a bike in double precision, an arbitrary frame trajectory in space
    if rear == "fourbar":
        if not os.path.exists(LINKAGE):
            pytest.skip("no linkage file")
        from idlml.bike.fourbar import FourBar, load_linkage
        bike = Bike(cal(), linkage=FourBar(load_linkage(LINKAGE)[0], 0.16)).double()
    else:
        bike = Bike(cal()).double()
    t0 = torch.tensor([0.3, 1.1, 2.0])
    h = 1e-4

    def world(t):
        r0, x0 = frame_pose(t)  # the rigid chassis
        q, _, _ = latents(t)
        out = {}
        poses = bike.poses(q)
        poses["imu0"] = bike.frame_pose(q)  # the frame pod, flexing on the chassis
        for name, (r, p) in poses.items():
            out[name] = (r0 @ r, x0 + (r0 @ p[..., None])[..., 0])
        return r0, x0, out

    (r0m, x0m, wm), (r0, x0, w), (r0p, x0p, wp) = world(t0 - h), world(t0), world(t0 + h)
    # frame: body rate, angular acceleration, specific force, all by central differences
    rate = lambda ra, rb: so3.vee(r0.transpose(-1, -2) @ (rb - ra) / (2 * h))
    w0 = rate(r0m, r0p)
    w0m = so3.vee(r0m.transpose(-1, -2) @ (r0 - frame_pose(t0 - 2 * h)[0]) / (2 * h))
    w0p = so3.vee(r0p.transpose(-1, -2) @ (frame_pose(t0 + 2 * h)[0] - r0) / (2 * h))
    a0 = (w0p - w0m) / (2 * h)
    q, qd, qdd = latents(t0)
    # the frame pod's own accelerometer: its flexing world motion, differenced
    ri, xi = w["imu0"]
    f_meas = (ri.transpose(-1, -2) @ ((wp["imu0"][1] - 2 * xi + wm["imu0"][1]) / h**2 - GW)[..., None])[..., 0]

    # Act: the model sees the chassis rate and the frame pod's own accelerometer
    bike_readings = pod_readings(bike, q, qd, qdd, w0, a0, f_meas)

    # Assert: the frame pod's gyro as it reads, given its flex
    gyro_i = so3.vee(ri.transpose(-1, -2) @ (wp["imu0"][0] - wm["imu0"][0]) / (2 * h))
    assert torch.allclose(bike_readings["frame"][0], gyro_i, atol=1e-5), bike_readings["frame"][0] - gyro_i

    # Assert: each pod's world motion, differenced, read in its own axes
    for name in ("fork", "rear"):
        rw, xw = w[name]
        gyro = so3.vee(rw.transpose(-1, -2) @ (wp[name][0] - wm[name][0]) / (2 * h))
        acc = (rw.transpose(-1, -2) @ ((wp[name][1] - 2 * xw + wm[name][1]) / h**2 - GW)[..., None])[..., 0]
        g_pred, a_pred = bike_readings[name]
        assert torch.allclose(g_pred, gyro, atol=1e-5), (name, g_pred - gyro)
        assert torch.allclose(a_pred, acc, atol=2e-3), (name, a_pred - acc)


def test_spline_derivatives_are_exact():
    # Arrange
    torch.manual_seed(0)
    coef = torch.randn(2, 40)
    dt, m = 0.001, 50
    n = 1800
    # Act
    q, qd, qdd = evaluate(coef, m, n, dt)
    # Assert: central differences agree up to their own error, which peaks at
    # the knots where a cubic spline's third derivative jumps (about 1% here)
    fd1 = (q[2:] - q[:-2]) / (2 * dt)
    fd2 = (qd[2:] - qd[:-2]) / (2 * dt)
    assert (qd[1:-1] - fd1).abs().max() < 1e-3 * qd.abs().max()
    assert (qdd[1:-1] - fd2).abs().max() < 3e-2 * qdd.abs().max()


def test_delay_shifts_a_sine_by_a_fraction_of_a_sample():
    # Arrange
    rate, tau, f = 400.0, 0.0007, 30.0
    t = torch.arange(4000) / rate
    x = torch.sin(2 * math.pi * f * t)[:, None]
    # Act
    y = delay(x, torch.tensor(tau), rate)
    # Assert (away from the ends)
    exp = torch.sin(2 * math.pi * f * (t - tau))[:, None]
    assert (y - exp)[200:-200].abs().max() < 1e-3


def test_fork_mount_correction_about_the_steer_axis_is_inert():
    # Arrange: a fork mount correction with a part about the steer axis
    # (which is exactly a steer offset, so it must not act)
    bike = Bike(cal()).double()
    q, _, _ = latents(torch.tensor([0.2, 0.9]))
    ap = bike.fork_mount.T @ so3.unit(bike.steer_axis)
    perp = torch.tensor([0.01, -0.02, 0.015])
    perp = perp - ap * (ap @ perp)
    with torch.no_grad():
        bike.mount_err["fork"].copy_(perp)
    r1, p1 = bike.front_pose(q)
    # Act: add a large rotation about the steer axis
    with torch.no_grad():
        bike.mount_err["fork"].copy_(perp + 0.3 * ap)
    r2, p2 = bike.front_pose(q)
    # Assert: same pose; and the perpendicular part does act
    assert torch.allclose(r1, r2, atol=1e-12) and torch.allclose(p1, p2)
    with torch.no_grad():
        bike.mount_err["fork"].zero_()
    assert (bike.front_pose(q)[0] - r1).abs().max() > 1e-3
