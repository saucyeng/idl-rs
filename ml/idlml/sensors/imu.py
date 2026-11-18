"""What a real IMU does to the true motion, and the frame pod as an input.

`PodSensor` turns a pod's true readings into what it records: scale and
axis misalignment, bias, and its clock offset (a fractional delay, so the
offset itself is a learnable parameter). Noise and clipping are added only
when simulating; when estimating, clipped samples are masked out instead.
"""
import torch
from torch import nn

G = 9.80665


def delay(x, seconds, rate):
    """x [T, C] delayed by `seconds` (may be fractional, differentiable):
    y(t) = x(t - seconds). FFT phase shift on a zero-padded copy."""
    t = x.shape[0]
    n = 2 * t
    xf = torch.fft.rfft(x, n=n, dim=0)
    f = torch.fft.rfftfreq(n, d=1.0 / rate).to(x.device)
    ph = torch.exp(-2j * torch.pi * f * seconds)
    return torch.fft.irfft(xf * ph[:, None], n=n, dim=0)[:t]


class PodSensor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gyro_bias = nn.Parameter(torch.zeros(3))
        self.accel_bias = nn.Parameter(torch.zeros(3))
        self.gyro_mis = nn.Parameter(torch.zeros(3, 3))  # I + M: scale on the diagonal, misalignment off it
        self.accel_mis = nn.Parameter(torch.zeros(3, 3))
        self.clock = nn.Parameter(torch.zeros(()))  # s; recorded(t) = true(t - clock)

    def forward(self, gyro, accel, rate):
        eye = torch.eye(3, device=gyro.device)
        g = gyro @ (eye + self.gyro_mis).T + self.gyro_bias
        a = accel @ (eye + self.accel_mis).T + self.accel_bias
        if self.clock.requires_grad or self.clock.abs() > 0:
            ga = delay(torch.cat([g, a], 1), self.clock, rate)
            g, a = ga[:, :3], ga[:, 3:]
        return g, a


def clip_mask(x, full_scale, frac=0.98):
    """True where every axis of the sample is inside the sensor's range."""
    return (x.abs() < frac * full_scale).all(-1)


def frame_motion(w0, f0, rate, gyro_bias=None):
    """The frame pod as the forward model's input: bias-corrected rate,
    angular acceleration (central difference) and specific force."""
    w = w0 - (gyro_bias if gyro_bias is not None else 0)
    a = torch.zeros_like(w)
    a[1:-1] = (w[2:] - w[:-2]) * (rate / 2)
    a[0], a[-1] = a[1], a[-2]
    return w, a, f0
