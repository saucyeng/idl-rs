"""Rotation helpers, batched over any leading dimensions. Matrices map a
body's axes into its parent's (v_parent = R @ v_body)."""
import torch


def skew(v):
    z = torch.zeros_like(v[..., 0])
    x, y, w = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack([
        torch.stack([z, -w, y], -1),
        torch.stack([w, z, -x], -1),
        torch.stack([-y, x, z], -1),
    ], -2)


def vee(m):
    """Inverse of skew, taken from the skew-symmetric part of m."""
    a = 0.5 * (m - m.transpose(-1, -2))
    return torch.stack([a[..., 2, 1], a[..., 0, 2], a[..., 1, 0]], -1)


def rot_axis(axis, angle):
    """Rotation by `angle` (...) about the unit `axis` (3 or ...x3), Rodrigues.
    Smooth in angle at zero, so it is safe under forward- and reverse-mode AD."""
    axis = axis.expand(*angle.shape, 3) if axis.dim() == 1 else axis
    k = skew(axis)
    s = torch.sin(angle)[..., None, None]
    c = torch.cos(angle)[..., None, None]
    eye = torch.eye(3, dtype=angle.dtype, device=angle.device)
    return eye + s * k + (1 - c) * (k @ k)


def exp(w):
    """Rotation vector -> matrix (small-angle safe)."""
    th2 = (w * w).sum(-1)
    # clamp, not "+ 1e-30": adding a Python float promotes forward-mode tangents to float64
    safe = torch.clamp(th2, min=1e-30)
    th = torch.sqrt(safe)
    small = th2 < 1e-12
    a = torch.where(small, 1 - th2 / 6, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / safe)
    k = skew(w)
    eye = torch.eye(3, dtype=w.dtype, device=w.device)
    return eye + a[..., None, None] * k + b[..., None, None] * (k @ k)


def quat_to_mat(q):
    """Unit quaternion [w, x, y, z] -> matrix (the convention of bikes/*.json)."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def unit(v):
    return v / v.norm(dim=-1, keepdim=True)
