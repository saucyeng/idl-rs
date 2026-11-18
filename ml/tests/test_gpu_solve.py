import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from idlml.solve import gn  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("n,bw,ng,block", [(1000, 75, 15, 192), (777, 30, 0, 64), (5000, 75, 25, 75)])
def test_gpu_block_cholesky_matches_lapack(n, bw, ng, block):
    # Arrange: a random SPD banded-plus-border system, like the normal equations
    rng = np.random.default_rng(0)
    m = n + ng
    J = np.zeros((3 * m, m))
    for r in range(3 * m):
        c = rng.integers(0, n)
        lo, hi = max(0, c - bw // 2), min(n, c + bw // 2 + 1)
        J[r, lo:hi] = rng.normal(size=hi - lo)
        if ng:
            J[r, n:] = rng.normal(size=ng) * 0.1
    H = J.T @ J + np.eye(m) * 1e-3
    ab = np.zeros((bw + 1, n))
    for d in range(bw + 1):
        ab[d, : n - d] = np.diagonal(H[:n, :n], -d)
    g = rng.normal(size=m)
    dev = "cuda"
    t = lambda x: torch.as_tensor(x, dtype=torch.float64, device=dev)
    args = (t(ab), t(H[:n, n:]), t(H[n:, n:]), t(g[:n]), t(g[n:]), 1e-4)
    # Act
    xl_c, xg_c = gn.solve_border(*args)
    xl_g, xg_g = gn.solve_border_gpu(*args, block=block)
    # Assert
    assert np.allclose(xl_g.cpu().numpy(), xl_c, rtol=1e-6, atol=1e-8 * np.abs(xl_c).max())
    if ng:
        assert np.allclose(xg_g.cpu().numpy(), xg_c, rtol=1e-6, atol=1e-8 * np.abs(xg_c).max())
