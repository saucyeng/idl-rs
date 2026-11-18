"""Does torch.compile fuse the 36-direction derivative pass, and does it agree?"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
sys.argv = [sys.argv[0], "8000"]
import torch
exec(open(os.path.join(ROOT, "experiments", "profile_gn.py")).read().split("# --- inside one forward pass ---")[0].split("with torch.no_grad():\n    timed(\"loss")[0])
from torch.func import jvp, vmap
from idlml.solve import gn
lay = gn.Layout(o, v)
local = o.local_inputs(v); glob = tuple(v[k] for k in lay.globals); prim = tuple(p.detach() for p in local + glob)
tans = [torch.randn((36,) + p.shape, device=dev) for p in prim]
def dirs(*t): return vmap(lambda *tt: jvp(o.sample_residuals, prim, tt)[1])(*t)
with torch.no_grad():
    ref = dirs(*tans); torch.cuda.synchronize()
    t = time.perf_counter(); dirs(*tans); torch.cuda.synchronize(); print("eager     %.3f s" % (time.perf_counter() - t))
    try:
        c = torch.compile(dirs, dynamic=False)
        t = time.perf_counter(); out = c(*tans); torch.cuda.synchronize(); print("compile (first call) %.1f s" % (time.perf_counter() - t))
        for _ in range(2):
            t = time.perf_counter(); out = c(*tans); torch.cuda.synchronize(); print("compiled  %.3f s" % (time.perf_counter() - t))
        print("max rel diff", float((out - ref).abs().max() / ref.abs().max()))
    except Exception as e:
        print("compile failed:", type(e).__name__, str(e)[:400])
