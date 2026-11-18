"""Why doesn't torch.compile capture the derivative pass? Graph-break report."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
sys.argv = [sys.argv[0], "2000"]
import torch, torch._dynamo as dynamo
exec(open(os.path.join(ROOT, "experiments", "profile_gn.py")).read().split("# --- inside one forward pass ---")[0].split("with torch.no_grad():\n    timed(\"loss")[0])
from torch.func import jvp, vmap
from idlml.solve import gn
lay = gn.Layout(o, v)
prim = tuple(p.detach() for p in o.local_inputs(v) + tuple(v[k] for k in lay.globals))
tans = [torch.randn((36,) + p.shape, device=dev) for p in prim]
def dirs(*t): return vmap(lambda *tt: jvp(o.sample_residuals, prim, tt)[1])(*t)
def fwd(*p): return o.sample_residuals(*p)
with torch.no_grad():
    for name, fn, args in (("derivative pass", dirs, tans), ("plain forward", fwd, prim)):
        e = dynamo.explain(fn)(*args)
        print(f"== {name}: graphs {e.graph_count}, breaks {e.graph_break_count}, ops captured {e.op_count}")
        for b in e.break_reasons[:6]:
            print("   ", str(b.reason)[:200])
