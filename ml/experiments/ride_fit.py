"""Two stages on a real ride: learn the bike, then estimate with it frozen.

1. Bike fit: the busiest windows of the ride (most frame rotation, so the
   lever-arm terms are strongest), solved together with shared geometry and
   biases (gauss_newton_multi). Saved to bikes/gt-force-fit.json and reused
   on later runs unless --refit.
2. Estimate: freeze the fitted geometry and solve the target window
   (default Lower Broken Bomber) alone.

Reports: loss by term, travel percentiles, travel through each airborne
event, frame-pod flex size and spectrum, residual RMS per pod channel.

  --air          anchor travel near 0 in free fall (Weights.air_sigma)
  --flex-scale   multiply the frame-flex prior sigmas (smaller = stiffer)
  --tag          name for the saved latents / residuals in runs/real/
"""
import argparse
import datetime
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from idlml.bike.model import D, FITTABLE, FRAME_ROT, FRAME_TRANS, S  # noqa: E402
from idlml.data.cache import load  # noqa: E402
from idlml.data.real import (FIT_FILE, airborne, airtime_windows, apply_fit, busy_windows, default_fit, gt_force, load_fit,  # noqa: E402
                             save_fit, sensor_cal, window_objective)
from idlml.factors.objective import Weights  # noqa: E402
from idlml.solve.gn import gauss_newton, gauss_newton_multi  # noqa: E402


def travel_report(q, d, sl, w):
    rate = float(d["rate"])
    mask, events = airborne(d, sl)
    for i, nm, top in ((D, "front", w.travel_max[0]), (S, "rear", w.travel_max[1])):
        x = q[:, i] * 1000
        airm = f";  airborne median {np.median(x[mask]):.0f}" if mask.any() else ""
        print(f"    {nm} travel mm: 1% {np.percentile(x, 1):.0f}  median {np.median(x):.0f}  99% {np.percentile(x, 99):.0f}"
              f"  max {x.max():.0f} (stop {top * 1000:.0f}){airm}")
    t = d["t"][sl]
    print(f"    {len(events)} airborne events (>=0.12 s, frame <0.6 g, both pods quiet); travel mm front / rear at take-off, minimum in the air, at landing:")
    for i0, i1 in events:
        f, r = q[i0:i1, D] * 1000, q[i0:i1, S] * 1000
        print(f"      {t[i0]:8.2f} s  {(i1 - i0) / rate:4.2f} s   front {f[0]:4.0f} {f.min():4.0f} {f[-1]:4.0f}"
              f"   rear {r[0]:4.0f} {r.min():4.0f} {r[-1]:4.0f}")
    return mask


def flex_report(q, rate):
    """The frame pod's flex against the chassis: size and where its energy is."""
    from scipy.signal import welch
    names = ["rot x", "rot y", "rot z", "trans x", "trans y", "trans z"]
    print("    frame-pod flex vs chassis (rot deg, trans mm): rms, rms below 5 Hz, peak frequency")
    for j, i in enumerate(FRAME_ROT + FRAME_TRANS):
        x = q[:, i] * (np.degrees(1) if i in FRAME_ROT else 1000)
        f, p = welch(x, fs=rate, nperseg=1024)
        df = f[1] - f[0]
        slow = np.sqrt(p[(f > 0) & (f < 5)].sum() * df + x.mean() ** 2)
        band = (f > 5) & (f < 150)
        print(f"      {names[j]:8s} {x.std():7.4f}  {slow:7.4f}  {f[band][p[band].argmax()]:5.0f} Hz")


def main():
    # line-buffered output, so a run that is stopped part-way still leaves its log
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="173461a2")
    ap.add_argument("--start", type=float, default=5784.3)
    ap.add_argument("--end", type=float, default=5834.4)
    ap.add_argument("--fit-windows", type=int, default=4)
    ap.add_argument("--air-windows", type=int, default=2, help="with --air: this many of the fit windows chosen for airtime")
    ap.add_argument("--fit-seconds", type=float, default=30.0)
    ap.add_argument("--fit-iters", type=int, default=20)
    ap.add_argument("--refit", action="store_true", help="redo the bike fit even if a saved one exists")
    ap.add_argument("--fit-file", default=FIT_FILE)
    ap.add_argument("--air", action="store_true")
    ap.add_argument("--cal", action="store_true", help="bike fit also estimates pod mount rotations and accel scales")
    ap.add_argument("--flex-scale", type=float, default=1.0)
    ap.add_argument("--tag", default="target")
    a = ap.parse_args()
    dev = "cuda"
    d = load(a.session)
    rate = float(d["rate"])
    t = d["t"]
    target = slice(int(np.searchsorted(t, a.start)), int(np.searchsorted(t, a.end)))
    w = Weights(gyro_sigma=0.05, accel_sigma=1.0, frame_accel_sigma=1.0)
    w.frame_flex_rot_sigma *= a.flex_scale
    w.frame_flex_trans_sigma *= a.flex_scale
    bike = gt_force()

    # --- 1. the bike: saved fit, or learn it over the busiest windows (not the target) ---
    if os.path.exists(a.fit_file) and not a.refit:
        saved = load_fit(a.fit_file)
        apply_fit(bike, saved["values"])
        cal = sensor_cal(saved["values"])
        print(f"bike geometry from {os.path.basename(a.fit_file)} ({saved.get('note', '')}):")
        for k, v in saved["values"].items():
            print(f"    {k:12s} {np.round(v, 4).tolist()}")
    else:
        fit = default_fit(bike) + (["mount_err.fork", "mount_err.rear"] if a.cal else [])
        n_air = min(a.air_windows, a.fit_windows) if a.air else 0
        wins = airtime_windows(d, a.fit_seconds, n_air, exclude=target) if n_air else []
        wins = sorted(wins + busy_windows(d, a.fit_seconds, a.fit_windows - len(wins), exclude=[target] + wins),
                      key=lambda s: s.start)
        print(f"bike fit over {len(wins)} windows of {a.fit_seconds:.0f} s: " + ", ".join(f"{t[s.start]:.0f}s" for s in wins))
        objs = [window_objective(d, s, bike, w, fit=fit, dev=dev, air=a.air, accel_scale=a.cal) for s in wins]
        vl, info = gauss_newton_multi(objs, iters=a.fit_iters)
        print(f"  bike fit: {info['iterations']} iterations, {info['seconds']:.0f} s;"
              f" airtime per window: {[round(float(airborne(d, s)[0].sum()) / rate, 2) for s in wins]} s")
        fitted = {}
        for k in fit:
            v0 = objs[0].prior[f"bike.{k}"][0].detach().cpu().numpy().reshape(-1)
            v1 = vl[0][f"bike.{k}"].cpu().numpy().reshape(-1)
            fitted[k] = vl[0][f"bike.{k}"]
            print(f"    {k:12s} start {np.round(v0, 3).tolist()}  fitted {np.round(v1, 3).tolist()}  (prior sigma {FITTABLE[k]})")
        cal = {}
        if a.cal:
            for pod in ("fork", "rear"):
                cal[pod] = vl[0][f"{pod}.accel_scale"]
                fitted[f"sensor.{pod}.accel_scale"] = cal[pod]
                print(f"    {pod} accel scale error %  {np.round(cal[pod].cpu().numpy() * 100, 2).tolist()}")
            for pod in ("fork", "rear"):
                print(f"    {pod} mount correction deg {np.round(np.degrees(vl[0][f'bike.mount_err.{pod}'].cpu().numpy()), 2).tolist()}"
                      + ("  (part about the steer axis is unused)" if pod == "fork" else ""))
        del objs
        torch.cuda.empty_cache()
        apply_fit(bike, fitted)
        save_fit(fitted, {"note": f"fitted {datetime.date.today()}", "session": a.session,
                          "windows_s": [float(t[s.start]) for s in wins], "window_seconds": a.fit_seconds,
                          "iterations": info["iterations"], "flex_scale": a.flex_scale, "air_anchor": a.air, "cal": a.cal}, a.fit_file)
        print(f"  saved to {a.fit_file}")

    # --- 2. target window, geometry frozen ---
    print(f"\ntarget {a.start:.1f}-{a.end:.1f} s, geometry frozen; airtime anchor {'on' if a.air else 'off'},"
          f" frame-flex sigma x{a.flex_scale}; pod accel scales {'fitted' if cal else 'nominal'}:")
    obj = window_objective(d, target, bike, w, fit=(), dev=dev, air=a.air, sensor_cal=cal)
    v, info = gauss_newton(obj)
    print(f"  {info['iterations']} iterations, {info['seconds']:.1f} s for {(target.stop - target.start) / rate:.1f} s of data; loss {info['loss'][-1]:.0f}")
    with torch.no_grad():
        terms = {k: float(x) for k, x in obj.terms(v).items()}
        q = obj.latents(v)[0].cpu().numpy()
    print("    loss by term: " + ", ".join(f"{k} {x:.0f}" for k, x in terms.items()))
    air = travel_report(q, d, target, w)
    flex_report(q, rate)
    print("    biases:", {k: np.round(x.cpu().numpy(), 3).tolist() for k, x in v.items() if "bias" in k})
    # residuals per pod channel, in sensor units: what the model still misses
    with torch.no_grad():
        segs = {sg.name: sg.r.cpu().numpy() for sg in obj.residuals(v) if sg.r.dim() == 2}
    acc_sig = (w.accel_sigma**2 + w.frame_accel_sigma**2) ** 0.5
    for name in ("fork", "rear"):
        r = segs[f"pod.{name}"]
        m = obj.pods[name]["mask"].cpu().numpy()
        print(f"    {name} residual RMS: gyro {np.round(np.degrees(r[m, :3] * w.gyro_sigma).std(0), 1)} deg/s,"
              f" accel {np.round((r[m, 3:] * acc_sig).std(0), 2)} m/s^2")
    np.savez(os.path.join(ROOT, "runs", "real", f"ride_fit_{a.tag}.npz"), q=q, air_mask=air, terms=np.array(list(terms.items()), dtype=object),
             **{k: x for k, x in segs.items()}, t=d["t"][target],
             mask_fork=obj.pods["fork"]["mask"].cpu().numpy(), mask_rear=obj.pods["rear"]["mask"].cpu().numpy())


if __name__ == "__main__":
    main()
