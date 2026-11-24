"""PE D5 width curve (, 23 Nov 2026). Same estimator/features/target as pe_specificity.py arm 'all'
(D2.pe_gain, nested calibration, raw X7 features) but over the full PCA-width grid D2.LS = 1..256,full, and WITHOUT the ocular-row restriction
(all X7 trials). Per width L: gain, circular-shift floor, noise floor, plus random-subspace-free reference not needed (PCA is fitted in-fold).
L90 = smallest L whose gain >= 0.9 x full-width gain. G1: L=64 and full gain are compared to saved pe_specificity 'all' (different rows: reported, not asserted)."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path("/path/to/eegaudit"); os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "experiments/ext")); sys.path.insert(0, str(ROOT))
import d2_pe_tasks as D2
ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default=""); ap.add_argument("--out", default="results/ext/pe_width"); a = ap.parse_args()
out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True); info = D2.dataset("ds003458"); LS = D2.LS
for s in ([int(x) for x in a.subjects.split(",")] if a.subjects else range(1, 24)):
    f = out / f"width_sub-{s:03d}.json"
    if f.exists(): continue
    t0 = time.time(); fz = np.load(ROOT / f"results/features/raw/sub-{s:03d}.npz"); tri = fz["tri"]; F = fz["F"].astype(np.float64); n = len(tri)
    y = info["qtil"](s, tri).astype(np.float64); Ys = np.column_stack([y] + [np.roll(y, round(n * q / 4)) for q in (1, 2, 3)])
    sc = D2.pe_gain(F, Ys, LS); nz = D2.pe_gain(D2.gnoise(F.shape, 1000 + s), Ys[:, :1], LS)
    rows = {str(L): dict(gain=float(sc[L][:, 0].mean()), shift_mean=float(sc[L][:, 1:].mean()), noise=float(nz[L][:, 0].mean())) for L in LS}
    json.dump(dict(participant=s, n_trials=int(n), LS=[str(L) for L in LS], rows=rows, runtime_s=round(time.time() - t0, 1)), open(f, "w"), indent=1)
    print(s, {k: round(v["gain"], 4) for k, v in rows.items()}, f"{time.time()-t0:.0f}s", flush=True)
