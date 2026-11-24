"""Misspecification twins (methodology r6, EEGNet read-out, 8 donors x 3 twins per family): per family, mean observed/predicted
posterior-variance ratio of the learning rate and mean held-out gain (nats per trial). Output: results/method_pilots/r6_summary.json"""
import glob, json
import numpy as np
from pathlib import Path
R = Path("/path/to/eegaudit/results/method_pilots"); out = {}
for f in sorted(glob.glob(str(R / "r6_misspec_g*.json"))):
    d = json.load(open(f))
    for key, t in d["twins"].items():
        fam = t["family"]; vr = t["var_ratio"]; vr0 = vr[0] if isinstance(vr, list) else vr
        out.setdefault(fam, {"ratio": [], "pred": [], "gain": []}); out[fam]["ratio"].append(float(vr0)); out[fam]["pred"].append(float(t["pred_ratio_at_rho_fit"])); out[fam]["gain"].append(float(t["gain_nats_per_trial"]))
summ = {fam: dict(n_twins=len(v["ratio"]), ratio_mean=float(np.mean(v["ratio"])), ratio_sd=float(np.std(v["ratio"], ddof=1)), pred_mean=float(np.mean(v["pred"])), gain_mnats_mean=1000 * float(np.mean(v["gain"]))) for fam, v in out.items()}
json.dump(summ, open(R / "r6_summary.json", "w"), indent=1)
for fam, s in summ.items(): print(f"{fam:14s} n={s['n_twins']:2d} ratio {s['ratio_mean']:.2f} (pred {s['pred_mean']:.2f}) gain {s['gain_mnats_mean']:+.1f} mnats")
