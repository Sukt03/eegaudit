"""Split-half identification without a trained read-out (smoke_splithalf.py), 10 subjects: raw voltage / random-init encoder / trained encoder.
Output: results/eegbits/splithalf_summary.json"""
import glob, json
from pathlib import Path
import numpy as np
R = Path("/path/to/eegaudit/results/eegbits"); rng = np.random.default_rng(0)
def ci(v):
    v = np.asarray(v, float); b = [rng.choice(v, len(v)).mean() for _ in range(2000)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
files = sorted(glob.glob(str(R / "splithalf_sub-*.json"))); out = {"n_subjects": len(files)}
for space in ["raw_voltage", "random_init_encoder", "trained_encoder"]:
    for k in (40, 1):
        for m in ("top1", "two_way", "bits"):
            vals = []
            for f in files:
                rows = json.load(open(f))["rows"]; vals.append(np.mean([r[m] for r in rows if r["space"] == space and r["k"] == k and r["condition"] == "eeg"]))
            out[f"{space}_k{k}_{m}"] = dict(mean=float(np.mean(vals)), ci95=ci(vals))
    pf = [np.mean([r["top1"] for r in json.load(open(f))["rows"] if r["space"] == space and r["condition"] == "shuffled_pair"]) for f in files]
    out[f"{space}_pair_top1"] = float(np.mean(pf))
    print(f"{space}: k=40 top-1 {100*out[f'{space}_k40_top1']['mean']:.1f}% [{100*out[f'{space}_k40_top1']['ci95'][0]:.1f},{100*out[f'{space}_k40_top1']['ci95'][1]:.1f}] 2-way {100*out[f'{space}_k40_two_way']['mean']:.1f}% bits {out[f'{space}_k40_bits']['mean']:.2f} | k=1 top-1 {100*out[f'{space}_k1_top1']['mean']:.1f}% | pair floor {100*out[f'{space}_pair_top1']:.1f}%")
(R / "splithalf_summary.json").write_text(json.dumps(out, indent=1))
