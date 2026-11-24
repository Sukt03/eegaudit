"""EEG vs MEG on the same DINOv2 target with the same linear read-out (run_meg_probe.py): per modality, mean and 95% CI over subjects of
yield (Wolpaw bits, soft bits, 2-way, top-1) at k = all repetitions and k = 1, controls, and the raw split-half ceiling.
Output: results/eegbits/modprobe_summary.json"""
import glob, json
from pathlib import Path
import numpy as np
R = Path("/path/to/eegaudit/results/eegbits"); rng = np.random.default_rng(0)
def ci(v):
    v = np.asarray(v, float)
    if len(v) < 2: return [float(v.mean()), float(v.mean())]
    b = [rng.choice(v, len(v)).mean() for _ in range(2000)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
out = {}
for mod in ["eeg", "meg"]:
    files = sorted(glob.glob(str(R / f"modprobe_{mod}_sub-*.json")))
    if not files: continue
    res = {"n_subjects": len(files)}
    for cond, kname in [("eeg", "kall"), ("noise", "kall"), ("shuffled_pair", "kall"), ("splithalf_raw", "half"), ("splithalf_raw_pair", "half")]:
        for m in ("top1", "two_way", "bits", "soft_bits"):
            vals = []
            for f in files:
                d = json.load(open(f)); rows = [r for r in d["rows"] if r["condition"] == cond and (r["k"] == d["n_test_reps"] if kname == "kall" else True)]
                if rows and m in rows[0]: vals.append(np.mean([r[m] for r in rows]))
            if vals: res[f"{cond}_{m}"] = dict(mean=float(np.mean(vals)), ci95=ci(vals))
    k1 = [np.mean([r["bits"] for r in json.load(open(f))["rows"] if r["condition"] == "eeg" and r["k"] == 1]) for f in files]
    res["eeg_k1_bits"] = dict(mean=float(np.mean(k1)), ci95=ci(k1)); res["n_test_reps"] = json.load(open(files[0]))["n_test_reps"]; res["feature_dim"] = json.load(open(files[0]))["feature_dim"]
    out[mod] = res
    print(f"{mod}: n={len(files)} D={res['feature_dim']} k={res['n_test_reps']}: 2-way {100*res['eeg_two_way']['mean']:.1f}% top-1 {100*res['eeg_top1']['mean']:.1f}% bits {res['eeg_bits']['mean']:.2f} soft {res.get('eeg_soft_bits',{}).get('mean',float('nan')):.2f} | noise 2-way {100*res['noise_two_way']['mean']:.1f}% pair {100*res['shuffled_pair_two_way']['mean']:.1f}% | raw split-half top-1 {100*res['splithalf_raw_top1']['mean']:.1f}% 2-way {100*res['splithalf_raw_two_way']['mean']:.1f}% (pair {100*res['splithalf_raw_pair_top1']['mean']:.1f}%) | k=1 bits {res['eeg_k1_bits']['mean']:.3f}")
(R / "modprobe_summary.json").write_text(json.dumps(out, indent=1))
