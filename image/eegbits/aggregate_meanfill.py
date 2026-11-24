"""Window and electrode axes under the in-distribution (mean_loo) fill vs zero fill, through the frozen encoders.
Inputs : results/eegbits/acq_{nice,nice_ga}_sub-XX_seed*.json and *_mean_loo.json, acq_atm_sub-08{,_mean_loo}.json
Output : results/eegbits/meanfill_summary.json (per decoder x fill x window: bits and top-1 at k=80, all63; occpar32 at 1.0 s;
         fraction of the 1.0-s yield at each window; mean and bootstrap CI over subjects (seeds averaged within subject))."""
import glob, json, re
from pathlib import Path
import numpy as np
R = Path("/path/to/eegaudit/results/eegbits"); rng = np.random.default_rng(0)

def ci(v):
    v = np.asarray(v, float)
    if len(v) < 2: return [float(v.mean()), float(v.mean())]
    b = [rng.choice(v, len(v)).mean() for _ in range(2000)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

def rows_of(files):
    per_subj = {}
    for f in files:
        d = json.load(open(f)); subj = d["subject"]
        per_subj.setdefault(subj, []).append(d["rows"])
    return per_subj

def pick(rows, **kw):
    out = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    return out

out = {}
for dec, pat in [("atm", "acq_atm_sub-08{}.json"), ("nice", "acq_nice_sub-*_seed141{}.json"), ("nice_ga", "acq_nice_ga_sub-*_seed*{}.json")]:
    for fill, suf in [("zero", ""), ("mean_loo", "_mean_loo")]:
        files = sorted(glob.glob(str(R / pat.format(suf)))); files = [f for f in files if (suf and f.endswith("_mean_loo.json")) or (not suf and not f.endswith("_mean_loo.json"))]
        if not files: continue
        ps = rows_of(files); res = {"n_subjects": len(ps), "n_files": len(files), "window": {}, "occpar32_w1": {}}
        for w in (0.2, 0.4, 0.6, 1.0):
            bits, top1, frac = [], [], []
            for subj, runs in ps.items():
                b = np.mean([np.mean([r["bits"] for r in pick(rows, axis="acquisition", k=80, channels="all63", window=w)]) for rows in runs])
                t = np.mean([np.mean([r["top1"] for r in pick(rows, axis="acquisition", k=80, channels="all63", window=w)]) for rows in runs])
                b1 = np.mean([np.mean([r["bits"] for r in pick(rows, axis="acquisition", k=80, channels="all63", window=1.0)]) for rows in runs])
                bits.append(b); top1.append(t); frac.append(b / b1 if b1 > 0 else np.nan)
            res["window"][str(w)] = dict(bits_mean=float(np.mean(bits)), bits_ci=ci(bits), top1_mean=float(np.mean(top1)), frac_of_full_mean=float(np.nanmean(frac)), frac_ci=ci([x for x in frac if np.isfinite(x)]))
        bits, frac = [], []
        for subj, runs in ps.items():
            b = np.mean([np.mean([r["bits"] for r in pick(rows, axis="acquisition", k=80, channels="occpar32", window=1.0)]) for rows in runs])
            b1 = np.mean([np.mean([r["bits"] for r in pick(rows, axis="acquisition", k=80, channels="all63", window=1.0)]) for rows in runs])
            bits.append(b); frac.append(b / b1)
        res["occpar32_w1"] = dict(bits_mean=float(np.mean(bits)), bits_ci=ci(bits), frac_of_full_mean=float(np.mean(frac)), frac_ci=ci(frac))
        for w in (0.4, 0.6):                                   # joint budget: 32 occipito-parietal electrodes AND a short window
            bits, frac = [], []
            for subj, runs in ps.items():
                b = np.mean([np.mean([r["bits"] for r in pick(rows, axis="acquisition", k=80, channels="occpar32", window=w)]) for rows in runs])
                b1 = np.mean([np.mean([r["bits"] for r in pick(rows, axis="acquisition", k=80, channels="all63", window=1.0)]) for rows in runs])
                bits.append(b); frac.append(b / b1)
            res[f"occpar32_w{w}"] = dict(bits_mean=float(np.mean(bits)), bits_ci=ci(bits), frac_of_full_mean=float(np.mean(frac)), frac_ci=ci(frac))
        out[f"{dec}_{fill}"] = res
        print(dec, fill, res["n_subjects"], "subj | frac of 1.0 s yield by window:", {w: round(v["frac_of_full_mean"], 2) for w, v in res["window"].items()}, "| occpar32 frac", round(res["occpar32_w1"]["frac_of_full_mean"], 2), "| occpar32+0.6s", round(res["occpar32_w0.6"]["frac_of_full_mean"], 2), "| occpar32+0.4s", round(res["occpar32_w0.4"]["frac_of_full_mean"], 2))
(R / "meanfill_summary.json").write_text(json.dumps(out, indent=1))
