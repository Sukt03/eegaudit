"""Retrain-vs-mask agreement with a second retrained seed, for occpar32 AND w06 (EXPERIMENTS agent, 2026-11-20). Same logic as
aggregate_retrain_seed.py (which stays as is for occpar32): per (subject, seed) with results/eegbits/retrain_<variant>_<s>_seed<seed>.json,
retrained native k=80 bits and k=1 bits vs the mean-fill masked frozen encoder of the SAME seed (acq_nice_ga_<s>_seed<seed>_mean_loo.json:
occpar32 = channels occpar32, window 1.0; w06 = channels all63, window 0.6) and the full model of the same seed (acq_nice_ga_<s>_seed<seed>.json).
Usage: python aggregate_retrain_seed_variants.py [sub-01,sub-05,sub-08] [141,297]
Output: results/method_pilots/retrain_seeds_<variant>.json/.md"""
import json, sys
from pathlib import Path
import numpy as np
R = Path("/path/to/eegaudit/results/eegbits"); OUT = Path("/path/to/eegaudit/results/method_pilots")
SUBJECTS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["sub-01", "sub-05", "sub-08"]; SEEDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [141, 297]
MASK = {"occpar32": ("occpar32", 1.0), "w06": ("all63", 0.6)}


def ci(v, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(v, float); b = [rng.choice(v, len(v)).mean() for _ in range(5000)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


for variant, (ch, win) in MASK.items():
    per = []
    for s in SUBJECTS:
        for seed in SEEDS:
            f = R / f"retrain_{variant}_{s}_seed{seed}.json"
            if not f.exists(): continue
            d = json.load(open(f)); assert d["variant"] == variant and d["subject"] == s and d["seed"] == seed
            rows = d["rows"]; r80 = float([r for r in rows if r["axis"] == "native"][0]["bits"]); r1 = float(np.mean([r["bits"] for r in rows if r["axis"] == "repetitions"]))
            noise_top1 = float(rows[-1]["top1"])
            mrows = json.load(open(R / f"acq_nice_ga_{s}_seed{seed}_mean_loo.json"))["rows"]
            m80 = float(np.mean([r["bits"] for r in mrows if r["axis"] == "acquisition" and r["k"] == 80 and r["channels"] == ch and r["window"] == win]))
            m1 = float(np.mean([r["bits"] for r in mrows if r["axis"] == "acquisition" and r["k"] == 1 and r["channels"] == ch and r["window"] == win]))
            frows = json.load(open(R / f"acq_nice_ga_{s}_seed{seed}.json"))["rows"]; f80 = float([r for r in frows if r["axis"] == "native"][0]["bits"]); f1 = float(np.mean([r["bits"] for r in frows if r["axis"] == "repetitions" and r["k"] == 1]))
            per.append(dict(subject=s, seed=seed, retrained_k80=r80, masked_k80=m80, full_k80=f80, retrained_k1=r1, masked_k1=m1, full_k1=f1, retrained_noise_top1=noise_top1))
    if not per: continue
    out = dict(variant=variant, masked_cell=dict(channels=ch, window=win), per_pair=per, n_pairs=len(per), summary={})
    md = [f"# Retrained {variant} vs masked frozen encoder ({ch}, {win} s), per (subject, seed)", "",
          "| subject | seed | full k=80 | masked k=80 | retrained k=80 | retr-mask | retr/full | mask/full | retrained noise top-1 |", "|---|---|---|---|---|---|---|---|---|"]
    for p in per:
        md.append(f"| {p['subject']} | {p['seed']} | {p['full_k80']:.3f} | {p['masked_k80']:.3f} | {p['retrained_k80']:.3f} | {p['retrained_k80']-p['masked_k80']:+.3f} | {p['retrained_k80']/p['full_k80']:.2f} | {p['masked_k80']/p['full_k80']:.2f} | {p['retrained_noise_top1']:.3f} |")
    for k in ("k80", "k1"):
        d = np.array([p[f"retrained_{k}"] - p[f"masked_{k}"] for p in per]); rs = np.array([p[f"retrained_{k}"] / p[f"full_{k}"] for p in per])
        bys = {sd: np.array([p[f"retrained_{k}"] - p[f"masked_{k}"] for p in per if p["seed"] == sd]) for sd in SEEDS}
        subj = {}
        for p in per: subj.setdefault(p["subject"], []).append(p[f"retrained_{k}"] - p[f"masked_{k}"])
        sd_ = [np.std([p[f"retrained_{k}"] for p in per if p["subject"] == s], ddof=1) for s in SUBJECTS if sum(p["subject"] == s for p in per) >= 2]
        out["summary"][k] = dict(retrained_minus_masked=dict(mean=float(d.mean()), ci95_boot=ci(d), n=len(d)), retrained_share=dict(mean=float(rs.mean()), ci95_boot=ci(rs)),
                                 per_seed_mean_diff={str(sd): (float(v.mean()) if len(v) else None) for sd, v in bys.items()},
                                 seed_averaged_subject_diff={s: float(np.mean(v)) for s, v in subj.items()},
                                 within_subject_seed_sd_retrained=float(np.mean(sd_)) if sd_ else None)
        md += ["", f"{k}: retrained - masked {d.mean():+.3f} bits, 95% CI [{ci(d)[0]:+.3f}, {ci(d)[1]:+.3f}] over {len(d)} (subject, seed) pairs; per seed " +
               ", ".join(f"{sd}: {v.mean():+.3f} (n={len(v)})" for sd, v in bys.items() if len(v)) + f"; retrained share of full {rs.mean():.2f}; within-subject retrained seed SD {out['summary'][k]['within_subject_seed_sd_retrained']}"]
    (OUT / f"retrain_seeds_{variant}.json").write_text(json.dumps(out, indent=1)); (OUT / f"retrain_seeds_{variant}.md").write_text("\n".join(md) + "\n"); print("\n".join(md)); print()
