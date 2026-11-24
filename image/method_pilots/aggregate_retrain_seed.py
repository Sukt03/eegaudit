"""Round C: retrain-vs-mask agreement for occpar32 with a seed CI. For each (subject, seed) with results/eegbits/retrain_occpar32_<s>_seed<seed>.json:
retrained bits (k=80, k=1) vs the mean-fill masked frozen encoder of the SAME seed (acq_nice_ga_<s>_seed<seed>_mean_loo.json, channels=occpar32,
window=1.0) and the full model of the same seed (acq_nice_ga_<s>_seed<seed>.json). Output: results/method_pilots/retrain_occpar32_seeds.{json,md}"""
import json, sys
from pathlib import Path
import numpy as np
R = Path("/path/to/eegaudit/results/eegbits"); OUT = Path("/path/to/eegaudit/results/method_pilots"); rng = np.random.default_rng(0)
SUBJECTS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["sub-01", "sub-05", "sub-08"]; SEEDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [141, 297, 777]


def ci(v):
    v = np.asarray(v, float); b = [rng.choice(v, len(v)).mean() for _ in range(2000)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


per = []
for s in SUBJECTS:
    for seed in SEEDS:
        f = R / f"retrain_occpar32_{s}_seed{seed}.json"
        if not f.exists(): continue
        rows = json.load(open(f))["rows"]; r80 = float([r for r in rows if r["axis"] == "native"][0]["bits"]); r1 = float(np.mean([r["bits"] for r in rows if r["axis"] == "repetitions"]))
        mrows = json.load(open(R / f"acq_nice_ga_{s}_seed{seed}_mean_loo.json"))["rows"]
        m80 = float(np.mean([r["bits"] for r in mrows if r["axis"] == "acquisition" and r["k"] == 80 and r["channels"] == "occpar32" and r["window"] == 1.0]))
        m1 = float(np.mean([r["bits"] for r in mrows if r["axis"] == "acquisition" and r["k"] == 1 and r["channels"] == "occpar32" and r["window"] == 1.0]))
        frows = json.load(open(R / f"acq_nice_ga_{s}_seed{seed}.json"))["rows"]; f80 = float([r for r in frows if r["axis"] == "native"][0]["bits"]); f1 = float(np.mean([r["bits"] for r in frows if r["axis"] == "repetitions" and r["k"] == 1]))
        per.append(dict(subject=s, seed=seed, retrained_k80=r80, masked_k80=m80, full_k80=f80, retrained_k1=r1, masked_k1=m1, full_k1=f1))
out = dict(per_pair=per, n_pairs=len(per), summary={})
md = ["# Retrained occpar32 vs masked frozen encoder, per (subject, seed)", "", "| subject | seed | full k=80 | masked k=80 | retrained k=80 | retr-mask | retr/full | mask/full | full k=1 | masked k=1 | retrained k=1 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for p in per:
    md.append(f"| {p['subject']} | {p['seed']} | {p['full_k80']:.3f} | {p['masked_k80']:.3f} | {p['retrained_k80']:.3f} | {p['retrained_k80']-p['masked_k80']:+.3f} | {p['retrained_k80']/p['full_k80']:.2f} | {p['masked_k80']/p['full_k80']:.2f} | {p['full_k1']:.3f} | {p['masked_k1']:.3f} | {p['retrained_k1']:.3f} |")
for k in ("k80", "k1"):
    d = np.array([p[f"retrained_{k}"] - p[f"masked_{k}"] for p in per]); rs = np.array([p[f"retrained_{k}"] / p[f"full_{k}"] for p in per]); ms = np.array([p[f"masked_{k}"] / p[f"full_{k}"] for p in per])
    out["summary"][k] = dict(retrained_minus_masked=dict(mean=float(d.mean()), ci95_boot=ci(d), n=len(d)), retrained_share=dict(mean=float(rs.mean()), ci95_boot=ci(rs)), masked_share=dict(mean=float(ms.mean()), ci95_boot=ci(ms)))
    sub_sets = {s: [p for p in per if p["subject"] == s] for s in SUBJECTS}
    out["summary"][k]["within_subject_seed_sd_retrained"] = float(np.mean([np.std([p[f"retrained_{k}"] for p in v], ddof=1) for v in sub_sets.values() if len(v) >= 2] or [np.nan]))
    out["summary"][k]["within_subject_seed_sd_masked"] = float(np.mean([np.std([p[f"masked_{k}"] for p in v], ddof=1) for v in sub_sets.values() if len(v) >= 2] or [np.nan]))
    md += ["", f"{k}: retrained - masked {d.mean():+.3f} bits, 95% CI {ci(d)} over {len(d)} pairs; retrained share of full {rs.mean():.2f} {ci(rs)}; masked share {ms.mean():.2f} {ci(ms)}; within-subject seed SD retrained {out['summary'][k]['within_subject_seed_sd_retrained']:.3f}, masked {out['summary'][k]['within_subject_seed_sd_masked']:.3f}"]
(OUT / "retrain_occpar32_seeds.json").write_text(json.dumps(out, indent=1)); (OUT / "retrain_occpar32_seeds.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
