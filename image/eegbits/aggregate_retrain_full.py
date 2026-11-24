"""Retraining check vs masking (auditor follow-up): NICE-GA retrained on 32 occipito-parietal electrodes (occpar32) or on the first
0.6 s (w06) versus the mean-fill masked frozen 63-channel/1.0-s encoder, same subjects, seed 141, k = 80 and k = 1.
Inputs : results/eegbits/retrain_<variant>_sub-XX_seed141.json ; acq_nice_ga_sub-XX_seed141_mean_loo.json (masked) ; acq_nice_ga_sub-XX_seed141.json (full)
Output : results/eegbits/retrain_summary.json  (per variant: retrained bits, masked bits, full bits, ratios, paired differences with CI)"""
import glob, json
from pathlib import Path
import numpy as np
R = Path("/path/to/eegaudit/results/eegbits"); rng = np.random.default_rng(0)


def ci(v):
    v = np.asarray(v, float)
    if len(v) < 2: return [float(v.mean()), float(v.mean())]
    b = [rng.choice(v, len(v)).mean() for _ in range(2000)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def masked(subject, variant, k):
    rows = json.load(open(R / f"acq_nice_ga_{subject}_seed141_mean_loo.json"))["rows"]
    if variant not in ("occpar32", "w06"):                 # no masked counterpart for the other variants: compare with the full model only
        return float("nan")
    ch, w = ("occpar32", 1.0) if variant == "occpar32" else ("all63", 0.6)
    return float(np.mean([r["bits"] for r in rows if r["axis"] == "acquisition" and r["k"] == k and r["channels"] == ch and r["window"] == w]))


def full(subject, k):
    rows = json.load(open(R / f"acq_nice_ga_{subject}_seed141.json"))["rows"]
    if k == 80: return float([r for r in rows if r["axis"] == "native"][0]["bits"])
    return float(np.mean([r["bits"] for r in rows if r["axis"] == "repetitions" and r["k"] == 1]))


out = {}
for variant in ["occpar32", "occpar16", "w06", "w04", "occ32w06", "pca100", "shuffle", "dinov2", "pixpca", "randtarget"]:
    files = sorted(glob.glob(str(R / f"retrain_{variant}_sub-*_seed141.json")))
    if not files: continue
    per = []
    for f in files:
        d = json.load(open(f)); s = d["subject"]; rows = d["rows"]
        r80 = float([r for r in rows if r["axis"] == "native"][0]["bits"]); r1 = float(np.mean([r["bits"] for r in rows if r["axis"] == "repetitions"]))
        nat = [r for r in rows if r["axis"] == "native"][0]
        per.append(dict(subject=s, retrained_k80=r80, masked_k80=masked(s, variant, 80), full_k80=full(s, 80),
                        retrained_k1=r1, masked_k1=masked(s, variant, 1), full_k1=full(s, 1),
                        top1_retrained=float(nat["top1"]), two_way_retrained=float(nat["two_way"]), soft_retrained=float(nat.get("soft_bits", float("nan"))), noise_top1=float(rows[-1]["top1"])))
    res = dict(n_subjects=len(per), per_subject=per)
    for k in ("k80", "k1"):
        rt = [p[f"retrained_{k}"] for p in per]; mk = [p[f"masked_{k}"] for p in per]; fu = [p[f"full_{k}"] for p in per]
        res[f"{k}_retrained_bits"] = dict(mean=float(np.mean(rt)), ci95=ci(rt)); res[f"{k}_masked_bits"] = dict(mean=float(np.mean(mk)), ci95=ci(mk)); res[f"{k}_full_bits"] = dict(mean=float(np.mean(fu)), ci95=ci(fu))
        res[f"{k}_retrained_share_of_full"] = dict(mean=float(np.mean(np.array(rt) / np.array(fu))), ci95=ci(np.array(rt) / np.array(fu)))
        res[f"{k}_masked_share_of_full"] = dict(mean=float(np.mean(np.array(mk) / np.array(fu))), ci95=ci(np.array(mk) / np.array(fu)))
        d = np.array(rt) - np.array(mk); d = d[np.isfinite(d)]
        res[f"{k}_retrained_minus_masked_bits"] = dict(mean=float(np.mean(d)) if len(d) else None, ci95=ci(d) if len(d) else None)
        mk2 = [x for x in mk if np.isfinite(x)]
        res[f"{k}_masked_bits"] = dict(mean=float(np.mean(mk2)) if mk2 else None, ci95=ci(mk2) if mk2 else None)
        res[f"{k}_masked_share_of_full"] = dict(mean=float(np.mean(np.array(mk2) / np.array(fu)[:len(mk2)])) if mk2 else None)
    out[variant] = res
    sr = [p["soft_retrained"] for p in per if np.isfinite(p["soft_retrained"])]
    res["k80_retrained_soft_bits"] = dict(mean=float(np.mean(sr)), ci95=ci(sr)) if sr else None
    res["top1_retrained_mean"] = float(np.mean([p["top1_retrained"] for p in per])); res["noise_top1_mean"] = float(np.mean([p["noise_top1"] for p in per]))
    mb = res['k80_masked_bits']['mean']; dd = res['k80_retrained_minus_masked_bits']
    print(f"{variant}: n={len(per)} k=80 retrained {res['k80_retrained_bits']['mean']:.2f} bits ({100*res['k80_retrained_share_of_full']['mean']:.0f}% of full CLIP model, top-1 {100*res['top1_retrained_mean']:.1f}%)" + (f" vs masked {mb:.2f} | diff {dd['mean']:+.2f} CI {dd['ci95']}" if mb is not None else "") + f" | k=1 retrained {res['k1_retrained_bits']['mean']:.3f} | noise top-1 {100*res['noise_top1_mean']:.1f}%")
# CLIP arm scored against the test-image features (same candidate set as the DINOv2 / pixel arms), from rescore_clip_testfeat.py
cf = sorted(glob.glob(str(R / "clip_testfeat_sub-*.json")))
if cf:
    v80, v1, vc = [], [], []
    for f in cf:
        rows = json.load(open(f))["rows"]
        v80.append([r["bits"] for r in rows if r["axis"] == "native" and r["candidates"] == "test_image_features"][0])
        vc.append([r["bits"] for r in rows if r["axis"] == "native" and r["candidates"] == "concept_centres"][0])
        v1.append(float(np.mean([r["bits"] for r in rows if r["axis"] == "repetitions"])))
    out["clip_testfeat"] = dict(n_subjects=len(cf), k80_bits=dict(mean=float(np.mean(v80)), ci95=ci(v80)), k1_bits=dict(mean=float(np.mean(v1)), ci95=ci(v1)),
                                k80_bits_vs_centres=dict(mean=float(np.mean(vc)), ci95=ci(vc)),
                                top1_mean=float(np.mean([[r["top1"] for r in json.load(open(f))["rows"] if r["axis"] == "native" and r["candidates"] == "test_image_features"][0] for f in cf])))
    print(f"clip vs test features: n={len(cf)} k=80 {out['clip_testfeat']['k80_bits']['mean']:.2f} bits (vs centres {out['clip_testfeat']['k80_bits_vs_centres']['mean']:.2f})")
(R / "retrain_summary_full.json").write_text(json.dumps(out, indent=1)); print("-> retrain_summary_full.json")
