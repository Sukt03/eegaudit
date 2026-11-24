"""Paired, seed-replicated target swap (method image loop, 2026-11-20). Per (subject, seed): Wolpaw bits (and soft bits when the file
has them) of NICE-GA trained and scored in each target space with the 200 test-image features as candidates:
  clip   : released NICE-GA checkpoint rescored vs test features (seed 141: results/eegbits/clip_testfeat_<s>.json;
           other seeds: results/method_pilots/clip_testfeat_<s>_seed<seed>.json)
  dinov2 : results/eegbits/retrain_dinov2_<s>_seed<seed>.json ;  pixpca : results/eegbits/retrain_pixpca_<s>_seed<seed>.json
Paired differences DINOv2-CLIP and CLIP-pixels with a 95 % CI over the complete (subject, seed) pairs: bootstrap over pairs (2000
resamples, seed 0) and the t-interval. Output: results/method_pilots/paired_target_swap.json and .md"""
import json, glob, re, sys
from pathlib import Path
import numpy as np
R = Path("/path/to/eegaudit/results"); OUT = R / "method_pilots"
SUBJECTS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["sub-01", "sub-05", "sub-08"]
SEEDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [141, 297, 777]
rng = np.random.default_rng(0)


def k80_k1(rows, native_filter=lambda r: True):
    nat = [r for r in rows if r["axis"] == "native" and native_filter(r)][0]
    k1 = [r for r in rows if r["axis"] == "repetitions"]
    return dict(bits=float(nat["bits"]), top1=float(nat["top1"]), two_way=float(nat["two_way"]), soft_bits=float(nat["soft_bits"]) if "soft_bits" in nat else None,
                k1_bits=float(np.mean([r["bits"] for r in k1])), k1_soft_bits=float(np.mean([r["soft_bits"] for r in k1])) if all("soft_bits" in r for r in k1) else None)


def load(space, s, seed):
    if space == "clip":
        f = R / "eegbits" / f"clip_testfeat_{s}.json" if seed == 141 else OUT / f"clip_testfeat_{s}_seed{seed}.json"
        if not f.exists(): return None
        d = json.load(open(f)); assert d["seed"] == seed
        return k80_k1(d["rows"], lambda r: r.get("candidates") == "test_image_features")
    f = R / "eegbits" / f"retrain_{space}_{s}_seed{seed}.json"
    if not f.exists(): return None
    d = json.load(open(f)); assert d["seed"] == seed and d["variant"] == space
    return k80_k1(d["rows"])


def ci_boot(v):
    v = np.asarray(v, float); b = [rng.choice(v, len(v)).mean() for _ in range(2000)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def ci_t(v):
    from scipy import stats
    v = np.asarray(v, float); n = len(v)
    if n < 2: return [float(v.mean()), float(v.mean())]
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n); return [float(v.mean() - h), float(v.mean() + h)]


cells = {}; complete = []
for s in SUBJECTS:
    for seed in SEEDS:
        c = {sp: load(sp, s, seed) for sp in ("clip", "dinov2", "pixpca")}
        cells[f"{s}_seed{seed}"] = c
        if all(c.values()): complete.append((s, seed))
out = dict(subjects=SUBJECTS, seeds=SEEDS, n_pairs_complete=len(complete), pairs_complete=[f"{s}_seed{sd}" for s, sd in complete], cells=cells, summary={})
md = ["# Paired target swap: NICE-GA trained and scored in CLIP / DINOv2 / pixel-PCA space (200 test-image candidates)", "",
      f"Complete (subject, seed) pairs: {len(complete)} of {len(SUBJECTS) * len(SEEDS)}.", "",
      "| subject | seed | CLIP bits | DINOv2 bits | pixels bits | DINOv2-CLIP | CLIP-pixels | CLIP top-1 | DINOv2 top-1 | pixels top-1 | CLIP k=1 | DINOv2 k=1 | pixels k=1 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for s in SUBJECTS:
    for seed in SEEDS:
        c = cells[f"{s}_seed{seed}"]; g = lambda sp, k: (f"{c[sp][k]:.3f}" if c[sp] else "-")
        dd = f"{c['dinov2']['bits'] - c['clip']['bits']:+.3f}" if c["dinov2"] and c["clip"] else "-"; dp = f"{c['clip']['bits'] - c['pixpca']['bits']:+.3f}" if c["pixpca"] and c["clip"] else "-"
        md.append(f"| {s} | {seed} | {g('clip','bits')} | {g('dinov2','bits')} | {g('pixpca','bits')} | {dd} | {dp} | {g('clip','top1')} | {g('dinov2','top1')} | {g('pixpca','top1')} | {g('clip','k1_bits')} | {g('dinov2','k1_bits')} | {g('pixpca','k1_bits')} |")
for metric in ("bits", "k1_bits", "top1", "soft_bits"):
    vals = {sp: np.array([cells[f"{s}_seed{sd}"][sp][metric] for s, sd in complete], dtype=float) for sp in ("clip", "dinov2", "pixpca")}
    if any(np.isnan(v).any() for v in vals.values()) or not complete: continue
    sm = {sp: dict(mean=float(v.mean()), ci95_boot=ci_boot(v)) for sp, v in vals.items()}
    for name, d in (("dinov2_minus_clip", vals["dinov2"] - vals["clip"]), ("clip_minus_pixpca", vals["clip"] - vals["pixpca"]), ("dinov2_minus_pixpca", vals["dinov2"] - vals["pixpca"])):
        sm[name] = dict(mean=float(d.mean()), ci95_boot=ci_boot(d), ci95_t=ci_t(d), n=int(len(d)), n_positive=int((d > 0).sum()), per_pair=d.tolist())
    # per-subject means over seeds, then paired over subjects (the seed-averaged version)
    bysub = {sp: np.array([np.mean([cells[f"{s}_seed{sd}"][sp][metric] for s2, sd in complete if s2 == s]) for s in SUBJECTS if any(s2 == s for s2, _ in complete)]) for sp in vals}
    sm["seed_averaged"] = {"dinov2_minus_clip": dict(mean=float((bysub["dinov2"] - bysub["clip"]).mean()), per_subject=(bysub["dinov2"] - bysub["clip"]).tolist()),
                           "clip_minus_pixpca": dict(mean=float((bysub["clip"] - bysub["pixpca"]).mean()), per_subject=(bysub["clip"] - bysub["pixpca"]).tolist())}
    # seed spread within subject (SD over seeds) per space
    sm["within_subject_seed_sd"] = {sp: float(np.mean([np.std([cells[f"{s}_seed{sd}"][sp][metric] for s2, sd in complete if s2 == s], ddof=1) for s in SUBJECTS if sum(s2 == s for s2, _ in complete) >= 2] or [np.nan])) for sp in vals}
    out["summary"][metric] = sm
    md += ["", f"## Summary over {len(complete)} pairs: {metric}", "", "| quantity | mean | 95% CI (bootstrap over pairs) | 95% CI (t) | n positive |", "|---|---|---|---|---|"]
    for sp in ("clip", "dinov2", "pixpca"):
        md.append(f"| {sp} | {sm[sp]['mean']:.3f} | [{sm[sp]['ci95_boot'][0]:.3f}, {sm[sp]['ci95_boot'][1]:.3f}] | | |")
    for name in ("dinov2_minus_clip", "clip_minus_pixpca", "dinov2_minus_pixpca"):
        d = sm[name]; md.append(f"| {name} | {d['mean']:+.3f} | [{d['ci95_boot'][0]:+.3f}, {d['ci95_boot'][1]:+.3f}] | [{d['ci95_t'][0]:+.3f}, {d['ci95_t'][1]:+.3f}] | {d['n_positive']}/{d['n']} |")
    md.append(f"\nWithin-subject SD over seeds: " + ", ".join(f"{sp} {v:.3f}" for sp, v in sm["within_subject_seed_sd"].items()))
(OUT / "paired_target_swap.json").write_text(json.dumps(out, indent=1)); (OUT / "paired_target_swap.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
