"""Target swap, every available pair (EXPERIMENTS agent, 2026-11-20). Same loaders and candidate set as aggregate_paired_target_swap.py
(200 test-image features; CLIP = released NICE-GA rescored vs test features; DINOv2 / pixel-PCA = NICE-GA retrained in that space), but each
paired difference uses every (subject, seed) that has BOTH of its arms, not only the cells with all three. Two levels of inference:
pairs (bootstrap over (subject, seed) pairs) and subjects (seeds averaged within subject first, bootstrap + t over subjects).
Output: results/method_pilots/paired_target_swap_allpairs.{json,md}. Does not modify paired_target_swap.json."""
import json
from pathlib import Path
import numpy as np
from scipy import stats
R = Path("/path/to/eegaudit/results"); OUT = R / "method_pilots"
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 11)]; SEEDS = [141, 297, 777]


def k80(rows, flt=lambda r: True):
    nat = [r for r in rows if r["axis"] == "native" and flt(r)][0]
    k1 = [r for r in rows if r["axis"] == "repetitions"]
    return dict(bits=float(nat["bits"]), top1=float(nat["top1"]), k1_bits=float(np.mean([r["bits"] for r in k1])))


def load(space, s, seed):
    if space == "clip":
        f = R / "eegbits" / f"clip_testfeat_{s}.json" if seed == 141 else OUT / f"clip_testfeat_{s}_seed{seed}.json"
        if not f.exists(): return None
        d = json.load(open(f)); assert d["seed"] == seed
        return k80(d["rows"], lambda r: r.get("candidates") == "test_image_features")
    f = R / "eegbits" / f"retrain_{space}_{s}_seed{seed}.json"
    if not f.exists(): return None
    d = json.load(open(f)); assert d["seed"] == seed and d["variant"] == space and d["subject"] == s
    return k80(d["rows"])


def boot(v, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(v, float); b = [rng.choice(v, len(v)).mean() for _ in range(5000)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def tci(v):
    v = np.asarray(v, float); n = len(v)
    if n < 2: return [float("nan")] * 2
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n); return [float(v.mean() - h), float(v.mean() + h)]


cells = {f"{s}_seed{sd}": {sp: load(sp, s, sd) for sp in ("clip", "dinov2", "pixpca")} for s in SUBJECTS for sd in SEEDS}
out = dict(note=__doc__, cells=cells, comparisons={})
md = ["# Target swap over every available pair (Wolpaw bits, k=80, 200 test-image candidates)", ""]
for name, a, b in (("dinov2_minus_clip", "dinov2", "clip"), ("clip_minus_pixpca", "clip", "pixpca"), ("dinov2_minus_pixpca", "dinov2", "pixpca")):
    for key in ("bits", "k1_bits"):
        pairs = [(k, c[a][key] - c[b][key]) for k, c in cells.items() if c[a] and c[b]]
        d = np.array([x for _, x in pairs])
        subj = {}
        for k, x in pairs: subj.setdefault(k.split("_")[0], []).append(x)
        ds = np.array([np.mean(v) for v in subj.values()])
        res = dict(n_pairs=len(d), pairs=[k for k, _ in pairs], mean_pairs=float(d.mean()), ci95_boot_pairs=boot(d), n_pos_pairs=int((d > 0).sum()),
                   n_subjects=len(ds), subjects=list(subj), mean_subjects=float(ds.mean()), ci95_boot_subjects=boot(ds), ci95_t_subjects=tci(ds),
                   n_pos_subjects=int((ds > 0).sum()), wilcoxon_p_subjects=float(stats.wilcoxon(ds).pvalue) if len(ds) >= 5 else None,
                   mean_a=float(np.mean([cells[k][a][key] for k, _ in pairs])), mean_b=float(np.mean([cells[k][b][key] for k, _ in pairs])))
        out["comparisons"][f"{name}_{key}"] = res
        md.append(f"- {name} [{key}]: pairs n={res['n_pairs']} mean {res['mean_pairs']:+.3f} CI {res['ci95_boot_pairs'][0]:+.3f},{res['ci95_boot_pairs'][1]:+.3f} ({res['n_pos_pairs']}/{res['n_pairs']} >0); "
                  f"subjects n={res['n_subjects']} mean {res['mean_subjects']:+.3f} boot CI {res['ci95_boot_subjects'][0]:+.3f},{res['ci95_boot_subjects'][1]:+.3f} t CI {res['ci95_t_subjects'][0]:+.3f},{res['ci95_t_subjects'][1]:+.3f} "
                  f"({res['n_pos_subjects']}/{res['n_subjects']} >0, Wilcoxon p={res['wilcoxon_p_subjects']}); arms {res['mean_a']:.3f} vs {res['mean_b']:.3f}")
md += ["", "| cell | CLIP | DINOv2 | pixels |", "|---|---|---|---|"]
for k, c in cells.items():
    if any(c.values()): md.append(f"| {k} | " + " | ".join(f"{c[sp]['bits']:.3f}" if c[sp] else "-" for sp in ("clip", "dinov2", "pixpca")) + " |")
(OUT / "paired_target_swap_allpairs.json").write_text(json.dumps(out, indent=1)); (OUT / "paired_target_swap_allpairs.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
