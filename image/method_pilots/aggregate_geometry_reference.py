"""Table for Round B: mean over available (subject, seed) files of bits / top-1 per (source, target) cell at k = 80 and k = 1,
next to the retrained yields from paired_target_swap.json. Output: results/method_pilots/geometry_reference_summary.{json,md}"""
import json, glob
from pathlib import Path
import numpy as np
OUT = Path("/path/to/eegaudit/results/method_pilots")
files = sorted(glob.glob(str(OUT / "geometry_reference_sub-*_seed*.json")))
acc = {}
for f in files:
    d = json.load(open(f))
    for r in d["rows"]:
        key = (r["source"], r["target"], r["k"]); acc.setdefault(key, {}).setdefault(f"{d['subject']}_seed{d['seed']}", []).append((r["bits"], r["top1"]))
summ = {}
for key, per in acc.items():
    b = [np.mean([x[0] for x in v]) for v in per.values()]; t = [np.mean([x[1] for x in v]) for v in per.values()]
    summ["|".join(map(str, key))] = dict(n_pairs=len(per), bits_mean=float(np.mean(b)), bits_sd=float(np.std(b, ddof=1)) if len(b) > 1 else None, top1_mean=float(np.mean(t)), per_pair_bits={k: float(np.mean([x[0] for x in v])) for k, v in per.items()})
pts = json.load(open(OUT / "paired_target_swap.json")) if (OUT / "paired_target_swap.json").exists() else None
md = [f"# Geometry reference (LOO ridge map of trained-encoder embeddings into each target space); {len(files)} (subject, seed) files", "",
      "| source | -> CLIP bits (top-1) | -> DINOv2 bits (top-1) | -> pixels bits (top-1) |", "|---|---|---|---|"]
for src in ["image_clip_features", "encoder_clip", "encoder_dinov2", "encoder_pixpca", "encoder_clip_shuffled", "encoder_dinov2_shuffled", "encoder_pixpca_shuffled"]:
    for k in ([None] if "image" in src else ([80] if "shuffled" in src else [80, 1])):
        cells = []
        for T in ("clip", "dinov2", "pixpca"):
            s = summ.get(f"{src}|{T}|{k}"); cells.append(f"{s['bits_mean']:.3f} ({100*s['top1_mean']:.1f}%)" if s else "-")
        md.append(f"| {src}{'' if k is None else f' k={k}'} | " + " | ".join(cells) + " |")
if pts and pts["summary"].get("bits"):
    sm = pts["summary"]["bits"]; md += ["", "Retrained-in-space yields (paired_target_swap.json, k=80 bits, same pairs where complete): " + ", ".join(f"{sp} {sm[sp]['mean']:.3f}" for sp in ("clip", "dinov2", "pixpca"))]
    for arm in ("clip", "dinov2", "pixpca"):
        s = summ.get(f"encoder_{arm}_direct|{arm}|80")
        if s: md.append(f"- encoder_{arm} direct cosine scoring in its own space (same files): {s['bits_mean']:.3f} bits over {s['n_pairs']} pairs")
(OUT / "geometry_reference_summary.json").write_text(json.dumps(dict(files=files, cells=summ), indent=1)); (OUT / "geometry_reference_summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
