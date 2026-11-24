"""S1: InfoNCE (soft) and Wolpaw yield for every RETRAINED NICE-GA variant checkpoint, k = 80 and k = 1 (5 draws), with the
full-input CLIP NICE-GA checkpoint of the same subject x seed as the reference ("full"). Retrain files that already carry
soft_bits (written after run.metrics gained it) are reused as-is; older ones are re-scored through the checkpoint.
Output: results/loop/image/s1_full_<subject>_seed<seed>.json, s1_var_<variant>_<subject>_seed<seed>.json (skip-if-exists),
then s1_table.md / s1_table.json via s1_aggregate().  Usage: python s1_rescore_variants.py [--only variant]"""
import json, re, sys, time
import numpy as np, torch
sys.path.insert(0, "/path/to/eegaudit/experiments/loop/image")
from common_image import EB, OUT, metrics, load_encoder, load_variant, save

VARIANTS = ["occpar32", "occpar16", "w06", "w04", "pca100", "shuffle", "dinov2", "pixpca", "randtarget", "occ32w06"]
only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
dev = torch.device("cuda")


def rows_for(embed, Y, X, seed):
    """Same draw sequence as retrain_nice_ga_variant.py: rng(seed % 1000), k=80 then five k=1 draws."""
    rng = np.random.default_rng(seed % 1000); rows = [dict(k=80, draw=0, **metrics(embed(X.mean(1)), Y))]
    for d in range(5):
        rows.append(dict(k=1, draw=d, **metrics(embed(X[:, rng.choice(80, 1)].mean(1)), Y)))
    rows.append(dict(k=80, draw=0, condition="noise", **metrics(embed(rng.standard_normal(X.shape[0:1] + X.shape[2:]).astype(np.float32)), Y)))
    return rows


def full_reference(subject, seed):
    p = OUT / f"s1_full_{subject}_seed{seed}.json"
    if p.exists():
        return
    t0 = time.time(); embed, Y, tag, X = load_encoder("nice_ga", subject, seed, dev); X = np.asarray(X)
    save(p.name, dict(variant="full_clip_nice_ga", subject=subject, seed=seed, rows=rows_for(embed, Y, X, seed), runtime_s=round(time.time() - t0, 1)))
    del embed; torch.cuda.empty_cache()


def variant_rescore(variant, subject, seed):
    p = OUT / f"s1_var_{variant}_{subject}_seed{seed}.json"
    if p.exists():
        return
    t0 = time.time(); meta = json.loads((EB / f"retrain_{variant}_{subject}_seed{seed}.json").read_text())
    if all("soft_bits" in r for r in meta["rows"]):
        rows = [dict(k=r["k"], draw=r["draw"], **({"condition": "noise"} if r.get("condition") == "noise" else {}), **{m: r[m] for m in ("top1", "two_way", "bits", "soft_bits")}) for r in meta["rows"]]
        src = "reused retrain json (already scored with soft_bits)"
    else:
        embed, Y, X, _ = load_variant(variant, subject, seed, dev); rows = rows_for(embed, Y, X, seed); src = "re-scored through the saved checkpoint"
        # consistency check against the retrain file's Wolpaw numbers at k=80
        r0 = [r for r in meta["rows"] if r["k"] == 80 and r["axis"] == "native"][0]
        assert abs(r0["top1"] - rows[0]["top1"]) < 0.02, (variant, subject, seed, r0["top1"], rows[0]["top1"])
        del embed; torch.cuda.empty_cache()
    save(p.name, dict(variant=variant, subject=subject, seed=seed, source=src, rows=rows, runtime_s=round(time.time() - t0, 1)))


def s1_aggregate():
    tab = {}; per = []
    for f in sorted(OUT.glob("s1_var_*.json")):
        v = json.loads(f.read_text()); full = OUT / f"s1_full_{v['subject']}_seed{v['seed']}.json"
        if not full.exists():
            continue
        F = json.loads(full.read_text())
        def agg(rows, k):
            rs = [r for r in rows if r["k"] == k and r.get("condition") != "noise"]
            return {m: float(np.mean([r[m] for r in rs])) for m in ("top1", "bits", "soft_bits")}
        rec = dict(variant=v["variant"], subject=v["subject"], seed=v["seed"])
        for k in (80, 1):
            a, b = agg(v["rows"], k), agg(F["rows"], k)
            rec[f"k{k}"] = dict(var=a, full=b, share_wolpaw=a["bits"] / b["bits"] if b["bits"] > 0 else float("nan"), share_soft=a["soft_bits"] / b["soft_bits"] if b["soft_bits"] > 0 else float("nan"))
        noise = [r for r in v["rows"] if r.get("condition") == "noise"]
        rec["noise_k80"] = {m: float(noise[0][m]) for m in ("top1", "bits", "soft_bits")} if noise else None
        per.append(rec); tab.setdefault(v["variant"], []).append(rec)
    lines = ["# S1: retrained NICE-GA variants, share of the full-input CLIP NICE-GA yield (same subject x seed), both bounds", "",
             "| variant | n (subj x seed) | k=80 Wolpaw bits var/full | share Wolpaw | k=80 soft bits var/full | share soft | k=1 Wolpaw share | k=1 soft share | noise k=80 soft bits |", "|---|---|---|---|---|---|---|---|---|"]
    summ = {}
    for v in ["occpar32", "occpar16", "w06", "w04", "occ32w06", "pca100", "shuffle", "dinov2", "pixpca", "randtarget"]:
        R = tab.get(v, [])
        if not R:
            continue
        def ms(key, k):
            x = np.array([r[f"k{k}"][key] for r in R]); x = x[np.isfinite(x)]
            return (float(x.mean()), float(x.min()), float(x.max())) if len(x) else (float("nan"),) * 3
        def mv(side, m, k):
            return float(np.mean([r[f"k{k}"][side][m] for r in R]))
        sw80, ss80, sw1, ss1 = ms("share_wolpaw", 80), ms("share_soft", 80), ms("share_wolpaw", 1), ms("share_soft", 1)
        nz = float(np.mean([r["noise_k80"]["soft_bits"] for r in R if r["noise_k80"]]))
        summ[v] = dict(n=len(R), k80=dict(wolpaw_var=mv("var", "bits", 80), wolpaw_full=mv("full", "bits", 80), soft_var=mv("var", "soft_bits", 80), soft_full=mv("full", "soft_bits", 80),
                       share_wolpaw=sw80, share_soft=ss80), k1=dict(wolpaw_var=mv("var", "bits", 1), wolpaw_full=mv("full", "bits", 1), soft_var=mv("var", "soft_bits", 1), soft_full=mv("full", "soft_bits", 1), share_wolpaw=sw1, share_soft=ss1), noise_soft_k80=nz)
        lines.append(f"| {v} | {len(R)} | {mv('var','bits',80):.2f} / {mv('full','bits',80):.2f} | {sw80[0]:.2f} ({sw80[1]:.2f}-{sw80[2]:.2f}) | {mv('var','soft_bits',80):.2f} / {mv('full','soft_bits',80):.2f} | {ss80[0]:.2f} ({ss80[1]:.2f}-{ss80[2]:.2f}) | {sw1[0]:.2f} ({sw1[1]:.2f}-{sw1[2]:.2f}) | {ss1[0]:.2f} ({ss1[1]:.2f}-{ss1[2]:.2f}) | {nz:.2f} |")
    lines += ["", "share = variant yield / full-input CLIP NICE-GA yield of the same subject and seed; k=1 is the mean of 5 single-trial draws (same draw sequence for variant and full).",
              "dinov2 / pixpca / randtarget are target swaps: their 'share' compares a DINOv2 / pixel-PCA / random-target 200-way retrieval with the CLIP one of the same EEG and seed."]
    (OUT / "s1_table.md").write_text("\n".join(lines) + "\n"); (OUT / "s1_table.json").write_text(json.dumps(dict(summary=summ, per_checkpoint=per), indent=1))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    t0 = time.time(); files = sorted(EB.glob("retrain_*_sub-*_seed*.json")); done = 0
    for f in files:
        m = re.match(r"retrain_(\w+?)_(sub-\d+)_seed(\d+)\.json", f.name)
        if not m:
            continue
        v, s, seed = m.group(1), m.group(2), int(m.group(3))
        if v not in VARIANTS or (only and v != only):
            continue
        try:
            full_reference(s, seed); variant_rescore(v, s, seed); done += 1
        except Exception as e:
            print(f"S1 ERROR {v} {s} {seed}: {type(e).__name__}: {e}", flush=True)
        print(f"S1 {v} {s} seed{seed} done ({time.time()-t0:.0f}s)", flush=True)
    s1_aggregate(); print(f"S1 finished {done} checkpoints in {time.time()-t0:.0f}s", flush=True)
