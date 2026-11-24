"""P6 image (ideator 13:35, approved): nuisance-partitioned information from the cached hard-pool files (no new decoding).
results/eegbits/acq_hardpool_nice_ga_<s>_seed<seed>.json (10 participants x 5 seeds; NICE-GA = ours) and acq_hardpool_atm_sub-08_seed0.json (ATM-S, 1 participant).
Per participant (mean over seeds and draws, k = 80): bits in same-category pools (hard) and random pools of the same N (2, 10, 50), shuffled-pair floor
of each; exemplar-specific bits = hard - hard floor; category-shared share = 1 - hard/random. Participant bootstrap (5000) over the 10 NICE-GA participants.
Pass rule: hard-pool bits > floor with participant CI excluding 0. Output: results/loop/p6_nuisance/p6_image.json"""
import json, glob, re
from pathlib import Path
import numpy as np
R = Path("/path/to/eegaudit/results"); OUT = R / "loop/p6_nuisance"


def boot(v):
    rng = np.random.default_rng(0); v = np.asarray(v, float); b = [rng.choice(v, len(v)).mean() for _ in range(5000)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def cell(rows, pool, cond, N, key):
    v = [r[key] for r in rows if r["axis"] == "hardpool" and r["pool"] == pool and r["condition"] == cond and r["N"] == N and r["k"] == 80]
    return float(np.mean(v)) if v else None


per = {}
for f in sorted(glob.glob(str(R / "eegbits/acq_hardpool_*.json"))):
    d = json.load(open(f)); key = f"{d['variant']}_{d['subject']}"; per.setdefault(key, []).append(d["rows"])
Ns = sorted({r["N"] for rs in per.values() for r in rs[0] if r["axis"] == "hardpool"})
res = dict(note=__doc__, Ns=Ns, per_participant={}, summary={})
for key, runs in per.items():
    out = dict(n_seeds=len(runs))
    for N in Ns:
        g = lambda pool, cond, k: float(np.mean([cell(rs, pool, cond, N, k) for rs in runs]))
        hard, rand, fh, fr = g("hard", "eeg", "bits"), g("random", "eeg", "bits"), g("hard", "shuffled_pair", "bits"), g("random", "shuffled_pair", "bits")
        out[f"N{N}"] = dict(hard_bits=hard, random_bits=rand, hard_floor=fh, random_floor=fr, exemplar_bits=hard - fh, category_shared_share=1 - hard / rand if rand > 0 else None,
                            hard_top1=g("hard", "eeg", "top1"), random_top1=g("random", "eeg", "top1"), hard_2way=g("hard", "eeg", "two_way"), random_2way=g("random", "eeg", "two_way"))
    res["per_participant"][key] = out
md = ["| N | NICE-GA hard bits | random bits | hard floor | exemplar bits [CI] | n > 0 | category-shared share [CI] | hard 2-way | random 2-way | ATM-S sub-08 hard / random / share |", "|---|---|---|---|---|---|---|---|---|---|"]
for N in Ns:
    P = [v[f"N{N}"] for k, v in res["per_participant"].items() if k.startswith("nice_ga")]
    ex = [p["exemplar_bits"] for p in P]; sh = [p["category_shared_share"] for p in P]
    a = res["per_participant"].get("atm_sub-08", {}).get(f"N{N}")
    s = dict(n=len(P), hard_bits=float(np.mean([p["hard_bits"] for p in P])), random_bits=float(np.mean([p["random_bits"] for p in P])), hard_floor=float(np.mean([p["hard_floor"] for p in P])),
             exemplar_bits=dict(mean=float(np.mean(ex)), ci95=boot(ex), n_pos=int(np.sum(np.array(ex) > 0))), category_shared_share=dict(mean=float(np.mean(sh)), ci95=boot(sh)),
             hard_2way=float(np.mean([p["hard_2way"] for p in P])), random_2way=float(np.mean([p["random_2way"] for p in P])), atm_sub08=a)
    s["pass"] = bool(s["exemplar_bits"]["ci95"][0] > 0); res["summary"][f"N{N}"] = s
    md.append(f"| {N} | {s['hard_bits']:.3f} | {s['random_bits']:.3f} | {s['hard_floor']:.3f} | {s['exemplar_bits']['mean']:.3f} [{s['exemplar_bits']['ci95'][0]:.3f}, {s['exemplar_bits']['ci95'][1]:.3f}] | {s['exemplar_bits']['n_pos']}/{len(P)} | {s['category_shared_share']['mean']:.2f} [{s['category_shared_share']['ci95'][0]:.2f}, {s['category_shared_share']['ci95'][1]:.2f}] | {s['hard_2way']:.3f} | {s['random_2way']:.3f} | " + (f"{a['hard_bits']:.3f} / {a['random_bits']:.3f} / {a['category_shared_share']:.2f}" if a else "-") + " |")
res["table_md"] = "\n".join(md)
(OUT / "p6_image.json").write_text(json.dumps(res, indent=1)); print("\n".join(md))
