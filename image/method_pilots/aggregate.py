"""Aggregate the method pilots (rounds 1-3) into the tables for outputs/METHOD_ITERATIONS_20261120.md."""
import json, glob
import numpy as np
R = "/path/to/eegaudit/results/method_pilots"
EST = ("bits", "soft_bits", "two_way_adj")
def adj(r): r = dict(r); r["two_way_adj"] = r["two_way"] - 0.5; return r
def mean_rows(rows, **kw):
    v = [adj(r) for r in rows if all(r.get(a) == b for a, b in kw.items())]
    return {e: float(np.mean([r[e] for r in v])) for e in EST + ("top1", "nn_two_way")} if v else None

print("# Round 1: input-side PCA bottleneck through frozen NICE-GA (k = 80), fraction of full-width yield")
print("| subject | basis | L | var expl | top-1 | Wolpaw bits (frac) | soft bits (frac) | 2-way adj (frac) | nn 2-way |"); print("|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(f"{R}/r1_input_bottleneck_*.json")):
    if "smoke" in f: continue
    d = json.load(open(f)); full = adj([r for r in d["rows"] if r["L"] == "full" and r["k"] == 80][0])
    for r in d["rows"]:
        if r["condition"] != "eeg" or r["k"] != 80 or r["L"] == "full": continue
        r = adj(r)
        print(f"| {d['subject']} | {d['basis_subject']} | {r['L']} | {r['var_explained']:.2f} | {r['top1']:.3f} | {r['bits']:.2f} ({r['bits']/full['bits']:.2f}) | {r['soft_bits']:.2f} ({r['soft_bits']/full['soft_bits']:.2f}) | {r['two_way_adj']:.3f} ({r['two_way_adj']/full['two_way_adj']:.2f}) | {r['nn_two_way']:.2f} |")
    print(f"| {d['subject']} | {d['basis_subject']} | full | 1.00 | {full['top1']:.3f} | {full['bits']:.2f} (1.00) | {full['soft_bits']:.2f} (1.00) | {full['two_way_adj']:.3f} (1.00) | {full['nn_two_way']:.2f} |")
    ctrl = max(r["top1"] for r in d["rows"] if r["condition"] in ("noise", "shuffled_pair"))
    print(f"| {d['subject']} | {d['basis_subject']} | controls max top-1 over L | | {ctrl:.3f} | | | | |")
print("\nL at which 90 % of full-width yield is first reached (k = 80):")
for f in sorted(glob.glob(f"{R}/r1_input_bottleneck_*.json")):
    if "smoke" in f: continue
    d = json.load(open(f)); full = adj([r for r in d["rows"] if r["L"] == "full" and r["k"] == 80][0])
    out = {}
    for e in EST:
        out[e] = next((r["L"] for r in d["rows"] if r["condition"] == "eeg" and r["k"] == 80 and r["L"] != "full" and adj(r)[e] / full[e] >= 0.9), ">4000")
    print(f"  {d['subject']} basis {d['basis_subject']}: L90 Wolpaw={out['bits']}  soft={out['soft_bits']}  2-way adj={out['two_way_adj']}")

print("\n# Round 2: estimator dependence of the price axes (NICE-GA, mean_loo fill)")
files = [f for f in sorted(glob.glob(f"{R}/r2_estimator_axes_*.json")) if "smoke" not in f]
KS = [1, 2, 4, 8, 16, 32, 80]; SOA = 0.2
per = []
for f in files:
    d = json.load(open(f)); rows = d["rows"]; e = {}
    for k in KS:
        e[("rep", k)] = mean_rows(rows, axis="repetitions", order="avg_then_encode", k=k)
        if k > 1: e[("repE", k)] = mean_rows(rows, axis="repetitions", order="encode_then_avg", k=k)
    for ch in ("all63", "occpar32"):
        for w in (0.2, 0.4, 0.6, 1.0):
            e[("acq", ch, w)] = mean_rows(rows, axis="acquisition", channels=ch, window=w, k=80)
            e[("acq1", ch, w)] = mean_rows(rows, axis="acquisition", channels=ch, window=w, k=1)
    e["name"] = f"{d['subject']} s{d['seed']}"; per.append(e)
print(f"files: {[p['name'] for p in per]}")
print("\n## Repetitions: mean over files of bits per query and per second (k x 0.2 s), under each estimator")
print("| k | Wolpaw bits | Wolpaw bits/s | soft bits | soft bits/s | 2-way adj | encode-then-avg Wolpaw | encode-then-avg soft |"); print("|---|---|---|---|---|---|---|---|")
for k in KS:
    a = {e: np.mean([p[("rep", k)][e] for p in per]) for e in EST}
    b = {e: np.mean([p[("repE", k)][e] for p in per]) for e in EST} if k > 1 else a
    print(f"| {k} | {a['bits']:.3f} | {a['bits']/(k*SOA):.3f} | {a['soft_bits']:.3f} | {a['soft_bits']/(k*SOA):.3f} | {a['two_way_adj']:.3f} | {b['bits']:.3f} ({b['bits']-a['bits']:+.3f}) | {b['soft_bits']:.3f} ({b['soft_bits']-a['soft_bits']:+.3f}) |")
print("\nOptimum k for yield per second, per file (Wolpaw / soft):")
for p in per:
    kw = max(KS, key=lambda k: p[("rep", k)]["bits"] / k); ks = max(KS, key=lambda k: p[("rep", k)]["soft_bits"] / k)
    print(f"  {p['name']}: k*={kw} / {ks};  ratio k=80/k=1: Wolpaw {p[('rep',80)]['bits']/p[('rep',1)]['bits']:.1f}x, soft {p[('rep',80)]['soft_bits']/p[('rep',1)]['soft_bits']:.1f}x, 2-way adj {p[('rep',80)]['two_way_adj']/p[('rep',1)]['two_way_adj']:.1f}x")
print("\n## Window / electrodes at k = 80: share of the 1.0 s, 63-ch yield under each estimator (mean over files; min-max)")
print("| condition | Wolpaw share | soft share | 2-way adj share | nn 2-way (abs) |"); print("|---|---|---|---|---|")
for ch in ("all63", "occpar32"):
    for w in (0.2, 0.4, 0.6, 1.0):
        sh = {e: [p[("acq", ch, w)][e] / p[("acq", "all63", 1.0)][e] for p in per] for e in EST}
        nn = np.mean([p[("acq", ch, w)]["nn_two_way"] for p in per])
        print(f"| {ch} {w} s | " + " | ".join(f"{np.mean(sh[e]):.2f} ({min(sh[e]):.2f}-{max(sh[e]):.2f})" for e in EST) + f" | {nn:.3f} |")
print("\nSmallest window reaching 90 % of the 1.0-s yield (63 ch), per file: Wolpaw / soft / 2-way adj")
for p in per:
    o = []
    for e in EST:
        o.append(next((w for w in (0.2, 0.4, 0.6, 1.0) if p[("acq", "all63", w)][e] / p[("acq", "all63", 1.0)][e] >= 0.9), None))
    print(f"  {p['name']}: {o}")
print("\n## Same at k = 1 (single trial; mean over draws): share of the 1.0 s, 63-ch k=1 yield")
print("| condition | Wolpaw share | soft share | 2-way adj share |"); print("|---|---|---|---|")
for ch in ("all63", "occpar32"):
    for w in (0.2, 0.4, 0.6, 1.0):
        sh = {e: [p[("acq1", ch, w)][e] / p[("acq1", "all63", 1.0)][e] for p in per] for e in EST}
        print(f"| {ch} {w} s | " + " | ".join(f"{np.mean(sh[e]):.2f} ({min(sh[e]):.2f}-{max(sh[e]):.2f})" for e in EST) + " |")

print("\n# Round 3: masking fill (NICE-GA seed 141, k = 80, all 200 queries; mean over 3 subjects)")
files = [f for f in sorted(glob.glob(f"{R}/r3_fill_*.json")) if "smoke" not in f]
ds = [json.load(open(f)) for f in files]
conds = [("all63", 0.2), ("all63", 0.4), ("all63", 0.6), ("occpar32", 1.0), ("occpar32", 0.4)]
for queries in ("all200", "q100-199"):
    print(f"\n## queries = {queries}: Wolpaw bits / soft bits / 2-way (mean over subjects)")
    print("| fill | " + " | ".join(f"{c} {w}" for c, w in conds) + " |"); print("|---|" + "---|" * len(conds))
    for fill in ("none", "zero", "mean_loo", "mean_half", "mean_othersub", "mean_10rep"):
        cells = []
        for c, w in conds:
            if fill == "none": c, w = "all63", 1.0
            v = [mean_rows(d["rows"], fill=fill, channels=c, window=w, k=80, queries=queries) for d in ds]
            cells.append(f"{np.mean([x['bits'] for x in v]):.2f} / {np.mean([x['soft_bits'] for x in v]):.2f} / {np.mean([x['two_way_adj'] for x in v])+0.5:.3f}")
        print(f"| {fill} | " + " | ".join(cells) + " |")
print("\nPaired difference mean_half - mean_loo on queries 100-199 (the fill never saw these images), per subject and condition: Wolpaw / soft bits")
for d in ds:
    o = []
    for c, w in conds:
        a = mean_rows(d["rows"], fill="mean_half", channels=c, window=w, k=80, queries="q100-199"); b = mean_rows(d["rows"], fill="mean_loo", channels=c, window=w, k=80, queries="q100-199")
        o.append(f"{a['bits']-b['bits']:+.3f}/{a['soft_bits']-b['soft_bits']:+.3f}")
    print(f"  {d['subject']}: {o}")
print("\nk = 1 (mean over 5 draws, all 200 queries): soft bits by fill")
print("| fill | " + " | ".join(f"{c} {w}" for c, w in conds) + " |"); print("|---|" + "---|" * len(conds))
for fill in ("zero", "mean_loo", "mean_half", "mean_othersub", "mean_10rep"):
    cells = []
    for c, w in conds:
        v = [mean_rows(d["rows"], fill=fill, channels=c, window=w, k=1, queries="all200") for d in ds]
        cells.append(f"{np.mean([x['bits'] for x in v]):.3f} / {np.mean([x['soft_bits'] for x in v]):.2f}")
    print(f"| {fill} | " + " | ".join(cells) + " |")
