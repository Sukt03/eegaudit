"""Post-hoc usable-budget and alignment-clarity pass over saved results (no training). 22 Nov 2026.
A. UBP image: raw vs baseline-subtracted (random-init ridge twin) yield at k=40 and k=80 repeats, per participant; fraction lost to the twin;
   saturation (ratio k80/k40, raw vs net). The noise-input twin was never run for UBP (source retrain = {}) -> null. 'twin' columns below are the untrained-encoder baseline, not a noise-input twin.
B. NICE-GA image (p1_rank_bound): raw yield at k=1 vs k=80 repeats, no twin saved -> null for twin columns.
C. PE (ds003458) eye-quiet control: kept ratio and quiet-minus-random for the two eye rules, read from summary_b1.json.
D. Alignment clarity (true - within-swap, within - across, true - twin): needs per-trial predictions that are not on disk -> null.
Descriptive rule A (stated post hoc, 22 Nov; the earlier 0.02 rule could not fire): the baseline matters if its share of raw yield has CI lower bound > 0.05; saturation differs if the CI of (net - raw) ratio excludes 0. Not a pre-registered falsifier. Part C uses L64 only; the secondary eye rule has n=7 of 23 and comparator overlap 0.48. Part B averages 5 k1 draws per participant; k1 and k80 share the test pool.
Provenance: UBP is trained in-dataset on THINGS-EEG2 (not a pretrained foundation model); no FM rows here.
Usage: srun ... python eu_bac_workshop.py [--n 3] --out PATH
"""
import argparse, hashlib, json, sys
from pathlib import Path
import torch

R = Path("/path/to/eegaudit/results")
dev = "cuda"
assert torch.cuda.is_available(), "GPU required"


def boot(x, B=10000, seed=0):
    x = torch.tensor(x, dtype=torch.float64, device=dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    idx = torch.randint(0, len(x), (B, len(x)), device=dev, generator=g)
    m = x[idx].mean(1)
    q = torch.quantile(m, torch.tensor([0.025, 0.975], dtype=torch.float64, device=dev))
    return {"mean": x.mean().item(), "lo": q[0].item(), "hi": q[1].item(), "n": len(x)}


def ratio_boot(num, den, B=10000, seed=1):
    n = torch.tensor(num, dtype=torch.float64, device=dev); d = torch.tensor(den, dtype=torch.float64, device=dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    idx = torch.randint(0, len(n), (B, len(n)), device=dev, generator=g)
    r = n[idx].mean(1) / d[idx].mean(1)
    q = torch.quantile(r, torch.tensor([0.025, 0.975], dtype=torch.float64, device=dev))
    return {"ratio_of_means": (n.mean() / d.mean()).item(), "lo": q[0].item(), "hi": q[1].item()}


ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=0); ap.add_argument("--out", required=True)
a = ap.parse_args()
out = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "smoke_n": a.n or None}

# A. UBP
subs = {}
for f in ("ubp_card_full_a.json", "ubp_card_full_b.json"):
    p = R / "baselines/ubp" / f
    if p.exists():
        subs.update(json.load(open(p))["subjects"])
ids = sorted(subs, key=int)
if a.n: ids = ids[: a.n]
rel = [subs[i]["released"] for i in ids]; tw = [subs[i]["random_init_ridge"] for i in ids]
raw = {k: [r[f"k{k}_bits"] for r in rel] for k in (40, 80)}
twin = {k: [t[f"k{k}_bits"] for t in tw] for k in (40, 80)}
net = {k: [x - y for x, y in zip(raw[k], twin[k])] for k in (40, 80)}
A = {"participants": ids, "twin": "random_init_ridge only; noise-input twin not saved (null)",
     "raw": {k: boot(raw[k]) for k in raw}, "baseline_yield": {k: boot(twin[k]) for k in twin},
     "net": {k: boot(net[k]) for k in net},
     "baseline_fraction": {k: ratio_boot(twin[k], raw[k]) for k in raw}}
sat_raw = [r80 / r40 for r40, r80 in zip(raw[40], raw[80])]
sat_net = [n80 / n40 for n40, n80 in zip(net[40], net[80])]
A["saturation_ratio_k80_over_k40"] = {"raw": boot(sat_raw), "net": boot(sat_net),
                                       "diff_net_minus_raw": boot([b - c for b, c in zip(sat_net, sat_raw)])}
sd = A["saturation_ratio_k80_over_k40"]["diff_net_minus_raw"]; fal = A["baseline_fraction"][80]["lo"] <= 0.05 and sd["lo"] <= 0 <= sd["hi"]
A["rule_A_baseline_irrelevant"] = bool(fal)
A["verdict"] = "baseline share <=5% and no saturation shift" if fal else "baseline removes >5% of yield and/or shifts saturation (descriptive, post hoc)"
out["A_ubp_budget"] = A

# B. NICE-GA p1 k1 vs k80
j = json.load(open(R / "loop/image_p1/p1_rank_bound.json"))["per_participant"]["nice_ga"]
sid = sorted(j)[: a.n or None]
k1 = [sum(d["I_NCE"] for d in j[s]["k1"]) / len(j[s]["k1"]) for s in sid]
k80 = [j[s]["k80"]["I_NCE"] for s in sid]
out["B_nice_ga_budget"] = {"participants": sid, "k1": boot(k1), "k80": boot(k80), "baseline_columns": None,
                           "note": "no twin or specificity result saved for this read-out"}

# C. PE eye-quiet
sb = json.load(open(R / "ext/eyequiet_v3/summary_b1.json"))
C = {}
for rule in ("primary_eog_mad", "secondary_fp75"):
    L = sb[rule]["L64"]
    C[rule] = {"n_used": sb[rule]["n_used"], "kept_ratio_of_means": L["kept_ratio_of_means"], "kept_ci": L["kept_ci"],
               "quiet_minus_random": L["quiet_minus_random"], "falsifier_triggered": L["falsifier_triggered"]}
out["C_pe_eyequiet_v3"] = C
out["D_alignment_clarity"] = None
out["D_note"] = "per-trial predictions for within/across swaps are not saved on disk; not computed (no placeholder numbers)"
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({"A": {k: A[k] for k in ("raw", "net", "baseline_fraction", "verdict")}, "B": out["B_nice_ga_budget"]["k80"]}, indent=1)[:2500])
