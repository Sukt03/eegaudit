"""L4: ds004595 replication with EEGNet LOPO next to the ensemble. Re-analysis only: reads the existing group_stats files
(results/group_stats_ds004595: 50 participants; results/group_stats: ds003458, 23) and the exact ceiling curves, and writes
one table: coupling with CI, prevalence, individually significant participants, paired ens-eegnet difference, increment in
bits at the coupling (ds004595 curve where present, else flagged), and the ratio ds004595/ds003458 per decoder."""
import json, numpy as np
from pathlib import Path
ROOT = Path("/path/to/eegaudit"); OUT = ROOT / "results/loop/learner"; OUT.mkdir(parents=True, exist_ok=True)
import sys; sys.path.insert(0, str(ROOT / "experiments/loop/learner")); from common import make_incr
G5 = json.load(open(ROOT / "results/group_stats_ds004595/results.json")); G3 = json.load(open(ROOT / "results/group_stats/results.json"))
curves = {"ds003458": ROOT / "results/ceiling_RWD_v2/results.json", "ds004595": ROOT / "results/ceiling_RWD_ds004595/results.json"}
inc = {}
for k, p in curves.items():
    try: inc[k] = make_incr(p)
    except Exception as e: inc[k] = None; print("no curve for", k, e)
res = {"sources": {"ds004595": "results/group_stats_ds004595/results.json", "ds003458": "results/group_stats/results.json"}, "rows": {}}
for ds, G in (("ds003458", G3), ("ds004595", G5)):
    for dec in ("ens", "eegnet", "raw", "eegnet_post"):
        if dec not in G["decoders"]: continue
        D = G["decoders"][dec]; row = {"n": len(D["r"]), "rho_eff": D["rho_eff_pooled"], "rho_ci": D["rho_eff_ci"], "mean_r": float(np.mean(D["r"])), "gamma0": D["prevalence"]["gamma0"],
                                     "p_global": D["prevalence"]["p_global"], "p_majority": D["prevalence"]["p_majority"], "indiv": D["n_individually_above_95pct_null"], "tau2": D["meta"]["tau2"]}
        if inc[ds]: f, rmax = inc[ds]; row["increment_bits"] = f(D["rho_eff_pooled"]); row["increment_bits_ci"] = [f(x) for x in D["rho_eff_ci"]]; row["curve_max_rho"] = rmax
        res["rows"][f"{ds}/{dec}"] = row
    res["rows"][f"{ds}/pair ens-eegnet"] = G["pairs"].get("ens-eegnet")
for dec in ("ens", "eegnet", "raw"):
    a, b = res["rows"][f"ds004595/{dec}"], res["rows"][f"ds003458/{dec}"]
    res["rows"][f"ratio ds004595/ds003458 {dec}"] = {"rho_ratio": a["rho_eff"] / b["rho_eff"], "bits_ratio": (a["increment_bits"] / b["increment_bits"]) if "increment_bits" in a and "increment_bits" in b else None}
res["real_verify_eegnet"] = "not available: results/real_verify_ds004595_* exist for ens only"
(OUT / "l4_ds004595_eegnet.json").write_text(json.dumps(res, indent=1))
for k, v in res["rows"].items(): print(k, json.dumps(v)[:300])
