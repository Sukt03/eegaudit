"""Means over subjects of the estimator comparison (audit A) so the ledger can register them."""
import json, numpy as np
src = "/path/to/eegaudit/outputs/PROTOCOL_AUDIT_A_estimators_20261120-0007.json"
d = json.load(open(src))
full = d["full"]; out = {"n_subjects": len(full), "source": src}
for k in ["top1", "wolpaw", "mi_soft", "two_way", "two_way_bits", "mean_rank"]:
    v = np.array([r[k] for r in full]); out[k + "_mean"] = float(v.mean()); out[k + "_sd"] = float(v.std(ddof=1))
out["mi_soft_eff200"] = out["mi_soft_mean"] / np.log2(200); out["wolpaw_eff200"] = out["wolpaw_mean"] / np.log2(200)
for N in [50, 100, 200]:
    rows = [r for r in d["subsample"] if r["N"] == N]
    if rows:
        out[f"soft_eff_N{N}"] = float(np.mean([r["mi_soft"] for r in rows]) / np.log2(N))
        out[f"wolpaw_eff_N{N}"] = float(np.mean([r["wolpaw"] for r in rows]) / np.log2(N))
json.dump(out, open("/path/to/eegaudit/results/protocol_audit/A_summary.json", "w"), indent=1); print(out)
