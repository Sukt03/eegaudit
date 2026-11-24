"""Measured learning-target ceiling (methodology round 4): the read-out's own split-half reliability bounds the attainable coupling,
rho_max = sqrt(Spearman-Brown reliability of two half-data read-outs). Combines the per-participant rows of
results/method_pilots/r4_*.json (23 participants) and prices the ensemble increment against the ceiling increment at
rho_max (mean and median), using the same exact joint curve (ceiling_RWD_v2) and bits definition as price_list.py.
Output: results/method_pilots/r4_summary.json"""
import glob, json, sys
from pathlib import Path
import numpy as np
ROOT = Path("/path/to/eegaudit"); sys.path.insert(0, str(ROOT))
from worth_ceiling import load_curve
rows = {}
for f in sorted(glob.glob(str(ROOT / "results/method_pilots/r4_*.json"))):
    if "smoke" in f: continue
    d = json.load(open(f))
    if isinstance(d, dict):
        if "efficiency" in d:
            ens_inc = d["efficiency"]["convention_0.9"]["ens_increment_bits"]; rho_eff = d["rho_eff_ens"]
        rr = d.get("rows", [])
    else:
        rr = d                                                     # partial file: a bare list of per-participant rows
    for r in rr:
        rows[r["subject"]] = r
rm = np.array([r["rho_max_decoder"] for r in rows.values()]); assert len(rm) == 23, len(rm)
rhos, cur, *_ = load_curve(ROOT / "results/ceiling_RWD_v2/results.json"); c = cur["logit_alpha"]; bits = -0.5 * np.log2(1 - c); inc = bits - bits[0]
def inc_at(r): return float(np.interp(r, rhos, inc))
out = dict(n_participants=int(len(rm)), rho_eff_ens=rho_eff, ens_increment_bits=ens_inc,
           rho_max_decoder_mean=float(rm.mean()), rho_max_decoder_median=float(np.median(rm)), rho_max_decoder_sd=float(rm.std(ddof=1)),
           rho_max_decoder_min=float(rm.min()), rho_max_decoder_max=float(rm.max()), n_below_0_9=int((rm < 0.9).sum()),
           ceiling_increment_at_mean_rho_max=inc_at(rm.mean()), efficiency_at_mean_rho_max_pct=100 * ens_inc / inc_at(rm.mean()),
           ceiling_increment_at_median_rho_max=inc_at(np.median(rm)), efficiency_at_median_rho_max_pct=100 * ens_inc / inc_at(np.median(rm)),
           efficiency_convention_pct=100 * ens_inc / inc_at(0.9))
(ROOT / "results/method_pilots/r4_summary.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
