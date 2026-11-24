"""Monte Carlo spread of the exact-ceiling quantities across seeds (RUN-1 of EXPERIMENTS_todo_20261120.md).
Reads results/ceiling_RWD_v2/results.json (seed 7) and results/todo_pilots/ceiling_RWD_v2_seed*_rho*/results.json (seeds 8, 9),
reports per seed: bits at rho 0 / 0.3 / 0.4 / 0.9, ceiling increment, ensemble increment at rho_eff (interpolated 0.3-0.4 as
price_list.py does) and efficiency; then SD across seeds. Run with python -I; writes results/todo_pilots/mc_summary.json."""
import glob, json, math, sys
import numpy as np
P = "/path/to/eegaudit"
RHO_ENS = 0.3773  # science-ensemble rho_eff (results/track_decision/price_list.json)
bits = lambda c: -0.5 * math.log2(1 - c)

def c_of(d, rho, p="logit_alpha"):
    j = d["joint"]
    k = str(rho) if str(rho) in j else f"{rho:g}"
    return j[k]["exact"][p]["contraction"] if k in j else None

seeds = {7: json.load(open(f"{P}/results/ceiling_RWD_v2/results.json"))}
parts = {}
for f in glob.glob(f"{P}/results/todo_pilots/ceiling_RWD_v2_seed*_rho*/results.json"):
    s = int(f.split("seed")[1].split("_")[0]); d = json.load(open(f))
    parts.setdefault(s, {"joint": {}, "behaviour": d["behaviour"]})["joint"].update(d["joint"])
seeds.update(parts)
rows = {}
for s, d in sorted(seeds.items()):
    r = {}
    for rho in (0.0, 0.3, 0.4, 0.9):
        c = c_of(d, rho); r[f"bits_rho{rho}"] = bits(c) if c is not None else None
    if r["bits_rho0.0"] is not None and r["bits_rho0.9"] is not None:
        r["ceiling_increment"] = r["bits_rho0.9"] - r["bits_rho0.0"]
    if all(r.get(k) is not None for k in ("bits_rho0.0", "bits_rho0.3", "bits_rho0.4")):
        c3, c4 = c_of(d, 0.3), c_of(d, 0.4)
        c_ens = float(np.interp(RHO_ENS, [0.3, 0.4], [c3, c4]))
        r["ens_increment"] = bits(c_ens) - r["bits_rho0.0"]
        if "ceiling_increment" in r:
            r["ens_efficiency"] = r["ens_increment"] / r["ceiling_increment"]
    r["ess_rho0.9"] = d["joint"].get("0.9", {}).get("ess_median"); r["ess_rho0"] = d["joint"].get("0.0", {}).get("ess_median")
    rows[s] = r
summary = {"per_seed": rows, "sd_across_seeds": {}, "mean_across_seeds": {}, "n_seeds": {}}
for key in ("bits_rho0.0", "bits_rho0.9", "ceiling_increment", "ens_increment", "ens_efficiency"):
    v = [r[key] for r in rows.values() if r.get(key) is not None]
    if len(v) >= 2:
        summary["sd_across_seeds"][key] = float(np.std(v, ddof=1)); summary["mean_across_seeds"][key] = float(np.mean(v)); summary["n_seeds"][key] = len(v)
json.dump(summary, open(f"{P}/results/todo_pilots/mc_summary.json", "w"), indent=1)
for s, r in rows.items():
    print("seed", s, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
print("SD across seeds:", {k: round(v, 4) for k, v in summary["sd_across_seeds"].items()}, "n:", summary["n_seeds"])
