"""R5: measured-ceiling efficiency for EEGNet LOPO (the lock's primary learning decoder), re-analysis of existing files.
Part A (decoder-independent): target reliability rho_max_target per participant from the r4 partial files (PE series under posterior
draws), ceiling increment incr(rho_max_target) vs EEGNet's increment incr(rho_eff_eegnet) (rho_eff from worth_ceiling_RWD_v2).
Part B (decoder determination) for EEGNet needs two independently trained EEGNet decoders per fold; the cached EEGNet artefacts
(results/features/eegnet/sub-XXX.npz) hold only the LOPO prediction (476 x 1), so part B is reported for the ens read-out (r4) and
flagged as not measured for EEGNet unless the split-half EEGNet run below was done (field 'eegnet_split_half': null)."""
import json, glob, numpy as np
R = "/path/to/eegaudit/results"
rows = []; seen = set()
for f in sorted(glob.glob(f"{R}/method_pilots/r4_partial_*-*.json")):
    for r in json.load(open(f)):
        if r["subject"] not in seen: rows.append(r); seen.add(r["subject"])
rows.sort(key=lambda r: r["subject"])
W = json.load(open(f"{R}/worth_ceiling_RWD_v2/results.json"))["decoders"]
C = json.load(open(f"{R}/ceiling_RWD_v2/results.json"))
keys = sorted(C["joint"], key=float); rhos = [float(k) for k in keys]; cs = [C["joint"][k]["exact"]["logit_alpha"]["contraction"] for k in keys]
bits = lambda c: -0.5 * np.log2(1 - c)
def incr(rho):
    if rho <= 0.9: return float(bits(np.interp(rho, rhos, cs)) - bits(cs[0]))
    s = lambda r: r * r / (1 - r * r); A, B = 8.15, 9.38
    return float(0.5 * np.log2(A + B * s(rho)) - 0.5 * np.log2(A))
A = {k: np.array([r[k] for r in rows]) for k in ("rho_max_target", "rho_max_decoder", "r_full")}
out = {"n_participants": len(rows), "source_rows": "results/method_pilots/r4_partial_*.json (decoder=ens for part B; part A is decoder-independent)",
       "rho_max_target": {"mean": float(A["rho_max_target"].mean()), "median": float(np.median(A["rho_max_target"])), "min": float(A["rho_max_target"].min()), "sd": float(A["rho_max_target"].std(ddof=1))},
       "rho_max_decoder_ens": {"mean": float(A["rho_max_decoder"].mean()), "median": float(np.median(A["rho_max_decoder"]))}, "decoders": {}}
for dec in ("eegnet", "ens", "raw"):
    rho = W[dec]["rho_eff"]; lo, hi = W[dec]["rho_ci"]; d = {"rho_eff": rho, "rho_ci": [lo, hi], "increment_bits": incr(rho), "increment_bits_ci": [incr(lo), incr(hi)], "efficiency_pct": {}}
    for name, rc in (("convention_0.9", 0.9), ("rho_max_target_mean", out["rho_max_target"]["mean"]), ("rho_max_target_median", out["rho_max_target"]["median"]),
                     ("rho_max_target_min", out["rho_max_target"]["min"]), ("rho_max_decoder_ens_mean", out["rho_max_decoder_ens"]["mean"]), ("rho_max_decoder_ens_median", out["rho_max_decoder_ens"]["median"])):
        d["efficiency_pct"][name] = {"ceiling_rho": rc, "ceiling_increment_bits": incr(rc), "efficiency": 100 * incr(rho) / incr(rc), "efficiency_ci": [100 * incr(lo) / incr(rc), 100 * incr(hi) / incr(rc)]}
    d["rho_eff_below_rho_max_target_min"] = bool(rho < out["rho_max_target"]["min"]); d["rho_ci_hi_below_rho_max_target_min"] = bool(hi < out["rho_max_target"]["min"])
    out["decoders"][dec] = d
out["eegnet_split_half"] = None
json.dump(out, open(f"{R}/loop/r5_eegnet_ceiling/r5_eegnet_ceiling_partA.json", "w"), indent=1)
e = out["decoders"]["eegnet"]
print(f"n={len(rows)} rho_max_target mean {out['rho_max_target']['mean']:.3f} median {out['rho_max_target']['median']:.3f} min {out['rho_max_target']['min']:.3f}")
print(f"EEGNet rho_eff {e['rho_eff']:.3f} [{e['rho_ci'][0]:.3f},{e['rho_ci'][1]:.3f}] increment {e['increment_bits']:.3f} bits")
for k, v in e["efficiency_pct"].items(): print(f"  {k}: rho_c {v['ceiling_rho']:.3f} ceiling {v['ceiling_increment_bits']:.3f} bits -> efficiency {v['efficiency']:.1f}% [{v['efficiency_ci'][0]:.1f},{v['efficiency_ci'][1]:.1f}]")
print("ens convention", round(out["decoders"]["ens"]["efficiency_pct"]["convention_0.9"]["efficiency"], 1), "ens at rho_max_decoder mean", round(out["decoders"]["ens"]["efficiency_pct"]["rho_max_decoder_ens_mean"]["efficiency"], 1))
