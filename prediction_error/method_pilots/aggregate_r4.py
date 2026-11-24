"""Merge the three r4 ranges and recompute the ensemble efficiency under measured ceilings."""
import json, glob, numpy as np
R = "/path/to/eegaudit/results/method_pilots"
rows = []; incr_src = None
seen = set()
for f in sorted(glob.glob(f"{R}/r4_partial_*-*.json")):
    for r in json.load(open(f)):
        if r["subject"] not in seen: rows.append(r); seen.add(r["subject"])
rho_eff = json.load(open("/path/to/eegaudit/results/worth_ceiling_RWD_v2/results.json"))["decoders"]["ens"]["rho_eff"]
rows.sort(key=lambda r: r["subject"])
C = json.load(open("/path/to/eegaudit/results/ceiling_RWD_v2/results.json"))
keys = sorted(C["joint"], key=float); rhos = [float(k) for k in keys]; cs = [C["joint"][k]["exact"]["logit_alpha"]["contraction"] for k in keys]
bits = lambda c: -0.5 * np.log2(1 - c)
def incr(rho):
    if rho <= 0.9: return bits(np.interp(rho, rhos, cs)) - bits(cs[0])
    s = lambda r: r * r / (1 - r * r); A, B = 8.15, 9.38
    return 0.5 * np.log2(A + B * s(rho)) - 0.5 * np.log2(A)
print(f"n participants = {len(rows)}; rho_eff(ens) = {rho_eff:.3f}; ens increment = {incr(rho_eff):.3f} bits")
print("| participant | r (full CV) | r half-data | r12 half-decoders | rel_dec (SB) | rho_max_dec | PE rel across posterior draws | rho_max_target | posterior sd (logit a, logit d, log b, kappa) |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['subject']} | {r['r_full']:.3f} | {r['r_half_data']:.3f} | {r['r12_half_decoders']:.3f} | {r['rel_decoder_sb']:.3f} | {r['rho_max_decoder']:.3f} | {r['rel_target_pairs']:.3f} | {r['rho_max_target']:.3f} | {[round(x,2) for x in r['post_sd']]} |")
K = ("r_full", "r_half_data", "r12_half_decoders", "rel_decoder_sb", "rho_max_decoder", "rel_target_pairs", "rho_max_target")
A = {k: np.array([r[k] for r in rows]) for k in K}
print("\nmean (sd) [min, max]:")
for k in K: print(f"  {k}: {A[k].mean():.3f} ({A[k].std(ddof=1):.3f}) [{A[k].min():.3f}, {A[k].max():.3f}]")
print("\nEfficiency of the ensemble (increment / ceiling increment) under each ceiling:")
print("| ceiling | rho_c | ceiling increment (bits) | efficiency |"); print("|---|---|---|---|")
for name, rc in (("convention (paper)", 0.9), ("rho_max target, mean", A["rho_max_target"].mean()), ("rho_max target, min", A["rho_max_target"].min()),
                 ("rho_max decoder, mean", A["rho_max_decoder"].mean()), ("rho_max decoder, median", float(np.median(A["rho_max_decoder"]))),
                 ("rho_max decoder, min", A["rho_max_decoder"].min()), ("rho_max decoder, max", A["rho_max_decoder"].max()),
                 ("product of the two, mean", (A["rho_max_target"] * A["rho_max_decoder"]).mean())):
    print(f"| {name} | {rc:.3f} | {incr(rc):.3f} | {100*incr(rho_eff)/incr(rc):.1f} % |")
# per-participant efficiency with a per-participant decoder ceiling, using the participant's own r as its coupling (no null correction here)
pe = [100 * incr(min(r["r_full"], 0.999)) / incr(r["rho_max_decoder"]) for r in rows if r["r_full"] > 0]
print(f"\nPer-participant efficiency with own coupling r and own decoder ceiling: mean {np.mean(pe):.1f} %, median {np.median(pe):.1f} %, range {min(pe):.1f}-{max(pe):.1f} %")
print(f"Spearman-Brown check: r_full vs r_half_data mean ratio {np.mean(A['r_full']/np.maximum(A['r_half_data'],1e-6)):.2f}")
