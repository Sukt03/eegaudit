"""Aggregate r6 misspecification twins into one table (per family: contraction ratio, held-out gain) next to the real
data (results/real_verify_eegnet) and the correctly specified twins (results/real_twin_eegnet_a/b/c).
Usage: python r6_aggregate.py g1 g2 g3   -> results/method_pilots/r6_summary.json and r6_summary.md
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("/path/to/eegaudit")
OUT = ROOT / "results/method_pilots"
ORDER = ["correct", "unsigned", "mix_unsigned", "t3", "hetero", "ar1_0.3", "ar1_0.5", "alpha_n-0.6", "alpha_n-1.0",
         "alpha_n+1.0", "alpha_n_rand", "prev_0.5", "prev_1.0"]
LABEL = {"correct": "correctly specified (baseline)", "unsigned": "(i) unsigned |PE|", "mix_unsigned": "(i) signed + unsigned PE",
         "t3": "(ii) Student-t(3) noise", "hetero": "(ii) noise sd grows with |PE|", "ar1_0.3": "(iii) AR(1) noise phi=0.3",
         "ar1_0.5": "(iii) AR(1) noise phi=0.5", "alpha_n-0.6": "(iv) neural logit alpha = behavioural - 0.6",
         "alpha_n-1.0": "(iv) neural logit alpha = behavioural - 1.0", "alpha_n+1.0": "(iv) neural logit alpha = behavioural + 1.0",
         "alpha_n_rand": "(iv) neural alpha independent of behaviour", "prev_0.5": "(v) PE_t + 0.5 PE_{t-1}", "prev_1.0": "(v) PE_t + PE_{t-1}"}


def fam_stats(recs):
    vr = np.array([r["var_ratio"][0] for r in recs]); pr = np.array([r["pred_ratio_at_rho_fit"] for r in recs])
    gn = np.array([r["gain_nats_per_trial"] for r in recs])
    o = {"n_twins": len(recs), "ratio_mean": float(vr.mean()), "ratio_median": float(np.median(vr)), "ratio_se": float(vr.std(ddof=1) / np.sqrt(len(vr))),
         "pred_ratio_mean": float(pr.mean()), "obs_over_pred_mean": float((vr / pr).mean()),
         "gain_mean": float(gn.mean()), "gain_se": float(gn.std(ddof=1) / np.sqrt(len(gn))), "gain_n_pos": int((gn > 0).sum()),
         "gain_wilcoxon_p": float(stats.wilcoxon(gn).pvalue) if np.any(gn != 0) else 1.0,
         "rho_fit_mean": float(np.mean([r["rho_fit"] for r in recs])), "rho_target_mean": float(np.mean([r["rho_target"] for r in recs])),
         "corr_signal_q_mean": float(np.mean([r["corr_signal_q"] for r in recs])), "amp_mean": float(np.mean([r["amp"] for r in recs])),
         "ess_min": float(min(min(r["ess"]) for r in recs)),
         "alpha_shift_joint_minus_beh": float(np.mean([r["mean_joint"][0] - r["mean_beh"][0] for r in recs])),
         "alpha_err_beh": float(np.mean([abs(r["mean_beh"][0] - r["theta"][0]) for r in recs])),
         "alpha_err_joint": float(np.mean([abs(r["mean_joint"][0] - r["theta"][0]) for r in recs]))}
    return o


def main():
    tags = sys.argv[1:]
    recs = {}
    files = []
    for t in tags:
        p = OUT / f"r6_misspec_{t}.json"; files.append(str(p))
        R = json.loads(p.read_text())
        for r in R["twins"].values():
            recs.setdefault(r["family"], []).append(r)
    fams = [f for f in ORDER if f in recs] + [f for f in recs if f not in ORDER]
    S = {f: fam_stats(recs[f]) for f in fams}
    # real-data comparators (same quantities, same subset of participants and all 23)
    V = json.loads((ROOT / "results/real_verify_eegnet/results.json").read_text())
    subs = sorted({r["real"] for f in fams for r in recs[f]})
    real = {}
    for name, keys in (("all_23", list(V["subjects"])), ("pilot_subset", [str(s) for s in subs])):
        vr = np.array([V["subjects"][k]["h1_joint"]["var"][0] / V["subjects"][k]["h1_beh"]["var"][0] for k in keys])
        gn = np.array([(V["subjects"][k]["A_h1_to_h2"]["joint"] - V["subjects"][k]["A_h1_to_h2"]["beh"]) / V["subjects"][k]["A_h1_to_h2"]["n_test"] for k in keys])
        real[name] = {"n": len(keys), "ratio_mean": float(vr.mean()), "ratio_median": float(np.median(vr)), "gain_mean": float(gn.mean()),
                      "gain_n_pos": int((gn > 0).sum()), "rho_mean": float(np.mean([V["subjects"][k]["rho"] for k in keys]))}
    real["summary_C_alpha"] = V["summary"]["C"]["logit_alpha"]
    real["summary_A_h1_to_h2_joint"] = {k: V["summary"]["A_h1_to_h2_joint"][k] for k in ("mean_gain_nats_per_trial", "n_positive", "wilcoxon_p")}
    twins_cs = {}
    for t in ("a", "b", "c"):
        p = ROOT / f"results/real_twin_eegnet_{t}/results.json"
        if p.exists():
            Rt = json.loads(p.read_text()); Ss = list(Rt["subjects"].values())
            vr = np.array([s["A_joint"]["var"][0] / s["A_beh"]["var"][0] for s in Ss])
            gn = np.array([(s["A_joint"]["lpred"] - s["A_beh"]["lpred"]) / s["n_test"] for s in Ss])
            twins_cs[t] = {"n": len(Ss), "ratio_mean": float(vr.mean()), "gain_mean": float(gn.mean()), "gain_n_pos": int((gn > 0).sum())}
    out = {"inputs": files, "pilot_participants": subs, "families": S, "real": real, "correct_twins_existing": twins_cs}
    (OUT / "r6_summary.json").write_text(json.dumps(out, indent=1))
    lines = ["| misspecification | n twins | rho in fit (target) | alpha var ratio obs | predicted at rho_fit | obs/pred | held-out gain (nats/trial) | n gain>0 | Wilcoxon p | alpha shift joint-beh |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for f in fams:
        s = S[f]
        lines.append(f"| {LABEL.get(f, f)} | {s['n_twins']} | {s['rho_fit_mean']:.2f} ({s['rho_target_mean']:.2f}) | {s['ratio_mean']:.2f} (se {s['ratio_se']:.2f}; median {s['ratio_median']:.2f}) | "
                     f"{s['pred_ratio_mean']:.2f} | {s['obs_over_pred_mean']:.2f} | {s['gain_mean']:+.4f} (se {s['gain_se']:.4f}) | {s['gain_n_pos']}/{s['n_twins']} | {s['gain_wilcoxon_p']:.2f} | {s['alpha_shift_joint_minus_beh']:+.2f} |")
    r = real["all_23"]; rp = real["pilot_subset"]
    lines.append(f"| **real data, EEGNet, all 23** | 23 | {r['rho_mean']:.2f} | {r['ratio_mean']:.2f} (median {r['ratio_median']:.2f}) | {real['summary_C_alpha']['predicted_ratio_mean']:.2f} | "
                 f"{r['ratio_mean'] / real['summary_C_alpha']['predicted_ratio_mean']:.2f} | {r['gain_mean']:+.4f} | {r['gain_n_pos']}/23 | {real['summary_A_h1_to_h2_joint']['wilcoxon_p']:.2f} | |")
    lines.append(f"| real data, the {rp['n']} pilot participants | {rp['n']} | {rp['rho_mean']:.2f} | {rp['ratio_mean']:.2f} (median {rp['ratio_median']:.2f}) | | | {rp['gain_mean']:+.4f} | {rp['gain_n_pos']}/{rp['n']} | | |")
    for t, v in twins_cs.items():
        lines.append(f"| existing correct twins {t} | {v['n']} | | {v['ratio_mean']:.2f} | | | {v['gain_mean']:+.4f} | {v['gain_n_pos']}/{v['n']} | | |")
    md = "\n".join(lines)
    (OUT / "r6_summary.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
