"""Round 5, attack 2: does the retrain-vs-mask agreement survive seed variance?
retrain_summary.json compares ONE seed (141) in both arms (occpar32 retrained 0.46 vs masked 0.45; w06 0.51 vs 0.48). The masked
arm exists for five NICE-GA seeds (acq_nice_ga_sub-XX_seed{141,297,777,1024,2022}_mean_loo.json), the full 63-ch/1.0-s model too
(acq_nice_ga_sub-XX_seed*.json). Pure re-analysis of existing JSONs (no GPU):
  per subject: masked yield per seed, seed sd, seed range; full-model yield per seed and sd; retrained(141) - masked(141);
               retrained(141) - masked(mean of 5 seeds); where seed 141 sits in the five-seed distribution of the masked arm
  across subjects: mean difference and bootstrap CI under (a) the paper's one-seed pairing and (b) the five-seed masked mean;
               a seed-variance-aware CI: the seed sd of a one-seed difference is sqrt(2) x the masked seed sd (both arms are one seed),
               so the smallest difference the comparison could have detected is reported; top-1 granularity (1/200) in bits.
Output: results/method_pilots/r5_seed_variance.json"""
import json, glob
from pathlib import Path
import numpy as np
from common import save, wolpaw_bits
R = Path("/path/to/eegaudit/results/eegbits"); SEEDS = [141, 297, 777, 1024, 2022]; rng = np.random.default_rng(0)
summ = json.load(open(R / "retrain_summary.json"))


def ci(v, B=4000):
    v = np.asarray(v, float); b = [rng.choice(v, len(v)).mean() for _ in range(B)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def masked(subject, seed, variant, k):
    rows = json.load(open(R / f"acq_nice_ga_{subject}_seed{seed}_mean_loo.json"))["rows"]
    ch, w = ("occpar32", 1.0) if variant == "occpar32" else ("all63", 0.6)
    return float(np.mean([r["bits"] for r in rows if r["axis"] == "acquisition" and r["k"] == k and r["channels"] == ch and r["window"] == w]))


def full(subject, seed, k):
    rows = json.load(open(R / f"acq_nice_ga_{subject}_seed{seed}.json"))["rows"]
    if k == 80: return float([r for r in rows if r["axis"] == "native"][0]["bits"])
    return float(np.mean([r["bits"] for r in rows if r["axis"] == "repetitions" and r["k"] == 1]))


out = {"note": __doc__, "granularity": {}}
for p in (0.10, 0.13, 0.15, 0.20):
    out["granularity"][f"top1_{p:.2f}"] = {"bits": wolpaw_bits(p, 200), "bits_per_image": wolpaw_bits(p + 0.005, 200) - wolpaw_bits(p, 200)}
for variant in ("occpar32", "w06"):
    for k in (80, 1):
        per = []
        for p in summ[variant]["per_subject"]:
            s = p["subject"]; m = {sd: masked(s, sd, variant, k) for sd in SEEDS}; f = {sd: full(s, sd, k) for sd in SEEDS}
            mv = np.array(list(m.values())); fv = np.array(list(f.values()))
            ret = p[f"retrained_k{k}"]; assert abs(m[141] - p[f"masked_k{k}"]) < 1e-9 and abs(f[141] - p[f"full_k{k}"]) < 1e-9
            per.append(dict(subject=s, retrained_141=ret, masked_by_seed=m, masked_seed_mean=float(mv.mean()), masked_seed_sd=float(mv.std(ddof=1)),
                            masked_seed_range=[float(mv.min()), float(mv.max())], rank_of_141_in_masked=int((mv < m[141]).sum()) + 1,
                            full_by_seed=f, full_seed_mean=float(fv.mean()), full_seed_sd=float(fv.std(ddof=1)),
                            diff_141_pairing=ret - m[141], diff_vs_masked_5seed_mean=ret - float(mv.mean()),
                            masked_minus_full_by_seed=(mv - fv).tolist(), masked_minus_full_sd=float((mv - fv).std(ddof=1))))
        d1 = np.array([q["diff_141_pairing"] for q in per]); d5 = np.array([q["diff_vs_masked_5seed_mean"] for q in per])
        sd_seed = float(np.sqrt(np.mean([q["masked_seed_sd"] ** 2 for q in per])))           # pooled within-subject seed sd of the masked arm
        sd_full = float(np.sqrt(np.mean([q["full_seed_sd"] ** 2 for q in per])))
        n = len(per)
        res = dict(n_subjects=n, per_subject=per,
                   diff_one_seed_pairing=dict(mean=float(d1.mean()), ci95=ci(d1), sd_across_subjects=float(d1.std(ddof=1))),
                   diff_vs_five_seed_masked_mean=dict(mean=float(d5.mean()), ci95=ci(d5), sd_across_subjects=float(d5.std(ddof=1))),
                   masked_seed_sd_pooled=sd_seed, full_seed_sd_pooled=sd_full,
                   seed_sd_of_a_one_seed_difference=float(np.sqrt(2) * sd_seed),
                   # if the retrained arm's seed sd equals the masked arm's, a one-seed per-subject difference has seed sd sqrt(2)*sd;
                   # the mean over n subjects then has seed-only se sd*sqrt(2/n); 2*se is the smallest mean difference seeds alone could hide
                   seed_only_se_of_mean_diff=float(np.sqrt(2.0 / n) * sd_seed), min_detectable_mean_diff_2se=float(2 * np.sqrt(2.0 / n) * sd_seed),
                   share_of_subject_scatter_explained_by_seeds=float(min(1.0, 2 * sd_seed ** 2 / d1.var(ddof=1))),
                   masked_mean_141=float(np.mean([q["masked_by_seed"][141] for q in per])), masked_mean_5seed=float(np.mean([q["masked_seed_mean"] for q in per])),
                   retrained_mean=float(np.mean([q["retrained_141"] for q in per])),
                   seed141_rank_in_masked=[q["rank_of_141_in_masked"] for q in per])
        out[f"{variant}_k{k}"] = res
        print(f"{variant} k={k} (n={n}): retrained(141) {res['retrained_mean']:.3f} | masked(141) {res['masked_mean_141']:.3f} | masked(5-seed mean) {res['masked_mean_5seed']:.3f}")
        print(f"   diff one-seed pairing {res['diff_one_seed_pairing']['mean']:+.3f} CI {np.round(res['diff_one_seed_pairing']['ci95'],3)} | vs 5-seed mean {res['diff_vs_five_seed_masked_mean']['mean']:+.3f} CI {np.round(res['diff_vs_five_seed_masked_mean']['ci95'],3)}")
        print(f"   masked seed sd (pooled) {sd_seed:.3f}, full-model seed sd {sd_full:.3f}; seed sd of a one-seed difference {res['seed_sd_of_a_one_seed_difference']:.3f}; seed-only se of mean diff {res['seed_only_se_of_mean_diff']:.3f}; "
              f"subject scatter of the difference {res['diff_one_seed_pairing']['sd_across_subjects']:.3f}; share explained by seeds {100*res['share_of_subject_scatter_explained_by_seeds']:.0f}%")
        print("   seed-141 rank of masked arm within 5 seeds per subject:", res["seed141_rank_in_masked"])
        for q in per:
            print(f"      {q['subject']} ret {q['retrained_141']:.3f} masked {np.round(list(q['masked_by_seed'].values()),3)} sd {q['masked_seed_sd']:.3f} | full sd {q['full_seed_sd']:.3f} | d141 {q['diff_141_pairing']:+.3f} d5 {q['diff_vs_masked_5seed_mean']:+.3f}")
save("r5_seed_variance.json", out)
