"""Aggregate the checkpoint-based pilots into results/eegbits/pilots_summary.json (mean and 95% bootstrap CI over subjects; seeds
averaged within subject):
  controls : acq_controls_<tag>_sub-XX_seed*.json  -> eeg / noise / shuffled_pair at k=80 and k=1 per encoder (top-1, 2-way, bits)
  hardpool : acq_hardpool_<tag>_sub-XX_seed*.json  -> top-1 and 2-way inside hard vs random pools (N=2,10,50), eeg vs shuffled_pair
  fm_probe : fm_probe_<model>_sub-XX.json          -> frozen foundation-model rung (k=80, k=1, noise, shuffled_pair)
  atm20    : acq_atm_sub-08_draws20.json (+ _mean_loo_draws20) -> repetition curve with 20 draws (sd over draws)
Run via srun on a compute node; light."""
import glob, json, re
from pathlib import Path
import numpy as np
R = Path("/path/to/eegaudit/results/eegbits"); rng = np.random.default_rng(0)


def ci(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) < 2: return [float(v.mean()), float(v.mean())] if len(v) else [None, None]
    b = [rng.choice(v, len(v)).mean() for _ in range(2000)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def by_subject(files):
    d = {}
    for f in files:
        j = json.load(open(f)); d.setdefault(j["subject"], []).append(j["rows"])
    return d


def summ(vals):
    return dict(mean=float(np.mean(vals)), ci95=ci(vals), n_subjects=len(vals))


out = {"controls": {}, "hardpool": {}, "fm_probe": {}, "atm_draws20": {}}
# ---- controls
for tag in ["atm", "nice", "nice_ga"]:
    ps = by_subject(sorted(glob.glob(str(R / f"acq_controls_{tag}_sub-*_seed*.json"))))
    if not ps: continue
    res = {}
    for k in (80, 1):
        for cond in ("eeg", "noise", "shuffled_pair"):
            for m in ("top1", "two_way", "bits"):
                vals = [np.mean([np.mean([r[m] for r in rows if r["condition"] == cond and r["k"] == k]) for rows in runs]) for runs in ps.values()]
                res[f"k{k}_{cond}_{m}"] = summ(vals)
    out["controls"][tag] = res
    print(f"controls {tag}: n={len(ps)} k=80 top-1 eeg/noise/pair = {100*res['k80_eeg_top1']['mean']:.1f} / {100*res['k80_noise_top1']['mean']:.1f} / {100*res['k80_shuffled_pair_top1']['mean']:.1f} %; 2-way {100*res['k80_eeg_two_way']['mean']:.1f} / {100*res['k80_noise_two_way']['mean']:.1f} / {100*res['k80_shuffled_pair_two_way']['mean']:.1f}")
# ---- hard pools
for tag in ["atm", "nice", "nice_ga"]:
    ps = by_subject(sorted(glob.glob(str(R / f"acq_hardpool_{tag}_sub-*_seed*.json"))))
    if not ps: continue
    res = {}
    for k in (80, 1):
        for pool in ("hard", "random"):
            for N in (2, 10, 50):
                for cond in ("eeg", "shuffled_pair"):
                    for m in ("top1", "two_way", "bits"):
                        vals = [np.mean([np.mean([r[m] for r in rows if r["pool"] == pool and r["N"] == N and r["k"] == k and r["condition"] == cond]) for rows in runs]) for runs in ps.values()]
                        res[f"k{k}_{pool}{N}_{cond}_{m}"] = summ(vals)
    out["hardpool"][tag] = res
    print(f"hardpool {tag}: n={len(ps)} k=80 nearest-neighbour 2-way eeg {100*res['k80_hard2_eeg_two_way']['mean']:.1f}% (pair floor {100*res['k80_hard2_shuffled_pair_two_way']['mean']:.1f}%) | hard10 top-1 {100*res['k80_hard10_eeg_top1']['mean']:.1f}% vs random10 {100*res['k80_random10_eeg_top1']['mean']:.1f}% | hard50 bits {res['k80_hard50_eeg_bits']['mean']:.2f} vs random50 {res['k80_random50_eeg_bits']['mean']:.2f}")
# ---- fm probe
for model in ["cbramod", "labram", "eegpt"]:
    files = [f for f in sorted(glob.glob(str(R / f"fm_probe_{model}_sub-*.json"))) if "slice" not in f]
    if not files: continue
    ps = by_subject(files); res = {"feature_dim": json.load(open(files[0]))["feature_dim"]}
    for cond, k in (("eeg", 80), ("eeg", 1), ("noise", 80), ("shuffled_pair", 80)):
        for m in ("top1", "two_way", "bits"):
            vals = [np.mean([np.mean([r[m] for r in rows if r["condition"] == cond and r["k"] == k]) for rows in runs]) for runs in ps.values()]
            res[f"k{k}_{cond}_{m}"] = summ(vals)
    out["fm_probe"][model] = res
    print(f"fm_probe {model}: n={len(ps)} D={res['feature_dim']} k=80 2-way {100*res['k80_eeg_two_way']['mean']:.1f}% [{100*res['k80_eeg_two_way']['ci95'][0]:.1f},{100*res['k80_eeg_two_way']['ci95'][1]:.1f}] top-1 {100*res['k80_eeg_top1']['mean']:.1f}% bits {res['k80_eeg_bits']['mean']:.2f} | k=1 2-way {100*res['k1_eeg_two_way']['mean']:.1f}% | noise {100*res['k80_noise_two_way']['mean']:.1f}% pair {100*res['k80_shuffled_pair_two_way']['mean']:.1f}%")
# ---- ATM-S 20 draws
for suf in ["", "_mean_loo"]:
    f = R / f"acq_atm_sub-08{suf}_draws20.json"
    if f.exists():
        rows = json.load(open(f))["rows"]; res = {}
        for k in (1, 2, 4, 8, 16, 32, 80):
            b = [r["bits"] for r in rows if r["axis"] == "repetitions" and r["k"] == k]
            res[str(k)] = dict(bits_mean=float(np.mean(b)), bits_sd=float(np.std(b, ddof=1)) if len(b) > 1 else 0.0, n_draws=len(b), top1_mean=float(np.mean([r["top1"] for r in rows if r["axis"] == "repetitions" and r["k"] == k])))
        out["atm_draws20"]["mean_loo" if suf else "zero"] = res
        print(f"atm draws20{suf}:", {k: (round(v["bits_mean"], 2), round(v["bits_sd"], 2)) for k, v in res.items()})
(R / "pilots_summary.json").write_text(json.dumps(out, indent=1)); print("-> pilots_summary.json")
