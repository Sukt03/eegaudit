"""Inter-participant variation for all three targets, and figure paper/figs/fig_participants.pdf.
Image: NICE-GA bits per participant x seed (k=80 and k=1), ICC(1) = share of variance between participants (one-way random effects).
Sentence: per ZuCo participant, EEG minus noise-twin 2-way identification over feature settings (SR, NR, NR2).
Prediction error: per participant LOPO coupling r for EEGNet and REVE tokens vs the decoder's session-permutation null (95th pct).
Output: results/stats/participants.json"""
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path("/path/to/eegaudit"); R = ROOT / "results"; rng = np.random.default_rng(0)
sys.path.insert(0, str(ROOT / "experiments/eegbits")); import figstyle as fs; fs.setup(); import matplotlib.pyplot as plt

def icc1(M):   # M: participants x replicates
    M = np.asarray(M, float); n, k = M.shape; gm = M.mean()
    msb = k * ((M.mean(1) - gm) ** 2).sum() / (n - 1); msw = ((M - M.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return float((msb - msw) / (msb + (k - 1) * msw)), float(msb / msw)
def boot(x, B=5000):
    x = np.asarray(x, float); m = x[rng.integers(0, len(x), (B, len(x)))].mean(1); return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
out = {}
# ---- image
per = defaultdict(dict); per1 = defaultdict(dict); noise = defaultdict(list)
for f in glob.glob(str(R / "eegbits/acq_controls_nice_ga_sub-*_seed*.json")):
    d = json.load(open(f))
    for r in d["rows"]:
        if r["axis"] == "control" and r["k"] == 80 and r["condition"] == "eeg": per[d["subject"]][d["seed"]] = r["bits"]
        if r["axis"] == "control" and r["k"] == 80 and r["condition"] == "noise": noise[d["subject"]].append(r["bits"])
for f in glob.glob(str(R / "eegbits/acq_nice_ga_sub-*_seed*.json")):
    if "mean_loo" in f: continue
    d = json.load(open(f)); per1[d["subject"]][d["seed"]] = float(np.mean([r["bits"] for r in d["rows"] if r["axis"] == "repetitions" and r["k"] == 1]))
subs = sorted(per); seeds = sorted(set.intersection(*[set(per[s]) for s in subs]))
M = np.array([[per[s][x] for x in seeds] for s in subs]); M1 = np.array([[per1[s][x] for x in seeds] for s in subs])
ic, F = icc1(M); ic1, F1 = icc1(M1)
mean_s = M.mean(1)
out["image"] = dict(n_participants=len(subs), n_seeds=len(seeds), subjects=subs, k80_mean=mean_s.tolist(), k80_min=float(mean_s.min()), k80_max=float(mean_s.max()),
                    k80_ratio_max_min=float(mean_s.max() / mean_s.min()), k80_icc=ic, k80_F=F, k1_mean=M1.mean(1).tolist(), k1_min=float(M1.mean(1).min()), k1_max=float(M1.mean(1).max()), k1_icc=ic1,
                    k80_seed_sd_mean=float(M.std(1, ddof=1).mean()), k80_between_sd=float(mean_s.std(ddof=1)), noise_mean=float(np.mean([np.mean(noise[s]) for s in subs])))
# ---- sentence
txt = {}
for task in ["SR", "NR", "NR2"]:
    acc = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(str(R / f"track_text/within_{task}/ridge_*__seed*.json")):
        d = json.load(open(f)); key = (d["decoder"], d["subject"])
        for r in d["rows"]:
            if r["k"] == 1 and r.get("subjects_averaged", 1) == 1 and r["condition"] in ("eeg", "noise"): acc[key][r["condition"]].append(r["two_way"])
    bys = defaultdict(list)
    for (dec, s), v in acc.items():
        if v["eeg"] and v["noise"]: bys[s].append(np.mean(v["eeg"]) - np.mean(v["noise"]))
    txt[task] = {s: dict(diff=float(np.mean(x)), ci=boot(x), n_settings=len(x)) for s, x in sorted(bys.items())}
allp = [v for t in txt.values() for v in t.values()]
out["sentence"] = dict(tasks=txt, n_participant_task=len(allp), n_ci_above_zero=sum(v["ci"][0] > 0 for v in allp), n_ci_below_zero=sum(v["ci"][1] < 0 for v in allp),
                       diff_min=float(min(v["diff"] for v in allp)), diff_max=float(max(v["diff"] for v in allp)))
# ---- prediction error
G = json.load(open(R / "group_stats/results.json"))["decoders"]; PL = json.load(open(R / "track_decision/price_list.json"))["datasets"]["ds003458"]["controls"]
GA = json.load(open(R / "loop/learner/measure_bits_gauss.json"))["datasets"]["ds003458"]["decoders"]
pe = {}
for d in ["eegnet", "revetok", "raw"]:
    r = np.asarray(G[d]["r"], float); p95 = PL[d]["session_permutation_null"]["p95_r"]
    pe[d] = dict(r=r.tolist(), r_min=float(r.min()), r_max=float(r.max()), r_median=float(np.median(r)), null_p95=float(p95), n_above=int((r > p95).sum()), n=len(r),
                 tau_between_z=float(np.sqrt(GA[d]["tau2"])), bits_g_pp_median=GA[d]["bits_g_median_pp"])
rr = np.corrcoef(G["eegnet"]["r"], G["revetok"]["r"])[0, 1]
out["prediction_error"] = dict(decoders=pe, corr_eegnet_revetok_across_participants=float(rr))
(R / "stats/participants.json").write_text(json.dumps(out, indent=1))
# ---- figure
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.0), gridspec_kw=dict(width_ratios=[1, 1.15, 1.25], wspace=0.42))
ax = axs[0]; o = np.argsort(mean_s)
for i, j in enumerate(o):
    ax.scatter([i] * len(seeds), M[j], s=9, color=fs.SERIES["blue"], alpha=0.35, lw=0, zorder=2)
    ax.scatter(i, mean_s[j], s=30, marker="D", color=fs.SERIES["blue"], edgecolor=fs.SURFACE, lw=0.8, zorder=3)
ax.axhline(out["image"]["noise_mean"], color=fs.GREY2, lw=0.8, ls=":"); ax.text(0, out["image"]["noise_mean"] + 0.02, "noise floor", fontsize=6.8, color=fs.INK2)
ax.set_xticks(range(len(subs))); ax.set_xticklabels([subs[j].replace("sub-", "") for j in o], fontsize=6.5); ax.set_xlabel("participant (sorted)"); ax.set_ylabel("bits per 80-trial query")
ax.text(0.02, 0.97, f"ICC = {ic:.2f}", transform=ax.transAxes, fontsize=7, va="top", color=fs.INK2)
ax = axs[1]; x = 0; col = {"SR": fs.SERIES["blue"], "NR": fs.SERIES["orange"], "NR2": fs.SERIES["violet"]}; ticks = []
for t in ["SR", "NR", "NR2"]:
    for s, v in sorted(txt[t].items(), key=lambda kv: kv[1]["diff"]):
        ax.errorbar(x, 100 * v["diff"], yerr=[[100 * (v["diff"] - v["ci"][0])], [100 * (v["ci"][1] - v["diff"])]], fmt="o", ms=3.5, color=col[t], elinewidth=1, capsize=0); x += 1
    ax.text(x - len(txt[t]) / 2 - 0.5, 6.4, t, fontsize=7, color=col[t], ha="center"); x += 1.5
ax.axhline(0, color=fs.GREY2, lw=0.8, ls=":"); ax.set_xticks([]); ax.set_xlabel("ZuCo participant, by task"); ax.set_ylabel("EEG $-$ noise twin\n(2-way, points)"); ax.set_ylim(-7, 7.5)
ax = axs[2]; o = np.argsort(pe["eegnet"]["r"])
for d, c, mk, lab in [("eegnet", fs.SERIES["blue"], "o", "EEGNet"), ("revetok", fs.SERIES["violet"], "D", "REVE tokens"), ("raw", fs.SERIES["orange"], "s", "raw ERP")]:
    ax.scatter(range(23), np.asarray(pe[d]["r"])[o], s=14 if d != "raw" else 10, marker=mk, color=c, lw=0, alpha=0.9 if d != "raw" else 0.6, label=lab, zorder=3)
ax.axhspan(0, max(pe[d]["null_p95"] for d in pe), color=fs.GRID, alpha=0.7, lw=0); ax.text(22, 0.02, "null (95th pct)", fontsize=6.8, color=fs.INK2, ha="right")
ax.set_xticks([]); ax.set_xlabel("participant (sorted by EEGNet)"); ax.set_ylabel("coupling $r$ with the\nprediction error"); ax.legend(fontsize=6.5, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.2), handletextpad=0.2)
for a, l in zip(axs, "abc"): a.text(-0.02, 1.04, l, transform=a.transAxes, fontsize=10, fontweight="bold", ha="right")
fs.save(fig, "fig_participants", ROOT / "paper/figs")
print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))} for k, v in out.items()}, indent=1))
print("eeg-reve corr", rr, {d: (pe[d]["n_above"], pe[d]["r_min"], pe[d]["r_max"]) for d in pe})
