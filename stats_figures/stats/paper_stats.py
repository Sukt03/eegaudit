"""Statistics layer for the paper: per claim, unit-of-analysis mean, bootstrap 95% CI, n, test vs floor or paired contrast,
effect size (Cohen's d_z), and Holm-adjusted p within each modality. Reads existing result files only.
Output: results/stats/paper_stats.json"""
import glob, json, math, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
R = Path("/path/to/eegaudit/results"); rng = np.random.default_rng(0)

def boot(x, B=10000):
    x = np.asarray(x, float); i = rng.integers(0, len(x), (B, len(x))); m = x[i].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

def paired(name, a, b, alt="two-sided", tost=None):
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b; n = len(d)
    t = stats.ttest_1samp(d, 0.0, alternative=alt)
    try: w = stats.wilcoxon(d, alternative=alt, zero_method="zsplit").pvalue
    except ValueError: w = float("nan")
    sd = d.std(ddof=1) if n > 1 else float("nan")
    out = dict(name=name, n=n, mean_a=float(a.mean()), mean_b=float(b.mean()), diff=float(d.mean()), diff_ci=boot(d),
               t=float(t.statistic), df=n - 1, p_t=float(t.pvalue), p_wilcoxon=float(w), dz=float(d.mean() / sd) if sd > 0 else float("nan"),
               n_positive=int((d > 0).sum()))
    if tost is not None:   # two one-sided t-tests for equivalence within +-tost
        p_lo = stats.ttest_1samp(d, -tost, alternative="greater").pvalue; p_hi = stats.ttest_1samp(d, tost, alternative="less").pvalue
        out.update(tost_margin=tost, p_tost=float(max(p_lo, p_hi)))
    return out

def holm(rows, key="p_t"):
    ps = [(i, r[key]) for i, r in enumerate(rows) if r.get(key) == r.get(key)]
    ps.sort(key=lambda x: x[1]); m = len(ps); run = 0.0
    for rank, (i, p) in enumerate(ps):
        run = max(run, min(1.0, (m - rank) * p)); rows[i]["p_holm"] = float(run)
    return rows

S = {}
# ---------------- IMAGE ----------------
img = []
per = defaultdict(lambda: defaultdict(list))
for f in glob.glob(str(R / "eegbits/acq_controls_nice_ga_sub-*_seed*.json")):
    d = json.load(open(f)); s = d["subject"]
    for r in d["rows"]:
        if r["axis"] == "control" and r["k"] == 80: per[s][r["condition"]].append(r["bits"]); per[s][r["condition"] + "_2w"].append(r["two_way"])
for f in glob.glob(str(R / "eegbits/acq_nice_ga_sub-*_seed*.json")):
    if "mean_loo" in f: continue
    d = json.load(open(f)); s = d["subject"]
    per[s]["k1"].append(np.mean([r["bits"] for r in d["rows"] if r["axis"] == "repetitions" and r["k"] == 1]))
    per[s]["k1_2w"].append(np.mean([r["two_way"] for r in d["rows"] if r["axis"] == "repetitions" and r["k"] == 1]))
subs = sorted(s for s in per if per[s]["eeg"] and per[s]["noise"])
m = lambda s, k: float(np.mean(per[s][k]))
img.append(paired("image_k80_eeg_vs_noise_bits", [m(s, "eeg") for s in subs], [m(s, "noise") for s in subs], "greater"))
img.append(paired("image_k80_eeg_vs_shuffled_pair_bits", [m(s, "eeg") for s in subs], [m(s, "shuffled_pair") for s in subs], "greater"))
subs1 = sorted(s for s in per if per[s]["k1"])
img.append(paired("image_k1_two_way_vs_chance", [m(s, "k1_2w") for s in subs1], [0.5] * len(subs1), "greater"))
img.append(paired("image_k1_bits_vs_zero", [m(s, "k1") for s in subs1], [0.0] * len(subs1), "greater"))
# split-half ceiling by representation space (k = 40 vs 40 repetitions), top-1 and bits
sh = defaultdict(dict)
for f in glob.glob(str(R / "eegbits/splithalf_sub-*.json")):
    d = json.load(open(f))
    for r in d["rows"]:
        if r["k"] == 40 and r["condition"] == "eeg": sh[d["subject"]][r["space"]] = r
ss = sorted(sh)
for metric in ["top1", "bits"]:
    img.append(paired(f"image_splithalf_trained_vs_raw_{metric}", [sh[s]["trained_encoder"][metric] for s in ss], [sh[s]["raw_voltage"][metric] for s in ss], "greater"))
    img.append(paired(f"image_splithalf_raw_vs_randominit_{metric}", [sh[s]["raw_voltage"][metric] for s in ss], [sh[s]["random_init_encoder"][metric] for s in ss]))
# paired target swap (same EEG, same architecture, retrained per target)
P = json.load(open(R / "method_pilots/paired_target_swap.json")); cells = P["cells"]; keys = [k for k in P["pairs_complete"]]
g = lambda t, k="bits": [cells[c][t][k] for c in keys]
img.append(paired("image_target_clip_vs_pixels_bits", g("clip"), g("pixpca"), "greater"))
img.append(paired("image_target_dinov2_vs_pixels_bits", g("dinov2"), g("pixpca"), "greater"))
img.append(paired("image_target_dinov2_vs_clip_bits", g("dinov2"), g("clip")))
img.append(paired("image_target_clip_vs_pixels_k1_bits", g("clip", "k1_bits"), g("pixpca", "k1_bits"), "greater"))
# recording axis: retrained vs masked (equivalence, margin 0.05 bits pre-registered by the ideator, S2) and retrained vs full
RT = json.load(open(R / "eegbits/retrain_summary.json"))
for v in ["occpar32", "w06"]:
    ps = [p for p in RT[v]["per_subject"] if p["masked_k80"] == p["masked_k80"]]
    img.append(paired(f"image_recording_{v}_retrained_vs_masked", [p["retrained_k80"] for p in ps], [p["masked_k80"] for p in ps], tost=0.05))
    img.append(paired(f"image_recording_{v}_retrained_vs_full", [p["retrained_k80"] for p in ps], [p["full_k80"] for p in ps], tost=0.05))
for v in ["occpar16", "w04", "pca100"]:
    if v in RT:
        ps = RT[v]["per_subject"]
        img.append(paired(f"image_recording_{v}_retrained_vs_full", [p["retrained_k80"] for p in ps], [p["full_k80"] for p in ps], "less"))
for v in ["shuffle", "randtarget"]:
    if v in RT:
        ps = RT[v]["per_subject"]; img.append(paired(f"image_control_{v}_retrained_vs_full", [p["retrained_k80"] for p in ps], [p["full_k80"] for p in ps], "less"))
# yield protocol (results/stats/yield_stats.json from experiments/stats/yield_stats.py): bottleneck and usable-fraction tests
YS = json.load(open(R / "stats/yield_stats.json"))
_st = rng.bit_generator.state   # yield tests must not shift the bootstrap draws of the rows computed after them
bi = YS["image"]["input_nicega_InfoNCE"]; br = YS["image"]["representation_nicega_Wolpaw"]["L90_per_participant"]
cs = sorted(set(bi["L90_per_participant"]) & set(br))
img.append(paired("yield_image_L90_input_vs_representation_log10", [math.log10(bi["L90_per_participant"][s]) for s in cs], [math.log10(br[s]) for s in cs]))
pp = bi["eeg_per_participant"]
img.append(paired("yield_image_input_L100_vs_full_InfoNCE", [pp[s]["100"] for s in sorted(pp)], [pp[s]["full"] for s in sorted(pp)], "less"))
img.append(paired("yield_image_input_L1000_vs_full_InfoNCE", [pp[s]["1000"] for s in sorted(pp)], [pp[s]["full"] for s in sorted(pp)], "less"))
fk = YS["image"]["usable_fraction"]["nicega_k40_I_NCE"]["per_participant"]; fw = YS["image"]["usable_fraction"]["nicega_k40_I_F"]["per_participant"]
img.append(paired("yield_image_fraction_k40_InfoNCE_vs_Wolpaw", [fk[s] for s in sorted(fk)], [fw[s] for s in sorted(fk)]))
rng.bit_generator.state = _st
S["image"] = holm(img)
# ---------------- TEXT ----------------
txt = []
acc = defaultdict(lambda: defaultdict(list))
for f in glob.glob(str(R / "track_text/within_SR/ridge_*__seed*.json")):
    d = json.load(open(f)); key = (d["decoder"], d["subject"])
    for r in d["rows"]:
        if r["k"] == 1 and r.get("subjects_averaged", 1) == 1: acc[key][r["condition"]].append(r["two_way"])
keys2 = sorted(k for k in acc if acc[k]["eeg"] and acc[k]["noise"])
txt.append(paired("text_ridge_settings_eeg_vs_noise_two_way", [np.mean(acc[k]["eeg"]) for k in keys2], [np.mean(acc[k]["noise"]) for k in keys2], "greater"))
txt.append(paired("text_ridge_settings_eeg_vs_shuffled_pair_two_way", [np.mean(acc[k]["eeg"]) for k in keys2], [np.mean(acc[k]["shuffled_pair"]) for k in keys2], "greater"))
bysub = defaultdict(lambda: defaultdict(list))
for (dec, s), v in acc.items():
    for c in ("eeg", "noise"): bysub[s][c].append(np.mean(v[c]))
sb = sorted(bysub)
txt.append(paired("text_ridge_participants_eeg_vs_noise_two_way", [np.mean(bysub[s]["eeg"]) for s in sb], [np.mean(bysub[s]["noise"]) for s in sb], "greater"))
be = json.load(open(R / "track_text/brain_translator/full_eeg_results.json"))["identification"]; bn = json.load(open(R / "track_text/brain_translator/full_noise_results.json"))["identification"]
for mode in ["greedy", "teacher_forced"]:
    n = be["n_queries"]; x1 = round(be[mode]["top1"] * n); x2 = round(bn[mode]["top1"] * n)
    tab = np.array([[x1, n - x1], [x2, n - x2]]); p = stats.fisher_exact(tab, alternative="greater").pvalue
    txt.append(dict(name=f"text_braintranslator_{mode}_top1_eeg_vs_noise", n=int(n), mean_a=be[mode]["top1"], mean_b=bn[mode]["top1"], diff=be[mode]["top1"] - bn[mode]["top1"], p_t=float(p), test="Fisher exact, one-sided", chance=be["chance_top1"]))
_st = rng.bit_generator.state
for task in ["SR", "NR2"]:
    TB = YS["text"].get(f"protocol_zuco_{task}_ridge_mean_wordsall_bits_NCE")
    if TB:
        q = TB["per_participant"]; ps2 = sorted(q["eeg"])
        txt.append(paired(f"yield_text_{task}_full_eeg_vs_noise_InfoNCE", [q["eeg"][s]["full"] for s in ps2], [q["noise"][s]["full"] for s in ps2]))
        txt.append(paired(f"yield_text_{task}_full_eeg_vs_shuffled_InfoNCE", [q["eeg"][s]["full"] for s in ps2], [q["shuffled_pair"][s]["full"] for s in ps2]))
rng.bit_generator.state = _st
S["text"] = holm(txt)
# ---------------- PREDICTION ERROR ----------------
G = json.load(open(R / "group_stats/results.json"))["decoders"]
pe = []
z = lambda d: np.arctanh(np.clip(np.asarray(G[d]["r"], float), -0.999, 0.999))
nz = lambda d: np.asarray(G[d]["null_z"], float) if len(G[d].get("null_z", [])) == len(G[d]["r"]) else None
for d in ["eegnet", "revetok", "reve_lopo", "fmtok", "cbramod_ft", "ens", "raw", "fm", "eegpt", "labram"]:
    if d not in G: continue
    r = paired(f"pe_{d}_fisherz_vs_zero", z(d), np.zeros(len(G[d]["r"])), "greater")
    r.update(n_above_null95=int(G[d].get("n_individually_above_95pct_null", -1)), median_r=float(np.median(G[d]["r"])))
    pe.append(r)
for a, b in [("revetok", "eegnet"), ("eegnet", "raw"), ("fmtok", "fm"), ("eegnet", "fm"), ("fmtok", "labram"), ("ens", "eegnet"), ("cbramod_ft", "fm")]:
    if a in G and b in G: pe.append(paired(f"pe_{a}_vs_{b}_fisherz", z(a), z(b)))
_st = rng.bit_generator.state
pr = YS["pe"]["input_raw_rawr"]["eeg_per_participant"]; pn = YS["pe"]["input_ens_rawr"]["eeg_per_participant"]; ps_ = sorted(set(pr) & set(pn))
pe.append(paired("yield_pe_ens_vs_raw_L64_rawr", [pn[s]["64"] for s in ps_], [pr[s]["64"] for s in ps_]))
pe.append(paired("yield_pe_ens_L8_vs_L64_rawr", [pn[s]["8"] for s in ps_], [pn[s]["64"] for s in ps_], "less"))
fe = YS["pe"]["usable_fraction"]["eegnet"]["per_participant"]
pe.append(paired("yield_pe_eegnet_fraction_vs_zero", [fe[k] for k in sorted(fe)], [0.0] * len(fe), "greater"))
PB = [v for k, v in YS["pe"].items() if k.startswith("protocol_ds003458_raw_erp_ridge_bits_g")]
if PB:
    q = PB[0]["per_participant"]; ps2 = sorted(q["eeg"])
    pe.append(paired("yield_pe_gauss_raw_L64_vs_full", [q["eeg"][s]["64"] for s in ps2], [q["eeg"][s]["full"] for s in ps2], "less"))
    pe.append(paired("yield_pe_gauss_raw_pca_vs_random_L64", [q["eeg"][s]["64"] for s in ps2], [q["random"][s]["64"] for s in ps2], "greater"))
    pe.append(paired("yield_pe_gauss_raw_full_vs_perm95", [q["eeg"][s]["full"] for s in ps2], [q["perm_p95"][s]["full"] for s in ps2], "greater"))
rng.bit_generator.state = _st
S["prediction_error"] = holm(pe)
for rows in S.values():
    for r in rows:
        for k in ("p_holm", "p_tost", "p_t"):
            if k in r and r[k] == r[k] and r[k] > 0:
                e = math.floor(math.log10(r[k])); r[k + "_mant"] = round(r[k] / 10 ** e, 1); r[k + "_exp"] = e
out = R / "stats/paper_stats.json"; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(S, indent=1))
for mod, rows in S.items():
    print("==", mod)
    for r in rows:
        print(f"  {r['name']:55s} n={r['n']:4d} a={r['mean_a']:.3f} b={r['mean_b']:.3f} diff={r['diff']:+.3f} ci={[round(x,3) for x in r.get('diff_ci',[])]} p={r['p_t']:.2g} holm={r.get('p_holm',float('nan')):.2g} dz={r.get('dz',float('nan')):.2f} pos={r.get('n_positive','')}" + (f" TOST(±{r['tost_margin']}) p={r['p_tost']:.2g}" if 'p_tost' in r else "") + (f" above_null={r['n_above_null95']}" if 'n_above_null95' in r else ""))
