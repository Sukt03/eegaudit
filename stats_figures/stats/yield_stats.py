"""Yield curves, L90 and usable fraction per modality, per participant, with participant-bootstrap 95% CIs (PROTOCOL.md steps b-f).
Reads existing result files only; picks up new protocol-schema bottleneck files automatically (glob, see load_protocol_files).
Output: results/stats/yield_stats.json (every number in fig_yield / fig_usable and their captions comes from here).
Run: srun --overlap -N1 -n1 -c2 --mem=8G -t 15 <cogmodels python> -u experiments/stats/yield_stats.py"""
import glob, json, math, os
from collections import defaultdict
from pathlib import Path
import numpy as np
R = Path("/path/to/eegaudit/results"); rng = np.random.default_rng(0); B = 10000
bits_rho = lambda r: float(-0.5 * np.log2(1 - min(max(r, 0.0), 0.999999) ** 2))

def boot_mean(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return [float("nan")] * 2
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1); return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
def boot_median(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return [float("nan")] * 2
    m = np.median(x[rng.integers(0, len(x), (B, len(x)))], 1); return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
def summ(x):
    x = np.asarray(x, float); return dict(mean=float(np.nanmean(x)), ci95=boot_mean(x), median=float(np.nanmedian(x)), n=int(np.isfinite(x).sum()))

def l90(Ls, y, y_full, frac=0.9):
    """Smallest L reaching frac * full-width yield; linear interpolation on the monotone (running-max) envelope (PROTOCOL f)."""
    Ls = np.asarray(Ls, float); env = np.maximum.accumulate(np.asarray(y, float)); t = frac * y_full
    if not np.isfinite(y_full) or y_full <= 0: return float("nan")
    i = int(np.argmax(env >= t)) if (env >= t).any() else None
    if i is None: return float("nan")
    if i == 0: return float(Ls[0])
    x0, x1, e0, e1 = Ls[i - 1], Ls[i], env[i - 1], env[i]
    return float(x0 + (t - e0) / (e1 - e0) * (x1 - x0)) if e1 > e0 else float(x1)

def curve_block(per, Ls, conds, full_dim, unit, source, note=""):
    """per[participant][cond][L] -> value (L may be 'full'). Returns group curve + per-participant L90."""
    parts = sorted(p for p in per if "eeg" in per[p])
    out = dict(unit=unit, source=source, note=note, n=len(parts), participants=parts, full_dim=full_dim, Ls=[int(l) for l in Ls], curves={})
    for c in conds:
        rows = {}
        for L in list(Ls) + ["full"]:
            v = [per[p][c].get(L) for p in parts if c in per[p] and per[p][c].get(L) is not None]
            if v: rows[str(L)] = dict(mean=float(np.mean(v)), ci95=boot_mean(v), n=len(v))
        if rows: out["curves"][c] = rows
    L90 = {}
    for p in parts:
        e = per[p]["eeg"]; full = e.get("full", e.get(max(Ls)))
        L90[p] = l90([L for L in Ls if L in e] + [full_dim], [e[L] for L in Ls if L in e] + [full], full)
    v = np.array(list(L90.values()), float)
    out["L90_per_participant"] = L90; out["L90"] = dict(median=float(np.nanmedian(v)), median_ci95=boot_median(v), gmean=float(np.exp(np.nanmean(np.log(v)))), min=float(np.nanmin(v)), max=float(np.nanmax(v)), n=int(np.isfinite(v).sum()))
    fv = [per[p]["eeg"].get("full") for p in parts]; out["full_yield"] = summ(fv)
    out["eeg_per_participant"] = {p: {str(L): per[p]["eeg"][L] for L in per[p]["eeg"]} for p in parts}
    out["per_participant"] = {c: {p: {str(L): per[p][c][L] for L in per[p][c]} for p in parts if c in per[p]} for c in conds}
    return out

OUT = dict(image={}, pe={}, text={}, protocol_files=[])
# ======================================================================= IMAGE (THINGS-EEG2, 200-way, k = 80 trials per query)
# b1 input-side: frozen NICE-GA (ours, seed 141), PCA of the participant's own EEG (63 ch x 250 samples), 10 participants
IMG_FULL_DIM = 63 * 250
for metric, unit in [("soft_bits", "InfoNCE"), ("bits", "Wolpaw")]:
    per = defaultdict(lambda: defaultdict(dict)); Ls = set()
    for f in sorted(glob.glob(str(R / "method_pilots/r1_input_bottleneck_sub-*_basis-sub-*.json"))):
        if "smoke" in f: continue
        d = json.load(open(f))
        if d["subject"] != d["basis_subject"]: continue
        for r in d["rows"]:
            if r["k"] != 80: continue
            L = r["L"] if r["L"] == "full" else int(r["L"]); per[d["subject"]][r["condition"]][L] = r[metric]
            if L != "full": Ls.add(L)
    OUT["image"][f"input_nicega_{unit}"] = curve_block(per, sorted(Ls), ["eeg", "noise", "shuffled_pair"], IMG_FULL_DIM, f"{unit} bits per 80-trial query (200-way)",
                                                         "results/method_pilots/r1_input_bottleneck_sub-XX_basis-sub-XX.json", "frozen NICE-GA (ours), seed 141; full = 15,750 input dims")
# ATM-S released checkpoint, one participant (sub-08), same input-side bottleneck
A = json.load(open(R / "loop/image/s3_atm_input_bottleneck_sub-08.json"))
per = defaultdict(lambda: defaultdict(dict)); Ls = set()
rows = A.get("rows") or []
for r in rows:
    if r.get("k") != 80: continue
    L = r["L"] if r["L"] == "full" else int(r["L"]); per["sub-08"][r["condition"]].setdefault(L, []).append(r.get("soft_bits"))
    if L != "full": Ls.add(L)
for c in per["sub-08"]:
    for L in per["sub-08"][c]: per["sub-08"][c][L] = float(np.mean(per["sub-08"][c][L]))
if Ls: OUT["image"]["input_atms_InfoNCE"] = curve_block(per, sorted(Ls), ["eeg", "noise", "shuffled_pair"], IMG_FULL_DIM, "InfoNCE bits per 80-trial query (200-way)",
                                                       "results/loop/image/s3_atm_input_bottleneck_sub-08.json", "ATM-S released checkpoint, ONE participant (pilot)")
# b2 representation-side (NICE-GA embedding, Wolpaw only): L90 per participant from the summary file
S = json.load(open(R / "eegbits/summary.json"))["embedding"]
OUT["image"]["representation_nicega_Wolpaw"] = dict(L90_per_participant=S["L90_per_subject"], L90_median=S["L90_median"], source="results/eegbits/summary.json (embedding)", full_dim=S["full_L"])
# usable fraction = (yield - floor) / (ceiling - floor), per participant; ceiling = split-half of the trained-encoder code (P1), same estimator
P = json.load(open(R / "loop/image_p1/p1_rank_bound.json"))["per_participant"]
fr = {}
for est in ["I_NCE", "I_F"]:
    for dec, key in [("nicega", "nice_ga"), ("atms", "atm_native")]:
        vals, ys, cs, fs_ = {}, [], [], []
        for s in sorted(P[key]):
            e = P[key][s]["k80"] if key == "nice_ga" else P[key][s]; ce = P["splithalf_trained"][s]
            y = e[est]; fl = e["shuffled_pair"][est] if "shuffled_pair" in e else P["atm_native"][s]["shuffled_pair"][est]
            c = ce[est]; vals[s] = (y - fl) / (c - fl); ys.append(y); cs.append(c); fs_.append(fl)
        fr[f"{dec}_{est}"] = dict(per_participant=vals, fraction=summ(list(vals.values())), yield_=summ(ys), ceiling=summ(cs), floor=summ(fs_))
for k in fr: fr[k]["note"] = "k = 80 yield against the k = 40 split-half ceiling (mismatched k; reference only, not for print)"
# matched k (S3): yield at k = 40 (mean of the two 40-repetition halves) vs the k = 40 split-half ceiling, shuffled-pair floors
S3 = json.load(open(R / "loop/s3_fraction/s3_fraction_k40.json"))
for est in ["I_NCE", "I_F"]:
    vals = {r["subject"]: r["fraction"][est] for r in S3["rows"]}
    fr[f"nicega_k40_{est}"] = dict(per_participant=vals, fraction=summ(list(vals.values())), yield_=summ([r["yield_k40"][est] for r in S3["rows"]]),
                                   ceiling=summ([r["ceiling_k40"][est] for r in S3["rows"]]), floor=summ([r["floor"][est] for r in S3["rows"]]),
                                   note="matched k = 40 (results/loop/s3_fraction/s3_fraction_k40.json); NICE-GA (ours) seed 141")
OUT["image"]["usable_fraction"] = fr
# ATM-S minus NICE-GA fraction, paired over participants (same ceiling per participant)
d = [fr["atms_I_NCE"]["per_participant"][s] - fr["nicega_I_NCE"]["per_participant"][s] for s in fr["nicega_I_NCE"]["per_participant"]]
OUT["image"]["usable_fraction"]["atms_minus_nicega_I_NCE"] = dict(diff=summ(d), n_positive=int(np.sum(np.array(d) > 0)))
# input-side vs representation-side L90, paired per participant
ii = OUT["image"]["input_nicega_InfoNCE"]["L90_per_participant"]; rr = S["L90_per_subject"]
common = sorted(set(ii) & set(rr)); ratio = [ii[s] / rr[s] for s in common]
OUT["image"]["L90_input_over_representation"] = dict(per_participant=dict(zip(common, ratio)), median=float(np.median(ratio)), median_ci95=boot_median(ratio), min=float(min(ratio)), max=float(max(ratio)), n=len(common),
                                                   note="input side InfoNCE vs representation side Wolpaw (each L90 relative to its own full-width yield)")
# ======================================================================= PREDICTION ERROR (ds003458, 23 participants, per feedback trial)
# b1 interim: per-participant PCA-L of the raw-ERP / ensemble feature families + ridge (belief_bottleneck.py, FINAL run 18 Nov), RAW-target r
# -> Gaussian-channel yield -1/2 log2(1-r^2) ("information of a Gaussian channel with the measured correlation", not the normal-scores bound)
for fam in ["raw", "ens"]:
    per = defaultdict(lambda: defaultdict(dict)); Ls = set()
    for f in sorted(glob.glob(str(R / "rung2_bottleneck/sub-*.json"))):
        d = json.load(open(f)); s = os.path.basename(f)[:-5]; j = d["targets"].index("pe")
        for L, v in d["pca"][fam].items():
            L = int(L); Ls.add(L)
            per[s]["eeg"][L] = bits_rho(v["r"][j]); per[s]["null_mean"][L] = bits_rho(v["null"][j])
            per[s]["null_p95"][L] = bits_rho(v["null"][j] + 1.645 * v["null_sd"][j])
        per[s]["eeg"]["full"] = per[s]["eeg"][max(Ls)]
    OUT["pe"][f"input_{fam}_rawr"] = curve_block(per, sorted(Ls), ["eeg", "null_mean", "null_p95"], 64, "Gaussian-channel bits per trial (raw-target r)",
                                                  "results/rung2_bottleneck/sub-XXX.json (pca.<family>)", f"interim: family '{fam}', PCA-L then ridge, per participant; full = PCA-64 (the decoder's input)")
# read-out precision bottleneck (P5): EEGNet / REVE LOPO read-out quantised to b bits
P5 = json.load(open(R / "loop/p5/p5_rate_curve.json"))["pe"]
for dec in P5:
    per = defaultdict(lambda: defaultdict(dict)); bs = set()
    for p in P5[dec]["per_participant"]:
        for b, v in p["grid"].items(): per[p["subject"]]["eeg"][int(b)] = v; bs.add(int(b))
        per[p["subject"]]["eeg"]["full"] = p["bits_g_full"]
    OUT["pe"][f"precision_{dec}"] = curve_block(per, sorted(bs), ["eeg"], 32, "Gaussianised bits per trial (uncorrected)", "results/loop/p5/p5_rate_curve.json", "x = read-out precision b (bits), full = float32")
# usable fraction (PROTOCOL e-f): per-participant null-corrected bits_g vs the read-out's own split-half ceiling
MB = json.load(open(R / "loop/learner/measure_bits_gauss.json"))["datasets"]["ds003458"]["decoders"]
S16 = json.load(open(R / "loop/learner/s16_ceiling.json")); P5A = json.load(open(R / "loop/r5_eegnet_ceiling/r5_eegnet_ceiling_partA.json"))
rho_e = np.array([r["rho_max_decoder"] for r in json.load(open(R / "loop/r5_eegnet_ceiling/r5_eegnet_splithalf_30ep_all.json"))["rows"]], float)   # 30 epochs, n = 23 (matched to the EEGNet yield; METHOD CHANGE 3)
rho_e10 = np.array([r["rho_max_decoder"] for r in json.load(open(R / "loop/r5_eegnet_ceiling/r5_eegnet_splithalf_all.json"))["rows"]], float)   # 10 epochs, n = 23 (sensitivity)
def pe_fraction(dec, rho_ceiling_samples):
    dd = MB[dec]; bi = np.array([bits_rho(math.tanh(x)) for x in dd["per_participant_d"]]); fl = dd["floor_bits_g"]
    rs = np.atleast_1d(np.asarray(rho_ceiling_samples, float)); cmean = bits_rho(rs.mean())   # ceiling = bits at the mean rho_max (as S16 / PROTOCOL e)
    cs = np.array([cmean])
    pt = float(np.mean((bi - fl) / (cmean - fl)))
    bb = []
    for _ in range(B):   # two-sample bootstrap: participants (yield) and, where available, participants (ceiling rho_max)
        y = bi[rng.integers(0, len(bi), len(bi))].mean(); c = bits_rho(rs[rng.integers(0, len(rs), len(rs))].mean()); bb.append((y - fl) / (c - fl))
    return dict(fraction=dict(mean=pt, ci95=[float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))], n=len(bi)), yield_mean=float(bi.mean()), yield_ci95=boot_mean(bi),
                bits_g_mean_pp_file=dd["bits_g_mean_pp"], floor=fl, ceiling=float(cmean), rho_max_mean=float(rs.mean()), n_ceiling=int(len(rs)),
                per_participant={str(i + 1): float((b - fl) / (cmean - fl)) for i, b in enumerate(bi)})
OUT["pe"]["usable_fraction"] = dict(eegnet=pe_fraction("eegnet", rho_e), ens=pe_fraction("ens", [P5A["rho_max_decoder_ens"]["mean"]]))
OUT["pe"]["usable_fraction"]["eegnet"]["ceiling_note"] = "EEGNet own split-half, 23 participants, 30 epochs (r5_eegnet_splithalf_30ep_all.json), matched to the 30-epoch EEGNet yield"
OUT["pe"]["usable_fraction"]["eegnet_ceiling10"] = pe_fraction("eegnet", rho_e10); OUT["pe"]["usable_fraction"]["eegnet_ceiling10"]["ceiling_note"] = "sensitivity: EEGNet own split-half, 23 participants, 10 epochs (r5_eegnet_splithalf_all.json)"
OUT["pe"]["usable_fraction"]["ens"]["ceiling_note"] = "ensemble read-out split-half rho_max mean over 23 (r5 part A); fixed denominator"
# ======================================================================= TEXT (ZuCo SR, 40-way sentence identification, 1 reading per query)
per = defaultdict(lambda: defaultdict(list)); per2 = defaultdict(lambda: defaultdict(list))
for f in glob.glob(str(R / "track_text/within_SR/ridge_mean_wordsall__*seed*.json")):
    d = json.load(open(f))
    for r in d["rows"]:
        if r["k"] == 1 and r.get("subjects_averaged", 1) == 1: per[d["subject"]][r["condition"]].append(r["bits"]); per2[d["subject"]][r["condition"]].append(r["two_way"])
txt = dict(source="results/track_text/within_SR/ridge_mean_wordsall__*.json", decoder="ridge on mean 840-d word features (Wang & Ji), within SR", N=40, full_dim=840, unit="Wolpaw bits per sentence query (40-way)")
for c in ["eeg", "noise", "trial_shuffled", "shuffled_pair"]:
    txt[c] = summ([np.mean(per[s][c]) for s in sorted(per)]); txt[c + "_two_way"] = summ([np.mean(per2[s][c]) for s in sorted(per2)])
txt["n_participants"] = len(per)
X1 = json.load(open(R / "track_text/cross_participant_ceiling/SR.json")) if (R / "track_text/cross_participant_ceiling/SR.json").exists() else None
if X1: txt["cross_reader_splithalf_raw"] = {k: v for k, v in X1.items() if not isinstance(v, (list, dict))}
OUT["text"]["full_width_ridge"] = txt
# ======================================================================= protocol-schema bottleneck files (new sweeps land here automatically)
def load_protocol_files():
    found = []
    for f in glob.glob(str(R / "loop/**/*.json"), recursive=True) + glob.glob(str(R / "track_text/**/*.json"), recursive=True) + glob.glob(str(R / "yield/**/*.json"), recursive=True):
        if os.path.getsize(f) > 20e6: continue
        try: d = json.load(open(f))
        except Exception: continue
        if isinstance(d, dict) and {"dataset", "decoder", "participant", "bottleneck", "rows"} <= set(d): found.append((f, d))
    return found
groups = defaultdict(list)
for f, d in load_protocol_files(): groups[(d["dataset"], d["decoder"], d["bottleneck"], d.get("target", ""))].append((f, d))
for (ds, dec, bn, tg), items in groups.items():
    mod = "image" if "things" in ds.lower() else "text" if "zuco" in ds.lower() else "pe"
    for metric in ["bits_NCE", "bits_g", "bits_F"]:
        per = defaultdict(lambda: defaultdict(lambda: defaultdict(list))); Ls = set(); fd = None
        for f, d in items:
            fd = fd or d.get("full_dim") or d.get("input_dim")
            for r in d["rows"]:
                if metric not in r or r[metric] is None: continue
                L = r["L"] if r["L"] == "full" else int(r["L"]); per[str(d["participant"])][r["condition"]][L].append(r[metric])
                if L != "full": Ls.add(L)
        if not Ls: continue
        pp = {p: {c: {L: float(np.mean(v)) for L, v in cc.items()} for c, cc in pc.items()} for p, pc in per.items()}
        fd = fd or int(round(1.6 * max(Ls)))
        conds = sorted({c for p in pp.values() for c in p})
        key = f"protocol_{ds}_{dec}_{metric}".replace(" ", "_")
        OUT[mod][key] = curve_block(pp, sorted(Ls), conds, fd, metric, sorted(f for f, _ in items)[0].replace(str(R), "results") + f" (+{len(items) - 1} files)", f"{ds} / {dec} / {bn} / {tg}")
        OUT[mod][key]["dataset"] = ds; OUT[mod][key]["decoder"] = dec; OUT[mod][key]["bottleneck"] = bn
        # PROTOCOL f/g: L90 is defined only when the full-width yield's CI clears every floor at full width (random projection is a curve, not a floor)
        fl_full = [OUT[mod][key]["curves"][c]["full"]["mean"] for c in OUT[mod][key]["curves"] if c in ("noise", "shuffled_pair", "trial_shuffled", "perm_p95", "null_p95") and "full" in OUT[mod][key]["curves"][c]]
        OUT[mod][key]["floor_full_max"] = max(fl_full) if fl_full else None
        OUT[mod][key]["L90_defined"] = bool(fl_full) and OUT[mod][key]["full_yield"]["ci95"][0] > max(fl_full)
        ceil = [d.get("ceiling") for _, d in items if d.get("ceiling")]
        if ceil: OUT[mod][key]["ceiling_raw"] = ceil[0]
    OUT["protocol_files"].append(dict(dataset=ds, decoder=dec, bottleneck=bn, target=tg, n_files=len(items)))
(R / "stats").mkdir(exist_ok=True); (R / "stats/yield_stats.json").write_text(json.dumps(OUT, indent=1))
# ---- print a compact summary
for mod in ["image", "pe", "text"]:
    for k, v in OUT[mod].items():
        if isinstance(v, dict) and "L90" in v: print(f"{mod:5s} {k:32s} n={v['n']:2d} full={v['full_yield']['mean']:.3f} {v['full_yield']['ci95']} L90 median={v['L90']['median']:.1f} {v['L90']['median_ci95']} range=[{v['L90']['min']:.0f},{v['L90']['max']:.0f}]")
for k, v in OUT["image"]["usable_fraction"].items(): print("image frac", k, json.dumps({kk: vv for kk, vv in v.items() if kk != "per_participant"})[:300])
for k, v in OUT["pe"]["usable_fraction"].items(): print("pe frac", k, {kk: vv for kk, vv in v.items() if kk != "per_participant"})
print("L90 ratio", {k: v for k, v in OUT["image"]["L90_input_over_representation"].items() if k != "per_participant"})
print("text", {k: (v["mean"], v["ci95"]) for k, v in txt.items() if isinstance(v, dict) and "mean" in v})
print("protocol files:", OUT["protocol_files"])
