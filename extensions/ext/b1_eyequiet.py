"""b1 EYE-QUIET TRIAL SUBSET (METHOD NOVEL SOLUTIONS ROUND 1, b1; rung R6). EXECUTOR-3, 21 Nov 2026.
Question: does the PE yield survive when the read-out is trained AND scored only on trials without blinks or saccades?
v2 (METHOD ruling 13:00, fixed before any result was read): the ds003458 EOG channel is not on a uV scale (v1's 75 uV rule marked 0 quiet
  trials). PRIMARY quiet rule = scale-free robust rule on the vertical EOG: eye-movement trial iff EOG p2p > participant median + 3 x MAD;
  SECONDARY = max(Fp1, Fp2) p2p < 75 uV with the EEG unit check printed (continuous Fp SD in uV). No fixed percentile. Quiet fraction and
  outcome (reward) rate in quiet vs all trials printed. Same size-matched random-subset comparison; falsifier: kept < 80 %.
Quiet trial (threshold fixed in advance, rung0_ocular.py): pre-ICA vertical EOG peak-to-peak < 75 uV in -200..600 ms around feedback (raw
  ds003458 EEGLAB files, 0.1-30 Hz, average reference, 50 Hz, the rung-0 pipeline; cached per participant in results/ext/eyequiet/ptp_sub-XXX.npz).
Read-out and score: the X7 raw-ERP ridge (results/features/raw, 150-600 ms), locked PE log-score with nested calibration, 10 contiguous folds
  over the subset's rows in time order (training AND test restricted to the subset); widths PCA-64 (CHANGE 12 primary) and full.
Comparison: 20 size-matched random subsets of ALL trials (seeds 0-19, same size as the quiet subset, time order kept). Per participant:
  kept = (quiet yield - floor) / mean(random yield - floor); group = ratio of participant means with a GPU bootstrap CI.
Falsifier (pre-stated): the quiet-trial yield < 80 % of the size-matched random-subset yield (paired participant CI) -> the yield depends on
  eye-movement trials. Floors: circular shifts n/4, n/2, 3n/4 within the subset. G2: the 'all trials' arm = results/standard/pe_bottleneck.
Participants with < 100 quiet trials are reported and excluded from the group statistic (stated).
Usage: python b1_eyequiet.py [--subjects 1,2,3] --out DIR | --summarize --out DIR
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/path/to/eegaudit"); os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "experiments/ext")); sys.path.insert(0, str(ROOT))
import d2_pe_tasks as D2

DEV = "cuda"; LS = (64, "full"); NRAND = 20; THR = 75.0; MIN_Q = 100
DS = os.environ.get("B1_DS", "ds003458")     # second PE dataset: B1_DS=ds004595 (vertical EOG = "Below Eye"; features_ds004595/raw; no standard G2 file)


def eog_ptp(s, out):
    """Pre-ICA peak-to-peak in -200..600 ms per feedback epoch: vertical EOG channel and max(Fp1, Fp2) (rung-0 pipeline). Cached."""
    f = out / f"ptp2_sub-{s:03d}.npz"
    if f.exists(): z = np.load(f); return z["tri"], z["eog"], z["fp"], float(z["fp_sd_uV"])
    import mne
    if DS == "ds003458":
        from cogmodels import models as M
        from cogmodels.data import load_subject
        root = Path("data_raw/ds003458"); d = load_subject(root, s, M.reward_schedule(480).numpy()); fb = d["fb_onset"][:d["T"]]
    else:
        import ds004595_task as T4
        root = Path("data_raw/ds004595"); d = T4.load(s); fb = np.where(d["missing"], np.nan, d["fb_onset"])
    raw = mne.io.read_raw_eeglab(root / f"sub-{s:03d}/eeg/sub-{s:03d}_task-ThreeArmedBandit_eeg.set", preload=True, verbose="error")
    eog = [c for c in raw.ch_names if "EOG" in c.upper() or c == "Below Eye"]; assert eog, "no EOG channel"
    if DS != "ds003458":
        std = set(mne.channels.make_standard_montage("standard_1005").ch_names) | {"CPz"}
        raw.set_channel_types({c: "misc" for c in raw.ch_names if c not in std and c not in eog and c != "Above Eye"} | ({"Above Eye": "eog"} if "Above Eye" in raw.ch_names else {}), verbose="error")
    raw.set_channel_types({c: "eog" for c in eog}, verbose="error")
    if "CPz" not in raw.ch_names: mne.add_reference_channels(raw, "CPz", copy=False)
    # METHOD BUG HUNT B2: re-reference the vertical EOG to the common average BEFORE the EEG average reference is applied: EOG minus the mean of the
    # EEG channels on the original reference (incl. the re-added CPz reference channel) = the EOG on the average reference
    eegn = [c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"]
    raw._data[raw.ch_names.index(eog[0])] -= raw._data[[raw.ch_names.index(c) for c in eegn]].mean(0)
    raw.set_eeg_reference("average", verbose="error")
    raw.filter(0.1, 30.0, picks="all", verbose="error")
    ok = np.where(~np.isnan(fb))[0]; samp = np.round(fb[ok] * raw.info["sfreq"]).astype(int); ev = np.c_[samp, np.zeros_like(samp), np.ones_like(samp)]
    pk = [eog[0], "Fp1", "Fp2"]
    ep = mne.Epochs(raw, ev, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0), picks=pk, preload=True, reject=None, decim=int(round(raw.info["sfreq"] / 50)), verbose="error")
    X = ep.get_data(picks=pk) * 1e6; tri = ok[ep.selection]; P = X.max(2) - X.min(2)
    fp_sd = float(raw.get_data(picks=["Fp1", "Fp2"]).std() * 1e6)                     # EEG unit check (printed): continuous filtered Fp SD in uV
    np.savez(f, tri=tri, eog=P[:, 0], fp=P[:, 1:].max(1), fp_sd_uV=fp_sd); return tri, P[:, 0], P[:, 1:].max(1), fp_sd


def block(F, y, s):
    n = len(y); Ys = np.column_stack([y] + [np.roll(y, round(n * q / 4)) for q in (1, 2, 3)]); sc = D2.pe_gain(F, Ys, LS)
    return {str(L): dict(gain=float(sc[L][:, 0].mean()), shift_mean=float(sc[L][:, 1:].mean())) for L in LS}


def subject(s, out):
    t0 = time.time(); info = D2.dataset(DS); fz = np.load(ROOT / ("results/features/raw" if DS == "ds003458" else "results/features_ds004595/raw") / f"sub-{s:03d}.npz"); tri = fz["tri"]; F = fz["F"].astype(np.float64)
    etri, pe, pf, fp_sd = eog_ptp(s, out); me = dict(zip(etri.tolist(), pe.tolist())); mf = dict(zip(etri.tolist(), pf.tolist()))
    has = np.array([t in me for t in tri]); E = np.array([me.get(t, np.inf) for t in tri]); Pf = np.array([mf.get(t, np.inf) for t in tri])
    med = float(np.median(E[has])); mad = float(np.median(np.abs(E[has] - med)))
    quiet = {"primary_eog_mad": has & (E <= med + 3 * mad), "secondary_fp75": has & (Pf < THR)}
    if DS == "ds003458":
        Q = np.load(ROOT / "results/full_36318/behaviour_quantities_flow.npz"); rew = Q["reward"][s - 1][tri].astype(float)
    else:
        import ds004595_task as T4; rew = np.asarray(T4.load(s)["reward"], float)[tri]
    res = dict(participant=f"sub-{s:03d}", n_trials=int(len(tri)), n_with_eog=int(has.sum()), eog_ptp_median=med, eog_ptp_mad=mad, fp_sd_uV=fp_sd,
               fp_ptp_median_uV=float(np.median(Pf[has])), outcome_rate_all=float(rew.mean()), rules={})
    res["all"] = block(F, info["qtil"](s, tri).astype(np.float64), s)
    if DS == "ds003458":
        st = json.loads((ROOT / f"results/standard/pe_bottleneck/sub-{s:03d}.json").read_text())
        res["g2"] = {str(L): dict(old=next(r["gain"] for r in st["rows"] if r["input"] == "eeg" and r["mode"] == "pca" and r["L"] == L), new=res["all"][str(L)]["gain"]) for L in LS}
        res["g2"]["passed"] = bool(all(abs(v["old"] - v["new"]) < 1e-9 for k, v in res["g2"].items() if k != "passed"))
    else:
        res["g2"] = dict(passed=True, note="no standard per-participant file for " + DS)
    res["dataset"] = DS
    # FIXER (R2 ruling 4, b1): the random comparator is drawn STRATIFIED to the quiet subset's outcome x |PE|-decile mix (deciles over all
    # trials of the participant); the target range (SD of PE and of |PE|) of quiet, all and the matched comparator is printed
    pe_all = info["qtil"](s, tri).astype(np.float64); ape = np.abs(pe_all)
    dec = np.searchsorted(np.quantile(ape, np.linspace(0, 1, 11)[1:-1]), ape, side="right"); strat = np.unique(rew, return_inverse=True)[1] * 10 + dec
    def strat_draw(k, idx):
        rg = np.random.default_rng(k); pick = []
        for st_ in np.unique(strat[idx]):
            pick.append(rg.choice(np.where(strat == st_)[0], int((strat[idx] == st_).sum()), replace=False))
        return np.sort(np.concatenate(pick))
    for rule, qm in quiet.items():
        nq = int(qm.sum()); R_ = dict(n_quiet=nq, quiet_fraction=nq / len(tri), outcome_rate_quiet=float(rew[qm].mean()) if nq else None)
        if nq >= MIN_Q:
            idx = np.where(qm)[0]; R_["quiet"] = block(F[idx], info["qtil"](s, tri[idx]).astype(np.float64), s); rnd = []
            rs_ = []
            for k in range(NRAND):
                r = strat_draw(k, idx); assert len(r) == nq; rs_.append(r); rnd.append(block(F[r], info["qtil"](s, tri[r]).astype(np.float64), s))
            R_["comparator"] = "stratified: outcome x |PE| decile, matched to the quiet subset"
            R_["comparator_overlap_with_quiet"] = float(np.mean([np.isin(r, idx).mean() for r in rs_]))   # FIXER: share of comparator trials that are quiet
            R_["target_range"] = dict(sd_pe_all=float(pe_all.std()), sd_pe_quiet=float(pe_all[idx].std()), sd_pe_random=float(np.mean([pe_all[r].std() for r in rs_])),
                                      sd_abs_pe_all=float(ape.std()), sd_abs_pe_quiet=float(ape[idx].std()), sd_abs_pe_random=float(np.mean([ape[r].std() for r in rs_])),
                                      mean_abs_pe_all=float(ape.mean()), mean_abs_pe_quiet=float(ape[idx].mean()), mean_abs_pe_random=float(np.mean([ape[r].mean() for r in rs_])))
            R_["random"] = {str(L): dict(gain_mean=float(np.mean([x[str(L)]["gain"] for x in rnd])), gain_sd=float(np.std([x[str(L)]["gain"] for x in rnd])),
                                         shift_mean=float(np.mean([x[str(L)]["shift_mean"] for x in rnd]))) for L in LS}
        else:
            R_["excluded"] = f"< {MIN_Q} quiet trials"
        res["rules"][rule] = R_
    res["runtime_s"] = round(time.time() - t0, 1); (out / f"b1_sub-{s:03d}.json").write_text(json.dumps(res, indent=1))
    pr = res["rules"]["primary_eog_mad"]
    print(f"sub-{s:03d}", "quiet(primary)", pr["n_quiet"], "/", len(tri), "pca64 all", round(res["all"]["64"]["gain"], 4), "quiet", pr.get("quiet", {}).get("64", {}).get("gain"),
          "random", pr.get("random", {}).get("64", {}).get("gain_mean"), "fp_sd", round(fp_sd, 1), "G2", res["g2"]["passed"], "overlap", round(pr.get("comparator_overlap_with_quiet", float("nan")), 3), "target SD |PE| quiet/all/random", {k: round(v, 3) for k, v in pr.get("target_range", {}).items() if "sd_abs" in k}, f"{res['runtime_s']}s", flush=True)


def summarize(out):
    per = [json.loads(f.read_text()) for f in sorted(out.glob("b1_sub-*.json"))]; res = dict(n=len(per), g2_all_pass=bool(all(p["g2"]["passed"] for p in per)),
                                                                                  fp_sd_uV_median=float(np.median([p["fp_sd_uV"] for p in per])), outcome_rate_all=float(np.mean([p["outcome_rate_all"] for p in per])))
    T = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float64, device=DEV); q = lambda x: torch.quantile(x, torch.tensor([.025, .975], dtype=torch.float64, device=DEV)).cpu().tolist()
    for rule in ("primary_eog_mad", "secondary_fp75"):
        use = [p for p in per if "quiet" in p["rules"][rule]]; n = len(use)
        R = dict(comparator="stratified outcome x |PE| decile (FIXER v3)", target_range={k: float(np.mean([p["rules"][rule]["target_range"][k] for p in use])) for k in (use[0]["rules"][rule].get("target_range", {}) if use else {})}, comparator_overlap_with_quiet=float(np.mean([p["rules"][rule]["comparator_overlap_with_quiet"] for p in use])) if use else None, n_used=n, excluded=[p["participant"] for p in per if "quiet" not in p["rules"][rule]], quiet_fraction_mean=float(np.mean([p["rules"][rule]["quiet_fraction"] for p in per])), quiet_fraction_per_participant={p["participant"]: p["rules"][rule]["quiet_fraction"] for p in per},
                 outcome_rate_quiet=float(np.nanmean([p["rules"][rule]["outcome_rate_quiet"] if p["rules"][rule]["outcome_rate_quiet"] is not None else np.nan for p in per])))
        if n >= 3:
            g = torch.Generator(device=DEV); g.manual_seed(20261121); IX = torch.randint(0, n, (10000, n), generator=g, device=DEV)
            for L in ("64", "full"):
                Qd = T([p["rules"][rule]["quiet"][L]["gain"] - p["rules"][rule]["quiet"][L]["shift_mean"] for p in use]); Rm = T([p["rules"][rule]["random"][L]["gain_mean"] - p["rules"][rule]["random"][L]["shift_mean"] for p in use])
                kept = float(Qd.mean() / Rm.mean()); kci = q(Qd[IX].mean(1) / Rm[IX].mean(1))
                R["L" + L] = dict(quiet_minus_floor=float(Qd.mean()), random_minus_floor=float(Rm.mean()), kept_ratio_of_means=kept, kept_ci=kci,
                                  quiet_minus_random=dict(d=float((Qd - Rm).mean()), ci=q((Qd - Rm)[IX].mean(1)), n_pos=int((Qd > Rm).sum())),
                                  falsifier_triggered=bool(kci[1] < 0.8), reading="depends on eye-movement trials" if kci[1] < 0.8 else ("not shown to depend on eye-movement trials" if kept >= 0.8 else "inconclusive"))
        res[rule] = R
    (out / "summary_b1.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default=""); ap.add_argument("--out", required=True); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize: return summarize(out)
    for s in ([int(x) for x in a.subjects.split(",")] if a.subjects else D2.dataset(DS)["subs"]):
        if not (out / f"b1_sub-{s:03d}.json").exists(): subject(s, out)


if __name__ == "__main__":
    main()
