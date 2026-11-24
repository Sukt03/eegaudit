"""v2 (METHOD BUG HUNT B1/B2, 21 Nov ~14:00): VEOG re-referenced to the average; EVENT-predictor arm (blink + saccade flags, robust per participant);
ROW-PERMUTED-regressor null (5 permutations, same 70 columns + intercept, same in-fold OLS) as the denominator of the PRIMARY kept share;
"eog_only" renamed "ocular_features" (includes frontal brain signal). Outputs results/ext/eogreg_v2/.
PE EOG-REGRESSED YIELD, ds003458 (METHOD RESULTS REVIEW ROUND 1, PE row (c)(1): "the single most important check for the PE headline";
rung R6). EXECUTOR-3, 21 Nov 2026.
Question: how much of the raw-ERP ridge PE yield survives when the pre-ICA ocular signals are projected out of the features?
Data: X7 features results/features/raw (64 ch x 23 samples, 150-600 ms, 50 Hz, post-ICA), the 23 participants; target = the locked PE
  (s13b targets_ds003458, z-scored residualised PE). Ocular regressors O = rung0_ocular.ocular() (audited rung-0 set: PRE-ICA vertical
  EOG, Fp1-Fp2, F7-F8 at 150-600 ms / 50 Hz, blink flag; 70 columns); trials = X7 trials with an ocular epoch (the same rows in every arm).
Arms (same rows, same estimator):
  all              X7 features (reference; G2 = results/standard/pe_bottleneck when no trial is lost)
  all_eogreg       X7 features with O (+ intercept) regressed out of EVERY feature column IN-FOLD: coefficients fit by least squares on each
                   fit's training rows only, applied to its training and held-out rows (no test row and no target enters the regression)
  interior_eogreg  the 49 interior channels (rung-0 RING removed: Fp/AF/F7/F8/temporal ring), ocular-regressed in-fold (rung-0 arm)
  eog_only         O as the features (what the eyes alone carry)
Read-out/score: the locked PE log-score (nested calibration, 10 contiguous folds) with in-fold z-score + ridge LAM; widths PCA-64 (METHOD
  CHANGE 12 primary for ridge) and full (the StdPE ridge row). All 100 nested fits of an arm = one batched GPU call (d2_pe_tasks.ridge_batch);
  the in-fold regression is one batched least-squares solve over the 100 fits. Floors: circular shifts n/4, n/2, 3n/4; noise input.
Summary (--summarize): per arm participant mean [bootstrap CI, GPU index tensor], n above floor; paired all_eogreg - all and the retained
  share = mean(all_eogreg - floor) / mean(all - floor) (ratio of participant means) with a paired bootstrap CI; eog_only vs floor.
  Pre-stated reading (METHOD, before the result): "not ocular" iff all_eogreg keeps >= 80 % of the yield (ratio of participant means of
  yield - floor, bootstrap CI lower >= 0.60); "partly ocular" if it keeps < 50 %; else "mixed"; reported beside N1's ocular-proxy share 0.53.
  EEGNet arm: not in this script.
Usage: python pe_eogreg.py [--subjects 1,2] --out DIR | --summarize --out DIR
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/path/to/eegaudit"); os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "experiments/ext")); sys.path.insert(0, str(ROOT))
import d2_pe_tasks as D2

DEV = "cuda"; LS = (64, "full")
RING = ["Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "AFz", "F7", "F8", "FT9", "FT10", "T7", "T8", "TP9", "TP10"]


def eogreg(O):
    """Per-fit transform: regress [1, O] out of every feature column with coefficients fit on the fit's training rows (batched lstsq)."""
    Og = torch.as_tensor(np.c_[np.ones(len(O)), O], dtype=torch.float64, device=DEV)
    def t(F, M):
        m = M.double()[:, :, None]; Ot = Og[None] * m; G = Ot.transpose(1, 2) @ Ot + 1e-8 * torch.eye(Og.shape[1], dtype=torch.float64, device=DEV)
        B = torch.linalg.solve(G, Ot.transpose(1, 2) @ (F[None] * m))            # (K, p+1, D) from training rows only
        return F[None] - Og[None] @ B                                            # applied to every row
    return t


def ocular_v2(sub, trials_keep):
    """v2 (METHOD BUG HUNT B2): the rung-0 pre-ICA regressors with the vertical EOG RE-REFERENCED to the common average (EOG minus the mean of the
    EEG channels on the original reference, before the EEG average reference), 0.1-30 Hz, 150-600 ms at 50 Hz: VEOG (23) + Fp1-Fp2 (23) + F7-F8 (23)
    + blink flag = 70 columns; EVENT predictors (scale-free, per participant): blink = VEOG p2p (-200..800 ms) > median + 3 MAD, saccade = (F7-F8)
    p2p > median + 3 MAD. Returns O (n, 70), events (n, 2), keep mask."""
    import mne
    from cogmodels import models as M
    from cogmodels.data import load_subject
    root = Path("data_raw/ds003458"); d = load_subject(root, sub, M.reward_schedule(480).numpy()); fb = d["fb_onset"][:d["T"]]
    raw = mne.io.read_raw_eeglab(root / f"sub-{sub:03d}/eeg/sub-{sub:03d}_task-ThreeArmedBandit_eeg.set", preload=True, verbose="error")
    eog = [c for c in raw.ch_names if "EOG" in c.upper()]; raw.set_channel_types({c: "eog" for c in eog}, verbose="error")
    if "CPz" not in raw.ch_names: mne.add_reference_channels(raw, "CPz", copy=False)
    eegn = [c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"]
    raw._data[raw.ch_names.index(eog[0])] -= raw._data[[raw.ch_names.index(c) for c in eegn]].mean(0)
    raw.set_eeg_reference("average", verbose="error"); raw.filter(0.1, 30.0, picks="all", verbose="error")
    ok = np.where(~np.isnan(fb))[0]; samp = np.round(fb[ok] * raw.info["sfreq"]).astype(int); ev = np.c_[samp, np.zeros_like(samp), np.ones_like(samp)]
    picks = [eog[0], "Fp1", "Fp2", "F7", "F8"]
    ep = mne.Epochs(raw, ev, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), picks=picks, preload=True, reject=None, decim=int(round(raw.info["sfreq"] / 50)), verbose="error")
    X = ep.get_data(picks=picks) * 1e6; tri = ok[ep.selection]; w = (ep.times >= 0.15) & (ep.times <= 0.60)
    V, FP, FF = X[:, 0], X[:, 1] - X[:, 2], X[:, 3] - X[:, 4]
    rob = lambda v: v > np.median(v) + 3 * np.median(np.abs(v - np.median(v)))
    blink = rob(np.ptp(V, 1)).astype(float); sacc = rob(np.ptp(FF, 1)).astype(float)
    O = np.c_[V[:, w], FP[:, w], FF[:, w], blink]; Ev = np.c_[blink, sacc]
    pos = {t: i for i, t in enumerate(tri)}; keep = np.array([t in pos for t in trials_keep]); ix = [pos[t] for t in trials_keep[keep]]
    return O[ix], Ev[ix], keep


def permuted(O, seed):
    """Row-permuted regressors (METHOD BUG HUNT B1 null): same columns, rows shuffled -> no relation to the trial's EEG."""
    return O[np.random.default_rng(seed).permutation(len(O))]


def subject(s, out):
    t0 = time.time(); sub = f"sub-{s:03d}"; info = D2.dataset("ds003458")
    fz = np.load(ROOT / f"results/features/raw/{sub}.npz"); tri0 = fz["tri"]; F0 = fz["F"].astype(np.float64)
    ep = np.load(ROOT / f"results/eeg_measures/{sub}_epochs.npz"); chl = [str(c) for c in ep["ch"]]
    O, Ev, keep = ocular_v2(s, tri0); tri = tri0[keep]; F = F0[keep]; n = len(tri)
    inner = [i for i, c in enumerate(chl) if c not in RING]; Fi = F.reshape(n, len(chl), -1)[:, inner].reshape(n, -1)
    y = info["qtil"](s, tri).astype(np.float64); Ys = np.column_stack([y] + [np.roll(y, round(n * q / 4)) for q in (1, 2, 3)])
    T = eogreg(O.astype(np.float64)); rows = {}
    arms = [("all", F, None), ("all_eogreg", F, T), ("interior_eogreg", Fi, T), ("all_eventreg", F, eogreg(Ev.astype(np.float64))), ("ocular_features", O.astype(np.float64), None)]
    arms += [(f"all_permnull{k}", F, eogreg(permuted(O.astype(np.float64), 500 + k))) for k in range(5)]
    for arm, X, tf in arms:
        sc = D2.pe_gain(X, Ys, LS, transform=tf); nz = D2.pe_gain(D2.gnoise(X.shape, 1000 + s), Ys[:, :1], LS)
        rows[arm] = {str(L): dict(gain=float(sc[L][:, 0].mean()), shifted=[float(v) for v in sc[L][:, 1:].mean(0)], shift_mean=float(sc[L][:, 1:].mean()),
                                  noise=float(nz[L][:, 0].mean()), floor=max(float(sc[L][:, 1:].mean()), float(nz[L][:, 0].mean()))) for L in LS}
        rows[arm]["n_features"] = int(X.shape[1])
    st = json.loads((ROOT / f"results/standard/pe_bottleneck/{sub}.json").read_text())
    g2 = {}
    for L in LS:
        o = next(r for r in st["rows"] if r["input"] == "eeg" and r["mode"] == "pca" and r["L"] == L); g2[str(L)] = dict(old=o["gain"], new=rows["all"][str(L)]["gain"])
    g2 = dict(widths=g2, n_lost=int((~keep).sum()), passed=bool((~keep).sum() > 0 or max(abs(v["old"] - v["new"]) for v in g2.values()) < 1e-9))
    res = dict(dataset="ds003458", participant=sub, version="v2 (METHOD BUG HUNT B1/B2)", n_trials=int(n), n_ocular_features=int(O.shape[1]), blink_rate=float(Ev[:, 0].mean()),
               saccade_rate=float(Ev[:, 1].mean()), interior_channels=[chl[i] for i in inner], unit="bits per feedback trial",
               split="10 contiguous blocks, within participant; regressions fit in-fold on training rows", rows=rows, g2=g2, runtime_s=round(time.time() - t0, 1))
    (out / f"eogreg_{sub}.json").write_text(json.dumps(res, indent=1))
    pn = np.mean([rows[f"all_permnull{k}"]["64"]["gain"] for k in range(5)])
    print(sub, "pca64", {a: round(rows[a]["64"]["gain"], 4) for a in ("all", "all_eogreg", "all_eventreg", "ocular_features")}, "permnull", round(pn, 4), "G2", g2["passed"], f"{time.time() - t0:.0f}s", flush=True)


def summarize(out):
    per = [json.loads(f.read_text()) for f in sorted(out.glob("eogreg_sub-*.json"))]; res = dict(n=len(per), version="v2", participants=[p["participant"] for p in per])
    g = torch.Generator(device=DEV); g.manual_seed(20261121); n = len(per); IX = torch.randint(0, n, (10000, n), generator=g, device=DEV)
    T = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float64, device=DEV)
    q = lambda x: torch.quantile(x, torch.tensor([.025, .975], dtype=torch.float64, device=DEV)).cpu().tolist()
    dm = lambda arm, L: T([p["rows"][arm][L]["gain"] - p["rows"][arm][L]["floor"] for p in per])
    for L in ("64", "full"):
        R = {}
        for arm in ("all", "all_eogreg", "interior_eogreg", "all_eventreg", "ocular_features"):
            gn = T([p["rows"][arm][L]["gain"] for p in per]); fl = T([p["rows"][arm][L]["floor"] for p in per])
            R[arm] = dict(gain=float(gn.mean()), ci=q(gn[IX].mean(1)), floor=float(fl.mean()), n_above_floor=int((gn > fl).sum()))
        PN = torch.stack([dm(f"all_permnull{k}", L) for k in range(5)]).mean(0); R["all_permnull_mean"] = dict(gain_minus_floor=float(PN.mean()), ci=q(PN[IX].mean(1)))
        A_, E_, V_ = dm("all", L), dm("all_eogreg", L), dm("all_eventreg", L)
        kp = E_[IX].mean(1) / PN[IX].mean(1); kpv = float(E_.mean() / PN.mean())
        R["kept_vs_permuted_null"] = dict(share=kpv, ci=q(kp), note="PRIMARY (METHOD BUG HUNT B1): (EOG-regressed - floor) / (row-permuted-regressor null - floor)")
        R["kept_vs_all"] = dict(share=float(E_.mean() / A_.mean()), ci=q(E_[IX].mean(1) / A_[IX].mean(1)), note="v1 definition, secondary")
        R["eventreg_kept_vs_all"] = dict(share=float(V_.mean() / A_.mean()), ci=q(V_[IX].mean(1) / A_[IX].mean(1)))
        R["permnull_kept_vs_all"] = dict(share=float(PN.mean() / A_.mean()), ci=q(PN[IX].mean(1) / A_[IX].mean(1)), note="cost of regressing out 71 unrelated columns")
        ci = R["kept_vs_permuted_null"]["ci"]
        R["reading"] = dict(rule="METHOD (pre-stated), applied to the kept share vs the permuted null: 'not ocular' iff >= 0.80 with CI lower >= 0.60; 'partly ocular' if < 0.50; else 'mixed'",
                            verdict="not ocular" if (kpv >= 0.8 and ci[0] >= 0.6) else ("partly ocular" if kpv < 0.5 else "mixed"),
                            ocular_features_label="ocular regressors used as features (pre-ICA; includes frontal brain signal)")
        res["L" + L] = R
    res["g2_all_pass"] = bool(all(p["g2"]["passed"] for p in per)); res["blink_rate_mean"] = float(np.mean([p["blink_rate"] for p in per])); res["saccade_rate_mean"] = float(np.mean([p["saccade_rate"] for p in per]))
    (out / "summary_eogreg.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1)[:4000])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default=""); ap.add_argument("--out", required=True); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize: return summarize(out)
    for s in ([int(x) for x in a.subjects.split(",")] if a.subjects else range(1, 24)):
        if not (out / f"eogreg_sub-{s:03d}.json").exists(): subject(s, out)


if __name__ == "__main__":
    main()
