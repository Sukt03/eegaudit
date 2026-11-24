"""SPECIFICITY control for the PE EOG-regressed yield (METHOD AUDITS 12:40, after pe_eogreg PASS "partly ocular", kept 0.375). EXECUTOR-3, 21 Nov 2026.
Question: is the 62 % loss under ocular regression eye-specific, or does regressing out ANY 70 size-matched pre-ICA derivations remove as much?
Same rows, code and estimator as pe_eogreg.py (X7 features, locked PE log-score, nested calibration, PCA-64 primary + full, in-fold least-squares
regression of the regressors out of every feature column, fit on each fit's training rows only, batched GPU).
Control regressors (size-matched to the 70 ocular columns): PRE-ICA posterior derivations O1-O2, P7-P8, Pz-POz at 150-600 ms / 50 Hz
  (3 x 23 = 69 columns, the rung-0 pre-ICA pipeline: average reference, 0.1-30 Hz, -0.2..0.8 s epochs, baseline -0.2..0) + a dummy flag
  (constant 0, like the ocular blink flag's role) = 70 columns. Trials = the pe_eogreg rows (X7 trials with an ocular epoch).
Kept share = (control-regressed - floor) / (all - floor), ratio of participant means, GPU bootstrap CI, printed beside the ocular 0.375.
Reading: if the posterior regression also removes ~60 %, "partly ocular" is not eye-specific (it would be generic regression shrinkage).
Usage: python pe_specificity.py --out DIR [--subjects ..] | --summarize --out DIR
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/path/to/eegaudit"); os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "experiments/ext")); sys.path.insert(0, str(ROOT))
import d2_pe_tasks as D2
from pe_eogreg import eogreg, LS

DEV = "cuda"; PAIRS = [("O1", "O2"), ("P7", "P8"), ("Pz", "POz")]


def posterior_regs(sub, trials_keep):
    import mne
    from cogmodels import models as M
    from cogmodels.data import load_subject
    root = Path("data_raw/ds003458"); d = load_subject(root, sub, M.reward_schedule(480).numpy()); fb = d["fb_onset"][:d["T"]]
    raw = mne.io.read_raw_eeglab(root / f"sub-{sub:03d}/eeg/sub-{sub:03d}_task-ThreeArmedBandit_eeg.set", preload=True, verbose="error")
    eog = [c for c in raw.ch_names if "EOG" in c.upper()]; raw.set_channel_types({c: "eog" for c in eog}, verbose="error")
    if "CPz" not in raw.ch_names: mne.add_reference_channels(raw, "CPz", copy=False)
    raw.set_eeg_reference("average", verbose="error"); raw.filter(0.1, 30.0, picks="all", verbose="error")
    ok = np.where(~np.isnan(fb))[0]; samp = np.round(fb[ok] * raw.info["sfreq"]).astype(int); ev = np.c_[samp, np.zeros_like(samp), np.ones_like(samp)]
    picks = [c for p in PAIRS for c in p]; assert all(c in raw.ch_names for c in picks), [c for c in picks if c not in raw.ch_names]
    ep = mne.Epochs(raw, ev, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), picks=picks, preload=True, reject=None, decim=int(round(raw.info["sfreq"] / 50)), verbose="error")
    X = ep.get_data(picks=picks) * 1e6; tri = ok[ep.selection]; w = (ep.times >= 0.15) & (ep.times <= 0.60)
    feats = [X[:, picks.index(a), w] - X[:, picks.index(b), w] for a, b in PAIRS]; C = np.c_[np.concatenate(feats, 1), np.zeros(len(X))]
    pos = {t: i for i, t in enumerate(tri)}; keep = np.array([t in pos for t in trials_keep])
    return C[[pos[t] for t in trials_keep[keep]]], keep


def subject(s, out):
    import rung0_ocular as R0
    t0 = time.time(); sub = f"sub-{s:03d}"; info = D2.dataset("ds003458")
    fz = np.load(ROOT / f"results/features/raw/{sub}.npz"); tri0 = fz["tri"]; F0 = fz["F"].astype(np.float64)
    O, keep_o, _, _ = R0.ocular(s, tri0); tri1 = tri0[keep_o]
    Cp, keep_c = posterior_regs(s, tri1); assert keep_c.all(), "posterior epochs missing for some ocular rows"
    F = F0[keep_o]; n = len(tri1); y = info["qtil"](s, tri1).astype(np.float64); Ys = np.column_stack([y] + [np.roll(y, round(n * q / 4)) for q in (1, 2, 3)])
    rows = {}
    for arm, tf in (("all", None), ("all_eogreg", eogreg(O.astype(np.float64))), ("all_postreg", eogreg(Cp.astype(np.float64)))):
        sc = D2.pe_gain(F, Ys, LS, transform=tf); nz = D2.pe_gain(D2.gnoise(F.shape, 1000 + s), Ys[:, :1], LS)
        rows[arm] = {str(L): dict(gain=float(sc[L][:, 0].mean()), shift_mean=float(sc[L][:, 1:].mean()), noise=float(nz[L][:, 0].mean()),
                                  floor=max(float(sc[L][:, 1:].mean()), float(nz[L][:, 0].mean()))) for L in LS}
    res = dict(participant=sub, n_trials=int(n), n_control_columns=int(Cp.shape[1]), n_ocular_columns=int(O.shape[1]), control=[f"{a}-{b}" for a, b in PAIRS] + ["dummy flag"],
               rows=rows, runtime_s=round(time.time() - t0, 1))
    (out / f"spec_{sub}.json").write_text(json.dumps(res, indent=1))
    print(sub, {a: round(rows[a]["64"]["gain"], 4) for a in rows}, f"{res['runtime_s']}s", flush=True)


def summarize(out, sout=None):
    # FIXER (R2 ruling 4): the ocular reference is READ from the eogreg summary (no hard-coded 0.375); label "not diagnostic of ocular origin"
    EG = ROOT / "results/ext/eogreg/summary_eogreg.json"; eg = json.loads(EG.read_text())
    per = [json.loads(f.read_text()) for f in sorted(out.glob("spec_sub-*.json"))]; n = len(per)
    res = dict(n=n, ocular_kept_reference={("L" + L): eg["L" + L]["reading"]["kept_share"] for L in ("64", "full")}, ocular_kept_reference_source=str(EG),
               label="continuous regression of any 70 correlated EEG columns removes a similar share; not diagnostic of ocular origin")
    g = torch.Generator(device=DEV); g.manual_seed(20261121); IX = torch.randint(0, n, (10000, n), generator=g, device=DEV)
    T = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float64, device=DEV); q = lambda x: torch.quantile(x, torch.tensor([.025, .975], dtype=torch.float64, device=DEV)).cpu().tolist()
    for L in ("64", "full"):
        d = {a: T([p["rows"][a][L]["gain"] - p["rows"][a][L]["floor"] for p in per]) for a in ("all", "all_eogreg", "all_postreg")}
        R = {a: dict(gain=float(np.mean([p["rows"][a][L]["gain"] for p in per]))) for a in d}
        for a in ("all_eogreg", "all_postreg"):
            R[a].update(kept=float(d[a].mean() / d["all"].mean()), kept_ci=q(d[a][IX].mean(1) / d["all"][IX].mean(1)))
        diff = d["all_postreg"] - d["all_eogreg"]; R["postreg_minus_eogreg"] = dict(d=float(diff.mean()), ci=q(diff[IX].mean(1)), n_pos=int((diff > 0).sum()))
        R["comparison"] = "posterior regression removes less than ocular regression" if q(diff[IX].mean(1))[0] > 0 else "posterior regression removes a comparable share"
        R["reading"] = R["comparison"] + "; not diagnostic of ocular origin (both regressions remove brain signal)"; R["ocular_kept_reference"] = res["ocular_kept_reference"]["L" + L]
        res["L" + L] = R
    sout = Path(sout) if sout else out; sout.mkdir(parents=True, exist_ok=True)
    (sout / "summary_specificity.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default=""); ap.add_argument("--out", required=True); ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--summary_out", default="")   # FIXER: write the re-summary to a NEW folder (old summary untouched)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize: return summarize(out, a.summary_out or None)
    for s in ([int(x) for x in a.subjects.split(",")] if a.subjects else range(1, 24)):
        if not (out / f"spec_sub-{s:03d}.json").exists(): subject(s, out)


if __name__ == "__main__":
    main()
