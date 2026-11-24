"""Aggregate MI retest replicate rows (, 23 Nov 2026;). numpy only; run through srun.
R = within-participant (leave-one-run-pair-out) yield, Y = LPGO primary yield of the same read-out (mi_card_v3 / mi_fm_eogreg per-participant files).
Per participant gain = EEG bits - own design's noise-input blank. U = mean(Y - blank_Y) / mean(R - blank_R), participant bootstrap (10000, joint).
Check (i) same scorer AND budget: static, fails for every row (within = 2 run pairs, ~56 training trials, kernel-form logistic / 600 EEGNet steps;
LPGO = 9/10 of the cohort, PCA-256 primal logistic / 4000 EEGNet steps). (ii) lower 95% bound of mean(R - blank_R) > 0.
(iii) split-half reliability of R across participants: halves = alternating test trials within participant, Pearson r of the two half means
across participants, Spearman-Brown corrected, participant-bootstrap lower bound > 0.1 (needs per-trial bits; not available for the saved EEGNet row)."""
import glob, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/path/to/eegaudit"); G = ROOT / "results/gpu_fill_20261123"
rng = np.random.default_rng(0); B = 10000


def load_within(dirs, row):
    R, b, sh, pt = {}, {}, {}, {}
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/sub-*.json")):
            j = json.load(open(f)); s = j["participant"]; R[s] = j[row]["eeg"]["bits"]; b[s] = j[row]["noise"]["bits"]
            if "shuffle" in j[row]: sh[s] = j[row]["shuffle"]["bits"]
        z = Path(d) / "pertrial.npz"
        if z.exists():
            q = np.load(z); sidv = q["sid"]; Ps = q["participants"]; key = row.replace("|", "__") + "__eeg"
            for k, s in enumerate(Ps):
                v = q[key][sidv == k]; nk = key.replace("__eeg", "__noise")
                if nk in q.files: v = v - q[nk][sidv == k]   # reviewer fix: reliability of EEG minus noise-input per trial (a null row must not pass)
                pt[int(s)] = (v[0::2].mean(), v[1::2].mean())
    return R, b, sh, pt


def load_lpgo(fn, s_list, get):
    Y, bY = {}, {}
    for s in s_list:
        j = json.load(open(fn(s))); Y[s], bY[s] = get(j)
    return Y, bY


def ci(v): return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def analyse(name, label, R, bR, Y, bY, pt):
    S = sorted(set(R) & set(Y)); n = len(S)
    r = np.array([R[s] for s in S]); br = np.array([bR[s] for s in S]); y = np.array([Y[s] for s in S]); by = np.array([bY[s] for s in S])
    idx = rng.integers(0, n, (B, n)); out = dict(name=name, label=label, n=n)
    out["R"] = dict(mean=float(r.mean()), ci=ci(r[idx].mean(1))); out["R_blank"] = float(br.mean()); out["R_blank_max_abs_participant"] = float(np.abs(br).max())
    out["R_gain"] = dict(mean=float((r - br).mean()), ci=ci((r - br)[idx].mean(1)))
    out["Y"] = dict(mean=float(y.mean()), ci=ci(y[idx].mean(1))); out["Y_blank"] = float(by.mean())
    gy, gr = y - by, r - br; U = gy.mean() / gr.mean(); Ub = gy[idx].mean(1) / gr[idx].mean(1)
    den = ci(gr[idx].mean(1)); stable = bool(den[0] > 0)   # reviewer fix: U reported only when the denominator CI excludes 0
    out["U"] = dict(value=float(U) if stable else None, ci=ci(Ub) if stable else None, unstable=not stable, denominator_ci=den)
    out["check_i_same_scorer_budget"] = None   # reviewer fix: not a constant; within-participant run-pair retest differs from LPGO by design, so no usage ratio across designs is claimed
    out["check_ii_R_ci_excludes_blank"] = bool(out["R_gain"]["ci"][0] > 0)
    out["blank_gate_ok"] = bool(abs(br.mean()) <= 0.02 and abs(by.mean()) <= 0.02)
    if pt and len(set(S) & set(pt)) == n:
        a = np.array([pt[s][0] for s in S]); c = np.array([pt[s][1] for s in S])
        def sb(ix): rr = np.corrcoef(a[ix], c[ix])[0, 1]; return 2 * rr / (1 + rr)
        v = np.array([sb(i) for i in idx]); out["check_iii_split_half"] = dict(sb=float(sb(np.arange(n))), ci=ci(v), pass_lower_gt_0p1=bool(ci(v)[0] > 0.1))
    else: out["check_iii_split_half"] = None
    iii = out["check_iii_split_half"]
    out["anchor_ok"] = bool(out["check_ii_R_ci_excludes_blank"] and iii and iii["pass_lower_gt_0p1"])   # checks ii and iii only; same-session run-pair retest
    out["verdict"] = "same-session retest reproduces (checks ii, iii)" if out["anchor_ok"] else "same-session retest does not pass checks ii/iii"
    return out


def main():
    rows = []
    allp = [json.load(open(f))["participant"] for f in sorted(glob.glob(str(ROOT / "results/ext/mi_card_v3/primary_lpgo/sub-*.json")))]
    v3 = lambda row: (lambda s: str(ROOT / f"results/ext/mi_card_v3/primary_lpgo/sub-{s:03d}.json"))
    def v3get(row): return lambda j: (j[row]["eeg"]["bits"], j[row]["noise"]["bits"])
    spec = [("raw voltage", "raw", [G / "mi_retest_logistic"], "raw", "raw"), ("raw proxies-regressed", "raw_eogreg", [G / "mi_retest_logistic"], "raw_eogreg", "raw_eogreg"),
            ("EEGNet as recorded (saved within, 600 steps)", "eegnet", [ROOT / "results/ext/mi_card_within_eegnet/secondary_within"], "eegnet", "eegnet"),
            ("EEGNet regressed", "eegnet_eogreg", [G / f"mi_retest_eegnet_eogreg/s{i}" for i in range(8)], "eegnet_eogreg", "eegnet_eogreg")]
    for lab, wrow, dirs, lrow, _ in spec:
        dirs = [d for d in dirs if Path(d).exists()]
        if not dirs: continue
        R, bR, sh, pt = load_within(dirs, wrow); Y, bY = load_lpgo(v3(lrow), [s for s in R], v3get(lrow))
        o = analyse(lab, wrow, R, bR, Y, bY, pt); o["R_shuffle_mean"] = float(np.mean(list(sh.values()))) if sh else None; rows.append(o)
    for m, lab in (("cbramod", "disjoint"), ("reve", "disjoint"), ("labram", "transductive"), ("eegpt", "transductive")):
        d = G / f"mi_retest_fm_{m}"
        if not d.exists(): continue
        R, bR, sh, pt = load_within([d], f"fm_{m}"); fn = lambda s, m=m: str(ROOT / f"results/ext/mi_fm_eogreg/{m}/sub-{s:03d}.json")
        Y, bY = load_lpgo(fn, list(R), lambda j: (j["fm"]["eeg"]["bits"], j["fm"]["noise"]["bits"]))
        o = analyse(f"{m.upper()} ({lab}), regressed input", f"fm_{m}", R, bR, Y, bY, pt); o["R_shuffle_mean"] = float(np.mean(list(sh.values()))) if sh else None; rows.append(o)
    (G / "mi_retest_summary.json").write_text(json.dumps(rows, indent=1)); print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
