"""I-3 PLANTED SUB-THRESHOLD OCULAR PE SIGNAL: a positive control for the two eye controls (IDEATOR round 17:20, I-3). FIXER-2, 21 Nov 2026.
Question: which ocular contributions can each eye control detect? A PE-modulated ocular source is planted into ds003458 and the eye-quiet
  test (b1_eyequiet v3, primary MAD rule) and the event-based regression (pe_eogreg v2 'all_eventreg') are re-run on the planted data.
Plant (per participant; everything that defines it is label-free except the PE scaling itself, which IS the planted signal):
  - Pre-ICA epochs exactly as pe_eogreg.ocular_v2 (VEOG re-referenced to the common average before the EEG average reference, 0.1-30 Hz,
    -0.2..0.8 s, baseline -0.2..0, 50 Hz), but for ALL 64 EEG channels + VEOG.
  - Blink topography b_c = OLS slope of pre-ICA channel c on the VEOG over all (trial, sample) points of the participant's epochs
    (b_VEOG = 1). Fit on all rows, not per fold: it only DEFINES the simulated source (no target enters it; no read-out uses it).
  - Waveform w(t) = mean VEOG epoch of blink-flagged trials minus mean of the other trials (blink = the ocular_v2 event rule), scaled to
    unit peak-to-peak, positive peak; < 10 blink trials -> first temporal PC of the VEOG epochs (printed).
  - Planted VEOG increment P_t(t) = a * MAD * y_t * w(t), y_t = the locked PE target (D2 qtil, z-scored residualised PE), MAD = the b1
    primary-rule MAD of the VEOG p2p (-0.2..0.6 s). Added to the VEOG, to every pre-ICA channel used by the controls (Fp1, Fp2, F7, F8;
    b_c P_t) and to the post-ICA X7 features (64 ch x 150-600 ms; b_c P_t). So every ocular regressor, event flag and p2p is recomputed
    from the planted signals.
  - Sub-threshold criterion (printed per level): the plant alone stays within the eye-movement margin, a * max_t |y_t| <= 3 (plant p2p
    <= 3 MAD on every trial); the number of trials whose primary-rule quiet label flips is printed. If the calibrated amplitude breaks
    the criterion it is CAPPED at 3 / max|y| and the achieved increment is reported (flag 'capped').
  - Size (v3): signed amplitude a = the FIRST crossing (geometric scan x1.5 from 1e-3, then 16-step log-bisection) for each sign; the
    sign with the smaller crossing is used (chosen at 25 %, kept at 50 %). Reason (smokes): the increment is not monotone (a near-
    deterministic plant makes the nested calibration over-confident) and not sign-symmetric (a plant anti-aligned with the
    participant's own EEG PE code first cancels it). If the target is not reached within the cap, the scanned amplitude
    with the largest increment is used (flag 'capped'; the summary reports the achieved increment). Target: the planted increment of the reference arm (X7 ridge, PCA-64) equals
    frac x the registered group real yield (results/ext/eogreg_v2/summary_eogreg.json, L64 'all' gain - floor), frac in {0.25, 0.50}.
Controls re-run on every level (level 0 = no plant; same trials = the pe_eogreg trial set):
  all            reference read-out (X7 features; G2 = eogreg_v2 'all')
  eventreg       [1, blink, saccade] flags regressed out in-fold (G2 = eogreg_v2 'all_eventreg'); PRIMARY control of LOCK 7
  eogreg_cont    the 70 continuous ocular columns regressed out in-fold (G2 = eogreg_v2 'all_eogreg'); secondary (it also removes brain signal)
  quiet          b1 v3 primary rule (VEOG p2p <= median + 3 MAD), quiet subset vs 20 outcome x |PE|-decile stratified random subsets
                 (b1_eyequiet.block; target re-residualised within each subset as b1; G2 = eyequiet_v3 when no trial is lost)
Statistics (--summarize): per level and control, planted increment D_X = (Y_X(level) - floor) - (Y_X(0) - floor), participant
  means; removed share = 1 - mean(D_X) / mean(D_all), ratio of participant means with a GPU participant bootstrap CI (10,000).
  Eye-quiet: kept = (quiet - floor)/(random - floor) on the planted data (ratio of means, CI), and its increment D_quiet.
Pre-stated outcomes (IDEATOR I-3): (a) the eye-quiet test misses the sub-threshold plant (kept ~ 1) BY DESIGN; (b) the event-based
  regression removes >= 50 % of the planted increment. FALSIFIER: event regression removes < 50 % -> the eye sentence is limited to
  "supra-threshold eye movements". Read on the point estimate; the CI is printed.
Floors: circular shifts n/4, n/2, 3n/4 of the target (per arm, per level); smoke gate: |shift floor| <= 0.02 everywhere.
Usage: python i3_planted_ocular.py [--subjects 1,2] --out DIR | --summarize --out DIR
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/path/to/eegaudit"); os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "experiments/ext")); sys.path.insert(0, str(ROOT))
import d2_pe_tasks as D2
import pe_eogreg as EG
import b1_eyequiet as B1

VERSION = "v3b (signed first-crossing amplitude; max scanned increment when unreachable)"
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DEV = "cuda"; LS = (64, "full"); FRACS = (0.25, 0.50); NBIS = 16


def preica(s):
    """Pre-ICA epochs (n_ok, 65, 51) for 64 EEG (incl. CPz) + VEOG (last), ocular_v2 preprocessing; feedback-trial index; times."""
    import mne
    from cogmodels import models as M
    from cogmodels.data import load_subject
    root = Path("data_raw/ds003458"); d = load_subject(root, s, M.reward_schedule(480).numpy()); fb = d["fb_onset"][:d["T"]]
    raw = mne.io.read_raw_eeglab(root / f"sub-{s:03d}/eeg/sub-{s:03d}_task-ThreeArmedBandit_eeg.set", preload=True, verbose="error")
    eog = [c for c in raw.ch_names if "EOG" in c.upper()]; raw.set_channel_types({c: "eog" for c in eog}, verbose="error")
    if "CPz" not in raw.ch_names: mne.add_reference_channels(raw, "CPz", copy=False)
    eegn = [c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"]
    raw._data[raw.ch_names.index(eog[0])] -= raw._data[[raw.ch_names.index(c) for c in eegn]].mean(0)
    raw.set_eeg_reference("average", verbose="error"); raw.filter(0.1, 30.0, picks="all", verbose="error")
    ok = np.where(~np.isnan(fb))[0]; samp = np.round(fb[ok] * raw.info["sfreq"]).astype(int); ev = np.c_[samp, np.zeros_like(samp), np.ones_like(samp)]
    picks = eegn + [eog[0]]
    ep = mne.Epochs(raw, ev, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), picks=picks, preload=True, reject=None, decim=int(round(raw.info["sfreq"] / 50)), verbose="error")
    return ep.get_data(picks=picks) * 1e6, ok[ep.selection], eegn, ep.times


def rob(v):
    return v > np.median(v) + 3 * np.median(np.abs(v - np.median(v)))


def controls_inputs(V, Fp1, Fp2, F7, F8, times):
    """= pe_eogreg.ocular_v2 (O 70 columns, events) and b1 primary p2p (-0.2..0.6 s), from (possibly planted) waveforms."""
    w = (times >= 0.15) & (times <= 0.60); FP, FF = Fp1 - Fp2, F7 - F8
    blink = rob(np.ptp(V, 1)).astype(float); sacc = rob(np.ptp(FF, 1)).astype(float)
    O = np.c_[V[:, w], FP[:, w], FF[:, w], blink]; Ev = np.c_[blink, sacc]
    ptp_q = np.ptp(V[:, times <= 0.6 + 1e-9], 1)
    return O, Ev, ptp_q


def score_level(F, y, O, Ev, ptp_q, rew, sub, qt):
    """All controls on one (possibly planted) data set. Returns dict of arms with gain + shift_mean per width, and the quiet test."""
    n = len(y); Ys = np.column_stack([y] + [np.roll(y, round(n * q / 4)) for q in (1, 2, 3)]); R = {}
    for arm, tf in (("all", None), ("eventreg", EG.eogreg(Ev.astype(np.float64))), ("eogreg_cont", EG.eogreg(O.astype(np.float64)))):
        sc = D2.pe_gain(F, Ys, LS, transform=tf)
        R[arm] = {str(L): dict(gain=float(sc[L][:, 0].mean()), shift_mean=float(sc[L][:, 1:].mean())) for L in LS}
    med = float(np.median(ptp_q)); mad = float(np.median(np.abs(ptp_q - med))); qm = ptp_q <= med + 3 * mad; idx = np.where(qm)[0]; nq = len(idx)
    Q = dict(n_quiet=int(nq), quiet_fraction=nq / n, thr=med + 3 * mad, mad=mad)
    if nq >= B1.MIN_Q:
        ape = np.abs(y); dec = np.searchsorted(np.quantile(ape, np.linspace(0, 1, 11)[1:-1]), ape, side="right"); strat = np.unique(rew, return_inverse=True)[1] * 10 + dec
        Q["quiet"] = B1.block(F[idx], qt(idx), sub); rnd = []
        for k in range(B1.NRAND):                                            # = b1_eyequiet.strat_draw (v3), same seeds
            rg = np.random.default_rng(k); pick = [rg.choice(np.where(strat == st_)[0], int((strat[idx] == st_).sum()), replace=False) for st_ in np.unique(strat[idx])]
            r = np.sort(np.concatenate(pick)); rnd.append(B1.block(F[r], qt(r), sub))
        Q["random"] = {str(L): dict(gain_mean=float(np.mean([x[str(L)]["gain"] for x in rnd])), shift_mean=float(np.mean([x[str(L)]["shift_mean"] for x in rnd]))) for L in LS}
    else:
        Q["excluded"] = f"< {B1.MIN_Q} quiet trials"
    R["quiet_test"] = Q; return R, qm


def subject(s, out, target_all):
    t0 = time.time(); sub = f"sub-{s:03d}"; info = D2.dataset("ds003458")
    fz = np.load(ROOT / f"results/features/raw/{sub}.npz"); tri0 = fz["tri"]; F0 = fz["F"].astype(np.float64)
    chl = [str(c) for c in np.load(ROOT / f"results/eeg_measures/{sub}_epochs.npz")["ch"]]
    Xp, etri, eegn, times = preica(s); pos = {t: i for i, t in enumerate(etri)}; keep = np.array([t in pos for t in tri0]); ix = [pos[t] for t in tri0[keep]]
    tri = tri0[keep]; F = F0[keep]; n = len(tri); V = Xp[:, -1]; ci = {c: i for i, c in enumerate(eegn)}   # Xp / V: ALL feedback epochs (event rule medians as ocular_v2)
    y = info["qtil"](s, tri).astype(np.float64); qt = lambda r: info["qtil"](s, tri[r]).astype(np.float64)   # b1: target re-residualised within each subset
    rew = np.load(ROOT / "results/full_36318/behaviour_quantities_flow.npz")["reward"][s - 1][tri].astype(float)
    # blink topography (label-free): slope of each pre-ICA channel on the VEOG over all (trial, sample) points
    v = V.reshape(-1) - V.mean(); Xc = Xp[:, :-1].transpose(1, 0, 2).reshape(len(eegn), -1); Xc = Xc - Xc.mean(1, keepdims=True)
    b = (Xc @ v) / (v @ v); bt = {c: float(b[ci[c]]) for c in eegn}
    # waveform
    blink0 = rob(np.ptp(V, 1)); nbl = int(blink0.sum())
    if nbl >= 10: w = V[blink0].mean(0) - V[~blink0].mean(0); wsrc = "blink-trial mean minus other-trial mean"
    else: w = np.linalg.svd(V - V.mean(0), full_matrices=False)[2][0]; wsrc = "first temporal PC (< 10 blink trials)"
    w = w / np.ptp(w); w = w if w.max() >= -w.min() else -w
    win = (times >= 0.15) & (times <= 0.60); assert win.sum() * len(chl) == F.shape[1]
    bF = np.array([bt[c] for c in chl])                                       # features are channel-major (64 x 23), channel order of the epochs file
    ptp_base = np.ptp(V[ix][:, times <= 0.6 + 1e-9], 1); MAD = float(np.median(np.abs(ptp_base - np.median(ptp_base))))
    pick = {c: Xp[:, ci[c]] for c in ("Fp1", "Fp2", "F7", "F8")}

    def planted(a):
        Pk = a * MAD * y[:, None] * w[None]                                   # (n, 51) VEOG increment on the analysed rows
        P = np.zeros_like(V); P[ix] = Pk                                      # epochs without a feature row get no plant
        Fp = F + (bF[None, :, None] * Pk[:, None, win]).reshape(n, -1)
        return Fp, sel(controls_inputs(V + P, *(pick[c] + bt[c] * P for c in ("Fp1", "Fp2", "F7", "F8")), times))

    def sel(t): return tuple(x[ix] for x in t)

    def incr(a):                                                              # reference-arm increment, PCA-64, true target column only
        Fp = planted(a)[0]; return float(D2.pe_gain(Fp, y[:, None], (64,)).mean()) - base64

    base64 = float(D2.pe_gain(F, y[:, None], (64,)).mean()); cap = 3.0 / np.abs(y).max()
    res = dict(participant=sub, version=VERSION, code_sha256=CODE_SHA256, dataset="ds003458", n_trials=int(n), n_lost=int((~keep).sum()), waveform_source=wsrc, n_blink_trials=nbl,
               waveform=w.tolist(), waveform_energy_in_feature_window=float((w[win] ** 2).sum() / (w ** 2).sum()), veog_ptp_mad=MAD,
               topography={c: bt[c] for c in ("Fp1", "Fp2", "AF7", "AF8", "F7", "F8", "Fz", "Cz", "Pz", "CPz", "Oz")}, max_abs_y=float(np.abs(y).max()),
               amplitude_cap=float(cap), target_increment_bits=target_all, levels={})
    O0, Ev0, pq0 = sel(controls_inputs(V, pick["Fp1"], pick["Fp2"], pick["F7"], pick["F8"], times))
    R0, qm0 = score_level(F, y, O0, Ev0, pq0, rew, sub, qt); res["levels"]["0"] = dict(a=0.0, rows=R0, blink_rate=float(Ev0[:, 0].mean()), saccade_rate=float(Ev0[:, 1].mean()))
    def first_crossing(tgt, sg):
        """Smallest |a| (sign sg) whose reference-arm increment reaches tgt: geometric scan x1.5 from 1e-3 to 4 x cap, then NBIS-step
        log-bisection between the last point below and the first point above. Returns (|a| or None, scan)."""
        scan = []; prev = None
        for la in np.arange(-3.0, np.log10(cap * 4) + 1e-9, np.log10(1.5)):
            v = incr(sg * 10 ** la); scan.append((float(sg * 10 ** la), v))
            if v >= tgt:
                lo, hi = (prev if prev is not None else la - 1.0), la
                for _ in range(NBIS):
                    mid = 0.5 * (lo + hi); (lo, hi) = (mid, hi) if incr(sg * 10 ** mid) < tgt else (lo, mid)
                return 10 ** hi, scan
            prev = la
        return None, scan

    sign = None
    for fr in FRACS:
        # v3 (smokes sub-02): the increment is neither monotone in a nor sign-symmetric: (i) a plant whose direction is anti-aligned with
        # the participant's own EEG PE code first CANCELS yield; (ii) a near-deterministic plant makes the nested calibration
        # over-confident and the gain collapses. So the plant sign is the one with the smaller first-crossing amplitude (chosen at the
        # 25 % level, kept at 50 %), and the amplitude is that first crossing. No crossing below 4 x cap -> |a| = cap (flag 'capped').
        tgt = fr * target_all; scans = {}
        if sign is None:
            res_ = {sg: first_crossing(tgt, sg) for sg in (1.0, -1.0)}; scans = {str(int(k)): v[1] for k, v in res_.items()}
            ok = {sg: v[0] for sg, v in res_.items() if v[0] is not None}
            sign = min(ok, key=ok.get) if ok else max((1.0, -1.0), key=lambda sg: max(x[1] for x in res_[sg][1])); a_t = ok.get(sign)
        else:
            a_t, sc = first_crossing(tgt, sign); scans = {str(int(sign)): sc}
        note = "first-crossing scan + bisection"
        if a_t is None or a_t > cap:                                          # v3b: not reachable within the cap -> the scanned amplitude
            cand = [(v, x) for sc in scans.values() for x, v in sc if abs(x) <= cap and np.sign(x) == sign]   # with the LARGEST increment
            v_best, x_best = max(cand); note = f"target not reached within the cap (first crossing {'none' if a_t is None else f'{a_t:.3g}'}): amplitude at the max scanned increment {v_best:.4f}"
            a_t = cap * 4 if a_t is None else a_t; aa = abs(x_best)
        else:
            aa = a_t
        a = sign * aa; Fp, (O, Ev, pq) = planted(a); R, qm = score_level(Fp, y, O, Ev, pq, rew, sub, qt); scan = scans
        res["levels"][f"{fr:.2f}"] = dict(frac=fr, target_bits=tgt, a_target=float(a_t), a=float(a), capped=bool(a_t > cap), calib_note=note, calib_scan=scan,
                                           sign=float(sign), plant_ptp_max_in_mad=float(aa * np.abs(y).max()), sub_threshold=bool(aa * np.abs(y).max() <= 3.0 + 1e-9),
                                           quiet_flips=dict(quiet_to_eye=int((qm0 & ~qm).sum()), eye_to_quiet=int((~qm0 & qm).sum())),
                                           blink_rate=float(Ev[:, 0].mean()), saccade_rate=float(Ev[:, 1].mean()),
                                           blink_flag_pe_corr=float(np.corrcoef(Ev[:, 0], y)[0, 1]) if Ev[:, 0].std() > 0 else None, rows=R)
    # G2: level 0 vs the registered eogreg_v2 / eyequiet_v3 per-participant files
    g2 = {}
    eg = json.loads((ROOT / f"results/ext/eogreg_v2/eogreg_{sub}.json").read_text())["rows"]
    for mine, theirs in (("all", "all"), ("eventreg", "all_eventreg"), ("eogreg_cont", "all_eogreg")):
        g2[mine] = max(abs(R0[mine][str(L)]["gain"] - eg[theirs][str(L)]["gain"]) for L in LS)
    bq = json.loads((ROOT / f"results/ext/eyequiet_v3/b1_{sub}.json").read_text())["rules"]["primary_eog_mad"]
    if "quiet" in bq and "quiet" in R0["quiet_test"] and res["n_lost"] == 0:
        g2["quiet"] = max(abs(R0["quiet_test"]["quiet"][str(L)]["gain"] - bq["quiet"][str(L)]["gain"]) for L in LS)
        g2["random"] = max(abs(R0["quiet_test"]["random"][str(L)]["gain_mean"] - bq["random"][str(L)]["gain_mean"]) for L in LS)
        g2["n_quiet_b1"] = bq["n_quiet"]; g2["n_quiet_mine"] = R0["quiet_test"]["n_quiet"]
    g2["passed"] = bool(all(v < 1e-6 for k, v in g2.items() if k in ("all", "eventreg", "eogreg_cont", "quiet", "random")))
    res["g2"] = g2; res["runtime_s"] = round(time.time() - t0, 1)
    (out / f"i3_{sub}.json").write_text(json.dumps(res, indent=1))
    L = "64"; r0 = res["levels"]["0"]["rows"]
    msg = [f"{sub} n={n} wave={wsrc[:10]} cap={cap:.3f} G2={g2['passed']} ({ {k: (round(v, 9) if isinstance(v, float) else v) for k, v in g2.items()} })"]
    for k, lv in res["levels"].items():
        if k == "0": continue
        R = lv["rows"]; d = lambda arm: (R[arm][L]["gain"] - R[arm][L]["shift_mean"]) - (r0[arm][L]["gain"] - r0[arm][L]["shift_mean"])
        msg.append(f"  frac {k}: a={lv['a']:.4f} capped={lv['capped']} subthr={lv['sub_threshold']} flips={lv['quiet_flips']} D_all={d('all'):.4f} D_eventreg={d('eventreg'):.4f} D_cont={d('eogreg_cont'):.4f}"
                   + (f" quiet={R['quiet_test']['quiet'][L]['gain']:.4f} random={R['quiet_test']['random'][L]['gain_mean']:.4f}" if "quiet" in R["quiet_test"] else ""))
    print("\n".join(msg), f"{res['runtime_s']}s", flush=True)


def summarize(out):
    per = [json.loads(f.read_text()) for f in sorted(out.glob("i3_sub-*.json"))]; n = len(per)
    res = dict(code_sha256=CODE_SHA256, n=n, participants=[p["participant"] for p in per], code_sha256_per_participant=sorted({p["code_sha256"] for p in per}),
               g2_all_pass=bool(all(p["g2"]["passed"] for p in per)), target_increment_bits=per[0]["target_increment_bits"],
               waveform_sources={p["participant"]: p["waveform_source"] for p in per})
    g = torch.Generator(device=DEV); g.manual_seed(20261121); IX = torch.randint(0, n, (10000, n), generator=g, device=DEV)
    T = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float64, device=DEV)
    q = lambda x: torch.quantile(x, torch.tensor([.025, .975], dtype=torch.float64, device=DEV)).cpu().tolist()
    allfl = [abs(p["levels"][k]["rows"][arm][L]["shift_mean"]) for p in per for k in p["levels"] for arm in ("all", "eventreg", "eogreg_cont") for L in ("64", "full")]
    res["max_abs_shift_floor"] = float(max(allfl))
    for k in [k for k in per[0]["levels"] if k != "0"]:
        S = dict(n_capped=sum(p["levels"][k]["capped"] for p in per), n_sub_threshold=sum(p["levels"][k]["sub_threshold"] for p in per),
                 a_median=float(np.median([p["levels"][k]["a"] for p in per])), quiet_to_eye_flips_total=sum(p["levels"][k]["quiet_flips"]["quiet_to_eye"] for p in per),
                 blink_rate=float(np.mean([p["levels"][k]["blink_rate"] for p in per])), blink_rate_level0=float(np.mean([p["levels"]["0"]["blink_rate"] for p in per])))
        for L in ("64", "full"):
            dm = lambda p, lv, arm: p["levels"][lv]["rows"][arm][L]["gain"] - p["levels"][lv]["rows"][arm][L]["shift_mean"]
            D = {arm: T([dm(p, k, arm) - dm(p, "0", arm) for p in per]) for arm in ("all", "eventreg", "eogreg_cont")}
            R = dict(planted_increment_all=dict(mean=float(D["all"].mean()), ci=q(D["all"][IX].mean(1)), share_of_real_yield=float(D["all"].mean()) / res["target_increment_bits"]))
            for arm in ("eventreg", "eogreg_cont"):
                rm = 1 - D[arm][IX].mean(1) / D["all"][IX].mean(1)
                R[arm] = dict(increment=float(D[arm].mean()), increment_ci=q(D[arm][IX].mean(1)), removed_share=float(1 - D[arm].mean() / D["all"].mean()), removed_ci=q(rm))
            use = [p for p in per if "quiet" in p["levels"][k]["rows"]["quiet_test"] and "quiet" in p["levels"]["0"]["rows"]["quiet_test"]]
            if len(use) >= 3:
                IXu = IX[:, :len(use)] % len(use)
                qd = lambda p, lv: p["levels"][lv]["rows"]["quiet_test"]["quiet"][L]["gain"] - p["levels"][lv]["rows"]["quiet_test"]["quiet"][L]["shift_mean"]
                rd = lambda p, lv: p["levels"][lv]["rows"]["quiet_test"]["random"][L]["gain_mean"] - p["levels"][lv]["rows"]["quiet_test"]["random"][L]["shift_mean"]
                Qd, Rd = T([qd(p, k) for p in use]), T([rd(p, k) for p in use]); Dq = T([qd(p, k) - qd(p, "0") for p in use]); Da = T([dm(p, k, "all") - dm(p, "0", "all") for p in use])
                R["eye_quiet"] = dict(n_used=len(use), kept_on_planted=float(Qd.mean() / Rd.mean()), kept_ci=q(Qd[IXu].mean(1) / Rd[IXu].mean(1)),
                                      quiet_increment=float(Dq.mean()), quiet_increment_share_of_all=float(Dq.mean() / Da.mean()), share_ci=q(Dq[IXu].mean(1) / Da[IXu].mean(1)),
                                      falsifier_triggered_on_planted=bool(q(Qd[IXu].mean(1) / Rd[IXu].mean(1))[1] < 0.8))
            ev = R["eventreg"]
            R["reading"] = dict(rule="IDEATOR I-3 (pre-stated): (b) event regression must remove >= 50 % of the planted increment; falsifier = removes < 50 % (point estimate)",
                                falsifier_triggered=bool(ev["removed_share"] < 0.5),
                                sentence=("event-based regression does not cover sub-threshold ocular coding: limit the eye sentence to supra-threshold eye movements"
                                          if ev["removed_share"] < 0.5 else "event-based regression removes most of a sub-threshold ocular PE code"))
            S["L" + L] = R
        res[f"frac_{k}"] = S
    (out / "summary_i3.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1)[:6000])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default=""); ap.add_argument("--out", required=True); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize: return summarize(out)
    sm = json.loads((ROOT / "results/ext/eogreg_v2/summary_eogreg.json").read_text())["L64"]["all"]; target_all = float(sm["gain"] - sm["floor"])
    for s in ([int(x) for x in a.subjects.split(",")] if a.subjects else range(1, 24)):
        if not (out / f"i3_sub-{s:03d}.json").exists(): subject(s, out, target_all)


if __name__ == "__main__":
    main()
