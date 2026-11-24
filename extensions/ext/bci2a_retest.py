"""BCI-IV-2a retest + EOG check (, 23 Nov 2026).
Data: data_raw/bci_iv_2a/A0{1..9}{T,E}.gdf (22 EEG + 3 EOG, 250 Hz, two sessions on different days) and true labels for E (data_raw/bci_iv_2a/true_labels/A0xE.mat).
Question 1 (MI D4 retest): does the log-score gain of a read-out fitted on one session reproduce when fitted on the other (T->E vs E->T), per participant, n=9.
Question 2 (MI D6 with recorded EOG): yield of 22-ch EEG vs EEG with the 3 recorded EOG channels regressed out vs EOG alone.
Read-out: log-variance of 4 filter-bank bands (4-8,8-12,12-20,20-30 Hz) x 22 ch in cue+0.5..cue+2.5 s, multinomial logistic regression (C=0.01 after smoke showed C=1 overconfident, negative noise-floor bits; standardised on train),
yield = held-out mean log2 p(true class) - log2 prior (train class frequencies) in bits/trial. Floors: noise input (N(0,1) features), shuffled train labels.
Arms: eeg | eeg_eogreg (per-channel least squares on the 3 EOG time series, fitted on TRAIN trials only, applied to test) | eog_only (3 EOG ch features).
Usage: python bci2a_retest.py --out DIR [--subs 1,2,..]"""
import argparse, json, glob, os, sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.io import loadmat
import mne
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT = Path('/path/to/eegaudit'); D = ROOT / 'data_raw/bci_iv_2a'
BANDS = [(4, 8), (8, 12), (12, 20), (20, 30)]; FS = 250

def load(sub, ses):
    raw = mne.io.read_raw_gdf(D / f'A{sub:02d}{ses}.gdf', eog=['EOG-left', 'EOG-central', 'EOG-right'], preload=True, verbose='error'); X = raw.get_data() * 1e6
    names = raw.ch_names; ev, evid = mne.events_from_annotations(raw, verbose='error')
    inv = {v: k for k, v in evid.items()}; cue = [i for i in range(len(ev)) if inv[ev[i, 2]] in ('769', '770', '771', '772', '783')]
    onset = ev[cue, 0]; code = np.array([inv[ev[i, 2]] for i in cue])
    if ses == 'T': y = np.array([int(c) - 769 for c in code])
    else: y = loadmat(D / f'true_labels/A{sub:02d}E.mat')['classlabel'].ravel().astype(int) - 1; assert len(y) == len(onset), (len(y), len(onset))
    ok = np.ones(len(onset), bool)
    art = [i for i in range(len(ev)) if inv[ev[i, 2]] == '1023']   # rejected-trial flag in the competition files
    for a in art:
        j = np.searchsorted(onset, ev[a, 0])   # flag 1023 sits at trial start (768), ~2 s BEFORE the cue of the same trial: first cue at/after it
        if j < len(ok) and onset[j] - ev[a, 0] < 3 * FS: ok[j] = False
    eeg = X[:22]; eog = X[22:25]; assert len(names) == 25
    segs = [(eeg[:, o + int(.5 * FS):o + int(2.5 * FS)], eog[:, o + int(.5 * FS):o + int(2.5 * FS)]) for o in onset]
    return np.stack([s[0] for s in segs]), np.stack([s[1] for s in segs]), y, ok

def feats(E):  # E (n, ch, t) -> (n, ch*len(BANDS)) log-variance
    out = []
    for lo, hi in BANDS:
        sos = butter(4, [lo, hi], btype='band', fs=FS, output='sos'); F = sosfiltfilt(sos, E, axis=-1); out.append(np.log(F.var(-1) + 1e-6))
    return np.concatenate(out, 1)

def eogreg(Etr, Gtr, Ete, Gte):  # per-trial EEG minus EOG-explained part, coefficients from train trials only
    A = np.concatenate([g for g in Gtr], 1).T; B = np.concatenate([e for e in Etr], 1).T
    W = np.linalg.lstsq(A, B, rcond=None)[0]
    return np.stack([e - (W.T @ g) for e, g in zip(Ete, Gte)]), np.stack([e - (W.T @ g) for e, g in zip(Etr, Gtr)])

def yield_bits(Ftr, ytr, Fte, yte, seed=0, noise=False, shuffle=False):
    rng = np.random.default_rng(seed)
    if noise: Ftr, Fte = rng.standard_normal(Ftr.shape), rng.standard_normal(Fte.shape)
    if shuffle: ytr = rng.permutation(ytr)
    sc = StandardScaler().fit(Ftr); clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(Ftr), ytr)
    P = np.full((len(Fte), 4), 1e-6); P[:, clf.classes_] = clf.predict_proba(sc.transform(Fte)); lp = np.log2(P[np.arange(len(yte)), yte])
    prior = np.bincount(ytr, minlength=4) / len(ytr); return float(lp.mean() - np.log2(prior[yte]).mean())

def arm_scores(tr, te):
    (Etr, Gtr, ytr), (Ete, Gte, yte) = tr, te; r = {}
    Rte, Rtr = eogreg(Etr, Gtr, Ete, Gte)
    for arm, (A, B) in dict(eeg=(Etr, Ete), eeg_eogreg=(Rtr, Rte), eog_only=(Gtr, Gte)).items():
        Fa, Fb = feats(A), feats(B); r[arm] = yield_bits(Fa, ytr, Fb, yte)
        if arm == 'eeg': r['noise'] = float(np.mean([yield_bits(Fa, ytr, Fb, yte, s, noise=True) for s in range(3)])); r['shuffled'] = float(np.mean([yield_bits(Fa, ytr, Fb, yte, s, shuffle=True) for s in range(3)]))
    return r

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--subs', default='1,2,3,4,5,6,7,8,9'); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for s in [int(x) for x in a.subs.split(',')]:
        T = load(s, 'T'); E = load(s, 'E'); sel = lambda d: (d[0][d[3]], d[1][d[3]], d[2][d[3]])
        t, e = sel(T), sel(E); res = dict(sub=s, n_T=len(t[2]), n_E=len(e[2]), T_to_E=arm_scores(t, e), E_to_T=arm_scores(e, t))
        json.dump(res, open(out / f'A{s:02d}.json', 'w'), indent=1); print(s, {k: round(v, 3) for k, v in res['T_to_E'].items()}, {k: round(v, 3) for k, v in res['E_to_T'].items()}, flush=True)
if __name__ == '__main__': main()
