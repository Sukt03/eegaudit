"""Single-trial feedback EEG measures for ds003458: FRN (FCz), P300 (Pz), frontal theta (FCz)."""
from pathlib import Path

import mne
import numpy as np
from mne.time_frequency import tfr_array_morlet

FRN_WIN, P3_WIN, TH_WIN = (0.25, 0.35), (0.30, 0.50), (0.20, 0.50)


def set_path(root, sub):
    return Path(root) / f"sub-{sub:03d}" / "eeg" / f"sub-{sub:03d}_task-ThreeArmedBandit_eeg.set"


def trial_measures(root, sub, fb_onset, reject_uv=150.0):
    """Return dict of (T,) arrays (NaN where no feedback or artefact) aligned to behaviour trials."""
    raw = mne.io.read_raw_eeglab(set_path(root, sub), preload=True, verbose="error")
    non_eeg = [c for c in raw.ch_names if c.upper() in ("EOG", "ECG", "EKG", "EMG", "GSR", "SCR")]
    raw.set_channel_types({c: "misc" for c in non_eeg}, verbose="error")
    raw.pick("eeg").set_eeg_reference("average", verbose="error")
    sf = raw.info["sfreq"]
    T = len(fb_onset)
    ok = np.where(~np.isnan(fb_onset))[0]
    samples = np.round(fb_onset[ok] * sf).astype(int)
    events = np.c_[samples, np.zeros_like(samples), np.ones_like(samples)]
    out = {k: np.full(T, np.nan) for k in ("frn", "p300", "theta")}

    erp = raw.copy().filter(0.1, 30, verbose="error")
    ep = mne.Epochs(erp, events, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), picks=["FCz", "Pz"],
                    preload=True, reject=None, verbose="error")
    X = ep.get_data() * 1e6  # (n, 2, times) microvolts
    times = ep.times
    keep = ep.selection  # indices into events that survived (edge epochs dropped)
    bad = np.ptp(X, axis=2).max(1) > reject_uv
    win = lambda w: (times >= w[0]) & (times <= w[1])
    frn, p3 = X[:, 0, win(FRN_WIN)].mean(1), X[:, 1, win(P3_WIN)].mean(1)

    th = raw.copy().filter(1, 40, verbose="error")
    ept = mne.Epochs(th, events, tmin=-0.6, tmax=1.0, baseline=None, picks=["FCz"], preload=True,
                     reject=None, verbose="error")
    pw = tfr_array_morlet(ept.get_data(), sf, freqs=np.arange(4, 9), n_cycles=3, output="power",
                          verbose="error")[:, 0].mean(1)  # (n, times)
    tt = ept.times
    lp = np.log10(pw + 1e-30)
    theta = lp[:, (tt >= TH_WIN[0]) & (tt <= TH_WIN[1])].mean(1) - lp[:, (tt >= -0.4) & (tt <= -0.1)].mean(1)
    tkeep = ept.selection

    idx = ok[keep]
    out["frn"][idx] = np.where(bad, np.nan, frn)
    out["p300"][idx] = np.where(bad, np.nan, p3)
    tmap = dict(zip(ok[tkeep], theta))
    badset = set(idx[bad])
    for i, v in tmap.items():
        if i not in badset:
            out["theta"][i] = v
    out["n_rejected"] = int(bad.sum())
    return out


def cv_r2(y, X, k=10):
    """Interleaved k-fold cross-validated R^2 of OLS."""
    n = len(y)
    folds = np.arange(n) % k
    pred = np.zeros(n)
    for f in range(k):
        tr, te = folds != f, folds == f
        b, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        pred[te] = X[te] @ b
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
