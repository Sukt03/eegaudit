"""E3a: single-trial feedback EEG measures for one ds003458 subject (CPU).

Pipeline: restore CPz online reference -> average reference -> ICA ocular correction (EOG-correlated
components) -> feedback-locked measures:
  FRN    FCz mean amplitude 240-340 ms (0.1-30 Hz, baseline -200..0 ms)
  P300   Pz  mean amplitude 300-500 ms (same ERP epochs)
  delta  FCz log power 1-3 Hz, 250-500 ms (Morlet; epochs -1.5..1.5 s), z-scored across trials
  theta  FCz log power 4-8 Hz, 200-500 ms, z-scored across trials
Rejection: peak-to-peak > 150 uV at FCz/Pz/Fz/Cz in the ERP epoch or at FCz in -1..1 s of the TF epoch.
Output: eeg_measures/sub-XXX.npz aligned to behavioural trials (NaN where missing/rejected).

Usage: python eeg_v2.py --sub 1
"""
import argparse
from pathlib import Path

import mne
import numpy as np
from mne.preprocessing import ICA
from mne.time_frequency import tfr_array_morlet

from cogmodels import models as M
from cogmodels.data import load_subject

ROOT = Path("data_raw/ds003458")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, required=True)
    ap.add_argument("--out", default="results/eeg_measures")
    ap.add_argument("--reject_uv", type=float, default=150.0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    probs = M.reward_schedule(480).numpy()
    d = load_subject(ROOT, a.sub, probs)
    T = d["T"]
    fb = d["fb_onset"][:T]

    raw = mne.io.read_raw_eeglab(ROOT / f"sub-{a.sub:03d}/eeg/sub-{a.sub:03d}_task-ThreeArmedBandit_eeg.set",
                                 preload=True, verbose="error")
    types = {c: ("eog" if "EOG" in c.upper() else "misc") for c in raw.ch_names
             if any(k in c.upper() for k in ("EOG", "ECG", "EKG", "EMG", "GSR"))}
    std = set(mne.channels.make_standard_montage("standard_1005").ch_names) | {"CPz"}
    for c in raw.ch_names:                                   # any channel without a scalp position (e.g. SCR skin
        if c not in std and c not in types:                  # conductance) is not EEG: exclude from reference/ICA/rejection
            types[c] = "misc"
    raw.set_channel_types(types, verbose="error")
    if "CPz" not in raw.ch_names:
        mne.add_reference_channels(raw, "CPz", copy=False)
    raw.set_montage("standard_1005", on_missing="ignore", verbose="error")
    sf = raw.info["sfreq"]
    # bad channels (audit B3): robust z > 4 of log variance during the task (1-40 Hz), interpolated before referencing
    ok_fb = fb[~np.isnan(fb)]
    task = raw.copy().crop(max(ok_fb.min() - 2, 0), min(ok_fb.max() + 2, raw.times[-1])).filter(1.0, 40.0, picks="eeg", verbose="error")
    eeg_names = [c for c, t in zip(task.ch_names, task.get_channel_types()) if t == "eeg" and c != "CPz"]
    lv = np.log(task.get_data(picks=eeg_names).var(1))
    rz = (lv - np.median(lv)) / (1.4826 * np.median(np.abs(lv - np.median(lv))) + 1e-12)
    raw.info["bads"] = [eeg_names[k] for k in np.where(np.abs(rz) > 4)[0]]
    n_bad = len(raw.info["bads"])
    if n_bad:
        raw.interpolate_bads(reset_bads=True, verbose="error")
    raw.set_eeg_reference("average", verbose="error")

    # ICA ocular correction
    n_ica = 0
    if "eog" in raw.get_channel_types():
        rf = raw.copy().crop(max(ok_fb.min() - 2, 0), min(ok_fb.max() + 2, raw.times[-1])).filter(1.0, 40.0, verbose="error")
        ica = ICA(n_components=20, method="fastica", random_state=0, max_iter="auto", verbose="error")
        ica.fit(rf, picks="eeg", decim=4, reject=dict(eeg=500e-6), verbose="error")   # audit B2: task data only
        eogs = [c for c in rf.ch_names if c.upper().startswith("EOG")] + [c for c in ("Fp1", "Fp2", "F7", "F8") if c in rf.ch_names]
        bad, _ = ica.find_bads_eog(rf, ch_name=eogs, verbose="error")                      # vertical + horizontal proxies
        ica.exclude = bad[:3]
        n_ica = len(ica.exclude)
        ica.apply(raw, verbose="error")
    raw.pick(["eeg"])

    ok = np.where(~np.isnan(fb))[0]
    samp = np.round(fb[ok] * sf).astype(int)
    events = np.c_[samp, np.zeros_like(samp), np.ones_like(samp)]
    res = {k: np.full(T, np.nan) for k in ("frn", "p300", "delta", "theta")}

    erp = raw.copy().filter(0.1, 30.0, verbose="error")
    pick = [c for c in ("FCz", "Pz", "Fz", "Cz") if c in erp.ch_names]
    ep = mne.Epochs(erp, events, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), picks=pick, preload=True,
                    reject=None, verbose="error")
    X = ep.get_data() * 1e6
    tm = ep.times
    bad_erp = np.ptp(X, axis=2).max(1) > a.reject_uv
    w = lambda lo, hi: (tm >= lo) & (tm <= hi)
    idx = ok[ep.selection]
    res["frn"][idx] = np.where(bad_erp, np.nan, X[:, pick.index("FCz"), w(0.24, 0.34)].mean(1))
    res["p300"][idx] = np.where(bad_erp, np.nan, X[:, pick.index("Pz"), w(0.30, 0.50)].mean(1))
    # all-channel epochs at 50 Hz for decoding. Audit fixes: (i) 20 Hz low-pass before decimating to 50 Hz (no aliasing);
    # (ii) baseline = 200 ms before STIMULUS onset of the same trial, so the pre-feedback window (which follows the choice
    # and carries the expectation) is not subtracted from the feedback response and the pre-feedback control is valid.
    erp20 = raw.copy().filter(0.1, 20.0, verbose="error")
    epa = mne.Epochs(erp20, events, tmin=-0.2, tmax=0.8, baseline=None, picks="eeg", preload=True,
                     reject=None, decim=int(round(sf / 50)), verbose="error")
    A = (epa.get_data() * 1e6).astype(np.float32)
    so = d["stim_onset"][:T][ok]
    ssamp = np.round(so * sf).astype(int)
    sev = np.c_[ssamp, np.zeros_like(ssamp), np.ones_like(ssamp)]
    epb = mne.Epochs(erp20, sev, tmin=-0.2, tmax=0.0, baseline=None, picks="eeg", preload=True, reject=None,
                     verbose="error")
    base = {ok[k]: v for k, v in zip(epb.selection, epb.get_data().mean(2) * 1e6)}
    trl = ok[epa.selection]
    B = np.stack([base.get(t, np.full(A.shape[1], np.nan)) for t in trl]).astype(np.float32)
    has_base = np.isfinite(B).all(1)
    A = A - np.nan_to_num(B)[:, :, None]
    bad_a = (np.ptp(A, axis=2).max(1) > 2 * a.reject_uv) | ~has_base
    # true band power (audit B6): Morlet on long 500 Hz epochs, log power averaged over 150-600 ms after feedback
    epl = mne.Epochs(raw, events, tmin=-1.0, tmax=1.5, baseline=None, picks="eeg", preload=True, reject=None, verbose="error")
    Dl = epl.get_data()
    tl = epl.times; wl = (tl >= 0.15) & (tl <= 0.60)
    bands = ((np.array([1.5, 2.0, 3.0]), 2.0), (np.arange(4.0, 8.1, 1.0), 3.0), (np.arange(8.0, 13.1, 1.0), 5.0), (np.arange(14.0, 25.1, 2.0), 5.0))
    BP = np.stack([np.log10(tfr_array_morlet(Dl, sf, freqs=f, n_cycles=nc, output="power", decim=5, verbose="error")
                            [..., wl[::5]].mean((2, 3)) + 1e-30) for f, nc in bands], 2)       # (n, ch, 4)
    bp_map = {ok[k]: v for k, v in zip(epl.selection, BP)}
    BPa = np.stack([bp_map.get(t, np.full(BP.shape[1:], np.nan)) for t in trl]).astype(np.float32)
    bad_a = bad_a | ~np.isfinite(BPa).all((1, 2))
    np.savez_compressed(out / f"sub-{a.sub:03d}_epochs.npz", X=A, BP=np.nan_to_num(BPa), trial=ok[epa.selection],
                        bad=bad_a | np.isin(ok[epa.selection], idx[bad_erp]),
                        times=epa.times.astype(np.float32), ch=np.array(epa.ch_names), n_bad_channels=n_bad)

    ept = mne.Epochs(raw, events, tmin=-1.5, tmax=1.5, baseline=None, picks=["FCz"], preload=True,
                     reject=None, verbose="error")
    D = ept.get_data()
    tt = ept.times
    bad_tf = np.ptp(D[:, 0, (tt >= -1.0) & (tt <= 1.0)] * 1e6, axis=1) > a.reject_uv
    tidx = ok[ept.selection]
    for name, freqs, ncyc, win in (("delta", np.array([1.0, 1.5, 2.0, 2.5, 3.0]), 1.5, (0.25, 0.50)),
                                   ("theta", np.arange(4.0, 9.0), 3.0, (0.20, 0.50))):
        pw = tfr_array_morlet(D, sf, freqs=freqs, n_cycles=ncyc, output="power", verbose="error")[:, 0].mean(1)
        v = np.log10(pw[:, (tt >= win[0]) & (tt <= win[1])].mean(1) + 1e-30)
        v = np.where(bad_tf, np.nan, v)
        v = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)
        res[name][tidx] = v
    rej = np.zeros(T, bool)
    rej[idx[bad_erp]] = True
    np.savez(out / f"sub-{a.sub:03d}.npz", **res, n_ica_removed=n_ica, n_rejected=int(rej.sum()),
             n_feedback=len(ok), T=T)
    print(f"sub-{a.sub:03d}: T={T} feedback={len(ok)} ica_removed={n_ica} rejected_erp={int(bad_erp.sum())} "
          f"rejected_tf={int(bad_tf.sum())}", flush=True)


if __name__ == "__main__":
    main()
