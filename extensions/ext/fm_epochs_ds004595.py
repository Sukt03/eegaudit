"""200-Hz feedback epochs for ds004595 (the second PE dataset), for the frozen M3 PE grid (needs 13-45 Hz; the existing 50-Hz
results/eeg_measures_ds004595 epochs cannot carry it). EXECUTOR-2, 21 Nov 2026. Preprocessing only (data loading; PLAN §10).
Cleaning = eeg_ds004595.py exactly (channel typing, bad-channel interpolation by robust log-variance z > 4, average reference, ICA with
<= 3 EOG-correlated components removed); then, as legacy/phase0_belief_scripts/fm_epochs.py for ds003458: 60-Hz notch, 0.3-75 Hz,
epochs -1..1 s around feedback (responded trials), resample 200 Hz, uV / 100, float16 (n, ch, 2, 200), trial index saved.
Output results/fm_epochs_ds004595/sub-XXX.npz. Usage: python fm_epochs_ds004595.py 1 2 3 ..."""
import sys
from pathlib import Path
import mne
import numpy as np
from mne.preprocessing import ICA
sys.path.insert(0, "/path/to/eegaudit")
from ds004595_task import ROOT, load
OUT = Path("/path/to/eegaudit/results/fm_epochs_ds004595"); OUT.mkdir(parents=True, exist_ok=True)


def one(sub):
    if (OUT / f"sub-{sub:03d}.npz").exists(): return
    d = load(sub); fb, so_all, miss = d["fb_onset"], d["stim_onset"], d["missing"]
    raw = mne.io.read_raw_eeglab(ROOT / f"sub-{sub:03d}/eeg/sub-{sub:03d}_task-ThreeArmedBandit_eeg.set", preload=True, verbose="error")
    std = set(mne.channels.make_standard_montage("standard_1005").ch_names) | {"CPz"}; types = {}
    for c in raw.ch_names:
        if c in ("Above Eye", "Below Eye") or "EOG" in c.upper(): types[c] = "eog"
        elif c not in std: types[c] = "misc"
    raw.set_channel_types(types, verbose="error")
    if "CPz" not in raw.ch_names: mne.add_reference_channels(raw, "CPz", copy=False)
    raw.set_montage("standard_1005", on_missing="ignore", verbose="error")
    ok = np.where(~miss & np.isfinite(fb) & np.isfinite(so_all))[0]
    lo, hi = max(fb[ok].min() - 2, 0), min(fb[ok].max() + 2, raw.times[-1])
    task = raw.copy().crop(lo, hi).filter(1.0, 40.0, picks="eeg", verbose="error")
    eeg_names = [c for c, t in zip(task.ch_names, task.get_channel_types()) if t == "eeg" and c != "CPz"]
    lv = np.log(task.get_data(picks=eeg_names).var(1)); rz = (lv - np.median(lv)) / (1.4826 * np.median(np.abs(lv - np.median(lv))) + 1e-12)
    raw.info["bads"] = [eeg_names[k] for k in np.where(np.abs(rz) > 4)[0]]
    if raw.info["bads"]: raw.interpolate_bads(reset_bads=True, verbose="error")
    raw.set_eeg_reference("average", verbose="error")
    if "eog" in raw.get_channel_types():
        rf = raw.copy().crop(lo, hi).filter(1.0, 40.0, verbose="error")
        ica = ICA(n_components=20, method="fastica", random_state=0, max_iter="auto", verbose="error"); ica.fit(rf, picks="eeg", decim=4, reject=dict(eeg=500e-6), verbose="error")
        eogs = [c for c, t in zip(rf.ch_names, rf.get_channel_types()) if t == "eog"] + [c for c in ("Fp1", "Fp2", "F7", "F8") if c in rf.ch_names]
        bad, _ = ica.find_bads_eog(rf, ch_name=eogs, verbose="error"); ica.exclude = bad[:3]; ica.apply(raw, verbose="error")
    raw.pick(["eeg"]); raw.notch_filter(60.0, verbose="error"); raw.filter(0.3, 75.0, verbose="error")
    sf = raw.info["sfreq"]; samp = np.round(fb[ok] * sf).astype(int); ev = np.c_[samp, np.zeros_like(samp), np.ones_like(samp)]
    ep = mne.Epochs(raw, ev, tmin=-1.0, tmax=1.0 - 1.0 / sf, baseline=None, preload=True, reject=None, verbose="error"); ep.resample(200.0, verbose="error")
    X = ep.get_data(units="uV")[:, :, :400]; X = X.reshape(len(X), X.shape[1], 2, 200) / 100.0
    np.savez_compressed(OUT / f"sub-{sub:03d}.npz", X=X.astype(np.float16), trial=ok[ep.selection], ch=np.array(ep.ch_names))
    print(f"sub-{sub:03d}", X.shape, flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        try: one(int(s))
        except Exception as e: print("FAIL", s, repr(e)[:200], flush=True)
