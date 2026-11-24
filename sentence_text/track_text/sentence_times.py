"""Reading-time constants from official ZuCo 1.0 Matlab files (MATLAB v7, scipy.io.loadmat).
Per sentence: raw EEG duration (samples/500 Hz) = time the sentence was on screen; per word: gaze duration GD (ms).
Writes JSON with mean_GD_s, mean_sentence_duration_s and per-file stats. Run with python -I in the compute node."""
import sys, json, numpy as np
from scipy.io import loadmat
out = sys.argv[1]; files = sys.argv[2:]
res = {'files': {}}; all_dur, all_gd = [], []
for src in files:
    m = loadmat(src, squeeze_me=True, struct_as_record=False)
    sd = m['sentenceData']; durs, gds = [], []
    for s in np.atleast_1d(sd):
        raw = getattr(s, 'rawData', None)
        if raw is not None and np.ndim(raw) == 2 and max(raw.shape) > 10:
            durs.append(max(raw.shape) / 500.0)
        words = getattr(s, 'word', None)
        if words is None or np.ndim(words) == 0 and not hasattr(words, 'GD'):
            continue
        for w in np.atleast_1d(words):
            gd = getattr(w, 'GD', None)
            if gd is None:
                continue
            gd = np.asarray(gd, dtype=float).ravel()
            if gd.size and np.isfinite(gd[0]) and gd[0] > 0:
                gds.append(float(gd[0]))
    res['files'][src] = {'n_sentences': int(np.size(sd)), 'n_with_eeg': len(durs), 'mean_duration_s': float(np.mean(durs)) if durs else None,
                         'median_duration_s': float(np.median(durs)) if durs else None, 'n_gd': len(gds),
                         'mean_GD_ms': float(np.mean(gds)) if gds else None, 'median_GD_ms': float(np.median(gds)) if gds else None}
    all_dur += durs; all_gd += gds
    print(src, res['files'][src], flush=True)
res['mean_sentence_duration_s'] = float(np.mean(all_dur)) if all_dur else None
res['mean_GD_s'] = float(np.mean(all_gd)) / 1000.0 if all_gd else None
res['note'] = 'sentence duration = raw EEG samples/500 Hz while the sentence was on screen; GD = gaze duration per fixated word (ms); from the official ZuCo 1.0 .mat files listed'
json.dump(res, open(out, 'w'), indent=1); print(res['mean_sentence_duration_s'], res['mean_GD_s'])
