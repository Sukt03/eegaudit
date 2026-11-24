"""NA-cell class-B collector (, 23 Nov 2026, NEW, numpy only): recomputes from SAVED per-participant files (no model is run) the table cells that exist but were printed as dashes:
MI noise-input floors for raw_eogreg / eegnet / eegnet_eogreg (mi_card_v3), EEGNet kept share (eegnet_eogreg / eegnet), FM kept share after proxy regression
(mi_fm_eogreg / mi_fm unregressed; CBraMod, LaBraM, EEGPT; REVE unregressed is incomplete). Participant bootstrap 10000, seed 0."""
import json, glob, numpy as np
from pathlib import Path
R = Path('/path/to/eegaudit/results/ext'); rng = np.random.default_rng(0)
def boot(v, B=10000):
    v = np.asarray(v, float); m = v[rng.integers(0, len(v), (B, len(v)))].mean(1); return dict(mean=float(v.mean()), ci95=[float(np.quantile(m, .025)), float(np.quantile(m, .975))], n=len(v), max_abs=float(np.abs(v).max()))
def ratio(a, b, B=10000):
    a, b = np.asarray(a, float), np.asarray(b, float); ix = rng.integers(0, len(a), (B, len(a))); r = a[ix].mean(1) / b[ix].mean(1)
    return dict(value=float(a.mean() / b.mean()), ci95=[float(np.quantile(r, .025)), float(np.quantile(r, .975))], n=len(a), num=float(a.mean()), den=float(b.mean()))
def load(d):
    o = {}
    for f in sorted(glob.glob(str(d / 'sub-*.json'))):
        try: o[int(Path(f).stem[4:])] = json.load(open(f))
        except Exception: pass          # empty / partial file (e.g. interrupted REVE run): skipped, counted via n
    return o
O = {'NOT_CITABLE': 'until independent review'}
M3 = load(R / 'mi_card_v3/primary_lpgo'); O['mi_card_v3_n'] = len(M3)
O['noise_floor'] = {k: boot([p[k]['noise']['bits'] for p in M3.values()]) for k in ('raw', 'raw_eogreg', 'eegnet', 'eegnet_eogreg') if all(k in p and 'noise' in p[k] for p in M3.values())}
O['shuffle_floor'] = {k: boot([p[k]['shuffle']['bits'] for p in M3.values()]) for k in ('raw', 'raw_eogreg') if all(k in p and 'shuffle' in p[k] for p in M3.values())}
O['yield'] = {k: boot([p[k]['eeg']['bits'] for p in M3.values()]) for k in ('raw', 'raw_eogreg', 'eegnet', 'eegnet_eogreg') if all(k in p for p in M3.values())}
O['kept_eegnet'] = ratio([p['eegnet_eogreg']['eeg']['bits'] for p in M3.values()], [p['eegnet']['eeg']['bits'] for p in M3.values()])
O['kept_raw'] = ratio([p['raw_eogreg']['eeg']['bits'] for p in M3.values()], [p['raw']['eeg']['bits'] for p in M3.values()])
O['kept_fm'] = {}
for m in ('cbramod', 'labram', 'eegpt', 'reve'):
    U, Rg = load(R / 'mi_fm' / m), load(R / 'mi_fm_eogreg' / m); ks = sorted(set(U) & set(Rg))
    ok = [k for k in ks if 'fm' in U[k] and 'fm' in Rg[k]]
    if len(ok) < 105: O['kept_fm'][m] = dict(status='INCOMPLETE unregressed run', n_unregressed=len(U), n_both=len(ok)); continue
    O['kept_fm'][m] = dict(regressed=boot([Rg[k]['fm']['eeg']['bits'] for k in ok]), unregressed=boot([U[k]['fm']['eeg']['bits'] for k in ok]),
                           kept=ratio([Rg[k]['fm']['eeg']['bits'] for k in ok], [U[k]['fm']['eeg']['bits'] for k in ok]),
                           note='unregressed run = ext/mi_fm (flag False, unreviewed); regressed = ext/mi_fm_eogreg (CITABLE-flag caveat)')
Path('/path/to/eegaudit/results/gpu_fill_20261123/na_B_cells').mkdir(parents=True, exist_ok=True)
Path('/path/to/eegaudit/results/gpu_fill_20261123/na_B_cells/summary.json').write_text(json.dumps(O, indent=1)); print(json.dumps(O, indent=1))
