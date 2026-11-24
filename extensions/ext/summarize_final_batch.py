"""Summaries (, 23 Nov 2026) for pe_width, bci2a_retest, text_d5_width written next to the results so the paper reads tracked leaves.
Reviewed (PASS WITH CONDITIONS; bci2a after fixing the artefact-flag bug). Participant bootstrap 10000, seed 0."""
import json, glob, numpy as np
R = '/path/to/eegaudit/results/ext/'
rng = np.random.default_rng(0)
def ci(v):
    v = np.asarray(v, float); m = v[rng.integers(0, len(v), (10000, len(v)))].mean(1); return dict(mean=float(v.mean()), ci95=[float(np.quantile(m, .025)), float(np.quantile(m, .975))], n=len(v))
P = [json.load(open(f))['rows'] for f in sorted(glob.glob(R + 'pe_width/width_sub-*.json'))]
Ls = list(P[0]); S = {'n': len(P), 'gain_by_width': {L: ci([p[L]['gain'] for p in P]) for L in Ls}}
mc = {L: S['gain_by_width'][L]['mean'] for L in Ls}; S['L90_mean_curve'] = next(L for L in Ls if mc[L] >= .9 * mc['full'])
S['share_at_64_of_full'] = mc['64'] / mc['full']; json.dump(S, open(R + 'pe_width/summary.json', 'w'), indent=1)
B = [json.load(open(f)) for f in sorted(glob.glob(R + 'bci2a_retest/A*.json'))]
S = {'n': len(B), 'participants': [b['sub'] for b in B], 'note': 'A01 excluded: A01E.gdf corrupt'}
for k in ('eeg', 'eeg_eogreg', 'eog_only', 'noise', 'shuffled'):
    a = [b['T_to_E'][k] for b in B]; c = [b['E_to_T'][k] for b in B]; S[k] = dict(T_to_E=ci(a), E_to_T=ci(c), corr=float(np.corrcoef(a, c)[0, 1]))
json.dump(S, open(R + 'bci2a_retest/summary.json', 'w'), indent=1)
T = {}
for f in sorted(glob.glob(R + 'text_d5_width/*_results.json')):
    r = json.load(open(f)); k = int(r['tag'].split('_k')[1].split('_')[0]); i = r['identification']
    T.setdefault(str(k), dict(beam=[], greedy=[], tf=[]))
    T[str(k)]['beam'].append(i['beam']['bits']); T[str(k)]['greedy'].append(i['greedy']['bits']); T[str(k)]['tf'].append(i['teacher_forced']['bits'])
S = {'by_keep': {k: {m: dict(mean=float(np.mean(v)), per_seed=[float(x) for x in v]) for m, v in d.items()} for k, d in T.items()}, 'free_running_max': float(max(max(d['beam'] + d['greedy']) for d in T.values()))}
json.dump(S, open(R + 'text_d5_width/summary.json', 'w'), indent=1); print('ok', S['free_running_max'])
