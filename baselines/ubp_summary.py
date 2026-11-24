"""Aggregate the UBP card (results/baselines/ubp/ubp_card_full_{a,b}.json + ubp_card_retrain_subN.json) into one summary with participant
means and participant-bootstrap CI95 (10,000 resamples, one GPU tensor op). Card rows = released checkpoint (G2) and our CLEAN retrains
(eeg, noise; seeds averaged within participant); eeg_quirk = main.py reproduction check only. Writes results/baselines/ubp/ubp_summary.json."""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()  # every result JSON carries the exact code hash
import argparse, glob, json, os
import numpy as np, torch
R = '/path/to/eegaudit/results/baselines/ubp'; dev = torch.device('cuda')
ap = argparse.ArgumentParser(); ap.add_argument('--pattern', default=''); A = ap.parse_args()
FULL = f'{R}/ubp_card_soft12_sub*.json' if A.pattern == 'soft12' else f'{R}/ubp_card_full_*.json'
RET = f'{R}/ubp_card_soft12_sub*.json' if A.pattern == 'soft12' else f'{R}/ubp_card_retrain_sub*.json'
rel, rnd, ret = {}, {}, {}
for f in sorted(glob.glob(FULL)):
    d = json.load(open(f))
    for s, v in d['subjects'].items():
        rel[int(s)] = v['released']; rnd[int(s)] = v['random_init_ridge']
for f in sorted(glob.glob(RET)):
    for k, v in json.load(open(f))['retrain'].items():
        sub, arm, seed = k.split('/'); ret.setdefault(arm, {}).setdefault(int(sub[4:]), []).append(v)


def boot(x):
    x = torch.tensor(x, dtype=torch.float64, device=dev); n = len(x)
    if n < 2:
        return None
    idx = torch.randint(0, n, (10000, n), device=dev, generator=torch.Generator(device=dev).manual_seed(0))
    m = x[idx].mean(1); return [float(torch.quantile(m, 0.025)), float(torch.quantile(m, 0.975))]


def summ(rows, keys):
    subs = sorted(rows)
    return {'n_participants': len(subs), 'participants': subs,
            **{k: {'per_participant': [rows[s][k] for s in subs], 'mean': float(np.mean([rows[s][k] for s in subs])), 'ci95': boot([rows[s][k] for s in subs])}
               for k in keys if all(rows[s].get(k) is not None for s in subs)}}


K = ['k80_top1', 'k80_bits', 'k40_bits', 'k40_top1', 'ceiling_k40_bits', 'fraction_k40', 'k80_floor_bits', 'noise_input_floor_bits', 'k80_bits_t40', 'k40_bits_t40', 'ceiling_k40_bits_t40', 'fraction_k40_t40']
out = {'note': __doc__, 'released': summ(rel, K + ['G2_abs_diff_pt']), 'random_init_ridge': summ(rnd, K)}
out['released']['G2_pass_all'] = all(rel[s]['G2_pass'] for s in rel)
out['released']['published_top1_mean_json'] = float(np.mean([rel[s]['released_test_top1_json'] for s in rel]))
for arm, per in ret.items():
    rows = {s: {k: float(np.mean([v[k] for v in vs if v.get(k) is not None])) if any(v.get(k) is not None for v in vs) else None for k in K} for s, vs in per.items()}
    out[f'retrain_{arm}'] = summ(rows, K); out[f'retrain_{arm}']['n_seeds'] = {s: len(vs) for s, vs in per.items()}
if 'eeg' in ret and 'noise' in ret:
    common = sorted(set(ret['eeg']) & set(ret['noise']))
    d = [np.mean([v['k80_bits'] for v in ret['eeg'][s]]) - np.mean([v['k80_bits'] for v in ret['noise'][s]]) for s in common]
    out['retrain_eeg_minus_noise_k80_bits'] = {'per_participant': d, 'mean': float(np.mean(d)), 'ci95': boot(d)}
common = sorted(set(rel) & set(rnd))
out['released_minus_random_init_k80_bits'] = {'mean': float(np.mean([rel[s]['k80_bits'] - rnd[s]['k80_bits'] for s in common])),
                                              'ci95': boot([rel[s]['k80_bits'] - rnd[s]['k80_bits'] for s in common])}
out['random_init_share_of_released_k80'] = float(np.mean([rnd[s]['k80_bits'] for s in common]) / np.mean([rel[s]['k80_bits'] for s in common]))
out['code_sha256'] = CODE_SHA256; json.dump(out, open(f'{R}/ubp_summary' + ('_soft12' if A.pattern == 'soft12' else '') + '.json', 'w'), indent=1)
for k in ('released', 'random_init_ridge') + tuple(f'retrain_{a}' for a in ret):
    v = out[k]; print(k, v['n_participants'], {m: (round(v[m]['mean'], 3), [round(c, 3) for c in (v[m]['ci95'] or [])]) for m in K if m in v})
print('G2 all', out['released']['G2_pass_all'], 'eeg-noise', out.get('retrain_eeg_minus_noise_k80_bits', {}).get('mean'), 'rand share', round(out['random_init_share_of_released_k80'], 3))
