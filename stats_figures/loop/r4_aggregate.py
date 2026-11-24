"""Aggregate R4 contrastive runs over seeds: per input (eeg/noise/shuffled) mean +- sd over seeds of top-1, 2-way, bits, pooled and within task;
EEG minus noise per seed (paired by seed) with mean and seed range; query-paired bootstrap CI of EEG minus noise for each seed."""
import json, glob, numpy as np
R = '/path/to/eegaudit/results/loop/r4_contrastive'
runs = {}
for f in sorted(glob.glob(f'{R}/full_*_seed*_results.json')):
    d = json.load(open(f)); runs.setdefault(d['input'], {})[d['cfg']['seed']] = d
scopes = ['pooled'] + sorted(next(iter(runs['eeg'].values()))['within_task'])
get = lambda d, sc: d['pooled'] if sc == 'pooled' else d['within_task'][sc]
out = {'seeds': {k: sorted(v) for k, v in runs.items()}, 'by_input': {}, 'eeg_minus_noise': {}}
for inp, S in runs.items():
    out['by_input'][inp] = {}
    for sc in scopes:
        vals = {m: [get(S[s], sc)[m] for s in sorted(S)] for m in ('top1', 'two_way', 'bits', 'shuffled_pairs_two_way', 'shuffled_pairs_top1')}
        out['by_input'][inp][sc] = {m: {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else None, 'per_seed': v} for m, v in vals.items()}
        out['by_input'][inp][sc]['n_candidates'] = get(S[sorted(S)[0]], sc)['n_candidates']
rng = np.random.default_rng(0)
for sc in scopes:
    common = sorted(set(runs['eeg']) & set(runs.get('noise', {})))
    d2 = []; d1 = []; cis = {}
    for s in common:
        e, n = get(runs['eeg'][s], sc), get(runs['noise'][s], sc)
        d2.append(e['two_way'] - n['two_way']); d1.append(e['top1'] - n['top1'])
        a, b = np.array(e['per_query_two_way']), np.array(n['per_query_two_way']); diff = a - b
        bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)]
        cis[s] = {'two_way_diff': float(diff.mean()), 'ci95_query_bootstrap': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    out['eeg_minus_noise'][sc] = {'two_way': {'mean': float(np.mean(d2)), 'per_seed': d2, 'min': float(min(d2)), 'max': float(max(d2))}, 'top1': {'mean': float(np.mean(d1)), 'per_seed': d1}, 'per_seed_query_bootstrap': cis}
json.dump(out, open(f'{R}/aggregate.json', 'w'), indent=1)
print('| scope | N | input | seeds | top-1 mean (sd) | 2-way mean (sd) | bits mean | pair 2-way |'); print('|---|---|---|---|---|---|---|---|')
for sc in scopes:
    for inp in ('eeg', 'noise', 'shuffled'):
        if inp not in out['by_input']: continue
        v = out['by_input'][inp][sc]; f = lambda m: f"{v[m]['mean']:.3f} ({v[m]['sd']:.3f})" if v[m]['sd'] is not None else f"{v[m]['mean']:.3f}"
        print(f"| {sc} | {v['n_candidates']} | {inp} | {len(v['top1']['per_seed'])} | {f('top1')} | {f('two_way')} | {v['bits']['mean']:.3f} | {v['shuffled_pairs_two_way']['mean']:.3f} |")
print('\nEEG minus noise, 2-way (per seed; query-bootstrap 95% CI):')
for sc in scopes:
    e = out['eeg_minus_noise'][sc]; print(f"  {sc}: mean {e['two_way']['mean']:+.4f} per seed {[round(x,4) for x in e['two_way']['per_seed']]} | CIs " + '; '.join(f"s{s}: {c['two_way_diff']:+.3f} [{c['ci95_query_bootstrap'][0]:+.3f},{c['ci95_query_bootstrap'][1]:+.3f}]" for s, c in e['per_seed_query_bootstrap'].items()))
