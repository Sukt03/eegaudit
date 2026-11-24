"""S-L (b): bits at the pooled coupling rho_eff (random-effects mean of the null-corrected Fisher-z couplings d_i, tanh) vs the mean of
per-participant bits b(tanh d_i), per decoder, with participant-bootstrap CIs; also the raw-r version and the floor at the permutation p95 r."""
import json, numpy as np
R = '/path/to/eegaudit/results'; G = json.load(open(f'{R}/group_stats/results.json'))['decoders']; P = json.load(open(f'{R}/track_decision/price_list.json'))['datasets']['ds003458']
b = lambda r: float(-0.5 * np.log2(1 - min(abs(r), 0.999) ** 2)); rng = np.random.default_rng(0); out = {}
for k, g in G.items():
    d = np.array(g['d']); r = np.array(g['r']); rho = np.tanh(d); pb = np.array([b(x) for x in rho])
    bs = [np.mean(pb[rng.integers(0, len(pb), len(pb))]) for _ in range(4000)]
    c = P.get('controls', {}).get(k, {}).get('session_permutation_null', {})
    out[k] = dict(n=len(d), rho_eff_pooled=g['rho_eff_pooled'], bits_at_pooled=b(g['rho_eff_pooled']), bits_at_pooled_ci=[b(x) for x in g['rho_eff_ci']],
                  mean_per_participant_bits=float(pb.mean()), mean_per_participant_bits_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                  median_per_participant_bits=float(np.median(pb)), min_max_per_participant_bits=[float(pb.min()), float(pb.max())],
                  mean_raw_r_bits=float(np.mean([b(x) for x in r])), floor_p95_r=c.get('p95_r'), floor_bits=b(c['p95_r']) if c.get('p95_r') is not None else None,
                  n_participants_above_floor=int((np.abs(rho) > c['p95_r']).sum()) if c.get('p95_r') is not None else None)
json.dump(out, open(f'{R}/loop/learner/pooled_vs_mean_bits.json', 'w'), indent=1)
print('| decoder | n | rho_eff | bits at pooled [CI] | mean per-participant bits [CI] | median | floor bits (p95 r) | n above floor |'); print('|---|---|---|---|---|---|---|---|')
for k, v in sorted(out.items(), key=lambda kv: -kv[1]['bits_at_pooled']):
    print(f"| {k} | {v['n']} | {v['rho_eff_pooled']:.3f} | {v['bits_at_pooled']:.3f} [{v['bits_at_pooled_ci'][0]:.3f}, {v['bits_at_pooled_ci'][1]:.3f}] | {v['mean_per_participant_bits']:.3f} [{v['mean_per_participant_bits_ci'][0]:.3f}, {v['mean_per_participant_bits_ci'][1]:.3f}] | {v['median_per_participant_bits']:.3f} | {v['floor_bits'] if v['floor_bits'] is None else round(v['floor_bits'],4)} ({v['floor_p95_r']}) | {v['n_participants_above_floor']} |")
