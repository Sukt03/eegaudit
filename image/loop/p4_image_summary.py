"""P4 (image) summary from results/loop/image_p1/p1_rank_bound.json: per participant I_NCE (= I_V with H_V(T) = log2 200 at the LOO temperature; the
shuffled-pair soft bits are the empirical floor) for NICE-GA (k = 80, k = 1 mean of 5 draws) and ATM-S released embeddings; PVI median / IQR / fraction
positive per participant; ladder increment ATM-S - NICE-GA (transformer - CNN) per participant; the linear raw-voltage class has no cached soft bits
(Wolpaw only, results/eegbits/summary.json raw: 2.9 % top-1, 0.04 bits) and is listed as such. Output results/loop/p4/p4_image.json."""
import json, numpy as np
R = '/path/to/eegaudit/results/loop'; d = json.load(open(f'{R}/image_p1/p1_rank_bound.json'))['per_participant']; rng = np.random.default_rng(0)
rows = []
for s in sorted(d['nice_ga']):
    n80 = d['nice_ga'][s]['k80']; n1 = d['nice_ga'][s]['k1']; a = d['atm_native'][s]
    rows.append(dict(subject=s, nice_ga_k80_I_NCE=n80['I_NCE'], nice_ga_k80_floor_pairs=n80['shuffled_pair']['I_NCE'], nice_ga_k80_pvi=n80['pvi'], nice_ga_k1_I_NCE=float(np.mean([x['I_NCE'] for x in n1])), nice_ga_k1_pvi_median=float(np.mean([x['pvi']['median'] for x in n1])), nice_ga_k1_frac_pos=float(np.mean([x['pvi']['frac_positive'] for x in n1])),
                     atm_I_NCE=a['I_NCE'], atm_floor_pairs=a['shuffled_pair']['I_NCE'], atm_pvi=a['pvi'], ladder_atm_minus_nice=a['I_NCE'] - n80['I_NCE']))
def ci(v): v = np.asarray(v); return [float(np.percentile([v[rng.integers(0, len(v), len(v))].mean() for _ in range(4000)], q)) for q in (2.5, 97.5)]
S = {k: dict(mean=float(np.mean([r[k] for r in rows])), ci95=ci([r[k] for r in rows]), median=float(np.median([r[k] for r in rows]))) for k in ('nice_ga_k80_I_NCE', 'nice_ga_k1_I_NCE', 'atm_I_NCE', 'ladder_atm_minus_nice', 'nice_ga_k80_floor_pairs', 'atm_floor_pairs')}
S['n_participants'] = len(rows); S['ladder_n_positive'] = int(sum(r['ladder_atm_minus_nice'] > 0 for r in rows)); S['all_I_V_nonneg_after_floor'] = bool(all(r['nice_ga_k80_I_NCE'] - r['nice_ga_k80_floor_pairs'] >= 0 and r['atm_I_NCE'] - r['atm_floor_pairs'] >= 0 for r in rows))
S['pvi'] = dict(nice_ga_k80_median_mean=float(np.mean([r['nice_ga_k80_pvi']['median'] for r in rows])), nice_ga_k80_frac_pos_mean=float(np.mean([r['nice_ga_k80_pvi']['frac_positive'] for r in rows])), atm_median_mean=float(np.mean([r['atm_pvi']['median'] for r in rows])), atm_frac_pos_mean=float(np.mean([r['atm_pvi']['frac_positive'] for r in rows])), nice_ga_k1_frac_pos_mean=float(np.mean([r['nice_ga_k1_frac_pos'] for r in rows])))
S['raw_voltage_linear_class'] = 'no cached soft bits; Wolpaw 0.04 bits (summary.json raw), listed as the linear class without PVI'
json.dump({'summary': S, 'rows': rows}, open(f'{R}/p4/p4_image.json', 'w'), indent=1); print(json.dumps(S, indent=0))
