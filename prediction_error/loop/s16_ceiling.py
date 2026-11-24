"""S16: ceiling and fraction for the cognitive variable. rho_max (EEGNet's own split-half determinacy, Spearman-Brown, from
results/loop/r5_eegnet_ceiling/r5_eegnet_splithalf_0-9.json; participants 1-9, 10 epochs) -> ceiling = -1/2 log2(1 - rho_max^2);
fraction = (bits_g - floor) / (ceiling - floor) with the per-participant null-corrected bits_g of measure_bits_gauss.json (EEGNet, ds003458),
participant bootstrap over the 23 bits values with the 9 rho_max values resampled jointly; the ensemble ceiling (r4, rho_max 0.846, n = 23) beside it."""
import json, glob, numpy as np
R = '/path/to/eegaudit/results'; rng = np.random.default_rng(0); b = lambda r: float(-0.5 * np.log2(1 - min(abs(float(r)), 0.999) ** 2))
G = json.load(open(f'{R}/loop/learner/measure_bits_gauss.json'))['datasets']['ds003458']['decoders']['eegnet']
pp = np.array([b(np.tanh(x)) for x in G['per_participant_d']]); floor = G['floor_bits_g']
f = sorted(glob.glob(f'{R}/loop/r5_eegnet_ceiling/r5_eegnet_splithalf_0-9*.json'))
S = json.load(open(f[-1])); rows = [r for r in S['rows']]; rm = np.array([r['rho_max_decoder'] for r in rows])
r4 = json.load(open(f'{R}/method_pilots/r4_summary.json')); rm_ens = r4['rho_max_decoder_mean']
out = dict(source_splithalf=f[-1], n_splithalf=len(rows), epochs=S['epochs'], rho_max_eegnet=dict(per_participant=rm.round(3).tolist(), mean=float(rm.mean()), median=float(np.median(rm)), min=float(rm.min())),
           r12_mean=float(np.mean([r['r12_half_decoders'] for r in rows])), r_full_mean=float(np.mean([r['r_full'] for r in rows])), r_half_mean=float(np.mean([r['r_half_data'] for r in rows])),
           ceiling_bits_eegnet=b(rm.mean()), ceiling_bits_ens=b(rm_ens), rho_max_ens=rm_ens, floor_bits_g=floor, bits_g_mean_pp=float(pp.mean()))
def frac(bits, ceil): return (bits - floor) / (ceil - floor)
bs = [frac(pp[rng.integers(0, 23, 23)].mean(), b(rm[rng.integers(0, len(rm), len(rm))].mean())) for _ in range(4000)]
out['fraction_eegnet_ceiling'] = dict(point=frac(pp.mean(), b(rm.mean())), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
bs2 = [frac(pp[rng.integers(0, 23, 23)].mean(), b(rm_ens)) for _ in range(4000)]
out['fraction_ens_ceiling'] = dict(point=frac(pp.mean(), b(rm_ens)), ci95=[float(np.percentile(bs2, 2.5)), float(np.percentile(bs2, 97.5))], note='ensemble read-out ceiling, n = 23, labelled as such')
out['criterion'] = 'PASS if fraction in [0,1] and CI excludes 0 (ds003458)'; p = out['fraction_eegnet_ceiling']
out['verdict'] = 'PASS' if (0 <= p['point'] <= 1 and p['ci95'][0] > 0) else 'FAIL'
json.dump(out, open(f'{R}/loop/learner/s16_ceiling.json', 'w'), indent=1); print(json.dumps(out, indent=1))
