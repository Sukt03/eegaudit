"""S4: bits per second with the decoded window as denominator (not the 0.2 s SOA), and cognitive bits per feedback epoch.
Image (THINGS-EEG2): bits/s = bits / (k x window_s). Native (k = 80, 1.0 s window) and k = 1 for ATM-S and NICE-GA from results/track_image/summary.json
(which used stimulus_time_s = 0.2 SOA); retrained 0.6 s / 0.4 s windows (NICE-GA, k = 80 and k = 1) from results/eegbits/retrain_summary.json.
Text (ZuCo): bits per sentence / mean reading time per sentence (within_SR meta reading_times, ~3.9-4.0 s), best rung and mean rung from max_over_rungs.json.
Cognitive: bits per feedback epoch (the unit already; the decoded window 0-1.0 s post-feedback) from results/loop/learner/*.json."""
import json, numpy as np
R = '/path/to/eegaudit/results'
img = json.load(open(f'{R}/track_image/summary.json')); rt = json.load(open(f'{R}/eegbits/retrain_summary.json'))
out = {'image': {}, 'text': {}, 'cognitive': {}, 'note': 'bits/s = bits / (k x decoded window in s); the previous summary divided by the 0.2 s SOA'}
for dec, v in img['decoders'].items():
    o = {}
    for k in ('1', '80'):
        r = v['repetitions'].get(k) or (v['native'] if k == '80' else None)
        if r is None: continue
        b = r['bits']['mean']; o[f'k{k}_window1.0'] = dict(bits=b, bits_per_s_window=b / (int(k) * 1.0), bits_per_s_SOA_old=r.get('bits_per_second', {}).get('mean'), n_subjects=r['n_subjects'])
    for w in ('0.6', '0.4', '0.2'):
        for kk, key in (('80', 'window_k80'), ('1', 'window_k1')):
            r = v.get(key, {}).get(w)
            if r: o[f'masked_k{kk}_window{w}'] = dict(bits=r['bits']['mean'], bits_per_s_window=r['bits']['mean'] / (int(kk) * float(w)), n_subjects=r['n_subjects'])
    out['image'][dec] = o
for var, w in (('w06', 0.6), ('w04', 0.4)):
    if var in rt:
        out['image'][f'NICE-GA retrained {var}'] = {f'k80_window{w}': dict(bits=rt[var]['k80_retrained_bits']['mean'] if isinstance(rt[var]['k80_retrained_bits'], dict) else rt[var]['k80_retrained_bits'], n_subjects=rt[var]['n_subjects'])}
        o = out['image'][f'NICE-GA retrained {var}'][f'k80_window{w}']; o['bits_per_s_window'] = o['bits'] / (80 * w)
        b1 = rt[var]['k1_retrained_bits']['mean'] if isinstance(rt[var]['k1_retrained_bits'], dict) else rt[var]['k1_retrained_bits']
        out['image'][f'NICE-GA retrained {var}'][f'k1_window{w}'] = dict(bits=b1, bits_per_s_window=b1 / w, n_subjects=rt[var]['n_subjects'])
meta = json.load(open(f'{R}/track_text/within_SR/meta.json')); mx = json.load(open(f'{R}/track_text/within_SR/max_over_rungs.json'))
dur = float(np.mean([f['mean_duration_s'] for f in meta['reading_times']['files'].values()]))
out['text'] = dict(mean_reading_time_s=dur, best_rung_bits=mx['eeg']['max_bits'], best_rung_bits_per_s=mx['eeg']['max_bits'] / dur, noise_max_bits=mx['noise']['max_bits'], noise_max_bits_per_s=mx['noise']['max_bits'] / dur,
                   note='max over 110 settings, N = 40; the mean rung is at the floor (2-way 0.507)')
L = json.load(open(f'{R}/loop/learner/negentropy_pe_target_v2.json')); P = json.load(open(f'{R}/loop/learner/pooled_vs_mean_bits.json'))
out['cognitive'] = dict(unit='bits per feedback epoch (decoded window 0-1.0 s post-feedback -> also bits/s)', eegnet_bits_gauss=L['gaussianised_coupling']['eegnet']['r_gauss']['bits_at_fisher_mean'], eegnet_bits_pooled=P['eegnet']['bits_at_pooled'], floor=P['eegnet']['floor_bits'])
json.dump(out, open(f'{R}/loop/s4_rates/rates.json', 'w'), indent=1)
for dec, o in out['image'].items():
    for k, v in o.items(): print(f"image {dec:28s} {k:24s} bits {v['bits']:.3f} bits/s(window) {v['bits_per_s_window']:.3f}" + (f" old(SOA) {v['bits_per_s_SOA_old']:.3f}" if v.get('bits_per_s_SOA_old') is not None else '') + f" n={v['n_subjects']}")
print('text', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out['text'].items()}); print('cognitive', out['cognitive'])
