"""print compact tables from results/ext/glim_within_task/summary.json (read-only helper)"""
import json, sys
S = json.load(open('/path/to/eegaudit/results/ext/glim_within_task/summary.json'))
f = lambda x: f'{x:+.4f}'
c = lambda v: f'[{v[0]:+.3f},{v[1]:+.3f}]'
for gn in ('card12', 'ext100'):
    print('=====', gn)
    for sel, r in S['selections'].items():
        G = r['grids'][gn]; D = G['diffs']['eeg_minus_noise_test_time']; ci = G['ci95']
        k = 'diff:eeg_minus_noise_test_time'
        print(f"{sel:24s} n={r['n_records']} N={r['n_candidates']} P={r['n_participants']} eeg {f(G['arms']['eeg']['yield_bits'])} noise {f(G['arms']['noise_test_time']['yield_bits'])} pf {f(G['arms']['prompt_free']['yield_bits'])} | d {f(D['mean'])} S{c(ci['sentence'][k])} P{c(ci['participant'][k])} X{c(ci['crossed'][k])} pos {D['n_participants_positive']} | floors eeg {f(G['arms']['eeg']['floor_shuffled_bits'])} noise {f(G['arms']['noise_test_time']['floor_shuffled_bits'])} dfl {f(D['floor_shuffled_mean'])} | edge eeg {G['arms']['eeg']['frac_records_at_grid_upper_edge']:.2f} noise {G['arms']['noise_test_time']['frac_records_at_grid_upper_edge']:.2f} tau_med {G['arms']['eeg']['chosen_tau_median']:.3g}/{G['arms']['noise_test_time']['chosen_tau_median']:.3g}")
        for k2 in ('eeg_minus_prompt_free', 'prompt_free_minus_noise_test_time'):
            d = G['diffs'][k2]; kk = 'diff:' + k2
            print(f"      {k2}: {f(d['mean'])} S{c(ci['sentence'][kk])} P{c(ci['participant'][kk])} X{c(ci['crossed'][kk])} floor {f(d['floor_shuffled_mean'])}")
        print('      oracle', {k: round(v['bits_per_record'], 4) for k, v in r['oracle'].items()}, 'nan draws', G['ci_nan_draws'])
        print('      yield CIs (sentence)', {k: c(v) for k, v in ci['sentence'].items() if k.startswith('yield')})
