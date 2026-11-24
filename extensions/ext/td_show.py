"""print-only viewer of results/td_pilot/eval_summary.json and controls_summary.json (runner, 24 Nov 2026). NOT citable."""
import json, sys
R = '/path/to/eegaudit/results/td_pilot/'
E = json.load(open(R + sys.argv[1] if len(sys.argv) > 1 else R + 'eval_summary.json'))
f3 = lambda x: '%.3f' % x if isinstance(x, (int, float)) and x == x else str(x)
ci = lambda d: '%s [%s, %s]' % (f3(d['value']), f3(d['ci95'][0]), f3(d['ci95'][1]))
print('primary', E['primary'])
for fam, M in E['families'].items():
    for k, r in M.items():
        print(f'--- {fam} {k}: F1 {r["F1"]} (signed {r["F1_signed_descriptive"]}), F4 {r["F4"]}, F5 {r["F5"]}, F2 {r["F2_coverage"]["reading"]}, F3 {r["F3_null_interval_contains_0"]["reading"]}, F6 {r["F6"]["reading"]}')
        print('   F1 cells MAE:', {c: (ci(v['MAE']), v['MAE']['reading'][:4]) for c, v in r['F1_cells'].items()})
        print('   truth/q50:', {c: (f3(v['mean_truth']), f3(v['mean_q50'])) for c, v in r['F1_cells'].items()})
        print('   F2 coverage', ci(r['F2_coverage']), r['F2_strata_descriptive'], '| raw cov', f3(r['descriptive']['coverage_six_uncalibrated']), 'width raw/cal', f3(r['descriptive']['median_width_six_uncalibrated']), f3(r['descriptive']['median_width_six_calibrated']))
        print('   F3 null contains0', ci(r['F3_null_interval_contains_0']), 'n', r['F3_null_n_cells'], '| |med|<=.02', f3(r['F3_extra_null_abs_median_le_0.02']), '| OC med<=.02', f3(r['F3_extra_ocular_only_median_le_0.02']), 'OC contains 0', f3(r['F3_extra_ocular_only_interval_contains_0']))
        print('   F4', {c: (ci(v) if v['value'] == v['value'] else v['reading'], v['reading'][:4]) for c, v in r['F4_cells'].items()})
        print('   F5 width ratio', ci(r['F5_width_ratio_rho0.9_over_0.25']), r['F5_width_ratio_rho0.9_over_0.25']['reading'], '| cov rho .9', ci(r['F5_coverage_rho0.9']), r['F5_coverage_rho0.9']['reading'], r['F5_median_width'])
        g = r['F6']; print('   F6 MAE model/A1/A4/A0', f3(g['MAE_model']), f3(g['MAE_A1']), f3(g['MAE_A4']), f3(g['MAE_A0']), 'better', g['better_baseline'], 'ratio', ci(g['ratio_to_better']), '| to A1', ci(g['ratio_to_A1']), '| to A4', ci(g['ratio_to_A4']), g['reading'])
        print('   F8', r['F8'], 'f0 MAE', f3(r['descriptive']['f0_MAE']), 'f0 truth', f3(r['descriptive']['f0_truth_mean']), 'YO cov', f3(r['descriptive']['YO_coverage_six']), 'truth six mean', f3(r['descriptive']['mean_truth_six']), 'q50 six mean', f3(r['descriptive']['mean_q50_six']), 'max|truth|', f3(r['descriptive']['max_abs_truth_six']))
print('R4 test participants', E.get('R4_test_participants'))
print('VERDICTS', json.dumps(E.get('verdict_per_model'), indent=1))
