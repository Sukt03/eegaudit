import json
S=json.load(open('/path/to/eegaudit/results/td_screen/summary.json'))
f=lambda d:f"{d['value']:.4f} [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]"
print({k:S[k] for k in ('n_train_cells','n_test_cells','n_primary_test_cells','primary_model','primary_S1_verdict','VERDICT','jitter_floor64','VERDICT_requires_controls_ok','runtime_s','A4_stored_for_test_participants','rhofree_dup_maxdiff')})
print('control_checks',S['control_checks'])
def show(n,r):
    print(n,'MAE',f(r['MAE']),'cov',f(r['coverage90']),'ratioA1',f(r['MAE_over_A1']),'A1MAE',f(r['A1_MAE']))
    print('   ',r['verdict_readings'])
    for k in ('f0.25','f0.5','f0.75'): print('   ',k,'MAE',f(r[k]['MAE']),'cov',f(r[k]['coverage90']),'A1',f(r[k]['A1_MAE']),'truth',round(r[k]['mean_truth'],4),'q50',round(r[k]['mean_q50'],4))
    print('    f0 MAE',f(r['f0_MAE']),'cov',f(r['f0_interval_contains_Y']),' null MAE',f(r['null_MAE']),'null cov',f(r['null_interval_contains_Y']),'null contains0',r['null_interval_contains_0'],'null |med|<=.02',r['null_median_abs_le_0.02'])
for k in ('lin','mlp'):
    print(k,'cv',S['models_S1'][k]['cv_pinball_train'],'lam',S['models_S1'][k]['lam']); show('S1 '+k,S['models_S1'][k]['test'])
show('constant',S['control_constant_quantiles'])
a=S['ablation_S2_with_A1_inputs']; print('S2 primary',a['primary'])
for k in ('lin','mlp'): print('S2',k,'cv',a[k]['cv_pinball_train']); show('S2 '+k,a[k]['test'])
c=S['controls']
show('noise',c['noise_input']); print(c['noise_input']['primary_kind'])
print('shuffled MAE',[round(x,4) for x in c['shuffled_truth']['MAE_point']],'ratio',[round(x,3) for x in c['shuffled_truth']['ratio_to_A1_point']],'cov',[round(x,3) for x in c['shuffled_truth']['coverage_point']])
show('positive',c['positive_control_label_as_input'])
print('YO',S['YO_descriptive'])
