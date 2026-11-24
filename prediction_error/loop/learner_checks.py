"""Learner (ds003458, RWD) checks requested 13:05: re-analysis of existing result files, no refits.
S-L1 circular-shift control on the variance ratio: per participant, var(logit alpha | joint)/var(beh) and var(shifted-read-out joint)/var(beh),
     first-half fit (results/real_verify_eegnet/results.json: h1_joint, h1_shuffled, h1_beh). Pre-registered: the over-contraction is attributable
     to the read-out CONTENT (not to the i.i.d. Gaussian likelihood form) if the shifted ratio's median is >= 0.9 and the paired Wilcoxon
     (joint < shifted) has p < 0.05. Observed ratio reported in bits (0.5 log2(1/ratio)) next to the prediction, with a participant bootstrap CI
     and a Wilcoxon test against the predicted ratio.
S-L2 prediction at each participant's own EEG trial count: n_eeg in the first-half window (trials < 240) from results/features/eegnet/sub-XXX.npz
     ('tri'); predicted ratio (1-c(rho_i, n0))/(1-c(0, n0)) interpolated in rho on the exact curves at n0 = 120, 180, 240
     (results/ceiling_RWD_v2{_n0_120,_n0_180,}/results.json) and linearly in n0. Pre-registered: the trial-count explanation is void if the
     per-participant prediction changes the mean predicted ratio by < 0.02.
S-L3 power of the held-out-choice test under a correctly specified read-out: 92 simulated twins (4 per donor, 23 donors, the donor's EEG trial
     set and measured rho; results/real_twin_eegnet_{a,b,c}); expected gain (nats/trial) and its distribution; power = fraction of 23-twin sets
     (one twin per donor; the 4 disjoint sets and 2000 random one-per-donor sets) in which the one-sided Wilcoxon (joint > beh) has p < 0.05.
     Same for the reliability difference (odd/even r, joint minus beh) and the variance-ratio test (median ratio < 1, and vs the prediction)."""
import json, glob, numpy as np
from scipy import stats
R = '/path/to/eegaudit/results'; rng = np.random.default_rng(0)
V = json.load(open(f'{R}/real_verify_eegnet/results.json')); S = {int(k): v for k, v in V['subjects'].items()}; subs = sorted(S)
A = 0  # logit alpha
bits = lambda ratio: 0.5 * np.log2(1 / ratio)
def ci(x, f=np.mean, n=4000):
    x = np.asarray(x); b = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
# ---- curves
curves = {}
for n0, d in ((120, 'ceiling_RWD_v2_n0_120'), (180, 'ceiling_RWD_v2_n0_180'), (240, 'ceiling_RWD_v2')):
    C = json.load(open(f'{R}/{d}/results.json')); ks = sorted(C['joint'], key=float)
    curves[n0] = (np.array([float(k) for k in ks]), np.array([C['joint'][k]['exact']['logit_alpha']['contraction'] for k in ks]))
def pred_ratio(rho, n0):
    def at(n):
        rh, c = curves[n]; cr = np.interp(rho, rh, c); return (1 - cr) / (1 - c[0])
    ns = np.array([120, 180, 240]); return float(np.interp(n0, ns, [at(n) for n in ns]))
# ---- S-L1
r_joint = np.array([S[s]['h1_joint']['var'][A] / S[s]['h1_beh']['var'][A] for s in subs])
r_shift = np.array([S[s]['h1_shuffled']['var'][A] / S[s]['h1_beh']['var'][A] for s in subs])
rho = np.array([S[s]['rho'] for s in subs]); pred240 = np.array([pred_ratio(r, 240) for r in rho])
L1 = dict(n=len(subs), observed_ratio=dict(mean=float(r_joint.mean()), median=float(np.median(r_joint)), ci95_mean=ci(r_joint), ci95_median=ci(r_joint, np.median), per_participant=r_joint.round(3).tolist()),
          observed_bits=dict(mean_of_bits=float(np.mean(bits(r_joint))), bits_of_mean_ratio=float(bits(r_joint.mean())), bits_of_median_ratio=float(bits(np.median(r_joint))), ci95_mean_of_bits=ci(bits(r_joint))),
          predicted_ratio_at_240=dict(mean=float(pred240.mean()), bits_of_mean=float(bits(pred240.mean())), per_participant=pred240.round(3).tolist()),
          shifted_ratio=dict(mean=float(r_shift.mean()), median=float(np.median(r_shift)), ci95_median=ci(r_shift, np.median), per_participant=r_shift.round(3).tolist(), n_below_0_9=int((r_shift < 0.9).sum())),
          tests=dict(wilcoxon_obs_vs_pred_p=float(stats.wilcoxon(r_joint - pred240).pvalue), n_obs_below_pred=int((r_joint < pred240).sum()),
                     wilcoxon_joint_lt_shifted_p=float(stats.wilcoxon(r_joint - r_shift, alternative='less').pvalue), n_joint_below_shifted=int((r_joint < r_shift).sum()),
                     wilcoxon_shifted_vs_1_p=float(stats.wilcoxon(np.log(r_shift)).pvalue), wilcoxon_shifted_vs_pred_p=float(stats.wilcoxon(r_shift - pred240).pvalue)),
          ess_note='importance-sampling ESS per fit in the source file (ess_min %s, ess_median %s)' % (V['summary']['ess_min'], V['summary']['ess_median']))
L1['criterion'] = 'PASS if median shifted ratio >= 0.9 and Wilcoxon(joint < shifted) p < 0.05'
L1['verdict'] = 'PASS' if (L1['shifted_ratio']['median'] >= 0.9 and L1['tests']['wilcoxon_joint_lt_shifted_p'] < 0.05) else 'FAIL'
# ---- S-L2
n_h1 = np.array([int((np.load(f'{R}/features/eegnet/sub-{s:03d}.npz')['tri'] < 240).sum()) for s in subs]); n_all = np.array([S[s]['n_eeg'] for s in subs])
pred_own = np.array([pred_ratio(r, n) for r, n in zip(rho, n_h1)])
L2 = dict(n_eeg_first_half=dict(per_participant=n_h1.tolist(), median=float(np.median(n_h1)), min=int(n_h1.min()), max=int(n_h1.max())), n_eeg_session=dict(median=float(np.median(n_all)), min=int(n_all.min()), max=int(n_all.max())),
          predicted_ratio_240=float(pred240.mean()), predicted_ratio_own_count=float(pred_own.mean()), change=float(pred_own.mean() - pred240.mean()),
          predicted_bits_own_count=float(bits(pred_own.mean())), per_participant_pred_own=pred_own.round(3).tolist(),
          wilcoxon_obs_vs_pred_own_p=float(stats.wilcoxon(r_joint - pred_own).pvalue), n_obs_below_pred_own=int((r_joint < pred_own).sum()),
          note='the first-half fit window is trials 1-240; the ratio fit uses every EEG trial in that window (n_eeg here), see real_joint_fit.py test C')
L2['criterion'] = 'trial-count explanation void if |change| < 0.02'; L2['verdict'] = 'PASS' if abs(L2['change']) < 0.02 else 'FAIL'
# ---- S-L3
T = {}
for f in sorted(glob.glob(f'{R}/real_twin_eegnet_[abc]/results.json')): T.update(json.load(open(f))['subjects'])
donors = sorted({int(k.split('_')[0]) for k in T}); reps = sorted({int(k.split('_')[1]) for k in T})
gain = {k: (v['A_joint']['lpred'] - v['A_beh']['lpred']) / v['n_test'] for k, v in T.items()}
ratio_t = {k: v['A_joint']['var'][A] / v['A_beh']['var'][A] for k, v in T.items()}
def set_stats(keys):
    g = np.array([gain[k] for k in keys]); rt = np.array([ratio_t[k] for k in keys])
    o_b = np.array([T[k]['B_odd_beh']['mean'][A] for k in keys]); e_b = np.array([T[k]['B_even_beh']['mean'][A] for k in keys])
    o_j = np.array([T[k]['B_odd_joint']['mean'][A] for k in keys]); e_j = np.array([T[k]['B_even_joint']['mean'][A] for k in keys])
    return dict(A_mean_gain=float(g.mean()), A_p_greater=float(stats.wilcoxon(g, alternative='greater').pvalue), B_r_diff=float(stats.pearsonr(o_j, e_j)[0] - stats.pearsonr(o_b, e_b)[0]),
                C_median_ratio=float(np.median(rt)), C_p_lt_1=float(stats.wilcoxon(np.log(rt), alternative='less').pvalue), C_p_vs_pred=float(stats.wilcoxon(rt - pred240).pvalue))
disjoint = [set_stats([f'{d}_{r}' for d in donors if f'{d}_{r}' in T]) for r in reps]
random_sets = [set_stats([f'{d}_{rng.choice([r for r in reps if f"{d}_{r}" in T])}' for d in donors]) for _ in range(2000)]
allg = np.array(list(gain.values())); allr = np.array(list(ratio_t.values()))
L3 = dict(n_twins=len(T), n_donors=len(donors), twins_per_donor=len(reps), source='results/real_twin_eegnet_{a,b,c} (real_joint_ext.py --part twin: donor EEG trial set, donor rho, correctly specified read-out)',
          expected_gain_nats_per_trial=dict(mean=float(allg.mean()), sd_across_twins=float(allg.std(ddof=1)), ci95=ci(allg)),
          observed_real_gain_nats_per_trial=V['summary']['A_h1_to_h2_joint']['mean_gain_nats_per_trial'],
          power_A_disjoint_sets=float(np.mean([d['A_p_greater'] < 0.05 for d in disjoint])), power_A_random_sets=float(np.mean([d['A_p_greater'] < 0.05 for d in random_sets])),
          set_mean_gain_sd=float(np.std([d['A_mean_gain'] for d in random_sets])),
          B_r_diff=dict(mean=float(np.mean([d['B_r_diff'] for d in random_sets])), sd=float(np.std([d['B_r_diff'] for d in random_sets])), observed_real=V['summary']['B_logit_alpha']['joint_minus_beh']['r_diff']),
          C_twin_ratio=dict(median_all=float(np.median(allr)), mean_all=float(allr.mean()), ci95_median=ci(allr, np.median), power_lt_1=float(np.mean([d['C_p_lt_1'] < 0.05 for d in random_sets])),
                            frac_sets_differing_from_pred=float(np.mean([d['C_p_vs_pred'] < 0.05 for d in random_sets])), predicted_mean=float(pred240.mean())),
          disjoint_sets=disjoint)
L3['criterion'] = 'the held-out-choice null is informative only if power >= 0.8 under correct specification; otherwise the null cannot falsify the prediction'
L3['verdict'] = 'PASS (null informative)' if L3['power_A_random_sets'] >= 0.8 else 'FAIL (null uninformative: power %.2f)' % L3['power_A_random_sets']
out = dict(S_L1=L1, S_L2=L2, S_L3=L3)
json.dump(out, open(f'{R}/loop/learner/learner_checks.json', 'w'), indent=1)
for k, v in out.items(): print(k, v['verdict']); 
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ('disjoint_sets',) and not (isinstance(vv, dict) and 'per_participant' in vv and len(json.dumps(vv)) > 400)} for k, v in out.items()}, indent=0)[:6000])
