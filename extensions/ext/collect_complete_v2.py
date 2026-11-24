"""Prints headline numbers of the 23 Nov COMPLETE-FRAMEWORK runs (). Reads results/gpu_fill_20261123/*; NOT citable."""
import json, sys, glob
from pathlib import Path
R = Path('/path/to/eegaudit/results/gpu_fill_20261123')
f = lambda c: f"{c['mean']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]" if isinstance(c, dict) and 'mean' in c else str(c)
def J(p):
    p = R / p; return json.loads(p.read_text()) if p.exists() else None
for d in ('mi_untrained', 'pe_untrained'):
    S = J(f'{d}/summary.json')
    if S: print(f'## {d} n={S["n_participants"]} seeds={S["seeds"]} gate={ {k: v for k, v in S["gate"].items() if "pass" in k or "diff" in k} }\n untrained seedmean {f(S["untrained_seedmean"])}; trained(saved) {f(S["saved_trained_eegnet"])}; trained-untrained {f(S["trained_minus_untrained"])}; ratio {S["untrained_over_trained_ratio"]:.3f}; shuffle floor {f(S["bits"]["shuffle_rand0"])} max|p| {S["shuffle_floor_max_abs_participant"]:.4f}; per-seed {[round(x,4) for x in S["per_seed_cohort_means"]]}')
S = J('image_untrained/summary.json')
if S:
    print('## image_untrained n=', S['n_participants'], 'G1 max', S.get('G1_max_abs_diff'))
    for m in ('nice_ga', 'brainflora'):
        if m in S: d = S[m]; print(f' {m}: k80 {f(d["k80"])} k40 {f(d["k40"])} | trained k80 {f(d.get("trained_k80"))} k40 {f(d.get("trained_k40"))} | share k80 {d.get("ratio_of_means_k80")} k40 {d.get("ratio_of_means_k40")} | noise k80 {f(d["noise_k80"])} shuffled {f(d["shuffled_k80"])} floors_ok {d["floors_ok_le_0.02"]}')
for rho in ('0.5', '0.25', '0.9'):
    S = J(f'planted_v2_rho{rho}/summary.json')
    if not S: continue
    print(f'## planted rho={rho} n={S["n_participants"]} G1 {S["G1_max_abs_diff"]} FALSIFIER {S["FALSIFIER"]["fired"]} {S["FALSIFIER"]["fractions"]}')
    for k, r in S['rows'].items():
        if not k.startswith('L64') or 'null' in k and 'f_planted' not in r: 
            if k.startswith('L64') and 'null' in k: print(' ', k, 'null floor ok', r.get('null_floor_ok_le_0.02'), 'yield_all', round(r['mean_yield_all'], 4), 'eogreg', round(r['mean_yield_eogreg'], 4))
            continue
        ko = r.get('kept_ocular_only'); print(f"  {k}: f_true {r['f_true_measured']['value']:.3f} est_share {r['est_nuisance_share']:.3f} [{r['est_ci95'][0]:.3f},{r['est_ci95'][1]:.3f}] kept_mix {r['kept']['value']:.3f} kept_neural {r['kept_neural_only']['value']:.3f} [{r['kept_neural_only']['ci95'][0]:.3f},{r['kept_neural_only']['ci95'][1]:.3f}] kept_ocular {ko['value'] if ko else None} predicted {r['predicted_share']:.3f} pred-obs {r['predicted_minus_observed']:+.3f} miss_vs_true {r['est_minus_true']:+.3f}")
for m in ('cbramod', 'reve', 'labram', 'eegpt'):
    S = J(f'mi_fm_meanspec/{m}/summary.json')
    if S: print(f'## fm_meanspec {m} ({S["label"]}) G1 {S.get("G1_pass")} {S.get("G1_max_abs_diff")} power_ratio {S["power_ratio"]}\n ' + '; '.join(f'{k} {f(v)}' for k, v in S['rows'].items() if 'seedmean' in k or 'real' in k or 'saved' in k) + ' | n>0.02: ' + str({k: v for k, v in S.items() if 'n_participants_above' in k}))
S = J('ubp_heldout/ubp_card.json')
if S:
    import numpy as np
    rt = S['retrain']; print('## ubp_heldout', len(rt), 'members')
    for arm in ('eeg', 'noise'):
        rows = [v for v in rt.values() if v['arm'] == arm]; P = sorted({v['participant'] for v in rows})
        for key in ('bits_A_k80', 'bits_Bprime_k80', 'bits_A_k40', 'bits_Bprime_k40', 'top1_k80'):
            pm = [np.mean([v[key] for v in rows if v['participant'] == p]) for p in P]; b = np.random.default_rng(0).choice(pm, (10000, len(pm))).mean(1)
            print(f' {arm} {key}: {np.mean(pm):.4f} [{np.quantile(b,.025):.4f}, {np.quantile(b,.975):.4f}] n={len(P)}')
        if arm == 'eeg':
            gap = [np.mean([v['bits_A_k80'] - v['bits_Bprime_k80'] for v in rows if v['participant'] == p]) for p in P]; b = np.random.default_rng(1).choice(gap, (10000, len(gap))).mean(1)
            print(f'  A minus Bprime k80 {np.mean(gap):+.4f} [{np.quantile(b,.025):+.4f}, {np.quantile(b,.975):+.4f}]; tau Bprime mean {np.mean([v["tau_Bprime"] for v in rows]):.4f} oracle {np.mean([v["tau_oracle_test_k80"] for v in rows]):.4f}')
for a in ('0.3', '1.0'):
    p = R / f'zuco_plant/bt_plant_amp{a}_results.json'
    if p.exists():
        d = json.loads(p.read_text())['identification']; print(f'## zuco plant amp {a}: ' + ' | '.join(f"{m}: top1 {d[m]['top1']:.3f} bits {d[m]['bits']:.4f} shuffled {d[m]['shuffled_pairs_bits']:.4f}" for m in ('teacher_forced', 'greedy', 'beam')))
