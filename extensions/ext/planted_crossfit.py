"""E3 planted_crossfit (RUNNER, 24 Nov 2026). Nothing is retrained: reads the saved per-participant planted_v2 files.
Eq. 6 (paper/v2/sections/results.tex, label eq:share):  1 - K_Z = f (1 - kappa_Z) + (1 - f) (1 - kappa_T),
  f = measured planted nuisance fraction = sum(Yo) / sum(Yn + Yo) (ocular-only yield over the sum of the single-source yields),
  kappa_Z = kept ratio of the ocular-only cell = mean(ocular_only_eogreg) / mean(ocular_only_perm)  (1 if f = 0),
  kappa_T = kept ratio of the neural-only cell = mean(neural_only_eogreg) / mean(neural_only_perm),
  observed removed share = 1 - mean(mix_eogreg) / mean(mix_permnull)        (as gf_planted_calib_v2.summarize; ratio of participant means, seed-mean per participant).
In the saved evidence f, kappa_Z, kappa_T are measured on the very cells whose share they predict. Cross-fit: estimate them on one participant half, predict the
removed share of the other half, both directions. Halves = alternate positions of the sorted participant list (A = positions 0,2,4,..; B = 1,3,5,..).
PRIMARY prediction: f, kappa_Z, kappa_T all from the training half (same planted-fraction cell). Variants (reported): nominal f; f from the test half.
Pre-registration: results/ext/planted_crossfit/PREREG.json (+ PREREG_STAMP.json). Usage: python -I -u planted_crossfit.py --out DIR [--smoke]"""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, csv, json, os, time
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='/path/to/eegaudit/results/ext/planted_crossfit')
ap.add_argument('--n_boot', type=int, default=10000)
ap.add_argument('--n_split', type=int, default=200)
ap.add_argument('--smoke', action='store_true', help='300 resamples, 20 random splits, gates still evaluated')
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
dev = torch.device('cuda'); t0 = time.time()
B = 300 if a.smoke else a.n_boot; NSPLIT = 20 if a.smoke else a.n_split
R = Path('/path/to/eegaudit/results/gpu_fill_20261123')
RHOS = ['0.5', '0.9', '0.25']; PRIMARY_RHOS = ['0.5', '0.9']; FS = [0.25, 0.5, 0.75]; LS = ['64', 'full']
KEYS = dict(A='mix_all', E='mix_eogreg', PN='mix_permnull', Yn='neural_only', Yo='ocular_only', Ne='neural_only_eogreg', Np='neural_only_perm',
            Oe='ocular_only_eogreg', Op='ocular_only_perm')
T = lambda x: torch.tensor(x, dtype=torch.float64, device=dev)

# ---------------- load: D[rho] = dict(subs, L -> f -> key -> (P,) seed-mean per participant), plus saved summaries for the gate
D = {}; SUM = {}
for rho in RHOS:
    d = R / f'planted_v2_rho{rho}'; SUM[rho] = json.loads((d / 'summary.json').read_text())
    files = sorted(p for p in d.glob('sub-*.json')); res = [json.loads(p.read_text()) for p in files]
    subs = [r['participant'] for r in res]; assert subs == sorted(subs) and len(subs) == SUM[rho]['n_participants'], (rho, subs)
    assert all(abs(r['rho'] - float(rho)) < 1e-12 for r in res)
    D[rho] = dict(subs=subs, L={})
    for L in LS:
        D[rho]['L'][L] = {}
        for f in FS:
            cs = [[c for c in r['cells'] if c['f'] == f] for r in res]
            assert all(len(c) == 2 for c in cs), (rho, f, [len(c) for c in cs])      # two seeds per participant
            D[rho]['L'][L][f] = {k: T([float(np.mean([c[key][L] for c in cc])) for cc in cs]) for k, key in KEYS.items()}
    print(rho, 'participants', len(subs), subs, flush=True)


def stats(c, ix=None):
    """c: key -> (P,) ; ix: None or (P_sel,) index tensor or (B,P_sel) index tensor. Returns f_true, kappa_Z, kappa_T, observed removed share (ratio of means)."""
    g = (lambda v: v if ix is None else v[ix]); mean = lambda v: g(v).mean(-1)
    yn, yo = mean(c['Yn']), mean(c['Yo'])
    return dict(f=yo / (yn + yo), kz=mean(c['Oe']) / mean(c['Op']), kt=mean(c['Ne']) / mean(c['Np']), obs=1 - mean(c['E']) / mean(c['PN']))


def pred_eq6(f, kz, kt):
    return f * (1 - kz) + (1 - f) * (1 - kt)


gates = []
def gate(nm, exp, got, tol):
    gates.append(dict(name=nm, expected=exp, got=got, abs_diff=abs(got - exp), tol=tol, pass_=bool(abs(got - exp) <= tol)))


# ---------------- gate 1: in-sample Eq. 6 on all participants reproduces the saved summary rows (same definitions)
insample = {}
for rho in RHOS:
    insample[rho] = {}
    for L in LS:
        for f in FS:
            c = D[rho]['L'][L][f]; s = stats(c); row = SUM[rho]['rows'][f'L{L}|f={f}']
            p = float(pred_eq6(s['f'], s['kz'], s['kt']))
            for nm, got, exp in (('f_true', float(s['f']), row['f_true_measured']['value']), ('kappa_Z', float(s['kz']), row['kept_ocular_only']['value']),
                                 ('kappa_T', float(s['kt']), row['kept_neural_only']['value']), ('observed_share', float(s['obs']), row['est_nuisance_share']),
                                 ('predicted_share', p, row['predicted_share']), ('predicted_minus_observed', p - float(s['obs']), row['predicted_minus_observed'])):
                gate(f'in-sample rho {rho} L{L} f={f} {nm} vs saved summary', exp, got, 1e-4)
            insample[rho][f'L{L}|f={f}'] = dict(f=float(s['f']), kappa_Z=float(s['kz']), kappa_T=float(s['kt']), observed=float(s['obs']), predicted=p, abs_err=abs(p - float(s['obs'])))
# macros of the paper (3 decimals, informational): PlHalfLo/Hi from rho 0.5 L64 abs errors
ie = [insample['0.5'][f'L64|f={f}']['abs_err'] for f in FS]
gate('macro PlHalfLo = min |pred-obs| rho 0.5 L64 (3-dec rounding)', 0.016, min(ie), 6e-4)
gate('macro PlHalfHi = max |pred-obs| rho 0.5 L64 (3-dec rounding)', 0.040, max(ie), 6e-4)
gate('macro PlNineLo = min |pred-obs| rho 0.9 L64 (3-dec rounding, informational)', 0.002, min(insample['0.9'][f'L64|f={f}']['abs_err'] for f in FS), 6e-4)
gate('macro PlNineHi = max |pred-obs| rho 0.9 L64 (3-dec rounding, informational)', 0.002, max(insample['0.9'][f'L64|f={f}']['abs_err'] for f in FS), 6e-4)
print('in-sample abs errors rho 0.5 L64', [round(x, 4) for x in ie], 'rho 0.9', [round(insample['0.9'][f'L64|f={f}']['abs_err'], 4) for f in FS], flush=True)

# ---------------- cross-fit
def crossfit_point(rho, L, ixA, ixB):
    """both directions, three fractions. returns dict variant -> dir -> (3,) predicted, observed (3,)"""
    res = {}
    for dname, (tr, te) in (('A_to_B', (ixA, ixB)), ('B_to_A', (ixB, ixA))):
        pr = {'primary': [], 'nominal_f': [], 'test_half_f': [], 'baseline_share_equals_train_f': [], 'baseline_constant_train_mean_share': []}; ob = []
        trobs = []
        for f in FS:
            c = D[rho]['L'][L][f]; st = stats(c, tr); se = stats(c, te)
            pr['primary'].append(pred_eq6(st['f'], st['kz'], st['kt'])); pr['nominal_f'].append(pred_eq6(torch.tensor(f, dtype=torch.float64, device=dev), st['kz'], st['kt']))
            pr['test_half_f'].append(pred_eq6(se['f'], st['kz'], st['kt'])); pr['baseline_share_equals_train_f'].append(st['f']); trobs.append(st['obs']); ob.append(se['obs'])
        pr['baseline_constant_train_mean_share'] = [torch.stack(trobs).mean()] * 3
        res[dname] = dict(pred={k: torch.stack(v) for k, v in pr.items()}, obs=torch.stack(ob))
    return res


allP = {rho: len(D[rho]['subs']) for rho in RHOS}
summary = dict(NOT_CITABLE='new output, no blind review yet', code_sha256=CODE_SHA256, script=os.path.abspath(__file__), smoke=a.smoke, n_boot=B, n_random_splits=NSPLIT,
               eq6='1-K_Z = f(1-kappa_Z) + (1-f)(1-kappa_T), results.tex eq:share; saved evidence = results/gpu_fill_20261123/planted_v2_rho*/ (kappa and f measured on the cells they predict)',
               halves='A = positions 0,2,4.. of the sorted participant list, B = 1,3,5..', fractions=FS, primary_L='64', insample=insample, rho={})
rows = []
for rho in RHOS:
    P = allP[rho]; ixA = torch.arange(0, P, 2, device=dev); ixB = torch.arange(1, P, 2, device=dev)
    assert len(set(ixA.tolist()) & set(ixB.tolist())) == 0 and len(ixA) + len(ixB) == P
    for i in range(P):
        rows.append([rho, D[rho]['subs'][i], 'A' if i % 2 == 0 else 'B'])
    S = dict(n_participants=P, half_sizes=[len(ixA), len(ixB)], participants_A=[D[rho]['subs'][i] for i in ixA.tolist()], participants_B=[D[rho]['subs'][i] for i in ixB.tolist()], L={})
    for L in LS:
        pt = crossfit_point(rho, L, ixA, ixB)
        # bootstrap: participants resampled within each half; the same resample of A and of B serves both directions
        g = torch.Generator(device=dev); g.manual_seed(0)
        bA = ixA[torch.randint(0, len(ixA), (B, len(ixA)), device=dev, generator=g)]; bB = ixB[torch.randint(0, len(ixB), (B, len(ixB)), device=dev, generator=g)]
        bt = {'A_to_B': (bA, bB), 'B_to_A': (bB, bA)}
        boot = {}
        for dname, (tr, te) in bt.items():
            pr = {k: [] for k in pt['A_to_B']['pred']}; ob = []; trobs = []
            for f in FS:
                c = D[rho]['L'][L][f]; st = stats(c, tr); se = stats(c, te)
                pr['primary'].append(pred_eq6(st['f'], st['kz'], st['kt'])); pr['nominal_f'].append(pred_eq6(torch.full_like(st['f'], f), st['kz'], st['kt']))
                pr['test_half_f'].append(pred_eq6(se['f'], st['kz'], st['kt'])); pr['baseline_share_equals_train_f'].append(st['f']); trobs.append(st['obs']); ob.append(se['obs'])
            pr['baseline_constant_train_mean_share'] = [torch.stack(trobs).mean(0)] * 3
            boot[dname] = {k: torch.stack(v, 1) for k, v in pr.items()}, torch.stack(ob, 1)   # (B,3)
        SL = dict(per_direction={}, variants={});
        for var in pt['A_to_B']['pred']:
            maes = {}
            e_pt = {d: (pt[d]['pred'][var] - pt[d]['obs']).abs() for d in pt}                       # (3,)
            mae_dir_pt = {d: float(e_pt[d].mean()) for d in e_pt}
            mae_pt = (mae_dir_pt['A_to_B'] + mae_dir_pt['B_to_A']) / 2
            eb = {d: (boot[d][0][var] - boot[d][1]).abs() for d in boot}                            # (B,3)
            mae_b = (eb['A_to_B'].mean(1) + eb['B_to_A'].mean(1)) / 2
            ci = [float(torch.quantile(mae_b, 0.025)), float(torch.quantile(mae_b, 0.975))]
            SL['variants'][var] = dict(mae_mean_of_two_directions=mae_pt, mae_ci95_participant_bootstrap=ci, mae_by_direction=mae_dir_pt,
                                       abs_err_by_direction_and_f={d: {str(f): float(e_pt[d][j]) for j, f in enumerate(FS)} for d in e_pt},
                                       max_abs_err=float(torch.stack(list(e_pt.values())).max()),
                                       predicted={d: {str(f): float(pt[d]['pred'][var][j]) for j, f in enumerate(FS)} for d in pt})
        SL['observed_share_test_half'] = {d: {str(f): float(pt[d]['obs'][j]) for j, f in enumerate(FS)} for d in pt}
        # random half splits (report only)
        gr = torch.Generator(device=dev); gr.manual_seed(0); mm = []
        for _ in range(NSPLIT):
            pm = torch.randperm(P, device=dev, generator=gr); h = P // 2
            r = crossfit_point(rho, L, pm[:h], pm[h:])
            mm.append(float(sum((r[d]['pred']['primary'] - r[d]['obs']).abs().mean() for d in r) / 2))
        mm = np.array(mm); SL['random_splits_primary_mae'] = dict(n=NSPLIT, median=float(np.median(mm)), q05=float(np.quantile(mm, .05)), q95=float(np.quantile(mm, .95)), max=float(mm.max()), frac_le_0_05=float((mm <= 0.05).mean()))
        S['L'][L] = SL
        v = SL['variants']['primary']
        print(f"rho {rho} L{L} PRIMARY MAE {v['mae_mean_of_two_directions']:.4f} {[round(x, 4) for x in v['mae_ci95_participant_bootstrap']]} dirs {({k: round(x, 4) for k, x in v['mae_by_direction'].items()})} "
              f"| nominal-f {SL['variants']['nominal_f']['mae_mean_of_two_directions']:.4f} test-f {SL['variants']['test_half_f']['mae_mean_of_two_directions']:.4f} | baselines share=f "
              f"{SL['variants']['baseline_share_equals_train_f']['mae_mean_of_two_directions']:.4f} const {SL['variants']['baseline_constant_train_mean_share']['mae_mean_of_two_directions']:.4f} | rand splits median {SL['random_splits_primary_mae']['median']:.4f}", flush=True)
    summary['rho'][rho] = S

dec = {}
for rho in PRIMARY_RHOS:
    v = summary['rho'][rho]['L']['64']['variants']['primary']
    dec[rho] = dict(mae=v['mae_mean_of_two_directions'], ci95=v['mae_ci95_participant_bootstrap'], threshold=0.05, pass_=bool(v['mae_mean_of_two_directions'] <= 0.05))
summary['decision'] = dict(per_rho=dec, rho_025_reported_only=summary['rho']['0.25']['L']['64']['variants']['primary']['mae_mean_of_two_directions'],
                           PASS_prereg=bool(all(d['pass_'] for d in dec.values())))
summary['gates'] = gates
summary['gate_all_pass_hard'] = all(g['pass_'] for g in gates if 'macro PlNine' not in g['name'])
summary['gate_failures'] = [g for g in gates if not g['pass_']]
for g in gates:
    if not g['pass_']:
        print('GATE FAIL', g['name'], g['expected'], g['got'], flush=True)
print('GATES', sum(g['pass_'] for g in gates), 'of', len(gates), 'pass; hard gate all pass:', summary['gate_all_pass_hard'], flush=True)
print('DECISION', json.dumps(summary['decision']), flush=True)
summary['runtime_s'] = time.time() - t0
(out / 'summary.json').write_text(json.dumps(summary, indent=1))
with open(out / 'per_participant_halves.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['rho', 'participant', 'half']); w.writerows(rows)
print('wrote', out, f'{time.time()-t0:.0f}s', flush=True)
