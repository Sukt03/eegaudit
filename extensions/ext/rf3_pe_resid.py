"""RUNNER FIX 3 (round_20261123_auto; METHODS B6, IDEATION Idea 2): the SAME target ladder on three PE datasets with the d2 estimator used for ds007647
(d2_pe_tasks.pe_gain: in-fold z-score, PCA-64 + ridge (primary) and full width, 10 contiguous folds, nested calibration; window 0-600 ms).
Targets, each z-scored on the analysed trials:
  pe_raw          the dataset's locked model PE at the participant's MAP, NOT residualised
  outcome         reward
  pe_res_outcome  PE residualised on outcome only
  pe_res_oc       PE residualised on outcome, chosen option (dummies), option x outcome
  pe_res_full     the project's registered residualised target info['qtil'] (ds003458 / ds004595: outcome, chosen arm, arm x outcome, cubic trial index;
                  ds007647: outcome, choice, choice x outcome, condition, condition x outcome, cubic trial index)
  pe_res_full_re  my own recomputation of pe_res_full for ds003458 / ds004595 (code check: must equal qtil)
Floors per target: circular shifts n/4, n/2, 3n/4 (mean), one random permutation (seed 77 + s), N(0,1) input (seed 1000 + s).
Planted positive control: EEG features + 0.05 x column SD x pe_res_full x random-sign pattern -> the residualised target must be recovered (gain >> 0).
GATES: ds007647 'all' pe_res_full / pe_raw / outcome == results/ext/d2_ds007647_v2/sub-XXX.json (1e-4); ds003458 x7-window L64/full gain == results/standard/pe_bottleneck (1e-4).
Usage: --ds ds003458,ds004595,ds007647 --out DIR [--subjects 1,2]"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path('/path/to/eegaudit'); sys.path.insert(0, str(ROOT / 'experiments/ext')); sys.path.insert(0, str(ROOT))
import d2_pe_tasks as D2
from group_stats import resid
CODE_SHA256 = hashlib.sha256(open(__file__, 'rb').read()).hexdigest(); LS = (64, 'full'); DEV = 'cuda'
TG = ['pe_raw', 'outcome', 'pe_res_outcome', 'pe_res_oc', 'pe_res_full', 'pe_res_full_re']

def zs(v): v = np.asarray(v, float); return (v - v.mean()) / (v.std() + 1e-9)

def behaviour(ds, info):
    """-> f(label, tri) = (pe, reward, choice, T, cond or None) from the closure of the dataset's own qtil (no re-implementation of the model run)."""
    cl = dict(zip(info['qtil'].__code__.co_freevars, [c.cell_contents for c in info['qtil'].__closure__]))
    if ds == 'ds003458': Q, fq = cl['Q'], cl['full_q']; return lambda s, tri: (fq[s - 1][tri], Q['reward'][s - 1][tri], Q['choice'][s - 1][tri], 480.0, None)
    if ds == 'ds004595': B, pe, row, T = cl['B'], cl['pe'], cl['row'], cl['T']; return lambda s, tri: (pe[row[s]][tri], B['reward'][row[s]][tri], B['choice'][row[s]][tri], float(T), None)
    B, PE, row, T = cl['B'], cl['PE'], cl['row'], cl['T']; return lambda s, tri: (PE[row[s]][tri], B['reward'][row[s]][tri], B['choice'][row[s]][tri], float(T), B['cond'][row[s]][tri])

def targets(ds, info, beh, s, tri):
    pe, r, c, T, k = beh(s, tri); pe = np.asarray(pe, float); r = np.asarray(r, float); c = np.asarray(c)
    oh = [(c == u).astype(float) for u in np.unique(c)][1:]                     # choice dummies (first level = reference)
    tg = dict(pe_raw=zs(pe), outcome=zs(r), pe_res_outcome=zs(resid(pe, [r])), pe_res_oc=zs(resid(pe, [r] + oh + [o * r for o in oh])), pe_res_full=np.asarray(info['qtil'](s, tri), float))
    if ds in ('ds003458', 'ds004595'):
        tg['pe_res_full_re'] = zs(resid(pe, [r] + oh + [o * r for o in oh] + [(tri / T) ** p for p in (1, 2, 3)]))
    return tg, dict(reward_levels=int(len(np.unique(r))), choice_levels=int(len(np.unique(c))), corr_pe_outcome=float(np.corrcoef(pe, r)[0, 1]),
                    r2_pe_on_outcome=float(1 - resid(pe, [r]).var() / pe.var()), r2_pe_on_full=float(1 - resid(pe, [r] + oh + [o * r for o in oh] + [(tri / T) ** p for p in (1, 2, 3)]).var() / pe.var()))

def subject(ds, info, common, beh, s, out):
    dest = out / f'{ds}_sub-{s:03d}.json'
    if dest.exists(): return
    t0 = time.time(); X, tri, times, _ = D2.epochs(info, s, common); w = (times >= -1e-9) & (times <= 0.60 + 1e-9); Fw = X[:, :, w].reshape(len(X), -1); n = len(tri)
    tg, bstat = targets(ds, info, beh, s, tri); res = dict(code_sha256=CODE_SHA256, dataset=ds, participant=f'sub-{s:03d}', n_trials=int(n), dim=int(Fw.shape[1]), behaviour=bstat, targets={})
    if 'pe_res_full_re' in tg: res['qtil_recompute_max_abs_diff'] = float(np.abs(tg['pe_res_full_re'] - tg['pe_res_full']).max())
    rng = np.random.default_rng(77 + s); nzF = D2.gnoise(Fw.shape, 1000 + s)
    for tn, y in tg.items():
        if tn == 'pe_res_full_re': continue
        Ys = np.column_stack([y] + [np.roll(y, round(n * q / 4)) for q in (1, 2, 3)] + [rng.permutation(y)]); sc = D2.pe_gain(Fw, Ys, LS); nz = D2.pe_gain(nzF, Ys[:, :1], LS)
        res['targets'][tn] = {str(L): dict(gain=float(sc[L][:, 0].mean()), shift_mean=float(sc[L][:, 1:4].mean()), perm=float(sc[L][:, 4].mean()), noise=float(nz[L][:, 0].mean())) for L in LS}
    y = tg['pe_res_full']; u = torch.as_tensor(rng.integers(0, 2, Fw.shape[1]) * 2.0 - 1, device=DEV); Ft = torch.as_tensor(Fw, dtype=torch.float64, device=DEV)
    Fp = Ft + 0.05 * Ft.std(0) * torch.as_tensor(y, device=DEV)[:, None] * u[None]; sp = D2.pe_gain(Fp, y[:, None], LS); so = D2.pe_gain(Fp, tg['outcome'][:, None], LS)
    res['planted'] = {str(L): dict(pe_res_full_gain=float(sp[L][:, 0].mean()), outcome_gain=float(so[L][:, 0].mean())) for L in LS}
    if ds == 'ds003458':                                                        # D2's own G2: x7 window (150-600 ms), registered target
        wx = (times >= 0.15 - 1e-9) & (times <= 0.60 + 1e-9); gx = D2.pe_gain(X[:, :, wx].reshape(n, -1), y[:, None], LS)
        old = json.loads((ROOT / f'results/standard/pe_bottleneck/sub-{s:03d}.json').read_text()); g2 = {}
        for L in LS:
            o = next(r for r in old['rows'] if r['input'] == 'eeg' and r['mode'] == 'pca' and r['L'] == L); g2[str(L)] = dict(saved=o['gain'], this=float(gx[L][:, 0].mean()), abs_diff=abs(o['gain'] - float(gx[L][:, 0].mean())))
        res['gate_x7'] = dict(widths=g2, same_n=bool(old['n_trials'] == n), passed=bool(old['n_trials'] == n and max(v['abs_diff'] for v in g2.values()) < 1e-4))
    if ds == 'ds007647':
        old = json.loads((ROOT / f'results/ext/d2_ds007647_v2/sub-{s:03d}.json').read_text())['subsets']['all']; g = {}
        for a_, b_ in (('pe_res_full', 'pe_resid'), ('pe_raw', 'pe_raw'), ('outcome', 'outcome')):
            for L in ('64', 'full'):
                for q in ('gain', 'shift_mean', 'noise'): g[f'{a_}|{L}|{q}'] = abs(res['targets'][a_][L][q] - old[b_][L][q])
        res['gate_d2v2'] = dict(max_abs_diff=max(g.values()), passed=bool(max(g.values()) < 1e-4))
    res['runtime_s'] = round(time.time() - t0, 1); dest.write_text(json.dumps(res, indent=1))
    print(ds, res['participant'], n, {tn: round(v['64']['gain'], 4) for tn, v in res['targets'].items()}, 'planted', round(res['planted']['64']['pe_res_full_gain'], 3),
          'gate', res.get('gate_x7', res.get('gate_d2v2', {})).get('passed'), 'qtil diff', res.get('qtil_recompute_max_abs_diff'), f"{res['runtime_s']}s", flush=True)

def summarize(out, dss):
    from scipy.stats import t as tdist                                          # scalar quantile lookup only
    S = dict(NOT_CITABLE='until independent review', code_sha256=CODE_SHA256, estimator='d2_pe_tasks.pe_gain, window 0-600 ms; L64 = in-fold PCA-64 ridge (d2 primary), full = full width',
             bootstrap='participant, 10000 resamples, GPU (d2_pe_tasks.boot_ci)', datasets={})
    for ds in dss:
        P = [json.loads(f.read_text()) for f in sorted(out.glob(f'{ds}_sub-*.json'))]
        if not P: continue
        R = S['datasets'][ds] = dict(n=len(P), trials_mean=float(np.mean([p['n_trials'] for p in P])), behaviour={k: float(np.mean([p['behaviour'][k] for p in P])) for k in P[0]['behaviour']}, cells={})
        if 'gate_x7' in P[0]: R['gate_x7_all_pass'] = bool(all(p['gate_x7']['passed'] for p in P)); R['gate_x7_max_abs_diff'] = float(max(v['abs_diff'] for p in P for v in p['gate_x7']['widths'].values())); R['gate_x7_same_n'] = int(sum(p['gate_x7']['same_n'] for p in P))
        if 'gate_d2v2' in P[0]: R['gate_d2v2_all_pass'] = bool(all(p['gate_d2v2']['passed'] for p in P)); R['gate_d2v2_max_abs_diff'] = float(max(p['gate_d2v2']['max_abs_diff'] for p in P))
        if 'qtil_recompute_max_abs_diff' in P[0]: R['qtil_recompute_max_abs_diff'] = float(max(p['qtil_recompute_max_abs_diff'] for p in P))
        for L in ('64', 'full'):
            for tn in P[0]['targets']:
                g = np.array([p['targets'][tn][L]['gain'] for p in P]); fl = np.array([max(p['targets'][tn][L]['shift_mean'], p['targets'][tn][L]['noise']) for p in P]); h = float(tdist.ppf(.975, len(g) - 1)) * g.std(ddof=1) / np.sqrt(len(g))
                R['cells'][f'{tn}|L{L}'] = dict(gain=float(g.mean()), ci95=D2.boot_ci(g), t95=[float(g.mean() - h), float(g.mean() + h)], n_pos=int((g > 0).sum()), gain_minus_floor=float((g - fl).mean()), gain_minus_floor_ci95=D2.boot_ci(g - fl),
                                                n_above_floor=int((g > fl).sum()), shift_floor=float(np.mean([p['targets'][tn][L]['shift_mean'] for p in P])), perm_floor=float(np.mean([p['targets'][tn][L]['perm'] for p in P])),
                                                noise_floor=float(np.mean([p['targets'][tn][L]['noise'] for p in P])))
            G = lambda tn: np.array([p['targets'][tn][L]['gain'] for p in P])
            R[f'corr_participants_pe_raw_vs_outcome|L{L}'] = float(np.corrcoef(G('pe_raw'), G('outcome'))[0, 1])
            d = G('outcome') - G('pe_raw'); R[f'outcome_minus_pe_raw|L{L}'] = dict(d=float(d.mean()), ci95=D2.boot_ci(d), n_pos=int((d > 0).sum()))
            d = G('pe_res_full') - G('pe_raw'); R[f'pe_res_full_minus_pe_raw|L{L}'] = dict(d=float(d.mean()), ci95=D2.boot_ci(d)); R[f'kept_share_res_full_of_raw|L{L}'] = float(G('pe_res_full').mean() / G('pe_raw').mean())
            R[f'planted|L{L}'] = dict(pe_res_full_gain=float(np.mean([p['planted'][L]['pe_res_full_gain'] for p in P])), n_recovered_gt_0p1=int(sum(p['planted'][L]['pe_res_full_gain'] > 0.1 for p in P)),
                                      outcome_gain=float(np.mean([p['planted'][L]['outcome_gain'] for p in P])))
        R['floors_ok_abs_le_0.02'] = bool(all(abs(c[q]) <= 0.02 for c in R['cells'].values() for q in ('shift_floor', 'perm_floor', 'noise_floor')))
    (out / 'summary.json').write_text(json.dumps(S, indent=1)); print(json.dumps(S, indent=1)[:9000])

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--ds', default='ds003458,ds004595,ds007647'); ap.add_argument('--out', required=True); ap.add_argument('--subjects', default=''); ap.add_argument('--summarize', action='store_true')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dss = a.ds.split(',')
    if not a.summarize:
        for ds in dss:
            info = D2.dataset(ds); common = D2.common_channels(info); beh = behaviour(ds, info); subs = info['subs'] if not a.subjects else [s for s in info['subs'] if s in [int(x) for x in a.subjects.split(',')]]
            for s in subs: subject(ds, info, common, beh, s, out)
    summarize(out, dss)
