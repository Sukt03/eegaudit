"""B1 (IDEATOR_OTHER_METHODS_20261124, Card 1): sandwich bracket from STORED planted_v2 cells. No refit, no EEG, no new read-out call.
Rule fixed in results/ext/b1_sandwich_bracket/PREREG.json (sha256-stamped before any scored output).
Claim: the cohort-mean neural-only yield lies in [mean mix_eogreg, mean mix_all + 0.02] in every one of the 9 cells RHO {0.25,0.5,0.9} x f {0.25,0.5,0.75}.
Data: results/gpu_fill_20261123/planted_v2_rho{0.25,0.5,0.9}/sub-*.json (23 participants at RHO 0.5, 12 at 0.25 and 0.9), participant value = mean over the 2 seeds,
cohort mean = mean over participants, exactly as summarize() in gf_planted_calib_v2.py.
Usage: python -I -u b1_sandwich_bracket.py --out DIR [--smoke]   (smoke: 2 participants per RHO, writes only {ok, counts}, prints no bracket numbers)"""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np, torch

ROOT = '/path/to/eegaudit'
CODE = os.path.abspath(__file__)
CODE_SHA256 = hashlib.sha256(open(CODE, 'rb').read()).hexdigest()
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
RHOS = ['0.25', '0.5', '0.9']; FS = [0.25, 0.5, 0.75]; LS = ['64', 'full']
SLACK, GATE_TOL, B = 0.02, 1e-4, 10000
t0 = time.time(); log = lambda *m: print(f'[{time.time()-t0:6.1f}s]', *m, flush=True)
J = lambda p: json.load(open(p))
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def load(rho, smoke):
    d = f'{ROOT}/results/gpu_fill_20261123/planted_v2_rho{rho}'
    files = sorted(glob.glob(d + '/sub-*.json'))
    if smoke: files = files[:2]
    return d, files, [J(f) for f in files]


def cohort(R, f, L, key):
    """participant-level mean over seeds, per participant -> list (same as gf_planted_calib_v2.summarize)"""
    return [float(np.mean([c[key][L] for c in r['cells'] if c['f'] == f])) for r in R]


def boot(x, g):
    """participant bootstrap of the mean of a (n,) tensor; returns percentile 2.5/97.5"""
    ix = torch.randint(len(x), (B, len(x)), generator=g, device=DEV)
    m = x[ix].mean(1)
    return [float(m.quantile(.025)), float(m.quantile(.975))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    out = a.out; os.makedirs(out, exist_ok=True)
    pre = J(f'{ROOT}/results/ext/b1_sandwich_bracket/PREREG.json'); st = J(f'{ROOT}/results/ext/b1_sandwich_bracket/PREREG_STAMP.json')
    assert sha(f'{ROOT}/results/ext/b1_sandwich_bracket/PREREG.json') == st['prereg_sha256'], 'PREREG changed after stamp'
    assert st['extra_file_sha256'][CODE] == CODE_SHA256, 'script changed after stamp'
    g = torch.Generator(device=DEV); g.manual_seed(0)
    T = lambda v: torch.tensor(v, dtype=torch.float64, device=DEV)
    res = dict(NOT_CITABLE='until blind review', code_sha256=CODE_SHA256, prereg_sha256=st['prereg_sha256'], rows={}, gates={}, per_participant={}, null_cells={}, ocular_only_in_bracket={})
    gate_diffs = []
    for rho in RHOS:
        d, files, R = load(rho, a.smoke)
        S = J(d + '/summary.json')
        if not a.smoke: assert S['n_participants'] == len(R), (rho, S['n_participants'], len(R))
        for L in LS:
            for f in ['null'] + FS:
                A, E, N, O = (cohort(R, f, L, k) for k in ('mix_all', 'mix_eogreg', 'neural_only', 'ocular_only'))
                for k, v in zip(('mix_all', 'mix_eogreg', 'neural_only', 'ocular_only'), (A, E, N, O)):
                    for fn, pv in zip(files, v): res['per_participant'].setdefault(rho, {}).setdefault(os.path.basename(fn), {}).setdefault(f'L{L}|f={f}', {})[k] = pv
                # gate to stored summary means (not in smoke: the stored means use all participants)
                if not a.smoke:
                    sr = S['rows'][f'L{L}|f={f}']
                    for k, v, sk in (('mix_all', A, 'mean_yield_all'), ('mix_eogreg', E, 'mean_yield_eogreg'), ('neural_only', N, 'mean_neural_only'), ('ocular_only', O, 'mean_ocular_only')):
                        gate_diffs.append(dict(rho=rho, L=L, f=f, arm=k, computed=float(np.mean(v)), stored=sr[sk], absdiff=abs(float(np.mean(v)) - sr[sk])))
                At, Et, Nt, Ot = T(A), T(E), T(N), T(O)
                mA, mE, mN, mO = (float(x.mean()) for x in (At, Et, Nt, Ot))
                lo, hi = mE, mA + SLACK
                if f == 'null':
                    res['null_cells'][f'rho{rho}|L{L}'] = dict(mix_all=mA, mix_eogreg=mE, neural_only=mN, abs_max=max(abs(mA), abs(mE), abs(mN)))
                    continue
                row = dict(n=len(R), mean_mix_all=mA, mean_mix_eogreg=mE, mean_neural_only=mN, mean_ocular_only=mO, lower=lo, upper=hi, width=hi - lo,
                           inside=bool(lo <= mN <= hi), fail_below_lower=bool(mN < lo), fail_above_upper=bool(mN > hi),
                           ci95_mix_eogreg=boot(Et, g), ci95_neural_only=boot(Nt, g), ci95_neural_minus_eogreg=boot(Nt - Et, g), ci95_upper_minus_neural=boot(At + SLACK - Nt, g),
                           eogreg_lower_bound_gt_0=None)
                row['eogreg_lower_bound_gt_0'] = bool(row['ci95_mix_eogreg'][0] > 0)
                # descriptive second lower end: kept ratio (from stored summary) x raw yield (not part of the rule)
                if not a.smoke: row['descriptive_Kz_times_mix_all'] = S['rows'][f'L{L}|f={f}']['kept']['value'] * mA
                res['ocular_only_in_bracket'][f'rho{rho}|L{L}|f={f}'] = bool(lo <= mO <= hi)
                res['rows'][f'rho{rho}|L{L}|f={f}'] = row
    # ---------------- gates
    if not a.smoke:
        mx = max(x['absdiff'] for x in gate_diffs)
        res['gates'] = dict(stored_summary_reproduction_max_absdiff=mx, tol=GATE_TOL, n_compared=len(gate_diffs), pass_=bool(mx <= GATE_TOL))
        res['gates']['all_diffs'] = gate_diffs
    # null cells (descriptive control): |mix_all|,|mix_eogreg| and |neural_only| <= 0.02 at L64 as in the stored null gate
    res['null_cells_ok_le_0.02_L64'] = bool(all(v['abs_max'] <= 0.02 for k, v in res['null_cells'].items() if k.endswith('L64')))
    # ---------------- verdict
    cnt = {L: sum(r['inside'] for k, r in res['rows'].items() if f'|L{L}|' in k) for L in LS}
    n_cells = {L: sum(1 for k in res['rows'] if f'|L{L}|' in k) for L in LS}
    inf = [k for k, r in res['rows'].items() if '|L64|' in k and k.split('|')[0] in ('rho0.25', 'rho0.5') and r['eogreg_lower_bound_gt_0']]
    res['counts'] = dict(inside_L64=cnt['64'], inside_full=cnt['full'], n_cells=n_cells, informative_lower_cells_rho_le_0p5=len(inf), informative_cells=inf)
    res['ocular_only_inside_bracket_count_L64'] = sum(v for k, v in res['ocular_only_in_bracket'].items() if '|L64|' in k)
    res['ocular_only_inside_bracket_count_full'] = sum(v for k, v in res['ocular_only_in_bracket'].items() if '|Lfull|' in k)
    c1, c2, c3 = cnt['64'] == 9 and n_cells['64'] == 9, cnt['full'] >= 8 and n_cells['full'] == 9, len(inf) >= 4
    res['verdict'] = dict(C1_L64_9_of_9=bool(c1), C2_full_ge_8_of_9=bool(c2), C3_informative_lower_ge_4_of_6=bool(c3),
                          PASS=bool(c1 and c2 and c3 and (a.smoke or res['gates']['pass_'])),
                          rho0_9_reported_not_read=dict(width={f: res['rows'][f'rho0.9|L64|f={f}']['width'] for f in FS}, eogreg_ci_lower={f: res['rows'][f'rho0.9|L64|f={f}']['ci95_mix_eogreg'][0] for f in FS}))
    res['runtime_s'] = time.time() - t0
    if a.smoke:
        json.dump(dict(ok=True, smoke=True, n_rows=len(res['rows']), n_null=len(res['null_cells']), runtime_s=res['runtime_s'], code_sha256=CODE_SHA256, dev=DEV), open(out + '/smoke_ok.json', 'w'), indent=1)
        log('smoke ok', len(res['rows']), 'rows, device', DEV); return
    json.dump(res, open(out + '/summary.json', 'w'), indent=1)
    log('counts', json.dumps(res['counts']), 'verdict', json.dumps({k: v for k, v in res['verdict'].items() if k != 'rho0_9_reported_not_read'}), 'gate', res['gates']['stored_summary_reproduction_max_absdiff'])


if __name__ == '__main__':
    main()
