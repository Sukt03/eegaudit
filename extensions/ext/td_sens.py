"""td_sens.py: DESCRIPTIVE sensitivity of the pre-registered F1/F6 numbers (read per-participant files only; decides nothing). Per participant mean |error| of D1, D2, A1, A4, A0 on the six F1 cells of R0 / R2 / R3, and the
same cohort means without the single participant-seed whose stored truth is unstable (participant 2, seed 0, R0). NOT citable."""
import glob, json
import numpy as np
R = '/path/to/eegaudit/results/td_pilot/'
rows = []
for f in sorted(glob.glob(R + 'per_participant/sub-*.json')): rows += [dict(p=json.load(open(f))['participant'], **r) for r in json.load(open(f))['rows']]
six = lambda r: r['rho'] in (0.25, 0.5) and r['f'] in (0.25, 0.5, 0.75) and r['A1'] == r['A1']
out = {}
for fam in ('R0', 'R2', 'R3'):
    S = [r for r in rows if r['family'] == fam and six(r)]; P = sorted({r['p'] for r in S}); print('==', fam, 'n six cells', len(S))
    tab = {}
    for p in P:
        T = [r for r in S if r['p'] == p]; e = lambda g: float(np.mean([abs(g(r)) for r in T])); tab[p] = dict(D1=e(lambda r: r['D1_N_cal'][2] - r['YN']), D2=e(lambda r: r['D2_N_cal'][2] - r['YN']), A1=e(lambda r: r['A1'] - r['YN']), A4=e(lambda r: r['A4'] - r['YN']), truth=float(np.mean([r['YN'] for r in T])))
        print(' sub %02d' % p, ' '.join('%s %.3f' % (k, v) for k, v in tab[p].items()))
    for lab, drop in (('all participants', set()), ('without participant 2 (R0 seed 0 unstable stored truth)', {2})):
        keep = [p for p in P if p not in drop]; print(' ', lab, 'cohort mean:', ' '.join('%s %.4f' % (k, np.mean([tab[p][k] for p in keep])) for k in ('D1', 'D2', 'A1', 'A4')), '| D1/A4 %.2f D1/A1 %.2f' % (np.mean([tab[p]['D1'] for p in keep]) / np.mean([tab[p]['A4'] for p in keep]), np.mean([tab[p]['D1'] for p in keep]) / np.mean([tab[p]['A1'] for p in keep])))
    out[fam] = tab
json.dump(out, open(R + 'sensitivity_by_participant.json', 'w'), indent=1)
