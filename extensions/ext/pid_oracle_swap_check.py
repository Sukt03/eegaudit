import sys, json
sys.path.insert(0, '/path/to/eegaudit/experiments/ext')
import pid_oracle_swap as P
G = P.G; D2 = P.D2
S = json.load(open(P.RES + '/pid3/summary.json'))
print('LAM', D2.LAM, 'eta', P.ETA)
for r in ('0.25', '0.5', '0.9'):
    c = S['cells'][f'rho{r}|f=0.0']['hat']; print('C2', r, 'R', c['R'], 'U_O', c['U_O'])
print('C1 closed', json.dumps({k: {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()} for k, v in S['criteria']['C1_oracle_closed']['cells'].items()}))
for k in ('rho0.5|f=0.5', 'rho0.9|f=0.5', 'rho0.25|f=0.5'):
    for kind in ('swapO_matched', 'swapO_unmatched'):
        d = S['cells'][k]['dose'][kind]; print(k, kind, 'Yhat', [round(x, 3) for x in d['Y_hat']], 'Yclosed', [round(x, 3) for x in d['Y_oracle_closed']], 'slope ci', [round(x, 3) for x in d['slope_hat_ci95']], 'tf', round(d['slope_trialfrac_hat'], 3), round(d['trialfrac_planted_oracle_closed'], 3))
    h = S['cells'][k]['hat']; print('   atoms hat U_O', round(h['U_O']['mean'], 3), 'R', round(h['R']['mean'], 3), 'Y_O', round(h['Y_O'], 3))
print('C4 unmatched trialfrac closed', json.dumps({k: round(v['ratio'], 2) for k, v in S['criteria']['C4_swapO_unmatched_oracle_closed']['trialfrac_cells'].items()}))
print('C4 matched trialfrac closed', json.dumps({k: round(v['ratio'], 2) for k, v in S['criteria']['C4_swapO_matched_oracle_closed']['trialfrac_cells'].items()}))
# slope code check: synthetic read-out with LAM = 0 must reproduce the closed-form slopes
d = G.load(1, 0.25, 0); lam, uz, vz = P.T(d['lam']), P.T(d['uz']), P.T(d['vz'])
for L in (D2.LAM, 0.0):
    D2.LAM = L; syn = P.oracle_synth(lam, uz, vz, 0.25, 123)
    for f in (0.25, 0.5, 0.75):
        a, b = G.amp(f); oc = P.oracle_closed(lam, uz, vz, a, b, 0.25)
        print('LAM', L, 'f', f, {k: (round(oc[k]['slope0'], 4), round(syn[f][k]['slope0'], 4)) for k in ('swapO_matched', 'swapO_unmatched', 'swapN_matched')}, 'I_NO', round(oc['I_NO'], 4), round(syn[f]['I_NO'], 4))
print('lam PCs 20,35,51,63', [round(float(lam[i]), 3) for i in (20, 35, 51, 63)])
