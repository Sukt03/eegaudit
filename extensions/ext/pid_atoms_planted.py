"""Score-PID atoms (MMI redundancy) from the stored planted_v2 per-participant cells (, 23 Nov 2026; step 1 of IDEATOR_MIXED_ATTRIBUTION). GPU torch.
Per cell (rho, f, L64): Y_N = neural_only, Y_O = ocular_only, Y_NO = mix_all (held-out log-score gain, bits, same scorer/prior).
R = min(Y_N, Y_O); U_N = Y_N - R; U_O = Y_O - R; S = Y_NO - Y_N - Y_O + R. Per participant (seed-mean), participant bootstrap 10000, seed 0. Atoms with a source score <= 0.02 bit are flagged undefined-floor, not clipped.
Oracle atoms from the generator are NOT computed here (needs gf_planted_calib.py): only f_planted and the stored measures are printed beside the atoms."""
import json, glob, sys
import numpy as np, torch
ROOT = '/path/to/eegaudit/results/gpu_fill_20261123'; dev = 'cuda'; G = torch.Generator(device=dev); G.manual_seed(0)
out = {}
for rho in ('0.25', '0.5', '0.9'):
    P = [json.load(open(f))['cells'] for f in sorted(glob.glob(f'{ROOT}/planted_v2_rho{rho}/sub-*.json'))]
    fs = sorted({str(c['f']) for c in P[0]}); out[rho] = {'n': len(P)}
    for f in fs:
        A = []
        for cells in P:
            cs = [c for c in cells if str(c['f']) == f]
            A.append([np.mean([c['neural_only']['64'] for c in cs]), np.mean([c['ocular_only']['64'] for c in cs]), np.mean([c['mix_all']['64'] for c in cs])])
        A = torch.tensor(A, device=dev, dtype=torch.float64); i = torch.randint(0, len(A), (10000, len(A)), device=dev, generator=G)
        def atoms(X):
            yn, yo, yno = X[..., 0], X[..., 1], X[..., 2]; r = torch.minimum(yn, yo); return torch.stack([r, yn - r, yo - r, yno - yn - yo + r, yno], -1)
        pt = atoms(A.mean(0)); bs = atoms(A[i].mean(1)); lo = bs.quantile(0.025, 0); hi = bs.quantile(0.975, 0)
        out[rho][f] = {nm: dict(mean=float(pt[k]), ci95=[float(lo[k]), float(hi[k])]) for k, nm in enumerate(['R_redundant', 'U_neural', 'U_ocular', 'S_synergy', 'Y_both'])}
        out[rho][f]['Y_neural_only'] = float(A[:, 0].mean()); out[rho][f]['Y_ocular_only'] = float(A[:, 1].mean())
        out[rho][f]['undefined_floor'] = bool(min(A[:, 0].mean(), A[:, 1].mean()) <= 0.02)
        print(rho, f, {k: round(v['mean'], 3) for k, v in out[rho][f].items() if isinstance(v, dict)}, 'floor' if out[rho][f]['undefined_floor'] else '', flush=True)
json.dump(out, open(f'{ROOT}/pid_atoms_planted.json', 'w'), indent=1)
