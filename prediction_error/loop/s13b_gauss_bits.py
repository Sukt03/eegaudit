"""S13b: Gaussianised-target coupling rho_g with the same session-permutation null correction as group_stats.py, both datasets, one JSON.
Per participant i and decoder: read-out P (LOPO single-column read-outs used as they are; feature decoders: the per-target PCA-64 ridge
cv_predict_gpu of group_stats.py trained on the RAW targets, exactly as the paper's rho). rho_g,i = corr(P_i, normal-scores(y_i));
exact null: the same read-out against the normal-scores of every other participant j's residualised PE on the same trial indices
(r_g,ij); d_i = atanh(r_g,i) - mean_j atanh(r_g,ij), v_i as group_stats; DerSimonian-Laird pooled rho_g (rho_g_ci); bits_g = -1/2 log2(1 - rho^2)
at the pooled value and as the mean of per-participant null-corrected bits with a participant bootstrap CI; floor_bits_g = bits at the 95th
percentile of all null r_g; n_above_floor = participants whose r_g exceeds the 95th percentile of their own null.
Output results/loop/learner/measure_bits_gauss.json: datasets.<ds>.decoders.<name>.{rho_g, rho_g_ci, bits_g, bits_g_ci, bits_g_mean_pp,
bits_g_mean_pp_ci, floor_bits_g, floor_p95_r_g, n_above_floor, n, rho_raw_recomputed, bits_raw_recomputed, per_participant_rho_g, per_participant_d}.
Usage (repo root): python -I experiments/loop/s13b_gauss_bits.py [--smoke]"""
import sys, json, time, numpy as np, torch
from pathlib import Path
from scipy import stats
ROOT = Path('/path/to/eegaudit'); sys.path.insert(0, str(ROOT))
from group_stats import cv_predict_gpu, dersimonian_laird, resid, corr, QUANT
from cogmodels import models as M

from belief_bottleneck import LAM, folds
from group_stats import DEV
def cv_predict_fast(F, Y, L=64, colw=None):
    """Identical to group_stats.cv_predict_gpu (same folds, standardisation, colw, PCA-L, ridge LAM) but the PCA subspace is taken from the
    n x n Gram matrix when D > 3000 (exact: V_k = A_c^T U_k / S_k), so D = 32768 costs matmuls instead of a 430 x 32768 SVD per fold."""
    F = torch.tensor(F, dtype=torch.float64, device=DEV); Y = torch.tensor(Y, dtype=torch.float64, device=DEV)
    cw = None if colw is None else torch.tensor(colw, dtype=torch.float64, device=DEV)
    f = torch.tensor(folds(len(F)), device=DEV); P = torch.zeros_like(Y)
    for b in range(10):
        tr, te = f != b, f == b
        mu, sd = F[tr].mean(0), F[tr].std(0, unbiased=False) + 1e-6
        A, B = (F[tr] - mu) / sd, (F[te] - mu) / sd
        if cw is not None: A, B = A * cw, B * cw
        m = A.mean(0); Ac = A - m
        if A.shape[1] > 3000:
            K = Ac @ Ac.T; ev, U = torch.linalg.eigh(K); idx = torch.argsort(ev, descending=True)[:min(L, len(ev))]
            S = torch.sqrt(ev[idx].clamp(min=1e-12)); Uk = U[:, idx]
            Atr = Uk * S + ((m @ Ac.T) @ Uk / S)[None]; Ate = (B @ Ac.T) @ Uk / S
        else:
            _, _, Vt = torch.linalg.svd(Ac, full_matrices=False); Vk = Vt[:min(L, Vt.shape[0])].T; Atr, Ate = A @ Vk, B @ Vk
        k = Atr.shape[1]
        W = torch.linalg.solve(Atr.T @ Atr + LAM * len(Atr) * torch.eye(k, dtype=torch.float64, device=DEV), Atr.T @ (Y[tr] - Y[tr].mean(0)))
        P[te] = Ate @ W
    return P.cpu().numpy()
SMOKE = '--smoke' in sys.argv; t0 = time.time(); rng = np.random.default_rng(0)
b = lambda r: float(-0.5 * np.log2(1 - min(abs(float(r)), 0.999) ** 2))
nscore = lambda y: stats.norm.ppf((stats.rankdata(y) - 0.5) / len(y))
def targets_ds003458():
    Q = np.load(ROOT / QUANT); model = M.EXTRA_MODELS['RWD']; Z = torch.tensor(np.load(ROOT / 'results/map_RWD.npy')).float(); probs = M.reward_schedule(480); nS = Z.shape[0]
    full_q = []
    for i in range(nS):
        with torch.no_grad():
            full_q.append(M.run(model, Z[i][None], probs, torch.tensor(Q['choice'][i])[None], torch.tensor(Q['reward'][i])[None], torch.tensor(Q['missing'][i])[None], record=True)['q']['pe'][0].numpy())
    def qtil(j, tri):   # j = participant label (1-based)
        j = j - 1; c, r = Q['choice'][j][tri], Q['reward'][j][tri]
        cols = [r] + [(c == k).astype(float) for k in range(3)] + [(c == k).astype(float) * r for k in range(3)] + [(tri / 480.0) ** p for p in (1, 2, 3)]
        v = resid(full_q[j][tri].astype(float), cols); return (v - v.mean()) / (v.std() + 1e-9)
    subs = list(range(1, nS + 1)); return qtil, subs, (lambda dec, s: ROOT / f'results/features/{dec}/sub-{s:03d}.npz'), {s: i for i, s in enumerate(subs)}
def targets_ds004595():
    import deep_decoders_ds004595 as D2
    B = D2.B; model = M.EXTRA_MODELS['RWD']; Z = torch.tensor(np.load(ROOT / 'results/ds004595/map_RWD.npy')).float(); T = int(B['T'])
    with torch.no_grad():
        pe = D2.run_reset(model, Z, torch.full((3, T), 0.5), torch.tensor(B['resets']).float(), torch.tensor(B['choice']), torch.tensor(B['reward']).float(), torch.tensor(B['missing']), record=True)['q']['pe'].numpy()
    row = {int(s): i for i, s in enumerate(B['sub'])}
    def qtil(j, tri):   # j = participant label
        i = row[j]; c, r = B['choice'][i][tri], B['reward'][i][tri]; oh = np.eye(3)[c]
        C = np.column_stack([np.ones(len(tri)), r, oh, oh * r[:, None], tri / T, (tri / T) ** 2, (tri / T) ** 3])
        y = D2.resid(pe[i][tri].astype(float), C); return (y - y.mean()) / (y.std() + 1e-9)
    subs = [int(s) for s in D2.SUBS]; return qtil, subs, (lambda dec, s: ROOT / f'results/features_ds004595/{dec}/sub-{s:03d}.npz'), None
DATASETS = {'ds003458': (targets_ds003458, ['eegnet', 'raw', 'revetok', 'reve_lopo', 'cbramod_ft', 'fmtok', 'fm', 'labram', 'eegpt', 'ens', 'eegnet_post', 'p3_lopo', 'rewp_lopo']),
            'ds004595': (targets_ds004595, ['eegnet', 'raw', 'ens', 'eegnet_post'])}
if SMOKE: DATASETS = {'ds003458': (targets_ds003458, ['eegnet', 'raw']), 'ds004595': (targets_ds004595, ['eegnet'])}
out = {'formula': 'bits_g = -0.5*log2(1-rho_g^2); rho_g = corr(read-out, normal-scores(residualised PE)); null correction and pooling as group_stats.py (session-permutation null = other participants\' targets on the same trials)',
       'generated_by': 'experiments/loop/s13b_gauss_bits.py', 'datasets': {}}
for ds, (tf, decs) in DATASETS.items():
    qtil, subs, fpath, _ = tf(); 
    if SMOKE: subs_use = subs[:4]
    else: subs_use = subs
    Ycache = {}
    out['datasets'][ds] = {'n_participants': len(subs_use), 'decoders': {}}
    for dec in decs:
        R, Rraw, d, v, NL = [], [], [], [], []
        try:
            for s in subs_use:
                fz = np.load(fpath(dec, s)); F, colw, tri = fz['F'].astype(np.float64), fz['colw'].astype(np.float64), fz['tri']
                key = (s, tuple(tri[:5]), len(tri))
                if key not in Ycache: Ycache[key] = np.column_stack([qtil(j, tri) for j in subs])
                Y = Ycache[key]; Yg = np.column_stack([nscore(Y[:, j]) for j in range(Y.shape[1])]); me = subs.index(s)
                if F.shape[1] == 1: P = np.repeat(F, Y.shape[1], axis=1)
                else: P = cv_predict_fast(F, Y, 64, colw=colw)
                rg = np.array([corr(P[:, j], Yg[:, j]) for j in range(Y.shape[1])]); rr = corr(P[:, me], Y[:, me])
                r = rg[me]; nl = np.array([rg[j] for j in range(Y.shape[1]) if j != me]); zn = np.arctanh(np.clip(nl, -0.999, 0.999))
                R.append(float(r)); Rraw.append(float(rr)); NL.append(nl); d.append(float(np.arctanh(np.clip(r, -0.999, 0.999)) - zn.mean())); v.append(float(zn.var(ddof=1) * (1 + 1 / len(zn))))
        except FileNotFoundError as e:
            print(ds, dec, 'missing', e); continue
        d, v = np.array(d), np.array(v); meta = dersimonian_laird(d, v); allnull = np.concatenate(NL)
        pp_bits = np.array([b(np.tanh(x)) for x in d]); bs = [pp_bits[rng.integers(0, len(pp_bits), len(pp_bits))].mean() for _ in range(4000)]
        p95 = float(np.quantile(allnull, 0.95))
        out['datasets'][ds]['decoders'][dec] = dict(n=len(d), rho_g=float(np.tanh(meta['mu'])), rho_g_ci=[float(np.tanh(x)) for x in meta['ci']], bits_g=b(np.tanh(meta['mu'])), bits_g_ci=[b(np.tanh(x)) for x in meta['ci']],
            bits_g_mean_pp=float(pp_bits.mean()), bits_g_mean_pp_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], bits_g_median_pp=float(np.median(pp_bits)),
            floor_p95_r_g=p95, floor_bits_g=b(p95), null_mean_r_g=float(allnull.mean()), n_null=int(len(allnull)),
            n_above_floor=int(sum(R[i] > np.quantile(NL[i], 0.95) for i in range(len(R)))), p_meta=meta['p'], tau2=meta['tau2'],
            rho_raw_recomputed_fisher_mean=float(np.tanh(np.arctanh(np.clip(Rraw, -0.999, 0.999)).mean())), bits_raw_recomputed=b(np.tanh(np.arctanh(np.clip(Rraw, -0.999, 0.999)).mean())),
            per_participant_rho_g=[round(x, 4) for x in R], per_participant_d=[round(x, 4) for x in d.tolist()])
        o = out['datasets'][ds]['decoders'][dec]
        print(f"{ds} {dec:12s} n={o['n']} rho_g {o['rho_g']:.3f} [{o['rho_g_ci'][0]:.3f},{o['rho_g_ci'][1]:.3f}] bits_g {o['bits_g']:.3f} [{o['bits_g_ci'][0]:.3f},{o['bits_g_ci'][1]:.3f}] mean_pp {o['bits_g_mean_pp']:.3f} [{o['bits_g_mean_pp_ci'][0]:.3f},{o['bits_g_mean_pp_ci'][1]:.3f}] floor {o['floor_bits_g']:.4f} above {o['n_above_floor']}/{o['n']} | raw-target rho {o['rho_raw_recomputed_fisher_mean']:.3f} ({time.time()-t0:.0f}s)", flush=True)
json.dump(out, open(ROOT / ('results/loop/learner/measure_bits_gauss' + ('_smoke' if SMOKE else '') + '.json'), 'w'), indent=1)
print('done', time.time() - t0)
