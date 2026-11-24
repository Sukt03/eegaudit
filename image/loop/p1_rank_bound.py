"""P1 (image): three information bounds on the same rankings. For each (participant, condition) the 200 x 200 cosine matrix S between EEG
embeddings and candidates gives per-query ranks; I_F = Wolpaw bits from top-1; I_NCE = run.soft_bits(S) (LOO temperature); I_R = log2 200 -
H_MM(rank), H_MM = plug-in entropy of the rank histogram + (m - 1)/(2 n ln 2) (Miller-Madow, m occupied bins). Floors: shuffled pairs (rows of
the candidate matrix permuted) under all three. Conditions: NICE-GA seed 141 (10 participants; candidates = the paper's CLIP class centres, as
in I1 / run_acq_controls.load_encoder), k = 80 and k = 1 (5 draws, same rng as retrain_nice_ga_variant); ATM-S released test embeddings (10
participants) vs CLIP ViT-H/14 test features; ceiling = split-half of the trained NICE-GA space (reps 0-39 vs 40-79, cosine, 200-way).
Retrained axes under all three (seed 141): occpar32, w06 (9 participants), occpar16, w04 (3), pca100 (9), with the full checkpoint as reference.
Output results/loop/image_p1/p1_rank_bound.json. Usage: python -I p1_rank_bound.py [--smoke]"""
import sys, json, time, numpy as np, torch
sys.path.insert(0, '/path/to/eegaudit/experiments/eegbits'); sys.path.insert(0, '/path/to/eegaudit/experiments/loop/image')
from run import EMB, CLIP, load_torch, soft_bits, wolpaw_bits, _TAUS
from run_acq_controls import load_encoder
from common_image import load_variant
SMOKE = '--smoke' in sys.argv; dev = torch.device('cuda'); t0 = time.time(); OUT = '/path/to/eegaudit/results/loop/image_p1/p1_rank_bound' + ('_smoke' if SMOKE else '') + '.json'
def norm(A): return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
def h_mm(rank, n_bins):
    c = np.bincount(rank - 1, minlength=n_bins); p = c[c > 0] / c.sum(); return float(-(p * np.log2(p)).sum() + (len(p) - 1) / (2 * len(rank) * np.log(2)))
BINS = [1, 2, 3, 5, 9, 17, 33, 65, 129, 201]   # log2-spaced rank bins: {1},{2},{3,4},{5..8},...,{129..200}: 9 bins
def kl_bins(rank, n):
    """KL(p_obs || p_chance) over log2-spaced rank bins, p_chance = bin width / n (uniform rank); Miller-Madow-type bias (m-1)/(2 n_q ln 2) subtracted.
    Equals log2 n - H(rank) when every bin is a single rank; a coarsening, so a lower bound on the information in the rank."""
    b = np.digitize(rank, BINS) - 1; c = np.bincount(b, minlength=len(BINS) - 1); w = np.diff(BINS) / n; p = c / c.sum(); m = (c > 0).sum()
    return float((p[c > 0] * np.log2(p[c > 0] / w[c > 0])).sum() - (m - 1) / (2 * len(rank) * np.log(2)))
def bounds(S):
    n = S.shape[0]; diag = S[np.arange(n), np.arange(n)]; rank = (S > diag[:, None]).sum(1) + 1
    # I_Rb: information in the binned rank = H(bin under a uniform rank) - H_MM(observed bin); under a uniform rank the bin probabilities are the bin widths / n
    return dict(top1=float((rank == 1).mean()), two_way=float(((S < diag[:, None]).sum(1) / (n - 1)).mean()), I_F=wolpaw_bits(float((rank == 1).mean()), n), I_NCE=soft_bits(S), I_R=float(np.log2(n) - h_mm(rank, n)),
                I_Rb=kl_bins(rank, n), rank_hist=np.bincount(rank - 1, minlength=n).tolist(), mean_rank=float(rank.mean()))
def pvi_stats(S):
    """P4: per-query pointwise V-information under the LOO-temperature softmax of run.soft_bits: PVI_q = log2 N - CE_q(tau_LOO(q)); mean = I_NCE."""
    n = S.shape[0]; rows = np.arange(n)
    def ce_all(tau):
        Z = S / tau; Z = Z - Z.max(1, keepdims=True); lp = Z - np.log(np.exp(Z).sum(1, keepdims=True)); return -lp[rows, rows] / np.log(2)
    CE = np.stack([ce_all(t) for t in _TAUS]); tot = CE.sum(1, keepdims=True); loo = (tot - CE) / (n - 1); t_idx = loo.argmin(0)
    pvi = np.log2(n) - CE[t_idx, rows]
    return dict(mean=float(pvi.mean()), median=float(np.median(pvi)), frac_positive=float((pvi > 0).mean()), q10=float(np.percentile(pvi, 10)), q90=float(np.percentile(pvi, 90)))
def scored(E, C, rng):
    S = norm(E) @ norm(C).T; o = bounds(S); o['pvi'] = pvi_stats(S); perm = rng.permutation(len(C)); o['shuffled_pair'] = {k: v for k, v in bounds(norm(E) @ norm(C[perm]).T).items() if k != 'rank_hist'}; return o
subjects = [f'sub-{i:02d}' for i in range(1, 11)]; res = {'nice_ga': {}, 'atm_native': {}, 'splithalf_trained': {}, 'variants': {}}
subs_use = subjects[:1] if SMOKE else subjects
Yte = load_torch(CLIP / 'ViT-H-14_features_test.pt')
for s in subs_use:
    rng = np.random.default_rng(141 % 1000); embed, Y, tag, X = load_encoder('nice_ga', s, 141, dev); X = np.asarray(X); Y = np.asarray(Y)
    r = {'k80': scored(embed(X.mean(1)), Y, rng)}
    r['k1'] = [scored(embed(X[:, rng.choice(80, 1)].mean(1)), Y, rng) for d in range(5)]
    E1, E2 = embed(X[:, :40].mean(1)), embed(X[:, 40:].mean(1)); r['splithalf'] = scored(E1, E2, rng)
    res['nice_ga'][s] = {'k80': r['k80'], 'k1': r['k1']}; res['splithalf_trained'][s] = r['splithalf']
    Xa = np.asarray(load_torch(EMB / f'ATM_S_eeg_features_{s}_test.pt')); res['atm_native'][s] = scored(Xa, np.asarray(Yte), rng)
    print(f"{s}: NICE-GA k80 I_F {r['k80']['I_F']:.2f} I_NCE {r['k80']['I_NCE']:.2f} I_R {r['k80']['I_R']:.2f} (pair I_R {r['k80']['shuffled_pair']['I_R']:.2f}) | k1 I_R {np.mean([x['I_R'] for x in r['k1']]):.2f} | splithalf I_R {r['splithalf']['I_R']:.2f} | ATM native I_F {res['atm_native'][s]['I_F']:.2f} I_NCE {res['atm_native'][s]['I_NCE']:.2f} I_R {res['atm_native'][s]['I_R']:.2f} ({time.time()-t0:.0f}s)", flush=True)
VAR = {'occpar32': 9, 'w06': 9, 'occpar16': 3, 'w04': 3, 'pca100': 9}
if not SMOKE:
    import glob, re
    for v in VAR:
        res['variants'][v] = {}
        for f in sorted(glob.glob(f'/path/to/eegaudit/results/eegbits/retrain_{v}_sub-*_seed141.json')):
            s = re.search(r'(sub-\d\d)', f).group(1)
            try:
                embed, Y, X, meta = load_variant(v, s, 141, dev); rng = np.random.default_rng(141 % 1000)
                o = {'k80': scored(embed(X.mean(1)), Y, rng)}; o['k1'] = [scored(embed(X[:, rng.choice(80, 1)].mean(1)), Y, rng) for d in range(5)]
                res['variants'][v][s] = o; print(f"  {v} {s}: k80 I_F {o['k80']['I_F']:.2f} I_NCE {o['k80']['I_NCE']:.2f} I_R {o['k80']['I_R']:.2f} ({time.time()-t0:.0f}s)", flush=True)
            except Exception as e: print('  skip', v, s, repr(e)[:120], flush=True)
# ---- aggregate
rng = np.random.default_rng(0)
def agg(vals):
    vals = np.asarray(vals, float); bs = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(4000)]; ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return dict(mean=float(vals.mean()), ci95=ci, n=len(vals), ci_width_over_mean=float((ci[1] - ci[0]) / max(abs(vals.mean()), 1e-9)))
summ = {}
for name, D, key in (('nice_ga_k80', res['nice_ga'], 'k80'), ('atm_native', res['atm_native'], None), ('splithalf_trained', res['splithalf_trained'], None)):
    summ[name] = {m: agg([(d[key] if key else d)[m] for d in D.values()]) for m in ('I_F', 'I_NCE', 'I_R', 'I_Rb', 'top1')}
    summ[name]['shuffled_pair'] = {m: float(np.mean([(d[key] if key else d)['shuffled_pair'][m] for d in D.values()])) for m in ('I_F', 'I_NCE', 'I_R', 'I_Rb')}
summ['nice_ga_k1'] = {m: agg([np.mean([x[m] for x in d['k1']]) for d in res['nice_ga'].values()]) for m in ('I_F', 'I_NCE', 'I_R', 'I_Rb', 'top1')}
if res['variants']:
    full = res['nice_ga']; summ['variant_shares'] = {}
    for v, D in res['variants'].items():
        sh = {m: [D[s]['k80'][m] / full[s]['k80'][m] for s in D if s in full and full[s]['k80'][m] > 0] for m in ('I_F', 'I_NCE', 'I_R', 'I_Rb')}
        summ['variant_shares'][v] = {m: dict(mean=float(np.mean(x)), min=float(np.min(x)), max=float(np.max(x)), n=len(x)) for m, x in sh.items() if x}
    axes = {'electrodes': ['full', 'occpar32', 'occpar16'], 'window': ['full', 'w06', 'w04'], 'input_rank': ['full', 'pca100']}
    summ['order_check'] = {}
    for ax, rungs in axes.items():
        for m in ('I_NCE', 'I_R', 'I_Rb'):
            vals = [1.0 if r == 'full' else summ['variant_shares'].get(r, {}).get(m, {}).get('mean', np.nan) for r in rungs]
            summ['order_check'][f'{ax}_{m}'] = dict(rungs=rungs, shares=vals, monotone_nonincreasing=bool(all(np.nan_to_num(vals[i], nan=-1) >= np.nan_to_num(vals[i + 1], nan=-1) - 1e-9 for i in range(len(vals) - 1))))
    summ['order_same_I_R_vs_I_NCE'] = all(summ['order_check'][f'{ax}_I_R']['monotone_nonincreasing'] == summ['order_check'][f'{ax}_I_NCE']['monotone_nonincreasing'] for ax in axes)
k = summ['nice_ga_k80']; fl = summ['nice_ga_k80']['shuffled_pair']
summ['pass_rule'] = dict(cv_I_R=k['I_R']['ci_width_over_mean'], cv_I_Rb=k['I_Rb']['ci_width_over_mean'], cv_I_NCE=k['I_NCE']['ci_width_over_mean'], floor_I_R=fl['I_R'], floor_I_Rb=fl['I_Rb'], floor_I_NCE=fl['I_NCE'],
                         I_R_floor_at_chance=bool(abs(fl['I_R']) < 0.05), I_Rb_floor_at_chance=bool(abs(fl['I_Rb']) < 0.05), order_same=summ.get('order_same_I_R_vs_I_NCE'))
summ['pass_rule']['headline'] = 'I_Rb' if (summ['pass_rule']['I_Rb_floor_at_chance'] and k['I_Rb']['ci_width_over_mean'] <= k['I_NCE']['ci_width_over_mean'] and summ.get('order_same_I_R_vs_I_NCE')) else 'I_NCE'
json.dump({'summary': summ, 'per_participant': res, 'runtime_s': time.time() - t0}, open(OUT, 'w'), indent=1)
print(json.dumps({k: v for k, v in summ.items() if k != 'variant_shares'}, indent=0)[:3000]); print('done', time.time() - t0)
