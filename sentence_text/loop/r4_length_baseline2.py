"""Stronger length-only predictors for R4 (no EEG): (a) ridge from a polynomial basis of the word count (n, n^2 .. n^5, log n, standardized)
to the MiniLM embedding, fit on training records, alpha by dev MRR-free rule (fixed grid, pick by dev 2-way); (b) kNN-in-length window mean
(training sentences with |count - n| <= w, w in 1,2,3). Scoring as contrastive_retrieval.py (pooled and within task)."""
import json, pickle, sys, numpy as np
sys.path.insert(0, '/path/to/eegaudit/experiments/track_text')
from pilot_retrieval import retrieval_scores, wolpaw_bits
R = '/path/to/eegaudit/results'
D = pickle.load(open(f'{R}/track_text/brain_translator/cache_full.pkl', 'rb')); recs = D['recs']
E = np.load(f'{R}/loop/r4_contrastive/minilm_full.npz')['E']; E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
sp = {s: [r for r in recs if r['split'] == s] for s in ('train', 'dev', 'test')}
n_of = lambda rs: np.array([r['words'].shape[0] for r in rs], float)
def basis(n): 
    n = np.asarray(n, float); return np.column_stack([n ** k for k in range(1, 6)] + [np.log(n)])
Xtr = basis(n_of(sp['train'])); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Ytr = E[[r['g'] for r in sp['train']]]
def ridge_pred(alpha, rs):
    A = (Xtr - mu) / sd; A1 = np.c_[np.ones(len(A)), A]; W = np.linalg.solve(A1.T @ A1 + alpha * np.eye(A1.shape[1]), A1.T @ Ytr)
    B = (basis(n_of(rs)) - mu) / sd; return np.c_[np.ones(len(B)), B] @ W
rng = np.random.default_rng(0)
def score(Z, rs, perm=True):
    G = np.array([r['g'] for r in rs]); cand = sorted(set(G.tolist())); pos = {g: j for j, g in enumerate(cand)}; idx = np.array([pos[g] for g in G]); C = E[cand]; N = len(cand)
    t1, tw = retrieval_scores(Z, C, idx); out = dict(n_candidates=N, n_queries=len(G), top1=t1, two_way=tw, bits=wolpaw_bits(t1, N))
    if perm:
        sh = [retrieval_scores(Z, C, rng.permutation(idx)) for _ in range(20)]; out.update(shuffled_pairs_two_way=float(np.mean([s[1] for s in sh])))
        P = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12); S = P @ C.T; ts = S[np.arange(len(idx)), idx]; twq = 1 - (S > ts[:, None]).sum(1) / (N - 1)
        bs = [twq[rng.integers(0, len(idx), len(idx))].mean() for _ in range(1000)]; out['two_way_ci95'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out
res = {}
# (a) ridge on length basis, alpha picked on dev two-way
best = max([(score(ridge_pred(al, sp['dev']), sp['dev'], False)['two_way'], al) for al in [1e-2, 1e-1, 1, 10, 100, 1000]])
Z = ridge_pred(best[1], sp['test']); res['ridge_length_poly'] = {'alpha': best[1], 'dev_two_way': best[0], 'pooled': score(Z, sp['test'])}
T = np.array([r['task'] for r in sp['test']])
res['ridge_length_poly']['within_task'] = {t: score(Z[T == t], [r for r, m in zip(sp['test'], T == t) if m]) for t in sorted(set(T))}
# (b) window kNN in length
ntr = n_of(sp['train'])
for w in (1, 2, 3):
    Zw = np.stack([Ytr[np.abs(ntr - n) <= w].mean(0) if (np.abs(ntr - n) <= w).any() else Ytr.mean(0) for n in n_of(sp['test'])])
    res[f'window_knn_w{w}'] = {'pooled': score(Zw, sp['test']), 'within_task': {t: score(Zw[T == t], [r for r, m in zip(sp['test'], T == t) if m]) for t in sorted(set(T))}}
json.dump(res, open(f'{R}/loop/r4_contrastive/length_only_baseline2.json', 'w'), indent=1)
for k, v in res.items():
    print(k, 'pooled 2way %.4f [%.3f,%.3f] top1 %.4f' % (v['pooled']['two_way'], *v['pooled']['two_way_ci95'], v['pooled']['top1']), {t: round(s['two_way'], 4) for t, s in v['within_task'].items()})
