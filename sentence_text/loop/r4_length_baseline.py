"""R4 follow-up: how much of the above-floor identification of the contrastive model (and of its N(0,1) twin) is sentence LENGTH?
The twin sees only the padding mask, i.e. the number of fixated words. Length-only retriever: for a test record with n words,
predict the mean MiniLM embedding of all TRAINING sentences whose records have the same word count (nearest count if none);
score exactly as contrastive_retrieval.py (pooled and within task, shuffled-pair floor, bootstrap over queries). No EEG, no training."""
import json, pickle, sys, numpy as np
sys.path.insert(0, '/path/to/eegaudit/experiments/track_text')
from pilot_retrieval import retrieval_scores, wolpaw_bits
R = '/path/to/eegaudit/results'
D = pickle.load(open(f'{R}/track_text/brain_translator/cache_full.pkl', 'rb')); texts, recs = D['texts'], D['recs']
E = np.load(f'{R}/loop/r4_contrastive/minilm_full.npz')['E']; E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
tr = [r for r in recs if r['split'] == 'train']; te = [r for r in recs if r['split'] == 'test']
by_len = {}
for r in tr: by_len.setdefault(r['words'].shape[0], []).append(E[r['g']])
lens = np.array(sorted(by_len)); mean_by_len = {n: np.mean(v, 0) for n, v in by_len.items()}
def pred(n): return mean_by_len[int(lens[np.argmin(np.abs(lens - n))])]
Z = np.stack([pred(r['words'].shape[0]) for r in te]); G = np.array([r['g'] for r in te]); T = np.array([r['task'] for r in te])
rng = np.random.default_rng(0)
def score(Z, G):
    cand = sorted(set(G.tolist())); pos = {g: j for j, g in enumerate(cand)}; idx = np.array([pos[g] for g in G]); C = E[cand]; N = len(cand)
    t1, tw = retrieval_scores(Z, C, idx); sh = [retrieval_scores(Z, C, rng.permutation(idx)) for _ in range(20)]
    P = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12); S = P @ C.T; ts = S[np.arange(len(idx)), idx]; twq = 1 - (S > ts[:, None]).sum(1) / (N - 1)
    bs = [twq[rng.integers(0, len(idx), len(idx))].mean() for _ in range(1000)]
    return dict(n_candidates=N, n_queries=len(G), top1=t1, two_way=tw, bits=wolpaw_bits(t1, N), two_way_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                shuffled_pairs_top1=float(np.mean([s[0] for s in sh])), shuffled_pairs_two_way=float(np.mean([s[1] for s in sh])))
out = {'method': 'length-only retriever (mean MiniLM embedding of training sentences with the same fixated-word count)', 'pooled': score(Z, G), 'within_task': {}}
for task in sorted(set(T.tolist())):
    m = T == task; out['within_task'][task] = score(Z[m], G[m])
# how well does word count alone separate test sentences: same-length pairs among test sentences
tl = {}; [tl.setdefault(r['g'], r['words'].shape[0]) for r in te]
out['test_sentences_sharing_word_count_frac'] = float(np.mean([sum(1 for h in tl.values() if h == v) > 1 for v in tl.values()]))
json.dump(out, open(f'{R}/loop/r4_contrastive/length_only_baseline.json', 'w'), indent=1)
print('pooled', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out['pooled'].items()})
for t, s in out['within_task'].items(): print(t, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
