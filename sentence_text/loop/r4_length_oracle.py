"""Upper bound of a length-only retriever on the R4 test set: the per-length predictor is fitted on the TEST records themselves
(mean MiniLM embedding of the test sentences of records with that fixated-word count), then scored on the test set. Any map from
word count to an embedding, however nonlinear, cannot exceed this on the pooled set (it uses test labels)."""
import json, pickle, sys, numpy as np
sys.path.insert(0, '/path/to/eegaudit/experiments/track_text')
from pilot_retrieval import retrieval_scores
R = '/path/to/eegaudit/results'
D = pickle.load(open(f'{R}/track_text/brain_translator/cache_full.pkl', 'rb')); recs = D['recs']
E = np.load(f'{R}/loop/r4_contrastive/minilm_full.npz')['E']; E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
te = [r for r in recs if r['split'] == 'test']; n = np.array([r['words'].shape[0] for r in te]); G = np.array([r['g'] for r in te]); T = np.array([r['task'] for r in te])
def score(mask):
    g = G[mask]; nn_ = n[mask]; cand = sorted(set(g.tolist())); pos = {c: j for j, c in enumerate(cand)}; idx = np.array([pos[c] for c in g])
    Z = np.stack([E[g[nn_ == k]].mean(0) for k in nn_]); t1, tw = retrieval_scores(Z, E[cand], idx); return dict(N=len(cand), top1=t1, two_way=tw)
out = {'pooled': score(np.ones(len(te), bool)), 'within_task': {t: score(T == t) for t in sorted(set(T))}}
# also: how many distinct word counts per task, and the spread of n within a sentence across subjects
out['n_distinct_counts'] = {t: int(len(set(n[T == t].tolist()))) for t in sorted(set(T))}
out['within_sentence_count_sd_mean'] = float(np.mean([n[G == g].std() for g in set(G.tolist())]))
print(json.dumps(out)); json.dump(out, open(f'{R}/loop/r4_contrastive/length_oracle_bound.json', 'w'), indent=1)
