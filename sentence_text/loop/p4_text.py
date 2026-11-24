"""P4 (text): V-information with the contrastive encoder class, within-task pools, from saved test embeddings (contrastive_retrieval.py --save_z).
Per task: S = cos(Z_query, E_candidates) (candidates = that task's test sentences); H_V(T|X) = LOO-temperature softmax cross-entropy per query
(run.soft_bits machinery); H_V(T) = the same for the N(0,1) twin run (the twin is the no-EEG member of the class, as specified) and for shuffled pairs;
I_V = H_V(T)_twin - H_V(T|X)_eeg; PVI per query = CE_twin(q) - CE_eeg(q) paired by record; sentence-cluster bootstrap over the task's sentences.
Also per-participant I_V (records grouped by subject). Output results/loop/p4/p4_text.json."""
import sys, json, numpy as np
sys.path.insert(0, '/path/to/eegaudit/experiments/eegbits'); from run import _TAUS
R = '/path/to/eegaudit/results/loop'; rng = np.random.default_rng(0)
E = np.load(f'{R}/r4_contrastive/minilm_full.npz')['E']; E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
Z = {m: np.load(f'{R}/p4/text/z_{m}_seed0_testZ.npz', allow_pickle=True) for m in ('eeg', 'noise')}
import pickle; recs = [r for r in pickle.load(open('/path/to/eegaudit/results/track_text/brain_translator/cache_full.pkl', 'rb'))['recs'] if r['split'] == 'test']
subj = np.array([r['subject'] for r in recs]); assert len(subj) == len(Z['eeg']['g'])
def ce_rows(S, idx):
    n_q, N = S.shape; rows = np.arange(n_q)
    def ce(tau): A = S / tau; A = A - A.max(1, keepdims=True); lp = A - np.log(np.exp(A).sum(1, keepdims=True)); return -lp[rows, idx] / np.log(2)
    CE = np.stack([ce(t) for t in _TAUS]); tot = CE.sum(1, keepdims=True); loo = (tot - CE) / (n_q - 1); t_idx = loo.argmin(0); return CE[t_idx, rows]
out = {'tasks': {}, 'per_participant': {}}
for task in sorted(set(Z['eeg']['task'].tolist())):
    m = Z['eeg']['task'] == task; g = Z['eeg']['g'][m]; cand = sorted(set(g.tolist())); pos = {c: j for j, c in enumerate(cand)}; idx = np.array([pos[x] for x in g]); C = E[cand]; N = len(cand)
    ce = {}
    for k in ('eeg', 'noise'):
        Q = Z[k]['Z'][m]; Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12); ce[k] = ce_rows(Q @ C.T, idx)
    Qe = Z['eeg']['Z'][m]; Qe = Qe / (np.linalg.norm(Qe, axis=1, keepdims=True) + 1e-12); ce['pairs'] = np.mean([ce_rows(Qe @ C.T, rng.permutation(idx)) for _ in range(10)], 0)
    pvi = ce['noise'] - ce['eeg']; pvi_pairs = ce['pairs'] - ce['eeg']
    clusters = [np.where(idx == j)[0] for j in range(N)]; bs = []
    for _ in range(2000): sel = np.concatenate([clusters[j] for j in rng.integers(0, N, N)]); bs.append(pvi[sel].mean())
    out['tasks'][task] = dict(N=N, n_queries=int(m.sum()), H_cond_eeg=float(ce['eeg'].mean()), H_twin=float(ce['noise'].mean()), H_pairs=float(ce['pairs'].mean()), log2N=float(np.log2(N)),
                              I_V_vs_twin=float(pvi.mean()), I_V_vs_twin_ci95_cluster=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], I_V_vs_pairs=float(pvi_pairs.mean()),
                              I_NCE_eeg=float(np.log2(N) - ce['eeg'].mean()), I_NCE_twin=float(np.log2(N) - ce['noise'].mean()), pvi_median=float(np.median(pvi)), pvi_iqr=[float(np.percentile(pvi, 25)), float(np.percentile(pvi, 75))], frac_pvi_positive=float((pvi > 0).mean()))
    for s in sorted(set(subj[m].tolist())):
        mm = subj[m] == s; out['per_participant'][s] = dict(task=task, n=int(mm.sum()), I_V_vs_twin=float(pvi[mm].mean()), I_NCE_eeg=float(np.log2(N) - ce['eeg'][mm].mean()), I_NCE_twin=float(np.log2(N) - ce['noise'][mm].mean()))
    print(task, {k: round(v, 4) if isinstance(v, float) else v for k, v in out['tasks'][task].items()})
pp = np.array([v['I_V_vs_twin'] for v in out['per_participant'].values()]); out['n_participants_nonneg'] = int((pp >= 0).sum()); out['n_participants'] = len(pp)
json.dump(out, open(f'{R}/p4/p4_text.json', 'w'), indent=1); print('participants with I_V >= 0:', out['n_participants_nonneg'], '/', len(pp))
