"""EEG-minus-noise-twin difference in free-running sentence identification for the autoregressive models (BrainTranslator, DeWave-lite),
with a CLUSTER bootstrap over test sentences (all records of a sentence resampled together; 1349 records of 103 sentences are not
independent queries). Re-embeds the saved generations with MiniLM; the identification is the one of train_bt.py / train_dewave.py.
Usage: python -I text_cluster_bootstrap.py <eeg_generations.jsonl> <noise_generations.jsonl> <out.json>"""
import json, sys, pickle, numpy as np
sys.path.insert(0, '/path/to/eegaudit/experiments/track_text')
from pilot_retrieval import embed_sentences
R = '/path/to/eegaudit/results'; MINILM = '/path/to/eegaudit/data_raw/zuco/minilm'
f_eeg, f_noise, f_out = sys.argv[1:4]
rows = {k: [json.loads(l) for l in open(f)] for k, f in (('eeg', f_eeg), ('noise', f_noise))}
assert [r['g'] for r in rows['eeg']] == [r['g'] for r in rows['noise']]
g = np.array([r['g'] for r in rows['eeg']]); cand = sorted(set(g.tolist())); pos = {c: j for j, c in enumerate(cand)}; idx = np.array([pos[x] for x in g]); N = len(cand)
texts = {r['g']: r['target'] for r in rows['eeg']}
E_c, _ = embed_sentences([texts[c] for c in cand], MINILM); E_c = E_c / np.linalg.norm(E_c, axis=1, keepdims=True)
def per_query(hyps):
    Q, _ = embed_sentences([h if h else '.' for h in hyps], MINILM); Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    S = Q @ E_c.T; ts = S[np.arange(len(idx)), idx]; beat = (S > ts[:, None]).sum(1)
    return (beat == 0).astype(float), 1 - beat / (N - 1)
rng = np.random.default_rng(0); out = {'n_queries': len(g), 'n_sentences': N, 'modes': {}}
clusters = [np.where(idx == j)[0] for j in range(N)]
for mode in ('teacher_forced', 'greedy', 'beam'):
    pq = {k: per_query([r[mode] for r in rows[k]]) for k in rows}
    d_top1 = pq['eeg'][0].mean() - pq['noise'][0].mean(); d_two = pq['eeg'][1].mean() - pq['noise'][1].mean()
    bt, bw = [], []
    for _ in range(2000):
        sel = np.concatenate([clusters[j] for j in rng.integers(0, N, N)])
        bt.append(pq['eeg'][0][sel].mean() - pq['noise'][0][sel].mean()); bw.append(pq['eeg'][1][sel].mean() - pq['noise'][1][sel].mean())
    out['modes'][mode] = {k: {'top1': float(pq[k][0].mean()), 'two_way': float(pq[k][1].mean())} for k in rows}
    out['modes'][mode].update(eeg_minus_noise_top1=float(d_top1), eeg_minus_noise_top1_ci95=[float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                              eeg_minus_noise_two_way=float(d_two), eeg_minus_noise_two_way_ci95=[float(np.percentile(bw, 2.5)), float(np.percentile(bw, 97.5))])
    # length confound: correlation between generated-length and target-length per arm, and 2-way of a length-only guess is in r4_length_baseline
    out['modes'][mode]['corr_len_hyp_target'] = {k: float(np.corrcoef([len(r[mode].split()) for r in rows[k]], [len(r['target'].split()) for r in rows[k]])[0, 1]) for k in rows}
    print(mode, json.dumps({k: v for k, v in out['modes'][mode].items()}))
json.dump(out, open(f_out, 'w'), indent=1)
