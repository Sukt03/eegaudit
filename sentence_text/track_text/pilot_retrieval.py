"""TRACK 3 (EEG->text) pilot: sentence-retrieval price list on ZuCo word-level EEG features (Wang & Ji 2022 pickles).

Query = EEG for a held-out sentence (mean or concatenation over its first k fixated words, or the sentence-level mean
feature), averaged over m subjects. Decoder = ridge regression EEG -> sentence embedding (MiniLM mean-pooled, or TF-IDF/LSA
fallback), nearest candidate by cosine among the N held-out test sentences (pooled over tasks). Controls at every point:
noise input (train+test N(0,1), Jo et al. 2024), trial-shuffled training, shuffled pairs. Output: PROTOCOL.md JSON schema.
Run with python -I inside the GPU node (CPU only is enough)."""
import argparse, json, os, pickle, sys, time
import numpy as np
from sklearn.linear_model import Ridge

BANDS = ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2']
TASKS = {'SR': 'task1-SR-dataset.pickle', 'NR': 'task2-NR-dataset.pickle', 'NR2': 'task2-NR-2.0-dataset.pickle'}
WORD_KS = [1, 2, 4, 8, 'all']
SUBJ_MS = [1, 3, 'all']

def wolpaw_bits(P, N):
    """Wolpaw information per query; 0 at or below chance (the formula rises again for P < 1/N, which is not information)."""
    if P <= 1.0 / N:
        return 0.0
    P = min(P, 1 - 1e-9)
    return np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1))

def word_vec(w, eeg_type='GD'):
    try:
        parts = [np.asarray(w['word_level_EEG'][eeg_type][eeg_type + b], dtype=np.float64).ravel() for b in BANDS]
    except KeyError:
        return None
    v = np.concatenate(parts)
    if v.size != 840 or not np.all(np.isfinite(v)):
        return None
    return v

def sent_vec(s):
    try:
        v = np.concatenate([np.asarray(s['sentence_level_EEG']['mean' + b], dtype=np.float64).ravel() for b in BANDS])
    except KeyError:
        return None
    return v if (v.size == 840 and np.all(np.isfinite(v))) else None

def load_task(path, task, max_subjects=None):
    """Return sentences (list of text in index order), split per index, and records: dict (subject, idx) -> (word_vecs list, sent_vec)."""
    d = pickle.load(open(path, 'rb'))
    subjects = sorted(d.keys())
    if max_subjects:
        subjects = subjects[:max_subjects]
    n = max(len(d[s]) for s in subjects)
    texts = [None] * n
    recs = {}
    for sj in subjects:
        for i, s in enumerate(d[sj]):
            if s is None:
                continue
            if texts[i] is None:
                texts[i] = s['content'].strip()
            wv = [v for v in (word_vec(w) for w in s['word']) if v is not None]
            sv = sent_vec(s)
            if len(wv) == 0 or sv is None:
                continue
            recs[(sj, i)] = (np.stack(wv), sv)
    keep = [i for i in range(n) if texts[i] is not None]
    # Wang & Ji split: first 80% of sentence indices train, next 10% dev, last 10% test (unique sentences, no overlap)
    n_k = len(keep); tr = keep[: int(0.8 * n_k)]; dv = keep[int(0.8 * n_k): int(0.9 * n_k)]; te = keep[int(0.9 * n_k):]
    split = {i: 'train' for i in tr}; split.update({i: 'dev' for i in dv}); split.update({i: 'test' for i in te})
    return texts, split, recs, subjects

def feat(wv, sv, k, mode):
    if mode == 'sentlevel':
        return sv
    W = wv if k == 'all' else wv[:k]
    if mode == 'mean':
        return W.mean(0)
    kk = k if k != 'all' else wv.shape[0]
    out = np.zeros(840 * kk); out[: 840 * W.shape[0]] = W.ravel(); return out

def embed_sentences(texts, minilm_dir):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(minilm_dir); mdl = AutoModel.from_pretrained(minilm_dir).eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=128, return_tensors='pt')
                h = mdl(**b).last_hidden_state; m = b['attention_mask'].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1); out.append(torch.nn.functional.normalize(e, dim=1).numpy())
        return np.concatenate(out), 'all-MiniLM-L6-v2 (mean pooling, transformers)'
    except Exception as e:
        print('MiniLM failed, falling back to TF-IDF/LSA:', repr(e), flush=True)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        X = TfidfVectorizer(sublinear_tf=True).fit_transform(texts)
        Z = TruncatedSVD(100, random_state=0).fit_transform(X)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
        return Z, 'TF-IDF + LSA(100) on ZuCo sentences'

def retrieval_scores(pred, cand, true_idx):
    """pred: (Q, D) predicted embeddings; cand: (N, D) candidate embeddings; true_idx: (Q,) index into cand."""
    P = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12)
    S = P @ cand.T
    true_s = S[np.arange(len(true_idx)), true_idx]
    rank = (S > true_s[:, None]).sum(1)  # number of candidates beating the truth
    N = cand.shape[0]
    return float((rank == 0).mean()), float((1 - rank / (N - 1)).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/path/to/eegaudit/data_raw/zuco/eeg2text_pickles')
    ap.add_argument('--minilm', default='/path/to/eegaudit/data_raw/zuco/minilm')
    ap.add_argument('--times', default='/path/to/eegaudit/results/track_text/reading_times.json')
    ap.add_argument('--out', default='/path/to/eegaudit/results/track_text')
    ap.add_argument('--tasks', default='SR,NR,NR2')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--modes', default='mean,concat,sentlevel')
    ap.add_argument('--smoke', action='store_true', help='2 subjects per task, 1 seed, mean mode only')
    ap.add_argument('--n_draws_m3', type=int, default=20)
    a = ap.parse_args()
    t0 = time.time()
    tasks = a.tasks.split(','); seeds = [int(s) for s in a.seeds.split(',')]; modes = a.modes.split(',')
    if a.smoke:
        seeds = [0]; modes = ['mean']
    # ---- data
    all_texts, recs, split, task_of = [], {}, {}, {}
    for t in tasks:
        texts, sp, rc, subjects = load_task(os.path.join(a.data, TASKS[t]), t, max_subjects=2 if a.smoke else None)
        off = len(all_texts)
        idx_map = {}
        for i, tx in enumerate(texts):
            if tx is not None:
                idx_map[i] = len(all_texts); all_texts.append(tx); split[idx_map[i]] = sp[i]; task_of[idx_map[i]] = t
        for (sj, i), v in rc.items():
            recs[(t + ':' + sj, idx_map[i])] = v
        print(f'{t}: {len(idx_map)} sentences, {len(subjects)} subjects, {len(rc)} usable (subject,sentence) records', flush=True)
    # duplicate texts across tasks (ZuCo re-used some sentences): collapse so no test sentence is also a train sentence
    first = {}
    for g, tx in enumerate(all_texts):
        first.setdefault(tx, g)
    dup_test = [g for g in range(len(all_texts)) if split[g] == 'test' and first[all_texts[g]] != g]
    for g in range(len(all_texts)):
        if first[all_texts[g]] != g:
            split[g] = 'drop'
    print('dropped duplicate sentences:', sum(1 for g in range(len(all_texts)) if split[g] == 'drop'), 'of which test', len(dup_test), flush=True)
    E, enc_name = embed_sentences(all_texts, a.minilm)
    print('sentence encoder:', enc_name, E.shape, flush=True)
    train_g = [g for g in split if split[g] == 'train']; dev_g = [g for g in split if split[g] == 'dev']; test_g = [g for g in split if split[g] == 'test']
    N = len(test_g); Ndev = len(dev_g)
    print(f'train {len(train_g)} dev {Ndev} test N={N} sentences', flush=True)
    test_pos = {g: j for j, g in enumerate(test_g)}; dev_pos = {g: j for j, g in enumerate(dev_g)}
    by_sent = {}
    for (sj, g) in recs:
        by_sent.setdefault(g, []).append(sj)
    times = json.load(open(a.times)) if os.path.exists(a.times) else {}
    mean_gd_s = times.get('mean_GD_s'); mean_sent_s = times.get('mean_sentence_duration_s')
    n_words_all = float(np.mean([recs[k][0].shape[0] for k in recs]))
    print('mean fixated words per (subject,sentence):', n_words_all, flush=True)
    alphas = [1e1, 1e2, 1e3, 1e4, 1e5]
    results = {}  # (decoder, subject_label, seed) -> rows
    def add(dec, subj, seed, row):
        results.setdefault((dec, subj, seed), []).append(row)
    for mode in modes:
        ks = ['all'] if mode == 'sentlevel' else ([k for k in WORD_KS if k != 'all'] if mode == 'concat' else WORD_KS)
        for k in ks:
            dec = f'ridge_{mode}' if mode == 'sentlevel' else f'ridge_{mode}_words{k}'
            Xtr = np.stack([feat(*recs[(sj, g)], k, mode) for g in train_g for sj in by_sent.get(g, [])])
            Ytr = np.stack([E[g] for g in train_g for sj in by_sent.get(g, [])])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
            z = lambda X: (X - mu) / sd
            Xtr_z = z(Xtr)
            Xdv = z(np.stack([feat(*recs[(sj, g)], k, mode) for g in dev_g for sj in by_sent.get(g, [])]))
            ydv = np.array([dev_pos[g] for g in dev_g for sj in by_sent.get(g, [])])
            # alpha by dev top-1 retrieval (single-subject queries)
            best = None
            for al in alphas:
                m = Ridge(alpha=al).fit(Xtr_z, Ytr); t1, _ = retrieval_scores(m.predict(Xdv), E[dev_g], ydv)
                if best is None or t1 > best[0]:
                    best = (t1, al)
            alpha = best[1]
            print(f'[{dec}] dims {Xtr_z.shape} dev top1 {best[0]:.3f} alpha {alpha:g}  ({time.time()-t0:.0f}s)', flush=True)
            # time price for this amount of EEG
            if mode == 'sentlevel' or k == 'all':
                stim_s = mean_sent_s
            else:
                stim_s = (k * mean_gd_s) if mean_gd_s else None
            for seed in seeds:
                rng = np.random.default_rng(seed)
                models = {}
                models['eeg'] = Ridge(alpha=alpha).fit(Xtr_z, Ytr)
                perm = rng.permutation(len(Ytr)); models['trial_shuffled'] = Ridge(alpha=alpha).fit(Xtr_z, Ytr[perm])
                models['noise'] = Ridge(alpha=alpha).fit(rng.standard_normal(Xtr_z.shape), Ytr)
                for m_sub in SUBJ_MS:
                    # build query sets: list of draws, each draw = (X (Q,D), true_idx (Q,), label)
                    draws = []
                    if m_sub == 1:
                        subs = sorted({sj for g in test_g for sj in by_sent.get(g, [])})
                        for sj in subs:
                            gs = [g for g in test_g if sj in by_sent.get(g, [])]
                            if len(gs) < 5:
                                continue
                            draws.append((z(np.stack([feat(*recs[(sj, g)], k, mode) for g in gs])), np.array([test_pos[g] for g in gs]), sj))
                    elif m_sub == 'all':
                        X = []; y = []
                        for g in test_g:
                            sjs = by_sent.get(g, [])
                            if not sjs:
                                continue
                            X.append(z(np.stack([feat(*recs[(sj, g)], k, mode) for sj in sjs])).mean(0)); y.append(test_pos[g])
                        if len(X) >= 5:
                            draws.append((np.stack(X), np.array(y), 'all'))
                    else:
                        for dr in range(a.n_draws_m3 if not a.smoke else 2):
                            X = []; y = []
                            for g in test_g:
                                sjs = by_sent.get(g, [])
                                if len(sjs) < m_sub:
                                    continue
                                pick = rng.choice(sjs, m_sub, replace=False)
                                X.append(z(np.stack([feat(*recs[(sj, g)], k, mode) for sj in pick])).mean(0)); y.append(test_pos[g])
                            if len(X) >= 5:
                                draws.append((np.stack(X), np.array(y), f'draw{dr}'))
                    k_rep = m_sub if m_sub != 'all' else int(np.mean([len(by_sent.get(g, [])) for g in test_g]))
                    for di, (Xq, yq, label) in enumerate(draws):
                        for cond in ['eeg', 'noise', 'trial_shuffled', 'shuffled_pair']:
                            if cond == 'noise':
                                pred = models['noise'].predict(rng.standard_normal(Xq.shape)); yy = yq
                            elif cond == 'trial_shuffled':
                                pred = models['trial_shuffled'].predict(Xq); yy = yq
                            elif cond == 'shuffled_pair':
                                pred = models['eeg'].predict(Xq); yy = rng.permutation(yq)
                            else:
                                pred = models['eeg'].predict(Xq); yy = yq
                            t1, tw = retrieval_scores(pred, E[test_g], yy)
                            bits = wolpaw_bits(t1, N)
                            row = {'axis': 'repetitions' if cond == 'eeg' else 'control', 'k': int(k_rep), 'condition': cond,
                                   'channels': f'105ch x 8 bands; words={k}; mode={mode}', 'window_s': stim_s, 'draw': di,
                                   'top1': t1, 'two_way': tw, 'bits': float(bits),
                                   'bits_per_second': (float(bits) / (k_rep * stim_s)) if stim_s else None,
                                   'norm_eff': float(bits) / np.log2(N), 'words': k, 'subjects_averaged': m_sub, 'n_queries': int(len(yq)),
                                   'alpha': alpha, 'draw_label': label}
                            add(dec, label if m_sub == 1 else 'pooled', seed, row)
    os.makedirs(a.out, exist_ok=True)
    meta = {'track': 'text', 'n_candidates': N, 'n_dev': Ndev, 'n_train_sentences': len(train_g), 'tasks': tasks,
            'sentence_encoder': enc_name, 'eeg_feature': 'GD (gaze-duration) 8-band Hilbert amplitude, 105 ch -> 840-d per word (Wang & Ji 2022 pickles)',
            'split': 'Wang & Ji 2022: per task first 80% sentence indices train, 10% dev, 10% test; duplicates across tasks dropped',
            'reading_times': times, 'mean_fixated_words': n_words_all,
            'test_counts_per_task': {t: sum(1 for g in test_g if task_of[g] == t) for t in tasks},
            'test_subjects_per_task': {t: len({sj for g in test_g if task_of[g] == t for sj in by_sent.get(g, [])}) for t in tasks}, 'smoke': a.smoke, 'runtime_s': time.time() - t0}
    tag = '_smoke' if a.smoke else ''
    for (dec, subj, seed), rows in results.items():
        f = {'track': 'text', 'decoder': dec, 'subject': subj, 'seed': seed, 'n_candidates': N, 'stimulus_time_s': mean_sent_s, 'rows': rows}
        json.dump(f, open(os.path.join(a.out, f'{dec}__{subj}__seed{seed}{tag}.json'), 'w'))
    json.dump(meta, open(os.path.join(a.out, f'meta{tag}.json'), 'w'), indent=1)
    # quick summary: eeg vs controls per decoder, pooled over draws/seeds
    print('\nSUMMARY (N=%d candidates; chance top1=%.3f)' % (N, 1 / N))
    agg = {}
    for (dec, subj, seed), rows in results.items():
        for r in rows:
            agg.setdefault((dec, r['subjects_averaged'], r['condition']), []).append((r['top1'], r['two_way'], r['bits']))
    for key in sorted(agg, key=lambda x: (str(x[0]), str(x[1]), x[2])):
        v = np.array(agg[key]); print(f'{key[0]:28s} m={str(key[1]):4s} {key[2]:15s} top1 {v[:,0].mean():.3f} 2way {v[:,1].mean():.3f} bits {v[:,2].mean():.3f} (n={len(v)})')
    print('done in %.0fs' % (time.time() - t0))

if __name__ == '__main__':
    main()
