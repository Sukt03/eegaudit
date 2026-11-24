"""Text-track target swap (ideation T1+T2): same EEG features, same ridge decoder, different TARGET spaces.

Data: ZuCo task1-SR word-level EEG (Wang & Ji 2022 pickle). EEG feature per (subject, sentence) = concat of the first 8
fixated words' 840-d GD band features (as in pilot_retrieval.py mode=concat, k=8). Wang & Ji split by sentence index.
Targets: (a) sentence identity via MiniLM embedding -> retrieval top-1 among N test sentences, Wolpaw bits;
         (b) sentiment: SKIPPED unless a label field exists (the pickle has none; see report);
         (c) continuous sentence properties: mean word length, mean log word frequency (counts over all ZuCo sentence
             texts in the three pickles), mean fixation count per word (subject-specific, from nFixations / n words).
Inputs: EEG; gaze-only (nFixations of the first 8 fixated words + n fixated words + total fixations: the only oculomotor
fields in the pickle; no durations are stored); noise (N(0,1) train+test, 3 seeds); shuffled pairs (EEG model, targets
permuted across the subject's test sentences).
Metric (continuous): out-of-sample R^2 = 1 - sum (y - yhat)^2 / sum (y - mean(y_test))^2 per subject over its test
sentences; bits = 0.5*log2(1/(1-R^2)) if R^2>0 else 0 (Gaussian channel analogue). Mean and 95% CI (t) over subjects.
Alpha: picked per (input, target) on a held-out training subject (first subject): fit on the other subjects' train
sentences, score on the held-out subject's train sentences; then refit on all subjects' train sentences.
Run: srun ... python -I target_swap.py"""
import json, os, pickle, sys, time, math
import numpy as np
from sklearn.linear_model import Ridge
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_retrieval import load_task, feat, embed_sentences, retrieval_scores, wolpaw_bits, TASKS

DATA = '/path/to/eegaudit/data_raw/zuco/eeg2text_pickles'
MINILM = '/path/to/eegaudit/data_raw/zuco/minilm'
OUT = '/path/to/eegaudit/results/track_text/target_swap'
K = 8; SEEDS = [0, 1, 2]; ALPHAS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]

def norm_word(w):
    return ''.join(ch for ch in w.lower() if ch.isalnum())

def r2_bits(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    bits = 0.5 * math.log2(1 / (1 - r2)) if (r2 == r2 and r2 > 0) else 0.0
    return r2, bits

def ci95(v):
    v = np.asarray([x for x in v if x == x], float); n = len(v)
    if n < 2:
        return float(v.mean()) if n else float('nan'), float('nan'), float('nan'), n
    from scipy import stats
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / math.sqrt(n)
    return float(v.mean()), float(v.mean() - h), float(v.mean() + h), n

def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    # ---- word frequency table over all ZuCo sentence texts on disk (texts only; pickles loaded one at a time)
    from collections import Counter
    freq = Counter(); n_sent_all = 0
    for t, fn in TASKS.items():
        d = pickle.load(open(os.path.join(DATA, fn), 'rb'))
        seen = set()
        for sj in d:
            for s in d[sj]:
                if s is None: continue
                tx = s['content'].strip()
                if tx in seen: continue
                seen.add(tx); n_sent_all += 1
                for w in s['word_tokens_all']:
                    nw = norm_word(w)
                    if nw: freq[nw] += 1
        del d
    tot = sum(freq.values()); print(f'freq table: {len(freq)} types, {tot} tokens, {n_sent_all} unique sentences ({time.time()-t0:.0f}s)', flush=True)
    # ---- SR data (reload to also get word_tokens_all and nFixations, which load_task drops)
    texts, split, recs, subjects = load_task(os.path.join(DATA, TASKS['SR']), 'SR')
    d = pickle.load(open(os.path.join(DATA, TASKS['SR']), 'rb'))
    has_label = any(k in d[subjects[0]][0] for k in ('label', 'sentiment', 'sentiment_label'))
    print('sentiment label field present:', has_label, flush=True)
    tokens_all, gaze, fixpw = {}, {}, {}
    for sj in subjects:
        for i, s in enumerate(d[sj]):
            if s is None or (sj, i) not in recs: continue
            if i not in tokens_all: tokens_all[i] = s['word_tokens_all']
            nf = np.array([float(w.get('nFixations', 0) or 0) for w in s['word']])
            v = np.zeros(K); v[:min(K, len(nf))] = nf[:K]
            gaze[(sj, i)] = np.concatenate([v, [len(nf), nf.sum()]])
            fixpw[(sj, i)] = nf.sum() / max(1, len(s['word_tokens_all']))
    del d
    # sentence-level targets (same for all subjects)
    wlen = {i: float(np.mean([len(norm_word(w)) for w in tk if norm_word(w)])) for i, tk in tokens_all.items()}
    wfreq = {i: float(np.mean([math.log(freq[norm_word(w)] / tot) for w in tk if norm_word(w)])) for i, tk in tokens_all.items()}
    E, enc = embed_sentences([tx if tx else '' for tx in texts], MINILM); print('encoder', enc, flush=True)
    keys = sorted(recs); train_k = [k for k in keys if split[k[1]] == 'train']; test_k = [k for k in keys if split[k[1]] == 'test']
    test_g = sorted({k[1] for k in test_k}); test_pos = {g: j for j, g in enumerate(test_g)}; N = len(test_g)
    print(f'{len(subjects)} subjects, {len(train_k)} train records, {len(test_k)} test records, N={N} test sentences', flush=True)
    X_in = {'eeg': {k: feat(*recs[k], K, 'concat') for k in keys}, 'gaze': gaze}
    Y_t = {'identity': {k: E[k[1]] for k in keys}, 'word_length': {k: wlen[k[1]] for k in keys},
           'log_word_freq': {k: wfreq[k[1]] for k in keys}, 'fix_per_word': {k: fixpw[k] for k in keys}}
    hold = subjects[0]
    results = {'meta': {'task': 'SR', 'subjects': subjects, 'held_out_alpha_subject': hold, 'n_train_records': len(train_k),
                        'n_test_records': len(test_k), 'n_test_sentences': N, 'k_words': K, 'encoder': enc, 'alphas': ALPHAS,
                        'sentiment': 'skipped: no label/sentiment field in the pickle', 'freq_types': len(freq), 'freq_tokens': tot,
                        'freq_unique_sentences': n_sent_all, 'gaze_features': 'nFixations of first 8 fixated words, n fixated words, total fixations',
                        'r2_estimator': 'per subject over its test sentences: 1 - SSres/SStot, SStot about the test mean; bits = 0.5*log2(1/(1-R2)), 0 if R2<=0',
                        'identity_metric': 'top-1 retrieval among N test sentences by cosine; Wolpaw bits at N'},
               'per_subject': [], 'summary': {}}
    for target, Yd in Y_t.items():
        for inp in ['eeg', 'gaze', 'noise', 'shuffled']:
            base_in = 'eeg' if inp in ('noise', 'shuffled') else inp
            Xtr = np.stack([X_in[base_in][k] for k in train_k]); Ytr = np.stack([Yd[k] for k in train_k])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8; z = lambda X: (X - mu) / sd; Xtr_z = z(Xtr)
            # alpha on held-out training subject
            m_ho = np.array([k[0] == hold for k in train_k]); best = None
            for al in ALPHAS:
                m = Ridge(alpha=al).fit(Xtr_z[~m_ho], Ytr[~m_ho]); p = m.predict(Xtr_z[m_ho])
                if target == 'identity':
                    gs = [k[1] for k, h in zip(train_k, m_ho) if h]; ug = sorted(set(gs)); pos = {g: j for j, g in enumerate(ug)}
                    sc, _ = retrieval_scores(p, E[ug], np.array([pos[g] for g in gs]))
                else:
                    sc, _ = r2_bits(Ytr[m_ho], p)
                if best is None or sc > best[0]: best = (sc, al)
            alpha = best[1]
            for seed in (SEEDS if inp in ('noise', 'shuffled') else [0]):
                rng = np.random.default_rng(seed)
                Xfit = rng.standard_normal(Xtr_z.shape) if inp == 'noise' else Xtr_z
                model = Ridge(alpha=alpha).fit(Xfit, Ytr)
                for sj in subjects:
                    ks = [k for k in test_k if k[0] == sj]
                    if len(ks) < 5: continue
                    Xq = z(np.stack([X_in[base_in][k] for k in ks]))
                    if inp == 'noise': Xq = rng.standard_normal(Xq.shape)
                    p = model.predict(Xq); y = np.stack([Yd[k] for k in ks])
                    if target == 'identity':
                        ti = np.array([test_pos[k[1]] for k in ks])
                        if inp == 'shuffled': ti = rng.permutation(ti)
                        t1, _ = retrieval_scores(p, E[test_g], ti); score = t1; bits = wolpaw_bits(t1, N)
                    else:
                        if inp == 'shuffled': y = y[rng.permutation(len(y))]
                        score, bits = r2_bits(y, p)
                    results['per_subject'].append({'target': target, 'input': inp, 'seed': seed, 'subject': sj, 'n_test': len(ks),
                                                   'score': float(score), 'bits': float(bits), 'alpha': alpha})
            print(f'[{target:14s} {inp:8s}] alpha {alpha:g} ho-score {best[0]:.3f} ({time.time()-t0:.0f}s)', flush=True)
    for target in Y_t:
        for inp in ['eeg', 'gaze', 'noise', 'shuffled']:
            rows = [r for r in results['per_subject'] if r['target'] == target and r['input'] == inp]
            # per subject average over seeds first, then CI over subjects
            per = {}
            for r in rows: per.setdefault(r['subject'], []).append((r['score'], r['bits']))
            sc = [np.mean([a for a, b in v]) for v in per.values()]; bt = [np.mean([b for a, b in v]) for v in per.values()]
            ms, lo, hi, n = ci95(sc); mb, blo, bhi, _ = ci95(bt)
            results['summary'][f'{target}|{inp}'] = {'score_mean': ms, 'score_lo': lo, 'score_hi': hi, 'bits_mean': mb, 'bits_lo': blo,
                                                      'bits_hi': bhi, 'n_subjects': n, 'n_pos_subjects': int(sum(1 for s in sc if s > 0)),
                                                      'alpha': rows[0]['alpha'] if rows else None}
            print(f'{target:14s} {inp:8s} score {ms:.3f} [{lo:.3f},{hi:.3f}] bits {mb:.3f} [{blo:.3f},{bhi:.3f}] n={n}', flush=True)
    results['meta']['chance_top1'] = 1 / N; results['meta']['runtime_s'] = time.time() - t0
    json.dump(results, open(os.path.join(OUT, 'target_swap_SR.json'), 'w'), indent=1)
    print('wrote', os.path.join(OUT, 'target_swap_SR.json'))

if __name__ == '__main__':
    main()
