"""TEXT smokes T1-T3 (loop 2026-11-20) on ZuCo word-level features (Wang & Ji 2022 pickles).
Stage 'cache <task>': load the pickle once -> results/loop/text/cache_<task>.npz (EEG concat-8 / mean-all features, gaze counts,
sentence texts, MiniLM embeddings, Wang & Ji split).  The pickles store nFixations per fixated word but NO durations, so gaze =
fixation counts (per word, totals, number of fixated words, fraction of tokens fixated).
T1 <task>: identification within task from gaze-only, EEG, EEG+gaze; noise and shuffled-pair controls, 3 seeds.
T2: target swap on SR: identity (MiniLM) vs sentence length vs mean log word frequency (counts over all three pickles' texts)
    vs mean word length; sentiment label if the pickle has one; out-of-sample R^2 and 0.5*log2(1/(1-R^2)) bits; gaze-only control.
T3: within-subject vs leave-one-subject-out ridge on SR, candidates restricted to +-2 words of the query length.
Run with python -I. CPU only."""
import sys, os, json, time, math, pickle, re
from collections import Counter
import numpy as np

ROOT = '/path/to/eegaudit'
DATA = ROOT + '/data_raw/zuco/eeg2text_pickles'
MINILM = ROOT + '/data_raw/zuco/minilm'
OUT = ROOT + '/results/loop/text'
TASKS = {'SR': 'task1-SR-dataset.pickle', 'NR': 'task2-NR-dataset.pickle', 'NR2': 'task2-NR-2.0-dataset.pickle'}
BANDS = ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2']
ALPHAS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
SEEDS = [0, 1, 2]
K = 8

def wolpaw_bits(P, N):
    if N <= 1 or P <= 1.0 / N: return 0.0
    P = min(P, 1 - 1e-9)
    return float(np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1)))

def word_vec(w):
    try: parts = [np.asarray(w['word_level_EEG']['GD']['GD' + b], dtype=np.float64).ravel() for b in BANDS]
    except KeyError: return None
    v = np.concatenate(parts)
    return v if (v.size == 840 and np.all(np.isfinite(v))) else None

# ---------------------------------------------------------------- cache
def build_cache(task):
    t0 = time.time(); d = pickle.load(open(os.path.join(DATA, TASKS[task]), 'rb')); subjects = sorted(d.keys())
    print(f'[{task}] pickle loaded {time.time()-t0:.0f}s, subjects {subjects}', flush=True)
    n = max(len(d[s]) for s in subjects); texts = [None] * n; sent_keys = Counter(); rows = []
    for sj in subjects:
        for i, s in enumerate(d[sj]):
            if s is None: continue
            sent_keys.update(s.keys())
            if texts[i] is None: texts[i] = s['content'].strip()
            wv, nf = [], []
            for w in s['word']:
                v = word_vec(w)
                if v is None: continue
                wv.append(v); nf.append(float(w.get('nFixations', 1) or 1))
            if not wv: continue
            W = np.stack(wv); nf = np.array(nf); ntok = len(s['content'].split())
            c8 = np.zeros(840 * K); c8[:840 * min(K, len(W))] = W[:K].ravel()
            f8 = np.zeros(K); f8[:min(K, len(nf))] = nf[:K]
            gaze = np.concatenate([[len(W), nf.sum(), nf.mean(), len(W) / max(ntok, 1), nf.max(), len(s['word'])], f8])
            lab = None
            for lk in ('label', 'sentiment', 'sentiment_label'):
                if lk in s: lab = s[lk]
            rows.append((sj, i, c8, W.mean(0), gaze, ntok, lab))
    keep = [i for i in range(n) if texts[i] is not None]; nk = len(keep)
    split = {}
    for j, i in enumerate(keep): split[i] = 'train' if j < int(0.8 * nk) else ('dev' if j < int(0.9 * nk) else 'test')
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(2); tok = AutoTokenizer.from_pretrained(MINILM); mdl = AutoModel.from_pretrained(MINILM).eval()
    E = np.zeros((n, 384), dtype=np.float32)
    with torch.no_grad():
        for a in range(0, len(keep), 64):
            ids = keep[a:a + 64]; b = tok([texts[i] for i in ids], padding=True, truncation=True, max_length=128, return_tensors='pt')
            h = mdl(**b).last_hidden_state; m = b['attention_mask'].unsqueeze(-1).float()
            E[ids] = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=1).numpy()
    np.savez_compressed(os.path.join(OUT, f'cache_{task}.npz'),
        subj=np.array([r[0] for r in rows]), sent=np.array([r[1] for r in rows]), c8=np.stack([r[2] for r in rows]).astype(np.float32),
        mean=np.stack([r[3] for r in rows]).astype(np.float32), gaze=np.stack([r[4] for r in rows]).astype(np.float32),
        ntok=np.array([r[5] for r in rows]), label=np.array([json.dumps(r[6]) for r in rows]),
        texts=np.array([t or '' for t in texts]), E=E, split=np.array([split.get(i, 'none') for i in range(n)]), subjects=np.array(subjects),
        sentence_keys=np.array(json.dumps(dict(sent_keys))))
    print(f'[{task}] cache written: {len(rows)} records, {nk} sentences, sentence keys {dict(sent_keys)} ({time.time()-t0:.0f}s)', flush=True)

def load_cache(task):
    z = np.load(os.path.join(OUT, f'cache_{task}.npz'), allow_pickle=False)
    return {k: z[k] for k in z.files}

# ---------------------------------------------------------------- ridge (multi-alpha, dual when p > n)
def ridge_predict(Xtr, Ytr, Xtes, alphas):
    """z-score X on train, centre Y; return {alpha: [pred for each Xte]}. Xtes: list of arrays."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Z = (Xtr - mu) / sd; ym = Ytr.mean(0); Yc = Ytr - ym
    Zs = [(X - mu) / sd for X in Xtes]; n, p = Z.shape; out = {}
    if p > n:
        Kk = Z @ Z.T; w, V = np.linalg.eigh(Kk); VtY = V.T @ Yc
        for a in alphas:
            A = V @ (VtY / (w + a)[:, None]); out[a] = [(Zt @ Z.T) @ A + ym for Zt in Zs]
    else:
        G = Z.T @ Z; w, V = np.linalg.eigh(G); VtXY = V.T @ (Z.T @ Yc)
        for a in alphas:
            Wt = V @ (VtXY / (w + a)[:, None]); out[a] = [Zt @ Wt + ym for Zt in Zs]
    return out

def ident(pred, cand, true_idx, mask=None):
    """top-1, 2-way among candidates; mask (Q,N) bool restricts candidates per query (truth always included)."""
    P = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12); S = P @ cand.T
    if mask is not None: S = np.where(mask, S, -np.inf)
    ts = S[np.arange(len(true_idx)), true_idx]; beat = (S > ts[:, None]).sum(1)
    Ne = (np.isfinite(S)).sum(1) if mask is not None else np.full(len(true_idx), cand.shape[0])
    two = np.where(Ne > 1, 1 - beat / np.maximum(Ne - 1, 1), np.nan)
    return float((beat == 0).mean()), float(np.nanmean(two)), float(Ne.mean())

def r2(y, p):
    ss = ((y - p) ** 2).sum(); st = ((y - y.mean()) ** 2).sum()
    return float(1 - ss / st) if st > 0 else float('nan')

def r2_bits(v): return 0.5 * math.log2(1 / (1 - v)) if (v == v and 0 < v < 1) else 0.0

def ci(v):
    v = np.asarray([x for x in v if x == x], float)
    return (float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0, int(len(v)))

def fmt(m): return f'{m[0]:.3f} ± {m[1]:.3f}'

# ---------------------------------------------------------------- T1
def run_T1(task):
    t0 = time.time(); C = load_cache(task); subj, sent, split = C['subj'], C['sent'], C['split']; E = C['E']
    inputs = {'eeg_c8': C['c8'], 'eeg_mean': C['mean'], 'gaze': C['gaze'], 'gaze_nwords': C['gaze'][:, :1],
              'eeg_c8+gaze': np.concatenate([C['c8'], C['gaze']], 1)}
    sp = np.array([split[i] for i in sent]); tr, dv, te = sp == 'train', sp == 'dev', sp == 'test'
    dev_g = sorted(set(sent[dv])); test_g = sorted(set(sent[te])); N = len(test_g)
    dpos = {g: j for j, g in enumerate(dev_g)}; tpos = {g: j for j, g in enumerate(test_g)}
    di = np.array([dpos[g] for g in sent[dv]]); ti = np.array([tpos[g] for g in sent[te]]); subjects = sorted(set(subj))
    res = {'meta': {'task': task, 'N_test': N, 'n_train_records': int(tr.sum()), 'n_subjects': len(subjects), 'chance_top1': 1 / N,
                    'log2N': math.log2(N), 'split': 'Wang & Ji 80/10/10 by sentence index', 'alpha_grid': ALPHAS, 'seeds': SEEDS,
                    'gaze_features': '14-d: n fixated words, total nFix, mean nFix, fraction of tokens fixated, max nFix, n word entries, nFix of first 8 fixated words (no durations in the pickles)',
                    'inputs': list(inputs)}, 'rows': [], 'summary': {}}
    def score(pred_te, cond, inp, seed, perm=None):
        for s in subjects:
            m = subj[te] == s
            if m.sum() < 5: continue
            idx = ti[m]
            if perm is not None: idx = perm.permutation(idx)
            t1, two, _ = ident(pred_te[m], E[test_g], idx)
            res['rows'].append({'input': inp, 'condition': cond, 'seed': seed, 'subject': s, 'n': int(m.sum()), 'top1': t1, 'two_way': two, 'bits': wolpaw_bits(t1, N)})
    for inp, X in inputs.items():
        Ytr = E[sent[tr]]; P = ridge_predict(X[tr], Ytr, [X[dv], X[te]], ALPHAS)
        best = max(ALPHAS, key=lambda a: ident(P[a][0], E[dev_g], di)[0] + 1e-3 * ident(P[a][0], E[dev_g], di)[1])
        score(P[best][1], 'real', inp, 0)
        for sd in SEEDS: score(P[best][1], 'shuffled_pairs', inp, sd, np.random.default_rng(sd))
        for sd in SEEDS:
            rng = np.random.default_rng(100 + sd); Xn_tr = rng.standard_normal(X[tr].shape); Xn_te = rng.standard_normal(X[te].shape)
            Pn = ridge_predict(Xn_tr, Ytr, [Xn_te], [best]); score(Pn[best][0], 'noise', inp, sd)
        res['summary'][inp] = {'alpha': best}
        print(f'[{task} T1] {inp} alpha {best:g} ({time.time()-t0:.0f}s)', flush=True)
    for inp in inputs:
        for cond in ('real', 'shuffled_pairs', 'noise'):
            rr = [r for r in res['rows'] if r['input'] == inp and r['condition'] == cond]
            res['summary'][inp][cond] = {k: ci([r[k] for r in rr]) for k in ('top1', 'two_way', 'bits')}
    res['meta']['runtime_s'] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, f'T1_{task}.json'), 'w'), indent=1)
    print(f'| input | top-1 real | top-1 noise | 2-way real | 2-way noise | 2-way shuffled | bits real | bits noise |')
    for inp in inputs:
        S = res['summary'][inp]
        print(f"| {inp} | {fmt(S['real']['top1'])} | {fmt(S['noise']['top1'])} | {fmt(S['real']['two_way'])} | {fmt(S['noise']['two_way'])} | {fmt(S['shuffled_pairs']['two_way'])} | {fmt(S['real']['bits'])} | {fmt(S['noise']['bits'])} |")

# ---------------------------------------------------------------- T2
def norm_word(w): return re.sub(r'[^a-z]', '', w.lower())

def run_T2():
    t0 = time.time(); freq = Counter(); nsent = 0
    for t in TASKS:
        Ct = load_cache(t)
        for tx in Ct['texts']:
            if tx: nsent += 1; freq.update(nw for nw in (norm_word(w) for w in tx.split()) if nw)
    tot = sum(freq.values())
    C = load_cache('SR'); subj, sent, split, E, texts = C['subj'], C['sent'], C['split'], C['E'], C['texts']
    labels = [json.loads(x) for x in C['label']]; has_label = any(l is not None for l in labels)
    length = np.array([len(texts[i].split()) for i in sent], float)
    lfreq = np.array([np.mean([math.log(freq[nw] / tot) for nw in (norm_word(w) for w in texts[i].split()) if nw]) for i in sent])
    wlen = np.array([np.mean([len(nw) for nw in (norm_word(w) for w in texts[i].split()) if nw]) for i in sent])
    targets = {'identity_minilm': E[sent], 'sentence_length': length[:, None], 'mean_log_word_freq': lfreq[:, None], 'mean_word_length': wlen[:, None]}
    if has_label:
        lab = np.array([float(l) if l is not None else np.nan for l in labels]); targets['sentiment_label'] = lab[:, None]
    inputs = {'eeg_c8': C['c8'], 'eeg_mean': C['mean'], 'gaze': C['gaze'], 'eeg_c8+gaze': np.concatenate([C['c8'], C['gaze']], 1)}
    sp = np.array([split[i] for i in sent]); tr, dv, te = sp == 'train', sp == 'dev', sp == 'test'
    dev_g = sorted(set(sent[dv])); test_g = sorted(set(sent[te])); N = len(test_g)
    dpos = {g: j for j, g in enumerate(dev_g)}; tpos = {g: j for j, g in enumerate(test_g)}
    di = np.array([dpos[g] for g in sent[dv]]); ti = np.array([tpos[g] for g in sent[te]]); subjects = sorted(set(subj))
    res = {'meta': {'task': 'SR', 'N_test': N, 'n_subjects': len(subjects), 'freq_types': len(freq), 'freq_tokens': tot, 'freq_sentences': nsent,
                    'sentiment_label_in_pickle': has_label, 'sentence_keys_in_pickle': json.loads(str(C['sentence_keys'])),
                    'r2': 'per subject over its test sentences, out of sample; bits = 0.5*log2(1/(1-R2)), 0 if R2<=0; identity: top-1 + Wolpaw bits at N',
                    'target_corr_with_length': {k: float(np.corrcoef(v[:, 0], length)[0, 1]) for k, v in targets.items() if k != 'identity_minilm'},
                    'gaze_nwords_corr_with_length': float(np.corrcoef(C['gaze'][:, 0], length)[0, 1]), 'alpha_grid': ALPHAS, 'seeds': SEEDS}, 'rows': [], 'summary': {}}
    def score(pred, tgt, Y, cond, inp, seed, perm=None):
        for s in subjects:
            m = subj[te] == s
            if m.sum() < 5: continue
            if tgt == 'identity_minilm':
                idx = ti[m]
                if perm is not None: idx = perm.permutation(idx)
                t1, two, _ = ident(pred[m], E[test_g], idx); row = {'score': t1, 'two_way': two, 'bits': wolpaw_bits(t1, N)}
            else:
                y = Y[te][m][:, 0]
                if perm is not None: y = y[perm.permutation(len(y))]
                v = r2(y, pred[m][:, 0]); row = {'score': v, 'bits': r2_bits(v)}
            res['rows'].append(dict(row, target=tgt, input=inp, condition=cond, seed=seed, subject=s, n=int(m.sum())))
    for tgt, Y in targets.items():
        ok = np.isfinite(Y).all(1)
        for inp, X in inputs.items():
            P = ridge_predict(X[tr & ok], Y[tr & ok], [X[dv], X[te]], ALPHAS)
            if tgt == 'identity_minilm': best = max(ALPHAS, key=lambda a: ident(P[a][0], E[dev_g], di)[1])
            else: best = max(ALPHAS, key=lambda a: r2(Y[dv][:, 0], P[a][0][:, 0]))
            score(P[best][1], tgt, Y, 'real', inp, 0)
            for sd in SEEDS: score(P[best][1], tgt, Y, 'shuffled_pairs', inp, sd, np.random.default_rng(sd))
            for sd in SEEDS:
                rng = np.random.default_rng(100 + sd); Pn = ridge_predict(rng.standard_normal(X[tr & ok].shape), Y[tr & ok], [rng.standard_normal(X[te].shape)], [best])
                score(Pn[best][0], tgt, Y, 'noise', inp, sd)
            res['summary'][f'{tgt}|{inp}'] = {'alpha': best}
            print(f'[T2] {tgt} {inp} alpha {best:g} ({time.time()-t0:.0f}s)', flush=True)
    for key in list(res['summary']):
        tgt, inp = key.split('|')
        for cond in ('real', 'shuffled_pairs', 'noise'):
            rr = [r for r in res['rows'] if r['target'] == tgt and r['input'] == inp and r['condition'] == cond]
            res['summary'][key][cond] = {k: ci([r[k] for r in rr]) for k in ('score', 'bits')}
            res['summary'][key][cond]['n_subjects_positive'] = int(sum(1 for r in rr if r['score'] > (1 / N if tgt == 'identity_minilm' else 0)))
    res['meta']['runtime_s'] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, 'T2_SR.json'), 'w'), indent=1)
    print('| target | input | score real (top-1 or R2) | score noise | score shuffled | bits real | bits noise |')
    for key, S in res['summary'].items():
        print(f"| {key.replace('|', ' | ')} | {fmt(S['real']['score'])} | {fmt(S['noise']['score'])} | {fmt(S['shuffled_pairs']['score'])} | {fmt(S['real']['bits'])} | {fmt(S['noise']['bits'])} |")

# ---------------------------------------------------------------- T3
def run_T3():
    t0 = time.time(); C = load_cache('SR'); subj, sent, split, E, texts = C['subj'], C['sent'], C['split'], C['E'], C['texts']
    X = C['c8']; length = np.array([len(texts[i].split()) for i in range(len(texts))])
    sp = np.array([split[i] for i in sent]); subjects = sorted(set(subj))
    test_g = sorted(set(sent[sp == 'test'])); N = len(test_g); tpos = {g: j for j, g in enumerate(test_g)}
    dev_g = sorted(set(sent[sp == 'dev'])); dpos = {g: j for j, g in enumerate(dev_g)}
    Lc = length[test_g]
    def masks(gq): return np.abs(Lc[None, :] - length[gq][:, None]) <= 2
    res = {'meta': {'task': 'SR', 'N_test': N, 'n_subjects': len(subjects), 'feature': 'EEG concat of first 8 fixated words (6720-d)',
                    'restriction': 'candidates = test sentences within +-2 words of the query sentence length (truth included)',
                    'conditions': {'within': 'ridge trained on the held-out subject\'s own train-split rows',
                                   'loso_unseen': 'ridge trained on the other subjects\' train-split rows (test sentences unseen)',
                                   'loso_seen': 'ridge trained on the other subjects\' rows for all sentences incl. the test sentences',
                                   'template': 'no language model: candidate = mean EEG of the other subjects for that test sentence, cosine'},
                    'alpha_grid': ALPHAS, 'seeds': SEEDS}, 'rows': [], 'summary': {}}
    def add(cond, s, pred, cand, idx, seed, ctrl, Xq=None):
        for restricted in (False, True):
            m = masks(sent_q) if restricted else None
            t1, two, Ne = ident(pred, cand, idx, m)
            res['rows'].append({'condition': cond, 'control': ctrl, 'seed': seed, 'subject': s, 'restricted': restricted, 'n': len(idx),
                                'N_eff': Ne, 'chance_top1': float(np.mean(1 / (m.sum(1) if restricted else np.full(len(idx), N)))),
                                'top1': t1, 'two_way': two, 'bits': wolpaw_bits(t1, Ne)})
    for s in subjects:
        q = (subj == s) & (sp == 'test'); sent_q = sent[q]; idx = np.array([tpos[g] for g in sent_q]); Xq = X[q]
        if q.sum() < 5: continue
        own_dev = (subj == s) & (sp == 'dev'); di = np.array([dpos[g] for g in sent[own_dev]])
        trainsets = {'within': (subj == s) & (sp == 'train'), 'loso_unseen': (subj != s) & (sp == 'train'), 'loso_seen': (subj != s)}
        for cond, trm in trainsets.items():
            P = ridge_predict(X[trm], E[sent[trm]], [X[own_dev], Xq], ALPHAS)
            best = max(ALPHAS, key=lambda a: ident(P[a][0], E[dev_g], di)[1])
            add(cond, s, P[best][1], E[test_g], idx, 0, 'real')
            for sd in SEEDS:
                add(cond, s, P[best][1], E[test_g], np.random.default_rng(sd).permutation(idx), sd, 'shuffled_pairs')
                rng = np.random.default_rng(100 + sd); Pn = ridge_predict(rng.standard_normal(X[trm].shape), E[sent[trm]], [rng.standard_normal(Xq.shape)], [best])
                add(cond, s, Pn[best][0], E[test_g], idx, sd, 'noise')
        # template: other subjects' mean EEG per test sentence (z-scored on other subjects' rows)
        oth = subj != s; mu, sd_ = X[oth].mean(0), X[oth].std(0) + 1e-6
        T = np.stack([((X[oth & (sent == g)] - mu) / sd_).mean(0) if (oth & (sent == g)).any() else np.zeros(X.shape[1]) for g in test_g])
        T = T - T.mean(0); Zq = (Xq - mu) / sd_ - T.mean(0)
        add('template', s, Zq, T, idx, 0, 'real')
        for sd in SEEDS:
            add('template', s, Zq, T, np.random.default_rng(sd).permutation(idx), sd, 'shuffled_pairs')
            add('template', s, np.random.default_rng(100 + sd).standard_normal(Zq.shape), T, idx, sd, 'noise')
        print(f'[T3] subject {s} done ({time.time()-t0:.0f}s)', flush=True)
    for cond in ('within', 'loso_unseen', 'loso_seen', 'template'):
        for restricted in (False, True):
            for ctrl in ('real', 'shuffled_pairs', 'noise'):
                rr = [r for r in res['rows'] if r['condition'] == cond and r['restricted'] == restricted and r['control'] == ctrl]
                res['summary'][f'{cond}|{"restricted" if restricted else "all"}|{ctrl}'] = {k: ci([r[k] for r in rr]) for k in ('top1', 'two_way', 'bits', 'N_eff', 'chance_top1')}
    res['meta']['runtime_s'] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, 'T3_SR.json'), 'w'), indent=1)
    print('| condition | pool | N_eff | chance top-1 | top-1 real | top-1 noise | 2-way real | 2-way noise | 2-way shuffled | bits real | bits noise |')
    for cond in ('within', 'loso_unseen', 'loso_seen', 'template'):
        for pool in ('all', 'restricted'):
            R, Nn, Sh = (res['summary'][f'{cond}|{pool}|{c}'] for c in ('real', 'noise', 'shuffled_pairs'))
            print(f"| {cond} | {pool} | {R['N_eff'][0]:.1f} | {R['chance_top1'][0]:.3f} | {fmt(R['top1'])} | {fmt(Nn['top1'])} | {fmt(R['two_way'])} | {fmt(Nn['two_way'])} | {fmt(Sh['two_way'])} | {fmt(R['bits'])} | {fmt(Nn['bits'])} |")

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True); stage = sys.argv[1]
    if stage == 'cache': build_cache(sys.argv[2])
    elif stage == 'T1': run_T1(sys.argv[2])
    elif stage == 'T2': run_T2()
    elif stage == 'T3': run_T3()
