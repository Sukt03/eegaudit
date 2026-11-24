"""Text rows of the BASELINE YIELD CARD (IMPLEMENTER, 21 Nov 2026; METHOD B3 PASS 08:45 + IMPLEMENTER G1 spec in EXTENSIONS_LOOP).
Models: BrainTranslator (Wang & Ji 2022, our BART-base re-implementation), DeWave-style (our simplified re-implementation), GLIM
(released generations only: EEG vs test-time noise). Arms per model: eeg | noise (trained on N(0,1), identical recipe; Jo et al. 2024) |
random_init (BART re-initialised, identical recipe) | shuffled (DeWave only). Free-running decoding only for the yield (greedy, beam);
teacher-forced numbers are printed as the published-protocol metric, never as a yield.
Estimator (CHANGE 13, sentence identity): MiniLM embedding of each generation vs the MiniLM embeddings of the candidate test sentences;
InfoNCE log-score = log2 N - cross-entropy of a softmax over cosines, temperature chosen per sentence on the OTHER sentences' records
(leave-one-sentence-out, 40-point grid). Unit: bits per record (one participant reading one sentence). Candidate sets: pooled test set
and within-task sets. Floors: shuffled pairs (generations permuted across records). Reference: length-only read-out through the same
estimator. Paired EEG - twin difference per record, CI by sentence-cluster bootstrap (5000, one GPU tensor op).
Leakage audit: sentence overlap across tasks and across train/dev/test of the reference 80/10/10 sentence-index split (per task, pooled
as in Wang & Ji) vs OUR split (cross-task duplicates removed, prep_cache.py), exact and normalised text.
All numerics in torch on the GPU. Usage: python -I -u text_yield_card.py --out results/baselines/text"""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()  # every result JSON carries the exact code hash
import argparse, csv, json, math, os, pickle, re, sys, time
from collections import Counter, defaultdict
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='/path/to/eegaudit/results/baselines/text')
ap.add_argument('--cache', default='/path/to/eegaudit/results/track_text/brain_translator/cache_full.pkl')
ap.add_argument('--minilm', default='/path/to/eegaudit/data_raw/zuco/minilm')
ap.add_argument('--n_boot', type=int, default=5000)
ap.add_argument('--smoke', action='store_true', help='first 300 records per arm only')
ap.add_argument('--embed', choices=['minilm', 'tfidf'], default='minilm', help='N5: tfidf = lexical read-out (TF-IDF fitted on TRAINING sentences only, sublinear tf, l2) beside MiniLM')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = torch.device('cuda')
torch.manual_seed(0)
t0 = time.time()
R = '/path/to/eegaudit/results'
BT = f'{R}/track_text/brain_translator'; DW = f'{R}/loop/job0_dewave'; RB = f'{R}/baselines/text'
# FIXER (METHOD R2 ruling 5): EXPLICIT tag -> arm table (printed into the JSON as 'tag_map'). One row per run prefix:
# (model, run, arm, prefix, recipe note). run = seed, or '<seed>_<label>' for a REPLICATE of an already-present seed with the identical
# recipe (scored and printed, but not counted as an extra seed). Every identically-run seed is included; no seed selection.
# random_init = EEG encoder frozen at init (METHOD G1 a, PRIMARY); lm_prior_ablation = BART re-initialised (appendix only).
TAGS = []
def _t(model, run, arm, prefix, note): TAGS.append(dict(model=model, run=str(run), arm=arm, prefix=prefix, recipe=note))
_t('braintranslator', 0, 'eeg', f'{BT}/full_eeg', 'train_bt eeg, pretrained BART, seed 0 (track run)')
_t('braintranslator', 0, 'noise', f'{BT}/full_noise', 'train_bt noise, seed 0 (track run)')
for s_ in (0, 1, 2):
    _t('braintranslator', s_, 'random_init', f'{RB}/bt_frozen_eeg_s{s_}', f'train_bt --freeze_encoder, seed {s_}')
    _t('braintranslator', s_, 'lm_prior_ablation', f'{RB}/bt_random_eeg' + ('' if s_ == 0 else f'_s{s_}'), f'train_bt --bart_init random, seed {s_}')
for s_ in (1, 2, 3, 4):
    _t('braintranslator', s_, 'eeg', f'{RB}/bt_eeg_s{s_}', f'train_bt eeg, pretrained BART, seed {s_}'); _t('braintranslator', s_, 'noise', f'{RB}/bt_noise_s{s_}', f'train_bt noise, seed {s_}')
# bt_eeg_s0 = a byte-identical COPY of track_text/brain_translator/full_eeg (checked below): an alias of seed-0 eeg, scored once
ALIASES = {'bt_eeg_s0': f'{BT}/full_eeg'}
for arm, pre, note in (('eeg', f'{RB}/bt_eeg_s0_ck', 'train_bt eeg seed 0, checkpointed rerun (reimpl2)'), ('noise', f'{BT}/full_noise', 'seed-0 noise twin'),
                       ('lm_prior_ablation', f'{RB}/bt_rand_s0_ck', 'train_bt --bart_init random seed 0, checkpointed rerun (reimpl2)')):
    _t('braintranslator', '0_ck', arm, pre, note)
_t('dewave_style', 0, 'eeg', f'{DW}/full_eeg', 'train_dewave eeg seed 0 (job0)'); _t('dewave_style', 0, 'noise', f'{DW}/full_noise', 'train_dewave noise seed 0 (job0)')
_t('dewave_style', 0, 'shuffled', f'{DW}/full_shuffled', 'train_dewave shuffled seed 0 (job0)')
for s_ in (0, 1, 2):
    _t('dewave_style', s_, 'random_init', f'{RB}/dw_frozen_eeg_s{s_}', f'train_dewave --freeze_encoder, seed {s_}')
    _t('dewave_style', s_, 'lm_prior_ablation', f'{RB}/dw_random_eeg' + ('' if s_ == 0 else f'_s{s_}'), f'train_dewave --bart_init random, seed {s_}')
for s_ in (1, 2, 3, 4):
    _t('dewave_style', s_, 'eeg', f'{RB}/dw_eeg_s{s_}', f'train_dewave eeg seed {s_}'); _t('dewave_style', s_, 'noise', f'{RB}/dw_noise_s{s_}', f'train_dewave noise seed {s_}')
for arm, pre, note in (('eeg', f'{DW}/full_eeg', 'seed-0 eeg (job0)'), ('lm_prior_ablation', f'{RB}/dw_rand_s0_ck', 'train_dewave --bart_init random seed 0, checkpointed rerun (reimpl2)')):
    _t('dewave_style', '0_ck', arm, pre, note)
for s_ in (42, 43, 44):
    _t('glim_style', s_, 'eeg', f'{RB}/glim_eeg_s{s_}', f'glim_style eeg --epochs 40 seed {s_}'); _t('glim_style', s_, 'noise', f'{RB}/glim_noise_s{s_}', f'glim_style noise --epochs 40 seed {s_}')
    _t('glim_style', s_, 'random_init', f'{RB}/glim_frozen_eeg_s{s_}', f'glim_style --init random --epochs 40 seed {s_}')
# glim_*_r: a DIFFERENT recipe (200 epochs, default seed 42) -> its own model row, never pooled with the 40-epoch seeds
_t('glim_style_ep200', 42, 'eeg', f'{RB}/glim_eeg_r', 'glim_style eeg, 200 epochs, seed 42 (reimpl)'); _t('glim_style_ep200', 42, 'noise', f'{RB}/glim_noise_r', 'glim_style noise, 200 epochs, seed 42 (reimpl)')
_t('glim_style_ep200', 42, 'random_init', f'{RB}/glim_rand_r', 'glim_style --init random, 200 epochs, seed 42 (reimpl)')
for s_ in (0, 1, 2):
    _t('eeg2text_style', s_, 'eeg', f'{RB}/e2t_eeg_s{s_}', f'e2t_style eeg seed {s_}'); _t('eeg2text_style', s_, 'noise', f'{RB}/e2t_noise_s{s_}', f'e2t_style noise seed {s_}')
    _t('eeg2text_style', s_, 'random_init', f'{RB}/e2t_frozen_eeg_s{s_}', f'e2t_style --freeze_encoder seed {s_}')
for arm, pre, note in (('eeg', f'{RB}/e2t_eeg_s0', 'seed-0 eeg'), ('noise', f'{RB}/e2t_noise_s0', 'seed-0 noise'),
                       ('random_init', f'{RB}/e2t_rand_s0', 'e2t_style --freeze_encoder seed 0 (reimpl tag e2t_rand_s0; same recipe as e2t_frozen_eeg_s0)')):
    _t('eeg2text_style', '0_rand', arm, pre, note)
for arm, pre, note in (('eeg', f'{RB}/e2t_eeg_s0_ck', 'e2t_style eeg seed 0, checkpointed rerun (reimpl2)'), ('noise', f'{RB}/e2t_noise_s0', 'seed-0 noise'),
                       ('random_init', f'{RB}/e2t_rand_s0', 'seed-0 frozen twin (e2t_rand_s0)')):
    _t('eeg2text_style', '0_ck', arm, pre, note)
MODELS = {}
for r_ in TAGS: MODELS.setdefault(r_['model'], {}).setdefault(r_['run'], {})[r_['arm']] = r_['prefix']
REPLICATE = {(r_['model'], f"seed{r_['run']}") for r_ in TAGS if '_' in r_['run']}
for r_ in TAGS: r_['generations_exist'] = os.path.exists(r_['prefix'] + '_generations.jsonl'); r_['replicate_of_existing_seed'] = '_' in r_['run']
_mapped = {os.path.basename(r_['prefix']) for r_ in TAGS if r_['prefix'].startswith(RB)}
ALIAS_CHECK = {k: (open(f'{RB}/{k}_generations.jsonl', 'rb').read() == open(v + '_generations.jsonl', 'rb').read()) for k, v in ALIASES.items() if os.path.exists(f'{RB}/{k}_generations.jsonl')}
assert all(ALIAS_CHECK.values()), f'alias is not a byte-identical copy: {ALIAS_CHECK}'
_mapped |= set(ALIASES)
UNMAPPED = sorted(f[:-len('_generations.jsonl')] for f in os.listdir(RB) if f.endswith('_generations.jsonl') and f[:-len('_generations.jsonl')] not in _mapped)
print('TAG MAP:', len(TAGS), 'rows,', sum(r_['generations_exist'] for r_ in TAGS), 'with generations; UNMAPPED tags in', RB, ':', UNMAPPED or 'none', flush=True)
GLIM = '/path/to/eegaudit/data_raw/third_party/glim_results'
TAUS = torch.logspace(-2.5, 0, int(os.environ.get('CARD_NTAU', '12')), device=dev)  # FIXER L1 (LOCK §3): 12-point LOO grid PRIMARY; CARD_NTAU=40 = sensitivity
TAUS40 = torch.logspace(-2.5, 0, 40, device=dev)  # labelled sensitivity field (*_t40)
TAUS_LEN = torch.logspace(-1, 2, 40, device=dev)  # length-only reference: integer gaps need a wider grid
LOG2 = math.log(2)


def norm_text(s):
    return ' '.join(re.findall(r"[a-z0-9]+", s.lower()))


# ---------------- leakage audit (texts only; blocks from prep_full.log: SR 0-399, NR 400-699, NR2 700-1048)
D = pickle.load(open(a.cache, 'rb')); texts = D['texts']
blocks = {'SR': (0, 400), 'NR': (400, 700), 'NR2': (700, 1049)}
assert len(texts) == 1049, len(texts)
ref_split, task_of = {}, {}
for t, (s, e) in blocks.items():
    n = e - s
    for j, g in enumerate(range(s, e)):
        task_of[g] = t; ref_split[g] = 'train' if j < int(0.8 * n) else ('dev' if j < int(0.9 * n) else 'test')
our_split = {r['g']: r['split'] for r in D['recs']}
test_recs_by_g = Counter(r['g'] for r in D['recs'] if r['split'] == 'test')
del D


def audit(split, key):
    by = defaultdict(lambda: {'tasks': set(), 'splits': set(), 'gs': []})
    for g, sp in split.items():
        k = key(texts[g]); by[k]['tasks'].add(task_of[g]); by[k]['splits'].add(sp); by[k]['gs'].append(g)
    multi_task = [k for k, v in by.items() if len(v['tasks']) > 1]
    multi_split = [k for k, v in by.items() if len(v['splits']) > 1]
    test_leak = [k for k, v in by.items() if 'test' in v['splits'] and len(v['splits']) > 1]
    return dict(n_sentence_slots=len(split), n_unique=len(by), in_more_than_one_task=len(multi_task), in_more_than_one_split=len(multi_split),
                test_sentences_also_in_train_or_dev=len(test_leak), test_leak_examples=[by[k]['gs'] for k in test_leak[:5]],
                leaked_keys=set(test_leak))


leak = {}
for name, sp in (('reference_wang_ji_80_10_10', ref_split), ('ours_dedup', our_split)):
    for kn, key in (('exact', lambda s: s.strip()), ('normalised', norm_text)):
        leak[f'{name}/{kn}'] = audit(sp, key)
leaked_norm_ours = leak['ours_dedup/normalised'].pop('leaked_keys')
for v in leak.values():
    v.pop('leaked_keys', None)
print('LEAKAGE', json.dumps(leak), flush=True)

# ---------------- MiniLM on the GPU (same model and pooling as pilot_retrieval.embed_sentences)
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained(a.minilm); mdl = AutoModel.from_pretrained(a.minilm).to(dev).eval()
_cache = {}


if a.embed == 'tfidf':
    from sklearn.feature_extraction.text import TfidfVectorizer
    _vec = TfidfVectorizer(sublinear_tf=True, token_pattern=r"[a-z0-9]+(?:'[a-z]+)?").fit([texts[g] for g, sp in our_split.items() if sp == 'train'])
    def emb(ts):
        X = torch.from_numpy(_vec.transform([t.lower() for t in ts]).toarray()).float().to(dev)
        return torch.nn.functional.normalize(X, dim=1)
    _emb_minilm = None
else:
    _emb_minilm = True


@torch.no_grad()
def _minilm_emb(ts):
    new = [t for t in dict.fromkeys(ts) if t not in _cache]
    for i in range(0, len(new), 256):
        b = tok(new[i:i + 256], padding=True, truncation=True, max_length=128, return_tensors='pt').to(dev)
        h = mdl(**b).last_hidden_state; m = b['attention_mask'].unsqueeze(-1).float()
        e = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=1)
        _cache.update(zip(new[i:i + 256], e))
    return torch.stack([_cache[t] for t in ts])


if _emb_minilm:
    emb = _minilm_emb


def toks(s):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", s.lower())


def infonce(S, true, cl, taus=None):
    """S (Q, N) similarities, true (Q,) index, cl (Q,) sentence cluster id. Per-record log-score in bits with the temperature chosen per
    cluster on all OTHER clusters' records (leave-one-sentence-out)."""
    taus = TAUS if taus is None else taus
    Z = S[None] / taus[:, None, None]
    lp = Z - torch.logsumexp(Z, -1, keepdim=True)
    CE = -lp[:, torch.arange(len(true), device=dev), true] / LOG2          # (n_tau, Q)
    ncl = int(cl.max()) + 1
    csum = torch.zeros(len(taus), ncl, device=dev).index_add_(1, cl, CE); ccnt = torch.bincount(cl, minlength=ncl).float()
    loo = (CE.sum(1, keepdim=True) - csum) / (len(true) - ccnt).clamp(min=1)  # (n_tau, ncl)
    t_idx = loo.argmin(0)[cl]
    return math.log2(S.shape[1]) - CE[t_idx, torch.arange(len(true), device=dev)]


def ident(S, true):
    t = S[torch.arange(len(true), device=dev), true][:, None]
    beat = (S > t).sum(1).float(); tie = (S == t).sum(1).float() - 1
    return torch.where(beat == 0, 1.0 / (1 + tie), torch.zeros_like(beat)), 1 - (beat + 0.5 * tie) / (S.shape[1] - 1)


def cboot(vals, cl, n_boot=a.n_boot):
    """vals: dict name -> (Q,) tensor; cluster bootstrap of the mean, returns [2.5, 97.5] per name. One tensor op."""
    ncl = int(cl.max()) + 1
    pick = torch.randint(0, ncl, (n_boot, ncl), device=dev)
    w = torch.zeros(n_boot, ncl, device=dev).scatter_add_(1, pick, torch.ones_like(pick, dtype=torch.float))  # cluster multiplicity
    wr = w[:, cl]                                                                                                # (n_boot, Q)
    out = {}
    for k, v in vals.items():
        m = (wr * v[None]).sum(1) / wr.sum(1)
        out[k] = [float(torch.quantile(m, 0.025)), float(torch.quantile(m, 0.975))]
    return out


def score_set(rows_by_arm, mode, sel_task=None, exclude=None):
    """rows_by_arm: arm -> list of dicts with g/target/task/<mode>; all arms list the same records in the same order."""
    arms = list(rows_by_arm)
    base = rows_by_arm[arms[0]]
    keep = [i for i, r in enumerate(base) if (sel_task is None or r['task'] == sel_task) and (exclude is None or norm_text(r['target']) not in exclude)]
    if len(keep) < 10:
        return None
    tg = [base[i]['target'] for i in keep]
    U = list(dict.fromkeys(tg)); pos = {u: j for j, u in enumerate(U)}
    true = torch.tensor([pos[t] for t in tg], device=dev); cl = true.clone(); N = len(U)
    EU = emb(U)
    lu = torch.tensor([len(toks(u)) for u in U], device=dev, dtype=torch.float)
    res = {'n_records': len(keep), 'n_candidates': N, 'arms': {}}
    per = {}
    g = torch.Generator(device=dev); g.manual_seed(0)
    for arm in arms:
        hyp = [rows_by_arm[arm][i][mode] or '.' for i in keep]
        assert [rows_by_arm[arm][i]['target'] for i in keep] == tg
        S = emb(hyp) @ EU.T
        y = infonce(S, true, cl); t1, tw = ident(S, true)
        perm = torch.randperm(len(keep), device=dev, generator=g)
        y_sh = infonce(S[perm], true, cl)
        nk = f'noise_input_{mode}'
        if all(nk in rows_by_arm[arm][i] for i in keep):    # LOCK §4 noise-input floor: the trained model fed N(0,1) at test time
            Sn = emb([rows_by_arm[arm][i][nk] or '.' for i in keep]) @ EU.T; y_ni = infonce(Sn, true, cl)
        else:
            y_ni = None  # shuffled pairs: generations permuted across records
        lh = torch.tensor([len(toks(h)) for h in hyp], device=dev, dtype=torch.float)
        S_len = -(lh[:, None] - lu[None]).abs() + 1e-3 * torch.rand(len(keep), N, device=dev, generator=g)  # tiny jitter breaks ties
        y_len = infonce(S_len, true, cl, taus=TAUS_LEN)  # length gaps are integers: wider temperature grid
        per[arm] = y
        res['arms'][arm] = dict(yield_bits=float(y.mean()), yield_bits_t40=float(infonce(S, true, cl, taus=TAUS40).mean()), top1=float(t1.mean()), two_way=float(tw.mean()), chance_top1=1 / N,
                                floor_shuffled_bits=float(y_sh.mean()), length_only_bits=float(y_len.mean()), n_unique_generations=len(set(hyp)),
                                yield_minus_length_only=float((y - y_len).mean()))
        res['arms'][arm]['noise_input_floor_bits'] = float(y_ni.mean()) if y_ni is not None else 'not available (no saved weights for this run)'
        res['arms'][arm]['ci95'] = cboot({'y': y, 'sh': y_sh, 'yl': y - y_len}, cl)
    for arm in arms:
        if arm == 'eeg':
            continue
        d = per['eeg'] - per[arm]
        res[f'eeg_minus_{arm}'] = dict(mean=float(d.mean()), ci95=cboot({'d': d}, cl)['d'])
    return res


def bleu_share(rows_by_arm, mode, twin='noise'):
    """Corpus BLEU-1 (train_bt.py definition) per arm, and the EEG share (BLEU_eeg - BLEU_twin)/BLEU_eeg with a sentence-cluster bootstrap."""
    def counts(rows):
        m, hl, rl = [], [], []
        for r in rows:
            h, t = toks(r[mode]), toks(r['target']); ch, ct = Counter(h), Counter(t)
            m.append(sum(min(c, ct[w]) for w, c in ch.items())); hl.append(len(h)); rl.append(len(t))
        return [torch.tensor(x, device=dev, dtype=torch.float) for x in (m, hl, rl)]
    base = rows_by_arm['eeg']; U = list(dict.fromkeys(r['target'] for r in base)); pos = {u: j for j, u in enumerate(U)}
    cl = torch.tensor([pos[r['target']] for r in base], device=dev); ncl = len(U)
    pick = torch.randint(0, ncl, (a.n_boot, ncl), device=dev)
    w = torch.zeros(a.n_boot, ncl, device=dev).scatter_add_(1, pick, torch.ones_like(pick, dtype=torch.float))[:, cl]
    def bleu(c, wt=None):
        m, hl, rl = c
        if wt is None:
            M, H, Rr = m.sum(), hl.sum(), rl.sum()
        else:
            M, H, Rr = (wt * m).sum(1), (wt * hl).sum(1), (wt * rl).sum(1)
        bp = torch.where(H > Rr, torch.ones_like(H), torch.exp(1 - Rr / H.clamp(min=1)))
        return bp * M / H.clamp(min=1)
    ce, ct = counts(rows_by_arm['eeg']), counts(rows_by_arm[twin])
    be, bt = bleu(ce), bleu(ct); bse, bst = bleu(ce, w), bleu(ct, w); sh = (bse - bst) / bse
    return dict(bleu1_eeg=float(be), bleu1_twin=float(bt), eeg_share=float((be - bt) / be),
                eeg_share_ci95=[float(torch.quantile(sh, 0.025)), float(torch.quantile(sh, 0.975))])


def load_jsonl(p):
    rows = [json.loads(l) for l in open(p)]
    for r in rows:
        r['task'] = r['subject'].split(':')[0]
    return rows[:300] if a.smoke else rows


out = {'note': __doc__, 'tau_grid_points': len(TAUS), 'tag_map': TAGS, 'tag_aliases_identical_copies': ALIASES, 'unmapped_tags': UNMAPPED, 'leakage': leak, 'test_records_per_sentence_ours': dict(Counter(test_recs_by_g.values())), 'models': {}}
for model, by_seed in MODELS.items():
  out['models'][model] = {}
  for seed, arms in by_seed.items():
    rows = {k: load_jsonl(p + '_generations.jsonl') for k, p in arms.items() if os.path.exists(p + '_generations.jsonl')}
    missing = [k for k, p in arms.items() if not os.path.exists(p + '_generations.jsonl')]
    if 'eeg' not in rows or len(rows) < 2:
        out['models'][model][f'seed{seed}'] = {'arms_missing': missing}; continue
    order = [r['g'] for r in rows['eeg']]
    for k in rows:
        assert [r['g'] for r in rows[k]] == order, (model, seed, k)
    pub = {k: json.load(open(p + '_results.json')) for k, p in arms.items() if os.path.exists(p + '_results.json')}
    M = {'arms_present': list(rows), 'arms_missing': missing, 'n_records_per_arm': {k: len(v) for k, v in rows.items()},   # FIXER: n per arm
         'published_protocol_metrics': {k: {m: {'bleu1': v['modes'][m]['bleu1'], 'rouge1_f': v['modes'][m]['rouge1']['f']} for m in v['modes']}
                                        for k, v in pub.items()},
         'best_epoch': {k: v.get('best_epoch') for k, v in pub.items()}, 'yield': {}}
    for mode in ('greedy', 'beam'):
        M['yield'][mode] = {'pooled': score_set(rows, mode), 'pooled_label': 'includes task identity', 'within_label': 'within-task rows = the C1 test'}   # METHOD 17:20 labels
        for t in ('SR', 'NR', 'NR2'):
            M['yield'][mode][f'within_{t}'] = score_set(rows, mode, sel_task=t)
        M['yield'][mode]['pooled_excluding_leaked'] = score_set(rows, mode, exclude=leaked_norm_ours) if leaked_norm_ours else 'no leaked sentence in our split'
        if 'noise' in rows:
            M['yield'][mode]['bleu_share_vs_noise'] = bleu_share(rows, mode)
    M['yield']['teacher_forced_NOT_A_DECODER_OUTPUT'] = {'pooled': score_set(rows, 'teacher_forced')}
    out['models'][model][f'seed{seed}'] = M
    print(model, f'seed{seed}', 'best_epoch', M['best_epoch'], 'n per arm', M['n_records_per_arm'], flush=True)
    p = M['yield']['beam']['pooled']
    print(model, f'seed{seed}', 'beam pooled', {k: round(v['yield_bits'], 4) for k, v in p['arms'].items()},
          {k: v for k, v in p.items() if k.startswith('eeg_minus')}, f'({time.time()-t0:.0f}s)', flush=True)

# ---------------- cross-seed summary: mean over seeds, seed t-interval (95 %), and each seed's sentence-cluster CI
from scipy import stats as _st
def _tci(v):
    v = np.asarray(v, float)
    if len(v) < 2: return None
    h = _st.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)); return [float(v.mean() - h), float(v.mean() + h)]
out['summary_over_seeds'] = {}
for model in MODELS:
    seeds = {k: v for k, v in out['models'][model].items() if 'yield' in v and (model, k) not in REPLICATE}   # FIXER: replicates of a seed are printed per run, not counted twice
    if not seeds: continue
    S = {'n_seeds': len(seeds), 'seeds': list(seeds), 'replicates_not_counted': [k for k in out['models'][model] if (model, k) in REPLICATE and 'yield' in out['models'][model][k]]}
    for mode in ('greedy', 'beam'):
        for cset in ('pooled', 'within_SR', 'within_NR', 'within_NR2'):
            per = [v['yield'][mode][cset] for v in seeds.values() if v['yield'][mode].get(cset)]
            if not per: continue
            arms = sorted({k for p_ in per for k in p_['arms']})
            row = {f'{arm}_yield_bits': [p_['arms'][arm]['yield_bits'] for p_ in per if arm in p_['arms']] for arm in arms}
            row.update({f'{arm}_floor_bits': [p_['arms'][arm]['floor_shuffled_bits'] for p_ in per if arm in p_['arms']] for arm in arms})
            row['length_only_bits'] = [p_['arms']['eeg']['length_only_bits'] for p_ in per]
            for k in [k for k in per[0] if k.startswith('eeg_minus_')]:
                vals = [p_[k]['mean'] for p_ in per if k in p_]
                row[k] = {'per_seed': vals, 'mean': float(np.mean(vals)), 'seed_t_ci95': _tci(vals), 'per_seed_cluster_ci95': [p_[k]['ci95'] for p_ in per if k in p_]}
            S[f'{mode}/{cset}'] = {k: ({'per_seed': v, 'mean': float(np.mean(v)), 'seed_t_ci95': _tci(v)} if isinstance(v, list) else v) for k, v in row.items()}
        bl = [v['yield'][mode].get('bleu_share_vs_noise') for v in seeds.values() if v['yield'][mode].get('bleu_share_vs_noise')]
        if bl:
            S[f'{mode}/bleu1'] = {k: {'per_seed': [b_[k] for b_ in bl], 'mean': float(np.mean([b_[k] for b_ in bl])), 'seed_t_ci95': _tci([b_[k] for b_ in bl])}
                                  for k in ('bleu1_eeg', 'bleu1_twin', 'eeg_share')}
    tf = [v['published_protocol_metrics'] for v in seeds.values()]
    S['teacher_forced_bleu1'] = {arm: [t[arm]['teacher_forced']['bleu1'] for t in tf if arm in t] for arm in ('eeg', 'noise', 'random_init')}
    out['summary_over_seeds'][model] = S
    b_ = S.get('beam/pooled', {})
    print('SUMMARY', model, len(seeds), 'seeds | beam pooled', {k: (round(v['mean'], 4), v.get('seed_t_ci95')) for k, v in b_.items() if isinstance(v, dict) and 'mean' in v and ('yield' in k or 'minus' in k)}, flush=True)

# ---------------- GLIM released generations: EEG vs test-time noise (released arms only; no trained twin exists)
g_rows = {}
for arm, fn in (('eeg', 'wandb_export_gen_samples_glim.csv'), ('noise_test_time', 'wandb_export_gen_samples_noise_input_test.csv')):
    rr = list(csv.DictReader(open(os.path.join(GLIM, fn), encoding='utf-8')))
    occ = defaultdict(int); keyed = {}
    for r in rr:
        k = (r['Task'], r['Dataset'], r['Subject'], r['Input Text']); occ[k] += 1
        keyed[k + (occ[k],)] = {'target': r['Input Text'], 'gen': r['Generated Text'], 'task': r['Task'] + r['Dataset'], 'bleu1': float(r['Bleu1'])}
    g_rows[arm] = keyed
common = sorted(set(g_rows['eeg']) & set(g_rows['noise_test_time']))
rows = {arm: [dict(g_rows[arm][k], g=i) for i, k in enumerate(common)] for arm in g_rows}
G = {'n_common_records': len(common), 'published_protocol_bleu1_mean_per_sample': {arm: float(np.mean([r['bleu1'] for r in rows[arm]])) for arm in rows},
     'yield': {'free_running': {'pooled': score_set(rows, 'gen')}}}
for t in sorted({r['task'] for r in rows['eeg']}):
    G['yield']['free_running'][f'within_{t}'] = score_set(rows, 'gen', sel_task=t)
AUD = '/path/to/eegaudit/results/baselines/text/glim_split_audit.json'
if os.path.exists(AUD):  # METHOD condition 2: re-score without test sentences whose text also occurs under another GLIM text uid
    lk = {norm_text(t) for t in json.load(open(AUD))['leaked_test_texts']}
    G['yield']['free_running']['pooled_excluding_glim_cross_uid_texts'] = score_set(rows, 'gen', exclude=lk); G['n_excluded_texts'] = len(lk)
out['models']['glim_released'] = G
print('glim pooled', {k: round(v['yield_bits'], 4) for k, v in G['yield']['free_running']['pooled']['arms'].items()}, flush=True)
out['runtime_s'] = time.time() - t0
fn = os.path.join(a.out, ('text_yield_card_smoke' if a.smoke else 'text_yield_card') + ('_tfidf' if a.embed == 'tfidf' else '') + '.json')
out['code_sha256'] = CODE_SHA256; json.dump(out, open(fn, 'w'), indent=1)
print('wrote', fn, f'{time.time()-t0:.0f}s', flush=True)
