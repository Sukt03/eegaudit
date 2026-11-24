"""G2 for the GLIM-style re-implementation under GLIM's OWN data/eval protocol (IMPLEMENTER, 21 Nov 2026; coordinator "reimplement again").
Same model as glim_style.py (GLIM modules.py architecture, frozen flan-t5-large). Protocol from GLIM data/__STEP4_eeg_text_match_split.ipynb and
data/datamodule.py (both read): split by `text uid` per (dataset, task) with uids shared by > 1 label row put in train, unique uids permuted
with default_rng(42) and cut at k = 0.8 - 0.2 (n - a) / a and (a + k a) / 2 (set-iteration order of Python ints reproduced as in the
notebook); TRAINING records expanded x 8, one per LLM paraphrase target (lexical simplification v0/v1, semantic clarity v0/v1, syntax
simplification v0/v1, naive rewritten, naive simplified) from GLIM's label table; T5 input text = "To English: <sentence>"; batches hold
distinct text uids (their GLIMSampler; here: greedy de-duplication of a shuffled order); evaluation = 2-beam generation scored by per-sample
torchmetrics BLEU-1 against the 8 variants (BLEU1@MTV, their cal_gen_metrics), averaged; also vs the raw sentence.
Not reproducible here (stated): raw 128-ch sentence EEG (~97 GB, not downloaded) -> our word-level band features; TSR tasks (not in our pickles);
compute: GLIM = 49,600 steps x 576 samples; here --max_steps x 72 (default 20,000 x 72). G2 target = the released GLIM generations'
mean per-sample BLEU1@MTV (data_raw/third_party/glim_results)."""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()  # every result JSON carries the exact code hash
import argparse, json, math, os, pickle, re, sys, time
from collections import Counter
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument('--cache', default='/path/to/eegaudit/results/track_text/brain_translator/cache_full.pkl')
ap.add_argument('--out_dir', default='/path/to/eegaudit/results/baselines/text'); ap.add_argument('--tag', required=True)
ap.add_argument('--input', choices=['eeg', 'noise'], default='eeg'); ap.add_argument('--init', choices=['trained', 'random'], default='trained')
ap.add_argument('--t5', default='/path/to/eegaudit/data_raw/third_party/flan_t5_large')
ap.add_argument('--epochs', type=int, default=200); ap.add_argument('--bs', type=int, default=72); ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--seed', type=int, default=42); ap.add_argument('--max_steps', type=int, default=20000); ap.add_argument('--max_train', type=int, default=None); ap.add_argument('--max_eval', type=int, default=None)
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device('cuda'); t0 = time.time()
os.makedirs(a.out_dir, exist_ok=True); logf = open(os.path.join(a.out_dir, f'{a.tag}_train_log.txt'), 'w')
def log(*s):
    m = ' '.join(str(x) for x in s); print(m, flush=True); logf.write(m + '\n'); logf.flush()

D = pickle.load(open(a.cache, 'rb')); texts, recs = D['texts'], D['recs']; del D
import io, pickletools, re as _re, csv as _csv
LT = '/path/to/eegaudit/data_raw/third_party/glim_repo/zuco_label_8variants.df'
_g = set(); _p = []; _m = {}
for op, arg, _ in pickletools.genops(io.BytesIO(open(LT, 'rb').read())):   # same global scan as glim_split_audit.py (pandas/numpy only)
    if op.name in ('SHORT_BINUNICODE', 'BINUNICODE', 'UNICODE', 'BINUNICODE8'): _p.append(arg)
    elif op.name == 'MEMOIZE': _m[len(_m)] = _p[-1] if _p else None
    elif op.name in ('BINGET', 'LONG_BINGET', 'GET'): _p.append(_m.get(arg))
    elif op.name == 'STACK_GLOBAL': _g.add(f'{_p[-2]}.{_p[-1]}'); _p.append(None)
    elif op.name == 'GLOBAL': _g.add(arg.replace(' ', '.')); _p.append(None)
    else: _p.append(None)
assert all(x.startswith(('pandas.', 'numpy.', 'builtins.slice')) for x in _g), _g
import pandas as pd
LTdf = pd.read_pickle(LT)
VARS = ['lexical simplification (v0)', 'lexical simplification (v1)', 'semantic clarity (v0)', 'semantic clarity (v1)', 'syntax simplification (v0)', 'syntax simplification (v1)', 'naive rewritten', 'naive simplified']
norm = lambda t: ' '.join(_re.findall(r"[a-z0-9]+", str(t).lower()))
uid_counts = LTdf['text uid'].value_counts(); dup = set(uid_counts[uid_counts > 1].index.tolist())
phase_of_uid = {}
for (dset, task), grp in LTdf.groupby(['dataset', 'task']):
    text_uids = grp['text uid'].values; uset = list(set(text_uids.tolist()))
    uniq = [u for u in uset if u not in dup]; nn_, aa = len(uset), len(uniq); k = 0.8 - 0.2 * (nn_ - aa) / aa
    sh = np.random.default_rng(seed=42).permutation(uniq); trn, val, tst = np.split(sh, [int(k * aa), int((aa + k * aa) / 2)])
    for u in trn.tolist() + [u for u in uset if u in dup]: phase_of_uid[u] = 'train'
    for u in val.tolist(): phase_of_uid.setdefault(u, 'val')
    for u in tst.tolist(): phase_of_uid.setdefault(u, 'test')
DS = {'SR': ('ZuCo1', 'task1'), 'NR': ('ZuCo1', 'task2'), 'NR2': ('ZuCo2', 'task2')}   # GLIM label table: ZuCo2 NR = task2, TSR = task3
row_of = {}
for col in ('raw text', 'input text'):   # raw text first (matches 1,036 / 1,049 of our sentences), revised input text as fallback
    for _, r in LTdf.iterrows():
        row_of.setdefault((r['dataset'], r['task'], norm(r[col])), r)
TASK_OF = {}
for t_, (s0, e0) in {'SR': (0, 400), 'NR': (400, 700), 'NR2': (700, 1049)}.items():
    for g_ in range(s0, e0): TASK_OF[g_] = t_
tr, te, unmatched = [], [], 0
for r in recs:
    lr = row_of.get(DS[TASK_OF[r['g']]] + (norm(texts[r['g']]),))
    if lr is None: unmatched += 1; continue
    ph = phase_of_uid.get(int(lr['text uid']))
    if ph == 'train':
        for v in VARS: tr.append(dict(r, uid=int(lr['text uid']), tgt=str(lr[v])))
    elif ph == 'test':
        te.append(dict(r, uid=int(lr['text uid']), tgt=str(lr['input text']), variants=[str(lr[v]) for v in VARS]))
tr = tr[:a.max_train]; te = te[:a.max_eval]
print(f'GLIM split on our records: train x8 {len(tr)}, test {len(te)}, unmatched records {unmatched}', flush=True)
Xtr = np.concatenate([r['words'] for r in tr]); MU = Xtr.mean(0); SD = Xtr.std(0) + 1e-6; del Xtr
L_IN = 56
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
tok = AutoTokenizer.from_pretrained(a.t5)
t5 = T5ForConditionalGeneration.from_pretrained(a.t5, torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)


def collate(batch):
    X = np.zeros((len(batch), L_IN, 840), np.float32); Mk = np.zeros((len(batch), L_IN), np.int64)
    for i, r in enumerate(batch):
        w = ((r['words'] - MU) / SD)[:L_IN]; X[i, :len(w)] = w; Mk[i, :len(w)] = 1
    X = torch.from_numpy(X); Mk = torch.from_numpy(Mk)
    if a.input == 'noise':
        X = torch.randn_like(X)
    return X, Mk, ['To English: ' + texts[r['g']] for r in batch], [r['tgt'] for r in batch], [r['g'] for r in batch], [r['subject'] for r in batch], batch


def sincos(d, n):
    om = 1.0 / 10000 ** (np.arange(d // 2) / (d / 2)); o = np.einsum('m,d->md', np.arange(n), om)
    return torch.from_numpy(np.concatenate([np.sin(o), np.cos(o)], 1)).float()


class Mlp(nn.Module):
    def __init__(self, d, r=4):
        super().__init__(); self.fc1 = nn.Linear(d, d * r); self.fc2 = nn.Linear(d * r, d)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate='tanh'))


class EncBlock(nn.Module):  # adaLN block with prompt injection and temporal modulation (GLIM EncoderBlock)
    def __init__(self, d, L, heads=8):
        super().__init__()
        self.n1 = nn.LayerNorm(d, eps=1e-6, elementwise_affine=False); self.n2 = nn.LayerNorm(d, eps=1e-6, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True); self.mlp = Mlp(d)
        self.ada = nn.Linear(d, 6 * d); self.tada = nn.Linear(d, 2 * L)
        for m in (self.ada, self.tada):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x, mask, p):
        sc, sh = self.tada(F.silu(p)).chunk(2, 1); x = x * (1 + sc[:, :, None]) + sh[:, :, None]
        a1, b1, g1, a2, b2, g2 = self.ada(F.silu(p)).chunk(6, 1)
        h = self.n1(x) * (1 + b1[:, None]) + a1[:, None]
        x = x + g1[:, None] * self.attn(h, h, h, key_padding_mask=(mask == 0), need_weights=False)[0]
        return x + g2[:, None] * self.mlp(self.n2(x) * (1 + b2[:, None]) + a2[:, None])


class DecBlock(nn.Module):  # causal self-attention over queries + cross-attention to EEG tokens (GLIM DecoderBlock)
    def __init__(self, d, heads=8):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d, eps=1e-6), nn.LayerNorm(d, eps=1e-6), nn.LayerNorm(d, eps=1e-6)
        self.sa = nn.MultiheadAttention(d, heads, batch_first=True); self.ca = nn.MultiheadAttention(d, heads, batch_first=True); self.mlp = Mlp(d)
    def forward(self, q, x, mask):
        L = q.shape[1]; cm = torch.triu(torch.full((L, L), float('-inf'), device=q.device), 1)
        h = self.n1(q); q = q + self.sa(h, h, h, attn_mask=cm, need_weights=False)[0]
        q = q + self.ca(self.n2(q), x, x, key_padding_mask=(mask == 0), need_weights=False)[0]
        return q + self.mlp(self.n3(q))


class GLIMLite(nn.Module):
    def __init__(self, din=128, dh=256, de=1024, nq=96):
        super().__init__()
        self.inp = nn.Linear(840, din); self.prompt = nn.Parameter(torch.randn(din) * 0.02)
        self.register_buffer('pos', sincos(din, L_IN)[None])
        self.inb = nn.ModuleList([EncBlock(din, L_IN) for _ in range(6)])
        self.xp = nn.Linear(din, dh); self.norm1 = nn.LayerNorm(dh, eps=1e-6)
        self.q = nn.Parameter(torch.randn(1, nq, dh)); self.outb = nn.ModuleList([DecBlock(dh) for _ in range(6)]); self.norm2 = nn.LayerNorm(dh, eps=1e-6)
        # aligner
        self.al_in = nn.Linear(dh, de); self.qx = nn.Parameter(torch.randn(1, 1, de)); self.qy = nn.Parameter(torch.randn(1, 1, de))
        self.cax = nn.MultiheadAttention(de, 8, batch_first=True); self.cay = nn.MultiheadAttention(de, 8, batch_first=True)
    def encoder_params(self):
        return [p for n, p in self.named_parameters() if not (n.startswith('al_') or n.startswith('qx') or n.startswith('qy') or n.startswith('ca'))]
    def eeg(self, X, Mk):
        x = self.inp(X) + self.pos; p = self.prompt.expand(X.shape[0], -1)
        for b in self.inb:
            x = b(x, Mk, p)
        x = self.norm1(self.xp(x)); q = self.q.expand(X.shape[0], -1, -1)
        for b in self.outb:
            q = b(q, x, Mk)
        e = self.al_in(self.norm2(q))                                        # (n, 96, 1024) -> T5 decoder
        v = self.cax(self.qx.expand(len(e), -1, -1), e, e, need_weights=False)[0][:, 0]
        return e, v
    def text_vec(self, y, ym):
        return self.cay(self.qy.expand(len(y), -1, -1), y, y, key_padding_mask=(ym == 0), need_weights=False)[0][:, 0]


model = GLIMLite().to(dev)
if a.init == 'random':
    for p in model.encoder_params():
        p.requires_grad_(False)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.Adam(params, lr=a.lr, weight_decay=0)
log(f'cfg {vars(a)}'); log(f'train {len(tr)} test {len(te)} records; trainable {sum(p.numel() for p in params)/1e6:.1f}M ({time.time()-t0:.0f}s)')


def tok_ids(ts, n):
    e = tok(ts, max_length=n, padding='max_length', truncation=True, return_tensors='pt'); return e['input_ids'].to(dev), e['attention_mask'].to(dev)


def step(X, Mk, T, TG, train=True):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        ids, im = tok_ids(T, 96)
        with torch.no_grad():
            y = t5.get_encoder()(input_ids=ids, attention_mask=im).last_hidden_state
        e, v = model.eeg(X, Mk); w = model.text_vec(y, im)
        xn, yn = F.normalize(v.float(), dim=1), F.normalize(w.float(), dim=1); lg = xn @ yn.T; lab = torch.arange(len(T), device=dev)
        l_clip = (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab)) / 2
        tid, _ = tok_ids(TG, 64); labels = tid.masked_fill(tid == tok.pad_token_id, -100)
        out = t5(encoder_outputs=BaseModelOutput(e.to(torch.bfloat16)), labels=labels)
    return 0.5 * l_clip + 0.5 * out.loss.float(), l_clip, out


loader = lambda rs, sh, bs: torch.utils.data.DataLoader(rs, batch_size=bs, shuffle=sh, collate_fn=collate, num_workers=0, drop_last=sh)
def uid_batches(rs, bs, gen):
    """GLIMSampler: shuffled order, batches of DISTINCT text uids (greedy; leftovers carried to the next pass)."""
    order = torch.randperm(len(rs), generator=gen).tolist(); out = []; pool = order
    while len(pool) >= bs:
        cur, used, rest = [], set(), []
        for i_ in pool:
            if rs[i_]['uid'] not in used and len(cur) < bs:
                cur.append(i_); used.add(rs[i_]['uid'])
                if len(cur) == bs:
                    out.append(cur); cur, used = [], set()
            else:
                rest.append(i_)
        if not out or len(rest) == len(pool):
            break
        pool = rest + cur
    return out
gen_ = torch.Generator().manual_seed(a.seed); stepn = 0; tl = tc = 0; nb = 0
while stepn < a.max_steps:
    for bidx in uid_batches(tr, a.bs, gen_):
        X, Mk, T, TG, *_ = collate([tr[i_] for i_ in bidx])
        model.train(); loss, lc, _ = step(X.to(dev), Mk.to(dev), T, TG)
        opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item(); tc += lc.item(); nb += 1; stepn += 1
        if stepn % 1000 == 0:
            log(f'step {stepn}/{a.max_steps} loss {tl/nb:.4f} clip {tc/nb:.4f} ({time.time()-t0:.0f}s)'); tl = tc = 0; nb = 0
        if stepn >= a.max_steps:
            break
model.eval()
try:
    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, f'{a.out_dir}/{a.tag}_trainable.pt')  # coordinator: keep trainable weights (frozen T5 not duplicated)
except Exception as e:
    print(f'checkpoint save FAILED ({e}); run continues', flush=True)


def toks(s):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", s.lower())
def bleu1(h, r):
    m = hl = rl = 0
    for x, y in zip(h, r):
        x, y = toks(x), toks(y); hl += len(x); rl += len(y); cy = Counter(y); m += sum(min(c, cy[w]) for w, c in Counter(x).items())
    return 0.0 if hl == 0 else float((1.0 if hl > rl else math.exp(1 - rl / max(1, hl))) * m / hl)
def rouge1(h, r):
    P, R, Fm = [], [], []
    for x, y in zip(h, r):
        x, y = toks(x), toks(y); ov = sum((Counter(x) & Counter(y)).values()); p = ov / len(x) if x else 0.0; q = ov / len(y) if y else 0.0
        P.append(p); R.append(q); Fm.append(2 * p * q / (p + q) if p + q else 0.0)
    return {'p': float(np.mean(P)), 'r': float(np.mean(R)), 'f': float(np.mean(Fm))}


rows = []
with torch.no_grad():
    for X, Mk, T, TG, G, S, B in loader(te, False, 48):
        X, Mk = X.to(dev), Mk.to(dev)
        _, _, out = step(X, Mk, T, TG, train=False)
        tf = tok.batch_decode(out.logits.argmax(-1), skip_special_tokens=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            e, _ = model.eeg(X, Mk); enc = BaseModelOutput(e.to(torch.bfloat16))
            bm = tok.batch_decode(t5.generate(encoder_outputs=enc, num_beams=2, min_length=0, max_length=64), skip_special_tokens=True)
            gr = tok.batch_decode(t5.generate(encoder_outputs=BaseModelOutput(e.to(torch.bfloat16)), num_beams=1, do_sample=False, max_length=64), skip_special_tokens=True)
            en, _ = model.eeg(torch.randn_like(X), Mk)   # LOCK §4 noise-input floor
            nbm = tok.batch_decode(t5.generate(encoder_outputs=BaseModelOutput(en.to(torch.bfloat16)), num_beams=2, min_length=0, max_length=64), skip_special_tokens=True)
            ngr = tok.batch_decode(t5.generate(encoder_outputs=BaseModelOutput(en.to(torch.bfloat16)), num_beams=1, do_sample=False, max_length=64), skip_special_tokens=True)
        for i in range(len(T)):
            rows.append({'g': G[i], 'subject': S[i], 'target': TG[i], 'variants': B[i]['variants'], 'teacher_forced': tf[i].strip(), 'greedy': gr[i].strip(), 'beam': bm[i].strip(),
                         'noise_input_greedy': ngr[i].strip(), 'noise_input_beam': nbm[i].strip()})
with open(os.path.join(a.out_dir, f'{a.tag}_generations.jsonl'), 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')
refs = [r['target'] for r in rows]
from torchmetrics.functional.text import bleu_score as _bleu
MTV = {m: float(np.mean([float(_bleu([r[m]], [r['variants']], n_gram=1)) for r in rows])) for m in ('beam', 'greedy', 'teacher_forced', 'noise_input_beam')}
RAW = {m: float(np.mean([float(_bleu([r[m]], [[r['target']]], n_gram=1)) for r in rows])) for m in ('beam', 'greedy')}
REL = list(_csv.DictReader(open('/path/to/eegaudit/data_raw/third_party/glim_results/wandb_export_gen_samples_glim.csv', encoding='utf-8')))
G2T = float(np.mean([float(r['Bleu1']) for r in REL]))
res = {'bleu1_at_MTV_per_sample_mean': MTV, 'bleu1_vs_raw_per_sample_mean': RAW, 'G2_target_released_glim_bleu1_mtv': G2T, 'G2_rel_diff_beam': (MTV['beam'] - G2T) / G2T, 'G2_pass_10pct': abs(MTV['beam'] - G2T) / G2T <= 0.10, 'max_steps': a.max_steps, 'tag': a.tag, 'input': a.input, 'init': a.init, 'cfg': vars(a), 'n_train_records': len(tr), 'n_test_records': len(te), 'best_epoch': a.epochs,
       'modes': {m: {'bleu1': bleu1([r[m] for r in rows], refs), 'rouge1': rouge1([r[m] for r in rows], refs), 'n_unique_outputs': len({r[m] for r in rows})}
                 for m in ('teacher_forced', 'greedy', 'beam')}, 'runtime_s': time.time() - t0}
res['code_sha256'] = CODE_SHA256; json.dump(res, open(os.path.join(a.out_dir, f'{a.tag}_results.json'), 'w'), indent=1)
try:  # Track A: save the trained weights (T5 is frozen and not saved) in fp16
    torch.save({k_: v_.detach().half().cpu() for k_, v_ in model.state_dict().items()}, os.path.join(a.out_dir, f'{a.tag}_weights_fp16.pt'))
except Exception as e_:
    print('weight save failed', repr(e_), flush=True)
log(json.dumps({m: (round(v['bleu1'], 4), round(v['rouge1']['f'], 4), v['n_unique_outputs']) for m, v in res['modes'].items()}), f'done {time.time()-t0:.0f}s')
