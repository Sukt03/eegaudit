"""GLIM-style EEG-to-text — OUR RE-IMPLEMENTATION of Liu et al. 2025 (arXiv 2505.17099; architecture after github.com/justin-xzliu/GLIM
model/modules.py, CC BY-NC-SA 4.0, research use) on OUR leak-free ZuCo split and word-level band features (the released checkpoint needs
~97 GB of raw sentence-level .mat files, not downloaded). IMPLEMENTER, 21 Nov 2026.
Kept from GLIM (train.py defaults): input_dim 128, hidden_dim 256, 96 learnable queries, 6 adaLN encoder blocks with prompt injection and
temporal modulation, 6 causal-self-attention decoder blocks with cross-attention to the EEG tokens, aligner (Linear 256 -> 1024, one learned
query + cross-attention per modality), loss 0.5 x CLIP (cosine, symmetric CE, no temperature) + 0.5 x LM through the FROZEN flan-t5-large
(encoder_outputs = aligned EEG tokens), commitment weight 0, dropout 0, Adam lr 1e-4 wd 0, bf16 autocast, 200 epochs, last checkpoint,
generation with 2 beams, max 64 tokens. Prompt dropout (0, 1, 1) as in train.py, so only the task prompt (<NR> for all of SR/NR/NR2) is used.
Changes: input = word-level band features (n, <= 56 words, 840) projected to 128 per word (GLIM: 1280 raw samples x 128 channels);
positions = word index; target text = the input sentence (GLIM: one of 8 LLM paraphrase variants); batch 72 on one GPU (GLIM: 72 x 8 GPUs).
Arms: eeg | noise (fresh N(0,1) features of identical shape) | random_init (EEG encoder frozen at its seeded init; aligner trained).
Writes <tag>_generations.jsonl and <tag>_results.json in the train_bt.py format, so text_yield_card.py scores it unchanged."""
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
ap.add_argument('--seed', type=int, default=42); ap.add_argument('--max_train', type=int, default=None); ap.add_argument('--max_eval', type=int, default=None)
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device('cuda'); t0 = time.time()
os.makedirs(a.out_dir, exist_ok=True); logf = open(os.path.join(a.out_dir, f'{a.tag}_train_log.txt'), 'w')
def log(*s):
    m = ' '.join(str(x) for x in s); print(m, flush=True); logf.write(m + '\n'); logf.flush()

D = pickle.load(open(a.cache, 'rb')); texts, recs = D['texts'], D['recs']; del D
tr = [r for r in recs if r['split'] == 'train'][:a.max_train]; te = [r for r in recs if r['split'] == 'test'][:a.max_eval]
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
    return X, Mk, [texts[r['g']] for r in batch], [r['g'] for r in batch], [r['subject'] for r in batch]


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


def step(X, Mk, T, train=True):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        ids, im = tok_ids(T, 96)
        with torch.no_grad():
            y = t5.get_encoder()(input_ids=ids, attention_mask=im).last_hidden_state
        e, v = model.eeg(X, Mk); w = model.text_vec(y, im)
        xn, yn = F.normalize(v.float(), dim=1), F.normalize(w.float(), dim=1); lg = xn @ yn.T; lab = torch.arange(len(T), device=dev)
        l_clip = (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab)) / 2
        tid, _ = tok_ids(T, 64); labels = tid.masked_fill(tid == tok.pad_token_id, -100)
        out = t5(encoder_outputs=BaseModelOutput(e.to(torch.bfloat16)), labels=labels)
    return 0.5 * l_clip + 0.5 * out.loss.float(), l_clip, out


loader = lambda rs, sh, bs: torch.utils.data.DataLoader(rs, batch_size=bs, shuffle=sh, collate_fn=collate, num_workers=0, drop_last=sh)
for ep in range(a.epochs):
    model.train(); tl = tc = 0; nb = 0
    for X, Mk, T, _, _ in loader(tr, True, a.bs):
        loss, lc, _ = step(X.to(dev), Mk.to(dev), T)
        opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item(); tc += lc.item(); nb += 1
    if ep % 10 == 0 or ep == a.epochs - 1:
        log(f'epoch {ep+1}/{a.epochs} loss {tl/nb:.4f} clip {tc/nb:.4f} ({time.time()-t0:.0f}s)')
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
    for X, Mk, T, G, S in loader(te, False, 48):
        X, Mk = X.to(dev), Mk.to(dev)
        _, _, out = step(X, Mk, T, train=False)
        tf = tok.batch_decode(out.logits.argmax(-1), skip_special_tokens=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            e, _ = model.eeg(X, Mk); enc = BaseModelOutput(e.to(torch.bfloat16))
            bm = tok.batch_decode(t5.generate(encoder_outputs=enc, num_beams=2, min_length=0, max_length=64), skip_special_tokens=True)
            gr = tok.batch_decode(t5.generate(encoder_outputs=BaseModelOutput(e.to(torch.bfloat16)), num_beams=1, do_sample=False, max_length=64), skip_special_tokens=True)
            en, _ = model.eeg(torch.randn_like(X), Mk)   # LOCK §4 noise-input floor
            nbm = tok.batch_decode(t5.generate(encoder_outputs=BaseModelOutput(en.to(torch.bfloat16)), num_beams=2, min_length=0, max_length=64), skip_special_tokens=True)
            ngr = tok.batch_decode(t5.generate(encoder_outputs=BaseModelOutput(en.to(torch.bfloat16)), num_beams=1, do_sample=False, max_length=64), skip_special_tokens=True)
        for i in range(len(T)):
            rows.append({'g': G[i], 'subject': S[i], 'target': T[i], 'teacher_forced': tf[i].strip(), 'greedy': gr[i].strip(), 'beam': bm[i].strip(),
                         'noise_input_greedy': ngr[i].strip(), 'noise_input_beam': nbm[i].strip()})
with open(os.path.join(a.out_dir, f'{a.tag}_generations.jsonl'), 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')
refs = [r['target'] for r in rows]
res = {'tag': a.tag, 'input': a.input, 'init': a.init, 'cfg': vars(a), 'n_train_records': len(tr), 'n_test_records': len(te), 'best_epoch': a.epochs,
       'modes': {m: {'bleu1': bleu1([r[m] for r in rows], refs), 'rouge1': rouge1([r[m] for r in rows], refs), 'n_unique_outputs': len({r[m] for r in rows})}
                 for m in ('teacher_forced', 'greedy', 'beam')}, 'runtime_s': time.time() - t0}
res['code_sha256'] = CODE_SHA256; json.dump(res, open(os.path.join(a.out_dir, f'{a.tag}_results.json'), 'w'), indent=1)
try:  # Track A: save the trained weights (T5 is frozen and not saved) in fp16
    torch.save({k_: v_.detach().half().cpu() for k_, v_ in model.state_dict().items()}, os.path.join(a.out_dir, f'{a.tag}_weights_fp16.pt'))
except Exception as e_:
    print('weight save failed', repr(e_), flush=True)
log(json.dumps({m: (round(v['bleu1'], 4), round(v['rouge1']['f'], 4), v['n_unique_outputs']) for m, v in res['modes'].items()}), f'done {time.time()-t0:.0f}s')
