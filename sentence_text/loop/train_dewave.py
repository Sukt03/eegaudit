"""DeWave-style (Duan et al. 2023) discrete-codex EEG-to-text on ZuCo word-level GD features, built on the BrainTranslator script
(experiments/track_text/brain_translator/train_bt.py) so that data, decoder, and the Jo et al. 2024 dissection are identical.
Pipeline: word features (840) -> 1-layer TransformerEncoder (as BrainTranslator) -> Linear(840, d_code) -> vector quantisation against a
learned codebook (K x d_code, straight-through, codebook + 0.25 commitment loss, van den Oord 2017) -> Linear(d_code, d_bart) -> BART-base
encoder via inputs_embeds -> BART decoder.
Simplifications relative to DeWave: (1) the conformer raw-wave encoder is replaced by the same 1-layer transformer over word-level GD
features (DeWave's word-level variant also uses eye-tracking-aligned word features), (2) no contrastive codex-to-text alignment loss,
(3) BART-base instead of BART-large, (4) plain VQ-VAE codebook loss instead of DeWave's wav2vec-style product quantisation.
Inputs: eeg | noise (N(0,1) twin of identical shape, Jo et al. 2024) | shuffled (training EEG permuted against sentences; dev/test intact).
Run with python -I inside a GPU node."""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()  # every result JSON carries the exact code hash
import argparse, json, os, sys, time, pickle, copy, re, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '/path/to/eegaudit/experiments/track_text')
from pilot_retrieval import embed_sentences, retrieval_scores, wolpaw_bits

ap = argparse.ArgumentParser()
ap.add_argument('--cache', required=True); ap.add_argument('--out_dir', required=True); ap.add_argument('--tag', required=True)
ap.add_argument('--input', choices=['eeg', 'noise', 'shuffled'], default='eeg')
ap.add_argument('--bart', default='/path/to/eegaudit/data_raw/bart_base')
ap.add_argument('--minilm', default='/path/to/eegaudit/data_raw/zuco/minilm')
ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--lr', type=float, default=5e-5); ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--max_words', type=int, default=56); ap.add_argument('--max_tokens', type=int, default=64)
ap.add_argument('--beams', type=int, default=4); ap.add_argument('--max_eval', type=int, default=None)
ap.add_argument('--bart_init', choices=['pretrained', 'random'], default='pretrained')
ap.add_argument('--freeze_encoder', action='store_true', help='random-init twin (METHOD G1 a, PRIMARY): EEG-side encoder frozen at its seeded random init; BART + projection trained identically')  # random = random-init twin: BART re-initialised from its config (no language pretraining), same recipe
ap.add_argument('--K', type=int, default=2048); ap.add_argument('--d_code', type=int, default=512); ap.add_argument('--beta', type=float, default=0.25)
ap.add_argument('--vq_lr', type=float, default=1e-3)
ap.add_argument('--keep', type=int, default=840, help='D5 width: keep this many of the 840 input dims (fixed random subset, seed 1234), zero the rest; 840 = unchanged')
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(a.out_dir, exist_ok=True)
t0 = time.time()
log_f = open(os.path.join(a.out_dir, f'{a.tag}_train_log.txt'), 'w')
def log(*s):
    msg = ' '.join(str(x) for x in s); print(msg, flush=True); log_f.write(msg + '\n'); log_f.flush()

# ---------------- data (identical to train_bt.py)
D = pickle.load(open(a.cache, 'rb')); texts, recs = D['texts'], D['recs']
tr = [r for r in recs if r['split'] == 'train']; dv = [r for r in recs if r['split'] == 'dev']; te = [r for r in recs if r['split'] == 'test']
if a.max_eval:
    te = te[:a.max_eval]
if a.input == 'shuffled':  # trial-shuffled training: record i keeps its EEG, gets the sentence of record perm[i]; dev/test untouched
    perm = np.random.default_rng(a.seed + 1).permutation(len(tr))
    tr = [dict(r, g=tr[j]['g']) for r, j in zip(tr, perm)]
Xtr = np.concatenate([r['words'] for r in tr]); mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6; del Xtr
from transformers import BartTokenizer, BartForConditionalGeneration
tok = BartTokenizer.from_pretrained(a.bart)
KEEPMASK = torch.zeros(840); KEEPMASK[torch.from_numpy(np.random.default_rng(1234).permutation(840)[:a.keep])] = 1
def collate(batch):
    n = min(a.max_words, max(r['words'].shape[0] for r in batch))
    X = np.zeros((len(batch), n, 840), np.float32); M = np.zeros((len(batch), n), np.int64)
    for i, r in enumerate(batch):
        w = ((r['words'] - mu) / sd)[:n]; X[i, :len(w)] = w; M[i, :len(w)] = 1
    X = torch.from_numpy(X); M = torch.from_numpy(M)
    if a.keep < 840: X = X * KEEPMASK
    if a.input == 'noise':
        X = torch.randn_like(X)
    enc = tok([texts[r['g']] for r in batch], padding=True, truncation=True, max_length=a.max_tokens, return_tensors='pt')
    labels = enc['input_ids'].clone(); labels[enc['attention_mask'] == 0] = -100
    return X, M, labels, [texts[r['g']] for r in batch], [r['g'] for r in batch], [r['subject'] for r in batch]
def loader(rs, shuffle, bs):
    return torch.utils.data.DataLoader(rs, batch_size=bs, shuffle=shuffle, collate_fn=collate, num_workers=0)

# ---------------- model
class VQ(nn.Module):
    def __init__(self, K, d, beta):
        super().__init__(); self.emb = nn.Embedding(K, d); nn.init.uniform_(self.emb.weight, -1 / K, 1 / K); self.beta = beta
    def forward(self, z, M):
        # z: (B, T, d). nearest codebook entry per word position; straight-through estimator; losses on valid positions only
        flat = z.reshape(-1, z.shape[-1])
        d2 = (flat ** 2).sum(1, keepdim=True) - 2 * flat @ self.emb.weight.t() + (self.emb.weight ** 2).sum(1)[None]
        idx = d2.argmin(1); q = self.emb(idx).view_as(z)
        m = M.float().unsqueeze(-1)
        codebook = (((q - z.detach()) ** 2) * m).sum() / m.sum().clamp(min=1) / z.shape[-1]
        commit = (((z - q.detach()) ** 2) * m).sum() / m.sum().clamp(min=1) / z.shape[-1]
        q_st = z + (q - z).detach()
        return q_st, codebook + self.beta * commit, idx.view(z.shape[:2])
class DeWaveLite(nn.Module):
    def __init__(self, bart, K, d_code, beta):
        super().__init__()
        self.pre = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=840, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True), num_layers=1)
        self.to_code = nn.Linear(840, d_code); self.vq = VQ(K, d_code, beta); self.fc = nn.Linear(d_code, bart.config.d_model); self.bart = bart
    def embed(self, X, M):
        h = self.pre(X, src_key_padding_mask=(M == 0)); z = self.to_code(h); q, vql, idx = self.vq(z, M)
        return torch.relu(self.fc(q)), vql, idx
    def forward(self, X, M, labels):
        e, vql, idx = self.embed(X, M); out = self.bart(inputs_embeds=e, attention_mask=M, labels=labels); return out, vql, idx
    @torch.no_grad()
    def generate(self, X, M, **kw):
        e, _, _ = self.embed(X, M); return self.bart.generate(inputs_embeds=e, attention_mask=M, **kw)
if a.bart_init == 'random':
    from transformers import BartConfig
    bart = BartForConditionalGeneration(BartConfig.from_pretrained(a.bart))  # weights drawn from the config's init (seeded above)
else:
    bart = BartForConditionalGeneration.from_pretrained(a.bart)
model = DeWaveLite(bart, a.K, a.d_code, a.beta).to(dev)
if a.freeze_encoder:
    for n_, p_ in model.named_parameters():
        if any(n_.startswith(x) for x in ['pre.', 'to_code.', 'vq.']):
            p_.requires_grad_(False)
    print('frozen encoder params', sum(p_.numel() for p_ in model.parameters() if not p_.requires_grad), flush=True)
n_par = sum(p.numel() for p in model.parameters())
log(f'cfg {vars(a)}'); log(f'train {len(tr)} dev {len(dv)} test {len(te)} records; test sentences {len({r["g"] for r in te})}; params {n_par/1e6:.1f}M; device {dev}')
vq_params = list(model.vq.parameters()); other = [p for n, p in model.named_parameters() if not n.startswith('vq.')]
opt = torch.optim.AdamW([{'params': other, 'lr': a.lr}, {'params': vq_params, 'lr': a.vq_lr}], weight_decay=0.01)
steps = a.epochs * math.ceil(len(tr) / a.bs)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, 0.05 * steps)) * max(0.0, (steps - s) / max(1, steps * 0.95)))

def dev_loss():
    model.eval(); tot = 0; n = 0; used = set()
    with torch.no_grad():
        for X, M, L, *_ in loader(dv, False, 64):
            out, _, idx = model(X.to(dev), M.to(dev), L.to(dev)); k = (L != -100).sum().item(); tot += out.loss.item() * k; n += k
            used |= set(idx[M.to(dev) == 1].tolist())
    model.train(); return tot / max(1, n), len(used)

# ---------------- train
best = (1e9, None, -1, 0); step = 0
for ep in range(a.epochs):
    model.train(); run = 0; runv = 0; nb = 0; used = set()
    for X, M, L, *_ in loader(tr, True, a.bs):
        Mg = M.to(dev); out, vql, idx = model(X.to(dev), Mg, L.to(dev)); loss = out.loss + vql
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        run += out.loss.item(); runv += vql.item(); nb += 1; step += 1; used |= set(idx[Mg == 1].unique().tolist())
    dl, dused = dev_loss()
    log(f'epoch {ep+1}/{a.epochs} train_loss {run/nb:.4f} vq_loss {runv/nb:.4f} codes_used_train {len(used)}/{a.K} dev_loss {dl:.4f} dev_ppl {math.exp(dl):.1f} codes_used_dev {dused} ({time.time()-t0:.0f}s)')
    if dl < best[0]:
        best = (dl, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}), ep + 1, dused)
model.load_state_dict(best[1]); model.eval(); log(f'best epoch {best[2]} dev_loss {best[0]:.4f}')
try:
    torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in best[1].items()}, f'{a.out_dir}/{a.tag}_best.pt'); log(f'checkpoint saved (fp16): {a.out_dir}/{a.tag}_best.pt')
except Exception as e:
    log(f'checkpoint save FAILED ({e}); run continues')  # coordinator: a full disk must not lose the run's results

# ---------------- metrics (identical to train_bt.py)
def toks(s):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", s.lower())
from collections import Counter
def bleu1(hyps, refs):
    match = 0; hl = 0; rl = 0
    for h, r in zip(hyps, refs):
        h = toks(h); r = toks(r); hl += len(h); rl += len(r)
        ch, cr = Counter(h), Counter(r); match += sum(min(c, cr[w]) for w, c in ch.items())
    if hl == 0: return 0.0
    p = match / hl; bp = 1.0 if hl > rl else math.exp(1 - rl / max(1, hl)); return float(bp * p)
def rouge1(hyps, refs):
    P = []; R = []; Fs = []
    for h, r in zip(hyps, refs):
        h = toks(h); r = toks(r); ov = sum((Counter(h) & Counter(r)).values())
        p = ov / len(h) if h else 0.0; rc = ov / len(r) if r else 0.0; f = 2 * p * rc / (p + rc) if p + rc else 0.0
        P.append(p); R.append(rc); Fs.append(f)
    return {'p': float(np.mean(P)), 'r': float(np.mean(R)), 'f': float(np.mean(Fs))}

rows = []; test_codes = []
with torch.no_grad():
    for X, M, L, T, G, S in loader(te, False, 32):
        X, M, L = X.to(dev), M.to(dev), L.to(dev)
        out, _, idx = model(X, M, L); test_codes += idx[M == 1].tolist()
        pred = out.logits.argmax(-1)
        tf = tok.batch_decode(torch.where(L == -100, tok.pad_token_id, pred), skip_special_tokens=True)
        gr = tok.batch_decode(model.generate(X, M, num_beams=1, do_sample=False, max_length=a.max_tokens), skip_special_tokens=True)
        bm = tok.batch_decode(model.generate(X, M, num_beams=a.beams, do_sample=False, max_length=a.max_tokens), skip_special_tokens=True)
        Xn = torch.randn_like(X)  # LOCK §4 noise-input floor: the trained model fed N(0,1) at test time (same mask)
        ngr = tok.batch_decode(model.generate(Xn, M, num_beams=1, do_sample=False, max_length=a.max_tokens), skip_special_tokens=True)
        nbm = tok.batch_decode(model.generate(Xn, M, num_beams=a.beams, do_sample=False, max_length=a.max_tokens), skip_special_tokens=True)
        for i in range(len(T)):
            rows.append({'g': G[i], 'subject': S[i], 'target': T[i], 'teacher_forced': tf[i].strip(), 'greedy': gr[i].strip(), 'beam': bm[i].strip(),
                         'noise_input_greedy': ngr[i].strip(), 'noise_input_beam': nbm[i].strip()})
log(f'generated for {len(rows)} test records ({time.time()-t0:.0f}s)')
with open(os.path.join(a.out_dir, f'{a.tag}_generations.jsonl'), 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
refs = [r['target'] for r in rows]
cnt = Counter(test_codes); p = np.array(list(cnt.values()), float); p /= p.sum()
res = {'tag': a.tag, 'input': a.input, 'cfg': vars(a), 'n_train_records': len(tr), 'n_dev_records': len(dv), 'n_test_records': len(te),
       'best_epoch': best[2], 'best_dev_loss': best[0], 'best_dev_ppl': math.exp(best[0]), 'params_M': n_par / 1e6,
       'codex': {'K': a.K, 'codes_used_test': len(cnt), 'code_perplexity_test': float(np.exp(-(p * np.log(p)).sum()))}, 'modes': {}}
for mode in ['teacher_forced', 'greedy', 'beam']:
    hyps = [r[mode] for r in rows]
    res['modes'][mode] = {'bleu1': bleu1(hyps, refs), 'rouge1': rouge1(hyps, refs), 'n_unique_outputs': len(set(hyps)),
                          'mean_hyp_len': float(np.mean([len(toks(h)) for h in hyps])), 'mean_ref_len': float(np.mean([len(toks(r)) for r in refs]))}
test_g = sorted({r['g'] for r in rows}); pos = {g: j for j, g in enumerate(test_g)}; N = len(test_g)
E_c, enc_name = embed_sentences([texts[g] for g in test_g], a.minilm)
true_idx = np.array([pos[r['g']] for r in rows]); rng = np.random.default_rng(a.seed)
res['identification'] = {'n_candidates': N, 'n_queries': len(rows), 'encoder': enc_name, 'chance_top1': 1 / N}
for mode in ['teacher_forced', 'greedy', 'beam']:
    E_q, _ = embed_sentences([r[mode] if r[mode] else '.' for r in rows], a.minilm)
    t1, tw = retrieval_scores(E_q, E_c, true_idx)
    sh = [retrieval_scores(E_q, E_c, rng.permutation(true_idx)) for _ in range(20)]
    res['identification'][mode] = {'top1': t1, 'two_way': tw, 'bits': wolpaw_bits(t1, N),
                                   'shuffled_pairs_top1': float(np.mean([s[0] for s in sh])), 'shuffled_pairs_two_way': float(np.mean([s[1] for s in sh])),
                                   'shuffled_pairs_bits': float(np.mean([wolpaw_bits(s[0], N) for s in sh]))}
res['runtime_s'] = time.time() - t0
res['code_sha256'] = CODE_SHA256; json.dump(res, open(os.path.join(a.out_dir, f'{a.tag}_results.json'), 'w'), indent=1)
try:  # Track A: save the trained weights in fp16 (try/except)
    torch.save({k_: v_.detach().half().cpu() for k_, v_ in model.state_dict().items() if not k_.startswith('bart.') or not a.__dict__.get('freeze_lm', False)},
               os.path.join(a.out_dir, f'{a.tag}_weights_fp16.pt'))
except Exception as e_:
    print('weight save failed', repr(e_), flush=True)

log(json.dumps(res['codex']))
log(json.dumps({m: {'bleu1': round(v['bleu1'], 4), 'rouge1_f': round(v['rouge1']['f'], 4), 'uniq': v['n_unique_outputs']} for m, v in res['modes'].items()}))
log(json.dumps({m: {k: round(v, 4) for k, v in res['identification'][m].items()} for m in ['teacher_forced', 'greedy', 'beam']}))
log(f'done {time.time()-t0:.0f}s')
