"""S11: encoder-side scoring of the autoregressive EEG-to-text models (BrainTranslator and the DeWave-style codex variant), i.e. the
information in the ENCODER output rather than in generated text. Trains the model exactly as train_bt.py / train_dewave.py for --epochs
epochs (all full runs selected epoch 1 by dev loss), then takes the BART encoder's last hidden state, masked-mean over word positions, for
train / dev / test records; ridge (alpha by dev 2-way among a grid) from that vector to the MiniLM sentence embedding; identification
within task (candidates = test sentences of the task) and pooled; top-1, 2-way, Wolpaw bits, shuffled-pair floor; per-query 2-way saved
for the EEG-minus-noise sentence-cluster bootstrap (done by s11_compare.py). Inputs eeg | noise (N(0,1) twin) | shuffled."""
import argparse, json, os, sys, time, pickle, math
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, '/path/to/eegaudit/experiments/track_text')
from pilot_retrieval import retrieval_scores, wolpaw_bits
ap = argparse.ArgumentParser()
ap.add_argument('--arch', choices=['bt', 'dewave'], required=True); ap.add_argument('--input', choices=['eeg', 'noise', 'shuffled'], default='eeg')
ap.add_argument('--cache', default='/path/to/eegaudit/results/track_text/brain_translator/cache_full.pkl')
ap.add_argument('--emb', default='/path/to/eegaudit/results/loop/r4_contrastive/minilm_full.npz')
ap.add_argument('--out_dir', required=True); ap.add_argument('--tag', required=True); ap.add_argument('--epochs', type=int, default=1)
ap.add_argument('--bart', default='/path/to/eegaudit/data_raw/bart_base'); ap.add_argument('--lr', type=float, default=5e-5); ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--max_words', type=int, default=56); ap.add_argument('--max_tokens', type=int, default=64); ap.add_argument('--max_train', type=int, default=None)
a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device('cuda'); os.makedirs(a.out_dir, exist_ok=True); t0 = time.time()
D = pickle.load(open(a.cache, 'rb')); texts, recs = D['texts'], D['recs']
tr = [r for r in recs if r['split'] == 'train']; dv = [r for r in recs if r['split'] == 'dev']; te = [r for r in recs if r['split'] == 'test']
if a.max_train: tr = tr[:a.max_train]; dv = dv[:200]; te = te[:300]
if a.input == 'shuffled':
    perm = np.random.default_rng(a.seed + 1).permutation(len(tr)); tr = [dict(r, g=tr[j]['g']) for r, j in zip(tr, perm)]
Xtr = np.concatenate([r['words'] for r in tr]); mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6; del Xtr
from transformers import BartTokenizer, BartForConditionalGeneration
tok = BartTokenizer.from_pretrained(a.bart)
def collate(batch):
    n = min(a.max_words, max(r['words'].shape[0] for r in batch)); X = np.zeros((len(batch), n, 840), np.float32); M = np.zeros((len(batch), n), np.int64)
    for i, r in enumerate(batch):
        w = ((r['words'] - mu) / sd)[:n]; X[i, :len(w)] = w; M[i, :len(w)] = 1
    X = torch.from_numpy(X); M = torch.from_numpy(M)
    if a.input == 'noise': X = torch.randn_like(X)
    enc = tok([texts[r['g']] for r in batch], padding=True, truncation=True, max_length=a.max_tokens, return_tensors='pt')
    labels = enc['input_ids'].clone(); labels[enc['attention_mask'] == 0] = -100
    return X, M, labels, [r['g'] for r in batch], [r['task'] for r in batch]
loader = lambda rs, sh, bs: torch.utils.data.DataLoader(rs, batch_size=bs, shuffle=sh, collate_fn=collate, num_workers=0)
class VQ(nn.Module):
    def __init__(self, K, d, beta):
        super().__init__(); self.emb = nn.Embedding(K, d); nn.init.uniform_(self.emb.weight, -1 / K, 1 / K); self.beta = beta
    def forward(self, z, M):
        flat = z.reshape(-1, z.shape[-1]); d2 = (flat ** 2).sum(1, keepdim=True) - 2 * flat @ self.emb.weight.t() + (self.emb.weight ** 2).sum(1)[None]
        q = self.emb(d2.argmin(1)).view_as(z); m = M.float().unsqueeze(-1)
        loss = (((q - z.detach()) ** 2) * m).sum() / m.sum().clamp(min=1) / z.shape[-1] + self.beta * (((z - q.detach()) ** 2) * m).sum() / m.sum().clamp(min=1) / z.shape[-1]
        return z + (q - z).detach(), loss
class Model(nn.Module):
    def __init__(self, bart, arch):
        super().__init__(); self.arch = arch
        self.pre = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=840, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True), num_layers=1)
        if arch == 'bt': self.fc = nn.Linear(840, bart.config.d_model)
        else: self.to_code = nn.Linear(840, 512); self.vq = VQ(2048, 512, 0.25); self.fc = nn.Linear(512, bart.config.d_model)
        self.bart = bart
    def embed(self, X, M):
        h = self.pre(X, src_key_padding_mask=(M == 0))
        if self.arch == 'bt': return torch.relu(self.fc(h)), torch.zeros((), device=X.device)
        q, l = self.vq(self.to_code(h), M); return torch.relu(self.fc(q)), l
    def forward(self, X, M, labels):
        e, l = self.embed(X, M); return self.bart(inputs_embeds=e, attention_mask=M, labels=labels), l
    @torch.no_grad()
    def encode(self, X, M):
        e, _ = self.embed(X, M); H = self.bart.get_encoder()(inputs_embeds=e, attention_mask=M).last_hidden_state
        m = M.float().unsqueeze(-1); return (H * m).sum(1) / m.sum(1).clamp(min=1)
model = Model(BartForConditionalGeneration.from_pretrained(a.bart), a.arch).to(dev)
params = [{'params': [p for n, p in model.named_parameters() if not n.startswith('vq.')], 'lr': a.lr}]
if a.arch == 'dewave': params.append({'params': list(model.vq.parameters()), 'lr': 1e-3})
opt = torch.optim.AdamW(params, weight_decay=0.01); steps = a.epochs * math.ceil(len(tr) / a.bs)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, 0.05 * steps)) * max(0.0, (steps - s) / max(1, steps * 0.95)))
print(f'{a.tag}: train {len(tr)} dev {len(dv)} test {len(te)}; arch {a.arch} input {a.input}', flush=True)
for ep in range(a.epochs):
    model.train(); run = 0; nb = 0
    for X, M, L, *_ in loader(tr, True, a.bs):
        out, vql = model(X.to(dev), M.to(dev), L.to(dev)); loss = out.loss + vql; opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); run += out.loss.item(); nb += 1
    print(f'epoch {ep+1} train_loss {run/nb:.4f} ({time.time()-t0:.0f}s)', flush=True)
model.eval()
def encode_all(rs):
    Z = []; G = []; T = []
    with torch.no_grad():
        for X, M, L, g, task in loader(rs, False, 64): Z.append(model.encode(X.to(dev), M.to(dev)).cpu().numpy()); G += g; T += task
    return np.concatenate(Z), np.array(G), np.array(T)
Ztr, Gtr, _ = encode_all(tr); Zdv, Gdv, _ = encode_all(dv); Zte, Gte, Tte = encode_all(te); print(f'encoded ({time.time()-t0:.0f}s)', flush=True)
E = np.load(a.emb)['E']; E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
zm, zs = Ztr.mean(0), Ztr.std(0) + 1e-6; A = (Ztr - zm) / zs; Y = E[Gtr]
def ridge(alpha): 
    W = np.linalg.solve(A.T @ A + alpha * len(A) * np.eye(A.shape[1]), A.T @ (Y - Y.mean(0))); return lambda Z: ((Z - zm) / zs) @ W + Y.mean(0)
rng = np.random.default_rng(a.seed)
def score(P, G, cand, perm=True):
    pos = {g: j for j, g in enumerate(cand)}; idx = np.array([pos[g] for g in G]); C = E[cand]; N = len(cand); t1, tw = retrieval_scores(P, C, idx)
    o = dict(n_candidates=N, n_queries=len(G), top1=t1, two_way=tw, bits=wolpaw_bits(t1, N))
    if perm:
        sh = [retrieval_scores(P, C, rng.permutation(idx)) for _ in range(20)]; o['shuffled_pairs_top1'] = float(np.mean([s[0] for s in sh])); o['shuffled_pairs_two_way'] = float(np.mean([s[1] for s in sh]))
        Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12); S = Pn @ C.T; ts = S[np.arange(len(idx)), idx]; o['per_query_two_way'] = (1 - (S > ts[:, None]).sum(1) / (N - 1)).tolist(); o['per_query_hit'] = ((S > ts[:, None]).sum(1) == 0).astype(int).tolist()
    return o
dev_c = sorted(set(Gdv.tolist())); best = max((score(ridge(al)(Zdv), Gdv, dev_c, False)['two_way'], al) for al in [1e-3, 1e-2, 1e-1, 1, 10, 100])
pred = ridge(best[1]); Pte = pred(Zte)
res = dict(tag=a.tag, arch=a.arch, input=a.input, epochs=a.epochs, alpha=best[1], dev_two_way=best[0], n_train=len(tr), n_test=len(te), enc_dim=int(Ztr.shape[1]),
           pooled=score(Pte, Gte, sorted(set(Gte.tolist()))), within_task={t: score(Pte[Tte == t], Gte[Tte == t], sorted(set(Gte[Tte == t].tolist()))) for t in sorted(set(Tte.tolist()))},
           test_query_g=Gte.tolist(), test_query_task=Tte.tolist(), runtime_s=time.time() - t0)
json.dump(res, open(os.path.join(a.out_dir, f'{a.tag}_results.json'), 'w'))
f = lambda o: {k: round(v, 4) for k, v in o.items() if isinstance(v, float)}
print('pooled', f(res['pooled'])); [print(t, f(s)) for t, s in res['within_task'].items()]; print(f'done {time.time()-t0:.0f}s')
