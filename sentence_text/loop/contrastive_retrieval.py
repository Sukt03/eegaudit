"""R4: contrastive EEG-encoder -> MiniLM sentence-embedding retrieval on ZuCo (line of Brain-CLIPLM stage 1 / Collautti et al.),
on the same cache as BrainTranslator (Wang & Ji 840-d GD features per fixated word, <= 56 words, same train/dev/test split).
Encoder: Linear(840, d) -> 2-layer TransformerEncoder(d, 4 heads, ff 2d) -> masked mean pool -> Linear(d, 384); InfoNCE (symmetric)
against frozen all-MiniLM-L6-v2 sentence embeddings of the in-batch sentences (duplicates of the same sentence in a batch are masked
out of the negatives), learned temperature; early stop on dev retrieval top-1 (dev candidates = all dev sentences).
Inputs: eeg | noise (N(0,1) twin, same shape and mask; Jo et al. 2024) | shuffled (training EEG permuted against sentences; dev/test intact).
Scoring on test: pooled over tasks (N = all test sentences) and within task (candidates = test sentences of that task);
top-1, 2-way, Wolpaw bits; shuffled-pairs floor (20 permutations); bootstrap over test queries (1000 draws) for the per-mode CIs.
Run with python -I inside a GPU node."""
import argparse, json, os, sys, time, pickle, copy, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '/path/to/eegaudit/experiments/track_text')
from pilot_retrieval import embed_sentences, retrieval_scores, wolpaw_bits

ap = argparse.ArgumentParser()
ap.add_argument('--cache', required=True); ap.add_argument('--out_dir', required=True); ap.add_argument('--tag', required=True)
ap.add_argument('--input', choices=['eeg', 'noise', 'shuffled'], default='eeg')
ap.add_argument('--minilm', default='/path/to/eegaudit/data_raw/zuco/minilm')
ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--bs', type=int, default=128)
ap.add_argument('--d', type=int, default=256); ap.add_argument('--layers', type=int, default=2); ap.add_argument('--dropout', type=float, default=0.2)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--max_words', type=int, default=56); ap.add_argument('--max_eval', type=int, default=None)
ap.add_argument('--emb_cache', default=None, help='npz with MiniLM embeddings of all cache texts (created if missing)'); ap.add_argument('--save_z', action='store_true', help='save test-query embeddings, g, task (for P4 text PVI)')
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); os.makedirs(a.out_dir, exist_ok=True); t0 = time.time()
log_f = open(os.path.join(a.out_dir, f'{a.tag}_log.txt'), 'w')
def log(*s):
    msg = ' '.join(str(x) for x in s); print(msg, flush=True); log_f.write(msg + '\n'); log_f.flush()

D = pickle.load(open(a.cache, 'rb')); texts, recs = D['texts'], D['recs']
tr = [r for r in recs if r['split'] == 'train']; dv = [r for r in recs if r['split'] == 'dev']; te = [r for r in recs if r['split'] == 'test']
if a.max_eval: te = te[:a.max_eval]
if a.input == 'shuffled':
    perm = np.random.default_rng(a.seed + 1).permutation(len(tr)); tr = [dict(r, g=tr[j]['g']) for r, j in zip(tr, perm)]
Xtr = np.concatenate([r['words'] for r in tr]); mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6; del Xtr
# sentence embeddings (frozen target)
if a.emb_cache and os.path.exists(a.emb_cache):
    E_all = np.load(a.emb_cache)['E']; enc_name = str(np.load(a.emb_cache)['name'])
else:
    E_all, enc_name = embed_sentences([t if t else '.' for t in texts], a.minilm)
    if a.emb_cache: np.savez(a.emb_cache, E=E_all, name=enc_name)
E_all = E_all / (np.linalg.norm(E_all, axis=1, keepdims=True) + 1e-12); E_t = torch.from_numpy(E_all).float().to(dev)
def collate(batch):
    n = min(a.max_words, max(r['words'].shape[0] for r in batch))
    X = np.zeros((len(batch), n, 840), np.float32); M = np.zeros((len(batch), n), np.int64)
    for i, r in enumerate(batch):
        w = ((r['words'] - mu) / sd)[:n]; X[i, :len(w)] = w; M[i, :len(w)] = 1
    X = torch.from_numpy(X); M = torch.from_numpy(M)
    if a.input == 'noise': X = torch.randn_like(X)
    return X, M, torch.tensor([r['g'] for r in batch]), [r['task'] for r in batch], [r['subject'] for r in batch]
def loader(rs, shuffle, bs): return torch.utils.data.DataLoader(rs, batch_size=bs, shuffle=shuffle, collate_fn=collate, num_workers=0)

class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(840, a.d), nn.LayerNorm(a.d), nn.Dropout(a.dropout))
        self.pos = nn.Parameter(torch.zeros(1, a.max_words, a.d)); nn.init.normal_(self.pos, std=0.02)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(a.d, 4, 2 * a.d, a.dropout, batch_first=True, norm_first=True), a.layers)
        self.out = nn.Linear(a.d, E_all.shape[1]); self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
    def forward(self, X, M):
        h = self.inp(X) + self.pos[:, :X.shape[1]]; h = self.tf(h, src_key_padding_mask=(M == 0))
        m = M.float().unsqueeze(-1); h = (h * m).sum(1) / m.sum(1).clamp(min=1); return F.normalize(self.out(h), dim=-1)
model = Enc().to(dev); n_par = sum(p.numel() for p in model.parameters())
log(f'cfg {vars(a)}'); log(f'train {len(tr)} dev {len(dv)} test {len(te)} records; params {n_par/1e6:.2f}M; target {enc_name} {E_all.shape}')
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
steps = a.epochs * math.ceil(len(tr) / a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
def info_nce(z, g):
    s = model.logit_scale.exp().clamp(max=100) * z @ E_t[g].T          # (B, B)
    same = (g[:, None] == g[None, :]); eye = torch.eye(len(g), dtype=torch.bool, device=dev)
    s = s.masked_fill(same & ~eye, -1e4)                               # same sentence from another subject is not a negative
    lab = torch.arange(len(g), device=dev); return 0.5 * (F.cross_entropy(s, lab) + F.cross_entropy(s.T, lab))
@torch.no_grad()
def predict(rs):
    model.eval(); Z = []; G = []; T = []; S = []
    for X, M, g, task, subj in loader(rs, False, 256):
        Z.append(model(X.to(dev), M.to(dev)).cpu().numpy()); G += g.tolist(); T += task; S += subj
    model.train(); return np.concatenate(Z), np.array(G), np.array(T), S
def score(Z, G, cand_g, rng=None, n_perm=20):
    pos = {g: j for j, g in enumerate(cand_g)}; idx = np.array([pos[g] for g in G]); C = E_all[cand_g]
    t1, tw = retrieval_scores(Z, C, idx); N = len(cand_g); out = {'n_candidates': N, 'n_queries': len(G), 'top1': t1, 'two_way': tw, 'bits': wolpaw_bits(t1, N)}
    if rng is not None:
        sh = [retrieval_scores(Z, C, rng.permutation(idx)) for _ in range(n_perm)]
        out.update(shuffled_pairs_top1=float(np.mean([s[0] for s in sh])), shuffled_pairs_two_way=float(np.mean([s[1] for s in sh])),
                   shuffled_pairs_bits=float(np.mean([wolpaw_bits(s[0], N) for s in sh])))
        # bootstrap over queries
        P = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12); Sm = P @ C.T; ts = Sm[np.arange(len(idx)), idx]
        hit = (Sm > ts[:, None]).sum(1) == 0; twq = 1 - ((Sm > ts[:, None]).sum(1) / (N - 1))
        bs = [(hit[b].mean(), twq[b].mean()) for b in (rng.integers(0, len(idx), len(idx)) for _ in range(1000))]
        out.update(top1_ci95=[float(np.percentile([b[0] for b in bs], 2.5)), float(np.percentile([b[0] for b in bs], 97.5))],
                   two_way_ci95=[float(np.percentile([b[1] for b in bs], 2.5)), float(np.percentile([b[1] for b in bs], 97.5))],
                   per_query_hit=hit.astype(int).tolist(), per_query_two_way=twq.tolist())
    return out
dev_g = sorted({r['g'] for r in dv})
best = (-1, None, -1)
for ep in range(a.epochs):
    run = 0; nb = 0
    for X, M, g, *_ in loader(tr, True, a.bs):
        loss = info_nce(model(X.to(dev), M.to(dev)), g.to(dev)); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        run += loss.item(); nb += 1
    Zd, Gd, _, _ = predict(dv); s = score(Zd, Gd, dev_g)
    log(f'epoch {ep+1}/{a.epochs} loss {run/nb:.4f} temp {1/model.logit_scale.exp().item():.3f} dev top1 {s["top1"]:.4f} two_way {s["two_way"]:.4f} (N={s["n_candidates"]}) ({time.time()-t0:.0f}s)')
    if s['top1'] > best[0]: best = (s['top1'], copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}), ep + 1)
model.load_state_dict(best[1]); log(f'best epoch {best[2]} dev top1 {best[0]:.4f}')
Zt, Gt, Tt, St = predict(te); rng = np.random.default_rng(a.seed)
res = {'tag': a.tag, 'input': a.input, 'cfg': vars(a), 'params_M': n_par / 1e6, 'best_epoch': best[2], 'best_dev_top1': best[0], 'encoder_target': enc_name,
       'n_train_records': len(tr), 'n_dev_records': len(dv), 'n_test_records': len(te), 'pooled': score(Zt, Gt, sorted(set(Gt.tolist())), rng), 'within_task': {}}
for task in sorted(set(Tt.tolist())):
    m = Tt == task; res['within_task'][task] = score(Zt[m], Gt[m], sorted(set(Gt[m].tolist())), rng)
res['test_query_order'] = [{'g': int(g), 'task': t, 'subject': s} for g, t, s in zip(Gt, Tt, St)]
if a.save_z: np.savez(os.path.join(a.out_dir, f'{a.tag}_testZ.npz'), Z=Zt, g=Gt, task=Tt)
res['runtime_s'] = time.time() - t0
json.dump(res, open(os.path.join(a.out_dir, f'{a.tag}_results.json'), 'w'))
log('pooled', json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res['pooled'].items() if not k.startswith('per_query')}))
for task, s in res['within_task'].items():
    log(task, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if not k.startswith('per_query')}))
log(f'done {time.time()-t0:.0f}s')
