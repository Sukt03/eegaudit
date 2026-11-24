"""RUNNER FIX 4 (round_20261123_auto; EXPERIMENTS G2(b) ii-iii, METHODS B3b): text ridge probe WITHIN task + subspace-planted control.
Probe = experiments/ext/text_d5_probe.py (record mean of the 840-d ZuCo word features -> ridge -> MiniLM sentence embedding; soft12 LOO-temperature InfoNCE against the
unique test sentences; bits per test record), re-implemented with torch-GPU scoring. GATE: pooled real / planted(amp 1, all dims) / noise at K = 13, 52, 210, 840 ==
results/ext/text_d5_probe/summary.json to 1e-4 (the gate uses the original's ridge, which has NO intercept: bug list).
Cells:
  pooled | SR | NR | NR2      train and test records restricted to the task; candidates = that task's unique test sentences
  arms: real | noise (N(0,1) features, seed 0) | shuffled (training sentence embeddings permuted across training records, seed 3)
  subspace plant: the gf_zuco_plant_v2 lexical code (hash-of-token Gaussian vector, per-feature training SD) added ONLY to the first m of the probe's nested dims
     (m = 13, 52; amp 0.3, 1.0), on the real features and on N(0,1) features ('pure'); expected: yield rises until K = m then is flat, i.e. L90 = m.
  ridge variants: 'orig' (no intercept, as the saved probe) and 'c' (targets centred on the fitted rows, prediction + mean) as a sensitivity row.
CI: cluster bootstrap over test SENTENCES (10000, GPU). L90 = smallest K with yield >= 0.9 x yield(840); printed only if the K=840 lower bound exceeds the noise arm.
Usage: --out DIR"""
import argparse, hashlib, json, pickle, re, sys, time, zlib
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path('/path/to/eegaudit'); CODE_SHA256 = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--quick', action='store_true'); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
dev = 'cuda'; t0 = time.time(); log = lambda *m: print(f'[{time.time()-t0:6.1f}s]', *m, flush=True)
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained(str(ROOT / 'data_raw/zuco/minilm')); mm = AutoModel.from_pretrained(str(ROOT / 'data_raw/zuco/minilm')).to(dev).eval()
def emb(txts):
    o = []
    for i in range(0, len(txts), 64):
        e = tok(txts[i:i + 64], padding=True, truncation=True, return_tensors='pt').to(dev)
        with torch.no_grad(): h = mm(**e).last_hidden_state
        m = e['attention_mask'].unsqueeze(-1).float(); o.append(F.normalize((h * m).sum(1) / m.sum(1), dim=1))
    return torch.cat(o).double()
D = pickle.load(open(ROOT / 'results/track_text/brain_translator/cache_full.pkl', 'rb')); recs, texts = D['recs'], D['texts']
E = emb(texts)                                                                        # one embedding per sentence slot
Xall = torch.tensor(np.stack([r['words'].mean(0) for r in recs]).astype(np.float64), device=dev); gid = np.array([r['g'] for r in recs]); split = np.array([r['split'] for r in recs]); task = np.array([r['task'] for r in recs])
# plant matrix: sum_{i<n} v(token_i) / n_rows per record (= what gf_zuco_plant_v2 adds to the record mean, per unit amp x SD)
sd = torch.tensor(np.concatenate([r['words'] for r in recs if r['split'] == 'train']).std(0) + 1e-6, device=dev, dtype=torch.float64)
norm = lambda w: re.sub(r"[^a-z0-9']", '', w.lower()) or '_'; vc = {}
def vec(t):
    if t not in vc: vc[t] = np.random.default_rng(zlib.crc32(t.encode())).standard_normal(840).astype(np.float32)
    return vc[t]
PM = np.zeros((len(recs), 840), np.float32)
for i, r in enumerate(recs):
    tk = [norm(w) for w in texts[r['g']].split()]; n = min(len(r['words']), len(tk))
    if n: PM[i] = np.sum([vec(t) for t in tk[:n]], 0) / len(r['words'])
PM = torch.tensor(PM, device=dev, dtype=torch.float64); log('features', tuple(Xall.shape), 'tokens', len(vc))
KS = [13, 52, 210, 840]; perm840 = np.random.default_rng(1234).permutation(840); TAUS = torch.logspace(-2.5, 0, 12, dtype=torch.float64, device=dev)
assert not (set(gid[split == 'train']) & set(gid[split == 'test'])) and not (set(texts[g] for g in gid[split == 'train']) & set(texts[g] for g in gid[split == 'test'])), 'train/test sentence leak'

def ridge(Xtr, Ytr, Xte, centre, seed=0):
    mu, s = Xtr.mean(0), Xtr.std(0) + 1e-6; A = (Xtr - mu) / s; T = (Xte - mu) / s          # numpy std (ddof 0) as the original
    perm = torch.randperm(len(A), generator=torch.Generator().manual_seed(seed)).to(dev); nv = len(A) // 10; va, trn = perm[:nv], perm[nv:]; best = (-9, 1.0); I = torch.eye(A.shape[1], device=dev, dtype=torch.float64)
    for lam in torch.logspace(0, 6, 13, dtype=torch.float64, device=dev):
        ym = Ytr[trn].mean(0) if centre else 0; W = torch.linalg.solve(A[trn].T @ A[trn] + lam * I, A[trn].T @ (Ytr[trn] - ym)); c = float(F.cosine_similarity(A[va] @ W + ym, Ytr[va], dim=1).mean())
        if c > best[0]: best = (c, float(lam))
    ym = Ytr.mean(0) if centre else 0; W = torch.linalg.solve(A.T @ A + best[1] * I, A.T @ (Ytr - ym)); return T @ W + ym, best[1]

def rowbits(P, C, idx):                                                               # soft12 leave-one-query-out temperature, per-record bits
    S = F.normalize(P, dim=1) @ C.T; n = len(S); r = torch.arange(n, device=dev); Z = S[None] / TAUS[:, None, None]; CE = -(Z - torch.logsumexp(Z, -1, keepdim=True))[:, r, idx] / np.log(2)
    loo = (CE.sum(1, keepdim=True) - CE) / (n - 1); return np.log2(C.shape[0]) - CE[loo.argmin(0), r]

def clboot(b, idx, nc, B=10000):                                                      # cluster bootstrap over sentences
    g = torch.Generator(device=dev); g.manual_seed(0); s = torch.zeros(nc, dtype=torch.float64, device=dev).index_add_(0, idx, b); c = torch.zeros(nc, dtype=torch.float64, device=dev).index_add_(0, idx, torch.ones_like(b))
    ix = torch.randint(0, nc, (B, nc), generator=g, device=dev); m = s[ix].sum(1) / c[ix].sum(1); return [float(m.quantile(.025)), float(m.quantile(.975))]

def cell(name, tasks, centre):
    tr = torch.tensor(np.where((split == 'train') & np.isin(task, tasks))[0], device=dev); te = torch.tensor(np.where((split == 'test') & np.isin(task, tasks))[0], device=dev)
    gte = gid[te.cpu().numpy()]; ug = sorted(set(gte)); C = E[torch.tensor(ug, device=dev)]; idx = torch.tensor([ug.index(x) for x in gte], device=dev); Ytr = E[torch.tensor(gid[tr.cpu().numpy()], device=dev)]
    res = dict(n_train=len(tr), n_test=len(te), n_candidates=len(ug), chance_ceiling_bits=float(np.log2(len(ug))), arms={})
    def run(arm, X, Y=Ytr):
        o = {}
        for K in KS:
            keep = torch.tensor(perm840[:K].copy(), device=dev); P, lam = ridge(X[tr][:, keep], Y, X[te][:, keep], centre); b = rowbits(P, C, idx); o[str(K)] = dict(bits=float(b.mean()), ci95=clboot(b, idx, len(ug)), lam=lam)
        res['arms'][arm] = o; return o
    run('real', Xall)
    for K in KS:                                                                     # original: fresh rng(0) per K, train block then test block, of that K's shape
        rng = np.random.default_rng(0); keep = torch.tensor(perm840[:K].copy(), device=dev); a_ = torch.tensor(rng.standard_normal((len(tr), K)), device=dev); b_ = torch.tensor(rng.standard_normal((len(te), K)), device=dev)
        P, lam = ridge(a_, Ytr, b_, centre); b = rowbits(P, C, idx); res['arms'].setdefault('noise', {})[str(K)] = dict(bits=float(b.mean()), ci95=clboot(b, idx, len(ug)), lam=lam)
    pg = torch.randperm(len(tr), generator=torch.Generator().manual_seed(3)).to(dev); run('shuffled', Xall, Ytr[pg])
    Xnz = torch.randn(Xall.shape, generator=torch.Generator(device=dev).manual_seed(9), device=dev, dtype=torch.float64)
    plants = [('planted_all_amp1.0', 840, 1.0, Xall)] + ([] if a.quick else [(f'plant_m{m}_amp{amp}_{nm}', m, amp, base) for m in (13, 52) for amp in (0.3, 1.0) for nm, base in (('real', Xall), ('pure', Xnz))])
    for nm, m, amp, base in plants:
        mask = torch.zeros(840, dtype=torch.float64, device=dev); mask[torch.tensor(perm840[:m].copy(), device=dev)] = 1; o = run(nm, base + amp * sd * mask * PM)
        full = o['840']; fl = res['arms']['noise']['840']['bits']; o['L90'] = next(K for K in KS if o[str(K)]['bits'] >= 0.9 * full['bits']) if full['ci95'][0] > max(fl, 0) else None; o['planted_dims'] = m
    full = res['arms']['real']['840']; fl = res['arms']['noise']['840']['bits']
    res['real_L90'] = next(K for K in KS if res['arms']['real'][str(K)]['bits'] >= 0.9 * full['bits']) if full['ci95'][0] > max(fl, 0) else None
    res['real_above_noise_at_840'] = bool(full['ci95'][0] > max(fl, 0))
    log(name, 'centre' if centre else 'orig', f"n_te {len(te)} cand {len(ug)}", {arm: [round(v[str(K)]['bits'], 3) for K in KS] + [v.get('L90')] for arm, v in res['arms'].items()}); return res

S = dict(NOT_CITABLE='until independent review', code_sha256=CODE_SHA256, KS=KS, cells={})
for centre in (False, True):
    for name, tasks in (('pooled', ['SR', 'NR', 'NR2']), ('SR', ['SR']), ('NR', ['NR']), ('NR2', ['NR2'])):
        S['cells'][f"{name}|{'c' if centre else 'orig'}"] = cell(name, tasks, centre)
sv = json.loads((ROOT / 'results/ext/text_d5_probe/summary.json').read_text()); G = S['cells']['pooled|orig']['arms']
gate = {f'{x}|{K}': abs(G[y][str(K)]['bits'] - sv[x][str(K)]) for x, y in (('real', 'real'), ('noise', 'noise'), ('planted', 'planted_all_amp1.0')) for K in KS}
S['gate'] = dict(max_abs_diff=max(gate.values()), per=gate, passed=bool(max(gate.values()) < 1e-4)); S['runtime_s'] = round(time.time() - t0, 1)
(out / 'summary.json').write_text(json.dumps(S, indent=1)); log('GATE', S['gate'])
