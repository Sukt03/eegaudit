"""Text D5 ridge-probe width curve with planted positive control (, 23 Nov 2026; ideator cell 5).
Probe: mean over word rows of the 840-d ZuCo features of a record -> ridge -> MiniLM sentence embedding (fit on TRAIN records, penalty by 90/10 split of train),
scored on test records against the test sentences' MiniLM embeddings (InfoNCE soft12 LOO temperature as eegbits.run.soft_bits; bits per record, 103 candidates).
Width: keep K of 840 dims (fixed random subset, seed 1234, nested, same mask as train_dewave --keep). Arms: real cache | planted cache (amp 1.0, from gf_zuco_plant_v2) | noise N(0,1) of identical shape.
L90 = smallest K whose real-arm yield >= 0.9 x full-width yield (undefined if full-width yield CI/floor says no yield: reported as 'at floor')."""
import json, pickle, sys, argparse, numpy as np, torch
from pathlib import Path
ROOT = Path('/path/to/eegaudit'); sys.path.insert(0, str(ROOT / 'experiments/eegbits'))
from run import soft_bits
ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(ROOT / 'results/ext/text_d5_probe')); ap.add_argument('--planted', default=str(ROOT / 'results/gpu_fill_20261123/zuco_plant/cache_amp1.0.pkl')); a = ap.parse_args()
Path(a.out).mkdir(parents=True, exist_ok=True); dev = 'cuda'
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained(str(ROOT / 'data_raw/zuco/minilm')); mm = AutoModel.from_pretrained(str(ROOT / 'data_raw/zuco/minilm')).to(dev).eval()
def emb(txts):
    out = []
    for i in range(0, len(txts), 64):
        e = tok(txts[i:i + 64], padding=True, truncation=True, return_tensors='pt').to(dev)
        with torch.no_grad(): h = mm(**e).last_hidden_state
        m = e['attention_mask'].unsqueeze(-1).float(); v = (h * m).sum(1) / m.sum(1); out.append(torch.nn.functional.normalize(v, dim=1).cpu())
    return torch.cat(out).numpy()
def feats(D):
    tr = [r for r in D['recs'] if r['split'] == 'train']; te = [r for r in D['recs'] if r['split'] == 'test']
    f = lambda rs: np.stack([r['words'].mean(0) for r in rs]).astype(np.float64)
    return f(tr), emb([D['texts'][r['g']] for r in tr]), f(te), emb([D['texts'][r['g']] for r in te]), [r['g'] for r in te]
def ridge_bits(Xtr, Ytr, Xte, Yte, seed=0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; A = torch.tensor((Xtr - mu) / sd, device=dev); B = torch.tensor(Ytr, device=dev, dtype=torch.float64); T = torch.tensor((Xte - mu) / sd, device=dev)
    perm = torch.randperm(len(A), generator=torch.Generator().manual_seed(seed)).to(dev); nv = len(A) // 10; va, trn = perm[:nv], perm[nv:]; best = (-9, 1.0)
    for lam in torch.logspace(0, 6, 13, dtype=torch.float64, device=dev):
        W = torch.linalg.solve(A[trn].T @ A[trn] + lam * torch.eye(A.shape[1], device=dev, dtype=torch.float64), A[trn].T @ B[trn]); c = float(torch.nn.functional.cosine_similarity(A[va] @ W, B[va], dim=1).mean())
        if c > best[0]: best = (c, float(lam))
    W = torch.linalg.solve(A.T @ A + best[1] * torch.eye(A.shape[1], device=dev, dtype=torch.float64), A.T @ B); P = (T @ W).cpu().numpy()
    return P
def score(P, Yte, g):  # query = test record, candidates = unique test sentences, truth by sentence id
    ug = sorted(set(g)); C = np.stack([Yte[g.index(u)] for u in ug]); idx = np.array([ug.index(x) for x in g])
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12); S = Pn @ C.T
    return float(np.mean(np.log2(len(ug)) - ce_rows(S, idx)))
def ce_rows(S, idx):  # soft12 leave-one-query-out temperature, per-row CE bits
    taus = np.logspace(-2.5, 0, 12); n = len(S); CE = []
    for t in taus:
        Z = S / t; Z = Z - Z.max(1, keepdims=True); lp = Z - np.log(np.exp(Z).sum(1, keepdims=True)); CE.append(-lp[np.arange(n), idx] / np.log(2))
    CE = np.stack(CE); loo = (CE.sum(1, keepdims=True) - CE) / (n - 1); return CE[loo.argmin(0), np.arange(n)]
KS = [13, 52, 210, 840]; perm840 = np.random.default_rng(1234).permutation(840); res = {}
Dreal = pickle.load(open(ROOT / 'results/track_text/brain_translator/cache_full.pkl', 'rb')); arms = {'real': Dreal}
if Path(a.planted).exists(): arms['planted'] = pickle.load(open(a.planted, 'rb'))
for nm, D in arms.items():
    Xtr, Ytr, Xte, Yte, g = feats(D); res[nm] = {}
    for K in KS:
        keep = perm840[:K]; P = ridge_bits(Xtr[:, keep], Ytr, Xte[:, keep], Yte); res[nm][str(K)] = score(P, Yte, g)
        rng = np.random.default_rng(0); Pn = ridge_bits(rng.standard_normal(Xtr[:, keep].shape), Ytr, rng.standard_normal(Xte[:, keep].shape), Yte); res.setdefault('noise', {})[str(K)] = score(Pn, Yte, g) if nm == 'real' else res['noise'][str(K)]
        print(nm, K, round(res[nm][str(K)], 4), 'noise', round(res['noise'][str(K)], 4), flush=True)
json.dump(res, open(Path(a.out) / 'summary.json', 'w'), indent=1)
