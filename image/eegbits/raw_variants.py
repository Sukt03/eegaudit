"""Debug step (AIDE loop): the raw-EEG linear probe yields ~0 bits and shows no gain from averaging repetitions, which
contradicts published linear decoders on THINGS-EEG2. Test preprocessing variants on one subject, same ridge, same test:
  A  per-sample z-score, full window (current pipeline)
  B  centre only, global scale (no per-sample standardisation), full window
  C  B + window 0-0.8 s
  D  C + decimate to 100 Hz (block average by 2.5 -> use every 2nd and 3rd sample mean: implemented as reshape mean over 5 -> 50 Hz)
  E  B + window 0-0.8 s + 50 Hz block average, channels occipito-parietal only
For each: ridge (dual form on the GPU) at full rank, k = 1 and k = 80, 2-way / top-1 / bits. Also PCA-L curve for the best variant.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from run import CLIP, load_torch, load_raw, metrics, pca_scores, ridge_predict, pick_alpha, ALPHAS

dev = torch.device("cuda")
subject = sys.argv[1] if len(sys.argv) > 1 else "sub-01"
Ytr, Yte = load_torch(CLIP / "ViT-H-14_features_train.pt"), load_torch(CLIP / "ViT-H-14_features_test.pt")
Xtr4, Xte80, meta = load_raw(subject)
t = np.linspace(meta["times"][0], meta["times"][1], Xtr4.shape[-1])
ch = meta["ch_names"]
occ = [i for i, c in enumerate(ch) if c in ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8"]]


def feats(X, window=None, block=1, chans=None, zscore=False, stats=None):
    if chans is not None:
        X = X[:, chans]
    if window is not None:
        m = (t >= window[0]) & (t < window[1]); X = X[..., m]
    if block > 1:
        T = (X.shape[-1] // block) * block
        X = X[..., :T].reshape(*X.shape[:-1], T // block, block).mean(-1)
    F = X.reshape(X.shape[0], -1)
    if stats is None:
        mu = F.mean(0); sd = F.std(0) + 1e-6 if zscore else np.full(F.shape[1], F.std(), dtype=np.float32)
        stats = (mu, sd)
    return (F - stats[0]) / stats[1], stats


def kernel_ridge_eval(Ftr, Fte_list, alphas=(1e2, 1e3, 1e4, 1e5, 1e6)):
    """Dual ridge on the GPU: K = F Fᵀ; pick alpha on a held-out tenth; return metrics per test set."""
    Xt = torch.from_numpy(Ftr).to(dev); Yt = torch.from_numpy(Ytr - Ytr.mean(0)).to(dev)
    n = Xt.shape[0]; idx = torch.randperm(n, device=dev); cut = int(0.9 * n); tr, va = idx[:cut], idx[cut:]
    K = Xt @ Xt.T
    best, best_a = -1, None
    for a in alphas:
        Ktr = K[tr][:, tr]; A = torch.linalg.solve(Ktr + a * torch.eye(len(tr), device=dev), Yt[tr])
        pred = (K[va][:, tr] @ A).cpu().numpy() + Ytr.mean(0)
        s = metrics(pred, Ytr[va.cpu().numpy()])["mrr"]
        if s > best: best, best_a = s, a
    A = torch.linalg.solve(K + best_a * torch.eye(n, device=dev), Yt)
    out = []
    for Fte in Fte_list:
        pred = (torch.from_numpy(Fte).to(dev) @ Xt.T @ A).cpu().numpy() + Ytr.mean(0)
        out.append(metrics(pred, Yte))
    del K; torch.cuda.empty_cache()
    return best_a, out


Xtr = Xtr4.mean(1); Xte1, Xte80m = Xte80[:, :1].mean(1), Xte80.mean(1)
variants = {
    "A_zscore_full": dict(zscore=True),
    "B_global_full": dict(zscore=False),
    "C_global_0-0.8s": dict(zscore=False, window=(0.0, 0.8)),
    "D_global_0-0.8s_50Hz": dict(zscore=False, window=(0.0, 0.8), block=5),
    "E_global_0-0.8s_50Hz_occpar": dict(zscore=False, window=(0.0, 0.8), block=5, chans=occ),
}
res = {}
for name, kw in variants.items():
    t0 = time.time()
    Ftr, st = feats(Xtr, **kw); F1, _ = feats(Xte1, stats=st, **kw); F80, _ = feats(Xte80m, stats=st, **kw)
    a, (m1, m80) = kernel_ridge_eval(Ftr.astype(np.float32), [F1.astype(np.float32), F80.astype(np.float32)])
    res[name] = dict(D=int(Ftr.shape[1]), alpha=a, k1=m1, k80=m80, seconds=round(time.time() - t0, 1))
    print(f"{name:28s} D={Ftr.shape[1]:6d} alpha={a:g}  k=1: 2way {100*m1['two_way']:.1f} top1 {100*m1['top1']:.1f}  | k=80: 2way {100*m80['two_way']:.1f} top1 {100*m80['top1']:.1f} bits {m80['bits']:.2f}  ({res[name]['seconds']}s)", flush=True)
Path("/path/to/eegaudit/results/eegbits").mkdir(exist_ok=True)
json.dump(dict(subject=subject, variants=res), open(f"/path/to/eegaudit/results/eegbits/raw_variants_{subject}.json", "w"), indent=1)
