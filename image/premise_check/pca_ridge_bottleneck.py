"""Premise check for the ideation cards (not the paper experiment).

Question: does a closed-form rank-L bottleneck (PCA-L on ATM EEG embeddings -> ridge -> CLIP ViT-H/14 image embedding)
beat matched controls at small L? The trained two-layer bottleneck in bbeeg-atm was at chance for L <= 50 on sub-08;
three blind reviewers suspect an under-trained head rather than an information limit.

Design (leak-free for this question): fit PCA + ridge on ATM *training* features (16,540 images, mean of 4 repetitions)
against ViT-H/14 train targets; evaluate on ATM's 200 held-out test images (80-repetition averages) against the 200 test targets.
PCA bases are nested, so one SVD per subject serves every L (column slice).
Controls: (noise) N(0,1) input of the same shape, train and test; (trial-shuffled) EEG rows permuted against targets at
training time; (shuffled-pair) predictions scored against permuted test targets = chance floor.
Metrics on the 200-way test set: top-1, top-5, 2-way identification, mean reciprocal rank, matched cosine.
"""
import collections, json, sys, time
from pathlib import Path
import numpy as np
import torch

EMB = Path("/path/to/atm/data/hf_probe/emb_eeg")
GEN = Path("/path/to/atm/data/hf_probe")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results"
LS = [2, 5, 10, 25, 50, 100, 250, 1024]
SUBS = [f"sub-{i:02d}" for i in range(1, 11)]
SEEDS = [0, 1, 2]
ALPHAS = [1e-1, 1, 10, 100, 1000]


def to_np(x):
    if isinstance(x, dict):
        for k in ("img_features", "image_features", "features"):
            if k in x:
                x = x[k]
                break
        else:
            x = next(v for v in x.values() if hasattr(v, "shape"))
    return x.float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=np.float32)


def metrics(pred, tgt):
    p = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    t = tgt / np.linalg.norm(tgt, axis=1, keepdims=True)
    S = p @ t.T
    n = S.shape[0]
    diag = np.diag(S)
    rank = (S > diag[:, None]).sum(1) + 1
    twoway = ((S < diag[:, None]).sum(1) / (n - 1)).mean()
    return dict(top1=float((rank == 1).mean()), top5=float((rank <= 5).mean()), two_way=float(twoway),
                mrr=float((1.0 / rank).mean()), cosine=float(diag.mean()))


def pca_scores(Xtr, Xte):
    """Standardise, one SVD; return nested PCA scores for train and test (columns ordered by variance)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    U, S, Vt = np.linalg.svd(Ztr, full_matrices=False)
    return Ztr @ Vt.T, Zte @ Vt.T


def ridge_predict(Ztr, Ytr, Zte, alpha):
    ym = Ytr.mean(0)
    A = Ztr.T @ Ztr + alpha * np.eye(Ztr.shape[1], dtype=Ztr.dtype)
    W = np.linalg.solve(A, Ztr.T @ (Ytr - ym))
    return Zte @ W + ym


def pick_alpha(Ztr, Ytr, rng):
    idx = rng.permutation(len(Ztr)); cut = int(0.9 * len(idx))
    tr, va = idx[:cut], idx[cut:]
    return max(ALPHAS, key=lambda a: metrics(ridge_predict(Ztr[tr], Ytr[tr], Ztr[va], a), Ytr[va])["mrr"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Ytr = to_np(torch.load(GEN / "ViT-H-14_features_train.pt", map_location="cpu"))
    Yte = to_np(torch.load(GEN / "ViT-H-14_features_test.pt", map_location="cpu"))
    print("targets", Ytr.shape, Yte.shape, flush=True)
    rows = []
    for s in SUBS:
        t0 = time.time()
        Xtr_rep = to_np(torch.load(EMB / f"ATM_S_eeg_features_{s}_train.pt", map_location="cpu"))
        Xte = to_np(torch.load(EMB / f"ATM_S_eeg_features_{s}_test.pt", map_location="cpu"))
        assert Xtr_rep.shape[0] == 4 * Ytr.shape[0], (Xtr_rep.shape, Ytr.shape)
        Xtr = Xtr_rep.reshape(Ytr.shape[0], 4, -1).mean(1)          # image-major, 4 consecutive repetitions
        Ztr, Zte = pca_scores(Xtr, Xte)
        noise = []
        for seed in SEEDS:
            r = np.random.default_rng(100 + seed)
            noise.append(pca_scores(r.standard_normal(Xtr.shape).astype(np.float32), r.standard_normal(Xte.shape).astype(np.float32)))
        rng = np.random.default_rng(0)
        for L in LS:
            alpha = pick_alpha(Ztr[:, :L], Ytr, rng)
            pred = ridge_predict(Ztr[:, :L], Ytr, Zte[:, :L], alpha)
            rows.append(dict(subject=s, L=L, condition="eeg", seed=0, alpha=alpha, **metrics(pred, Yte)))
            for seed in SEEDS:
                r = np.random.default_rng(200 + seed)
                Ntr, Nte = noise[seed]
                rows.append(dict(subject=s, L=L, condition="noise", seed=seed, alpha=alpha, **metrics(ridge_predict(Ntr[:, :L], Ytr, Nte[:, :L], alpha), Yte)))
                perm = r.permutation(len(Ztr))
                rows.append(dict(subject=s, L=L, condition="trial_shuffled", seed=seed, alpha=alpha, **metrics(ridge_predict(Ztr[perm, :L], Ytr, Zte[:, :L], alpha), Yte)))
                rows.append(dict(subject=s, L=L, condition="shuffled_pair", seed=seed, alpha=alpha, **metrics(pred, Yte[r.permutation(len(Yte))])))
        print(f"{s} done in {time.time()-t0:.0f}s  (eeg top-1 by L: " + ", ".join(f"{L}:{100*[x for x in rows if x['subject']==s and x['L']==L and x['condition']=='eeg'][0]['top1']:.1f}" for L in LS) + ")", flush=True)
    (OUT / "pca_ridge_bottleneck.json").write_text(json.dumps(rows, indent=1))
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["condition"], r["L"])].append(r)
    lines = ["| condition | L | top-1 % | top-5 % | 2-way % | MRR | cosine |", "|---|---|---|---|---|---|---|"]
    for cond in ["eeg", "noise", "trial_shuffled", "shuffled_pair"]:
        for L in LS:
            g = agg[(cond, L)]
            m = {k: np.mean([x[k] for x in g]) for k in ["top1", "top5", "two_way", "mrr", "cosine"]}
            lines.append(f"| {cond} | {L} | {100*m['top1']:.1f} | {100*m['top5']:.1f} | {100*m['two_way']:.1f} | {m['mrr']:.3f} | {m['cosine']:.3f} |")
    per = []
    for s in SUBS:
        Lstar = None
        for L in LS:
            e = [r for r in rows if r["subject"] == s and r["L"] == L and r["condition"] == "eeg"][0]["top1"]
            nmax = max(r["top1"] for r in rows if r["subject"] == s and r["L"] == L and r["condition"] in ("noise", "trial_shuffled"))
            if e - nmax > 0.02:
                Lstar = L
                break
        per.append(f"{s}: L*={Lstar}")
    md = ("# Premise check: closed-form PCA-L + ridge bottleneck, ATM embeddings -> CLIP, 10 subjects, 200-way held-out test\n\n"
          "Mean over 10 subjects (controls also over 3 seeds). Chance: top-1 0.5%, top-5 2.5%, 2-way 50%, MRR ≈ 0.029.\n\n"
          + "\n".join(lines) + "\n\nSmallest L where EEG top-1 beats the best control by > 2 pp: " + "; ".join(per) + "\n")
    (OUT / "PREMISE_CHECK_pca_ridge.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
