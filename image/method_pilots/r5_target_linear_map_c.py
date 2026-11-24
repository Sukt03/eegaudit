"""Round 5, attack 1 (part c): the EEG-fitted version of the target swap without retraining. ONE frozen EEG embedding (ATM-S,
1024-d; the only encoder with cached TRAINING embeddings: 16,540 images x 4 repetitions, test 200 images x 80 repetitions) is read
out by a full-width ridge fitted on the training images into EVERY target space, then scored 200-way against that space's test
features (the paper's own embedding -> ridge -> CLIP pipeline of run.py, with the target swapped):
  targets: CLIP ViT-L/14 (NICE's clip_feature_maps, the swap's CLIP arm), DINOv2 ViT-B/14, pixel-PCA-768, and ATM-S's own training
           target CLIP ViT-H/14 as a reference; alpha chosen on a held-out tenth of the training images by MRR (as run.py.pick_alpha)
  controls: shuffled-pair floor; the ViT-H/14 image features themselves through the same ridge into every space (no EEG: the
           alignability of the spaces); image-order sanity: ViT-H -> ViT-L ridge must identify the test images almost perfectly
Reading: the retrained NICE-GA swap gives pixels 0.28 < CLIP 0.72 < DINOv2 0.81 bits. If the same frozen EEG embedding, read linearly,
yields the same ordering, the swap is a property of (EEG code, target geometry) and needs no retraining; if DINOv2 < CLIP here,
the retrained DINOv2 arm's extra bits come from the training objective, and the ordering cannot be called geometry.
Output: results/method_pilots/r5_target_linear_map_c.json ; usage: python r5_target_linear_map_c.py [sub-01 ...]"""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from common import score, save, _norm
sys.path.insert(0, "/path/to/eegaudit/experiments/eegbits")
from run import load_torch, EMB, CLIP
import run_acq_nice as N
R = Path("/path/to/eegaudit/results/eegbits"); NICE = N.NICE
SUBJECTS = sys.argv[1:] or [f"sub-{i:02d}" for i in range(1, 11)]
ALPHAS = [1e-1, 1, 10, 100, 1000, 1e4, 1e5, 1e6]; dev = torch.device("cuda"); t0 = time.time()
TR = {"clip_vitL": np.squeeze(np.load(NICE / "dnn_feature/clip_feature_maps_training.npy", allow_pickle=True)).astype(np.float32),
      "dinov2": np.load(R / "dinov2_feature/dinov2_feature_maps_training.npy").astype(np.float32), "pixpca": np.load(R / "pixpca_feature/pixpca_feature_maps_training.npy").astype(np.float32),
      "clip_vitH": load_torch(CLIP / "ViT-H-14_features_train.pt")}
TE = {"clip_vitL": np.squeeze(np.load(NICE / "dnn_feature/clip_feature_maps_test.npy", allow_pickle=True)).astype(np.float32),
      "dinov2": np.load(R / "dinov2_feature/dinov2_feature_maps_test.npy").astype(np.float32), "pixpca": np.load(R / "pixpca_feature/pixpca_feature_maps_test.npy").astype(np.float32),
      "clip_vitH": load_torch(CLIP / "ViT-H-14_features_test.pt")}
TR = {k: _norm(v) for k, v in TR.items()}; TE = {k: _norm(v) for k, v in TE.items()}
print({k: v.shape for k, v in TR.items()}, flush=True)


def ridge_all(Xtr, Xte):
    """Standardise on training rows, ridge to every target; alpha per target on a held-out tenth (MRR). Returns test predictions."""
    rng = np.random.default_rng(0); p = rng.permutation(len(Xtr)); ho, tr = p[:len(Xtr) // 10], p[len(Xtr) // 10:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Z = torch.from_numpy((Xtr - mu) / sd).to(dev).double(); Zte = torch.from_numpy((Xte - mu) / sd).to(dev).double()
    out = {}
    for t, Y in TR.items():
        Yt = torch.from_numpy(Y).to(dev).double()
        def fit(rows, a):
            X_, Y_ = Z[rows], Yt[rows]; xm, ym = X_.mean(0), Y_.mean(0)
            W = torch.linalg.solve((X_ - xm).T @ (X_ - xm) + a * torch.eye(Z.shape[1], device=dev, dtype=torch.float64), (X_ - xm).T @ (Y_ - ym)); return W, ym - xm @ W
        best, bm = None, -1
        for a in ALPHAS:
            W, b = fit(torch.from_numpy(tr).to(dev), a); P = (Z[torch.from_numpy(ho).to(dev)] @ W + b).float().cpu().numpy()
            S = _norm(P) @ Y[ho].T; rank = (S > np.diag(S)[:, None]).sum(1) + 1; m = float((1.0 / rank).mean())
            if m > bm: best, bm = a, m
        W, b = fit(torch.arange(len(Xtr), device=dev), best)
        out[t] = dict(pred=(Zte @ W + b).float().cpu().numpy(), alpha=best, heldout_mrr=bm)
    return out


def line(label, res):
    print(f"  {label:28s} " + " | ".join(f"{t}: {res[t]['bits']:.2f} b top1 {100*res[t]['top1']:.1f}% 2way {100*res[t]['two_way']:.1f}% soft {res[t]['soft_bits']:.2f} a={res[t]['alpha']:g}" for t in TR), flush=True)


out = dict(note=__doc__, alphas=ALPHAS, references={}, subjects={}); rng = np.random.default_rng(0)
ref = ridge_all(TR["clip_vitH"], TE["clip_vitH"])
out["references"]["vitH_features_through_ridge"] = {t: dict(**score(_norm(v["pred"]) @ TE[t].T), alpha=v["alpha"], heldout_mrr=v["heldout_mrr"]) for t, v in ref.items()}
line("ref: ViT-H feats -> ridge", out["references"]["vitH_features_through_ridge"])
for subject in SUBJECTS:
    Xtr = load_torch(EMB / f"ATM_S_eeg_features_{subject}_train.pt").reshape(len(TR["clip_vitL"]), 4, -1).mean(1); Xte = load_torch(EMB / f"ATM_S_eeg_features_{subject}_test.pt")
    r = ridge_all(Xtr, Xte); res = {}
    for t, v in r.items():
        S = _norm(v["pred"]) @ TE[t].T; res[t] = dict(**score(S), alpha=v["alpha"], heldout_mrr=v["heldout_mrr"], shuffled_pair=score(S[:, rng.permutation(200)]))
    res["native_vitH"] = score(_norm(Xte) @ TE["clip_vitH"].T)        # ATM-S's own zero-shot retrieval (its embedding lives in ViT-H space)
    out["subjects"][subject] = res; line(f"{subject} ATM-S emb -> ridge", res); print(f"     native ATM-S vs ViT-H test feats: {res['native_vitH']['bits']:.2f} b top1 {100*res['native_vitH']['top1']:.1f}%  ({time.time()-t0:.0f}s)", flush=True)
    out["runtime_s"] = round(time.time() - t0, 1); save("r5_target_linear_map_c.json", out)
print("\nMEAN OVER SUBJECTS (n=%d): " % len(out["subjects"]) + " | ".join(f"{t}: {np.mean([out['subjects'][s][t]['bits'] for s in out['subjects']]):.3f} b (top1 {100*np.mean([out['subjects'][s][t]['top1'] for s in out['subjects']]):.1f}%, soft {np.mean([out['subjects'][s][t]['soft_bits'] for s in out['subjects']]):.2f})" for t in TR))
print("per-subject ordering (bits) clip_vitL / dinov2 / pixpca:", {s: [round(out["subjects"][s][t]["bits"], 2) for t in ("clip_vitL", "dinov2", "pixpca")] for s in out["subjects"]})
print(f"done ({out['runtime_s']}s)")
