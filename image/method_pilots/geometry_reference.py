"""Round B of the method image loop (2026-11-20): the geometry reference for the target swap, done with real embeddings.
For one subject and each seed, take the TRAINED encoders
  clip   : released NICE-GA (CLIP target, NICE/model) ; dinov2 / pixpca : the retrained NICE-GA of that arm (results/eegbits/nice_ga_variants)
embed the TRAINING EEG (16540 images, the 4 repetitions averaged exactly as NICE trains) and fit a ridge map from that embedding into each
target space T in {clip, dinov2, pixpca} (unit-normalised rows on both sides, intercept, ridge strength by 5-fold CV mean-squared error on
the training set, never by identification), then apply it to the 200 TEST embeddings (k = 80 mean; k = 1, 5 draws) and identify each test
image among the 200 test-image features of T. The clip -> clip cell is the encoder's own space re-read linearly (should track the direct
yield); encoder_clip -> dinov2 says how much of the DINOv2-space yield the CLIP-trained embedding already carries, i.e. how much of the
target-swap difference is the target's geometry rather than new EEG information. No-EEG row: the CLIP image features of the training images
mapped the same way into each space and applied to the CLIP features of the test images (pure image geometry). Shuffled rows (test
embeddings permuted against the images) must be at chance.
First version (kept as results/method_pilots/geometry_reference_LOO-smoke_sub-01_seed141.json) fitted a leave-one-out ridge on the 200 test
embeddings only: at chance even for encoder_clip -> clip (2.0 % top-1 vs 16.5 % direct), because a held-out prediction is a combination of
the other 199 targets and is pushed away from the true one; abandoned.
Output: results/method_pilots/geometry_reference_<subject>_seed<seed>.json (one per seed). Usage: python geometry_reference.py sub-01 141,297,777"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/path/to/eegaudit/experiments/eegbits")
from run import metrics
import run_acq_controls as C
import run_acq_nice as N
R = Path("/path/to/eegaudit/results"); OUT = R / "method_pilots"; VAR = R / "eegbits/nice_ga_variants"
subject, seeds = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); t0 = time.time(); sub = int(subject[-2:])
NICEF = N.NICE / "dnn_feature"
test_feat = {"clip": np.squeeze(np.load(NICEF / "clip_feature_maps_test.npy", allow_pickle=True)).astype(np.float32),
             "dinov2": np.load(R / "eegbits/dinov2_feature/dinov2_feature_maps_test.npy").astype(np.float32),
             "pixpca": np.load(R / "eegbits/pixpca_feature/pixpca_feature_maps_test.npy").astype(np.float32)}
train_feat = {"clip": np.squeeze(np.load(NICEF / "clip_feature_maps_training.npy", allow_pickle=True)).astype(np.float32),
              "dinov2": np.load(R / "eegbits/dinov2_feature/dinov2_feature_maps_training.npy").astype(np.float32),
              "pixpca": np.load(R / "eegbits/pixpca_feature/pixpca_feature_maps_training.npy").astype(np.float32)}
for k in test_feat: assert test_feat[k].shape == (200, 768) and train_feat[k].shape == (16540, 768), k


def load_train_mean(subject):
    """Training EEG averaged over the 4 repetitions, float32 (16540, 63, 250); cached next to the test cache."""
    c = C.CACHE / f"{subject}_train_mean4.npy"
    if not c.exists():
        d = np.load(C.RAW / subject / "preprocessed_eeg_training.npy", allow_pickle=True); d = d.item() if isinstance(d, np.ndarray) else d
        a = d["preprocessed_eeg_data"]; assert a.shape[:2] == (16540, 4), a.shape
        np.save(c, a.mean(1).astype(np.float32)); del d, a
    return np.load(c, mmap_mode="r")


Xte = C.load_test(subject); Xtr = load_train_mean(subject); print("train mean loaded", Xtr.shape, f"{time.time()-t0:.0f}s", flush=True)
Enc_eeg, Proj_eeg = N.load_classes("nice_ga")


def encoder(arm, seed):
    enc, proj = Enc_eeg().to(dev).eval(), Proj_eeg().to(dev).eval()
    base = (N.NICE / "model" / f"nice_ga_dgx_v2_sub{sub:02d}_seed{seed}") if arm == "clip" else (VAR / f"nice_ga_{arm}_sub{sub:02d}_seed{seed}")
    enc.load_state_dict(torch.load(str(base) + "_Enc_eeg_cls.pth", map_location=dev), strict=True)
    proj.load_state_dict(torch.load(str(base) + "_Proj_eeg_cls.pth", map_location=dev), strict=True)
    @torch.no_grad()
    def embed(x, bs=1024):
        return np.concatenate([proj(enc(torch.from_numpy(np.ascontiguousarray(x[i:i + bs])).unsqueeze(1).to(dev))).float().cpu().numpy() for i in range(0, len(x), bs)])
    return embed


def unit(a): return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)


def ridge_fit(Ztr, Ytr, lambdas=np.logspace(-4, 2, 13), folds=5):
    """Ridge (with intercept) from unit rows Ztr to unit rows Ytr; lambda (scaled by trace(Z'Z)/p) by 5-fold CV MSE. Returns (W, b, lam, cv)."""
    Z = torch.from_numpy(unit(Ztr)).double().to(dev); Y = torch.from_numpy(unit(Ytr)).double().to(dev); n, p = Z.shape
    idx = torch.from_numpy(np.random.default_rng(0).permutation(n)).to(dev); fold = torch.zeros(n, dtype=torch.long, device=dev); fold[idx] = torch.arange(n, device=dev) % folds
    def solve(Zs, Ys, lam):
        mz, my = Zs.mean(0, keepdim=True), Ys.mean(0, keepdim=True); Zc, Yc = Zs - mz, Ys - my; G = Zc.T @ Zc; sc = torch.trace(G) / p
        W = torch.linalg.solve(G + lam * sc * torch.eye(p, device=dev, dtype=G.dtype), Zc.T @ Yc); return W, my - mz @ W
    cv = []
    for lam in lambdas:
        err = 0.0
        for f in range(folds):
            tr, va = fold != f, fold == f; W, b = solve(Z[tr], Y[tr], lam); err += float(((Z[va] @ W + b - Y[va]) ** 2).sum(1).mean()) / folds
        cv.append(err)
    lam = float(lambdas[int(np.argmin(cv))]); W, b = solve(Z, Y, lam)
    return W, b, lam, cv


def apply(W, b, Zte):
    return (torch.from_numpy(unit(Zte)).double().to(dev) @ W + b).float().cpu().numpy()


rows = []
# no-EEG geometry rows: CLIP image features of the training images -> each space, applied to the CLIP features of the test images
for T in test_feat:
    W, b, lam, cv = ridge_fit(train_feat["clip"], train_feat[T]); m = metrics(apply(W, b, test_feat["clip"]), test_feat[T])
    rows.append(dict(seed=None, source="image_clip_features", target=T, k=None, draw=0, ridge_lambda=lam, cv_mse=float(min(cv)), **m))
    print(f"image CLIP features -> {T}: top-1 {100*m['top1']:.1f}% bits {m['bits']:.2f} (lambda {lam:g})", flush=True)
for seed in seeds:
    rng = np.random.default_rng(seed % 1000); seed_rows = [dict(r) for r in rows]
    for arm in ("clip", "dinov2", "pixpca"):
        try:
            embed = encoder(arm, seed)
        except FileNotFoundError as e:
            print("missing checkpoint", arm, seed, e, flush=True); continue
        Ztr = embed(Xtr); Z80 = embed(Xte.mean(1)); Z1 = [embed(Xte[:, rng.choice(80, 1)].mean(1)) for _ in range(5)]; perm = rng.permutation(200)
        m = metrics(Z80, test_feat[arm]); seed_rows.append(dict(seed=seed, source=f"encoder_{arm}_direct", target=arm, k=80, draw=0, ridge_lambda=None, cv_mse=None, **m))
        for T in test_feat:
            W, b, lam, cv = ridge_fit(Ztr, train_feat[T])
            m = metrics(apply(W, b, Z80), test_feat[T]); seed_rows.append(dict(seed=seed, source=f"encoder_{arm}", target=T, k=80, draw=0, ridge_lambda=lam, cv_mse=float(min(cv)), **m))
            print(f"seed {seed} encoder {arm} -> {T}: top-1 {100*m['top1']:.1f}% bits {m['bits']:.2f} (lambda {lam:g}, {time.time()-t0:.0f}s)", flush=True)
            for d, z in enumerate(Z1):
                seed_rows.append(dict(seed=seed, source=f"encoder_{arm}", target=T, k=1, draw=d, ridge_lambda=lam, cv_mse=None, **metrics(apply(W, b, z), test_feat[T])))
            seed_rows.append(dict(seed=seed, source=f"encoder_{arm}_shuffled", target=T, k=80, draw=0, ridge_lambda=lam, cv_mse=None, **metrics(apply(W, b, Z80[perm]), test_feat[T])))
        del Ztr
    out = dict(subject=subject, seed=seed, rows=seed_rows, runtime_s=round(time.time() - t0, 1),
               note="ridge (intercept) from unit-normalised training-EEG embeddings (4 reps averaged) to unit-normalised training target features; lambda by 5-fold CV MSE over logspace(-4,2,13) x trace(Z'Z)/p; applied to the 200 test embeddings")
    (OUT / f"geometry_reference_{subject}_seed{seed}.json").write_text(json.dumps(out, indent=1))
    print("GEOMREF", subject, seed, f"({out['runtime_s']}s)", flush=True)
