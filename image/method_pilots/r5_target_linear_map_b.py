"""Round 5, attack 1 (part b): the leave-one-out ridge on the 200 test images (r5_target_linear_map.py) is uninformative: the 200
test images are nearly orthogonal in DINOv2 / pixel-PCA space (mean off-diagonal cosine 0.013 / 0.007), so no map fitted on 199 of
them can predict the 200th; even the noise-free CLIP test features give 0.00 bits in DINOv2 space through a 199-image LOO ridge.
Here the linear change of target space is fitted WITHOUT EEG on the 16,540 TRAINING images (ridge, intercept, standardised source,
alpha on a held-out tenth of the training images by top-1 among those images) and applied to the SAME test EEG embedding:
  maps     : clip->dinov2, clip->pixpca, dinov2->clip, dinov2->pixpca  (image features only)
  no-EEG   : CLIP test features through each map (alignability ceiling); CLIP test features + isotropic noise at the geometry
             control's matched sigma through each map (5 draws); the same for DINOv2 test features
  EEG      : CLIP-trained NICE-GA embedding (5 seeds) scored natively vs CLIP test features and, through the maps, vs DINOv2 and
             pixel-PCA test features; the DINOv2-trained embedding (seed 141) natively vs DINOv2 and through the maps vs CLIP / pixels
Reading: if the CLIP encoder's embedding, read linearly into DINOv2 space, keeps or gains bits relative to its native CLIP yield,
the retrained ordering (CLIP 0.72 < DINOv2 0.81) is target geometry; if it loses bits, the DINOv2 arm's extra bits come from training
on DINOv2 targets and the swap is a statement about the training objective, not about geometry alone.
Output: results/method_pilots/r5_target_linear_map_b.json ; usage: python r5_target_linear_map_b.py [sub-01 ...]"""
import sys, time, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from common import load_encoder, load_test, score, save, _norm
import run_acq_nice as N
R = Path("/path/to/eegaudit/results/eegbits"); NICE = N.NICE; VAR = R / "nice_ga_variants"
SUBJECTS = sys.argv[1:] or ["sub-01", "sub-02", "sub-03", "sub-05", "sub-06", "sub-07", "sub-08", "sub-09", "sub-10"]
SEEDS = [141, 297, 777, 1024, 2022]; ALPHAS = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
dev = torch.device("cuda"); t0 = time.time()
TR = {"clip": np.squeeze(np.load(NICE / "dnn_feature/clip_feature_maps_training.npy", allow_pickle=True)).astype(np.float32),
      "dinov2": np.load(R / "dinov2_feature/dinov2_feature_maps_training.npy").astype(np.float32), "pixpca": np.load(R / "pixpca_feature/pixpca_feature_maps_training.npy").astype(np.float32)}
TE = {"clip": np.squeeze(np.load(NICE / "dnn_feature/clip_feature_maps_test.npy", allow_pickle=True)).astype(np.float32),
      "dinov2": np.load(R / "dinov2_feature/dinov2_feature_maps_test.npy").astype(np.float32), "pixpca": np.load(R / "pixpca_feature/pixpca_feature_maps_test.npy").astype(np.float32)}
TR = {k: _norm(v) for k, v in TR.items()}; TE = {k: _norm(v) for k, v in TE.items()}
SIG = float(json.load(open(R / "target_geometry_control.json"))["matched"]["sigma"])


class RidgeMap:
    """y = W z + b with z = standardised x; alpha by top-1 identification among a held-out tenth of the training images."""
    def __init__(self, X, Y):
        rng = np.random.default_rng(0); p = rng.permutation(len(X)); ho, tr = p[:len(X) // 10], p[len(X) // 10:]
        self.mu, self.sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Z = (X - self.mu) / self.sd
        Xt = torch.from_numpy(Z[tr]).to(dev).double(); Yt = torch.from_numpy(Y[tr]).to(dev).double()
        xm, ym = Xt.mean(0), Yt.mean(0); G = (Xt - xm).T @ (Xt - xm); C = (Xt - xm).T @ (Yt - ym)
        self.fits = {}; best = None
        for a in ALPHAS:
            W = torch.linalg.solve(G + a * torch.eye(G.shape[0], device=dev, dtype=torch.float64), C); b = ym - xm @ W
            P = (torch.from_numpy(Z[ho]).to(dev).double() @ W + b).float().cpu().numpy()
            m = score(_norm(P) @ Y[ho].T); self.fits[a] = (W.float().cpu().numpy(), b.float().cpu().numpy(), m["top1"], m["two_way"])
            if best is None or m["top1"] > self.fits[best][2]: best = a
        self.alpha = best; self.W, self.b = self.fits[best][0], self.fits[best][1]
        self.heldout = {str(a): dict(top1=v[2], two_way=v[3]) for a, v in self.fits.items()}

    def __call__(self, X):
        return ((X - self.mu) / self.sd) @ self.W + self.b


maps = {}
for s, t in [("clip", "dinov2"), ("clip", "pixpca"), ("dinov2", "clip"), ("dinov2", "pixpca")]:
    maps[(s, t)] = RidgeMap(TR[s], TR[t]); print(f"map {s}->{t}: alpha {maps[(s, t)].alpha:g}, held-out training top-1 {100*maps[(s,t)].fits[maps[(s,t)].alpha][2]:.1f}% (n=1654)", flush=True)


def through(E, src):
    """Score an embedding living in `src` space natively and through every map out of `src`."""
    res = {src: score(_norm(E) @ TE[src].T)}
    for (s, t), M in maps.items():
        if s == src: res[t] = score(_norm(M(E)) @ TE[t].T)
    return res


def line(label, res):
    print(f"  {label:36s} " + " | ".join(f"{t}: {res[t]['bits']:.2f} b top1 {100*res[t]['top1']:.1f}% 2way {100*res[t]['two_way']:.1f}% soft {res[t]['soft_bits']:.2f}" for t in ("clip", "dinov2", "pixpca") if t in res), flush=True)


out = dict(note=__doc__, matched_sigma=SIG, maps={f"{s}->{t}": dict(alpha=M.alpha, heldout_training=M.heldout) for (s, t), M in maps.items()}, references={}, subjects={})
rng = np.random.default_rng(0)
out["references"]["clip_testfeat"] = through(TE["clip"], "clip"); line("ref: CLIP test features", out["references"]["clip_testfeat"])
out["references"]["dinov2_testfeat"] = through(TE["dinov2"], "dinov2"); line("ref: DINOv2 test features", out["references"]["dinov2_testfeat"])
for src in ("clip", "dinov2"):
    dr = [through(TE[src] + SIG * rng.standard_normal((200, 768)).astype(np.float32) / np.sqrt(768), src) for _ in range(5)]
    out["references"][f"{src}_plus_isonoise"] = {t: {k: float(np.mean([d[t][k] for d in dr])) for k in ("bits", "top1", "two_way", "soft_bits")} for t in dr[0]}
    line(f"ref: {src} + iso noise sigma={SIG:g} (5)", out["references"][f"{src}_plus_isonoise"])
Enc, Proj = N.load_classes("nice_ga")
for subject in SUBJECTS:
    print(subject, f"({time.time()-t0:.0f}s)", flush=True); sub = {}
    X = load_test(subject); xm = np.ascontiguousarray(X[:, :80].mean(1))
    for seed in SEEDS:
        embed, Yc, _, _ = load_encoder("nice_ga", subject, seed, dev); E = embed(xm)
        sub[f"clip_enc_seed{seed}"] = through(E, "clip"); sub[f"clip_enc_seed{seed}"]["clip_centres"] = score(_norm(E) @ _norm(Yc).T)
        line(f"CLIP-trained encoder seed {seed}", sub[f"clip_enc_seed{seed}"])
    tag = f"nice_ga_dinov2_sub{int(subject[-2:]):02d}_seed141"
    if (VAR / f"{tag}_Enc_eeg_cls.pth").exists():
        enc, proj = Enc().to(dev).eval(), Proj().to(dev).eval()
        enc.load_state_dict(torch.load(VAR / f"{tag}_Enc_eeg_cls.pth", map_location=dev), strict=True); proj.load_state_dict(torch.load(VAR / f"{tag}_Proj_eeg_cls.pth", map_location=dev), strict=True)
        with torch.no_grad():
            E = proj(enc(torch.from_numpy(xm).unsqueeze(1).to(dev))).float().cpu().numpy()
        sub["dinov2_enc_seed141"] = through(E, "dinov2"); line("DINOv2-trained encoder seed 141", sub["dinov2_enc_seed141"])
    out["subjects"][subject] = sub; out["runtime_s"] = round(time.time() - t0, 1); save("r5_target_linear_map_b.json", out)
print("\nMEAN OVER SUBJECTS (bits; top-1 %):")
for src in [f"clip_enc_seed{s}" for s in SEEDS] + ["dinov2_enc_seed141"]:
    S = [out["subjects"][s][src] for s in out["subjects"] if src in out["subjects"][s]]
    print(f"  {src:20s} n={len(S)} " + " | ".join(f"{t}: {np.mean([d[t]['bits'] for d in S]):.3f} ({100*np.mean([d[t]['top1'] for d in S]):.1f}%)" for t in S[0] if t != "clip_centres"))
# seed-pooled CLIP encoder: mean over 5 seeds then subjects
P = {t: np.mean([[out["subjects"][s][f"clip_enc_seed{sd}"][t]["bits"] for sd in SEEDS] for s in out["subjects"]]) for t in ("clip", "dinov2", "pixpca")}
print("  CLIP encoder, 5-seed x subject mean:", {t: round(float(v), 3) for t, v in P.items()}); print(f"done ({out['runtime_s']}s)")
