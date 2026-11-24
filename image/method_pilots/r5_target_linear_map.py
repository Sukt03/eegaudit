"""Round 5, attack 1: a decoder-shaped no-EEG reference for the target swap.
The published geometry control adds ISOTROPIC noise to each target embedding; a decoder's errors are not isotropic, so that
control cannot say whether pixels < CLIP < DINOv2 (0.28 / 0.72 / 0.81 bits, retrained encoders) is a property of the EEG
read-out or of how the 200 test images sit in each space. Here ONE EEG embedding is mapped linearly to EVERY target space:
  source embeddings (200 test images, 80-rep mean):  trained NICE-GA CLIP encoder (5 seeds), retrained NICE-GA DINOv2 encoder (seed 141)
  no-EEG references:                                 the CLIP test features themselves; CLIP test features + isotropic noise at the
                                                     geometry control's matched sigma (5 draws)
  map:  ridge (intercept, standardised source) fitted on 199 images, scored on the held-out one; exact leave-one-out via the hat
        matrix of the linear smoother, alpha grid reported in full, "selected" alpha = best LOO mean cosine (label-free w.r.t. top-1)
  targets: CLIP test features, DINOv2 test features, pixel-PCA test features (all L2-normalised, 200 x 768)
Scored 200-way (top-1, 2-way, Wolpaw bits, InfoNCE soft bits, 40-tau grid) against each target space.
If a single EEG embedding gives the same ordering through a linear map, the target swap needs no retraining and the ordering is a
property of (EEG code, target geometry); if it does not, the retrained ordering depends on the training objective.
Output: results/method_pilots/r5_target_linear_map.json ; usage: python r5_target_linear_map.py [sub-01 sub-02 ...]"""
import sys, time, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from common import load_encoder, load_test, score, save, _norm
import run_acq_nice as N
R = Path("/path/to/eegaudit/results/eegbits"); NICE = N.NICE
VAR = R / "nice_ga_variants"
SUBJECTS = sys.argv[1:] or ["sub-01", "sub-02", "sub-03", "sub-05", "sub-06", "sub-07", "sub-08", "sub-09", "sub-10"]
SEEDS = [141, 297, 777, 1024, 2022]
ALPHAS = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
dev = torch.device("cuda"); t0 = time.time()
T = {"clip": np.squeeze(np.load(NICE / "dnn_feature/clip_feature_maps_test.npy", allow_pickle=True)).astype(np.float32),
     "dinov2": np.load(R / "dinov2_feature/dinov2_feature_maps_test.npy").astype(np.float32),
     "pixpca": np.load(R / "pixpca_feature/pixpca_feature_maps_test.npy").astype(np.float32)}
T = {k: _norm(v) for k, v in T.items()}
geom = json.load(open(R / "target_geometry_control.json")); SIG = float(geom["matched"]["sigma"])


def loo_ridge(E, Y):
    """Exact leave-one-out ridge predictions (n x q) for every alpha: linear smoother H = Z (Z'Z + aI)^-1 Z' = K (K + aI)^-1,
    y_loo_i = (H y - h_ii y_i) / (1 - h_ii). Z = standardised E plus an (almost) unpenalised intercept column."""
    n = len(E); Z = (E - E.mean(0)) / (E.std(0) + 1e-6); Z = np.hstack([Z, np.full((n, 1), 1e3, np.float32)])
    K = (Z @ Z.T).astype(np.float64); out = {}
    for a in ALPHAS:
        H = K @ np.linalg.inv(K + a * np.eye(n)); h = np.diag(H)
        P = H @ Y; out[a] = ((P - h[:, None] * Y) / (1 - h)[:, None]).astype(np.float32)
    return out


def map_and_score(E, label):
    res = {}
    for tname, Y in T.items():
        preds = loo_ridge(E, Y); per_alpha = {}
        for a, P in preds.items():
            S = _norm(P) @ Y.T; m = score(S); m["mean_cos"] = float(np.diag(S).mean()); per_alpha[str(a)] = m
        best = max(per_alpha, key=lambda a: per_alpha[a]["mean_cos"])
        res[tname] = dict(per_alpha=per_alpha, selected_alpha=float(best), **per_alpha[best])
    print(f"  {label:38s} " + " | ".join(f"{t}: {res[t]['bits']:.2f} b top1 {100*res[t]['top1']:.1f}% soft {res[t]['soft_bits']:.2f} a={res[t]['selected_alpha']:g}" for t in T), flush=True)
    return res


out = dict(note=__doc__, matched_sigma=SIG, alphas=ALPHAS, subjects={}, references={})
# ---- no-EEG references (subject-independent)
rng = np.random.default_rng(0)
out["references"]["clip_testfeat_itself"] = map_and_score(T["clip"], "ref: CLIP test features -> ridge")
noisy = []
for d in range(5):
    Eo = T["clip"] + SIG * rng.standard_normal(T["clip"].shape).astype(np.float32) / np.sqrt(768)
    r = map_and_score(Eo, f"ref: CLIP + iso noise sigma={SIG:g} draw {d}")
    r["native_vs_clip"] = score(_norm(Eo) @ T["clip"].T); noisy.append(r)
out["references"]["clip_plus_isonoise_draws"] = noisy
out["references"]["clip_plus_isonoise_mean"] = {t: {k: float(np.mean([d[t][k] for d in noisy])) for k in ("bits", "top1", "soft_bits", "two_way")} for t in T}
out["references"]["clip_plus_isonoise_native_bits"] = float(np.mean([d["native_vs_clip"]["bits"] for d in noisy]))

Enc, Proj = N.load_classes("nice_ga")
for subject in SUBJECTS:
    print(subject, f"({time.time()-t0:.0f}s)", flush=True); sub = {}
    X = load_test(subject); xm = np.ascontiguousarray(X[:, :80].mean(1))                 # (200, 63, 250) 80-rep mean
    for seed in SEEDS:
        embed, Yc, _, _ = load_encoder("nice_ga", subject, seed, dev); E = embed(xm)
        nat = {"vs_centres": score(_norm(E) @ _norm(Yc).T), "vs_testfeat": score(_norm(E) @ T["clip"].T)}
        sub[f"clip_enc_seed{seed}"] = dict(native=nat, mapped=map_and_score(E, f"CLIP-trained encoder seed {seed}"))
        print(f"     native vs test feats: {nat['vs_testfeat']['bits']:.2f} b (vs centres {nat['vs_centres']['bits']:.2f})", flush=True)
    tag = f"nice_ga_dinov2_sub{int(subject[-2:]):02d}_seed141"
    if (VAR / f"{tag}_Enc_eeg_cls.pth").exists():
        enc, proj = Enc().to(dev).eval(), Proj().to(dev).eval()
        enc.load_state_dict(torch.load(VAR / f"{tag}_Enc_eeg_cls.pth", map_location=dev), strict=True)
        proj.load_state_dict(torch.load(VAR / f"{tag}_Proj_eeg_cls.pth", map_location=dev), strict=True)
        with torch.no_grad():
            E = proj(enc(torch.from_numpy(xm).unsqueeze(1).to(dev))).float().cpu().numpy()
        nat = {"vs_testfeat": score(_norm(E) @ T["dinov2"].T)}
        sub["dinov2_enc_seed141"] = dict(native=nat, mapped=map_and_score(E, "DINOv2-trained encoder seed 141"))
        print(f"     native vs DINOv2 test feats: {nat['vs_testfeat']['bits']:.2f} b", flush=True)
    out["subjects"][subject] = sub
    out["runtime_s"] = round(time.time() - t0, 1); save("r5_target_linear_map.json", out)
# ---- summary table
rows = []
for src in [f"clip_enc_seed{s}" for s in SEEDS] + ["dinov2_enc_seed141"]:
    vals = {t: [out["subjects"][s][src]["mapped"][t]["bits"] for s in out["subjects"] if src in out["subjects"][s]] for t in T}
    if vals["clip"]:
        rows.append((src, {t: (round(float(np.mean(v)), 3), len(v)) for t, v in vals.items()}))
print("\nMEAN LOO-RIDGE BITS BY SOURCE EMBEDDING AND TARGET SPACE (n subjects):"); [print(f"  {r[0]:22s}", r[1]) for r in rows]
print("  no-EEG CLIP itself       ", {t: round(out["references"]["clip_testfeat_itself"][t]["bits"], 3) for t in T})
print("  no-EEG CLIP + iso noise  ", {t: round(out["references"]["clip_plus_isonoise_mean"][t]["bits"], 3) for t in T}, "native", round(out["references"]["clip_plus_isonoise_native_bits"], 3))
print(f"done ({out['runtime_s']}s)")
