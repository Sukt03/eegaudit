"""KM3 KRONECKER-RESTRICTED CATEGORY ERASURE, image (METHOD "GRID & THRESHOLD ITERATION — ROUND 2 PRE-RULING" KM3; rungs R6 x R5).
EXECUTOR-3, 21 Nov 2026 (shares the LEACE construction with a2_leace.py).
Question: after removing ALL linearly decodable THINGS high-level category information from the EEG, how much exemplar yield is left?
Data: THINGS-EEG2 dev participants sub-01..03 (raw grid 63 ch x 250 samples; training = mean of the 4 repetitions, as the base path);
  target CLIP ViT-H/14; categories = THINGS category27_manual (data_raw/third_party/things_categories; mapped by concept NAME), as a
  multi-hot vector (a concept in several categories carries several 1s; LEACE erases the span of all its indicator columns), plus an
  explicit "other" column for concepts with no category (METHOD (5); coverage printed).
Erasers, ALL fit on TRAINING rows only (test images are used only to evaluate):
  none      identity (= the base raw ridge path; G2: reproduces results/eegbits/raw_sub-XX.json L = 1024, k = 80 exactly)
  full      LEACE (Belrose et al. 2023) on the read-out input (training PCA-1024 scores): x' = x - W^+ P W (x - mu),
            W = Sigma_xx^-1/2, P = projector onto colspace(W Sigma_xz), z = multi-hot category (+ other)
  spatial   A (x) I: LEACE on channel vectors (rows = training image x time sample, 63-d), the same 63 x 63 map at every sample, then PCA-1024
  temporal  I (x) B: LEACE on time courses (rows = training image x channel, 250-d), the same 250 x 250 map on every channel, then PCA-1024
  permprof  CONTROL (METHOD BUG HUNT B3; replaces the random-subspace control): LEACE on category profiles PERMUTED across training concepts
            (5 permutations; same profile frequencies, no relation to the concept's EEG)
The read-out is RE-FIT after every eraser (run.py rule: PCA on training rows, alpha by run.pick_alpha's rule with rng 0 through the L
  sequence, ridge), batched on the GPU (d1_ridge_curves.pca_batch / fit_batch); full + permprof x 5 share one batched fit. Floors: candidates permuted WITHIN exact-profile pools.
Scores on the 200 test images (k = 80): 200-way InfoNCE soft12 (primary estimator); EXEMPLAR yield = within-category identification.
  PRIMARY pools (METHOD G1 FIX KM3) = test images with the EXACT same multi-hot category profile (pool >= 2; coverage printed);
  SECONDARY pools ("partly category-confounded") = the test images sharing >= 1 category with the query (incl. itself), bits = log2 p(true | pool) - log2(1/|pool|),
  temperature LOO over queries on the 12-point grid; shuffled-pair floor (candidates permuted inside pools). Residual category
  decodability (G2/G3 of erasure): ridge one-vs-rest probe of the multi-hot category, trained on 90 % of the TRAINING images and tested on
  the other 10 % (fixed seed), balanced accuracy averaged over categories with >= 5 held-out positives; the G2 uses erasers REFIT on the probe's
  90 % (an eraser fit on all rows pushes held-out decodability below chance by construction); full LEACE must be within 0.02 of 0.5.
Kept share = (erased exemplar yield - floor) / (none - floor), ratio of participant means; claim rule (METHOD): "removing all linearly
  decodable category information keeps X % [CI] of the exemplar yield"; never "exemplar = category + noise" unless the kept CI is below the permuted-profile control.
All numerics in torch on the GPU (bootstraps as index tensors); CPU = loading and the category table.
Usage: python km3_category_erasure.py --subjects 1,2,3 --out DIR | --summarize --out DIR
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/path/to/eegaudit")
sys.path.insert(0, str(ROOT / "experiments/ext")); sys.path.insert(0, str(ROOT / "experiments/eegbits"))
import d1_ridge_curves as D1
import run as R

DEV = "cuda"; NSEED = 5; LS = [2, 5, 10, 25, 50, 100, 250, 1024]
CATD = ROOT / "data_raw/third_party/things_categories"; META = ROOT / "data_raw/things_eeg2_images/images_set/image_metadata.npy"


def categories():
    uid = [m["uniqueID"] for m in csv.DictReader(open(CATD / "concepts-metadata_things.tsv"), delimiter="\t")]
    cat = np.loadtxt(CATD / "category27_manual.tsv", delimiter="\t", skiprows=1); names = open(CATD / "category27_manual.tsv").readline().strip().split("\t")
    im = np.load(META, allow_pickle=True).item(); pos = {u: i for i, u in enumerate(uid)}
    def mh(concepts):
        M = np.stack([cat[pos[c.split("_", 1)[1]]] for c in concepts]); other = (M.sum(1) == 0).astype(float)
        return np.c_[M, other]
    tr = mh(im["train_img_concepts_THINGS"]); te = mh(np.array(im["test_img_concepts_THINGS"]))
    cov = dict(train_images=len(tr), train_no_category=int(tr[:, -1].sum()), train_multi=int((tr[:, :-1].sum(1) > 1).sum()),
               test_images=len(te), test_no_category=int(te[:, -1].sum()), test_multi=int((te[:, :-1].sum(1) > 1).sum()))
    cov["train_concept_ids"] = None
    prof = (te[:, None, :] == te[None, :, :]).all(-1).sum(1); cov["test_exact_profile_pooled"] = int((prof >= 2).sum()); cov["test_exact_profile_pools"] = int(len({tuple(r) for r, k in zip(te, prof) if k >= 2}))
    cid = np.unique(np.array(im["train_img_concepts_THINGS"]), return_inverse=True)[1]; cov.pop("train_concept_ids")
    return tr, te, names + ["other"], cov, cid


def leace(A, Z, seeds=()):
    """A (n, D) GPU float64 training rows, Z (n, k) multi-hot. Returns mu, [full map] + [random same-rank maps], rank (row form: x' = mu + (x - mu) @ M)."""
    mu = A.mean(0); Ac = A - mu; n = len(A); Zc = Z - Z.mean(0)
    Sxx = Ac.T @ Ac / n; Sxz = Ac.T @ Zc / n
    e, U = torch.linalg.eigh(Sxx); keep = e > 1e-10 * e.max(); U, e = U[:, keep], e[keep]
    W = U @ torch.diag(e ** -0.5) @ U.T; Wp = U @ torch.diag(e ** 0.5) @ U.T
    Q, Rr = torch.linalg.qr(W @ Sxz); r = int((Rr.diagonal().abs() > 1e-10 * Rr.diagonal().abs().max()).sum()); Q = Q[:, :r]
    I = torch.eye(A.shape[1], dtype=torch.float64, device=DEV); maps = [I - (Wp @ Q @ Q.T @ W).T]
    for sd in seeds:
        g = torch.Generator(device=DEV); g.manual_seed(int(sd)); Qr = torch.linalg.qr(U @ torch.randn((U.shape[1], r), generator=g, device=DEV, dtype=torch.float64))[0]
        maps.append(I - (Wp @ Qr @ Qr.T @ W).T)
    return mu, maps, r


def exemplar_bits(S, C, exact=False):
    """S (..., 200, 200) cosine; C (200, k) multi-hot test categories. Pools: exact=True (PRIMARY, METHOD G1 FIX KM3) = images with the
    IDENTICAL multi-hot profile; exact=False (SECONDARY, "partly category-confounded") = images sharing >= 1 category. Per-query bits at LOO temperature."""
    share = ((C[:, None, :] == C[None, :, :]).all(-1)) if exact else ((C @ C.T) > 0); has = share.sum(1) >= 2                                     # pool = images sharing >= 1 category; queries need >= 2
    t = torch.as_tensor(R._TAUS, dtype=torch.float64, device=DEV); mask = torch.where(share, 0.0, float("-inf"))
    lp = torch.log_softmax(S[..., None, :, :] / t[:, None, None] + mask, -1)               # (..., T, 200, 200)
    CE = -torch.diagonal(lp, dim1=-2, dim2=-1)[..., has] / np.log(2); m = has.sum()
    loo = (CE.sum(-1, keepdim=True) - CE) / (m - 1); pool = share.sum(1)[has].double()
    bits = torch.log2(pool) - CE.gather(-2, loo.argmin(-2, keepdim=True)).squeeze(-2)
    return bits.mean(-1)


def cat_decodability(Ftr, Ctr, seed=0):
    """Ridge one-vs-rest probe on erased TRAINING features (B, n, d): 90/10 split of training images (fixed seed); mean balanced accuracy."""
    n = Ftr.shape[1]; g = torch.Generator(device=DEV); g.manual_seed(seed); p = torch.randperm(n, generator=g, device=DEV); te, tr = p[: n // 10], p[n // 10:]
    A, B = Ftr[:, tr].double(), Ftr[:, te].double(); mu, sd = A.mean(1, keepdim=True), A.std(1, keepdim=True) + 1e-6; A, B = (A - mu) / sd, (B - mu) / sd
    Y = Ctr[tr]; Yt = Ctr[te]; ok = Yt.sum(0) >= 5
    W = torch.linalg.solve(A.transpose(1, 2) @ A + 1.0 * len(tr) * torch.eye(A.shape[2], dtype=torch.float64, device=DEV), A.transpose(1, 2) @ (Y - Y.mean(0)))
    Ptr, Pte = A @ W, B @ W; thr = 0.5 * ((Ptr * Y).sum(1) / Y.sum(0) + (Ptr * (1 - Y)).sum(1) / (1 - Y).sum(0))
    hit = (Pte > thr[:, None]).double(); tpr = (hit * Yt).sum(1) / Yt.sum(0).clamp_min(1); tnr = ((1 - hit) * (1 - Yt)).sum(1) / (1 - Yt).sum(0).clamp_min(1)
    return (0.5 * (tpr + tnr))[:, ok].mean(1)


def perm_profiles(Ctr, cid, seed):
    """METHOD BUG HUNT B3 control: the category profile of every training CONCEPT replaced by another concept's profile (permutation across
    training concepts), expanded to that concept's images: same profile frequencies and ranks, no relation to the EEG of the concept."""
    nc = cid.max() + 1; prof = np.zeros((nc, Ctr.shape[1])); prof[cid] = Ctr; return prof[np.random.default_rng(seed).permutation(nc)][cid]


def perm_within(Cte, seed):
    """Test-candidate permutation WITHIN exact-profile classes (METHOD: floors permuted within pools)."""
    rng = np.random.default_rng(seed); p = np.arange(len(Cte)); keys = [tuple(r) for r in Cte]
    for k in set(keys):
        ix = np.array([i for i, kk in enumerate(keys) if kk == k]); p[ix] = ix[rng.permutation(len(ix))]
    return p


def subject(s, out, Ctr, Cte, cov, cid):
    t0 = time.time(); sub = f"sub-{s:02d}"
    Ytr, Yte = R.load_torch(R.CLIP / "ViT-H-14_features_train.pt"), R.load_torch(R.CLIP / "ViT-H-14_features_test.pt")
    Yg = torch.as_tensor(Ytr, dtype=torch.float64, device=DEV); Yte_g = torch.as_tensor(Yte, dtype=torch.float64, device=DEV)
    Xtr, Xk80, _, _, ch = D1.load_gpu(sub); n = len(Xtr); print(sub, "loaded", f"{time.time() - t0:.0f}s", flush=True)
    Cg, Ct = torch.as_tensor(Ctr, dtype=torch.float64, device=DEV), torch.as_tensor(Cte, dtype=torch.float64, device=DEV)
    rng = np.random.default_rng(0); perms = {L: rng.permutation(n) for L in LS}
    res = dict(target="image", participant=sub, coverage=cov, unit="bits per query (k = 80)", conditions={})
    def fit_score(Ztr, Zte, names):
        fits = D1.fit_batch(Ztr, [Zte], Yg, LS, perms); P = fits[1024][1][0]; S = D1.gcos(P, Yte_g)            # (B, 200, 200)
        sc = D1.gscore(S); ex = exemplar_bits(S, Ct); exx = exemplar_bits(S, Ct, exact=True)
        shp = torch.stack([torch.as_tensor(perm_within(Cte, 200 + k), device=DEV) for k in range(3)])         # within-pool floor (METHOD)
        exf = torch.stack([exemplar_bits(S[:, :, q], Ct) for q in shp], 1).mean(1); exxf = torch.stack([exemplar_bits(S[:, :, q], Ct, exact=True) for q in shp], 1).mean(1); f200 = torch.stack([D1.gscore(S[:, :, q])["soft12"] for q in shp], 1).mean(1)
        dec = cat_decodability(Ztr, Cg)
        for i, nm in enumerate(names):
            res["conditions"][nm] = dict(alpha=fits[1024][0][i], soft12=float(sc["soft12"][i]), top1=float(sc["top1"][i]), two_way=float(sc["two_way"][i]), mrr=float(sc["mrr"][i]),
                                         floor_soft12=float(f200[i]), exemplar_exact=float(exx[i]), exemplar_exact_floor=float(exxf[i]),
                                         exemplar=float(ex[i]), exemplar_floor=float(exf[i]), category_balanced_acc=float(dec[i]))
            print(sub, nm, {k: round(v, 4) for k, v in res["conditions"][nm].items() if isinstance(v, float)}, f"{time.time() - t0:.0f}s", flush=True)
    F = Xtr.reshape(n, -1); Ztr, (Zte,) = D1.pca_batch(F[None], [Xk80.reshape(200, -1)[None]], 1024)                    # base PCA-1024 (none)
    fit_score(Ztr, Zte, ["none"])
    A = Ztr[0].double(); mu, maps, r = leace(A, Cg); res["rank_full"] = r
    Zp = [torch.as_tensor(perm_profiles(Ctr, cid, 300 + k), dtype=torch.float64, device=DEV) for k in range(NSEED)]
    for k in range(NSEED):
        mk, mp, rk = leace(A, Zp[k]); assert torch.allclose(mk, mu); maps += mp; res.setdefault("rank_permprof", []).append(rk)
    M = torch.stack(maps); Zf = (mu + (A - mu) @ M).float(); Zft = (mu + (Zte[0].double() - mu) @ M).float()           # (6, n, 1024)
    fit_score(Zf, Zft, ["full"] + [f"permprof{k}" for k in range(NSEED)]); del Zf, Zft, M
    # G2 of erasure on HELD-OUT training rows: the eraser must not see the probe's test rows (an eraser fit on all rows makes the held-out
    # cross-covariance the negative of the in-sample one, i.e. below-chance balanced accuracy). Refit LEACE / random on the probe's 90 % only.
    g = torch.Generator(device=DEV); g.manual_seed(0); pp = torch.randperm(n, generator=g, device=DEV); tr90 = pp[n // 10:]
    mu9, maps9, r9 = leace(A[tr90], Cg[tr90])
    for k in range(NSEED): maps9 += leace(A[tr90], Zp[k][tr90])[1]
    dec9 = cat_decodability((mu9 + (A - mu9) @ torch.stack(maps9)).float(), Cg)
    for i, nm in enumerate(["full"] + [f"permprof{k}" for k in range(NSEED)]):
        res["conditions"][nm]["category_balanced_acc_heldout_eraser"] = float(dec9[i])
    res["conditions"]["none"]["category_balanced_acc_heldout_eraser"] = res["conditions"]["none"]["category_balanced_acc"]
    print(sub, "held-out-eraser decodability", {nm: round(res["conditions"][nm]["category_balanced_acc_heldout_eraser"], 3) for nm in ("none", "full", "permprof0")}, flush=True)
    del A, Ztr, Zte; torch.cuda.empty_cache()
    # spatial (A x I): rows = image x time, 63-d channel vectors
    Xs = Xtr.permute(0, 2, 1).reshape(-1, 63).double(); Zs = Cg.repeat_interleave(250, 0)
    mu_s, (Ms,), r_s = leace(Xs, Zs); del Xs, Zs
    Fs = (mu_s + (Xtr.permute(0, 2, 1).double() - mu_s) @ Ms).permute(0, 2, 1).float().reshape(n, -1)
    Fst = (mu_s + (Xk80.permute(0, 2, 1).double() - mu_s) @ Ms).permute(0, 2, 1).float().reshape(200, -1)
    Zs_tr, (Zs_te,) = D1.pca_batch(Fs[None], [Fst[None]], 1024); del Fs, Fst; fit_score(Zs_tr, Zs_te, ["spatial"]); del Zs_tr, Zs_te; torch.cuda.empty_cache()
    # temporal (I x B): rows = image x channel, 250-d time courses
    Xt = Xtr.reshape(-1, 250).double(); Zt_ = Cg.repeat_interleave(63, 0)
    mu_t, (Mt,), r_t = leace(Xt, Zt_); del Xt, Zt_
    Ft = (mu_t + (Xtr.double() - mu_t) @ Mt).float().reshape(n, -1); Ftt = (mu_t + (Xk80.double() - mu_t) @ Mt).float().reshape(200, -1)
    Zt_tr, (Zt_te,) = D1.pca_batch(Ft[None], [Ftt[None]], 1024); del Ft, Ftt; fit_score(Zt_tr, Zt_te, ["temporal"])
    res["rank_spatial"], res["rank_temporal"] = r_s, r_t
    old = json.loads((ROOT / f"results/eegbits/raw_{sub}.json").read_text()); o = next(x for x in old["rows"] if x["L"] == 1024 and x["k"] == 80 and x["condition"] == "eeg")
    nn_ = res["conditions"]["none"]; res["g2_identity"] = dict(old=dict(alpha=o["alpha"], top1=o["top1"], two_way=o["two_way"]), new=dict(alpha=nn_["alpha"], top1=nn_["top1"], two_way=nn_["two_way"]),
                                                             passed=bool(abs(o["top1"] - nn_["top1"]) < 1e-9 and abs(o["two_way"] - nn_["two_way"]) < 1e-3))
    fb = res["conditions"]["full"]["category_balanced_acc_heldout_eraser"]
    res["g2_erasure"] = dict(full_balanced_acc_heldout_eraser=fb, full_balanced_acc_allrows_eraser=res["conditions"]["full"]["category_balanced_acc"],
                             passed=bool(abs(fb - 0.5) <= 0.02), note="all-rows eraser value is biased BELOW chance by construction; the held-out-eraser value is the gate")
    res["runtime_s"] = round(time.time() - t0, 1); (out / f"km3_{sub}.json").write_text(json.dumps(res, indent=1))
    print(sub, "G2 identity", res["g2_identity"]["passed"], "G2 erasure", res["g2_erasure"], "ranks", r, r_s, r_t, f"{res['runtime_s']}s", flush=True)


def summarize(out):
    per = [json.loads(f.read_text()) for f in sorted(out.glob("km3_sub-*.json"))]; n = len(per); res = dict(n=n, participants=[p["participant"] for p in per], coverage=per[0]["coverage"])
    T = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float64, device=DEV); g = torch.Generator(device=DEV); g.manual_seed(20261121)
    IX = torch.randint(0, n, (10000, n), generator=g, device=DEV); q = lambda x: torch.quantile(x, torch.tensor([.025, .975], dtype=torch.float64, device=DEV)).cpu().tolist()
    get = lambda c, k: T([p["conditions"][c][k] for p in per]); CONDS = ["full", "spatial", "temporal"] + [f"permprof{k}" for k in range(NSEED)]
    res["pools"] = dict(primary="exact multi-hot category profile (pool >= 2); floors permuted within pools", secondary="share >= 1 category: partly category-confounded")
    res["control"] = "LEACE on category profiles permuted across training concepts (5 permutations; METHOD BUG HUNT B3)"
    for key, lab in (("exemplar_exact", "primary"), ("exemplar", "secondary_partly_category_confounded")):
        base = get("none", key) - get("none", key + "_floor"); blk = dict(none=dict(exemplar=float(get("none", key).mean()), floor=float(get("none", key + "_floor").mean())))
        for c in CONDS:
            d = get(c, key) - get(c, key + "_floor"); blk[c] = dict(exemplar=float(get(c, key).mean()), kept_share=float(d.mean() / base.mean()), kept_share_ci=q(d[IX].mean(1) / base[IX].mean(1)),
                                                                   soft12=float(get(c, "soft12").mean()), category_balanced_acc_heldout_eraser=float(np.mean([p["conditions"][c].get("category_balanced_acc_heldout_eraser", np.nan) for p in per])))
        pc = torch.stack([get(f"permprof{k}", key) - get(f"permprof{k}", key + "_floor") for k in range(NSEED)]).mean(0); fu = get("full", key) - get("full", key + "_floor")
        blk["full_minus_permprof_bits"] = dict(d=float((fu - pc).mean()), ci=q((fu - pc)[IX].mean(1)))
        blk["full_kept_minus_permprof_kept"] = dict(d=float(fu.mean() / base.mean() - pc.mean() / base.mean()), ci=q(fu[IX].mean(1) / base[IX].mean(1) - pc[IX].mean(1) / base[IX].mean(1)))
        if lab == "primary": res.update(blk)                                                           # METHOD C1: top-level keys = PRIMARY exact pools
        else: res[lab] = blk
    res["none"]["soft12"] = float(get("none", "soft12").mean()); res["none"]["category_balanced_acc"] = float(get("none", "category_balanced_acc").mean())
    res["g2_all_pass"] = bool(all(p["g2_identity"]["passed"] and p["g2_erasure"]["passed"] for p in per))
    (out / "summary_km3.json").write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k != "secondary_partly_category_confounded"}, indent=1)[:4000])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default="1,2,3"); ap.add_argument("--out", required=True); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize: return summarize(out)
    Ctr, Cte, names, cov, cid = categories(); print("coverage", cov, flush=True)
    import gc
    failed = []
    for s in [int(x) for x in a.subjects.split(",")]:
        if (out / f"km3_sub-{s:02d}.json").exists(): continue
        # FIXER (R2 ruling 6): free GPU memory between participants and catch an OOM per participant (3 tries, 3 min apart),
        # so one OOM cannot end the remaining participants. The estimator is unchanged.
        for attempt in range(3):
            oom = False
            try: subject(s, out, Ctr, Cte, cov, cid)
            except torch.OutOfMemoryError as e: oom = True; print(f"sub-{s:02d} OOM (try {attempt + 1}/3): {str(e)[:200]}", flush=True)
            gc.collect(); torch.cuda.empty_cache()
            if not oom: break
            time.sleep(180)
        else: failed.append(s)
    if failed: print("KM3 FAILED (OOM x3):", failed, flush=True); sys.exit(1)


if __name__ == "__main__":
    main()
