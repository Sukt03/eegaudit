"""N-B. CATEGORY ERASURE THROUGH FROZEN TRAINED DECODERS (image; rung R6 x R7; extends KM3 beyond the linear ridge). FIXER-2, 21 Nov 2026.
Spec: EXTENSIONS_LOOP_20261121 "N-B" + the bug pre-review (4 issues), all four built in:
  (1) back-projection WITH the standardisation and the OBLIQUE LEACE map: the eraser is fit as KM3 does (training rows = mean of the 4
      training repetitions, raw 63 x 250 grid, standardised with training mean / sd (ddof 0, + 1e-6), PCA-1024 = top eigenvectors of the
      float64 Gram, km3_category_erasure.leace on the PCA scores with the multi-hot category27 (+ other) profile). Raw-space eraser:
        z = ((x - mu) / sd) P ;  x' = x + sd * [ ((z - mu_z) (M - I)) P^T ]
      i.e. the PCA coordinates become mu_z + (z - mu_z) M exactly (KM3's row form) and the residual outside span(P) is unchanged.
  (2) UBP lives in its own input space (released whitened 250 Hz data, 17 posterior channels): a SEPARATE eraser is fit in that space on
      UBP's own training data (train.pt, mean over repetitions, 17 x 250 = 4250-d, same standardise -> PCA-1024 -> LEACE), with the
      category profiles of its training images (concept order asserted against the THINGS metadata).
  (3) Wording: "after removing the linearly decodable category subspace of the top-1024 PCs" (information in the other dimensions and
      non-linearly decodable category information remain; the G3 probe is linear).
  (4) The eraser is affine, so it is applied AFTER averaging the 80 test repetitions (k = 80; identical to erasing each repetition).
Decoders (frozen, never refit): NICE-GA seed 141 and BrainFLORA (released) via s1_averaging.models on the THINGS-EEG2 test cache
  (200 x 80 x 63 x 250; candidates = their own 200 test targets); UBP (released intra-subject checkpoint, RN50 'medium', as w1_trialswap).
Erasers: none (identity), full (category LEACE), permprof0-4 (LEACE on category profiles permuted across training concepts; KM3 B3 control).
Scores per decoder x eraser (k = 80): 200-way InfoNCE soft12 (d1_ridge_curves.gscore) and the EXACT-profile exemplar yield
  (km3_category_erasure.exemplar_bits, exact=True) with matched shuffled-pair floors: global permutations for 200-way retrieval and within-profile permutations for exact-profile retrieval (3 seeds each).
Kept share = (erased - floor) / (none - floor); contrast full - mean(permprof), participant bootstrap (summary).
Falsifier (pre-stated): the trained decoders' exemplar kept share under full erasure overlaps the permuted-profile control (CI) ->
  "trained decoders use exemplar information outside the linearly decodable category subspace of the top-1024 PCs".
G2: identity eraser = un-erased input (max |x' - x| printed, must be 0); the un-erased k80 score is printed beside the registered value
  (NICE-GA: s1_v4 r1 = 2.103 with its known 0.11-bit gap, BrainFLORA: s1_v4 bf_registered within 1e-3; UBP: ubp_card_full k80 top-1
  exactly). G3: category decodability (KM3 cat_decodability, balanced accuracy) of held-out TRAINING rows after the RAW-SPACE round trip
  of an eraser fit on the other 90 % must be within 0.02 of 0.5 (per input space).
V3 NOTE (22 Nov): copy of km3_decoders.py as of 22 Nov 16:16 (global-permutation 200-way floor; the within-profile permutation is kept only as the separate field within_profile_shuffle_soft12). Root cause of the failed smoke (sha 39d91dd2...): its floor_soft12 was the WITHIN-PROFILE permutation, which keeps the truth in the same category pool.
Output: results/ext/km3_decoders_v3/{sub-XX.json, summary.json}, code_sha256 in each.
Usage: python km3_decoders.py --subjects 1 --out DIR [--no_ubp] | --summarize --out DIR
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as Fnn

ROOT = Path("/path/to/eegaudit"); sys.path.insert(0, str(ROOT / "experiments/ext")); sys.path.insert(0, str(ROOT / "experiments/baselines")); sys.path.insert(0, str(ROOT / "experiments/eegbits"))
CODE_SHA256 = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
import km3_category_erasure as K
import d1_ridge_curves as D1
import s1_averaging as S1
import w1_trialswap as W1
from safe_load import safe_load
DEV = "cuda"; NSEED = 5; L = 1024; UD = ROOT / "data_raw/third_party/ubp"


def fit_eraser(Xtr, Cg, Zp, idx=None):
    """Xtr (n, D) float32 GPU training rows; returns dict(mu, sd, P, mu_z, maps (1 + NSEED, L, L), rank). idx: rows to fit on (G3)."""
    X = Xtr if idx is None else Xtr[idx]; mu = X.mean(0, keepdim=True); sd = X.std(0, keepdim=True, unbiased=False) + 1e-6
    Z = (X - mu) / sd; C = Z.double().T @ Z.double(); _, V = torch.linalg.eigh(C); P = V.flip(-1)[:, :L].contiguous(); del C, V     # = D1.pca_batch basis (float64)
    A = Z.double() @ P; del Z
    C_ = Cg if idx is None else Cg[idx]; mu_z, maps, r = K.leace(A, C_)
    for k in range(NSEED):
        mk, mp, _ = K.leace(A, Zp[k] if idx is None else Zp[k][idx]); assert torch.allclose(mk, mu_z); maps += mp
    I = torch.eye(L, dtype=torch.float64, device=DEV)
    return dict(mu=mu.double(), sd=sd.double(), P=P, mu_z=mu_z, maps=torch.stack([I] + maps), rank=r, A=A)


def erase(E, x):
    """x (m, D) raw -> (B, m, D) erased by every map (index 0 = identity)."""
    x = x.double(); z = ((x - E["mu"]) / E["sd"]) @ E["P"]; dz = (z - E["mu_z"])[None] @ (E["maps"] - torch.eye(L, dtype=torch.float64, device=DEV)[None])
    return x[None] + E["sd"][None] * (dz @ E["P"].T)


def g3(Xtr, Cg, Zp):
    """Eraser (full) fit on 90 % of the training rows, applied in RAW space to all rows, re-projected on the 90 % basis -> held-out balanced acc."""
    n = len(Xtr); g = torch.Generator(device=DEV); g.manual_seed(0); pp = torch.randperm(n, generator=g, device=DEV); tr90 = pp[n // 10:]
    E = fit_eraser(Xtr, Cg, Zp, idx=tr90); E["maps"] = E["maps"][:2]; Zs = []
    for i in range(0, n, 2000):
        xe = erase(E, Xtr[i:i + 2000]); Zs.append((((xe - E["mu"]) / E["sd"]) @ E["P"]).float())
    Zs = torch.cat(Zs, 1); acc = K.cat_decodability(Zs, Cg)                    # probe's own 90/10 split (seed 0) = the eraser's split
    return dict(none=float(acc[0]), full_raw_roundtrip=float(acc[1]), passed=bool(abs(float(acc[1]) - 0.5) <= 0.02))


def score(S, Ct, Cte):
    sc = D1.gscore(S); ex = K.exemplar_bits(S, Ct, exact=True)
    shp = torch.stack([torch.as_tensor(K.perm_within(Cte, 200 + k), device=DEV) for k in range(3)])
    exf = torch.stack([K.exemplar_bits(S[:, :, q], Ct, exact=True) for q in shp], 1).mean(1)
    global_perms = [np.random.default_rng(200 + k).permutation(len(Cte)) for k in range(3)]
    f200 = torch.stack([D1.gscore(S.index_select(-1, torch.as_tensor(q, device=S.device)))["soft12"]
                        for q in global_perms], 1).mean(1)
    conditional_200 = torch.stack([D1.gscore(S[:, :, q])["soft12"] for q in shp], 1).mean(1)
    return dict(soft12=sc["soft12"].tolist(), top1=sc["top1"].tolist(), floor_soft12=f200.tolist(),
                within_profile_shuffle_soft12=conditional_200.tolist(),
                exemplar_exact=ex.tolist(), exemplar_exact_floor=exf.tolist())


def subject(s, out, Ctr, Cte, cid, no_ubp):
    t0 = time.time(); sub = f"sub-{s:02d}"; names = ["none", "full"] + [f"permprof{k}" for k in range(NSEED)]
    Cg, Ct = torch.as_tensor(Ctr, dtype=torch.float64, device=DEV), torch.as_tensor(Cte, dtype=torch.float64, device=DEV)
    Zp = [torch.as_tensor(K.perm_profiles(Ctr, cid, 300 + k), dtype=torch.float64, device=DEV) for k in range(NSEED)]
    res = dict(participant=sub, code_sha256=CODE_SHA256, erasers=names, unit="bits per query (k = 80)", decoders={},
               wording="after removing the linearly decodable category subspace of the top-1024 PCs (input space of each decoder)")
    # ---- our 63 x 250 grid (NICE-GA, BrainFLORA)
    Xtr, Xk80, _, _, ch = D1.load_gpu(sub); n = len(Xtr); Xtr = Xtr.reshape(n, -1)
    Xc = np.load(ROOT / f"data_raw/things_eeg2_250hz/test_cache/{sub}_test.npy", mmap_mode="r"); Xt = torch.as_tensor(np.asarray(Xc.mean(1)), device=DEV).float()
    res["test_cache_vs_preprocessed_k80_max_abs"] = float((Xt - Xk80).abs().max()); del Xk80
    res["g3_grid63"] = g3(Xtr, Cg, Zp); torch.cuda.empty_cache()
    E = fit_eraser(Xtr, Cg, Zp); res["rank_full_grid63"] = E["rank"]; del Xtr, E["A"]; torch.cuda.empty_cache()
    Xe = erase(E, Xt.reshape(200, -1)).float().reshape(len(names), 200, 63, 250); res["g2_identity_max_abs_grid63"] = float((Xe[0] - Xt).abs().max())
    print(sub, "eraser grid63 fit", f"rank {E['rank']}", "G3", res["g3_grid63"], f"{time.time()-t0:.0f}s", flush=True); del E; torch.cuda.empty_cache()
    MD = S1.models(sub, DEV); s1v4 = json.loads((ROOT / f"results/ext/s1_v4/{sub}.json").read_text())
    for m, (ft, _, Yc) in MD.items():
        Y = torch.as_tensor(Yc, dtype=torch.float64, device=DEV)
        with torch.no_grad():
            S = torch.stack([D1.gcos(S1.embed(ft, Xe[b].cpu().numpy()), Y) for b in range(len(names))])
        res["decoders"][m] = score(S, Ct, Cte)
        reg = s1v4["g2"].get("r1" if m == "nice_ga" else "bf_registered"); res["decoders"][m]["g2"] = dict(none_soft12=res["decoders"][m]["soft12"][0], registered=reg,
            abs_diff=abs(res["decoders"][m]["soft12"][0] - reg) if reg is not None else None, note="NICE-GA k80 vs r1 has a known undiagnosed gap (S1 v4: 0.11 bits)" if m == "nice_ga" else "")
        print(sub, m, {k: [round(v, 3) for v in vv[:3]] for k, vv in res["decoders"][m].items() if isinstance(vv, list)}, f"{time.time()-t0:.0f}s", flush=True)
    del Xe, MD; torch.cuda.empty_cache()
    # ---- UBP: its own whitened 17 x 250 space, its own training data
    if not no_ubp:
        meta = np.load(ROOT / "data_raw/things_eeg2_images/images_set/image_metadata.npy", allow_pickle=True).item()
        tr = safe_load(str(UD / f"things-eeg/Preprocessed_data_250Hz_whiten/{sub}/train.pt")); timg = np.asarray(tr["img"]); timg = timg[:, 0] if timg.ndim > 1 else timg
        con = [p.replace("\\", "/").split("/")[-2] for p in timg]; assert con == list(meta["train_img_concepts"]), "UBP train order != THINGS metadata order"
        Xu_tr = torch.from_numpy(np.ascontiguousarray(np.asarray(tr["eeg"])[:, :, W1.SEL])).float().mean(1)[..., 0:250].reshape(len(con), -1).to(DEV); del tr
        res["g3_ubp17"] = g3(Xu_tr, Cg, Zp); Eu = fit_eraser(Xu_tr, Cg, Zp); res["rank_full_ubp17"] = Eu["rank"]; del Xu_tr, Eu["A"]; torch.cuda.empty_cache()
        t = safe_load(str(UD / f"things-eeg/Preprocessed_data_250Hz_whiten/{sub}/test.pt")); imgs = list(np.asarray(t["img"])[:, 0]); assert imgs == sorted(imgs)
        assert [p.replace("\\", "/").split("/")[-2] for p in imgs] == list(meta["test_img_concepts"])
        FT = safe_load(str(UD / "things-eeg/Image_feature/FoveaBlur/eeg_intra-subject_ubp_EEGProjectLayer_RN50_test.pt"))["img_features"]
        Xu = torch.from_numpy(np.ascontiguousarray(np.asarray(t["eeg"])[:, :, W1.SEL])).mean(1).float()[..., 0:250].to(DEV); del t
        Xue = erase(Eu, Xu.reshape(200, -1)).float().reshape(len(names), 200, 17, 250); res["g2_identity_max_abs_ubp17"] = float((Xue[0] - Xu).abs().max()); del Eu
        ck = safe_load(str(UD / f"ckpt/{sub}/last.ckpt")); enc = W1.Enc().to(DEV); enc.load_state_dict({k[6:]: v for k, v in ck["state_dict"].items() if k.startswith("brain.")}); enc.eval()
        Yu = Fnn.normalize(torch.stack([FT["medium"][p] for p in imgs]).float().to(DEV), dim=1).double()
        with torch.no_grad():
            S = torch.stack([D1.gcos(enc(Xue[b]).double(), Yu) for b in range(len(names))])
        res["decoders"]["ubp"] = score(S, Ct, Cte)
        card = json.loads((ROOT / "results/baselines/ubp/ubp_card_full_a.json").read_text())["subjects"].get(str(s), {}).get("released")
        if card is None: card = json.loads((ROOT / "results/baselines/ubp/ubp_card_full_b.json").read_text())["subjects"].get(str(s), {}).get("released")
        res["decoders"]["ubp"]["g2"] = dict(none_top1=res["decoders"]["ubp"]["top1"][0], card_top1=card["k80_top1"] if card else None, none_soft12=res["decoders"]["ubp"]["soft12"][0],
                                            card_k80_bits=card["k80_bits"] if card else None, passed=bool(card and abs(res["decoders"]["ubp"]["top1"][0] - card["k80_top1"]) < 1e-6))
        print(sub, "ubp", {k: [round(v, 3) for v in vv[:3]] for k, vv in res["decoders"]["ubp"].items() if isinstance(vv, list)}, "G2", res["decoders"]["ubp"]["g2"], "G3", res["g3_ubp17"], flush=True)
    for m, d in res["decoders"].items():
        none_ = d["exemplar_exact"][0] - d["exemplar_exact_floor"][0]; kept = [(d["exemplar_exact"][i] - d["exemplar_exact_floor"][i]) / none_ if none_ != 0 else None for i in range(len(names))]
        n200 = d["soft12"][0] - d["floor_soft12"][0]
        d["kept_exemplar"] = dict(zip(names, kept)); d["kept_200way"] = dict(zip(names, [(d["soft12"][i] - d["floor_soft12"][i]) / n200 for i in range(len(names))]))
    res["runtime_s"] = round(time.time() - t0, 1); (out / f"{sub}.json").write_text(json.dumps(res, indent=1))
    print(sub, "kept exemplar", {m: {k: round(v, 3) for k, v in d["kept_exemplar"].items() if v is not None and k in ("full", "permprof0")} for m, d in res["decoders"].items()}, f"{res['runtime_s']}s", flush=True)


def summarize(out):
    P = [json.loads(f.read_text()) for f in sorted(out.glob("sub-*.json"))]; n = len(P); names = P[0]["erasers"]
    g = torch.Generator(device=DEV); g.manual_seed(20261121); ix = torch.randint(0, n, (10000, n), generator=g, device=DEV)
    T = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float64, device=DEV); q = lambda x: torch.quantile(x, torch.tensor([.025, .975], dtype=torch.float64, device=DEV)).tolist()
    res = dict(code_sha256=CODE_SHA256, n=n, participants=[p["participant"] for p in P], code_sha256_per_participant=sorted({p["code_sha256"] for p in P}),
               wording=P[0]["wording"], g3={k: dict(full_mean=float(np.mean([p[k]["full_raw_roundtrip"] for p in P if k in p])), n_pass=sum(p[k]["passed"] for p in P if k in p)) for k in ("g3_grid63", "g3_ubp17")},
               decoders={})
    for m in P[0]["decoders"]:
        R = {}
        for metric, fl in (("exemplar_exact", "exemplar_exact_floor"), ("soft12", "floor_soft12")):
            v = lambda i: T([p["decoders"][m][metric][i] - p["decoders"][m][fl][i] for p in P])
            none_ = v(0); full = v(1); perm = torch.stack([v(2 + k) for k in range(NSEED)]).mean(0)
            kf = full[ix].mean(1) / none_[ix].mean(1); kp = perm[ix].mean(1) / none_[ix].mean(1)
            R[metric] = dict(none_minus_floor=float(none_.mean()), kept_full=float(full.mean() / none_.mean()), kept_full_ci=q(kf), kept_permprof=float(perm.mean() / none_.mean()), kept_permprof_ci=q(kp),
                             full_minus_permprof=float((full - perm).mean() / none_.mean()), full_minus_permprof_ci=q(kf - kp), floors_max_abs=float(max(abs(p["decoders"][m][fl][i]) for p in P for i in range(len(names)))))
            if metric == "exemplar_exact":
                cf, cp = R[metric]["kept_full_ci"], R[metric]["kept_permprof_ci"]; ov = not (cf[1] < cp[0] or cp[1] < cf[0])
                R["falsifier"] = dict(rule="exemplar kept share under full erasure overlaps the permuted-profile control (CI) -> decoder uses exemplar information outside the linearly decodable category subspace of the top-1024 PCs",
                                      cis_overlap=bool(ov), triggered=bool(ov))
        R["g2"] = [p["decoders"][m].get("g2") for p in P]; res["decoders"][m] = R
    (out / "summary.json").write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k != "decoders"}, indent=1), json.dumps({m: {k: v for k, v in d.items() if k != "g2"} for m, d in res["decoders"].items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9,10"); ap.add_argument("--out", required=True); ap.add_argument("--summarize", action="store_true"); ap.add_argument("--no_ubp", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize: summarize(out)
    else:
        Ctr, Cte, _, _, cid = K.categories()
        for s in [int(x) for x in a.subjects.split(",")]:
            if not (out / f"sub-{s:02d}.json").exists(): subject(s, out, Ctr, Cte, cid, a.no_ubp)
