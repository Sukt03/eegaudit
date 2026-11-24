"""S1 AVERAGING-CURVE STRESS TEST (rungs R3/R4/R6) with N2 (category-level split-half ceiling) and N3 (k* crossing) inside.
EXECUTOR-2, 21 Nov 2026. Spec: "## S1 STRESS TEST", "## G1 SPEC S1" (METHOD 08:20), "## S1 SPEC CHECK" (METHOD 08:30), METHOD ROUND 2
N2/N3 rows, "## MASTER RESUME ROUTING (10:47)". Existing checkpoints only, no training of EEG models. GPU only (PLAN §10).

Models (frozen): NICE-GA seed 141 (ours; candidates = the paper's CLIP class centres, as p1/U1) and BrainFLORA released single-EEG (U1
adapter; native ViT-H test candidates). Test EEG = THINGS-EEG2 test cache (200 images x 80 repetitions).
k grid {1, 2, 4, 8, 16, 32, 40, 80}; 20 random k-subsets of the 80 repetitions per k (rng 20261121; k = 80 = the one full set), EEG averaged
over the subset, encoded, scores averaged over subsets.
Quantities per (model, k):
  yield200   InfoNCE log2 p(true) - log2(1/200), LOO temperature refit at each k and subset (= run.soft_bits; soft_bits_gpu); top-1 and
             Wolpaw as reference columns.
  floors     shuffled pairs (100 permutations of the candidate rows) and N(0,1) input of the same shape through the same model.
  ceiling    (k <= 40) split-half of the trained code: two DISJOINT k-subsets per image (20 random pairs; k = 1: 20 disjoint single pairs),
             cosine, the same InfoNCE refit at that k. k = 80: no matched ceiling; the k = 40 ceiling is printed, labelled "mismatched k".
  exemplar   within-category N-way (S1 SPEC CHECK 2): pools = test images with the EXACT same multi-hot THINGS category27 profile (as KM3's
             G1 FIX); N fixed (4, or the largest N keeping >= 80 % of categorised test images; printed); own image + N-1 same-pool distractors,
             50 draws per query (fixed seed); score log2 p(true) - log2(1/N) at the query's 200-way LOO temperature of the same k (refit-at-N
             printed as sensitivity). Floor = the score of a random same-pool image as "true" (labels shuffled within category), 100 draws
             -> mean and p95. Within-category ceiling = the same N-way construction on the split halves (k <= 40).
  category   category-level identification (N2): classes = exact category profiles with >= 2 test images; query embedding vs class
             centroids of the candidate features; InfoNCE over classes with LOO temperature; floor = query class labels permuted (100 ->
             mean, p95); category ceiling (N2) = half-A queries vs half-B class means (k <= 40).
  twin       random-init twin (same architecture, re-initialised, seed 0, frozen) + a ridge -> CLIP ViT-H probe trained on the 16,540 training
             4-repetition means (alpha by run.pick_alpha's dev rule on the GPU); the trained model's own embedding goes through the SAME probe
             recipe as an extra row ("trained + probe"); candidates ViT-H test features; scored at every k.
  fraction   (yield - floor) / (ceiling - floor) per k for yield200, exemplar and category; primary family k in {1, 4, 16, 40}.
N3: per participant, k* = smallest k on the grid whose yield exceeds that participant's floor p95, log-k linear interpolation, for exemplar
  and category separately (summary: group median with participant-bootstrap CI; paired log2 ratio k*_ex / k*_cat).
G2: NICE-GA k = 80 = results/method_pilots/r1 full-width InfoNCE (registered E2 group 2.40) per participant; NICE-GA k = 1 on the p1 draws
  (rng 141 sequence) = p1_rank_bound k1 I_NCE (group = ViNiceOne 0.472); BrainFLORA k = 80 = results/standard/summary.json brainflora_native
  per-participant k80 (group = StdBfKEighty 3.272). G3: floors within 0.02 at every k.
Stats (--summarize): participant bootstrap (10,000; GPU index tensors); Holm across k within (model x quantity); paired NICE-GA vs
  BrainFLORA per k. Falsifier: k = 1 exemplar yield >= 50 % of the k = 1 WITHIN-CATEGORY ceiling.
Usage: python s1_averaging.py --subjects 1 --out results/ext/s1 ; python s1_averaging.py --out results/ext/s1 --summarize
"""
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/path/to/eegaudit"); O = ROOT / "outputs/AGENT_LOOP_20261120"; SRC = O / "models/brainflora/source"
for p in (SRC, ROOT, ROOT / "experiments/ext", ROOT / "experiments/eegbits"):
    sys.path.insert(0, str(p))
import m3_yield_map as M
CODE_SHA256 = hashlib.sha256(open(__file__, "rb").read()).hexdigest()   # LOCK proposal item 6
DEV = "cuda"; KS = [1, 2, 4, 8, 16, 32, 40, 80]; NSUB = 20; CATD = ROOT / "data_raw/third_party/things_categories"
META = ROOT / "data_raw/things_eeg2_images/images_set/image_metadata.npy"   # as km3_category_erasure.py


def categories():
    """Exact multi-hot category27 profile per test image (as km3_category_erasure.categories)."""
    uid = [m["uniqueID"] for m in csv.DictReader(open(CATD / "concepts-metadata_things.tsv"), delimiter="\t")]
    cat = np.loadtxt(CATD / "category27_manual.tsv", delimiter="\t", skiprows=1); pos = {u: i for i, u in enumerate(uid)}
    im = np.load(META, allow_pickle=True).item()
    te = np.stack([cat[pos[c.split("_", 1)[1]]] for c in np.array(im["test_img_concepts_THINGS"])])
    return te


def pools_from(te):
    """Pools of identical non-empty profiles; N = 4 or the largest N keeping >= 80 % of categorised test images."""
    keyed = {}
    for i, r in enumerate(te):
        if r.sum() > 0: keyed.setdefault(tuple(r), []).append(i)
    ncat = int((te.sum(1) > 0).sum()); sizes = {i: len(v) for v in keyed.values() for i in v}
    N = 4 if sum(s >= 4 for s in sizes.values()) >= 0.8 * ncat else max([n for n in range(2, 40) if sum(s >= n for s in sizes.values()) >= 0.8 * ncat] or [2])
    pool_of = {i: v for v in keyed.values() for i in v if len(v) >= N}
    classes = [v for v in keyed.values() if len(v) >= 2]
    cov = dict(categorised=ncat, N=N, eligible_exemplar=len(pool_of), n_queries_all=len(pool_of), n_queries_matched=sum(len(v) >= N + 1 for v in pool_of.values()),
               n_classes=len(classes), in_classes=sum(len(v) for v in classes))
    return N, pool_of, classes, cov


def taus():
    """FIXER (METHOD 18:40, G1 FIX S1 CALIBRATION): the registered soft12 grid (run._TAUS, 10^-2.5..1, unchanged points) extended DOWN to
    10^-4.5 with 9 extra points, so a sharp construction (ceilings, large k) cannot sit at the lower edge. Where the old optimum was interior
    the LOO choice is unchanged (G2 vs the registered soft12 numbers still applies)."""
    import run as R
    # v6 (REVIEWER S1 ESTIMATOR RULING (b), 21 Nov ~18:00): the LOCKED soft12 grid for EVERY construction. A grid extended downward makes the
    # per-query LOO temperature fall to the lower edge exactly on a misidentified query of a near-saturated construction (unbounded loss:
    # v5 ceilings -44.8 bits; v4 biased ceilings down by up to 1.6 bits). Saturated constructions are labelled, not re-gridded.
    return torch.tensor(R._TAUS, dtype=torch.float64, device=DEV)


OLD_EDGE = 10 ** -2.5   # lower edge of the registered soft12 grid


def tau_info(t):
    """median fitted tau + whether it sits at the old soft12 lower edge / the extended grid's edges."""
    T = taus(); tt = torch.as_tensor(t, dtype=torch.float64).flatten(); m = float(tt.median())
    return dict(median=m, at_old_soft12_lower_edge=bool(m <= OLD_EDGE * (1 + 1e-9)), at_grid_edge=bool(m <= float(T.min()) * (1 + 1e-9) or m >= float(T.max()) * (1 - 1e-9)),
                n_q_lower_edge=int((tt <= float(T.min()) * (1 + 1e-9)).sum()), n_q=int(tt.numel()))   # v6 diagnostic (REVIEWER)


def loo_ce(S):
    """S (B, n, m) with the true candidate at column index = row index (n = m) -> CE (B, T, n) bits and LOO tau index (B, n)."""
    T = taus(); n = S.shape[-1]; Z = S[:, None] / T[None, :, None, None]
    CE = -torch.diagonal(torch.log_softmax(Z, dim=-1), dim1=-2, dim2=-1) / np.log(2); loo = (CE.sum(-1, keepdim=True) - CE) / (n - 1)
    return CE, loo.argmin(1)


def info200(S):
    """(B, 200, 200) -> (B,) InfoNCE bits, (B,) top-1, (B, 200) LOO tau per query."""
    CE, ti = loo_ce(S); n = S.shape[-1]; v = np.log2(n) - torch.gather(CE, 1, ti[:, None])[:, 0].mean(-1)
    top = (S.argmax(-1) == torch.arange(n, device=DEV)).double().mean(-1); return v, top, taus()[ti]


def nway(S, tau, pool_of, N, draws, rng_seed, shuffled=False):
    """Within-category N-way score. S (B, n, n) cosine query x candidate. Returns dict(all=(B,), matched=(B,), tau=float).
    FIXER (METHOD RULINGS BUG HUNT R2, ruling 1): tau=None -> ONE temperature per call (= per participant x k x construction), fit
    leave-one-QUERY-out on THIS N-way task over the taus() grid (score pooled over the batch and the draws); the same rule is applied
    to the yield, each null replicate and the ceiling. tau (B, n) -> the old fixed per-query 200-way temperature (sensitivity).
    matched = queries whose pool has >= N other images (the null's set); all = every pool >= N (yield only)."""
    rng = np.random.default_rng(rng_seed); q = sorted(pool_of)
    if shuffled: q = [i for i in q if len(pool_of[i]) >= N + 1]
    idx, qi = [], []
    for a, i in enumerate(q):
        others = [j for j in pool_of[i] if j != i]
        for _ in range(draws):
            if shuffled:
                # coordinator fix (bias -0.14 at k = 1): the null must NOT keep the query's own image among the candidates. Draw the fake
                # "true" image and the N-1 distractors from the OTHER same-pool images; pools with fewer than N other images are skipped.
                if len(others) < N: break
                pick = list(rng.choice(others, N, replace=False)); idx.append([i] + pick); qi.append(a); continue
            d = list(rng.choice(others, N - 1, replace=False)); idx.append([i, i] + d); qi.append(a)
    if not idx:
        nan = torch.full((S.shape[0],), float("nan"), dtype=torch.float64, device=DEV); return dict(all=nan, matched=nan, tau=float("nan"))
    I = torch.as_tensor(np.array(idx), device=DEV); rows, true, cand = I[:, 0], I[:, 1], torch.cat([I[:, 1:2], I[:, 2:]], 1)   # true first
    Q = torch.as_tensor(qi, device=DEV); nq = len(q); cnt = torch.bincount(Q, minlength=nq).double()
    mt = torch.as_tensor([len(pool_of[i]) >= N + 1 for i in q], device=DEV)
    sc = S[:, rows[:, None], cand]                                                                                             # (B, R, N)
    if tau is not None:
        lp = torch.log_softmax(sc / tau[:, rows][:, :, None], -1)[..., 0] / np.log(2) + np.log2(N)                             # (B, R)
        per = torch.zeros(S.shape[0], nq, dtype=torch.float64, device=DEV).index_add_(1, Q, lp) / cnt                         # (B, nq)
        return dict(all=per.mean(-1), matched=per[:, mt].mean(-1), tau=float("nan"))
    T = taus(); lp = torch.log_softmax(sc[:, None] / T[None, :, None, None], -1)[..., 0] / np.log(2) + np.log2(N)             # (B, T, R)
    per = torch.zeros(S.shape[0], len(T), nq, dtype=torch.float64, device=DEV).index_add_(2, Q, lp) / cnt                     # (B, T, nq)
    out = {}
    for name, m in (("all", torch.ones_like(mt)), ("matched", mt)):
        P = per[:, :, m]; pq = P.mean(0); n_ = pq.shape[1]                                                                     # (T, n_) pooled over B
        ti = ((pq.sum(1, keepdim=True) - pq) / max(n_ - 1, 1)).argmax(0)                                                       # LOO tau per query
        out[name] = P[:, ti, torch.arange(n_, device=DEV)].mean(-1)
        if name == "matched": out["tau"] = float(T[ti].median())
    return out


def catid(S_cent, labels, perm_seed=None, ret_tau=False):
    """S_cent (B, n, C) query x class-centroid cosine; labels (n,) class index -> (B,) InfoNCE over C classes with LOO tau over queries."""
    T = taus(); C = S_cent.shape[-1]; lab = torch.as_tensor(labels, device=DEV)
    if perm_seed is not None: lab = lab[torch.as_tensor(np.random.default_rng(perm_seed).permutation(len(labels)), device=DEV)]
    lp = torch.log_softmax(S_cent[:, None] / T[None, :, None, None], -1)                                                       # (B, T, n, C)
    CE = -lp[..., torch.arange(len(lab), device=DEV), lab] / np.log(2); loo = (CE.sum(-1, keepdim=True) - CE) / (len(lab) - 1); ti = loo.argmin(1)
    out_ = np.log2(C) - torch.gather(CE, 1, ti[:, None])[:, 0].mean(-1)
    return (out_, T[ti]) if ret_tau else out_


def models(sub, dev):
    import run_acq_nice as N
    from model.medformer_encoders import eeg_encoder
    out = {}
    Enc, Proj = N.load_classes("nice_ga"); tag = f"nice_ga_dgx_v2_{sub.replace('-', '')}_seed141"
    def nice(trained):
        torch.manual_seed(0); enc, proj = Enc().to(dev).eval(), Proj().to(dev).eval()
        if trained:
            enc.load_state_dict(torch.load(N.NICE / "model" / f"{tag}_Enc_eeg_cls.pth", map_location=dev, weights_only=True), strict=True)
            proj.load_state_dict(torch.load(N.NICE / "model" / f"{tag}_Proj_eeg_cls.pth", map_location=dev, weights_only=True), strict=True)
        return lambda t: proj(enc(t.unsqueeze(1)))
    def bfl(trained):
        torch.manual_seed(0); bf = eeg_encoder()
        if trained: bf.load_state_dict(torch.load(O / "models/brainflora/eeg_01-06_01-46_150.pth", map_location="cpu", weights_only=True), strict=True)
        bf = bf.eval().to(dev); return lambda t: bf(t)
    Yn = np.squeeze(np.load(N.NICE / "dnn_feature/center_clip.npy", allow_pickle=True)).astype(np.float64)
    Yb = torch.load(O / "models/brainflora/ViT-H-14_features_test.pt", map_location="cpu", weights_only=True); Yb = (Yb["img_features"] if isinstance(Yb, dict) else Yb).float().numpy()
    out["nice_ga"] = (nice(True), nice(False), Yn); out["brainflora"] = (bfl(True), bfl(False), Yb.astype(np.float64))
    return out


@torch.inference_mode()
def embed(f, X, bs=256):
    return torch.cat([f(torch.as_tensor(np.ascontiguousarray(X[i:i + bs]), device=DEV, dtype=torch.float32)).double() for i in range(0, len(X), bs)])


def nrm(Z):
    return Z / Z.norm(dim=-1, keepdim=True)


def subject(s, out):
    import run as R
    t0 = time.time(); sub = f"sub-{s:02d}"; X = np.load(ROOT / f"data_raw/things_eeg2_250hz/test_cache/{sub}_test.npy", mmap_mode="r"); assert X.shape == (200, 80, 63, 250)
    te = categories(); N, pool_of, classes, cov = pools_from(te); lab = np.full(200, -1)
    for c, v in enumerate(classes): lab[v] = c
    inc = np.where(lab >= 0)[0]
    MD = models(sub, DEV); rng = np.random.default_rng(20261121)
    subsets = {k: ([np.arange(80)] if k == 80 else [rng.choice(80, k, replace=False) for _ in range(NSUB)]) for k in KS}
    pairs = {k: [(lambda p: (p[:k], p[k:2 * k]))(rng.permutation(80)) for _ in range(NSUB)] for k in KS if k <= 40}
    # METHOD RULING (S1 ceiling at small k): ceiling(k) = k-repetition mean identified against the mean of the REMAINING 80-k repetitions
    rest = {k: [(lambda p: (p[:k], p[k:]))(np.random.default_rng(4000 + 97 * k + j).permutation(80)) for j in range(NSUB)] for k in KS if k <= 40}
    # twin probes: training 4-repetition means through trained and random-init encoders -> ridge -> ViT-H (dev alpha rule on the GPU)
    C = ROOT / "data_raw/things_eeg2_250hz/cache_f32"
    Xtr = (np.load(C / f"{sub}_train.npy", mmap_mode="r") if (C / f"{sub}_train.npy").exists() else R.load_raw(sub)[0])
    Ytr = torch.as_tensor(R.load_torch(R.CLIP / "ViT-H-14_features_train.pt"), dtype=torch.float64, device=DEV)
    Yvit = nrm(torch.as_tensor(R.load_torch(R.CLIP / "ViT-H-14_features_test.pt"), dtype=torch.float64, device=DEV))
    trm = np.concatenate([np.asarray(Xtr[i:i + 2000]).mean(1) for i in range(0, len(Xtr), 2000)]).astype(np.float32); del Xtr
    probes = {}
    for m, (ft, fr, _) in MD.items():
        for kind, f in (("trained_probe", ft), ("twin_probe", fr)):
            E = embed(f, trm); mu, sd = E.mean(0, keepdim=True), E.std(0, unbiased=False, keepdim=True) + 1e-6; Z = (E - mu) / sd
            a = M.fast_pick_alpha(Z, Ytr, np.random.default_rng(0), R)                                        # dev-MRR alpha (run.pick_alpha rule)
            ym = Ytr.mean(0); Wt = torch.linalg.solve(Z.T @ Z + a * torch.eye(Z.shape[1], dtype=Z.dtype, device=DEV), Z.T @ (Ytr - ym))
            probes[(m, kind)] = (f, mu, sd, Wt, ym, a)
    del trm; print(sub, "probes", {f"{m}/{k}": v[5] for (m, k), v in probes.items()}, f"{time.time()-t0:.0f}s", flush=True)
    perms = torch.as_tensor(np.stack([np.random.default_rng(300 + j).permutation(200) for j in range(100)]), device=DEV)
    res = dict(participant=sub, coverage=cov, code_sha256=CODE_SHA256, tau_grid=[float(t) for t in taus()], rows={})
    for m, (ft, fr, Yc) in MD.items():
        Yc = nrm(torch.as_tensor(Yc, device=DEV)); cent = nrm(torch.stack([Yc[v].mean(0) for v in classes]))
        for k in KS:
            E = torch.stack([nrm(embed(ft, np.asarray(X[:, ids].mean(1)))) for ids in subsets[k]])                    # (S, 200, d)
            S = E @ Yc.T; y, top, tau = info200(S)
            gN = torch.Generator(device=DEV); gN.manual_seed(1000 + k)
            En = nrm(embed(ft, torch.randn((200, 63, 250), generator=gN, device=DEV).cpu().numpy())); yn, _, _ = info200((En @ Yc.T)[None])
            ysh = torch.stack([info200(S[:, :, p])[0].mean() for p in perms]).mean()                              # shuffled pairs, 100 permutations
            exd = nway(S, None, pool_of, N, 50, 7 + k); ex = exd["matched"]                                         # FIXER: LOO-N-way tau, matched queries
            nl_ = [nway(S, None, pool_of, N, 5, 900 + 10 * k + r, shuffled=True) for r in range(20)]; exf = torch.stack([d_["matched"].mean() for d_ in nl_])   # (20,) floor draws
            ca, ca_tau = catid(E[:, inc] @ cent.T, lab[inc], ret_tau=True); caf = torch.stack([catid(E[:, inc] @ cent.T, lab[inc], perm_seed=500 + r).mean() for r in range(100)])
            row = dict(yield200=float(y.mean()), top1=float(top.mean()), wolpaw=float(np.mean([R.wolpaw_bits(float(t), 200) for t in top])), floor_shuffled=float(ysh), floor_noise=float(yn[0]),
                       exemplar=float(ex.mean()), exemplar_allq=float(exd["all"].mean()), exemplar_tau=exd["tau"], exemplar_floor=float(exf.mean()), exemplar_floor_p95=float(torch.quantile(exf, .95)),
                       category=float(ca.mean()), category_floor=float(caf.mean()), category_floor_p95=float(torch.quantile(caf, .95)))
            row["tau"] = dict(yield200=tau_info(tau), exemplar=tau_info([exd["tau"]]), exemplar_null=tau_info([d_["tau"] for d_ in nl_]), category=tau_info(ca_tau))   # FIXER calibration fields
            # FIXER sensitivity: the old per-query 200-way LOO temperature, same draws and matched queries (yield and null)
            row["exemplar_tau200"] = float(nway(S, tau, pool_of, N, 50, 7 + k)["matched"].mean())
            row["exemplar_floor_tau200"] = float(torch.stack([nway(S, tau, pool_of, N, 5, 900 + 10 * k + r, shuffled=True)["matched"].mean() for r in range(20)]).mean())
            for kind in ("trained_probe", "twin_probe"):
                f, mu, sd, Wt, ym, _ = probes[(m, kind)]
                P = torch.stack([nrm(((embed(f, np.asarray(X[:, ids].mean(1))) - mu) / sd) @ Wt + ym) for ids in subsets[k]]); row[kind] = float(info200(P @ Yvit.T)[0].mean())
            if k <= 40:
                A = torch.stack([nrm(embed(ft, np.asarray(X[:, a_].mean(1)))) for a_, _ in pairs[k]]); Bh = torch.stack([nrm(embed(ft, np.asarray(X[:, b_].mean(1)))) for _, b_ in pairs[k]])
                Sc = A @ Bh.transpose(1, 2); yc, _, tc = info200(Sc); row["ceiling200"] = float(yc.mean())
                ce_ = nway(Sc, None, pool_of, N, 50, 70 + k); row["ceiling_exemplar"] = float(ce_["matched"].mean()); row["ceiling_exemplar_allq"] = float(ce_["all"].mean())
                row["tau"]["ceiling200"] = tau_info(tc); row["tau"]["ceiling_exemplar"] = tau_info([ce_["tau"]])
                centB = torch.stack([nrm(torch.stack([Bh[j][v].mean(0) for v in classes])) for j in range(len(Bh))])
                cc_, cc_tau = catid(A[:, inc] @ centB.transpose(1, 2), lab[inc], ret_tau=True); row["ceiling_category"] = float(cc_.mean()); row["tau"]["ceiling_category"] = tau_info(cc_tau)  # coordinator: bug-review condition 2
                # keep the symmetric k-vs-k numbers as a labelled reliability curve; the ceiling is now k-mean vs rest-mean (= registered k40 ceiling at k = 40)
                for q in ("ceiling200", "ceiling_exemplar", "ceiling_category"): row["reliability_kvk_" + q.replace("ceiling_", "").replace("ceiling", "")] = row[q]
                A = torch.stack([nrm(embed(ft, np.asarray(X[:, a_].mean(1)))) for a_, _ in rest[k]]); Bh = torch.stack([nrm(embed(ft, np.asarray(X[:, b_].mean(1)))) for _, b_ in rest[k]])
                Sc = A @ Bh.transpose(1, 2); yc, _, tc = info200(Sc); row["ceiling200"] = float(yc.mean())
                ce_ = nway(Sc, None, pool_of, N, 50, 70 + k); row["ceiling_exemplar"] = float(ce_["matched"].mean()); row["ceiling_exemplar_allq"] = float(ce_["all"].mean())
                row["tau"]["ceiling200"] = tau_info(tc); row["tau"]["ceiling_exemplar"] = tau_info([ce_["tau"]])
                centB = torch.stack([nrm(torch.stack([Bh[j][v].mean(0) for v in classes])) for j in range(len(Bh))])
                cc_, cc_tau = catid(A[:, inc] @ centB.transpose(1, 2), lab[inc], ret_tau=True); row["ceiling_category"] = float(cc_.mean()); row["tau"]["ceiling_category"] = tau_info(cc_tau)  # coordinator: bug-review condition 2
                row["ceiling_construction"] = "k-mean vs rest-mean (80-k), 20 splits"
            res["rows"][f"{m}_k{k}"] = row
            print(sub, m, k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in row.items()}, f"{time.time()-t0:.0f}s", flush=True)
    # G2
    r1 = json.loads((ROOT / f"results/method_pilots/r1_input_bottleneck_{sub}_basis-{sub}.json").read_text())
    r1v = next(r["soft_bits"] for r in r1["rows"] if r["L"] == "full" and r["k"] == 80 and r["condition"] == "eeg")
    ft, _, Yn = MD["nice_ga"]; rr = np.random.default_rng(141); rr.permutation(200); d5 = []
    for d in range(5): d5.append(int(rr.choice(80, 1)[0])); rr.permutation(200)
    k1p1 = [float(info200((nrm(embed(ft, np.asarray(X[:, [d]].mean(1)))) @ nrm(torch.as_tensor(Yn, device=DEV)).T)[None])[0][0]) for d in d5]
    p1 = [x["I_NCE"] for x in json.loads((ROOT / "results/loop/image_p1/p1_rank_bound.json").read_text())["per_participant"]["nice_ga"][sub]["k1"]]
    bfk80 = next(r["k80"] for r in json.loads((ROOT / "results/standard/summary.json").read_text())["image"]["open_weight_models"]["brainflora_native"]["per_participant"] if r["subject"] == sub)
    res["g2"] = dict(nice_k80_vs_r1=abs(res["rows"]["nice_ga_k80"]["yield200"] - r1v), nice_k1_p1draws_max_abs=max(abs(a - b) for a, b in zip(k1p1, p1)),
                     brainflora_k80_vs_registered=abs(res["rows"]["brainflora_k80"]["yield200"] - bfk80), r1=r1v, bf_registered=bfk80)
    res["g2"]["pass"] = bool(res["g2"]["nice_k80_vs_r1"] < 1e-3 and res["g2"]["nice_k1_p1draws_max_abs"] < 1e-3 and res["g2"]["brainflora_k80_vs_registered"] < 1e-3)
    res["runtime_s"] = round(time.time() - t0, 1); (out / f"{sub}.json").write_text(json.dumps(res, indent=1)); print(sub, "G2", res["g2"], flush=True)


def kstar(ks, y, thr):
    """FIXER (13:20 fix 3 / R2 ruling 1c): thr is the PER-k floor p95 (array); crossing of y - thr, log-k linear interpolation."""
    lk = np.log2(ks); d = np.asarray(y, float) - np.broadcast_to(np.asarray(thr, float), (len(ks),)); above = d > 0
    if not above.any(): return float("nan")
    i = int(np.argmax(above))
    if i == 0: return float(ks[0])
    t = -d[i - 1] / (d[i] - d[i - 1]); return float(2 ** (lk[i - 1] + t * (lk[i] - lk[i - 1])))


def summarize(out):
    per = [json.loads(f.read_text()) for f in sorted(out.glob("sub-*.json"))]; n = len(per)
    g = torch.Generator(device=DEV); g.manual_seed(20261121); ix = torch.randint(0, n, (10000, n), generator=g, device=DEV)
    def st(v):
        t = torch.as_tensor(np.asarray(v, float), device=DEV); bm = t[ix].mean(1)
        return dict(mean=float(t.mean()), ci=torch.quantile(bm, torch.tensor([.025, .975], dtype=torch.float64, device=DEV)).tolist(), p_le0=float((bm <= 0).double().mean()))
    res = dict(n=n, code_sha256=CODE_SHA256, per_participant_code_sha256=sorted({p.get("code_sha256") for p in per}, key=str), participants=[p["participant"] for p in per], coverage=per[0]["coverage"], g2_all_pass=all(p["g2"]["pass"] for p in per), models={})
    for m in ("nice_ga", "brainflora"):
        blk = {}
        for k in KS:
            R_ = [p["rows"][f"{m}_k{k}"] for p in per]; f_ = lambda key: np.array([r[key] for r in R_])
            floor200 = np.maximum(f_("floor_shuffled"), f_("floor_noise")); row = {q: st(f_(q)) for q in R_[0] if isinstance(R_[0][q], (int, float))}  # coordinator: skip text fields (ceiling_construction)
            for q, fl, ce in (("yield200", floor200, "ceiling200"), ("exemplar", f_("exemplar_floor"), "ceiling_exemplar"), ("category", f_("category_floor"), "ceiling_category")):
                cek = ce if k <= 40 else None
                if cek is None and k == 80: c_ = np.array([p["rows"][f"{m}_k40"][ce] for p in per]); row[f"{q}_ceiling_note"] = "mismatched k (k = 40 ceiling)"
                else: c_ = f_(cek)
                row[f"{q}_fraction"] = dict(ratio_of_means=float((f_(q) - fl).mean() / (c_ - fl).mean()), per_participant=((f_(q) - fl) / (c_ - fl)).tolist())
            row["g3_floor_max_abs"] = float(np.abs(np.concatenate([f_("floor_shuffled"), f_("floor_noise")])).max())
            blk[f"k{k}"] = row
        for fam in ("yield200", "exemplar", "category"):   # FIXER (13:20 fix 1): step-down Holm, running max; primary family k in {1,4,16,40}
            for kf, lab_ in (([1, 4, 16, 40], "primary"), ([k for k in KS if k not in (1, 4, 16, 40)], "secondary")):
                ps = sorted((blk[f"k{k}"][fam]["p_le0"], k) for k in kf); run_ = 0.0
                for r, (p, k) in enumerate(ps):
                    run_ = max(run_, min(1.0, p * (len(ps) - r))); blk[f"k{k}"][fam]["p_holm"] = run_; blk[f"k{k}"][fam]["holm_family"] = lab_
        ks_ex = [kstar(KS, [p["rows"][f"{m}_k{k}"]["exemplar"] for k in KS], [p["rows"][f"{m}_k{k}"]["exemplar_floor_p95"] for k in KS]) for p in per]
        ks_ca = [kstar(KS, [p["rows"][f"{m}_k{k}"]["category"] for k in KS], [p["rows"][f"{m}_k{k}"]["category_floor_p95"] for k in KS]) for p in per]
        lr = np.log2(np.array(ks_ex) / np.array(ks_ca)); ok = np.isfinite(lr)
        blk["N3"] = dict(kstar_exemplar=ks_ex, kstar_category=ks_ca, median_ex=float(np.nanmedian(ks_ex)), median_cat=float(np.nanmedian(ks_ca)),
                         log2_ratio=st(lr[ok]) if ok.sum() > 1 else None, n_defined=int(ok.sum()))
        # FIXER (13:20 fix 2 + R2 ruling 2): floor-corrected falsifier, ratio of participant means, participant-bootstrap CI;
        # primary = 1-vs-79 ceiling (ceiling_exemplar at k = 1); "falsified" iff CI lower bound >= 0.50; 1-vs-1 = original pre-registration
        k1 = [p["rows"][f"{m}_k1"] for p in per]; ex1 = np.array([r["exemplar"] for r in k1]); fl1 = np.array([r["exemplar_floor"] for r in k1])
        def fals(ce):
            num = torch.as_tensor(ex1 - fl1, device=DEV); den = torch.as_tensor(np.array([r[ce] for r in k1]) - fl1, device=DEV); b = num[ix].mean(1) / den[ix].mean(1)
            ci = torch.quantile(b, torch.tensor([.025, .975], dtype=torch.float64, device=DEV)).tolist()
            return dict(ratio_of_means=float(num.mean() / den.mean()), ci=ci, falsified=bool(ci[0] >= 0.50), per_participant=(num / den).tolist(), ceiling_field=ce)
        blk["falsifier_k1"] = dict(primary_1v79=fals("ceiling_exemplar"), original_preregistration_1v1=fals("reliability_kvk_exemplar"),
                                   rule="(exemplar - exemplar_floor) / (ceiling - exemplar_floor) at k = 1; falsified iff bootstrap CI lower >= 0.50")
        # FIXER (METHOD 17:35 (3)): calibration across k for the exemplar construction. Stored per row: the yield's fitted N-way tau (median
        # LOO tau) and the null mean (the null/ceiling taus are refit per replicate and not stored). Pass = tau non-increasing in k with at most
        # one inversion inside the earlier k's bootstrap CI, AND the null within 0.02 bits of its k-mean with no adjacent jump > 0.02.
        cal = {}
        cons = sorted({c for p in per for k in KS for c in p["rows"][f"{m}_k{k}"].get("tau", {})})
        cal["tau_by_construction"] = {c: {f"k{k}": dict(median=float(np.median([p["rows"][f"{m}_k{k}"]["tau"][c]["median"] for p in per])),
                                                           n_at_old_soft12_lower_edge=int(sum(p["rows"][f"{m}_k{k}"]["tau"][c]["at_old_soft12_lower_edge"] for p in per)),
                                                           n_at_grid_edge=int(sum(p["rows"][f"{m}_k{k}"]["tau"][c]["at_grid_edge"] for p in per)))
                                          for k in KS if all(c in p["rows"][f"{m}_k{k}"].get("tau", {}) for p in per)} for c in cons}
        for k in KS:
            tk = torch.as_tensor([p["rows"][f"{m}_k{k}"]["exemplar_tau"] for p in per], dtype=torch.float64, device=DEV); bm = tk[ix].median(1).values
            cal[f"k{k}"] = dict(construction="exemplar yield (matched queries)", tau_median=float(tk.median()), tau_median_ci=torch.quantile(bm, torch.tensor([.025, .975], dtype=torch.float64, device=DEV)).tolist(),
                                null_mean=st([p["rows"][f"{m}_k{k}"]["exemplar_floor"] for p in per]))
        tm = [cal[f"k{k}"]["tau_median"] for k in KS]; inv = [i for i in range(1, len(KS)) if tm[i] > tm[i - 1]]
        inv_ok = len(inv) == 0 or (len(inv) == 1 and cal[f"k{KS[inv[0] - 1]}"]["tau_median_ci"][0] <= tm[inv[0]] <= cal[f"k{KS[inv[0] - 1]}"]["tau_median_ci"][1])
        nm_ = np.array([cal[f"k{k}"]["null_mean"]["mean"] for k in KS]); null_ok = bool(np.all(np.abs(nm_ - nm_.mean()) <= 0.02) and np.all(np.abs(np.diff(nm_)) <= 0.02))
        cal["check"] = dict(tau_inversions_at_k=[KS[i] for i in inv], tau_ok=bool(inv_ok), null_k_mean=float(nm_.mean()), null_max_dev=float(np.abs(nm_ - nm_.mean()).max()),
                            null_max_adjacent_jump=float(np.abs(np.diff(nm_)).max()), null_ok=null_ok, stable=bool(inv_ok and null_ok),
                            flag=None if (inv_ok and null_ok) else "calibration not stable across k")
        cal["check"]["constructions_at_grid_edge"] = {c: v for c, v in {c: sum(d["n_at_grid_edge"] for d in kk.values()) for c, kk in cal["tau_by_construction"].items()}.items() if v and c != "exemplar_null"}
        blk["calibration"] = cal
        # FIXER (13:20 fix 4): per-participant values and fractions at k = 1 and k = 40
        blk["per_participant_k1_k40"] = {f"k{k}": {q: [p["rows"][f"{m}_k{k}"][q] for p in per] for q in ("yield200", "exemplar", "exemplar_floor", "ceiling_exemplar", "reliability_kvk_exemplar", "category", "category_floor", "ceiling_category", "ceiling200")}
                                         | {f"{q}_fraction": blk[f"k{k}"][f"{q}_fraction"]["per_participant"] for q in ("yield200", "exemplar", "category")} for k in (1, 40)}
        res["models"][m] = blk
    res["paired_nice_minus_brainflora"] = {f"k{k}": st([p["rows"][f"nice_ga_k{k}"]["yield200"] - p["rows"][f"brainflora_k{k}"]["yield200"] for p in per]) for k in KS}
    # FIXER: new k = 40 ceiling (k-mean vs rest-mean) beside the registered ceiling_k40 per participant
    reg = dict(nice_ga={r["subject"]: r["ceiling_k40"]["I_NCE"] for r in json.loads((ROOT / "results/loop/s3_fraction/s3_fraction_k40.json").read_text())["rows"]},
               brainflora={r["subject"]: r["ceiling_k40"] for r in json.loads((ROOT / "results/standard/summary.json").read_text())["image"]["open_weight_models"]["brainflora_native"]["per_participant"]})
    res["ceiling_k40_vs_registered"] = {m: {p["participant"]: dict(new=p["rows"][f"{m}_k40"]["ceiling200"], registered=reg[m].get(p["participant"])) for p in per} for m in reg}
    (out / "summary.json").write_text(json.dumps(res, indent=1)); print(json.dumps({m: {k: {q: v[q]["mean"] for q in ("yield200", "exemplar", "category") } for k, v in b.items() if k.startswith("k")} for m, b in res["models"].items()}, indent=1))
    for m, b in res["models"].items():
        f_ = b["falsifier_k1"]; print(m, "FALSIFIER primary (1-vs-79, floor-corrected):", round(f_["primary_1v79"]["ratio_of_means"], 3), [round(x, 3) for x in f_["primary_1v79"]["ci"]], "falsified" if f_["primary_1v79"]["falsified"] else "not falsified",
              "| original pre-registration (1-vs-1):", round(f_["original_preregistration_1v1"]["ratio_of_means"], 3), [round(x, 3) for x in f_["original_preregistration_1v1"]["ci"]])
        print(m, "N3 k* exemplar (per-k floor p95):", b["N3"]["kstar_exemplar"])
        print(m, "CALIBRATION tau median by k:", {k: round(v["tau_median"], 4) for k, v in b["calibration"].items() if k != "check" and isinstance(v, dict) and "tau_median" in v},
              "null mean by k:", {k: (round(v["null_mean"]["mean"], 4), [round(x, 4) for x in v["null_mean"]["ci"]]) for k, v in b["calibration"].items() if k != "check" and isinstance(v, dict) and "null_mean" in v}, "CHECK", b["calibration"]["check"])
    for m, d in res["ceiling_k40_vs_registered"].items():
        for sub, v in d.items(): print("ceiling_k40", m, sub, "new", round(v["new"], 4), "registered", None if v["registered"] is None else round(v["registered"], 4))


KP = [1, 4, 16, 40]   # G-R1b k grid (k = 1 is identical in both modes: a check)


def subject_pred(s, out):
    """G-R1b (FIXER, METHOD 'G-R1b ACCEPTED'): PREDICTION averaging = encode every single repetition, then aggregate the outputs.
    Summing k single-repetition log-softmax scores at a common temperature equals a softmax over the MEAN single-repetition cosine
    (the per-row log-normaliser is constant over candidates), with the temperature absorbed by the same LOO refit. So the aggregated
    score matrix is S = mean_r cos(f(x_r), y) over the SAME k-subsets as input averaging (same rng stream), scored with the S1 v3
    estimator: yield200 (LOO tau), exemplar (LOO N-way tau, matched queries) + its null, category + its null."""
    t0 = time.time(); sub = f"sub-{s:02d}"; X = np.load(ROOT / f"data_raw/things_eeg2_250hz/test_cache/{sub}_test.npy", mmap_mode="r"); assert X.shape == (200, 80, 63, 250)
    te = categories(); N, pool_of, classes, cov = pools_from(te); lab = np.full(200, -1)
    for c, v in enumerate(classes): lab[v] = c
    inc = np.where(lab >= 0)[0]; MD = models(sub, DEV); rng = np.random.default_rng(20261121)
    subsets = {k: ([np.arange(80)] if k == 80 else [rng.choice(80, k, replace=False) for _ in range(NSUB)]) for k in KS}    # identical to subject()
    res = dict(participant=sub, coverage=cov, avg_mode="prediction", code_sha256=CODE_SHA256, rows={})
    for m, (ft, fr, Yc) in MD.items():
        Yc = nrm(torch.as_tensor(Yc, device=DEV)); cent = nrm(torch.stack([Yc[v].mean(0) for v in classes]))
        E1 = torch.stack([nrm(embed(ft, np.asarray(X[:, r]))) for r in range(80)])                                       # (80, 200, d) single repetitions
        S1r = E1 @ Yc.T; C1r = E1[:, inc] @ cent.T; del E1                                                                  # (80, 200, 200), (80, n_inc, C)
        for k in KP:
            S = torch.stack([S1r[ids].mean(0) for ids in subsets[k]]); Sc_ = torch.stack([C1r[ids].mean(0) for ids in subsets[k]])
            y, top, tau = info200(S); exd = nway(S, None, pool_of, N, 50, 7 + k)
            nl_ = [nway(S, None, pool_of, N, 5, 900 + 10 * k + r, shuffled=True) for r in range(20)]; exf = torch.stack([d_["matched"].mean() for d_ in nl_])
            ca, ca_tau = catid(Sc_, lab[inc], ret_tau=True); caf = torch.stack([catid(Sc_, lab[inc], perm_seed=500 + r).mean() for r in range(100)])
            row = dict(yield200=float(y.mean()), top1=float(top.mean()), exemplar=float(exd["matched"].mean()), exemplar_allq=float(exd["all"].mean()), exemplar_tau=exd["tau"],
                       exemplar_floor=float(exf.mean()), exemplar_floor_p95=float(torch.quantile(exf, .95)), category=float(ca.mean()), category_floor=float(caf.mean()))
            row["tau"] = dict(yield200=tau_info(tau), exemplar=tau_info([exd["tau"]]), exemplar_null=tau_info([d_["tau"] for d_ in nl_]), category=tau_info(ca_tau))
            res["rows"][f"{m}_k{k}"] = row; print(sub, "PRED", m, k, {a: round(b, 4) for a, b in row.items() if isinstance(b, float)}, f"{time.time()-t0:.0f}s", flush=True)
        del S1r, C1r; torch.cuda.empty_cache()
    res["runtime_s"] = round(time.time() - t0, 1); (out / f"{sub}.json").write_text(json.dumps(res, indent=1))


def summarize_pred(out, ref):
    """prediction - input per model x k x quantity (floor-corrected for exemplar/category), paired participant bootstrap;
    falsifier (pre-stated): output >= input at k = 16 with CI > 0. Reference = the S1 v3 input-averaging files."""
    P = {json.loads(f.read_text())["participant"]: json.loads(f.read_text()) for f in sorted(out.glob("sub-*.json"))}
    I = {json.loads(f.read_text())["participant"]: json.loads(f.read_text()) for f in sorted(Path(ref).glob("sub-*.json"))}
    subs = sorted(set(P) & set(I)); n = len(subs); g = torch.Generator(device=DEV); g.manual_seed(20261121); ix = torch.randint(0, n, (10000, n), generator=g, device=DEV)
    res = dict(n=n, participants=subs, reference=str(ref), contrasts={})
    for m in ("nice_ga", "brainflora"):
        for k in KP:
            for q, fl in (("yield200", None), ("exemplar", "exemplar_floor"), ("category", "category_floor")):
                val = lambda D, s_: D[s_]["rows"][f"{m}_k{k}"][q] - (D[s_]["rows"][f"{m}_k{k}"][fl] if fl else 0.0)
                d = torch.as_tensor([val(P, s_) - val(I, s_) for s_ in subs], dtype=torch.float64, device=DEV); bm = d[ix].mean(1)
                res["contrasts"][f"{m}_k{k}_{q}"] = dict(pred_minus_input=float(d.mean()), ci=torch.quantile(bm, torch.tensor([.025, .975], dtype=torch.float64, device=DEV)).tolist(),
                                                         n_pos=int((d > 0).sum()), per_participant=d.tolist())
        c = res["contrasts"][f"{m}_k16_yield200"]; res[f"falsifier_{m}"] = dict(rule="output (prediction) >= input averaging at k = 16, CI > 0", triggered=bool(c["ci"][0] > 0), d=c["pred_minus_input"], ci=c["ci"])
        res[f"k1_check_{m}"] = max(abs(res["contrasts"][f"{m}_k1_{q}"]["pred_minus_input"]) for q in ("yield200",))
    (out / "summary_pred.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "contrasts"}, indent=1)); print({k: (round(v["pred_minus_input"], 4), [round(x, 4) for x in v["ci"]]) for k, v in res["contrasts"].items()})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default=""); ap.add_argument("--out", required=True); ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--avg_mode", choices=["input", "prediction"], default="input"); ap.add_argument("--ref", default=str(ROOT / "results/ext/s1_v3"))   # G-R1b
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize: return summarize(out) if a.avg_mode == "input" else summarize_pred(out, a.ref)
    for s in ([int(x) for x in a.subjects.split(",")] if a.subjects else range(1, 11)):
        if not (out / f"sub-{s:02d}.json").exists(): (subject if a.avg_mode == "input" else subject_pred)(s, out)


if __name__ == "__main__":
    main()
