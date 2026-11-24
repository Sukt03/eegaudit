"""I-1 v2 DECODER-FREE EXEMPLAR CEILING, WHITENED (REVIEWER overnight I: G1 FIX). FIXER-3, 21 Nov 2026. THINGS-EEG2, 10 participants.
v1 (i1_eeg_ceiling.py) scored a raw 15,750-dim cosine that noise dimensions dominate (0.28 bits 200-way at k40), so "EEG ceiling <= decoder
ceiling" fired trivially. v2 PRIMARY ('white'): every step fitted on TRAINING data only (16,540 training images x 4 repetitions; the test
EEG is never used to fit anything):
  1. per-feature z-score with the mean/sd of the training image means (4-repetition means);
  2. PCA-256 of the z-scored training image means (torch.svd_lowrank, q 300, niter 6, seed 0; explained variance stored);
  3. whitening by the within-image repetition (noise) covariance of the training repetitions in the PCA-256 space
     (residual of each repetition around its image mean, unbiased /(n_img * 3); symmetric inverse square root, ridge 1e-6 x mean eigenvalue).
Test: each repetition of the 200 test images -> z (training stats) -> PCA-256 -> whiten; k-repetition mean vs the mean of the remaining
80 - k repetitions (the S1 split seeds 4000 + 97k + j, 20 splits), cosine, scored with the S1 estimators (s1_averaging imported unchanged,
v6 = locked soft12 grid): 200-way InfoNCE (per-query LOO tau) and the within-category N-way exemplar score (LOO N-way tau, matched queries).
Descriptive rows: 'pca' (steps 1-2, no whitening) and 'raw' (= v1: raw cosine, per-feature z with the across-image statistics of the 80-rep
test means; labelled secondary). k in {1, 4, 16, 40}. Per construction: top1, median tau, n_q_lower_edge, max_query_CE_bits (REVIEWER S1 ruling).
Summary: decoder-space ceilings from results/ext/s1_v6 (not v4: LOO-tau artefact) at the same k, decoder / EEG ratio per participant and as a
ratio of participant means with a participant-bootstrap CI (GPU); decoder ceilings at the locked tau floor with top-1 >= 0.99 are labelled
"saturated at the locked tau floor" with gap = log2 N - value. Output results/ext/i1_v2/{sub-XX.json, summary.json} with code_sha256."""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path("/path/to/eegaudit"); sys.path.insert(0, str(ROOT / "experiments/ext"))
import s1_averaging as S1
CODE_SHA256 = hashlib.sha256(open(__file__, "rb").read()).hexdigest(); KI = [1, 4, 16, 40]; NPC = 256
D = ROOT / "data_raw/things_eeg2_250hz"


def load_train(sub):
    c = D / f"cache_f32/{sub}_train.npy"
    if c.exists(): return np.load(c, mmap_mode="r")
    obj = np.load(D / f"Preprocessed_data_250Hz/{sub}/preprocessed_eeg_training.npy", allow_pickle=True)              # as run.load_dict
    return (obj.item() if isinstance(obj, np.ndarray) and obj.dtype == object else obj)["preprocessed_eeg_data"]


def fit_basis(sub, ch=1000):
    """Training-only z-score, PCA-256 and noise-whitening. Returns (mu, sd, V (F, 256), W (256, 256), info)."""
    X = load_train(sub); n = X.shape[0]; assert X.shape[1:] == (4, 63, 250), X.shape
    M = torch.empty((n, 63 * 250), dtype=torch.float32, device=S1.DEV)
    for i in range(0, n, ch):
        M[i:i + ch] = torch.as_tensor(np.ascontiguousarray(X[i:i + ch]), dtype=torch.float32, device=S1.DEV).reshape(-1, 4, 63 * 250).mean(1)
    mu = sum(M[i:i + ch].double().sum(0) for i in range(0, n, ch)) / n
    sd = (sum((M[i:i + ch].double() - mu).pow(2).sum(0) for i in range(0, n, ch)) / (n - 1)).sqrt() + 1e-6
    for i in range(0, n, ch): M[i:i + ch] = ((M[i:i + ch].double() - mu) / sd).float()     # M becomes Z in place
    Z = M; del M
    torch.manual_seed(0); U_, s_, V = torch.svd_lowrank(Z, q=300, niter=6)             # Z is centred (z-scored with the training means)
    tot = float(sum(torch.sum(Z[i:i + ch] ** 2, dtype=torch.float64) for i in range(0, n, ch))); ev = (s_.double() ** 2)[:NPC]; V = V[:, :NPC].double(); del Z, U_
    torch.cuda.empty_cache()
    C = torch.zeros((NPC, NPC), dtype=torch.float64, device=S1.DEV)
    for i in range(0, n, 250):
        x = torch.as_tensor(np.ascontiguousarray(X[i:i + 250]), dtype=torch.float64, device=S1.DEV).reshape(-1, 4, 63 * 250)
        P = ((x - x.mean(1, keepdim=True)) / sd) @ V; P = P.reshape(-1, NPC); C += P.T @ P
    C /= n * 3; lam, Q = torch.linalg.eigh(C); lam = lam.clamp_min(0) + 1e-6 * float(lam.mean())
    W = Q @ torch.diag(lam.rsqrt()) @ Q.T
    info = dict(n_train_images=int(n), pca_explained_var_frac=float(ev.sum() / tot), noise_eig_max_over_min=float(lam.max() / lam.min()),
                noise_cov_trace=float(lam.sum()), train_source=str(D / f"cache_f32/{sub}_train.npy") if (D / f"cache_f32/{sub}_train.npy").exists() else "Preprocessed_data_250Hz pickle")
    return mu, sd, V, W, info


def diag200(Sc):
    """S1.info200 plus the REVIEWER v6 per-construction fields."""
    y, top, tq = S1.info200(Sc); CE, ti = S1.loo_ce(Sc); per = torch.gather(CE, 1, ti[:, None])[:, 0]; T = S1.taus()
    return dict(value=float(y.mean()), top1=float(top.mean()), tau_median=float(tq.median()), n_q_lower_edge=int((ti == 0).sum()), n_q=int(ti.numel()),
                max_query_CE_bits=float(per.max()), tau_grid_min=float(T.min()))


def subject(s, out):
    t0 = time.time(); sub = f"sub-{s:02d}"; mu, sd, V, W, info = fit_basis(sub); tfit = time.time() - t0
    X = np.load(D / f"test_cache/{sub}_test.npy", mmap_mode="r")
    Xg = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32).reshape(200, 80, -1)          # (200, 80, 15750) on CPU (shared GPU)
    Zp = torch.stack([((Xg[i].to(S1.DEV).double() - mu) / sd) @ V for i in range(200)])                            # (200, 80, 256)
    del mu, sd, V; torch.cuda.empty_cache()
    spaces = dict(white=Zp @ W, pca=Zp)
    Mt = Xg.mean(1); tmu, tsd = Mt.mean(0), Mt.std(0) + 1e-6                                                       # v1 raw construction
    te = S1.categories(); N, pool_of, classes, cov = S1.pools_from(te)
    res = dict(participant=sub, code_sha256=CODE_SHA256, s1_code_sha256=S1.CODE_SHA256, tau_grid=S1.taus().tolist(), basis=info, coverage=cov,
               primary="white", labels=dict(white="PRIMARY: training-fit z + PCA-256 + noise whitening", pca="descriptive: training-fit z + PCA-256, no whitening",
                                            raw="secondary (= v1): raw 15,750-dim cosine, z with test across-image statistics (label-free)"), rows={})
    for k in KI:
        sp = [(lambda p: (p[:k], p[k:]))(np.random.default_rng(4000 + 97 * k + j).permutation(80)) for j in range(S1.NSUB)]
        row = {}
        for name in ("white", "pca", "raw"):
            if name == "raw":
                rz = lambda ix: S1.nrm(((Xg[:, torch.as_tensor(ix)].mean(1) - tmu) / tsd).double().to(S1.DEV))
                Sc = torch.stack([rz(a) @ rz(b).T for a, b in sp])
            else:
                Z = spaces[name]; Sc = torch.stack([S1.nrm(Z[:, a].mean(1)) @ S1.nrm(Z[:, b].mean(1)).T for a, b in sp])
            d2 = diag200(Sc); ce = S1.nway(Sc, None, pool_of, N, 50, 70 + k)
            row[name] = dict(ceiling200=d2["value"], ceiling_exemplar=float(ce["matched"].mean()), tau_exemplar=ce["tau"], diag200=d2)
        res["rows"][f"k{k}"] = row
        print(sub, k, {n_: (round(r["ceiling200"], 4), round(r["ceiling_exemplar"], 4), round(r["diag200"]["top1"], 3)) for n_, r in row.items()}, f"{time.time() - t0:.0f}s", flush=True)
    res["runtime_s"] = dict(fit=round(tfit, 1), total=round(time.time() - t0, 1))
    del Xg, Zp, spaces; torch.cuda.empty_cache(); (out / f"{sub}.json").write_text(json.dumps(res, indent=1))


def summarize(out, ref):
    P = [json.loads(f.read_text()) for f in sorted(out.glob("sub-*.json"))]; n = len(P); floor = float(S1.taus().min())
    g = torch.Generator(device=S1.DEV); g.manual_seed(20261121); ix = torch.randint(0, n, (10000, n), generator=g, device=S1.DEV)
    T = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float64, device=S1.DEV); q = lambda b: torch.quantile(b, torch.tensor([.025, .975], dtype=torch.float64, device=S1.DEV)).tolist()
    REF = {p["participant"]: json.loads((ref / f"{p['participant']}.json").read_text()) for p in P if (ref / f"{p['participant']}.json").exists()}
    res = dict(note=__doc__, code_sha256=CODE_SHA256, s1_code_sha256=sorted({p["s1_code_sha256"] for p in P}), decoder_ref=str(ref),
               decoder_ref_code_sha256=sorted({r.get("code_sha256") for r in REF.values()}), n=n, participants=[p["participant"] for p in P],
               n_with_decoder_ref=len(REF), basis={k_: float(np.mean([p["basis"][k_] for p in P])) for k_ in ("pca_explained_var_frac", "noise_eig_max_over_min")}, rows={})
    for k in KI:
        r = {}
        for sp in ("white", "pca", "raw"):
            for c in ("ceiling200", "ceiling_exemplar"):
                E = T([p["rows"][f"k{k}"][sp][c] for p in P]); r[f"eeg_{sp}_{c}"] = dict(mean=float(E.mean()), ci=q(E[ix].mean(1)), per_participant=E.tolist())
            r[f"eeg_{sp}_diag200"] = dict(top1=float(np.mean([p["rows"][f"k{k}"][sp]["diag200"]["top1"] for p in P])),
                                          n_q_lower_edge=int(sum(p["rows"][f"k{k}"][sp]["diag200"]["n_q_lower_edge"] for p in P)),
                                          max_query_CE_bits=float(max(p["rows"][f"k{k}"][sp]["diag200"]["max_query_CE_bits"] for p in P)))
        if len(REF) == n:
            for m in ("nice_ga", "brainflora"):
                for c in ("ceiling200", "ceiling_exemplar"):
                    Dm = T([REF[p["participant"]]["rows"][f"{m}_k{k}"][c] for p in P]); r[f"{m}_{c}"] = float(Dm.mean())
                    if c == "ceiling200":
                        tinfo = [REF[p["participant"]]["rows"][f"{m}_k{k}"].get("tau", {}).get("ceiling200", {}) for p in P]
                        sat = [bool(t.get("median", 1) <= floor * (1 + 1e-9)) for t in tinfo]
                        r[f"{m}_ceiling200_saturation"] = dict(n_median_tau_at_locked_floor=int(sum(sat)), gap_log2N_minus_value=(math.log2(200) - Dm).tolist(),
                                                              label="saturated at the locked tau floor (fraction vs a saturated ceiling = upper bound)" if any(sat) else "not saturated")
                    for sp in ("white", "pca", "raw"):
                        E = T([p["rows"][f"k{k}"][sp][c] for p in P])
                        r[f"{m}_{c}_over_eeg_{sp}"] = dict(ratio_of_means=float(Dm.mean() / E.mean()), ci=q(Dm[ix].mean(1) / E[ix].mean(1)), per_participant=(Dm / E).tolist(),
                                                           n_decoder_above_eeg=int((Dm > E).sum()))
        res["rows"][f"k{k}"] = r
    if len(REF) != n: res["decoder_comparison"] = f"SKIPPED: {len(REF)}/{n} participants have {ref}/sub-XX.json"
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: {a: (b["mean"] if isinstance(b, dict) and "mean" in b else b.get("ratio_of_means") if isinstance(b, dict) and "ratio_of_means" in b else None)
                          for a, b in v.items() if isinstance(b, dict)} for k, v in res["rows"].items()}, indent=1)[:4000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9,10"); ap.add_argument("--out", default=str(ROOT / "results/ext/i1_v2"))
    ap.add_argument("--ref", default=str(ROOT / "results/ext/s1_v6")); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize: summarize(out, Path(a.ref))
    else:
        for s in [int(x) for x in a.subjects.split(",")]:
            if not (out / f"sub-{s:02d}.json").exists(): subject(s, out)
