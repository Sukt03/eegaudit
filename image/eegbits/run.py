"""eegbits: how many linear dimensions (and bits per trial) of EEG does EEG-to-image decoding use?

One subject, one input type per call. Closed-form rank-L bottleneck: standardise -> PCA (one SVD; nested bases) -> ridge to
CLIP ViT-H/14 image embeddings -> 200-way zero-shot evaluation on THINGS-EEG2's held-out test images.

  --input embedding : ATM-S EEG embeddings (1024-D; train 16,540 images x 4 reps, test 200 images, 80-rep averages)
  --input raw       : preprocessed 250 Hz EEG epochs (63 ch x 250 samples = 15,750-D; train 16,540 x 4 reps, test 200 x 80 reps)
                      repetition axis on the test queries: k in --ks (mean of the first k of 80 repetitions)

Controls, every L: noise (N(0,1) input, train+test), trial_shuffled (train EEG rows permuted against targets), shuffled_pair
(EEG predictions scored against permuted targets). Metrics: top-1, top-5, 2-way identification, MRR, matched cosine, and
bits per query by the Wolpaw information-transfer formula on top-1 among N=200.
Primary metric: frac_at_L50 = chance-adjusted 2-way at L=50 / chance-adjusted 2-way at full width (EEG, k = all).

Usage:
  python run.py --input embedding --subject sub-01 --out /abs/results/eegbits/embedding_sub-01.json
  python run.py --input raw --subject sub-01 --slice 2000 --out /abs/results/eegbits/smoke_raw_sub-01.json
"""
import argparse, json, subprocess, time
from pathlib import Path
import numpy as np

EMB = Path("/path/to/atm/data/hf_probe/emb_eeg")
CLIP = Path("/path/to/atm/data/hf_probe")
RAW = Path("/path/to/eegaudit/data_raw/things_eeg2_250hz/Preprocessed_data_250Hz")
N_TEST = 200
ALPHAS = [1e-1, 1, 10, 100, 1000, 1e4, 1e5, 1e6, 1e7]


def load_torch(p):
    import torch
    x = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(x, dict):
        for k in ("img_features", "image_features", "features"):
            if k in x:
                x = x[k]
                break
        else:
            x = next(v for v in x.values() if hasattr(v, "shape"))
    return x.float().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=np.float32)


def load_raw(subject):
    """THINGS-EEG2 preprocessed (Gifford et al. 2022 / ATM repo format): dict with 'preprocessed_eeg_data'
    of shape (images, repetitions, channels, times)."""
    d = RAW / subject
    def load_dict(p):
        obj = np.load(p, allow_pickle=True)
        return obj.item() if isinstance(obj, np.ndarray) and obj.dtype == object else obj
    tr, te = load_dict(d / "preprocessed_eeg_training.npy"), load_dict(d / "preprocessed_eeg_test.npy")
    Xtr, Xte = np.asarray(tr["preprocessed_eeg_data"], dtype=np.float32), np.asarray(te["preprocessed_eeg_data"], dtype=np.float32)
    meta = dict(train_shape=list(Xtr.shape), test_shape=list(Xte.shape), ch_names=list(tr.get("ch_names", [])), times=[float(tr["times"][0]), float(tr["times"][-1])] if "times" in tr else None)
    return Xtr, Xte, meta


def metrics(pred, tgt):
    p = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    t = tgt / np.linalg.norm(tgt, axis=1, keepdims=True)
    S = p @ t.T
    n = S.shape[0]
    diag = np.diag(S)
    rank = (S > diag[:, None]).sum(1) + 1
    top1 = float((rank == 1).mean())
    out = dict(top1=top1, top5=float((rank <= 5).mean()), two_way=float(((S < diag[:, None]).sum(1) / (n - 1)).mean()),
               mrr=float((1.0 / rank).mean()), cosine=float(diag.mean()), bits=wolpaw_bits(top1, n))
    try:                                       # methodology change 2 (2026-11-20): every yield also under the InfoNCE soft bound
        out["soft_bits"] = soft_bits(S)
    except Exception:
        pass
    return out


_TAUS = np.logspace(-2.5, 0, 12)


def soft_bits(S):
    """InfoNCE / Barber-Agakov lower bound on I(EEG; candidate): log2 N minus the leave-one-query-out cross-entropy of a softmax
    over the cosine row, temperature chosen per held-out query on the other queries (protocol audit A estimator). 0 at chance."""
    n = S.shape[0]; rows = np.arange(n)
    def ce_all(tau):
        Z = S / tau; Z = Z - Z.max(1, keepdims=True)
        lp = Z - np.log(np.exp(Z).sum(1, keepdims=True))
        return -lp[rows, rows] / np.log(2)
    CE = np.stack([ce_all(t) for t in _TAUS]); tot = CE.sum(1, keepdims=True)
    loo = (tot - CE) / (n - 1); t_idx = loo.argmin(0)
    return float(np.log2(n) - CE[t_idx, rows].mean())


def wolpaw_bits(P, N):
    """Information transferred per selection among N equiprobable candidates with accuracy P (Wolpaw et al. 2002)."""
    P = min(max(P, 1e-9), 1 - 1e-9)
    return float(np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1)))


def _gpu():
    try:
        import torch
        return torch.device("cuda") if torch.cuda.is_available() else None
    except Exception:
        return None


def pca_scores(Xtr, Xte_list):
    """Standardise, one SVD; return nested PCA scores for train and test (columns ordered by variance).
    Runs on the GPU when one is visible (same algebra: Gram-matrix eigendecomposition for wide matrices)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ztr = (Xtr - mu) / sd
    dev = _gpu()
    if dev is not None:
        import torch
        Zt = torch.from_numpy(np.ascontiguousarray(Ztr)).to(dev)
        if Zt.shape[1] <= Zt.shape[0]:
            U, S, Vh = torch.linalg.svd(Zt, full_matrices=False)
            P = Vh.T
        else:
            G = (Zt @ Zt.T).double()
            w, U = torch.linalg.eigh(G)
            w, U = torch.flip(w, [0]).clamp_min(1e-12), torch.flip(U, [1])
            S = w.sqrt()
            P = ((U.T.float() @ Zt) / S.float()[:, None]).T
        out_tr = (Zt @ P).cpu().numpy()
        out_te = [((torch.from_numpy(np.ascontiguousarray((X - mu) / sd)).to(dev)) @ P).cpu().numpy() for X in Xte_list]
        Sv = S.float().cpu().numpy()
        del Zt, P; torch.cuda.empty_cache()
        return out_tr, out_te, Sv
    if Ztr.shape[1] <= Ztr.shape[0]:
        U, S, Vt = np.linalg.svd(Ztr, full_matrices=False)
    else:  # wide matrix: eigen-decompose the Gram matrix, which is smaller
        G = Ztr @ Ztr.T
        w, U = np.linalg.eigh(G)
        order = np.argsort(w)[::-1]
        w, U = np.clip(w[order], 1e-12, None), U[:, order]
        S = np.sqrt(w)
        Vt = (U.T @ Ztr) / S[:, None]
    P = Vt.T
    return Ztr @ P, [((X - mu) / sd) @ P for X in Xte_list], S


def ridge_predict(Ztr, Ytr, Zte, alpha):
    ym = Ytr.mean(0)
    dev = _gpu()
    if dev is not None and Ztr.shape[1] >= 256:
        import torch
        Zt = torch.from_numpy(np.ascontiguousarray(Ztr)).to(dev); Yt = torch.from_numpy(np.ascontiguousarray(Ytr - ym)).to(dev)
        A = Zt.T @ Zt + alpha * torch.eye(Zt.shape[1], device=dev)
        W = torch.linalg.solve(A, Zt.T @ Yt)
        return (torch.from_numpy(np.ascontiguousarray(Zte)).to(dev) @ W).cpu().numpy() + ym
    A = Ztr.T @ Ztr + alpha * np.eye(Ztr.shape[1], dtype=Ztr.dtype)
    W = np.linalg.solve(A, Ztr.T @ (Ytr - ym))
    return Zte @ W + ym


def pick_alpha(Ztr, Ytr, rng):
    idx = rng.permutation(len(Ztr)); cut = int(0.9 * len(idx))
    tr, va = idx[:cut], idx[cut:]
    return max(ALPHAS, key=lambda a: metrics(ridge_predict(Ztr[tr], Ytr[tr], Ztr[va], a), Ytr[va])["mrr"])


def git_commit():
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or "no-git"
    except Exception:
        return "no-git"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=["embedding", "raw", "randenc", "trainedenc"], required=True,
                    help="randenc / trainedenc: the NICE-GA encoder at random initialisation / trained (seed 141), applied to raw EEG, as the probe input (encoder-vs-EEG control)")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--Ls", default="2,5,10,25,50,100,250,1024")
    ap.add_argument("--ks", default="1,2,4,8,16,80", help="repetitions averaged for test queries (raw input only)")
    ap.add_argument("--seeds", default="0,1,2", help="seeds for the stochastic controls")
    ap.add_argument("--slice", type=int, default=0, help="use only the first N training images (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    LS = [int(x) for x in a.Ls.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    Ytr, Yte = load_torch(CLIP / "ViT-H-14_features_train.pt"), load_torch(CLIP / "ViT-H-14_features_test.pt")
    meta = {}
    if a.input == "embedding":
        Xtr = load_torch(EMB / f"ATM_S_eeg_features_{a.subject}_train.pt").reshape(len(Ytr), 4, -1).mean(1)
        Xte_by_k = {"80": load_torch(EMB / f"ATM_S_eeg_features_{a.subject}_test.pt")}
    elif a.input in ("randenc", "trainedenc"):
        # Encoder-vs-EEG control: the same bottleneck + ridge, but the input is the 768-d embedding of a NICE-GA encoder that is either
        # randomly initialised (seeded; nothing learned from EEG) or trained (seed 141 checkpoint). Both see the same raw EEG.
        import torch, run_acq_nice as N
        Xtr4, Xte80, meta = load_raw(a.subject)
        dev = torch.device("cuda"); torch.manual_seed(a.seed)
        Enc, Proj = N.load_classes("nice_ga"); enc, proj = Enc().to(dev).eval(), Proj().to(dev).eval()
        if a.input == "trainedenc":
            tag = f"nice_ga_dgx_v2_{a.subject.replace('-', '')}_seed141"
            enc.load_state_dict(torch.load(N.NICE / "model" / f"{tag}_Enc_eeg_cls.pth", map_location=dev), strict=True)
            proj.load_state_dict(torch.load(N.NICE / "model" / f"{tag}_Proj_eeg_cls.pth", map_location=dev), strict=True)
        @torch.no_grad()
        def _embed(x):
            out = []
            for b in range(0, len(x), 500):
                out.append(proj(enc(torch.from_numpy(np.ascontiguousarray(x[b:b + 500])).unsqueeze(1).to(dev))).float().cpu().numpy())
            return np.concatenate(out)
        Xtr = _embed(Xtr4.mean(1)); Xte_by_k = {"80": _embed(Xte80.mean(1))}
        meta["encoder"] = a.input; del Xtr4, Xte80
    else:
        Xtr4, Xte80, meta = load_raw(a.subject)
        Xtr = Xtr4.mean(1).reshape(Xtr4.shape[0], -1)
        ks = [int(k) for k in a.ks.split(",")]
        Xte_by_k = {str(k): Xte80[:, :k].mean(1).reshape(N_TEST, -1) for k in ks}
        del Xtr4, Xte80
    if a.slice:
        Xtr, Ytr = Xtr[: a.slice], Ytr[: a.slice]
    D = Xtr.shape[1]
    LS = [L for L in LS if L <= min(D, len(Xtr))] + ([min(D, len(Xtr))] if min(D, len(Xtr)) not in LS and min(D, len(Xtr)) < max(LS) else [])
    LS = sorted(set(LS))
    kkeys = list(Xte_by_k)
    Ztr, Zte_list, svals = pca_scores(Xtr, [Xte_by_k[k] for k in kkeys])
    Zte_by_k = dict(zip(kkeys, Zte_list))
    noise = {}
    for s in seeds:
        r = np.random.default_rng(100 + s)
        ntr, nte, _ = pca_scores(r.standard_normal(Xtr.shape).astype(np.float32), [r.standard_normal(Xte_by_k[kkeys[-1]].shape).astype(np.float32)])
        noise[s] = (ntr, nte[0])
    rng = np.random.default_rng(a.seed)
    rows = []
    for L in LS:
        alpha = pick_alpha(Ztr[:, :L], Ytr, rng)
        for k in kkeys:
            pred = ridge_predict(Ztr[:, :L], Ytr, Zte_by_k[k][:, :L], alpha)
            rows.append(dict(L=L, k=int(k), condition="eeg", seed=0, alpha=alpha, **metrics(pred, Yte)))
            if k == kkeys[-1]:
                for s in seeds:
                    r = np.random.default_rng(200 + s)
                    ntr, nte = noise[s]
                    rows.append(dict(L=L, k=int(k), condition="noise", seed=s, alpha=alpha, **metrics(ridge_predict(ntr[:, :L], Ytr, nte[:, :L], alpha), Yte)))
                    rows.append(dict(L=L, k=int(k), condition="trial_shuffled", seed=s, alpha=alpha, **metrics(ridge_predict(Ztr[r.permutation(len(Ztr)), :L], Ytr, Zte_by_k[k][:, :L], alpha), Yte)))
                    rows.append(dict(L=L, k=int(k), condition="shuffled_pair", seed=s, alpha=alpha, **metrics(pred, Yte[r.permutation(len(Yte))])))
    full = max(LS); kfull = kkeys[-1]
    eeg = {(r["L"], r["k"]): r for r in rows if r["condition"] == "eeg"}
    adj = lambda L: eeg[(L, int(kfull))]["two_way"] - 0.5
    frac = {str(L): float(adj(L) / adj(full)) if adj(full) > 0 else None for L in LS}
    L90 = next((L for L in LS if frac[str(L)] is not None and frac[str(L)] >= 0.9), None)
    ctrl_max_top1 = {str(L): max(r["top1"] for r in rows if r["L"] == L and r["condition"] in ("noise", "trial_shuffled")) for L in LS}
    Lstar = next((L for L in LS if eeg[(L, int(kfull))]["top1"] - ctrl_max_top1[str(L)] > 0.02), None)
    out = dict(experiment="eegbits", input=a.input, subject=a.subject, n_train=int(len(Xtr)), n_test=N_TEST, input_dim=int(D), slice=a.slice,
               Ls=LS, ks=[int(k) for k in kkeys], seeds=seeds, alphas=ALPHAS, split="THINGS-EEG2 official: 16,540 training images (4 reps) / 200 held-out test images (80 reps); no test data used for any selection",
               primary_metric="frac_at_L50", frac_at_L50=frac.get("50"), frac_by_L=frac, L90=L90, Lstar_top1_vs_controls=Lstar,
               full_width_eeg=eeg[(full, int(kfull))], rows=rows, raw_meta=meta, explained_var_top50=float((svals[:50] ** 2).sum() / (svals ** 2).sum()),
               git=git_commit(), runtime_s=round(time.time() - t0, 1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"{a.input} {a.subject} n_train={len(Xtr)} D={D}  frac_at_L50={out['frac_at_L50']}  L90={L90}  L*={Lstar}  full-width top1={eeg[(full, int(kfull))]['top1']:.3f} 2way={eeg[(full, int(kfull))]['two_way']:.3f} bits={eeg[(full, int(kfull))]['bits']:.2f}  ({out['runtime_s']}s) -> {a.out}")


if __name__ == "__main__":
    main()
