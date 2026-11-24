"""eegbits budget: price EEG-to-image decoding in real bits of EEG and in acquisition cost.

Builds on run.py (same data, same closed-form PCA-L + ridge decoder, same 200-way held-out test, same controls) and adds:
  (1) quantised bottleneck: each of the L PCA dimensions is uniformly quantised to b bits (range = train 0.1..99.9 percentiles)
      before ridge; budget = L*b bits per trial. Grid: L in --Ls, b in --bits plus float32.
  (2) raw input only, acquisition budget: electrode subsets (occipito-parietal 8/16/32, all 63, and size-matched random
      subsets) x post-stimulus windows (0-0.2, 0-0.4, 0-0.6, 0-1.0 s), at float32 bottleneck width --acq-L, for k = 1 and k = all.
Output JSON: rows with condition, L, bits_per_dim, budget_bits, channels, window, k, metrics (top1, top5, two_way, mrr, bits_out).
Usage: python run_budget.py --input raw --subject sub-01 --out /abs/results/eegbits/budget_raw_sub-01.json [--slice N]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run import CLIP, EMB, N_TEST, ALPHAS, load_torch, load_raw, metrics, pca_scores, ridge_predict, pick_alpha, git_commit

OCC_PAR = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
           "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "P9", "P10", "TP9", "TP10", "C5", "C3"]
SR = 250.0


def quantise(Ztr, Zte, b, lo_hi):
    lo, hi = lo_hi
    levels = 2 ** b
    step = (hi - lo) / levels
    q = lambda Z: np.clip(np.floor((Z - lo) / step), 0, levels - 1) * step + lo + step / 2
    return q(Ztr), q(Zte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=["embedding", "raw"], required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--Ls", default="5,10,25,50,100")
    ap.add_argument("--bits", default="1,2,4,8")
    ap.add_argument("--acq-L", type=int, default=50)
    ap.add_argument("--acq-train", type=int, default=5000, help="training images used for the acquisition grid (raw)")
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    LS = [int(x) for x in a.Ls.split(",")]; BITS = [int(x) for x in a.bits.split(",")]
    Ytr, Yte = load_torch(CLIP / "ViT-H-14_features_train.pt"), load_torch(CLIP / "ViT-H-14_features_test.pt")
    rows, meta = [], {}
    if a.input == "embedding":
        Xtr = load_torch(EMB / f"ATM_S_eeg_features_{a.subject}_train.pt").reshape(len(Ytr), 4, -1).mean(1)
        Xte_k = {"80": load_torch(EMB / f"ATM_S_eeg_features_{a.subject}_test.pt")}
        Xtr4 = Xte80 = None
    else:
        Xtr4, Xte80, meta = load_raw(a.subject)
        Xtr = Xtr4.mean(1).reshape(Xtr4.shape[0], -1)
        Xte_k = {"1": Xte80[:, :1].mean(1).reshape(N_TEST, -1), "80": Xte80.mean(1).reshape(N_TEST, -1)}
    if a.slice:
        Xtr, Ytr = Xtr[: a.slice], Ytr[: a.slice]
        if Xtr4 is not None:
            Xtr4 = Xtr4[: a.slice]
    kk = list(Xte_k)
    # (1) quantised bottleneck on the generic PCA directions
    Ztr, Zte_list, _ = pca_scores(Xtr, [Xte_k[k] for k in kk])
    Zte_k = dict(zip(kk, Zte_list))
    for L in LS:
        if L > min(Ztr.shape):
            continue
        alpha = pick_alpha(Ztr[:, :L], Ytr, rng)
        for k in kk:
            rows.append(dict(part="quant", condition="eeg_float32", L=L, bits_per_dim=32, budget_bits=32 * L, channels=None, window=None, k=int(k), alpha=alpha,
                             **metrics(ridge_predict(Ztr[:, :L], Ytr, Zte_k[k][:, :L], alpha), Yte)))
        lo_hi = (np.percentile(Ztr[:, :L], 0.1, axis=0), np.percentile(Ztr[:, :L], 99.9, axis=0))
        for b in BITS:
            qtr, _ = quantise(Ztr[:, :L], Ztr[:, :L], b, lo_hi)
            alpha_b = pick_alpha(qtr, Ytr, rng)
            for k in kk:
                _, qte = quantise(Ztr[:, :L], Zte_k[k][:, :L], b, lo_hi)
                rows.append(dict(part="quant", condition="eeg_quant", L=L, bits_per_dim=b, budget_bits=b * L, channels=None, window=None, k=int(k), alpha=alpha_b,
                                 **metrics(ridge_predict(qtr, Ytr, qte, alpha_b), Yte)))
        # noise control at the smallest and largest budget
        r = np.random.default_rng(100)
        ntr, nte, _ = pca_scores(r.standard_normal(Xtr.shape).astype(np.float32), [r.standard_normal(Xte_k[kk[-1]].shape).astype(np.float32)])
        rows.append(dict(part="quant", condition="noise_float32", L=L, bits_per_dim=32, budget_bits=32 * L, channels=None, window=None, k=int(kk[-1]), alpha=alpha,
                         **metrics(ridge_predict(ntr[:, :L], Ytr, nte[0][:, :L], alpha), Yte)))
    # (1b) informed bottleneck: reduced-rank regression. Full ridge first, then keep the top-L directions of the fitted
    #      targets (the supervised linear bottleneck of Mayo et al. 2024, in closed form). Same for the noise control.
    def rrr_rows(Ztr_full, Zte_by, cond):
        alpha_full = pick_alpha(Ztr_full, Ytr, rng)
        ym = Ytr.mean(0)
        A = Ztr_full.T @ Ztr_full + alpha_full * np.eye(Ztr_full.shape[1], dtype=Ztr_full.dtype)
        W = np.linalg.solve(A, Ztr_full.T @ (Ytr - ym))
        Fhat = Ztr_full @ W
        _, _, Vt = np.linalg.svd(Fhat, full_matrices=False)
        for L in LS + ([max(LS)] if False else []):
            if L > Vt.shape[0]:
                continue
            PL = Vt[:L].T @ Vt[:L]
            for k, Zte in Zte_by.items():
                rows.append(dict(part="rrr", condition=cond, L=L, bits_per_dim=32, budget_bits=32 * L, channels=None, window=None, k=int(k), alpha=alpha_full,
                                 **metrics((Zte @ W) @ PL + ym, Yte)))
    rrr_rows(Ztr, Zte_k, "eeg_rrr")
    rrr_rows(ntr, {kk[-1]: nte[0]}, "noise_rrr")
    # (2) acquisition budget: raw only
    if a.input == "raw":
        ch = meta["ch_names"]; nT = Xtr4.shape[-1]
        t_axis = np.linspace(meta["times"][0], meta["times"][1], nT)
        occ = [i for i, c in enumerate(ch) if c in OCC_PAR]
        subsets = {"occpar8": occ[:8], "occpar16": occ[:16], "occpar32": occ[:32], "all63": list(range(len(ch)))}
        for n in (8, 16, 32):
            subsets[f"random{n}"] = sorted(rng.choice(len(ch), n, replace=False).tolist())
        windows = {"0-0.2s": (0.0, 0.2), "0-0.4s": (0.0, 0.4), "0-0.6s": (0.0, 0.6), "0-1.0s": (0.0, 1.0)}
        # acquisition grid: 28 decompositions per subject, so fit on a fixed random subsample of training images
        sub_idx = np.sort(np.random.default_rng(7).choice(Xtr4.shape[0], min(a.acq_train, Xtr4.shape[0]), replace=False))
        Xtr4, Ytr_acq = Xtr4[sub_idx], Ytr[sub_idx]
        Xtr_avg = Xtr4.mean(1)                       # (n_sub, ch, T)
        Xte_1, Xte_80 = Xte80[:, :1].mean(1), Xte80.mean(1)
        for sname, idx in subsets.items():
            for wname, (t0w, t1w) in windows.items():
                tm = (t_axis >= t0w) & (t_axis < t1w)
                f = lambda X: X[:, idx][:, :, tm].reshape(X.shape[0], -1)
                Ztr_s, (Zte1, Zte80), _ = pca_scores(f(Xtr_avg), [f(Xte_1), f(Xte_80)])
                L = min(a.acq_L, Ztr_s.shape[1])
                alpha = pick_alpha(Ztr_s[:, :L], Ytr_acq, rng)
                for k, Zte in (("1", Zte1), ("80", Zte80)):
                    m = metrics(ridge_predict(Ztr_s[:, :L], Ytr_acq, Zte[:, :L], alpha), Yte)
                    samples = len(idx) * int(tm.sum())
                    rows.append(dict(part="acq", condition="eeg_float32", L=L, bits_per_dim=32, budget_bits=32 * L, channels=sname, n_channels=len(idx),
                                     window=wname, window_s=t1w - t0w, raw_samples=samples, bits_per_second=m["bits"] / (t1w - t0w) / int(k), k=int(k), alpha=alpha, **m))
    out = dict(experiment="eegbits_budget", input=a.input, subject=a.subject, n_train=int(len(Xtr)), slice=a.slice, Ls=LS, bits=BITS, acq_L=a.acq_L, acq_train=a.acq_train,
               split="THINGS-EEG2 official train/test; no test data used for any selection", rows=rows, raw_meta=meta, git=git_commit(), runtime_s=round(time.time() - t0, 1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    q = [r for r in rows if r["part"] == "quant" and r["k"] == int(kk[-1]) and r["condition"] == "eeg_quant"]
    best_small = sorted(q, key=lambda r: r["budget_bits"])
    print(f"{a.input} {a.subject}: {len(rows)} rows in {out['runtime_s']}s; quantised examples: " + "; ".join(f"L{r['L']}x{r['bits_per_dim']}b={r['budget_bits']}bits -> 2way {100*r['two_way']:.1f}%" for r in best_small[:4]) + f" -> {a.out}")


if __name__ == "__main__":
    main()
