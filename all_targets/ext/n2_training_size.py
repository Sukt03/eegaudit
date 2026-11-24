"""N2 YIELD vs TRAINING-SET SIZE (rung R9). EXECUTOR-2, 21 Nov 2026. Design = "## NEW IDEAS REVIEW" N2 (revised), cheap arms first.

Arm image_ridge (THINGS-EEG2, 10): run.py raw ridge path (63 ch x 250 samples, 4-rep training means, standardise -> PCA -> ridge -> CLIP
  ViT-H/14), k = 80 queries, 200-way InfoNCE (run.soft_bits). Training IMAGES subsampled uniformly WITHIN concept (all 1,654 concepts kept;
  10 images per concept; floor(10p) per concept plus one more for a random subset of concepts so the total is round(16,540 p)).
  Fractions 10 / 25 % (seeds 0, 1, 2), 50 % (seed 0), 100 %. PCA basis, alpha (run.pick_alpha, rng 0) re-fit on each subset;
  L = min(1024, n_train - 1). Test set and 200 candidates fixed. Floor per point: noise input (seed 100) and shuffled pairs (seed 200).
  G2: the 100 % point with run.py's alpha sequence reproduces results/eegbits/raw_sub-XX.json (L=1024, k=80: alpha, top1, two_way);
  the curve uses the 100 % point with a fresh rng-0 alpha pick (same rule as the subsets).
Arm pe_trials (ds003458, 23): X7 raw-ERP ridge, locked log-score (nested calibration, m3_yield_map.pe_gain logic) trained on 25 / 50 / 75 /
  100 % of every outer training fold (random rows within the fold, seeds 0, 1, 2; calibration inner 9-fold CV inside the subset); test folds
  fixed. Floor = target circularly shifted by n/4, n/2, 3n/4. G2: 100 % = results/standard/pe_bottleneck/sub-XXX.json full gain exactly.
Deferred to a second G1 (heavier, new code paths): NICE-GA retrained on fractions (retrain_nice_ga_variant `frac<p>s<seed>`) and the
  EEGNet LOPO participant-count arm (5 / 11 / 22 training participants).
Statistic (--summarize): participant-level curve (mean of seeds within participant), bootstrap CI; paired difference between the two largest
  sizes; saturating fit y = a - b n^-c on the participant-mean curve (least squares, c in (0, 3]) with a 2,000-resample participant
  bootstrap CI for the asymptote a (extrapolation, labelled). Reference line = the target's registered full-data ceiling
  (image: StdImgCeil = results/standard/summary.json image.ceiling.mean, NICE-GA split-half; PE: StdPE ceiling_ds003458.mean, EEGNet split-half).
  "Plateau below ceiling" only if the CI of a lies below the ceiling AND the last-step paired-difference CI includes 0.
GPU (PLAN §10): ridge and alpha selection run as torch on the GPU (gpu_pick / gpu_pred = run.pick_alpha / run.ridge_predict twins);
  PCA via run.pca_scores (GPU).
Usage: python n2_training_size.py --arm image_ridge|pe_trials [--subjects 1] --out DIR ; --summarize --arm ...
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy import optimize

ROOT = Path("/path/to/eegaudit")
for p in (ROOT, ROOT / "experiments/ext", ROOT / "experiments/eegbits", ROOT / "experiments/agent_loop_20261120", ROOT / "experiments/standard"):
    sys.path.insert(0, str(p))
IMG_PTS = [(0.1, 0), (0.1, 1), (0.1, 2), (0.25, 0), (0.25, 1), (0.25, 2), (0.5, 0), (1.0, 0)]
PE_PTS = [(f, s) for f in (0.25, 0.5, 0.75) for s in (0, 1, 2)] + [(1.0, 0)]


def gpu_pick(Z, Y, rng):
    """GPU twin of run.pick_alpha (same 90/10 rng split, max dev MRR over run.ALPHAS, first maximum): m3_yield_map.fast_pick_alpha."""
    import torch, run as R
    from m3_yield_map import fast_pick_alpha, DEV
    return fast_pick_alpha(torch.as_tensor(Z, dtype=torch.float64, device=DEV), torch.as_tensor(Y, dtype=torch.float64, device=DEV), rng, R)


def gpu_pred(Ztr, Ytr, Zte, a):
    """GPU twin of run.ridge_predict (same algebra, float64)."""
    from m3_yield_map import gpu_ridge
    return gpu_ridge(Ztr, Ytr, Zte, [a])[0].cpu().numpy()


def concept_subset(frac, seed, n_concepts=1654, per=10):
    if frac >= 1: return np.arange(n_concepts * per)
    rng = np.random.default_rng(seed); total = round(n_concepts * per * frac); base = int(np.floor(per * frac))
    k = np.full(n_concepts, base); k[rng.choice(n_concepts, total - base * n_concepts, replace=False)] += 1
    return np.sort(np.concatenate([c * per + rng.choice(per, k[c], replace=False) for c in range(n_concepts)]))


def image_subject(s, out):
    import run as R
    t0 = time.time(); sub = f"sub-{s:02d}"
    Ytr, Yte = R.load_torch(R.CLIP / "ViT-H-14_features_train.pt"), R.load_torch(R.CLIP / "ViT-H-14_features_test.pt"); assert len(Ytr) == 16540
    Xtr4, Xte80, _ = R.load_raw(sub); Xall = Xtr4.mean(1).reshape(len(Xtr4), -1); Xte = Xte80.mean(1).reshape(200, -1); del Xtr4, Xte80
    rows, g2 = [], {}
    for frac, seed in IMG_PTS:
        idx = concept_subset(frac, seed); Xtr, Yt = Xall[idx], Ytr[idx]
        Ztr, (Zte,), _ = R.pca_scores(Xtr, [Xte]); L = min(1024, len(idx) - 1)
        if frac == 1.0:
            rng = np.random.default_rng(0)
            for Lq in [2, 5, 10, 25, 50, 100, 250, 1024]: aq = gpu_pick(Ztr[:, :Lq], Yt, rng)
            m = R.metrics(gpu_pred(Ztr[:, :1024], Yt, Zte[:, :1024], aq), Yte)
            old = json.loads((ROOT / f"results/eegbits/raw_{sub}.json").read_text()); o = next(r for r in old["rows"] if r["L"] == 1024 and r["k"] == 80 and r["condition"] == "eeg")
            g2 = dict(old={k: o[k] for k in ("alpha", "top1", "two_way")}, new=dict(alpha=aq, top1=m["top1"], two_way=m["two_way"], soft_bits=m["soft_bits"]),
                      alpha_equal=bool(o["alpha"] == aq), pass_=bool(abs(o["top1"] - m["top1"]) < 1e-9 and abs(o["two_way"] - m["two_way"]) < 1e-3))
        a = gpu_pick(Ztr[:, :L], Yt, np.random.default_rng(0)); pred = gpu_pred(Ztr[:, :L], Yt, Zte[:, :L], a); m = R.metrics(pred, Yte)
        r = np.random.default_rng(100); ntr, (nte,), _ = R.pca_scores(r.standard_normal(Xtr.shape).astype(np.float32), [r.standard_normal(Xte.shape).astype(np.float32)])
        nz = R.metrics(gpu_pred(ntr[:, :L], Yt, nte[:, :L], a), Yte)["soft_bits"]; sh = R.metrics(pred, Yte[np.random.default_rng(200).permutation(200)])["soft_bits"]
        rows.append(dict(frac=frac, seed=seed, n_train=int(len(idx)), n_concepts=int(len(np.unique(idx // 10))), L=int(L), alpha=a, soft_bits=m["soft_bits"], top1=m["top1"],
                         noise=nz, shuffled=sh, floor=max(nz, sh)))
        print(sub, rows[-1], f"{time.time()-t0:.0f}s", flush=True)
    (out / f"image_ridge_{sub}.json").write_text(json.dumps(dict(arm="image_ridge", participant=sub, rows=rows, g2=g2, runtime_s=round(time.time() - t0, 1)), indent=1))
    print(sub, "G2", g2, flush=True)


def pe_subject(s, out, evaluate, fpath):
    import torch
    from belief_bottleneck import LAM, folds
    from pe_bottleneck_logscore import gain_rows
    DEV = "cuda"; sub = f"sub-{s:03d}"; fz = np.load(fpath("raw", s)); tri = fz["tri"]; n = len(tri)
    y = evaluate(s, tri).astype(np.float64); Ys = np.column_stack([y] + [np.roll(y, round(n * k / 4)) for k in (1, 2, 3)])
    Ft = torch.tensor(fz["F"].astype(np.float64), device=DEV); cw = torch.tensor(fz["colw"].astype(np.float64), device=DEV); Yt = torch.tensor(Ys, device=DEV); fo = folds(n)
    def pred(tr, te):
        tr, te = torch.tensor(tr, device=DEV), torch.tensor(te, device=DEV)
        A, B = Ft[tr], Ft[te]; mu, sd = A.mean(0), A.std(0, unbiased=False) + 1e-6; A = (A - mu) / sd * cw; B = (B - mu) / sd * cw
        W = torch.linalg.solve(A.T @ A + LAM * len(A) * torch.eye(A.shape[1], dtype=torch.float64, device=DEV), A.T @ (Yt[tr] - Yt[tr].mean(0)))
        return (B @ W).cpu().numpy()
    rows = []
    for frac, seed in PE_PTS:
        rng = np.random.default_rng(seed); sc = np.full((n, 4), np.nan)
        for b in range(10):
            tr = np.where(fo != b)[0]; te = np.where(fo == b)[0]
            if frac < 1: tr = np.sort(rng.choice(tr, round(frac * len(tr)), replace=False))
            P = pred(tr, te); inner = np.zeros((len(tr), 4)); fi = folds(len(tr), 9)
            for c in range(9): inner[fi == c] = pred(tr[fi != c], tr[fi == c])
            for j in range(4): sc[te, j] = gain_rows(Ys[tr, j], inner[:, j], Ys[te, j], P[:, j])
        rows.append(dict(frac=frac, seed=seed, gain=float(sc[:, 0].mean()), shift=float(sc[:, 1:].mean())))
    old = json.loads((ROOT / f"results/standard/pe_bottleneck/{sub}.json").read_text()); o = next(r for r in old["rows"] if r["input"] == "eeg" and r["mode"] == "pca" and r["L"] == "full")
    g2 = dict(old=o["gain"], new=rows[-1]["gain"], old_shift=float(np.mean(o["shifted"])), new_shift=rows[-1]["shift"]); g2["pass_"] = bool(abs(g2["old"] - g2["new"]) < 1e-9 and abs(g2["old_shift"] - g2["new_shift"]) < 1e-9)
    (out / f"pe_trials_{sub}.json").write_text(json.dumps(dict(arm="pe_trials", participant=sub, n_trials=int(n), rows=rows, g2=g2), indent=1))
    print(sub, [(r["frac"], r["seed"], round(r["gain"], 4)) for r in rows], "G2", g2, flush=True)


def summarize(arm, out):
    per = [json.loads(f.read_text()) for f in sorted(out.glob(f"{arm}_sub-*.json"))]; n = len(per)
    yk, fk = ("soft_bits", "floor") if arm == "image_ridge" else ("gain", "shift")
    fracs = sorted({r["frac"] for r in per[0]["rows"]})
    Y = np.array([[np.mean([r[yk] for r in p["rows"] if r["frac"] == f]) for f in fracs] for p in per])
    Fl = np.array([[np.mean([r[fk] for r in p["rows"] if r["frac"] == f]) for f in fracs] for p in per])
    rng = np.random.default_rng(20261121); ix = rng.integers(0, n, (10000, n))
    ci = lambda v: np.quantile(v[ix].mean(1), [.025, .975]).tolist()
    nfull = 16540 if arm == "image_ridge" else 1.0
    def fit(m):
        x = np.array(fracs) * nfull
        try:
            p, _ = optimize.curve_fit(lambda x, a, b, c: a - b * x ** (-c), x, m, p0=[m[-1], (m[-1] - m[0]) * x[0] ** 0.5, 0.5], bounds=([-10, -1e6, 1e-3], [10, 1e6, 3]), maxfev=20000)
            return p.tolist()
        except Exception:
            return [np.nan] * 3
    a_hat = fit(Y.mean(0)); ab = np.array([fit(Y[rng.integers(0, n, n)].mean(0))[0] for _ in range(2000)]); ab = ab[np.isfinite(ab)]
    reg = json.loads((ROOT / "results/standard/summary.json").read_text())
    ceiling = reg["image"]["ceiling"]["mean"] if arm == "image_ridge" else reg["prediction_error"]["ceiling_ds003458"]["mean"]
    d = Y[:, -1] - Y[:, -2]; a_ci = np.quantile(ab, [.025, .975]).tolist() if len(ab) else [np.nan, np.nan]
    res = dict(arm=arm, n=n, participants=[p["participant"] for p in per], g2_all_pass=all(p["g2"]["pass_"] for p in per), fractions=fracs,
               curve=[dict(frac=f, mean=float(Y[:, i].mean()), ci=ci(Y[:, i]), floor=float(Fl[:, i].mean()), excess_ci=ci(Y[:, i] - Fl[:, i])) for i, f in enumerate(fracs)],
               last_step=dict(fracs=fracs[-2:], diff_mean=float(d.mean()), diff_ci=ci(d), n_positive=int((d > 0).sum())),
               saturating_fit=dict(a=a_hat[0], b=a_hat[1], c=a_hat[2], a_ci=a_ci, n_boot_ok=int(len(ab)), x_unit="training images" if arm == "image_ridge" else "fraction of training fold",
                                   label="extrapolation"),
               ceiling_reference=dict(value=ceiling, source="results/standard/summary.json " + ("image.ceiling.mean (NICE-GA split-half)" if arm == "image_ridge" else "prediction_error.ceiling_ds003458.mean (EEGNet split-half)")))
    res["plateau_below_ceiling"] = bool(a_ci[1] < ceiling and d.size and ci(d)[0] <= 0 <= ci(d)[1])
    (out / f"summary_{arm}.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True, choices=["image_ridge", "pe_trials"]); ap.add_argument("--subjects", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize: return summarize(a.arm, out)
    if a.arm == "image_ridge":
        for s in ([int(x) for x in a.subjects.split(",")] if a.subjects else range(1, 11)):
            if not (out / f"image_ridge_sub-{s:02d}.json").exists(): image_subject(s, out)
    else:
        from i2_complementarity import targets
        evaluate, subs, fpath, _ = targets()
        for s in ([int(x) for x in a.subjects.split(",")] if a.subjects else subs):
            if not (out / f"pe_trials_sub-{s:03d}.json").exists(): pe_subject(s, out, evaluate, fpath)


if __name__ == "__main__":
    main()
