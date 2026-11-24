"""N1 ARTEFACT YIELD (rung R6; METHOD CHANGE 5): how much of the yield is reproducible from ocular-proxy channels alone?
EXECUTOR-2, 21 Nov 2026. Design = "## NEW IDEAS REVIEW" N1 (outputs/EXTENSIONS_LOOP_20261121.md). Sets fixed in experiments/ext/regions.py.

Image (THINGS-EEG2, 10): run.py raw ridge path (standardise -> PCA (all components up to 1024) -> ridge -> CLIP ViT-H/14, pick_alpha rng 0),
  k = 80 queries, InfoNCE soft bound (run.soft_bits), on three channel masks x 250 samples (0..996 ms after onset, the base path input; METHOD TIME-AXIS RULING):
  all63 (= run.py raw L = 1024: G2 reproduces results/eegbits/raw_sub-XX.json L=1024 k=80 alpha/top1/two_way), ocular_proxy
  {Fp1, Fp2, AF7, AF8} (frontal-pole proxy: THINGS-EEG2 has no EOG; an UPPER bound on the ocular contribution, since these channels
  also carry brain signal by volume conduction), posterior_control {O1, O2, PO7, PO8} (size-matched). Floors on every mask: noise input
  (3 seeds) and shuffled pairs (3 seeds); floor = max of the two means.
PE (ds003458, 23): X7 raw-ERP ridge (full width, LAM * n, 10 contiguous folds) on three inputs over the same trials:
  all = results/features/raw (64 ch x 23 samples, 150-600 ms, 50 Hz, post-ICA); ocular_proxy = rung0_ocular.ocular() PRE-ICA regressors
  (vertical EOG, Fp1-Fp2, F7-F8 at 150-600 ms / 50 Hz, blink flag; 70 features), the audited rung0 `eog_only` set; posterior_control =
  {Pz, POz, O1, O2} columns of the X7 features (92 features). Two estimators: (i) bits_g null-corrected (q1/X7: r_g = corr(read-out,
  normal-scored target), d = atanh r_g(own) - mean atanh r_g(other participants' targets on the same trials)); (ii) the locked log-score
  (density gain, nested calibration; m3_yield_map.pe_gain) with circular-shift floor. G2: 'all' reproduces results/yield/
  pe_bottleneck_ds003458/raw_sub-XXX.json (L=full eeg d) and results/standard/pe_bottleneck/sub-XXX.json (full gain) when no trial is lost.
Statistic (--summarize): per participant share = (proxy - floor)/(all - floor); participant bootstrap CI (10,000); proxy vs floor
  (one-sided t, Holm within target over {proxy, control}); paired proxy - control CI. Descriptive; no pass rule.
Text: no run (ZuCo pickles carry no ocular channels; gaze arms answer it).
GPU (PLAN §10): ridge and alpha selection run as torch on the GPU (gpu_pick / gpu_pred = run.pick_alpha / run.ridge_predict twins);
  PCA via run.pca_scores (GPU).
Usage: python n1_artefact_yield.py --target image|pe [--subjects 1] --out DIR ; --summarize
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path("/path/to/eegaudit")
for p in (ROOT, ROOT / "experiments/ext", ROOT / "experiments/eegbits", ROOT / "experiments/agent_loop_20261120", ROOT / "experiments/standard"):
    sys.path.insert(0, str(p))
import regions as RG


def gpu_pick(Z, Y, rng):
    """GPU twin of run.pick_alpha (same 90/10 rng split, max dev MRR over run.ALPHAS, first maximum): m3_yield_map.fast_pick_alpha."""
    import torch, run as R
    from m3_yield_map import fast_pick_alpha, DEV
    return fast_pick_alpha(torch.as_tensor(Z, dtype=torch.float64, device=DEV), torch.as_tensor(Y, dtype=torch.float64, device=DEV), rng, R)


def gpu_pred(Ztr, Ytr, Zte, a):
    """GPU twin of run.ridge_predict (same algebra, float64)."""
    from m3_yield_map import gpu_ridge
    return gpu_ridge(Ztr, Ytr, Zte, [a])[0].cpu().numpy()


def image_subject(s, out):
    import run as R
    t0 = time.time(); sub = f"sub-{s:02d}"
    Ytr, Yte = R.load_torch(R.CLIP / "ViT-H-14_features_train.pt"), R.load_torch(R.CLIP / "ViT-H-14_features_test.pt")
    Xtr4, Xte80, meta = R.load_raw(sub); ch = meta["ch_names"]
    masks = {"all63": list(range(len(ch))), **{k: RG.indices(ch, v) for k, v in RG.N1_IMAGE.items()}}
    res = dict(target="image", participant=sub, masks={k: [ch[i] for i in v] for k, v in masks.items()}, rows={})
    for name, idx in masks.items():
        Xtr = Xtr4[:, :, idx].mean(1).reshape(len(Xtr4), -1); Xte = Xte80[:, :, idx].mean(1).reshape(200, -1)
        Ztr, (Zte,), _ = R.pca_scores(Xtr, [Xte]); rng = np.random.default_rng(0)
        if name == "all63":
            for L in [2, 5, 10, 25, 50, 100, 250, 1024]:
                a = gpu_pick(Ztr[:, :L], Ytr, rng)
            L = 1024
        else:
            L = min(Ztr.shape[1], 1024); a = gpu_pick(Ztr[:, :L], Ytr, rng)
        pred = gpu_pred(Ztr[:, :L], Ytr, Zte[:, :L], a); m = R.metrics(pred, Yte)
        nz, sh = [], []
        for k in range(3):
            r = np.random.default_rng(100 + k)
            ntr, (nte,), _ = R.pca_scores(r.standard_normal(Xtr.shape).astype(np.float32), [r.standard_normal(Xte.shape).astype(np.float32)])
            nz.append(R.metrics(gpu_pred(ntr[:, :L], Ytr, nte[:, :L], a), Yte)["soft_bits"])
            sh.append(R.metrics(pred, Yte[np.random.default_rng(200 + k).permutation(200)])["soft_bits"])
        res["rows"][name] = dict(L=int(L), alpha=a, soft_bits=m["soft_bits"], top1=m["top1"], two_way=m["two_way"], mrr=m["mrr"], noise=float(np.mean(nz)),
                                 shuffled=float(np.mean(sh)), floor=float(max(np.mean(nz), np.mean(sh))))
        if name == "all63":
            old = json.loads((ROOT / f"results/eegbits/raw_{sub}.json").read_text())
            o = next(r for r in old["rows"] if r["L"] == 1024 and r["k"] == 80 and r["condition"] == "eeg")
            res["g2"] = dict(file=f"results/eegbits/raw_{sub}.json", old={k: o[k] for k in ("alpha", "top1", "two_way")}, new=dict(alpha=a, top1=m["top1"], two_way=m["two_way"]),
                             alpha_equal=bool(o["alpha"] == a), pass_=bool(abs(o["top1"] - m["top1"]) < 1e-9 and abs(o["two_way"] - m["two_way"]) < 1e-3))
        print(sub, name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["rows"][name].items()}, f"{time.time()-t0:.0f}s", flush=True)
    res["runtime_s"] = round(time.time() - t0, 1)
    (out / f"image_{sub}.json").write_text(json.dumps(res, indent=1))


def pe_subject(s, out, evaluate, subs, fpath):
    import torch
    from belief_bottleneck import LAM, folds
    from m3_yield_map import pe_gain
    os.chdir(ROOT); import rung0_ocular as R0
    t0 = time.time(); sub = f"sub-{s:03d}"; fz = np.load(fpath("raw", s)); tri0 = fz["tri"]; F0 = fz["F"].astype(np.float64)
    ep = np.load(ROOT / f"results/eeg_measures/{sub}_epochs.npz"); chl = [str(c) for c in ep["ch"]]; good = ~ep["bad"]
    win = (ep["times"] >= 0.15) & (ep["times"] <= 0.60); assert np.array_equal(ep["trial"][good], tri0)
    assert np.allclose(ep["X"][good][:, :, win].reshape(len(tri0), -1), F0, atol=1e-3)          # X7 column layout = (channel, sample)
    O, keep, _, _ = R0.ocular(s, tri0); tri = tri0[keep]; F = F0[keep]
    ctrl = F.reshape(len(tri), len(chl), -1)[:, RG.indices(chl, RG.N1_PE_CONTROL)].reshape(len(tri), -1)
    y = evaluate(s, tri).astype(np.float64); n = len(tri)
    Ys = np.column_stack([y] + [np.roll(y, round(n * k / 4)) for k in (1, 2, 3)])
    nscore = lambda v: stats.norm.ppf((stats.rankdata(v) - 0.5) / len(v))
    Yall = np.column_stack([evaluate(j, tri) for j in subs]); Yg = np.column_stack([nscore(Yall[:, j]) for j in range(Yall.shape[1])]); me = subs.index(s)
    DEV = "cuda"
    def bits_g(X):
        Ft = torch.tensor(X, dtype=torch.float64, device=DEV); Y = torch.tensor(Yall, dtype=torch.float64, device=DEV); f = folds(n); P = torch.zeros_like(Y)
        for b in range(10):
            tr, te = f != b, f == b; tr, te = torch.tensor(tr, device=DEV), torch.tensor(te, device=DEV)
            mu, sd = Ft[tr].mean(0), Ft[tr].std(0, unbiased=False) + 1e-6; A, B = (Ft[tr] - mu) / sd, (Ft[te] - mu) / sd
            W = torch.linalg.solve(A.T @ A + LAM * len(A) * torch.eye(A.shape[1], dtype=torch.float64, device=DEV), A.T @ (Y[tr] - Y[tr].mean(0))); P[te] = B @ W
        P = P.cpu().numpy(); rg = np.array([np.corrcoef(P[:, j], Yg[:, j])[0, 1] for j in range(Yg.shape[1])])
        zn = np.arctanh(np.clip(np.delete(rg, me), -0.999, 0.999)); d = float(np.arctanh(np.clip(rg[me], -0.999, 0.999)) - zn.mean())
        return dict(r_g=float(rg[me]), d=d, bits_g=float(-0.5 * np.log2(1 - min(np.tanh(max(d, 0.0)), 0.999) ** 2)), null_sd=float(zn.std(ddof=1)))
    rows = {}
    for name, X in (("all", F), ("ocular_proxy", O.astype(np.float64)), ("posterior_control", ctrl)):
        sc = pe_gain(X, np.ones(X.shape[1]), Ys, LAM)
        rows[name] = dict(n_features=int(X.shape[1]), gain=float(sc[:, 0].mean()), shift=float(sc[:, 1:].mean()), **bits_g(X))
        print(sub, name, {k: round(v, 4) for k, v in rows[name].items()}, f"{time.time()-t0:.0f}s", flush=True)
    q1 = json.loads((ROOT / f"results/yield/pe_bottleneck_ds003458/raw_{sub}.json").read_text()); qd = next(r["d"] for r in q1["rows"] if r["L"] == "full" and r["condition"] == "eeg")
    st = json.loads((ROOT / f"results/standard/pe_bottleneck/{sub}.json").read_text()); sg = next(r["gain"] for r in st["rows"] if r["input"] == "eeg" and r["mode"] == "pca" and r["L"] == "full")
    g2 = dict(n_lost=int((~keep).sum()), q1_d=qd, new_d=rows["all"]["d"], std_gain=sg, new_gain=rows["all"]["gain"])
    g2["pass_"] = bool(g2["n_lost"] > 0 or (abs(qd - rows["all"]["d"]) < 1e-6 and abs(sg - rows["all"]["gain"]) < 1e-9))
    res = dict(target="pe", participant=sub, n_trials=int(n), rows=rows, g2=g2, control_channels=RG.N1_PE_CONTROL, runtime_s=round(time.time() - t0, 1))
    (out / f"pe_{sub}.json").write_text(json.dumps(res, indent=1)); print(sub, "G2", g2, flush=True)


def summarize(target, out):
    per = [json.loads(f.read_text()) for f in sorted(out.glob(f"{target}_sub-*.json"))]; n = len(per)
    def ci(v):
        v = np.asarray(v); ix = np.random.default_rng(20261121).integers(0, n, (10000, n)); return np.quantile(v[ix].mean(1), [.025, .975]).tolist()
    res = dict(target=target, n=n, participants=[p["participant"] for p in per], g2_all_pass=all(p["g2"]["pass_"] for p in per))
    if target == "image":
        est = {"InfoNCE_k80": ("soft_bits", "floor")}
    else:
        est = {"log_score": ("gain", "shift"), "bits_g": ("bits_g", None)}
    allname = "all63" if target == "image" else "all"
    for e, (yk, fk) in est.items():
        blk = {}
        val = {m: np.array([p["rows"][m][yk] for p in per]) for m in per[0]["rows"]}
        flo = {m: (np.array([p["rows"][m][fk] for p in per]) if fk else np.zeros(n)) for m in per[0]["rows"]}
        ex = {m: val[m] - flo[m] for m in val}
        for m in val:
            blk[m] = dict(mean=float(val[m].mean()), ci=ci(val[m]), floor_mean=float(flo[m].mean()), excess_mean=float(ex[m].mean()), excess_ci=ci(ex[m]),
                          p_excess_gt0=float(stats.ttest_1samp(ex[m], 0, alternative="greater").pvalue), n_above_floor=int((ex[m] > 0).sum()))
        ps = sorted([(blk[m]["p_excess_gt0"], m) for m in val if m != allname])
        for r, (p, m) in enumerate(ps): blk[m]["p_holm"] = float(min(1, p * (len(ps) - r)))
        for m in val:
            if m == allname: continue
            sh = ex[m] / ex[allname]; blk[m]["share_mean_of_ratios"] = float(sh.mean()); blk[m]["share_ci"] = ci(sh)
            blk[m]["share_ratio_of_means"] = float(ex[m].mean() / ex[allname].mean())
            ix = np.random.default_rng(20261121).integers(0, n, (10000, n))      # participant bootstrap of the ratio of means
            blk[m]["share_ratio_of_means_ci"] = np.quantile(ex[m][ix].mean(1) / ex[allname][ix].mean(1), [.025, .975]).tolist()
        d = ex["ocular_proxy"] - ex["posterior_control"]; blk["proxy_minus_control"] = dict(mean=float(d.mean()), ci=ci(d), n_positive=int((d > 0).sum()))
        res[e] = blk
    (out / f"summary_{target}.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", required=True, choices=["image", "pe"]); ap.add_argument("--subjects", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(); out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    if a.summarize: return summarize(a.target, out)
    if a.target == "image":
        for s in ([int(x) for x in a.subjects.split(",")] if a.subjects else range(1, 11)):
            if not (out / f"image_sub-{s:02d}.json").exists(): image_subject(s, out)
    else:
        from i2_complementarity import targets
        evaluate, subs, fpath, _ = targets()
        for s in ([int(x) for x in a.subjects.split(",")] if a.subjects else subs):
            if not (out / f"pe_sub-{s:03d}.json").exists(): pe_subject(s, out, evaluate, subs, fpath)


if __name__ == "__main__":
    main()
