"""Q1 / X7-PE: prediction-error yield vs input bottleneck width, ds003458 (PROTOCOL step b; EXPERIMENTS agent, 2026-11-20).
Read-out = the paper's raw-ERP within-participant ridge (results/features/raw: 64 ch x 23 samples 150-600 ms = 1472 columns; 10 folds of
belief_bottleneck.folds, in-fold z-scoring and colw, ridge LAM), trained on the RAW residualised PE of every participant j on the same trials
(session-permutation design of group_stats / s13b). Bottleneck at width L: in-fold PCA-L (as cv_predict_gpu) or a fixed random orthonormal
L-dim projection (random-input bottleneck, seed 0); L = 'full' = all 1472 columns, no projection. Yield = Gaussianised bound exactly as
s13b_gauss_bits.py: r_g = corr(read-out, normal-scores(target)); d_i = atanh(r_g,ii) - mean_j atanh(r_g,ij) (null-corrected); bits_g = -1/2 log2(1 - tanh(d)^2)
(sign kept: negative d -> reported as 0 bits with d stored). Conditions per L: eeg (null-corrected), eeg_uncorrected, perm_p95 (bits at the 95th
percentile of the participant's own 22 null r_g), noise (N(0,1) features of the same shape through the same pipeline, null-corrected), random (random
projection, null-corrected). One protocol-schema JSON per participant: results/yield/pe_bottleneck_ds003458/raw_sub-XXX.json, plus summary.json."""
import sys, json, time, numpy as np, torch
from pathlib import Path
from scipy import stats
ROOT = Path('/path/to/eegaudit'); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments/loop'))
from belief_bottleneck import LAM, folds
from group_stats import DEV, dersimonian_laird, corr
import importlib.util
spec = importlib.util.spec_from_file_location("s13b_src", ROOT / "experiments/loop/s13b_gauss_bits.py")
src = open(ROOT / "experiments/loop/s13b_gauss_bits.py").read(); src = src[:src.index("DATASETS = {")]   # reuse target construction only
ns = {}; exec(compile(src, "s13b_head", "exec"), ns); targets_ds003458 = ns["targets_ds003458"]; nscore = ns["nscore"]
OUT = ROOT / "results/yield/pe_bottleneck_ds003458"; OUT.mkdir(parents=True, exist_ok=True)
LS = [1, 2, 4, 8, 16, 32, 64, 128, 256, "full"]; SMOKE = "--smoke" in sys.argv
gb = lambda d: float(-0.5 * np.log2(1 - min(np.tanh(max(d, 0.0)), 0.999) ** 2))


def cv_predict(F, Y, L, mode, colw, Rproj):
    F = torch.tensor(F, dtype=torch.float64, device=DEV); Y = torch.tensor(Y, dtype=torch.float64, device=DEV); cw = torch.tensor(colw, dtype=torch.float64, device=DEV)
    f = torch.tensor(folds(len(F)), device=DEV); P = torch.zeros_like(Y)
    for b in range(10):
        tr, te = f != b, f == b
        mu, sd = F[tr].mean(0), F[tr].std(0, unbiased=False) + 1e-6; A, B = (F[tr] - mu) / sd * cw, (F[te] - mu) / sd * cw
        if L == "full": Atr, Ate = A, B
        elif mode == "pca":
            _, _, Vt = torch.linalg.svd(A - A.mean(0), full_matrices=False); Vk = Vt[:min(L, Vt.shape[0])].T; Atr, Ate = A @ Vk, B @ Vk
        else:
            Vk = Rproj[:, :L]; Atr, Ate = A @ Vk, B @ Vk
        k = Atr.shape[1]
        W = torch.linalg.solve(Atr.T @ Atr + LAM * len(Atr) * torch.eye(k, dtype=torch.float64, device=DEV), Atr.T @ (Y[tr] - Y[tr].mean(0)))
        P[te] = Ate @ W
    return P.cpu().numpy()


def stats_for(P, Yg, me):
    rg = np.array([corr(P[:, j], Yg[:, j]) for j in range(Yg.shape[1])]); nl = np.delete(rg, me); zn = np.arctanh(np.clip(nl, -0.999, 0.999))
    d = float(np.arctanh(np.clip(rg[me], -0.999, 0.999)) - zn.mean()); v = float(zn.var(ddof=1) * (1 + 1 / len(zn)))
    return dict(r_g=float(rg[me]), d=d, v=v, p95=float(np.quantile(nl, 0.95)))


t0 = time.time(); qtil, subs, fpath, _ = targets_ds003458(); subs_use = subs[:3] if SMOKE else subs
rng = np.random.default_rng(0); per = {}
for s in subs_use:
    fz = np.load(fpath("raw", s)); F, colw, tri = fz["F"].astype(np.float64), fz["colw"].astype(np.float64), fz["tri"]
    Y = np.column_stack([qtil(j, tri) for j in subs]); Yg = np.column_stack([nscore(Y[:, j]) for j in range(Y.shape[1])]); me = subs.index(s)
    Rproj = torch.tensor(np.linalg.qr(np.random.default_rng(0).standard_normal((F.shape[1], F.shape[1])))[0], dtype=torch.float64, device=DEV)
    Fn = np.random.default_rng(1000 + s).standard_normal(F.shape)
    rows = []; per[s] = {}
    for L in LS:
        e = stats_for(cv_predict(F, Y, L, "pca", colw, Rproj), Yg, me); n = stats_for(cv_predict(Fn, Y, L, "pca", colw, Rproj), Yg, me)
        r = stats_for(cv_predict(F, Y, L, "random", colw, Rproj), Yg, me) if L != "full" else e
        per[s][str(L)] = dict(eeg=e, noise=n, random=r)
        for cond, val in (("eeg", gb(e["d"])), ("eeg_uncorrected", gb(np.arctanh(np.clip(e["r_g"], -0.999, 0.999)))), ("perm_p95", gb(np.arctanh(e["p95"]))), ("noise", gb(n["d"])), ("random", gb(r["d"]))):
            rows.append(dict(L=L, condition=cond, bits_g=val))
        rows[-5]["d"] = e["d"]; rows[-5]["r_g"] = e["r_g"]; rows[-2]["d"] = n["d"]; rows[-1]["d"] = r["d"]
    json.dump(dict(dataset="ds003458", decoder="raw_erp_ridge", participant=f"sub-{s:03d}", bottleneck="input_pca_in_fold (random = fixed random orthonormal projection)",
                   target="normal-scored residualised PE (RWD)", full_dim=int(F.shape[1]), n_epochs=int(len(tri)), estimator="Gaussianised bound, null-corrected (s13b)", rows=rows),
              open(OUT / f"raw_sub-{s:03d}.json", "w"), indent=1)
    print(f"sub-{s:03d} " + " ".join(f"L{L}:{gb(per[s][str(L)]['eeg']['d']):.3f}/{gb(per[s][str(L)]['random']['d']):.3f}/{gb(per[s][str(L)]['noise']['d']):.3f}" for L in LS) + f" ({time.time()-t0:.0f}s)", flush=True)
summ = {}
for L in LS:
    blk = {}
    for cond in ("eeg", "random", "noise"):
        d = np.array([per[s][str(L)][cond]["d"] for s in per]); v = np.array([per[s][str(L)][cond]["v"] for s in per]); meta = dersimonian_laird(d, v)
        pp = np.array([gb(x) for x in d]); bs = [pp[rng.integers(0, len(pp), len(pp))].mean() for _ in range(4000)]
        blk[cond] = dict(rho_g_pooled=float(np.tanh(meta["mu"])), bits_g_pooled=gb(meta["mu"]), bits_g_pooled_ci=[gb(x) for x in meta["ci"]], bits_g_mean_pp=float(pp.mean()),
                         bits_g_mean_pp_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], p_meta=float(meta["p"]))
    blk["n_above_own_p95"] = int(sum(per[s][str(L)]["eeg"]["r_g"] > per[s][str(L)]["eeg"]["p95"] for s in per)); summ[str(L)] = blk
plateau = max(summ[str(L)]["eeg"]["bits_g_mean_pp"] for L in LS)
L90 = next((L for L in LS if summ[str(L)]["eeg"]["bits_g_mean_pp"] >= 0.9 * plateau), None)
pp_L90 = {}
for s in per:
    vals = [gb(per[s][str(L)]["eeg"]["d"]) for L in LS]; m = max(vals); pp_L90[f"sub-{s:03d}"] = next((L for L, x in zip(LS, vals) if m > 0 and x >= 0.9 * m), None)
res = dict(note=__doc__, n=len(per), widths=LS, summary=summ, plateau_bits_g_mean_pp=plateau, L90=L90, L90_defined=bool(summ[str(L90)]["n_above_own_p95"] >= 20) if L90 is not None else False,
           per_participant_L90=pp_L90, eegnet_full_width_reference=dict(file="results/loop/learner/measure_bits_gauss.json", key="datasets.ds003458.decoders.eegnet.bits_g_mean_pp"), runtime_s=round(time.time() - t0, 1))
json.dump(res, open(OUT / ("summary_smoke.json" if SMOKE else "summary.json"), "w"), indent=1)
for L in LS: print(L, {c: round(summ[str(L)][c]["bits_g_mean_pp"], 4) for c in ("eeg", "random", "noise")}, summ[str(L)]["n_above_own_p95"])
print("L90", L90, "plateau", round(plateau, 4), "done", res["runtime_s"])
