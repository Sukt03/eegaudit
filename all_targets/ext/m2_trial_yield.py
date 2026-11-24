"""M2 steps 0 + 1 (rungs R3): save per-trial yields, then test their reliability beyond target content (M2.0 gate).
EXECUTOR-2, 21 Nov 2026. Design = "## M2 REVIEW" (METHOD CHANGE 7), "## M2 AMENDMENT" + CHECK, "## M2 v2 PROPOSAL/REVIEW" (METHOD CHANGE 9),
and the contrarian card outputs/cards/tool_trial_usefulness_contrarian.md (reliability first; no selection claim here).

STEP 0, per-trial yield (PVI) = the pointwise term of the locked score; saved to results/ext/m2/pvi_<target>_<readout>_sub-XX.npz
  image (THINGS-EEG2, 10): single test repetition r of image i, frozen encoder, 200 candidates (paper CLIP class centres, as p1_rank_bound),
        PVI = log2 200 - CE at the LOO temperature chosen over the 200 queries of the same repetition (run.soft_bits / p1 pvi_stats).
        Read-outs: NICE-GA seed 141 (primary), NICE seed 141, NICE-GA seed 297 (second seed). Array (200 images, 80 reps).
        G2: p1_rank_bound's k = 1 draws are single repetitions shared by all images (rng 141); the mean PVI of the drawn repetitions
        reproduces results/loop/image_p1/p1_rank_bound.json per_participant.nice_ga[s].k1[d].I_NCE, and the group mean of their
        5-draw means reproduces the registered ViNiceOne (results/loop/p4/p4_image.json summary.nice_ga_k1_I_NCE.mean).
  PE (ds003458, 23): feedback epoch t, PVI = held-out log2 Gaussian density of the residualised PE given a linear calibration of the
        read-out minus intercept-only density (gain_rows of experiments/standard/pe_bottleneck_logscore.py).
        Read-outs: raw-ERP ridge, full width (nested calibration, = m3_yield_map.pe_gain), G2 = results/standard/pe_bottleneck/sub-XXX.json
        ('eeg','pca','full') gain per participant, group = registered StdPERawFull; ridge_halfA / ridge_halfB = the same ridge trained on
        disjoint interleaved halves of every outer training fold (calibration by inner 9-fold CV inside that half); EEGNet LOPO cached
        read-out (results/features/eegnet) scored as E12 (within-participant train-fold calibration, common_log_yield), G2 = the
        common_log_yield.json per-participant gain, group = registered StdPEYield.
STEP 1, reliability (gate for any usefulness model; pass iff the participant-bootstrap lower 95 % CI > 0.1):
  per participant: partial Spearman between the two read-outs' PVI = Pearson of rank residuals after regressing the ranks on stratum
  dummies (image: image identity; PE: outcome x |PE| quintile) and, for PE, trial position (t/480, its square).
  Pairs: image PRIMARY NICE-GA141 vs NICE141, secondary NICE-GA141 vs NICE-GA297 (across read-outs; no second recording exists).
         PE PRIMARY ridge_halfA vs ridge_halfB, secondary ridge(full) vs EEGNet.
  Floor: the same statistic with one read-out permuted within stratum (100 permutations, mean). All statistics run on the GPU
  (partial_spearman_gpu, perm_within_gpu, bootstrap index matrices; PLAN §10); the numpy versions are kept as the reference. Group: mean over participants,
  10,000-resample participant bootstrap CI; Holm over the 2 targets on one-sided t p-values for mean > 0.1. Leakage diagnostic (PE):
  partial Spearman of ridge PVI with |y| and with y inside strata.
Sensitivity (image): repetition fixed effects partialled as well (per-repetition LOO temperature and session time are shared by both read-outs).
Usage: python m2_trial_yield.py --target image|pe [--subjects 1] --out DIR ; python m2_trial_yield.py --summarize --out DIR
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy import stats

ROOT = Path("/path/to/eegaudit")
for p in (ROOT, ROOT / "experiments/ext", ROOT / "experiments/eegbits", ROOT / "experiments/loop/image", ROOT / "experiments/agent_loop_20261120",
          ROOT / "experiments/standard"):
    sys.path.insert(0, str(p))
LN2 = np.log(2)


def pvi_rows(S, taus):
    """Per-query PVI with the LOO temperature (identical algebra to p1_rank_bound.pvi_stats / run.soft_bits)."""
    n = S.shape[0]; rows = np.arange(n)
    def ce_all(tau):
        Z = S / tau; Z = Z - Z.max(1, keepdims=True); lp = Z - np.log(np.exp(Z).sum(1, keepdims=True)); return -lp[rows, rows] / np.log(2)
    CE = np.stack([ce_all(t) for t in taus]); tot = CE.sum(1, keepdims=True); loo = (tot - CE) / (n - 1); t_idx = loo.argmin(0)
    return np.log2(n) - CE[t_idx, rows]


def pvi_rows_gpu(S, taus):
    """Batched pvi_rows on the GPU: S (B, n, n) -> (B, n)."""
    T = torch.as_tensor(np.asarray(taus), dtype=torch.float64, device=S.device); n = S.shape[-1]
    CE = -torch.diagonal(torch.log_softmax(S[:, None] / T[None, :, None, None], dim=-1), dim1=-2, dim2=-1) / np.log(2)   # (B, T, n)
    loo = (CE.sum(-1, keepdim=True) - CE) / (n - 1); t_idx = loo.argmin(1, keepdim=True)
    return np.log2(n) - torch.gather(CE, 1, t_idx)[:, 0]


def image_subject(s, out):
    from run import _TAUS, soft_bits
    from run_acq_controls import load_encoder
    dev = torch.device("cuda"); sub = f"sub-{s:02d}"; t0 = time.time(); g2 = {}
    norm = lambda A: A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    for name, var, seed in (("nicega141", "nice_ga", 141), ("nice141", "nice", 141), ("nicega297", "nice_ga", 297)):
        embed, Y, _, X = load_encoder(var, sub, seed, dev); X = np.asarray(X); Yn = torch.as_tensor(norm(np.asarray(Y)), dtype=torch.float64, device=dev)
        E = torch.stack([torch.as_tensor(norm(embed(X[:, r])), dtype=torch.float64, device=dev) for r in range(80)])     # (80, 200, d)
        P = pvi_rows_gpu(E @ Yn.T, _TAUS).T.cpu().numpy()                                                                 # (200, 80)
        S0 = (E[0] @ Yn.T).cpu().numpy(); assert abs(P[:, 0].mean() - soft_bits(S0)) < 1e-9 and np.abs(P[:, 0] - pvi_rows(S0, _TAUS)).max() < 1e-9
        np.savez(out / f"pvi_image_{name}_{sub}.npz", pvi=P)
        if name == "nicega141":
            rng = np.random.default_rng(141); rng.permutation(200); draws = []
            for d in range(5):
                draws.append(int(rng.choice(80, 1)[0])); rng.permutation(200)
            old = json.loads((ROOT / "results/loop/image_p1/p1_rank_bound.json").read_text())["per_participant"]["nice_ga"][sub]["k1"]
            new = [float(P[:, r].mean()) for r in draws]; oldv = [x["I_NCE"] for x in old]
            g2 = dict(file="results/loop/image_p1/p1_rank_bound.json", key=f"per_participant.nice_ga.{sub}.k1[*].I_NCE", draws=draws, old=oldv, new=new,
                      max_abs_diff=float(max(abs(a - b) for a, b in zip(oldv, new))), k1_mean_new=float(np.mean(new)), all_rep_mean=float(P.mean()))
            g2["pass"] = g2["max_abs_diff"] < 1e-4
            print(sub, "G2", g2, flush=True)
        print(sub, name, f"mean PVI {P.mean():.3f}", f"{time.time()-t0:.0f}s", flush=True)
    (out / f"g2_image_{sub}.json").write_text(json.dumps(g2, indent=1))


def pe_subject(s, out, evaluate, fpath, Q):
    import torch
    from belief_bottleneck import LAM, folds
    from pe_bottleneck_logscore import gain_rows
    from m3_yield_map import pe_gain
    DEV = "cuda"; sub = f"sub-{s:03d}"; fz = np.load(fpath("raw", s)); tri = fz["tri"]; n = len(tri)
    y = evaluate(s, tri).astype(np.float64); F = fz["F"].astype(np.float64); cw = fz["colw"].astype(np.float64)
    full = pe_gain(F, cw, y[:, None], LAM)[:, 0]
    # disjoint interleaved halves of each outer training fold
    Ft = torch.tensor(F, device=DEV); cwt = torch.tensor(cw, device=DEV); Yt = torch.tensor(y[:, None], device=DEV); fo = folds(n)
    def pred(tr, te):
        A, B = Ft[tr], Ft[te]; mu, sd = A.mean(0), A.std(0, unbiased=False) + 1e-6; A = (A - mu) / sd * cwt; B = (B - mu) / sd * cwt
        W = torch.linalg.solve(A.T @ A + LAM * len(A) * torch.eye(A.shape[1], dtype=torch.float64, device=DEV), A.T @ (Yt[tr] - Yt[tr].mean(0)))
        return (B @ W).cpu().numpy()[:, 0]
    halves = {}
    for h in (0, 1):
        sc = np.full(n, np.nan)
        for b in range(10):
            tr = np.where(fo != b)[0][h::2]; te = np.where(fo == b)[0]
            P = pred(tr, te); inner = np.zeros(len(tr)); fi = folds(len(tr), 9)
            for c in range(9): inner[fi == c] = pred(tr[fi != c], tr[fi == c])
            sc[te] = gain_rows(y[tr], inner, y[te], P)
        halves[h] = sc
    old = json.loads((ROOT / f"results/standard/pe_bottleneck/{sub}.json").read_text())
    o = next(r for r in old["rows"] if r["input"] == "eeg" and r["mode"] == "pca" and r["L"] == "full")["gain"]
    ez = np.load(fpath("eegnet", s)); xe, te_ = ez["F"].astype(float)[:, 0], ez["tri"]; ye = evaluate(s, te_); fe = folds(len(ye)); ge = np.full(len(ye), np.nan)
    for b in range(10): ge[fe == b] = gain_rows(ye[fe != b], xe[fe != b], ye[fe == b], xe[fe == b])
    common = {r["subject"]: r["gain"] for r in json.loads((ROOT / "outputs/AGENT_LOOP_20261120/results/common_log_yield.json").read_text())["behavioral"]["rows"]["eegnet"]}
    g2 = dict(ridge_old=o, ridge_new=float(full.mean()), ridge_abs_diff=abs(o - float(full.mean())), eegnet_old=common[s], eegnet_new=float(ge.mean()),
              eegnet_abs_diff=abs(common[s] - float(ge.mean())))
    g2["pass"] = g2["ridge_abs_diff"] < 1e-9 and g2["eegnet_abs_diff"] < 1e-9
    reward = Q["reward"][s - 1]
    np.savez(out / f"pvi_pe_ridge_{sub}.npz", pvi=full, tri=tri, y=y, reward=reward[tri])
    np.savez(out / f"pvi_pe_ridgehalves_{sub}.npz", pvi_a=halves[0], pvi_b=halves[1], tri=tri)
    np.savez(out / f"pvi_pe_eegnet_{sub}.npz", pvi=ge, tri=te_, y=ye, reward=reward[te_])
    (out / f"g2_pe_{sub}.json").write_text(json.dumps(g2, indent=1))
    print(sub, "G2", g2, f"halves mean {halves[0].mean():.4f} {halves[1].mean():.4f}", flush=True)


DEV = "cuda"


def _rank(X):
    """Row-wise ranks (0-based, ordinal ties) on the GPU; PVI values are continuous."""
    return torch.argsort(torch.argsort(X, dim=-1), dim=-1).double()


def partial_spearman_gpu(a, Bm, groups, extra=None):
    """a (n,), Bm (P, n): Pearson of rank residuals after removing group means and (Frisch-Waugh) within-group-demeaned covariates.
    Returns (P,) correlations. GPU twin of partial_spearman (equal up to rank ties)."""
    g = torch.as_tensor(np.unique(groups, return_inverse=True)[1], device=DEV); cnt = torch.bincount(g).double()
    dm = lambda V: V - (torch.zeros(V.shape[0], int(g.max()) + 1, dtype=V.dtype, device=DEV).index_add_(1, g, V) / cnt)[:, g]
    E = None if extra is None else dm(torch.as_tensor(np.atleast_2d(np.asarray(extra).T), dtype=torch.float64, device=DEV))   # (q, n)
    def res(V):
        R = dm(V)
        if E is None: return R
        coef = torch.linalg.lstsq(E.T, R.T).solution; return R - (E.T @ coef).T
    ra = res(_rank(torch.as_tensor(a, dtype=torch.float64, device=DEV)[None])); rb = res(_rank(torch.as_tensor(Bm, dtype=torch.float64, device=DEV)))
    return (rb @ ra[0]) / torch.sqrt((ra[0] @ ra[0]) * (rb * rb).sum(1) + 1e-300)


def perm_within_gpu(v, groups, P, seed):
    """P within-group permutations of v as one tensor op: sort by (group, random key)."""
    g = torch.as_tensor(np.unique(groups, return_inverse=True)[1], device=DEV).double(); gen = torch.Generator(device=DEV); gen.manual_seed(seed)
    key = g[None] + torch.rand((P, len(v)), generator=gen, device=DEV, dtype=torch.float64) * 0.5
    src = torch.argsort(key, dim=1); dst = torch.argsort(g, stable=True)          # i-th slot of each group (in sorted order) gets a random member
    out = torch.empty((P, len(v)), dtype=torch.float64, device=DEV); vt = torch.as_tensor(v, dtype=torch.float64, device=DEV)
    out[:, dst] = vt[src]; return out


def partial_spearman(a, b, groups, extra=None):
    """Pearson of rank residuals after regressing ranks on group dummies (+ extra covariates); Frisch-Waugh: demean within group,
    then partial the within-group-demeaned covariates."""
    _, g = np.unique(groups, return_inverse=True); cnt = np.bincount(g)
    dm = lambda v: v - (np.bincount(g, v) / cnt)[g]
    E = None if extra is None else np.column_stack([dm(c) for c in np.atleast_2d(extra.T)])
    def res(v):
        r = dm(v)
        return r if E is None else r - E @ np.linalg.lstsq(E, r, rcond=None)[0]
    ra, rb = res(stats.rankdata(a)), res(stats.rankdata(b))
    return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb) + 1e-300))


def perm_within(v, groups, rng):
    v = v.copy()
    for gval in np.unique(groups):
        i = np.where(groups == gval)[0]; v[i] = v[rng.permutation(i)]
    return v


def boot_ci(v, B=10000, seed=20261121):
    """Participant bootstrap on the GPU ([B, n] index matrix)."""
    vt = torch.as_tensor(np.asarray(v, dtype=float), device=DEV); gen = torch.Generator(device=DEV); gen.manual_seed(seed)
    ix = torch.randint(0, len(vt), (B, len(vt)), generator=gen, device=DEV)
    return torch.quantile(vt[ix].mean(1), torch.tensor([.025, .975], dtype=vt.dtype, device=DEV)).tolist()


def reliability(out):
    rng = np.random.default_rng(20261121); res = {}
    # image
    rows = {}
    for f in sorted(out.glob("pvi_image_nicega141_sub-*.npz")):
        sub = f.stem.split("_")[-1]; a = np.load(f)["pvi"]; g = np.repeat(np.arange(200), 80)
        r = {}
        for other in ("nice141", "nicega297"):
            b = np.load(out / f"pvi_image_{other}_{sub}.npz")["pvi"]
            rho = float(partial_spearman_gpu(a.ravel(), b.ravel()[None], g)[0]); fl = float(partial_spearman_gpu(a.ravel(), perm_within_gpu(b.ravel(), g, 100, 7), g).mean())
            r[other] = dict(rho=rho, floor=float(fl), unstratified=float(stats.spearmanr(a.ravel(), b.ravel())[0]))
        rep = np.tile(np.eye(80)[:, 1:], (200, 1)); b = np.load(out / f"pvi_image_nice141_{sub}.npz")["pvi"]
        r["nice141_repfe"] = dict(rho=float(partial_spearman_gpu(a.ravel(), b.ravel()[None], g, rep)[0]),
                                  floor=float(partial_spearman_gpu(a.ravel(), perm_within_gpu(b.ravel(), g, 20, 8), g, rep).mean()))
        r["item_splithalf_sb"] = (lambda q: 2 * q / (1 + q))(float(stats.spearmanr(a[:, 0::2].mean(1), a[:, 1::2].mean(1))[0]))
        rows[sub] = r
    if rows:
        res["image"] = dict(n=len(rows), per_participant=rows)
        for other in ("nice141", "nicega297"):
            v = np.array([rows[s][other]["rho"] for s in rows]); fl = np.array([rows[s][other]["floor"] for s in rows])
            res["image"][other] = dict(mean=float(v.mean()), ci=boot_ci(v), floor_mean=float(fl.mean()), floor_ci=boot_ci(fl),
                                       unstratified_mean=float(np.mean([rows[s][other]["unstratified"] for s in rows])), p_gt_0p1=float(stats.ttest_1samp(v, 0.1, alternative="greater").pvalue))
        v = np.array([rows[s]["nice141_repfe"]["rho"] for s in rows]); res["image"]["nice141_repfe"] = dict(mean=float(v.mean()), ci=boot_ci(v), floor_mean=float(np.mean([rows[s]["nice141_repfe"]["floor"] for s in rows])),
            note="sensitivity: repetition fixed effects partialled (removes per-repetition temperature / session-time effects)")
        res["image"]["item_splithalf_sb_mean"] = float(np.mean([rows[s]["item_splithalf_sb"] for s in rows]))
    # PE
    rows = {}
    for f in sorted(out.glob("pvi_pe_ridge_sub-*.npz")):
        sub = f.stem.split("_")[-1]; z = np.load(f); h = np.load(out / f"pvi_pe_ridgehalves_{sub}.npz"); e = np.load(out / f"pvi_pe_eegnet_{sub}.npz")
        tri, y, rw = z["tri"], z["y"], z["reward"]; q = np.minimum((stats.rankdata(np.abs(y)) - 1) * 5 // len(y), 4).astype(int); grp = rw.astype(int) * 5 + q
        pos = np.column_stack([tri / 480.0, (tri / 480.0) ** 2])
        rho = float(partial_spearman_gpu(h["pvi_a"], h["pvi_b"][None], grp, pos)[0])
        fl = float(partial_spearman_gpu(h["pvi_a"], perm_within_gpu(h["pvi_b"], grp, 100, 9), grp, pos).mean())
        common, ia, ib = np.intersect1d(tri, e["tri"], return_indices=True)
        rho_e = float(partial_spearman_gpu(z["pvi"][ia], e["pvi"][ib][None], grp[ia], pos[ia])[0])
        fl_e = float(partial_spearman_gpu(z["pvi"][ia], perm_within_gpu(e["pvi"][ib], grp[ia], 100, 10), grp[ia], pos[ia]).mean())
        leak_abs = float(partial_spearman_gpu(z["pvi"], np.abs(y)[None], grp, pos)[0]); leak_y = float(partial_spearman_gpu(z["pvi"], y[None], grp, pos)[0])
        q10 = np.minimum((stats.rankdata(np.abs(y)) - 1) * 10 // len(y), 9).astype(int); g10 = rw.astype(int) * 10 + q10   # decile refinement
        dec = dict(rho=float(partial_spearman_gpu(h["pvi_a"], h["pvi_b"][None], g10, pos)[0]),
                   floor=float(partial_spearman_gpu(h["pvi_a"], perm_within_gpu(h["pvi_b"], g10, 100, 11), g10, pos).mean()),
                   leak_abs_y=float(partial_spearman_gpu(z["pvi"], np.abs(y)[None], g10, pos)[0]), leak_y=float(partial_spearman_gpu(z["pvi"], y[None], g10, pos)[0]))
        posy = np.column_stack([pos, y, np.abs(y), y ** 2])                                   # AUDIT M2 (06:55): y, |y|, y^2 partialled too
        dec["rho_ycov"] = float(partial_spearman_gpu(h["pvi_a"], h["pvi_b"][None], g10, posy)[0])
        dec["floor_ycov"] = float(partial_spearman_gpu(h["pvi_a"], perm_within_gpu(h["pvi_b"], g10, 100, 12), g10, posy).mean())
        rows[sub] = dict(halves_deciles=dec, halves=dict(rho=rho, floor=float(fl), unstratified=float(stats.spearmanr(h["pvi_a"], h["pvi_b"])[0])),
                         ridge_vs_eegnet=dict(rho=rho_e, floor=float(fl_e), n_common=int(len(common))), leak_abs_y=leak_abs, leak_y=leak_y)
    if rows:
        res["pe"] = dict(n=len(rows), per_participant=rows)
        for k in ("halves", "ridge_vs_eegnet"):
            v = np.array([rows[s][k]["rho"] for s in rows]); fl = np.array([rows[s][k]["floor"] for s in rows])
            res["pe"][k] = dict(mean=float(v.mean()), ci=boot_ci(v), floor_mean=float(fl.mean()), floor_ci=boot_ci(fl), p_gt_0p1=float(stats.ttest_1samp(v, 0.1, alternative="greater").pvalue))
        for k in ("leak_abs_y", "leak_y"):
            v = np.array([rows[s][k] for s in rows]); res["pe"][k] = dict(mean=float(v.mean()), ci=boot_ci(v))
        dd = {k: np.array([rows[s]["halves_deciles"][k] for s in rows]) for k in ("rho", "floor", "leak_abs_y", "leak_y", "rho_ycov", "floor_ycov")}
        res["pe"]["halves_deciles_ycov"] = dict(mean=float(dd["rho_ycov"].mean()), ci=boot_ci(dd["rho_ycov"]), floor_mean=float(dd["floor_ycov"].mean()),
                                                floor_ci=boot_ci(dd["floor_ycov"]), note="decile strata; position, position^2, y, |y|, y^2 partialled")
        res["pe"]["halves_deciles"] = dict(mean=float(dd["rho"].mean()), ci=boot_ci(dd["rho"]), floor_mean=float(dd["floor"].mean()), floor_ci=boot_ci(dd["floor"]),
                                           leak_abs_y=dict(mean=float(dd["leak_abs_y"].mean()), ci=boot_ci(dd["leak_abs_y"])), leak_y=dict(mean=float(dd["leak_y"].mean()), ci=boot_ci(dd["leak_y"])),
                                           note="M2 REVIEW 2: the quintile leakage CI excludes 0 -> strata refined to outcome x |PE| decile")
    prim = [(t, k) for t, k in (("image", "nice141"), ("pe", "halves")) if t in res]
    ps = sorted([(res[t][k]["p_gt_0p1"], t, k) for t, k in prim])
    for r, (p, t, k) in enumerate(ps):
        res[t][k]["p_holm"] = float(min(1, p * (len(ps) - r))); res[t][k]["gate_pass"] = bool(res[t][k]["ci"][0] > 0.1)
    return res


def summarize(out):
    res = reliability(out)
    g2i = [json.loads(f.read_text()) for f in sorted(out.glob("g2_image_sub-*.json"))]; g2p = [json.loads(f.read_text()) for f in sorted(out.glob("g2_pe_sub-*.json"))]
    if g2i: res["g2_image"] = dict(n=len(g2i), all_pass=all(g["pass"] for g in g2i), k1_group_mean=float(np.mean([g["k1_mean_new"] for g in g2i])), max_abs_diff=max(g["max_abs_diff"] for g in g2i),
                                   all_rep_mean=float(np.mean([g["all_rep_mean"] for g in g2i])))
    if g2p: res["g2_pe"] = dict(n=len(g2p), all_pass=all(g["pass"] for g in g2p), ridge_group_mean=float(np.mean([g["ridge_new"] for g in g2p])),
                                eegnet_group_mean=float(np.mean([g["eegnet_new"] for g in g2p])), max_abs_diff=max(max(g["ridge_abs_diff"], g["eegnet_abs_diff"]) for g in g2p))
    (out / "summary_reliability.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({t: {k: v for k, v in res[t].items() if k != "per_participant"} for t in res}, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", choices=["image", "pe"]); ap.add_argument("--subjects", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--summarize", action="store_true"); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize: return summarize(out)
    if a.target == "image":
        for s in ([int(x) for x in a.subjects.split(",")] if a.subjects else range(1, 11)):
            if not (out / f"g2_image_sub-{s:02d}.json").exists(): image_subject(s, out)
    else:
        from i2_complementarity import targets
        from group_stats import QUANT
        evaluate, subs, fpath, _ = targets(); Q = np.load(ROOT / QUANT)
        for s in ([int(x) for x in a.subjects.split(",")] if a.subjects else subs):
            if not (out / f"g2_pe_sub-{s:03d}.json").exists(): pe_subject(s, out, evaluate, fpath, Q)


if __name__ == "__main__":
    main()
