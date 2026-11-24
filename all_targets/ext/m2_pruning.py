"""M2 step 2: TRAINING-side pruning by a label-free trial-usefulness score (rung R9; METHOD CHANGE 9; M2 v2 PROPOSAL + REVIEW).
EXECUTOR-2, 21 Nov 2026. PE arm (ds003458, X7 raw-ERP ridge, 23 participants). GPU only (PLAN §10): every ridge is a batched dual solve.

Gate: M2.0 passed (reliability 0.39 PE, 0.74 image; AUDIT 06:55).
Label-free per-trial features (results/eeg_measures epochs, post-ICA, the X7 source; z-scored WITHIN participant without labels):
  log variance (all channels, -200..800 ms), log max peak-to-peak over channels, mean channel kurtosis, frontal-pole delta log power
  (BP band 1, Fp1/Fp2/AF7/AF8, 150-600 ms), posterior alpha log power (BP band 3, O1/Oz/O2/PO7/PO8/Pz), beta log power (BP band 4, all ch).
  Trial position is excluded (M2 REVIEW 2).
Usefulness score for participant s = ridge (GPU, alpha by inner leave-one-participant-out over {0.1, 1, 10, 100, 1000}) from the features
  to the cross-fitted per-trial yield (PVI) of the OTHER 22 participants (results/ext/m2/pvi_pe_ridge_sub-XXX.npz: held-out per-trial
  log-score of the nested X7 ridge; each trial's value comes from a model that never saw it). s's PVI is never used.
G1 PASS conditions (METHOD 09:25): (1) PRIMARY = stratified pruning: top x % WITHIN each outcome x |PE|-decile stratum of the training
  rows, for every comparator; unstratified = secondary. (2) Score target = other participants' PVI minus its stratum mean. (3) Score alpha
  grid {0.1 ... 1e5}, inner-LOPO curve printed.
Pruning (TRAINING rows only): in every outer fold b, the training rows are ranked by a comparator score (higher = keep) and the top x %
  (x = 50, 75) are kept; the X7 ridge (in-fold z-scoring on the kept rows, LAM * n_kept, inner 9-fold calibration within the kept rows,
  contiguous folds in time order) is refit and scored on the FULL, untouched test fold with the locked log-score.
Comparators at the same x: usefulness score; random (5 seeds); peak-to-peak (keep the lowest); variance only (keep the lowest);
  decoder confidence (|inner-CV prediction - training mean| of the full-training-rows inner predictions of that fold: label-free for each
  trial, keep the highest); shuffled usefulness score (G3 floor). No rejection = x 100 %.
  The FAAR-style quality index is NOT implemented (its equations could not be checked against arXiv 2605.12408 in time): stated as absent.
Statistics: per participant yield per member; PRIMARY family {score - random(mean of 5), score - no rejection} x {50, 75}: paired
  participant bootstrap (10,000, GPU), Holm over 4. "Pruning helps" iff a primary Holm CI > 0 AND the estimate >= +0.005 bits/epoch for
  both primary contrasts at some x; "no useful signal" iff the 90 % CI upper bound < +0.005 for both; else "inconclusive".
  SECONDARY family {score - peak-to-peak, score - variance, score - confidence} x {50, 75}, Holm over 6.
G2: x = 100 % reproduces results/standard/pe_bottleneck/sub-XXX.json full gain exactly (|d| < 1e-9; dual vs primal algebra).
G3: shuffled-score pruning - random pruning ~ 0 (CI covers 0); random at 100 % = no rejection.
Usage: python m2_pruning.py --out results/ext/m2_pruning [--subjects 1,2,3] ; --summarize
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/path/to/eegaudit")
for p in (ROOT, ROOT / "experiments/ext", ROOT / "experiments/agent_loop_20261120", ROOT / "experiments/standard"):
    sys.path.insert(0, str(p))
import regions as RG
DEV = "cuda"; LN2 = np.log(2); XS = (0.5, 0.75); NRAND = 5


def features(s):
    e = np.load(ROOT / f"results/eeg_measures/sub-{s:03d}_epochs.npz"); good = ~e["bad"]; tri = e["trial"][good]
    X = torch.as_tensor(e["X"][good], dtype=torch.float64, device=DEV); BP = torch.as_tensor(e["BP"][good], dtype=torch.float64, device=DEV)
    ch = [str(c) for c in e["ch"]]; fp = RG.indices(ch, ["Fp1", "Fp2", "AF7", "AF8"]); po = RG.indices(ch, ["O1", "Oz", "O2", "PO7", "PO8", "Pz"])
    xc = X - X.mean(2, keepdim=True); kurt = ((xc ** 4).mean(2) / ((xc ** 2).mean(2) ** 2 + 1e-12)).mean(1)
    F = torch.stack([torch.log(X.var(2).mean(1) + 1e-12), torch.log((X.max(2).values - X.min(2).values).max(1).values + 1e-12), kurt,
                     BP[:, fp, 0].mean(1), BP[:, po, 2].mean(1), BP[:, :, 3].mean(1)], 1)
    F = (F - F.mean(0)) / (F.std(0) + 1e-12)                                      # label-free within-participant standardisation
    return tri, F


def strata(y, reward, q=10):
    """Outcome x |PE|-decile strata (deciles of |y| among the given rows)."""
    r = np.argsort(np.argsort(np.abs(y), kind="stable"), kind="stable"); dec = np.minimum(r * q // len(y), q - 1)
    return np.asarray(reward).astype(int) * q + dec


def strat_mean(v, g):
    out = np.empty_like(v, dtype=float)
    for gv in np.unique(g): out[g == gv] = v[g == gv].mean()
    return out


def ridge_fit(A, y, lam):
    A1 = torch.cat([A, torch.ones(len(A), 1, dtype=A.dtype, device=DEV)], 1); I = torch.eye(A1.shape[1], dtype=A.dtype, device=DEV); I[-1, -1] = 0
    return torch.linalg.solve(A1.T @ A1 + lam * I, A1.T @ y)


def ridge_pred(A, w):
    return torch.cat([A, torch.ones(len(A), 1, dtype=A.dtype, device=DEV)], 1) @ w


def usefulness_scores(subs):
    """LOPO usefulness score per participant: ridge from features to other participants' cross-fitted PVI (alpha by inner LOPO)."""
    D = {}
    for s in subs:
        tri, F = features(s); z = np.load(ROOT / f"results/ext/m2/pvi_pe_ridge_sub-{s:03d}.npz"); pos = {t: i for i, t in enumerate(tri)}
        keep = [i for i, t in enumerate(z["tri"]) if t in pos]
        pv = z["pvi"][keep] - strat_mean(z["pvi"][keep], strata(z["y"][keep], z["reward"][keep]))   # condition 2: usefulness beyond target content
        D[s] = (z["tri"][keep], F[[pos[t] for t in z["tri"][keep]]], torch.as_tensor(pv, dtype=torch.float64, device=DEV))
    lams = [0.1, 1, 10, 100, 1e3, 1e4, 1e5]; out = {}
    for s in subs:
        others = [j for j in subs if j != s]
        def cv_err(lam):   # inner leave-one-participant-out on the other participants
            e = 0.0
            for j in others:
                tr = [k for k in others if k != j]; w = ridge_fit(torch.cat([D[k][1] for k in tr]), torch.cat([D[k][2] for k in tr]), lam)
                e += float(((ridge_pred(D[j][1], w) - D[j][2]) ** 2).mean())
            return e
        curve = {lam: cv_err(lam) for lam in lams}; lam = min(curve, key=curve.get); w = ridge_fit(torch.cat([D[k][1] for k in others]), torch.cat([D[k][2] for k in others]), lam)
        tri, F = features(s); out[s] = dict(tri=tri, score=ridge_pred(F, w).cpu().numpy(), lam=lam, lam_curve={str(k): v for k, v in curve.items()}, feats=F.cpu().numpy())
        print(f"sub-{s:03d} score alpha {lam}; inner-LOPO MSE curve", {k: round(v, 6) for k, v in curve.items()}, flush=True)
    return out


def dual_members(F, Y, rows_tr, rows_te, LAM, cw):
    """Batched dual ridge: F (n, k), Y (n, J); rows_tr (M, m) training rows per member (equal m), rows_te (t,) shared test rows.
    In-fold z-scoring per member on its training rows (ddof 0) and colw, as pe_gain. Returns (M, t, J) predictions."""
    A = F[rows_tr]; mu, sd = A.mean(1, keepdim=True), A.std(1, unbiased=False, keepdim=True) + 1e-6
    Za = (A - mu) / sd * cw; Zb = (F[rows_te][None] - mu) / sd * cw; Yt = Y[rows_tr]; Yc = Yt - Yt.mean(1, keepdim=True)
    m = rows_tr.shape[1]; K = Za @ Za.transpose(1, 2)
    alpha = torch.linalg.solve(K + LAM * m * torch.eye(m, dtype=K.dtype, device=DEV), Yc)
    return (Zb @ Za.transpose(1, 2)) @ alpha


def pe_members(s, sc, evaluate, fpath):
    """All pruning members of one participant; returns dict member -> per-trial gains (n,) on the full test folds (+ G2)."""
    from belief_bottleneck import LAM, folds
    fz = np.load(fpath("raw", s)); tri = fz["tri"]; n = len(tri); y = evaluate(s, tri).astype(np.float64)
    F = torch.as_tensor(fz["F"].astype(np.float64), device=DEV); cw = torch.as_tensor(fz["colw"].astype(np.float64), device=DEV)
    Y = torch.as_tensor(y[:, None], device=DEV); fo = folds(n)
    pos = {t: i for i, t in enumerate(sc["tri"])}; assert all(t in pos for t in tri)
    useful = np.array([sc["score"][pos[t]] for t in tri]); feats = np.array([sc["feats"][pos[t]] for t in tri])
    var_, ptp_ = feats[:, 0], feats[:, 1]; rng = np.random.default_rng(1000 + s); shuffled = rng.permutation(useful)
    zr = np.load(ROOT / f"results/ext/m2/pvi_pe_ridge_sub-{s:03d}.npz"); rpos = {t: i for i, t in enumerate(zr["tri"])}; reward = np.array([zr["reward"][rpos[t]] for t in tri])
    def gains(rank_fn, x, strat=True):
        """rank_fn(b, tr) -> keep-priority per training row (higher kept). strat=True (PRIMARY, G1 condition 1): keep the top x % WITHIN each
        outcome x |PE|-decile stratum of the fold's TRAINING rows; strat=False: global top x % (secondary). Returns (n,) gains."""
        sc_ = np.full(n, np.nan)
        for b in range(10):
            tr = np.where(fo != b)[0]; te = np.where(fo == b)[0]
            if x < 1:
                pr = rank_fn(b, tr)
                if strat:
                    g = strata(y[tr], reward[tr]); keep = []
                    for gv in np.unique(g):
                        ii = np.where(g == gv)[0]; keep += list(ii[np.argsort(-pr[ii], kind="stable")[:round(x * len(ii))]])
                    tr = np.sort(tr[np.array(keep, int)])
                else:
                    tr = np.sort(tr[np.argsort(-pr, kind="stable")[:round(x * len(tr))]])
            P = dual_members(F, Y, torch.as_tensor(tr, device=DEV)[None], torch.as_tensor(te, device=DEV), LAM, cw)[0, :, 0]
            fi = folds(len(tr), 9); xin = torch.zeros(len(tr), dtype=torch.float64, device=DEV)
            for c in range(9):
                xin[torch.as_tensor(fi == c, device=DEV)] = dual_members(F, Y, torch.as_tensor(tr[fi != c], device=DEV)[None], torch.as_tensor(tr[fi == c], device=DEV), LAM, cw)[0, :, 0]
            yt, ye = Y[tr, 0], Y[te, 0]; xm, ym = xin.mean(), yt.mean(); slope = ((xin - xm) * (yt - ym)).sum() / ((xin - xm) ** 2).sum(); icpt = ym - slope * xm
            rv = ((yt - icpt - slope * xin) ** 2).mean(); vv = ((yt - ym) ** 2).mean()
            sc_[te] = ((torch.log(vv / rv) + (ye - ym) ** 2 / vv - (ye - icpt - slope * P) ** 2 / rv) / (2 * LN2)).cpu().numpy()
        return sc_
    # decoder confidence: |inner-CV prediction - training mean| from the full-training-rows inner fits of each outer fold (no own label)
    conf = {}
    for b in range(10):
        tr = np.where(fo != b)[0]; fi = folds(len(tr), 9); c_ = np.zeros(len(tr))
        for c in range(9):
            p = dual_members(F, Y, torch.as_tensor(tr[fi != c], device=DEV)[None], torch.as_tensor(tr[fi == c], device=DEV), LAM, cw)[0, :, 0]
            c_[fi == c] = np.abs(p.cpu().numpy())                                  # Yc centred on the training mean -> |pred - mean|
        conf[b] = dict(zip(tr, c_))
    res = {"none": gains(None, 1.0)}
    for st_, tag in ((True, ""), (False, "U")):          # "" = stratified (primary); "U" = unstratified (secondary)
        for x in XS:
            res[f"score{tag}_{x}"] = gains(lambda b, tr: useful[tr], x, st_)
            res[f"shuffled{tag}_{x}"] = gains(lambda b, tr: shuffled[tr], x, st_)
            res[f"ptp{tag}_{x}"] = gains(lambda b, tr: -ptp_[tr], x, st_)
            res[f"var{tag}_{x}"] = gains(lambda b, tr: -var_[tr], x, st_)
            res[f"conf{tag}_{x}"] = gains(lambda b, tr: np.array([conf[b][t] for t in tr]), x, st_)
            for r in range(NRAND):
                rr = np.random.default_rng(10 * s + r).random(n); res[f"random{r}{tag}_{x}"] = gains(lambda b, tr: rr[tr], x, st_)
    st = json.loads((ROOT / f"results/standard/pe_bottleneck/sub-{s:03d}.json").read_text())
    ref = next(r["gain"] for r in st["rows"] if r["input"] == "eeg" and r["mode"] == "pca" and r["L"] == "full")
    return {k: float(np.mean(v)) for k, v in res.items()}, abs(float(np.mean(res["none"])) - ref)


def summarize(out):
    per = [json.loads(f.read_text()) for f in sorted(out.glob("pe_sub-*.json"))]; n = len(per)
    g = torch.Generator(device=DEV); g.manual_seed(20261121); ix = torch.randint(0, n, (10000, n), generator=g, device=DEV)
    def stat(v, alt="greater"):
        t = torch.as_tensor(v, dtype=torch.float64, device=DEV); bm = t[ix].mean(1)
        ci = torch.quantile(bm, torch.tensor([.025, .975, .05, .95], dtype=torch.float64, device=DEV)).tolist()
        p = float((bm <= 0).double().mean())                                     # one-sided bootstrap p for mean > 0
        return dict(mean=float(t.mean()), ci95=ci[:2], ci90=ci[2:], p_boot=p, n_pos=int((t > 0).sum()))
    m = lambda k: np.array([p["members"][k] for p in per])
    rand = {(t, x): np.mean([m(f"random{r}{t}_{x}") for r in range(NRAND)], 0) for t in ("", "U") for x in XS}
    prim = {f"score-random_{x}": m(f"score_{x}") - rand[("", x)] for x in XS} | {f"score-none_{x}": m(f"score_{x}") - m("none") for x in XS}
    sec = {f"score-{c}_{x}": m(f"score_{x}") - m(f"{c}_{x}") for c in ("ptp", "var", "conf") for x in XS}
    sec |= {f"UNSTRAT score-random_{x}": m(f"scoreU_{x}") - rand[("U", x)] for x in XS} | {f"UNSTRAT score-none_{x}": m(f"scoreU_{x}") - m("none") for x in XS}
    res = dict(n=n, g2_max=max(p["g2"] for p in per), none_mean=float(m("none").mean()), primary={k: stat(v) for k, v in prim.items()}, secondary={k: stat(v) for k, v in sec.items()},
               floor_shuffled_minus_random={x: stat(m(f"shuffled_{x}") - rand[("", x)]) for x in XS}, members_mean={k: float(m(k).mean()) for k in per[0]["members"]},
               score_alpha={p["participant"]: p["lam"] for p in per})
    for fam in ("primary", "secondary"):
        ps = sorted((v["p_boot"], k) for k, v in res[fam].items())
        for r, (p, k) in enumerate(ps): res[fam][k]["p_holm"] = min(1.0, p * (len(ps) - r))
    MIN = 0.005; helps = any(all(res["primary"][f"{c}_{x}"]["p_holm"] < 0.05 and res["primary"][f"{c}_{x}"]["mean"] >= MIN for c in ("score-random", "score-none")) for x in XS)
    nosig = all(res["primary"][k]["ci90"][1] < MIN for k in res["primary"])
    res["verdict"] = "pruning helps" if helps else "no useful signal" if nosig else "inconclusive"
    (out / "summary_pe.json").write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k != "members_mean"}, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--subjects", default=""); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize: return summarize(out)
    from i2_complementarity import targets
    evaluate, allsubs, fpath, _ = targets(); t0 = time.time(); S = usefulness_scores(allsubs)
    for s in ([int(x) for x in a.subjects.split(",")] if a.subjects else allsubs):
        if (out / f"pe_sub-{s:03d}.json").exists(): continue
        mem, g2 = pe_members(s, S[s], evaluate, fpath); assert g2 < 1e-9, g2
        (out / f"pe_sub-{s:03d}.json").write_text(json.dumps(dict(participant=f"sub-{s:03d}", lam=S[s]["lam"], members=mem, g2=g2), indent=1))
        print(f"sub-{s:03d}", {k: round(v, 4) for k, v in mem.items() if not k.startswith("random")}, "G2", g2, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
