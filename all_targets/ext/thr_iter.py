"""GRID & THRESHOLD ITERATION, round 1 threshold variants (METHOD 05:55; rung R5). EXECUTOR-2, 21 Nov 2026. GPU bootstraps (PLAN §10).
Dev participants only: image sub-01..03 (frozen NICE-GA input-bottleneck curve r1, InfoNCE k = 80; results/method_pilots/
r1_input_bottleneck_sub-XX_basis-sub-XX.json), PE sub-001..003 (X7 raw-ERP PCA-L curve, locked log-score; results/standard/pe_bottleneck).
Summaries of a yield-vs-width curve y(L), on the participant-mean curve, with a participant-bootstrap CI (2,000 resamples, GPU):
  T1  L80 / L90 / L95: smallest L with y(L) >= q * max_L y(L)
  T2  TOST-sufficient width: smallest L whose yield is not below the full-width yield by more than 10 % (one-sided 95 % bootstrap lower
      bound of y(L) - 0.9 y(full) > 0)
  T3  AUC: mean over the log2-L grid points of clip(y(L) / y(full), 0, 1) (trapezoid on log2 L, normalised to [0, 1])
  T4  knee (kneedle): the L maximising the vertical distance between the normalised curve and its chord, on log2 L
  T5  PE: the same summaries with the X7 ridge re-fit on 50 % of every outer training fold (random rows, seed 100 + fold; inner folds
      within the subset; same PCA-L path as pe_bottleneck_logscore, on the GPU). Image: not defined for r1 (frozen decoder, no training).
Selection rule (fixed by METHOD before the smokes): the winner = the smallest relative CI width (CI width / estimate) among summaries that
change by < 25 % between 50 % and 100 % of the training rows (PE decides stability; image reports the CI widths).
G2: the 100 % PE curve recomputed through the 50 %-code path with frac = 1 equals results/standard/pe_bottleneck (|d| < 1e-9).
ROUND 2 (--round2; METHOD ROUND 1 RULING): L80/L90/L95 by PROTOCOL-f interpolation on the monotone envelope vs FULL width (yield_stats.l90),
knee on an interpolated log2-L grid, the TOST width interpolated, all as per-participant medians with participant-bootstrap CIs; AUC
definition fixed (area under y(L)/y(full) over log2 L from 1 to full). Finer PE grid (LS2); image keeps the stored r1 grid.
Output results/smoke_feasibility/thr_iter/summary.json (round 1), summary_round2.json. Usage: python thr_iter.py [--round2]
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/path/to/eegaudit")
for p in (ROOT, ROOT / "experiments/agent_loop_20261120", ROOT / "experiments/standard"):
    sys.path.insert(0, str(p))
DEV = "cuda"; OUT = ROOT / "results/smoke_feasibility/thr_iter"; OUT.mkdir(parents=True, exist_ok=True)
LS = [1, 2, 4, 8, 16, 32, 64, 128, 256, "full"]
LS2 = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, "full"]   # round 2: finer PE grid


def summaries(Ls, Y):
    """Ls numeric (full = its dim), Y (P, nL) participant curves (torch, GPU). Returns dict of summary -> (P_boot,) values for all rows of Y."""
    lx = torch.log2(torch.as_tensor(Ls, dtype=torch.float64, device=DEV)); Lt = torch.as_tensor(Ls, dtype=torch.float64, device=DEV)
    mx = Y.max(1, keepdim=True).values; full = Y[:, -1:]
    first = lambda mask: Lt[torch.argmax(mask.double(), 1)]
    out = {f"L{q}": first(Y >= q / 100 * mx) for q in (80, 90, 95)}
    r = torch.clamp(Y / full, 0, 1); w = (lx[1:] - lx[:-1]) / (lx[-1] - lx[0]); out["AUC"] = ((r[:, 1:] + r[:, :-1]) / 2 * w).sum(1)
    xn = (lx - lx[0]) / (lx[-1] - lx[0]); yn = (Y - Y[:, :1]) / (Y.max(1, keepdim=True).values - Y[:, :1] + 1e-12); out["knee"] = Lt[torch.argmax(yn - xn, 1)]
    return out


def boot(Yp, B=2000, seed=20261121):
    g = torch.Generator(device=DEV); g.manual_seed(seed); ix = torch.randint(0, Yp.shape[0], (B, Yp.shape[0]), generator=g, device=DEV)
    return Yp[ix].mean(1)                                                                   # (B, nL) bootstrap mean curves


def tost_L(Ls, Yp, Bc):
    """T2: smallest L with the one-sided 95 % bootstrap lower bound of y(L) - 0.9 y(full) > 0 (on mean curves)."""
    lo = torch.quantile(Bc - 0.9 * Bc[:, -1:], 0.05, dim=0); ok = (lo > 0).nonzero()
    return float(Ls[int(ok[0])]) if len(ok) else float("nan")


def block(Ls, Yp):
    m = Yp.mean(0, keepdim=True); Bc = boot(Yp); est = {k: float(v[0]) for k, v in summaries(Ls, m).items()}; bs = summaries(Ls, Bc)
    res = {}
    for k in est:
        ci = torch.quantile(bs[k], torch.tensor([.025, .975], dtype=torch.float64, device=DEV)).tolist()
        res[k] = dict(estimate=est[k], ci=ci, rel_ci_width=float((ci[1] - ci[0]) / abs(est[k])) if est[k] else None)
    t2 = tost_L(Ls, Yp, Bc)
    # bootstrap of T2 itself: recompute on resampled participant sets (nested resample of 200 x 200)
    g = torch.Generator(device=DEV); g.manual_seed(7); vals = []
    for _ in range(200):
        Yr = Yp[torch.randint(0, Yp.shape[0], (Yp.shape[0],), generator=g, device=DEV)]; vals.append(tost_L(Ls, Yr, boot(Yr, B=200)))
    v = torch.tensor(vals, dtype=torch.float64); v = v[~torch.isnan(v)]
    ci = torch.quantile(v, torch.tensor([.025, .975], dtype=torch.float64)).tolist() if len(v) else [float("nan")] * 2
    res["TOST_sufficient"] = dict(estimate=t2, ci=ci, rel_ci_width=float((ci[1] - ci[0]) / t2) if t2 == t2 and t2 else None)
    return res


def pe_curve(s, evaluate, fpath, frac, LS=LS):
    """X7 PCA-L curve (pe_bottleneck_logscore algebra, pca mode, own target + 3 shifts) with frac of every outer training fold."""
    from belief_bottleneck import LAM, folds
    from pe_bottleneck_logscore import gain_rows
    fz = np.load(fpath("raw", s)); tri = fz["tri"]; n = len(tri); y = evaluate(s, tri).astype(np.float64)
    Ys = np.column_stack([y] + [np.roll(y, round(n * k / 4)) for k in (1, 2, 3)])
    F = torch.tensor(fz["F"].astype(np.float64), device=DEV); cw = torch.tensor(fz["colw"].astype(np.float64), device=DEV); Y = torch.tensor(Ys, device=DEV)
    def preds(tr, te):
        A, B = F[tr], F[te]; mu, sd = A.mean(0), A.std(0, unbiased=False) + 1e-6; A = (A - mu) / sd * cw; B = (B - mu) / sd * cw
        Yc = Y[tr] - Y[tr].mean(0); _, _, Vt = torch.linalg.svd(A - A.mean(0), full_matrices=False); out = {}
        for L in LS:
            Atr, Ate = (A, B) if L == "full" else (A @ Vt[:min(L, Vt.shape[0])].T, B @ Vt[:min(L, Vt.shape[0])].T); k = Atr.shape[1]
            out[L] = (Ate @ torch.linalg.solve(Atr.T @ Atr + LAM * len(Atr) * torch.eye(k, dtype=torch.float64, device=DEV), Atr.T @ Yc)).cpu().numpy()
        return out
    fo = folds(n); sc = {L: np.full((n, 4), np.nan) for L in LS}
    for b in range(10):
        tr = np.where(fo != b)[0]; te = np.where(fo == b)[0]
        if frac < 1: tr = np.sort(np.random.default_rng(100 + b).choice(tr, round(frac * len(tr)), replace=False))
        outer = preds(torch.tensor(tr, device=DEV), torch.tensor(te, device=DEV)); inner = {L: np.zeros((len(tr), 4)) for L in LS}; fi = folds(len(tr), 9)
        for c in range(9):
            p = preds(torch.tensor(tr[fi != c], device=DEV), torch.tensor(tr[fi == c], device=DEV))
            for L in LS: inner[L][fi == c] = p[L]
        for L in LS:
            for j in range(4): sc[L][te, j] = gain_rows(Ys[tr, j], inner[L][:, j], Ys[te, j], outer[L][:, j])
    return [float(sc[L][:, 0].mean()) for L in LS], [float(sc[L][:, 1:].mean()) for L in LS]


def l90(Ls, y, y_full, frac=0.9):
    """= experiments/stats/yield_stats.l90 (PROTOCOL f): smallest L reaching frac * full-width yield; linear interpolation on the monotone
    (running-max) envelope."""
    Ls = np.asarray(Ls, float); env = np.maximum.accumulate(np.asarray(y, float)); t = frac * y_full
    if not np.isfinite(y_full) or y_full <= 0: return float("nan")
    i = int(np.argmax(env >= t)) if (env >= t).any() else None
    if i is None: return float("nan")
    if i == 0: return float(Ls[0])
    x0, x1, e0, e1 = Ls[i - 1], Ls[i], env[i - 1], env[i]
    return float(x0 + (t - e0) / (e1 - e0) * (x1 - x0)) if e1 > e0 else float(x1)


def per_participant(Ls, Y):
    """Round-2 summaries per participant (interpolated): L80/L90/L95 (PROTOCOL f vs full width), knee (kneedle on a 400-point
    piecewise-linear log2-L grid), AUC (fixed definition: area under y(L)/y(full), clipped to [0, 1], over log2 L from L = 1 to full)."""
    Ls = np.asarray(Ls, float); lx = np.log2(Ls); g = np.linspace(lx[0], lx[-1], 400); out = []
    for y in np.asarray(Y, float):
        yf = y[-1]; r = dict({f"L{q}": l90(Ls, y, yf, q / 100) for q in (80, 90, 95)})
        r["AUC"] = float(np.trapezoid(np.clip(y / yf, 0, 1), lx) / (lx[-1] - lx[0])) if yf > 0 else float("nan")
        yi = np.interp(g, lx, np.maximum.accumulate(y)); yn = (yi - yi[0]) / (yi[-1] - yi[0] + 1e-12); xn = (g - g[0]) / (g[-1] - g[0]); r["knee"] = float(2 ** g[int(np.argmax(yn - xn))])
        out.append(r)
    return out


def tost_interp(Ls, Y, B=2000, seed=20261121):
    """TOST-sufficient width, interpolated: smallest L where the one-sided 95 % participant-bootstrap lower bound of mean[y(L) - 0.9 y(full)]
    crosses 0 (linear interpolation in L between grid points). Bootstrap of the width itself by resampling participants (200)."""
    Yt = torch.as_tensor(np.asarray(Y, float), device=DEV)
    def width(Yp, seed_):
        g = torch.Generator(device=DEV); g.manual_seed(seed_); ix = torch.randint(0, Yp.shape[0], (B, Yp.shape[0]), generator=g, device=DEV)
        lb = torch.quantile(Yp[ix].mean(1) - 0.9 * Yp[ix].mean(1)[:, -1:], 0.05, dim=0).cpu().numpy()
        k = np.argmax(lb > 0) if (lb > 0).any() else None
        if k is None: return float("nan")
        if k == 0: return float(Ls[0])
        return float(Ls[k - 1] + (0 - lb[k - 1]) / (lb[k] - lb[k - 1]) * (Ls[k] - Ls[k - 1]))
    est = width(Yt, seed); gg = torch.Generator(device=DEV); gg.manual_seed(7)
    bs = [width(Yt[torch.randint(0, Yt.shape[0], (Yt.shape[0],), generator=gg, device=DEV)], seed + i) for i in range(200)]
    bs = np.array([b for b in bs if b == b]); return dict(estimate=est, ci=np.quantile(bs, [.025, .975]).tolist() if len(bs) else [float("nan")] * 2)


def block2(Ls, Y):
    """Per-participant medians with participant-bootstrap CIs (GPU index matrix; 10,000) + the interpolated TOST width."""
    P = per_participant(Ls, Y); res = {}
    g = torch.Generator(device=DEV); g.manual_seed(20261121); ix = torch.randint(0, len(P), (10000, len(P)), generator=g, device=DEV)
    for k in P[0]:
        v = torch.as_tensor([p[k] for p in P], dtype=torch.float64, device=DEV); med = float(v.median())
        ci = torch.quantile(v[ix].median(1).values, torch.tensor([.025, .975], dtype=torch.float64, device=DEV)).tolist()
        res[k] = dict(estimate=med, per_participant=v.cpu().numpy().tolist(), ci=ci, rel_ci_width=float((ci[1] - ci[0]) / abs(med)) if med else None)
    t = tost_interp(Ls, Y); res["TOST_sufficient"] = dict(t, rel_ci_width=float((t["ci"][1] - t["ci"][0]) / t["estimate"]) if t["estimate"] == t["estimate"] and t["estimate"] else None)
    return res


def main_round2():
    res = {}
    cur = []
    for s in (1, 2, 3):
        d = json.loads((ROOT / f"results/method_pilots/r1_input_bottleneck_sub-{s:02d}_basis-sub-{s:02d}.json").read_text())
        e = {r["L"]: r["soft_bits"] for r in d["rows"] if r["condition"] == "eeg" and r["k"] == 80 and r.get("draw", 0) == 0}
        Ls = sorted([L for L in e if L != "full"]) + ["full"]; cur.append([e[L] for L in Ls])
    Li = [L if L != "full" else 15750 for L in Ls]
    res["image"] = dict(source="r1 frozen NICE-GA input bottleneck, InfoNCE k80, sub-01..03 (stored L grid; interpolation only)", Ls=Li, curves=cur, frac100=block2(Li, cur))
    from i2_complementarity import targets
    evaluate, subs, fpath, _ = targets(); c100, c50 = [], []
    for s in (1, 2, 3):
        c100.append(pe_curve(s, evaluate, fpath, 1.0, LS2)[0]); c50.append(pe_curve(s, evaluate, fpath, 0.5, LS2)[0]); print("pe", s, flush=True)
    st = json.loads((ROOT / "results/smoke_feasibility/thr_iter/summary.json").read_text())["pe"]["curves100"]   # G2 on the shared grid points
    g2 = max(abs(c100[i][LS2.index(L)] - st[i][LS.index(L)]) for i in range(3) for L in LS)
    assert g2 < 1e-9, g2
    Lp = [L if L != "full" else 1472 for L in LS2]; b100, b50 = block2(Lp, c100), block2(Lp, c50)
    stab = {k: (abs(b50[k]["estimate"] - b100[k]["estimate"]) / abs(b100[k]["estimate"]) if b100[k]["estimate"] else None) for k in b100}
    elig = {k: b100[k]["rel_ci_width"] for k in b100 if stab[k] is not None and stab[k] < 0.25 and b100[k]["rel_ci_width"] is not None}
    res["pe"] = dict(source="X7 raw-ERP PCA-L, log-score, sub-001..003, finer L grid", Ls=Lp, curves100=c100, curves50=c50, g2_vs_round1=g2, frac100=b100, frac50=b50,
                     change_50_vs_100=stab, eligible_rel_ci=elig, winner=min(elig, key=elig.get) if elig else None)
    (OUT / "summary_round2.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({t: {k: v for k, v in r.items() if k not in ("curves", "curves100", "curves50")} for t, r in res.items()}, indent=1))


def main():
    res = {}
    # image (r1, frozen NICE-GA, InfoNCE k80)
    cur = []
    for s in (1, 2, 3):
        d = json.loads((ROOT / f"results/method_pilots/r1_input_bottleneck_sub-{s:02d}_basis-sub-{s:02d}.json").read_text())
        e = {r["L"]: r["soft_bits"] for r in d["rows"] if r["condition"] == "eeg" and r["k"] == 80 and r.get("draw", 0) == 0}
        Ls = sorted([L for L in e if L != "full"]) + ["full"]; cur.append([e[L] for L in Ls])
    Li = [L if L != "full" else 15750 for L in Ls]
    res["image"] = dict(source="r1 frozen NICE-GA input bottleneck, InfoNCE k80, sub-01..03", Ls=Li, curves=cur, frac100=block(Li, torch.tensor(cur, dtype=torch.float64, device=DEV)),
                        frac50="not defined (frozen decoder; no training rows)")
    # PE (X7) at 100 % (G2) and 50 %
    from i2_complementarity import targets
    evaluate, subs, fpath, _ = targets(); c100, c50, g2 = [], [], []
    for s in (1, 2, 3):
        y1, _ = pe_curve(s, evaluate, fpath, 1.0); y5, _ = pe_curve(s, evaluate, fpath, 0.5); c100.append(y1); c50.append(y5)
        st = json.loads((ROOT / f"results/standard/pe_bottleneck/sub-{s:03d}.json").read_text())
        ref = [next(r["gain"] for r in st["rows"] if r["input"] == "eeg" and r["mode"] == "pca" and r["L"] == L) for L in LS]; g2.append(max(abs(a - b) for a, b in zip(y1, ref)))
        print("pe", s, "G2", g2[-1], [round(v, 4) for v in y1], [round(v, 4) for v in y5], flush=True)
    assert max(g2) < 1e-9, g2
    Lp = [L if L != "full" else 1472 for L in LS]
    b100 = block(Lp, torch.tensor(c100, dtype=torch.float64, device=DEV)); b50 = block(Lp, torch.tensor(c50, dtype=torch.float64, device=DEV))
    stab = {k: (abs(b50[k]["estimate"] - b100[k]["estimate"]) / abs(b100[k]["estimate"]) if b100[k]["estimate"] else None) for k in b100}
    elig = {k: b100[k]["rel_ci_width"] for k in b100 if stab[k] is not None and stab[k] < 0.25 and b100[k]["rel_ci_width"] is not None}
    res["pe"] = dict(source="X7 raw-ERP PCA-L, log-score, sub-001..003", Ls=Lp, curves100=c100, curves50=c50, g2_max=max(g2), frac100=b100, frac50=b50,
                     change_50_vs_100=stab, eligible_rel_ci=elig, winner=min(elig, key=elig.get) if elig else None)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1)); print(json.dumps({t: {k: v for k, v in r.items() if k not in ("curves", "curves100", "curves50")} for t, r in res.items()}, indent=1))


def main_full():
    """FULL run (GRID ROUND 3 FREEZE + MASTER 05:28): L90 (PROTOCOL-f interpolation) PRIMARY, AUC registered companion; all participants
    (image r1 10; PE X7 23 on the finer LS2 grid at 100 % and 50 % training rows); per-participant medians (L90) / means (AUC) with
    participant-bootstrap CIs; the 50 % vs 100 % change. G2: PE 100 % curve on the stored grid points = results/standard/pe_bottleneck."""
    out = ROOT / "results/ext/thresholds"; out.mkdir(parents=True, exist_ok=True); res = {}
    cur, subs_i = [], []
    for s in range(1, 11):
        f = ROOT / f"results/method_pilots/r1_input_bottleneck_sub-{s:02d}_basis-sub-{s:02d}.json"
        if not f.exists(): continue
        d = json.loads(f.read_text()); e = {r["L"]: r["soft_bits"] for r in d["rows"] if r["condition"] == "eeg" and r["k"] == 80 and r.get("draw", 0) == 0}
        Ls = sorted([L for L in e if L != "full"]) + ["full"]; cur.append([e[L] for L in Ls]); subs_i.append(s)
    Li = [L if L != "full" else 15750 for L in Ls]
    res["image"] = dict(participants=subs_i, Ls=Li, curves=cur, summary=block2(Li, cur))
    from i2_complementarity import targets
    evaluate, subs, fpath, _ = targets(); c100, c50, g2 = [], [], []
    for s in subs:
        y1 = pe_curve(s, evaluate, fpath, 1.0, LS2)[0]; y5 = pe_curve(s, evaluate, fpath, 0.5, LS2)[0]; c100.append(y1); c50.append(y5)
        st = json.loads((ROOT / f"results/standard/pe_bottleneck/sub-{s:03d}.json").read_text())
        g2.append(max(abs(y1[LS2.index(L)] - next(r["gain"] for r in st["rows"] if r["input"] == "eeg" and r["mode"] == "pca" and r["L"] == L)) for L in LS))
        print("pe", s, "G2", g2[-1], flush=True)
    assert max(g2) < 1e-9, g2
    Lp = [L if L != "full" else 1472 for L in LS2]; b100, b50 = block2(Lp, c100), block2(Lp, c50)
    auc = lambda C: np.array([p["AUC"] for p in per_participant(Lp, C)])
    g = torch.Generator(device=DEV); g.manual_seed(20261121); ix = torch.randint(0, len(c100), (10000, len(c100)), generator=g, device=DEV)
    ci = lambda v: torch.quantile(torch.as_tensor(v, device=DEV)[ix].mean(1), torch.tensor([.025, .975], dtype=torch.float64, device=DEV)).tolist()
    a100, a50 = auc(c100), auc(c50)
    res["pe"] = dict(participants=list(subs), Ls=Lp, curves100=c100, curves50=c50, g2_max=max(g2), summary100=b100, summary50=b50,
                     L90_primary=dict(median=b100["L90"]["estimate"], ci=b100["L90"]["ci"], median_50=b50["L90"]["estimate"], change=abs(b50["L90"]["estimate"] - b100["L90"]["estimate"]) / b100["L90"]["estimate"]),
                     AUC_companion=dict(mean=float(a100.mean()), ci=ci(a100), mean_50=float(a50.mean()), change=float(abs(a50.mean() - a100.mean()) / a100.mean()), paired_diff_ci=ci(a50 - a100)))
    ai = np.array([p["AUC"] for p in per_participant(Li, cur)])
    gi = torch.Generator(device=DEV); gi.manual_seed(20261121); ixi = torch.randint(0, len(cur), (10000, len(cur)), generator=gi, device=DEV)
    res["image"]["L90_primary"] = dict(median=res["image"]["summary"]["L90"]["estimate"], ci=res["image"]["summary"]["L90"]["ci"])
    res["image"]["AUC_companion"] = dict(mean=float(ai.mean()), ci=torch.quantile(torch.as_tensor(ai, device=DEV)[ixi].mean(1), torch.tensor([.025, .975], dtype=torch.float64, device=DEV)).tolist(),
                                         frac50="not defined (frozen decoder)")
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({t: {k: v for k, v in r.items() if k in ("L90_primary", "AUC_companion", "g2_max", "participants")} for t, r in res.items()}, indent=1))


if __name__ == "__main__":
    if "--full" in sys.argv: main_full(); sys.exit()
    main_round2() if "--round2" in sys.argv else main()
