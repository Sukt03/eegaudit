"""IDEA II: RELIANCE vs INFORMATION MAP (rung R7), with N6 (channel-group reliance) inside. EXECUTOR-5, 21 Nov 2026.
Spec: outputs/EXTENSIONS_LOOP_20261121.md "Idea II" (METHOD 05:45), FRONTIER FINAL RULING (06:00: recompute the all-cell grid WITHOUT the
flagged power cells first), N6 ruling (channel groups), MASTER RESUME ROUTING 10:47.
Definitions (per participant, held-out rows only):
  reliance(cell)  = locked score of the TRAINED, FROZEN decoder - its score when that cell of every test row is replaced by the same cell of
                    another test row (a derangement; 5 draws, averaged). Forward passes only, batched over cells x draws on the GPU.
  information     = the cell-alone yield (same read-out family fit on that cell only, M3 rule).
  unique (B2)     = full yield - yield of the read-out RE-FIT without that cell (leave-one-cell-out).
  Sanity (adoption rule): for the ridge, Spearman(reliance, unique) >= 0.8.
Image (THINGS-EEG2, 200 test items, k = 80, locked score = InfoNCE soft bound, M3.soft_bits_gpu = run.soft_bits):
  cells = 8 named regions (regions.py) x 10 windows of 100 ms (raw-input decoders have no band axis).
  decoders: NICE-GA (BMVC retrain, seed 141; Enc_eeg -> Proj_eeg vs the 200 CLIP centres, as run_acq_nice.py) and the broadband ridge
            (63 ch x 50 x 20-ms bins = M3 broadband cell; M3 image_batch rule, G2 = M3 json broadband_cell yield).
  N6 groups: ocular-sensitive {Fp1, Fp2, AF7, AF8}; temporal (muscle-sensitive) {FT7, FT8, T7, T8, TP7, TP8, TP9, TP10}; 20 size-matched
            random groups per size drawn (seed 2026) from the channels outside both groups. Whole epoch replaced (5 draws). Ridge also
            RE-FIT without each group (information-side counterpart). Criterion (per decoder): group drop - mean random-group drop, CI > 0
            over participants.
  Not run: ATM-S (only released embeddings on disk, no weights), UBP (17 posterior channels in its input: zero reliance on both N6 groups
            by construction; its region x time map is limited to the 3 posterior regions).
PE (ds003458, locked score = M3 pe_gain: nested held-out Gaussian density gain, 10 contiguous folds, calibration on inner folds):
  cells = 8 regions x 5 bands x 6 windows (0-600 ms), M3 combined features, with the POWER columns of cells flagged "includes pre-event
  activity" (M3.edge_flag) REMOVED (FRONTIER FINAL RULING b). decoder = all-cell dual ridge (G2 = M3.pe_gain_batch on the same matrix);
  EEGNet LOPO not run (no saved weights; predictions only). Floors: target circular shifts n/4, n/2, 3n/4.
Usage: python idea2_reliance.py --modality image|pe --subjects 1,2,3 --out DIR"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/path/to/eegaudit/experiments/ext")
import frontier_smokes as FS
M3, RG, DEV, F64 = FS.M3, FS.RG, "cuda", torch.float64
from belief_bottleneck import LAM, folds
OCULAR = ["Fp1", "Fp2", "AF7", "AF8"]; TEMPORAL = ["FT7", "FT8", "T7", "T8", "TP7", "TP8", "TP9", "TP10"]
NDRAW = 5


def derangements(n, nd, seed):
    g = torch.Generator(device=DEV); g.manual_seed(seed); out = []
    for _ in range(nd):
        p = torch.randperm(n, generator=g, device=DEV); d = torch.empty_like(p); d[p] = p.roll(1); out.append(d)
    return torch.stack(out)                                                     # (nd, n): row i takes the cell of row d[i] != i


def spear(a, b): return FS.spearman(torch.as_tensor(a), torch.as_tensor(b))


# ------------------------------------------------------------------ image
def nice_ga(subject):
    sys.path.insert(0, "/path/to/eegaudit/experiments/eegbits")
    import run_acq_nice as A
    Enc, Proj = A.load_classes("nice_ga"); tag = f"nice_ga_dgx_v2_{subject.replace('-', '')}_seed141"
    enc, proj = Enc().to(DEV).eval(), Proj().to(DEV).eval()
    enc.load_state_dict(torch.load(A.NICE / "model" / f"{tag}_Enc_eeg_cls.pth", map_location=DEV, weights_only=True), strict=True)
    proj.load_state_dict(torch.load(A.NICE / "model" / f"{tag}_Proj_eeg_cls.pth", map_location=DEV, weights_only=True), strict=True)
    cen = torch.tensor(np.squeeze(np.load(A.NICE / "dnn_feature/center_clip.npy", allow_pickle=True)), dtype=torch.float32, device=DEV)
    @torch.no_grad()
    def embed(x, bs=500):                                                      # x (M, 63, 250) GPU float32
        return torch.cat([proj(enc(x[i:i + bs, None])).float() for i in range(0, len(x), bs)])
    return embed, cen / cen.norm(dim=1, keepdim=True)


def image_subject(s, out):
    import run as R
    t0 = time.time(); sub, Ytr, Yte, Xtr4, Xte80, ch = M3.image_load(s, R); ntr = len(Ytr)
    sets = {k: RG.indices(ch, v) for k, v in RG.REGIONS.items()}; regs = list(sets)
    cells = [(r, w) for r in regs for w in range(10)]; C = len(cells)
    oc, te_ = RG.indices(ch, OCULAR), RG.indices(ch, TEMPORAL); rest = [i for i in range(63) if i not in oc + te_]
    rng = np.random.default_rng(2026); groups = {"ocular": oc, "temporal": te_}
    for sz, nm in ((4, "rand4"), (8, "rand8")):
        for j in range(20): groups[f"{nm}_{j}"] = sorted(rng.choice(rest, sz, replace=False).tolist())
    Xm = torch.tensor(Xte80.mean(1), device=DEV)                                 # (200, 63, 250) repetition mean (NICE-GA input)
    der = derangements(200, NDRAW, 1000 + s); res = dict(participant=sub, cells=cells, groups={k: [ch[i] for i in v] for k, v in groups.items()})
    Yn = torch.as_tensor(Yte / np.linalg.norm(Yte, axis=1, keepdims=True), dtype=F64, device=DEV); Yg = torch.as_tensor(Ytr, dtype=F64, device=DEV)
    # ---- NICE-GA (frozen)
    embed, cen = nice_ga(sub); sb = lambda E: M3.soft_bits_gpu(FS.cosS(E.view(-1, 200, E.shape[-1]).double(), cen.double()))
    base = float(sb(embed(Xm))[0]); acq = json.loads((FS.ROOT / f"results/eegbits/acq_nice_ga_{sub}_seed141.json").read_text())
    nat = next(r for r in acq["rows"] if r["axis"] == "native"); res["nicega_g2"] = dict(this=base, registered_acq_native=nat.get("soft_bits"), abs_diff=abs(base - nat.get("soft_bits", np.nan)))
    def patched_scores(masks):                                                  # masks: list of (chs, t0, t1) -> (len, NDRAW) scores
        outs = []
        for chs, a, b in masks:
            Xp = Xm[None].repeat(NDRAW, 1, 1, 1); ci = torch.as_tensor(chs, device=DEV)
            for d in range(NDRAW): Xp[d][:, ci, a:b] = Xm[der[d]][:, ci, a:b]
            outs.append(Xp)
        X = torch.cat(outs).view(-1, 63, 250); E = embed(X); return sb(E).view(len(masks), NDRAW)
    cm = [(sets[r], 25 * w, 25 * w + 25) for r, w in cells]; gm = [(v, 0, 250) for v in groups.values()]
    rc = torch.cat([patched_scores(cm[i:i + 4]) for i in range(0, C, 4)]); rg = torch.cat([patched_scores(gm[i:i + 4]) for i in range(0, len(gm), 4)])
    res["nicega"] = dict(base=base, reliance=(base - rc.mean(1)).tolist(), reliance_draw_sd=rc.std(1).tolist(), group_reliance={k: float(base - rg[i].mean()) for i, k in enumerate(groups)})
    print(sub, "NICE-GA base", round(base, 4), "G2", res["nicega_g2"], f"{time.time()-t0:.0f}s", flush=True)
    # ---- broadband ridge (M3 broadband cell; frozen for reliance, re-fit for unique)
    Ftr, Fte = M3.image_broadband(Xtr4, Xte80); del Xtr4
    Ftr = torch.as_tensor(Ftr, device=DEV); Fte = torch.as_tensor(Fte, device=DEV); k = Ftr.shape[1]
    alpha, W, ym, mu, sd = FS.image_allcell_ridge(lambda r: Ftr[r], ntr, k, Yg, [Yg]); Ate = (Fte.double() - mu) / sd; P = Ate @ W + ym
    rb = float(M3.soft_bits_gpu(FS.cosS(P, Yn)[None])[0]); m3j = json.loads((FS.ROOT / f"results/ext/m3/image_{sub}.json").read_text())
    res["ridge_g2"] = dict(this=rb, m3_broadband_cell=m3j["broadband_cell"]["yield_"], abs_diff=abs(rb - m3j["broadband_cell"]["yield_"]), alpha=alpha, m3_alpha=m3j["broadband_cell"]["alpha"])
    cols = lambda chs, ws: torch.as_tensor([c * 50 + 5 * w + j for c in chs for w in ws for j in range(5)], device=DEV)
    ccols = [cols(sets[r], [w]) for r, w in cells]; gcols = [cols(v, range(10)) for v in groups.values()]
    def ridge_patch(colsets):
        sc = []
        for cc in colsets:
            dP = torch.stack([(Ate[der[d]][:, cc] - Ate[:, cc]) @ W[cc] for d in range(NDRAW)])         # (NDRAW, 200, 1024)
            sc.append(M3.soft_bits_gpu(FS.cosS(P[None] + dP, Yn)))
        return torch.stack(sc)
    rrc, rrg = ridge_patch(ccols), ridge_patch(gcols)
    # Gram once; leave-cell-out and leave-group-out re-fits by sub-indexing (z-scoring is per column, so it is unchanged by dropping columns)
    idx = np.random.default_rng(0).permutation(ntr); cut = int(0.9 * ntr); tr, va = torch.as_tensor(idx[:cut], device=DEV), torch.as_tensor(idx[cut:], device=DEV)
    A = (Ftr.double() - mu) / sd; G90 = A[tr].T @ A[tr]; ym90 = Yg[tr].mean(0); C90 = A[tr].T @ (Yg[tr] - ym90); Gall = A.T @ A; Call = A.T @ (Yg - Yg.mean(0))
    Av = A[va]; T = Yg[va]; T = T / T.norm(dim=1, keepdim=True); del A
    def refit(keep):
        I = torch.eye(len(keep), dtype=F64, device=DEV); G = G90[keep][:, keep]; Cc = C90[keep]; best, al = -1, None
        for a in R.ALPHAS:
            pv = Av[:, keep] @ torch.cholesky_solve(Cc, torch.linalg.cholesky(G + a * I)) + ym90; pv = pv / pv.norm(dim=1, keepdim=True)
            Sm = pv @ T.T; m = float((1.0 / ((Sm > Sm.diagonal()[:, None]).sum(1) + 1)).mean())
            if m > best: best, al = m, a
        Wk = torch.cholesky_solve(Call[keep], torch.linalg.cholesky(Gall[keep][:, keep] + al * I))
        return float(M3.soft_bits_gpu(FS.cosS(Ate[:, keep] @ Wk + Yg.mean(0), Yn)[None])[0])
    allc = torch.arange(k, device=DEV); notin = lambda cc: allc[~torch.isin(allc, cc)]
    full_refit = refit(allc); loco = [refit(notin(cc)) for cc in ccols]; logo = {nm: refit(notin(cc)) for nm, cc in zip(groups, gcols)}
    info = M3.image_batch(torch.stack([Ftr[:, cc] for cc in ccols]), torch.stack([Fte[:, cc] for cc in ccols]), Yg, Yn, R, 1)
    res["ridge"] = dict(base=rb, refit_base=full_refit, reliance=(rb - rrc.mean(1)).tolist(), unique=[full_refit - v for v in loco],
                        info=[x["yield_"] for x in info], info_floor=[x["floor"] for x in info],
                        group_reliance={nm: float(rb - rrg[i].mean()) for i, nm in enumerate(groups)}, group_unique={nm: full_refit - v for nm, v in logo.items()})
    rr, un, inf, rn = res["ridge"]["reliance"], res["ridge"]["unique"], res["ridge"]["info"], res["nicega"]["reliance"]
    res["spearman"] = dict(ridge_reliance_vs_unique=spear(rr, un), ridge_reliance_vs_info=spear(rr, inf), nicega_reliance_vs_info=spear(rn, inf),
                           nicega_reliance_vs_unique=spear(rn, un), nicega_vs_ridge_reliance=spear(rn, rr))
    def n6(d):
        r4 = np.mean([d[f"rand4_{j}"] for j in range(20)]); r8 = np.mean([d[f"rand8_{j}"] for j in range(20)])
        return dict(ocular=d["ocular"], ocular_minus_rand4=d["ocular"] - r4, temporal=d["temporal"], temporal_minus_rand8=d["temporal"] - r8, rand4_mean=r4, rand8_mean=r8)
    res["n6"] = dict(nicega_reliance=n6(res["nicega"]["group_reliance"]), ridge_reliance=n6(res["ridge"]["group_reliance"]), ridge_unique_refit=n6(res["ridge"]["group_unique"]))
    res["runtime_s"] = round(time.time() - t0, 1)
    print(sub, "ridge G2", res["ridge_g2"], "| spearman", {k: round(v, 3) for k, v in res["spearman"].items()}, "| N6", json.dumps(res["n6"]), f"{res['runtime_s']}s", flush=True)
    (out / f"image_{sub}.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------ PE
def pe_subject(s, out):
    t0 = time.time(); D = FS.load_pe(s); n = D["n"]; Z = D["Z"]; cells = D["cells"]; C = len(cells)
    keep = torch.ones(C, 36, dtype=torch.bool, device=DEV)
    for i, (r, b, t) in enumerate(cells):
        if M3.edge_flag("combined", M3.BNAMES.index(b), t): keep[i, :6] = False
    F = Z.permute(1, 0, 2)[:, keep].double(); cc = torch.arange(C, device=DEV)[:, None].expand(C, 36)[keep]; k = F.shape[1]
    OH = torch.nn.functional.one_hot(cc, C).double()                                                     # (k, C)
    Yq = D["y_raw"][torch.stack([torch.arange(n, device=DEV)] + [(torch.arange(n, device=DEV) - round(n * q / 4)) % n for q in (1, 2, 3)])].T   # (n, 4)
    Q = Yq.shape[1]; fo = folds(n); g = torch.Generator(device=DEV); g.manual_seed(3000 + s)
    def stdz(tr, te):
        A = F[tr]; mu, sd = A.mean(0), A.std(0, unbiased=False) + 1e-6; return (A - mu) / sd, (F[te] - mu) / sd
    def preds(tr, te, cells_chunk=24):
        """full prediction (nte, Q), leave-one-cell-out predictions (C, nte, Q), W (k, Q); dual ridge, z-scoring in-fold (M3 algebra)."""
        A, B = stdz(tr, te); l = LAM * len(tr); I = torch.eye(len(tr), dtype=F64, device=DEV); K, KB = A @ A.T, B @ A.T; Yc = Yq[tr] - Yq[tr].mean(0)
        al = torch.linalg.solve(K + l * I, Yc); Pf = KB @ al; Pl = torch.empty(C, len(te), Q, dtype=F64, device=DEV)
        for c0 in range(0, C, cells_chunk):
            M = OH[:, c0:c0 + cells_chunk].T[:, None, :]                                                    # (c, 1, k)
            Kc = (A[None] * M) @ A.T; KBc = (B[None] * M) @ A.T
            Pl[c0:c0 + cells_chunk] = (KB[None] - KBc) @ torch.linalg.solve(K[None] - Kc + l * I, Yc.expand(Kc.shape[0], -1, -1))
        return Pf, Pl, A.T @ al, B
    def cal(inner, yt):
        xm, ym = inner.mean(-2, keepdim=True), yt.mean(-2, keepdim=True); sl = ((inner - xm) * (yt - ym)).sum(-2, keepdim=True) / ((inner - xm) ** 2).sum(-2, keepdim=True)
        ic = ym - sl * xm; rv = ((yt - ic - sl * inner) ** 2).mean(-2, keepdim=True); vv = ((yt - ym) ** 2).mean(-2, keepdim=True); return sl, ic, rv, vv, ym
    def gain(Pp, ye, c):
        sl, ic, rv, vv, ym = c; return (torch.log(vv / rv) + (ye - ym) ** 2 / vv - (ye - ic - sl * Pp) ** 2 / rv) / (2 * np.log(2))
    gf = torch.zeros(n, Q, dtype=F64, device=DEV); gl = torch.zeros(C, n, Q, dtype=F64, device=DEV); gp = torch.zeros(C, n, Q, dtype=F64, device=DEV)
    for b in range(10):
        tr = np.where(fo != b)[0]; te = np.where(fo == b)[0]; Pf, Pl, W, B = preds(tr, te); fi = folds(len(tr), 9)
        inf_ = torch.zeros(len(tr), Q, dtype=F64, device=DEV); inl = torch.zeros(C, len(tr), Q, dtype=F64, device=DEV)
        for c in range(9):
            m = torch.as_tensor(fi == c, device=DEV); a_, b_, _, _ = preds(tr[fi != c], tr[fi == c]); inf_[m] = a_; inl[:, m] = b_
        yt, ye = Yq[tr], Yq[te]; cf, cl = cal(inf_, yt), cal(inl, yt[None]); gf[te] = gain(Pf, ye, cf); gl[:, te] = gain(Pl, ye[None], cl)
        for d in range(NDRAW):                                   # derangement within the test fold
            p = torch.randperm(len(te), generator=g, device=DEV); don = torch.empty_like(p); don[p] = p.roll(1)
            dBW = ((B[don] - B)[:, :, None] * W[None]).permute(0, 2, 1) @ OH                                 # (nte, Q, C)
            gp[:, te] += gain(Pf[None] + dBW.permute(2, 0, 1), ye[None], cf) / NDRAW
    E = M3.pe_gain_batch(F[None], Yq, LAM)[0]
    base = gf[:, 0].mean()
    # information: cell-alone yield on the kept columns, grouped by width (30 / 36)
    inf_y = torch.zeros(C, dtype=F64); inf_f = torch.zeros(C, dtype=F64)
    for wdt in (30, 36):
        ids = [i for i in range(C) if int(keep[i].sum()) == wdt]
        if not ids: continue
        Eb = M3.pe_gain_batch(torch.stack([F[:, cc == i] for i in ids]), Yq, LAM)
        for j, i in enumerate(ids): inf_y[i] = float(Eb[j, :, 0].mean()); inf_f[i] = float(Eb[j, :, 1:].mean())
    rel = (gf[None, :, 0] - gp[:, :, 0]).mean(1); uni = (gf[None, :, 0] - gl[:, :, 0]).mean(1)
    rel_f = (gf[None, :, 1:] - gp[:, :, 1:]).mean((1, 2)); uni_f = (gf[None, :, 1:] - gl[:, :, 1:]).mean((1, 2))
    m3j = D["m3_json"]
    res = dict(participant=D["sub"], cells=cells, n_trials=n, n_features=k, n_power_cols_removed=int((~keep).sum()),
               allcell_no_flagged=dict(this=float(base), m3_estimator=float(E[:, 0].mean()), max_abs_diff_per_trial=float(np.abs(E - gf.cpu().numpy()).max()),
                                       shift_floor=float(gf[:, 1:].mean()), m3_x7=m3j["all_feature_yield"], m3_broadband=m3j["broadband_cell"]["yield_"]),
               ridge=dict(reliance=rel.tolist(), unique=uni.tolist(), info=inf_y.tolist(), reliance_shift_floor=rel_f.tolist(), unique_shift_floor=uni_f.tolist(), info_floor=inf_f.tolist()),
               spearman=dict(ridge_reliance_vs_unique=spear(rel, uni), ridge_reliance_vs_info=spear(rel, inf_y), unique_vs_info=spear(uni, inf_y)),
               top_reliance=[cells[i] for i in rel.argsort(descending=True)[:5].tolist()], eegnet="not run: no saved EEGNet LOPO weights (predictions only)",
               runtime_s=round(time.time() - t0, 1))
    print(D["sub"], "all-cell (flagged power removed)", json.dumps(res["allcell_no_flagged"]), "| spearman", {a: round(v, 3) for a, v in res["spearman"].items()},
          "| top reliance", res["top_reliance"][:3], f"{res['runtime_s']}s", flush=True)
    (out / f"pe_{D['sub']}.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--modality", required=True, choices=["image", "pe"]); ap.add_argument("--subjects", default="1,2,3")
    ap.add_argument("--out", required=True); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for s in [int(x) for x in a.subjects.split(",")]:
        f = out / (f"image_sub-{s:02d}.json" if a.modality == "image" else f"pe_sub-{s:03d}.json")
        if f.exists(): continue
        (image_subject if a.modality == "image" else pe_subject)(s, out); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
